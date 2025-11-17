import gym
import yaml
import os
import sys
import torch
from gym import spaces
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../envs/aps/lib")))
from network_simlator import NetworkSimulator
from data_store import DataStore
from aps_utils import clip_abs, tpdv_parse, get_adj, generalized_mean


class Aps(gym.Env):
    def __init__(self, env_args=None, args=None, if_graph=False):
        self.env_args = env_args
        self.if_graph = if_graph
        tpdv_parse(self.env_args)
        self.simulator = NetworkSimulator(env_args.simulation_scenario)
        self.history_length = self.env_args.history_length
        self.datastore = DataStore(self.history_length, ['obs'])

        self.feature_length = 1
        self.feature_length += 1 if self.env_args.if_include_phase else 0
        self.feature_length += 1 if self.env_args.if_include_channel_rank else 0

        self.num_ues = self.simulator.scenario_conf.number_of_ues
        self.num_aps = self.simulator.scenario_conf.number_of_aps
        self.n_agents = self.num_ues * self.num_aps

        if self.if_graph:
            self.same_ue_edges, self.same_ap_edges = get_adj(self.num_ues, self.num_aps, if_transpose=False)

        self.action_space = [spaces.Discrete(2) for _ in range(self.n_agents)]
        self.observation_space = [
            spaces.Box(low=0, high=1, 
                       shape=(self.history_length * self.feature_length,), 
                       dtype=float)
            for _ in range(self.n_agents)]
        self.share_observation_space = [
            spaces.Box(low=0, high=1, 
                       shape=(self.n_agents * self.history_length * self.feature_length,), 
                       dtype=float)
            for _ in range(self.n_agents)]

        with open(self.env_args.simulation_scenario.data_normalization_config, 'r') as config_file:
            self.normalization_dict = yaml.safe_load(config_file)


    def step(self, actions):
        actions = torch.from_numpy(actions).to(self.env_args.simulation_scenario.device_sim)
        self.simulator.step(actions)

        obs, state, reward, mask, info = self.compute_state_reward()
        done = [False] * self.n_agents

        if self.if_graph:
            return obs, state, reward, done, info, mask, self.same_ue_edges, self.same_ap_edges
        else:
            return obs, state, reward, done, info, mask


    def compute_state_reward(self):
        # state calc
        simulator_info = self.simulator.datastore.get_last_k_elements()
        serving_mask = self.simulator.serving_mask.clone().detach().to(torch.int32)

        channel_coef = simulator_info['channel_coef']
        self.datastore.add(obs=channel_coef[-1])
        G = self.datastore.get_last_k_elements()['obs']
        G = clip_abs(G)

        x_mean = torch.tensor(self.normalization_dict['x_mean']).to(device=G.device)
        x_std = torch.tensor(self.normalization_dict['x_std']).to(device=G.device)

        if self.env_args.if_include_phase:
            x = torch.stack((torch.log2(torch.abs(G)), G.angle()), -1)
            x = (x - x_mean[:2]) / x_std[:2]
        else:
            x = torch.log2(torch.abs(G)).unsqueeze(-1)
            x = (x - x_mean[:1]) / x_std[:1]

        # G: [history_length, n_aps, n_ues, feature_dim]
        x = x.permute(1, 2, 0, 3).reshape(self.n_agents, self.history_length * self.feature_length)
        obs = x.clone()
        flat_global = obs.reshape(1, -1)          # [1, n_agents * feature_dim]
        state = flat_global.repeat(obs.size(0), 1)  # [n_agents, n_agents * feature_dim]

        # reward
        threshold = self.env_args.se_threshold
        se = torch.log2(1 + simulator_info['sinr']).mean(dim=0) # mean over different steps and ues
        eta = self.env_args.se_coef
        if self.env_args.if_full_cooperation:
            active_aps = serving_mask.sum(dim=1).sign().sum().float()
            activated_ap_ratio_ = active_aps / self.num_aps
            activated_ap_ratio = torch.full((self.n_agents, 1), activated_ap_ratio_, device=serving_mask.device, dtype=torch.float32)

            se_satis_ratio_clipped_ = torch.clamp(se / threshold, max=1.0).mean().item()
            se_satis_ratio_clipped = torch.full((self.n_agents, 1), se_satis_ratio_clipped_, device=serving_mask.device, dtype=torch.float32)

            reward = eta * se_satis_ratio_clipped - activated_ap_ratio

        else:
            active_aps = serving_mask.sum(dim=1).sign().float()
            if_corresponding_ap_is_on = active_aps.unsqueeze(1).repeat(1, self.num_ues).reshape(-1, 1)

            se_satis_ratio_clipped = torch.clamp(se / self.env_args.se_threshold, max=1.0)
            if_corresponding_se_is_satisfied = se_satis_ratio_clipped.unsqueeze(1).repeat(1, self.num_aps).reshape(-1, 1)

            reward = eta * if_corresponding_se_is_satisfied - if_corresponding_ap_is_on

        mask = self.simulator.channel_manager.measurement_mask.clone().detach() \
            .flatten().to(torch.int32).unsqueeze(1)

        info = {
            'se': torch.log2(1 + simulator_info['sinr']).mean(dim=0), # simulator_info['sinr'].mean(dim=0),
            'active_ap_count': serving_mask.sum(dim=1).sign().sum().float(),
            'reward': reward.mean(),
            'serving_ap_count': serving_mask.sum(dim=0).float(),
            'served_ue_count': serving_mask.sum(dim=1).float(),
        }

        return obs.to(torch.float32).cpu().numpy(), state.to(torch.float32).cpu().numpy(), reward.to(torch.float32).cpu().numpy(), mask.to(torch.float32).cpu().numpy(), info


    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.observation_space[0].shape[0]


    def get_avail_actions(self):
        return torch.ones((self.n_agents, self.get_total_actions()))


    def get_total_actions(self):
        return self.action_space[0].n


    def seed(self, seed):
        self.simulator.set_seed(seed)


    def reset(self):
        self.simulator.reset()
        obs, state, _, mask, info = self.compute_state_reward()

        if self.if_graph:
            return obs, state, mask, info, self.same_ue_edges, self.same_ap_edges
        else:
            return obs, state, mask, info


    def process_obs_graph(self, graph):
        x = graph['channel'].x[:, :2]
        if self.env_args.if_include_channel_rank:
            sorted_indices = torch.argsort(x[:, 0]).to(device=x.device)
            ranks = torch.empty_like(sorted_indices).to(device=x.device)        
            ranks[sorted_indices] = torch.arange(len(x[:, 0])).to(device=x.device)
            normalized_ranks = (ranks / (len(x[:, 0]) - 1)).unsqueeze(dim=1)
            x = torch.cat((x, normalized_ranks), dim=1)
        graph['channel'].x = x


class Aps_c(gym.Env):
    def __init__(self, env_args=None, args=None, if_graph=False):
        self.env_args = env_args
        self.if_graph = if_graph
        tpdv_parse(self.env_args)
        self.simulator = NetworkSimulator(env_args.simulation_scenario)
        self.history_length = self.env_args.history_length
        self.datastore = DataStore(self.history_length, ['obs'])

        self.feature_length = 1
        self.feature_length += 1 if self.env_args.if_include_phase else 0
        self.feature_length += 1 if self.env_args.if_include_channel_rank else 0

        self.num_ues = self.simulator.scenario_conf.number_of_ues
        self.num_aps = self.simulator.scenario_conf.number_of_aps
        self.n_agents = self.num_ues * self.num_aps

        if self.if_graph:
            self.same_ue_edges, self.same_ap_edges = get_adj(self.num_ues, self.num_aps, if_transpose=False)

        self.action_space = [spaces.Discrete(2) for _ in range(self.n_agents)]
        self.observation_space = [
            spaces.Box(low=0, high=1, 
                       shape=(self.history_length * self.feature_length,), 
                       dtype=float)
            for _ in range(self.n_agents)]
        self.share_observation_space = [
            spaces.Box(low=0, high=1, 
                       shape=(self.n_agents * self.history_length * self.feature_length,), 
                       dtype=float)
            for _ in range(self.n_agents)]

        with open(self.env_args.simulation_scenario.data_normalization_config, 'r') as config_file:
            self.normalization_dict = yaml.safe_load(config_file)


    def step(self, actions):
        actions = torch.from_numpy(actions).to(self.env_args.simulation_scenario.device_sim)
        self.simulator.step(actions)

        obs, state, reward, cost, mask, info = self.compute_state_reward()
        done = [False] * self.n_agents

        if self.if_graph:
            return obs, state, reward, cost, done, info, mask, self.same_ue_edges, self.same_ap_edges
        else:
            return obs, state, reward, cost, done, info, mask


    def compute_state_reward(self):
        # state
        simulator_info = self.simulator.datastore.get_last_k_elements()
        serving_mask = self.simulator.serving_mask.clone().detach().to(torch.int32)

        channel_coef = simulator_info['channel_coef']
        self.datastore.add(obs=channel_coef[-1])
        G = self.datastore.get_last_k_elements()['obs']
        G = clip_abs(G)

        x_mean = torch.tensor(self.normalization_dict['x_mean']).to(device=G.device)
        x_std = torch.tensor(self.normalization_dict['x_std']).to(device=G.device)

        if self.env_args.if_include_phase:
            x = torch.stack((torch.log2(torch.abs(G)), G.angle()), -1)
            x = (x - x_mean[:2]) / x_std[:2]
        else:
            x = torch.log2(torch.abs(G)).unsqueeze(-1)
            x = (x - x_mean[:1]) / x_std[:1]

        x = x.permute(1, 2, 0, 3).reshape(self.n_agents, self.history_length * self.feature_length)
        obs = x.clone()
        flat_global = obs.reshape(1, -1)          # [1, n_agents * feature_dim]
        state = flat_global.repeat(obs.size(0), 1)  # [n_agents, n_agents * feature_dim]

        # reward
        if self.env_args.if_full_cooperation:
            active_aps = serving_mask.sum(dim=1).sign().sum().float()
            reward_scalar = 1 - active_aps / self.num_aps
            reward = torch.full((self.n_agents, 1), reward_scalar, device=serving_mask.device, dtype=torch.float32)
        else:
            active_aps = serving_mask.sum(dim=1).sign().float()
            if_corresponding_ap_is_on = active_aps.unsqueeze(1).repeat(1, self.num_ues).reshape(-1, 1)
            reward = -if_corresponding_ap_is_on

        # cost
        threshold = self.env_args.se_threshold
        se = torch.log2(1 + simulator_info['sinr']).mean(dim=0) # mean over different steps and ues
        if self.env_args.if_full_cooperation:
            cost_scalar = (-se + threshold).mean().item()
            cost = torch.full((self.n_agents, 1), cost_scalar, device=serving_mask.device, dtype=torch.float32)
        else:
            minus_if_corresponding_se_is_satisfied = (-se + threshold).unsqueeze(1).repeat(1, self.num_aps).reshape(-1, 1)
            cost = minus_if_corresponding_se_is_satisfied

        # agent mask: set to 0 if no measurement
        mask = self.simulator.channel_manager.measurement_mask.clone().detach() \
            .flatten().to(torch.int32).unsqueeze(1)

        info = {
            'se': se,
            'active_ap_count': serving_mask.sum(dim=1).sign().sum().float(),
            'reward': reward.mean(),
            'serving_ap_count': serving_mask.sum(dim=0).float(),
            'served_ue_count': serving_mask.sum(dim=1).float(),
        }

        return obs.to(torch.float32).cpu().numpy(), state.to(torch.float32).cpu().numpy(), reward.to(torch.float32).cpu().numpy(), cost.to(torch.float32).cpu().numpy(), mask.to(torch.float32).cpu().numpy(), info


    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.observation_space[0].shape[0]


    def get_avail_actions(self):
        return torch.ones((self.n_agents, self.get_total_actions()))


    def get_total_actions(self):
        return self.action_space[0].n


    def seed(self, seed):
        self.simulator.set_seed(seed)


    def reset(self):
        self.simulator.reset()
        obs, state, _, _, mask, info = self.compute_state_reward()

        if self.if_graph:
            return obs, state, mask, info, self.same_ue_edges, self.same_ap_edges
        else:
            return obs, state, mask, info

