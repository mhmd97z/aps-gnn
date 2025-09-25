import time
import torch
import numpy as np
from typing import Tuple
from numpy import ndarray as arr
from onpolicy.runner.shared.base_runner import Runner
from torch_geometric.data import HeteroData, Batch


def _t2n(x):
    return x.detach().cpu().numpy()

def generate_graph_batch(obs, same_ap_adj, same_ue_adj, agent_id):
    n_envs = same_ap_adj.shape[0]
    agent_id = agent_id.reshape((n_envs, -1, 1))

    graphs_list = []
    for i in range(n_envs):
        data = HeteroData()
        data['channel'].x = torch.tensor(obs[i]).to(device="cuda:0")
        data['channel', 'same_ue', 'channel'].edge_index = torch.tensor(same_ue_adj[i]).to(device="cuda:0")
        data['channel', 'same_ap', 'channel'].edge_index = torch.tensor(same_ap_adj[i]).to(device="cuda:0")
        graphs_list.append(data)

    return Batch.from_data_list(graphs_list)


class ApsCRunner(Runner):
    """
    Runner class to perform training, evaluation and data
    collection for the MPEs. See parent class for details
    """

    def __init__(self, config):
        super(ApsCRunner, self).__init__(config)

    def run(self):
        self.warmup()

        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        # This is where the episodes are actually run.
        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    cost_preds,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    rnn_states_cost
                ) = self.collect(step)

                # Obs reward and next obs
                obs, states, rewards, costs, dones, infos, mask, same_ue, same_ap = self.envs.step(
                    actions.copy()
                )
                agent_id = torch.arange(states.shape[1]).unsqueeze(1).repeat(1, states.shape[0]).T.unsqueeze(2).numpy()
                batch = generate_graph_batch(obs, same_ap, same_ue, agent_id)
                data = (
                    batch,
                    agent_id,
                    rewards,
                    costs,
                    dones,
                    infos,
                    values,
                    cost_preds,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    rnn_states_cost
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            if not self.use_eval:
                self.compute()
                train_infos = self.train()

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            # log information
            if episode % self.log_interval == 0:
                avg_ep_rew = np.mean(self.buffer.rewards) * self.episode_length
                avg_ep_cost = np.mean(self.buffer.costs) * self.episode_length
                if not self.use_eval:
                    train_infos["average_episode_rewards"] = avg_ep_rew
                    train_infos["average_episode_costs"] = avg_ep_cost
                print(
                    f"Average episode rewards is {avg_ep_rew:.3f} \t"
                    f"Average episode costs is {avg_ep_cost:.3f} \t"
                    f"Total timesteps: {total_num_steps} \t "
                    f"Percentage complete {total_num_steps / self.num_env_steps * 100:.3f}"
                )
                if not self.use_eval:
                    self.log_train(train_infos, total_num_steps)
                self.log_env(infos, total_num_steps)

    def warmup(self):
        obs, state, _, _, same_ue, same_ap = self.envs.reset()
        agent_id = torch.arange(state.shape[1]).unsqueeze(1).repeat(1, state.shape[0]).T.unsqueeze(2).numpy()
        batch = generate_graph_batch(obs, same_ap, same_ue, agent_id)
        self.buffer.graph_storage.set_graph(0, batch)

    @torch.no_grad()
    def collect(self, step: int) -> Tuple[arr, arr, arr, arr, arr, arr]:
        if self.use_eval:
            deterministic=True
        else:
            deterministic=False
        self.trainer.prep_rollout()
        (
            value,
            cost_preds,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
            rnn_states_cost
        ) = self.trainer.policy.get_actions(
            self.buffer.graph_storage[step],
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.rnn_states_cost[step]),
            np.concatenate(self.buffer.masks[step]),
            deterministic=deterministic
        )

        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        cost_preds = np.array(np.split(_t2n(cost_preds), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )
        rnn_states_cost = np.array(
            np.split(_t2n(rnn_states_cost), self.n_rollout_threads)
        )

        return (
            values,
            cost_preds,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            rnn_states_cost
        )

    def insert(self, data):
        (
            batch,
            agent_id,
            rewards,
            costs,
            dones,
            infos,
            values,
            cost_preds,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            rnn_states_cost
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        self.buffer.insert(
            batch,
            agent_id,
            rnn_states,
            rnn_states_critic,
            rnn_states_cost,
            actions,
            action_log_probs,
            values,
            cost_preds,
            rewards,
            costs,
            masks,
        )

    @torch.no_grad()
    def compute(self):
        """Calculate returns for the collected data."""
        self.trainer.prep_rollout()
        next_values = self.trainer.policy.get_values(
            self.buffer.graph_storage[-1],
            np.concatenate(self.buffer.rnn_states_critic[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

        next_costs = self.trainer.policy.get_cost_values(
            self.buffer.graph_storage[-1],
            np.concatenate(self.buffer.rnn_states_cost[-1]),
            np.concatenate(self.buffer.masks[-1]),
        )
        next_costs = np.array(np.split(_t2n(next_costs), self.n_rollout_threads))
        self.buffer.compute_cost_returns(next_costs, self.trainer.cost_normalizer)
        self.buffer.fix_ue_ids()

    @torch.no_grad()
    def eval(self, total_num_steps: int):
        raise
        eval_episode_rewards = []
        eval_obs, eval_agent_id, eval_node_obs, eval_adj = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
        )

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_obs),
                np.concatenate(eval_node_obs),
                np.concatenate(eval_adj),
                np.concatenate(eval_agent_id),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_action), self.n_eval_rollout_threads)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads)
            )

            if self.eval_envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                for i in range(self.eval_envs.action_space[0].shape):
                    eval_uc_actions_env = np.eye(
                        self.eval_envs.action_space[0].high[i] + 1
                    )[eval_actions[:, :, i]]
                    if i == 0:
                        eval_actions_env = eval_uc_actions_env
                    else:
                        eval_actions_env = np.concatenate(
                            (eval_actions_env, eval_uc_actions_env), axis=2
                        )
            elif self.eval_envs.action_space[0].__class__.__name__ == "Discrete":
                eval_actions_env = np.squeeze(
                    np.eye(self.eval_envs.action_space[0].n)[eval_actions], 2
                )
            else:
                raise NotImplementedError

            # Obser reward and next obs
            (
                eval_obs,
                eval_agent_id,
                eval_node_obs,
                eval_adj,
                eval_rewards,
                eval_dones,
                eval_infos,
            ) = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), 1), dtype=np.float32
            )

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {}
        eval_env_infos["eval_average_episode_rewards"] = np.sum(
            np.array(eval_episode_rewards), axis=0
        )
        eval_average_episode_rewards = np.mean(
            eval_env_infos["eval_average_episode_rewards"]
        )
        print(
            "eval average episode rewards of agent: "
            + str(eval_average_episode_rewards)
        )
        self.log_env(eval_env_infos, total_num_steps)

    def log_env(self, env_infos, total_num_steps, prefix=""):
        """
        Log env info.
        :param env_infos: (dict) information about env state.
        :param total_num_steps: (int) total number of training env steps.
        """
        info_keys = env_infos[0].keys()
        for k in info_keys:
            v = []
            for info in env_infos:
                v.append(info[k])
            v = torch.stack(v)
            if k == 'se':
                for i in range(v.shape[1]):
                    self.writter.add_scalar(f"{k}/ue_{i}", v[:, i].mean(), total_num_steps)
            elif k == "serving_ap_count":
                for i in range(v.shape[1]):
                    self.writter.add_scalar(f"{k}/ue_{i}", v[:, i].mean(), total_num_steps)
            elif k == "served_ue_count":
                for i in range(v.shape[1]):
                    self.writter.add_scalar(f"{k}/ap_{i}", v[:, i].mean(), total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v.mean()}, total_num_steps)
