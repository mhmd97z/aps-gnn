import numpy as np
import torch
import torch.nn as nn
from onpolicy.utils.util import get_grad_norm, huber_loss, mse_loss
from onpolicy.algorithms.mat.utils.valuenorm import ValueNorm
from onpolicy.algorithms.mat.utils.util import check


class MATLTrainer:
    """
    Trainer class for MAT to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """
    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device("cpu")):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm       
        self.huber_delta = args.huber_delta

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self.dec_actor = args.dec_actor
        
        self.n_ues = args.env_args.simulation_scenario.number_of_ues
        self.lagrangian_coef = args.lagrangian_coef_rate
        self.lamda_lagr = torch.full((self.n_ues,), args.lamda_lagr, **self.tpdv)
        self.safety_bound = torch.tensor(0.0).to(**self.tpdv)
        self.if_per_ue = args.if_update_lagr_per_ue
        self.lambda_max = args.lambda_lagr_max

        if self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device=self.device)
            self.cost_normalizer = ValueNorm(1, device=self.device)
        else:
            self.value_normalizer = None
            self.cost_normalizer = None

        if args.if_pid_lagr_update:
            self.if_pid_lagr_update = True

            self.pid_ki = args.lagr_pid_ki
            self.pid_i = torch.full((self.n_ues,), args.lamda_lagr, **self.tpdv)

            self.pid_kp = args.lagr_pid_kp
            self.pid_p = torch.full((self.n_ues,), args.lamda_lagr, **self.tpdv)
            self.pid_p_avg_alpha = torch.tensor(0.90).to(**self.tpdv)   # 0 for hard update, 1 for no update.

            self.pid_kd = args.lagr_pid_kd
            self.cost_d = torch.zeros(self.n_ues).to(**self.tpdv)
            self.pid_d_avg_alpha = torch.tensor(0.90).to(**self.tpdv)   # 0 for hard update, 1 for no update.
        else:
            self.if_pid_lagr_update = False


    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """

        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)

        if self._use_valuenorm:
            self.value_normalizer.update(return_batch)
            error_clipped = self.value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = self.value_normalizer.normalize(return_batch) - values
        else:
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        # if self._use_value_active_masks and not self.dec_actor:
        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss


    def update_lagrangian_pid(self, cost, ue_idx):
        if self.if_per_ue:
            cost = cost.flatten()
            ue_idx = ue_idx.flatten()
            unique_indices, inverse = torch.unique(ue_idx, return_inverse=True, sorted=True)
            unique_indices = unique_indices.to(self.device)
            inverse = inverse.to(self.device)
            sum_per_ue = torch.zeros(unique_indices.shape[0], dtype=cost.dtype, device=self.device).scatter_add_(0, inverse, cost)
            count_per_ue = torch.zeros(unique_indices.shape[0], dtype=cost.dtype, device=self.device).scatter_add_(0, inverse, torch.ones_like(cost, dtype=cost.dtype))
            cost_mean_per_ue = sum_per_ue / count_per_ue

            delta = cost_mean_per_ue - self.safety_bound
            self.pid_p = self.pid_p * self.pid_p_avg_alpha + delta * (1 - self.pid_p_avg_alpha)
            self.pid_i = (self.pid_i + delta * self.pid_ki).clamp(0.0, self.lambda_max)
            cost_d = self.cost_d * self.pid_d_avg_alpha + cost_mean_per_ue * (1 - self.pid_d_avg_alpha)
            pid_d = (cost_d - self.cost_d).clamp(0.0, self.lambda_max)
            self.cost_d = cost_d
            self.lamda_lagr = (self.pid_kp * self.pid_p + self.pid_i + self.pid_kd * pid_d).clamp(0.0, self.lambda_max)
        else:
            cost_mean = cost.mean()
            delta = cost_mean - self.safety_bound
            self.pid_p = self.pid_p * self.pid_p_avg_alpha + delta * (1 - self.pid_p_avg_alpha)
            self.pid_i = (self.pid_i + delta * self.pid_ki).clamp(0.0, self.lambda_max)
            cost_d = self.cost_d * self.pid_d_avg_alpha + cost_mean * (1 - self.pid_d_avg_alpha)
            pid_d = (cost_d - self.cost_d).clamp(0.0, self.lambda_max)
            self.cost_d = cost_d
            self.lamda_lagr = (self.pid_kp * self.pid_p + self.pid_i + self.pid_kd * pid_d).clamp(0.0, self.lambda_max)


    def update_lagrangian_simple(self, cost, ue_idx):
        if self.if_per_ue:
            cost = cost.flatten()
            ue_idx = ue_idx.flatten()
            unique_indices, inverse = torch.unique(ue_idx, return_inverse=True, sorted=True)
            unique_indices = unique_indices.to(self.device)
            inverse = inverse.to(self.device)
            sum_per_ue = torch.zeros(unique_indices.shape[0], dtype=cost.dtype, device=self.device).scatter_add_(0, inverse, cost)
            count_per_ue = torch.zeros(unique_indices.shape[0], dtype=cost.dtype, device=self.device).scatter_add_(0, inverse, torch.ones_like(cost, dtype=cost.dtype))
            cost_mean_per_ue = sum_per_ue / count_per_ue
            self.lamda_lagr += self.lagrangian_coef * (cost_mean_per_ue - self.safety_bound) # we assume gamma is very small
            self.lamda_lagr = torch.clamp(self.lamda_lagr, 0.0, self.lambda_max)
        else:
            cost_mean = cost.mean()
            self.lamda_lagr += self.lagrangian_coef * (cost_mean - self.safety_bound) # we assume gamma is very small
            self.lamda_lagr = torch.clamp(self.lamda_lagr, 0.0, self.lambda_max)

    def ppol_update(self, sample):
        """
        Update actor and critic networks.
        :param sample: (Tuple) contains data batch with which to update networks.
        :update_actor: (bool) whether to update actor network.

        :return value_loss: (torch.Tensor) value function loss.
        :return critic_grad_norm: (torch.Tensor) gradient norm from critic up9date.
        ;return policy_loss: (torch.Tensor) actor(policy) loss value.
        :return dist_entropy: (torch.Tensor) action entropies.
        :return actor_grad_norm: (torch.Tensor) gradient norm from actor update.
        :return imp_weights: (torch.Tensor) importance sampling weights.
        """
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, rnn_states_cost_batch, actions_batch, \
        value_preds_batch, cost_preds_batch, return_batch, cost_return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, cost_adv_targ, available_actions_batch, ue_idx_batch = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        cost_adv_targ = check(cost_adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        cost_preds_batch = check(cost_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        cost_return_batch = check(cost_return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        ue_idx_batch = check(ue_idx_batch).to(self.device).to(torch.long)

        # Reshape to do in a single forward pass for all steps
        values, cost_values, action_log_probs, dist_entropy = self.policy.evaluate_actions(share_obs_batch,
                                                                              obs_batch,
                                                                              rnn_states_batch,
                                                                              rnn_states_critic_batch,
                                                                              rnn_states_cost_batch,
                                                                              actions_batch,
                                                                              masks_batch, 
                                                                              available_actions_batch,
                                                                              active_masks_batch)
        # actor update
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        lambda_val = self.lamda_lagr[ue_idx_batch].mean().item()
        lambda_vec = torch.full_like(cost_adv_targ, lambda_val)

        adv_targ_hybrid = (adv_targ - lambda_vec * cost_adv_targ)

        surr1 = imp_weights * adv_targ_hybrid
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_hybrid

        if self._use_policy_active_masks:
            policy_loss = (-torch.sum(torch.min(surr1, surr2),
                                      dim=-1,
                                      keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        # critic update
        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)

        # cost critic update
        cost_value_loss = self.cal_value_loss(cost_values, cost_preds_batch, cost_return_batch, active_masks_batch)

        loss = policy_loss - dist_entropy * self.entropy_coef + value_loss * self.value_loss_coef + cost_value_loss * self.value_loss_coef

        self.policy.optimizer.zero_grad()
        loss.backward()

        if self._use_max_grad_norm:
            grad_norm = nn.utils.clip_grad_norm_(self.policy.transformer.parameters(), self.max_grad_norm)
        else:
            grad_norm = get_grad_norm(self.policy.transformer.parameters())

        self.policy.optimizer.step()

        if self.if_pid_lagr_update:
            self.update_lagrangian_pid(cost_return_batch, ue_idx_batch)
        else:
            self.update_lagrangian_simple(cost_return_batch, ue_idx_batch)

        return value_loss, cost_value_loss, grad_norm, policy_loss, dist_entropy, imp_weights

    def train(self, buffer):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        advantages_copy = buffer.advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (buffer.advantages - mean_advantages) / (std_advantages + 1e-5)
        
        cost_advantages_copy = buffer.cost_advantages.copy()
        cost_advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_cost_advantages = np.nanmean(cost_advantages_copy)
        std_cost_advantages = np.nanstd(cost_advantages_copy)
        cost_advantages = (buffer.cost_advantages - mean_cost_advantages) / (std_cost_advantages + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['cost_value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['lamda_lagr'] = self.lamda_lagr.mean().item()

        for _ in range(self.ppo_epoch):
            data_generator = buffer.feed_forward_generator_transformer(advantages, cost_advantages, self.num_mini_batch)

            for sample in data_generator:

                value_loss, cost_value_loss, policy_loss, dist_entropy, actor_grad_norm, imp_weights \
                    = self.ppol_update(sample)

                train_info['value_loss'] += value_loss.item()
                train_info['cost_value_loss'] += cost_value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['lamda_lagr'] += self.lamda_lagr.mean().item()

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates
 
        return train_info

    def prep_training(self):
        self.policy.train()

    def prep_rollout(self):
        self.policy.eval()
