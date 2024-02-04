"""Based on https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/r_mappo.py. """
import numpy as np
import torch
import torch.nn as nn
from src.algo.utils.util import check
from src.utils.util import get_grad_norm, huber_loss, mse_loss
from src.utils.valuenorm import ValueNorm


class R_MAPPO(object):
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        
        # common parameters
        self.clip_param = args.clip_param
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.entropy_coef = args.entropy_coef
        self.value_loss_coef = args.value_loss_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.ppo_epoch = args.ppo_epoch
        
        self.novel_type = args.novel_type   # 1: prediction error; 2: TD error; 3: cnount-based; 4. RND
        assert self.novel_type in [0, 1, 2, 3, 4], "novel_type must be 0, 1, 2, 3 or 4"
        # r + c_1 * u_s + c_2 * u_o (+ c_3 * mi)
        self.self_coef = args.self_coef
        self.other_coef = args.other_coef
        self.novel_max = args.novel_max # maximum over all agents' novelty
        self.ir_coef = args.ir_coef     # hindsight term as intrinsic reward
        self.ad_coef = args.ad_coef     # hindsight term as advantage
        
        self.use_hdd = args.use_hdd
        self.hdd_log_weight = args.hdd_log_weight
        self.hdd_reduce = args.hdd_reduce
        self.hdd_count = args.hdd_count
        self.hdd_epoch = args.hdd_epoch
        self.hdd_batch_size = args.hdd_batch_size
        self.hdd_ratio_clip = args.hdd_ratio_clip
        self.hdd_norm_adv = args.hdd_norm_adv
        # mutual information between a and g (not weighted mutual information)
        self.hdd_weight_only = args.hdd_weight_only
        # return of others' novelty
        self.hdd_weight_one = args.hdd_weight_one
        
        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks
        self._use_novel_active_masks = args.use_novel_active_masks
        self._use_hdd_active_masks = args.use_hdd_active_masks
        
        assert (self._use_popart and self._use_valuenorm) == False, \
            ("self._use_popart and self._use_valuenorm can not be set True simultaneously")
        
        if self._use_popart:
            self.value_normalizer = self.policy.critic.v_out
        elif self._use_valuenorm:
            self.value_normalizer = ValueNorm(1, device = self.device)
        else:
            self.value_normalizer = None
                
    def cal_value_error(self, values_pred, values, active_masks_batch):
        value_error = (values_pred - values).pow(2)
        
        if self._use_value_active_masks:
            value_error = (value_error * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_error = value_error.mean()
        
        return value_error
        
    def cal_value_loss(self, values, value_preds_batch, value_normalizer, return_batch, active_masks_batch):
        value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                    self.clip_param)
        if self._use_popart or self._use_valuenorm:
            value_normalizer.update(return_batch)
            error_clipped = value_normalizer.normalize(return_batch) - value_pred_clipped
            error_original = value_normalizer.normalize(return_batch) - values
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

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss
    
    def cal_policy_loss(self, adv, action_log_probs, dist_entropy, old_action_log_probs_batch,
                        active_masks_batch):
        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)
        surr1 = imp_weights * adv
        surr2 = torch.clamp(imp_weights, 1.0-self.clip_param, 1.0+self.clip_param) * adv
        approx_kl = (old_action_log_probs_batch - action_log_probs).mean().item()
        
        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(torch.min(surr1, surr2), dim=-1, keepdim=True).mean()
        
        policy_loss = policy_action_loss - self.entropy_coef * dist_entropy
        
        return policy_loss, policy_action_loss, imp_weights, approx_kl
    
    def cal_rnd_loss(self, obs_batch, active_masks_batch):
        predict_features, target_features = self.policy.rnd(obs_batch)
        rnd_loss = torch.square(predict_features - target_features).mean(axis=1, keepdim=True)
        
        if self._use_novel_active_masks:
            rnd_loss = (rnd_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            rnd_loss = rnd_loss.mean()
        
        return rnd_loss

    def ppo_update(self, sample):
        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, \
        return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch \
        = sample
        
        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        
        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(
            share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch,
            actions_batch, masks_batch, available_actions_batch, active_masks_batch)
    
        # update policy
        policy_loss, policy_action_loss, imp_weights, _ = self.cal_policy_loss(
            adv_targ, action_log_probs, dist_entropy, old_action_log_probs_batch, active_masks_batch)
        
        self.policy.actor_optimizer.zero_grad()
        policy_loss.backward()
        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_grad_norm(self.policy.actor.parameters())
        self.policy.actor_optimizer.step()
        
        # update critic
        value_loss = self.cal_value_loss(values, value_preds_batch, self.value_normalizer, return_batch,
                                         active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_grad_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()
        
        # update rnd
        if self.policy.novel_type == 4:
            rnd_loss = self.cal_rnd_loss(obs_batch, active_masks_batch)
            self.policy.rnd_optimizer.zero_grad()
            rnd_loss.backward()
            if self._use_max_grad_norm:
                rnd_grad_norm = nn.utils.clip_grad_norm_(self.policy.rnd.parameters(), self.max_grad_norm)
            else:
                rnd_grad_norm = get_grad_norm(self.policy.rnd.parameters())
            self.policy.rnd_optimizer.step()
        
            return value_loss, critic_grad_norm, policy_action_loss, dist_entropy, actor_grad_norm, imp_weights, \
                   rnd_loss, rnd_grad_norm
        
        return value_loss, critic_grad_norm, policy_action_loss, dist_entropy, actor_grad_norm, imp_weights, None, None

    def hdd_update(self, buffer, buffer_hdd):
        if not self.use_hdd:
            return {}
        
        if buffer.dnovels is None:
            rewards_o = np.concatenate([buffer.novels[:buffer.agent_id, :-1], 
                                        buffer.novels[buffer.agent_id+1:, :-1]], axis=0)
        else:
            rewards_o = np.concatenate([buffer.dnovels[:buffer.agent_id, :-1], 
                                        buffer.dnovels[buffer.agent_id+1:, :-1]], axis=0)
        if self.hdd_reduce:
            rewards_o = np.sum(rewards_o, axis=0)
            returns_oh = buffer.returns_oh[:-1]
            returns_oh_reverse = buffer.returns_oh_reverse[:-1]
        else:
            returns_oh = buffer.returns_oh[:, :-1]
            returns_oh_reverse = buffer.returns_oh_reverse[:, :-1]
        
        if buffer.available_actions is None:
            available_actions = None
        else:
            available_actions = buffer.available_actions[:-1]
        if self.hdd_count:
            return self.policy.hdd.update(buffer.obs[:-1], buffer.actions, returns_oh, rewards_o,
                                          returns_oh_reverse, available_actions, buffer.active_masks[:-1])
        return self.policy.hdd.update(buffer_hdd, buffer.obs[:-1], buffer.actions, returns_oh, rewards_o,
                                      returns_oh_reverse, available_actions, buffer.active_masks[:-1])
    
    def cal_hdd_ratio(self, buffer):
        if buffer.dnovels is None:
            rewards_o = np.concatenate([buffer.novels[:buffer.agent_id, :-1], 
                                        buffer.novels[buffer.agent_id+1:, :-1]], axis=0)
        else:
            rewards_o = np.concatenate([buffer.dnovels[:buffer.agent_id, :-1], 
                                        buffer.dnovels[buffer.agent_id+1:, :-1]], axis=0)
            
        if self.hdd_reduce:
            rewards_o = np.sum(rewards_o, axis=0)
            returns_oh = buffer.returns_oh[:-1]
            returns_oh_reverse = buffer.returns_oh_reverse[:-1]
        else:
            returns_oh = buffer.returns_oh[:, :-1]
            returns_oh_reverse = buffer.returns_oh_reverse[:, :-1]
            
        h_probs = self.policy.hdd.evaluate_actions(buffer.obs[:-1], buffer.actions, returns_oh, rewards_o,
                                                   returns_oh_reverse, buffer.available_actions[:-1])
        a_probs = np.exp(buffer.action_log_probs)
        if self.hdd_ratio_clip is not None:
            buffer.ratios[:] = np.clip(a_probs / h_probs, 0., self.hdd_ratio_clip)
        else:
            buffer.ratios[:] = a_probs / h_probs
    
    def cal_hdd_advantage(self, buffer):
        if not self.use_hdd:
            return
                
        self.cal_hdd_ratio(buffer)
        
        weight = -np.log(buffer.ratios + 1e-6) if self.hdd_log_weight else (1 - buffer.ratios)
        if self.hdd_weight_one:
            weight = np.ones_like(weight)
        
        if self.hdd_weight_only:
            hdd_advantages = weight
        else:
            returns = buffer.returns_oa[:-1] if self.hdd_reduce else buffer.returns_oa[:, :-1]
            hdd_advantages = weight * returns
        
        if self._use_hdd_active_masks:
            hdd_advantages *= buffer.active_masks[:-1]
        
        buffer.hdd_advantages = hdd_advantages.copy() if self.hdd_reduce else hdd_advantages.sum(axis=0)
    
    def train(self, buffer):
        if self._use_popart or self._use_valuenorm:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        
        train_info = {}
        
        if self.use_hdd and self.ad_coef > 0.0:
            hdd_advantages = buffer.hdd_advantages.copy()
            
            if self.hdd_norm_adv:
                # normalize hdd_adv and adv respectively, then add them
                hdd_advantages_copy = hdd_advantages.copy()
                hdd_advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
                mean_hdd_advantages = np.nanmean(hdd_advantages_copy)
                std_hdd_advantages = np.nanstd(hdd_advantages_copy)
                hdd_advantages = (hdd_advantages - mean_hdd_advantages) / (std_hdd_advantages + 1e-5)
                
                advantages_copy = advantages.copy()
                advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
                mean_advantages = np.nanmean(advantages_copy)
                std_advantages = np.nanstd(advantages_copy)
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
                
                advantages += hdd_advantages * self.ad_coef
            else:
                # add hdd_adv and adv, then normalization
                advantages += hdd_advantages * self.ad_coef
                
                advantages_copy = advantages.copy()
                advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
                mean_advantages = np.nanmean(advantages_copy)
                std_advantages = np.nanstd(advantages_copy)
                advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        else:
            advantages_copy = advantages.copy()
            advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
            mean_advantages = np.nanmean(advantages_copy)
            std_advantages = np.nanstd(advantages_copy)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)
        
        train_info['value_loss'] = []
        train_info['policy_loss'] = []
        train_info['dist_entropy'] = []
        train_info['actor_grad_norm'] = []
        train_info['critic_grad_norm'] = []
        train_info['ratio'] = []
        
        for _ in range(self.ppo_epoch):
            if self._use_recurrent_policy:
                data_generator = buffer.recurrent_generator(advantages, self.num_mini_batch, self.data_chunk_length)
            elif self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch)

            for sample in data_generator:
                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, \
                rnd_loss, rnd_grad_norm = self.ppo_update(sample)
                
                train_info['value_loss'].append(value_loss.item())
                train_info['policy_loss'].append(policy_loss.item())
                train_info['dist_entropy'].append(dist_entropy.item())
                train_info['actor_grad_norm'].append(actor_grad_norm.item())
                train_info['critic_grad_norm'].append(critic_grad_norm.item())
                train_info['ratio'].append(imp_weights.mean().item())
                
                if rnd_loss is not None:
                    if 'rnd_loss' not in train_info:
                        train_info['rnd_loss'] = []
                        train_info['rnd_grad_norm'] = []
                    train_info['rnd_loss'].append(rnd_loss.item())
                    train_info['rnd_grad_norm'].append(rnd_grad_norm.item())

        for k in train_info.keys():
            train_info[k] = np.mean(train_info[k])
        
        return train_info
        
    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
