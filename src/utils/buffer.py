import numpy as np
import torch
from src.utils.util import get_shape_from_act_space, get_shape_from_obs_space


def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:])


class ReplayBuffer(object):
    def __init__(self, idx, args, obs_space, share_obs_space, act_space):
        self.agent_id = idx
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.n_agents_in_env = args.n_agents
        self.rnn_hidden_size = args.hidden_size
        self.recurrent_N = args.recurrent_N
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda
        self._use_gae = args.use_gae
        self._use_popart = args.use_popart
        self._use_valuenorm = args.use_valuenorm
        self._use_proper_time_limits = args.use_proper_time_limits

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        T, N = self.episode_length, self.n_rollout_threads
        self.share_obs = np.zeros((T + 1, N, *share_obs_shape), dtype=np.float32)
        self.obs = np.zeros((T + 1, N, *obs_shape), dtype=np.float32)

        self.rnn_states = np.zeros((T + 1, N, self.recurrent_N, self.rnn_hidden_size), dtype=np.float32)
        self.rnn_states_critic = np.zeros_like(self.rnn_states)

        self.value_preds = np.zeros((T + 1, N, 1), dtype=np.float32)
        self.returns = np.zeros((T + 1, N, 1), dtype=np.float32)
        
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.episode_length + 1, N, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = np.zeros((T, N, act_shape), dtype=np.float32)
        self.action_log_probs = np.zeros((T, N, act_shape), dtype=np.float32)
        self.ext_rewards = np.zeros((T, N, 1), dtype=np.float32)
        self.int_rewards = np.zeros((T, N, 1), dtype=np.float32)
        
        self.masks = np.ones((T + 1, N, 1), dtype=np.float32)
        self.bad_masks = np.ones_like(self.masks)
        self.active_masks = np.ones_like(self.masks)
        
        self.novels = np.zeros((self.n_agents_in_env, T + 1, N, 1), dtype=np.float32)

        self.step = 0
    
    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs, value_preds,
               rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.rnn_states[self.step + 1] = rnn_states.copy()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.ext_rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length
    
    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.rnn_states[0] = self.rnn_states[-1].copy()
        self.rnn_states_critic[0] = self.rnn_states_critic[-1].copy()
        self.masks[0] = self.masks[-1].copy()
        self.bad_masks[0] = self.bad_masks[-1].copy()
        self.active_masks[0] = self.active_masks[-1].copy()
        if self.available_actions is not None:
            self.available_actions[0] = self.available_actions[-1].copy()
    
    def compute_novel(self, visit_count, lookup_index, pow):
        if (visit_count == 0).all():
            return
        visit_count_lookup = self.obs[..., lookup_index: lookup_index + 2].copy()
        row, col = visit_count.shape
        visit_count_lookup[..., 0] *= row
        visit_count_lookup[..., 1] *= col
        visit_count_lookup = visit_count_lookup.astype(np.int32)
        novel = np.power(np.maximum(visit_count[visit_count_lookup[..., 0], visit_count_lookup[..., 1]], 1), -pow)
        self.novels[self.agent_id] = novel.reshape(self.novels[self.agent_id].shape)
    
    def set_all_novels(self, all_novels):
        self.novels[:] = all_novels.copy()
        
    def compute_returns(self, self_coef=0.0, other_coef=0.0, value_normalizer=None, novel_max=False, **kwargs):
        if self._use_gae:
            gae = 0.0
            for step in reversed(range(self.episode_length)):
                proxy_reward = self.ext_rewards[step].copy() 
                if novel_max:
                    proxy_reward += self_coef * self.novels[:, step + 1].max(axis=0) * self.masks[step + 1]
                else:
                    proxy_reward += self_coef * self.novels[self.agent_id, step + 1] * self.masks[step + 1]
                    proxy_reward += other_coef * (self.novels[:, step + 1].sum(axis=0) - 
                                                  self.novels[self.agent_id, step + 1]) * self.masks[step + 1]
                self.int_rewards[step] = proxy_reward - self.ext_rewards[step]
                    
                if self._use_popart or self._use_valuenorm:
                    delta = proxy_reward + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) \
                        * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    if self._use_proper_time_limits:
                        gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = proxy_reward + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] \
                        - self.value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    if self._use_proper_time_limits:
                        gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = self.value_preds[-1].copy()
            for step in reversed(range(self.episode_length)):
                proxy_reward = self.ext_rewards[step].copy()
                if novel_max:
                    proxy_reward += self_coef * self.novels[:, step + 1].max(axis=0) * self.masks[step + 1]
                else:
                    proxy_reward += self_coef * self.novels[self.agent_id, step + 1] * self.masks[step + 1]
                    proxy_reward += other_coef * (self.novels[:, step + 1].sum(axis=0) - 
                                                  self.novels[self.agent_id, step + 1]) * self.masks[step + 1]
                self.int_rewards[step] = proxy_reward - self.ext_rewards[step]
                    
                if self._use_proper_time_limits:
                    if self._use_popart:
                        self.returns[step] = (proxy_reward + self.gamma * self.returns[step + 1]) * \
                            self.bad_masks[step + 1] + value_normalizer.denormalize(self.value_preds[step]) * \
                            (1 - self.bad_masks[step + 1])
                    else:
                        self.returns[step] = (proxy_reward + self.gamma * self.returns[step + 1]) * \
                            self.bad_masks[step + 1] + self.value_preds[step] * (1 - self.bad_masks[step + 1])
                else:
                    self.returns[step] = proxy_reward + self.gamma * self.returns[step + 1] * self.masks[step + 1]
    
    def compute_other_novel_returns(self):
        pass
    
    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None):
        episode_length, n_rollout_threads = self.ext_rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch
        else:
            num_mini_batch = batch_size // mini_batch_size

        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        advantages = advantages.reshape(-1, 1)

        for indices in sampler:
            # obs size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[index,Dim]
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch
                  
    def naive_recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.ext_rewards.shape[1]
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch))
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads).numpy()
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs[:-1, ind])
                obs_batch.append(self.obs[:-1, ind])
                rnn_states_batch.append(self.rnn_states[0:1, ind])
                rnn_states_critic_batch.append(self.rnn_states_critic[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(self.available_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = np.stack(share_obs_batch, 1)
            obs_batch = np.stack(obs_batch, 1)
            actions_batch = np.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, 1)
            value_preds_batch = np.stack(value_preds_batch, 1)
            return_batch = np.stack(return_batch, 1)
            masks_batch = np.stack(masks_batch, 1)
            active_masks_batch = np.stack(active_masks_batch, 1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, 1)
            adv_targ = np.stack(adv_targ, 1)

            # States is just a (N, -1) from_numpy [N[1,dim]]
            rnn_states_batch = np.stack(rnn_states_batch, 1).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch, 1).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch
                  
    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.ext_rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")

        rand = torch.randperm(data_chunks).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 3:
            share_obs = self.share_obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.share_obs.shape[2:])
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states_critic.shape[2:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []

            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                obs_batch.append(obs[ind:ind+data_chunk_length])
                actions_batch.append(actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])

            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            # share_obs_batch = np.stack(share_obs_batch)
            # obs_batch = np.stack(obs_batch)

            # actions_batch = np.stack(actions_batch)
            # if self.available_actions is not None:
            #     available_actions_batch = np.stack(available_actions_batch)
            # value_preds_batch = np.stack(value_preds_batch)
            # return_batch = np.stack(return_batch)
            # masks_batch = np.stack(masks_batch)
            # active_masks_batch = np.stack(active_masks_batch)
            # old_action_log_probs_batch = np.stack(old_action_log_probs_batch)
            # adv_targ = np.stack(adv_targ)

            # These are all from_numpys of size (L, N, Dim)
            share_obs_batch = np.stack(share_obs_batch, axis=1)
            obs_batch = np.stack(obs_batch, axis=1)

            actions_batch = np.stack(actions_batch, axis=1)
            if self.available_actions is not None:
                available_actions_batch = np.stack(available_actions_batch, axis=1)
            value_preds_batch = np.stack(value_preds_batch, axis=1)
            return_batch = np.stack(return_batch, axis=1)
            masks_batch = np.stack(masks_batch, axis=1)
            active_masks_batch = np.stack(active_masks_batch, axis=1)
            old_action_log_probs_batch = np.stack(old_action_log_probs_batch, axis=1)
            adv_targ = np.stack(adv_targ, axis=1)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = np.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = np.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (L, N, ...) from_numpys to (L * N, ...)
            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)

            yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
                  value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
                  adv_targ, available_actions_batch


class Disc(object):
    def __init__(self, nbins, momentum=0.99):
        self.nbins = nbins
        self.bounds = np.zeros((nbins + 1,), dtype=np.float32)
        self.bounds[-1] = 1.0
        self.momentum = momentum
        self.init_flag = True
        
    def update(self, novels):
        novels_ = novels.copy().flatten()
        novels_.sort()
        indices = np.linspace(0, len(novels_) - 1, self.nbins + 1).astype(np.int32)
        bounds = novels_[indices][1:-1]
        
        if self.init_flag:
            self.bounds[1:-1] = bounds
            self.init_flag = False
        else:
            self.bounds[1:-1] = self.bounds[1:-1] * self.momentum + bounds * (1 - self.momentum)
        novels_in_bins = np.digitize(novels, self.bounds, right=True) - 1
        return (novels_in_bins + 0.5) / self.nbins


class HindsightReplayBuffer(ReplayBuffer):
    def __init__(self, idx, args, obs_space, share_obs_space, act_space):
        super().__init__(idx, args, obs_space, share_obs_space, act_space)
        
        T, N = self.episode_length, self.n_rollout_threads
        self.novels = np.zeros((self.n_agents_in_env, T + 1, N, 1), dtype=np.float32)
        
        self.disc_na = args.discrete_novel_in_adv
        self.disc_nh = args.discrete_novel_in_hd
        self.disc, self.dnovels = None, None
        if self.disc_na or self.disc_nh:
            self.disc = [Disc(args.discrete_nbins, args.discrete_momentum) for _ in range(self.n_agents_in_env)]
            self.dnovels = np.zeros_like(self.novels)
        self.hdd_ret_padding = args.hdd_ret_padding
        self.hdd_gamma = args.hdd_gamma
        
        self.other_reduce = args.hdd_reduce
        if self.other_reduce:
            self.ratios = np.zeros((T, N, 1), dtype=np.float32)
            self.returns_oa = np.zeros_like(self.returns)   # return of others in adv
            self.returns_oh = np.zeros_like(self.returns)   # return of others in ratio
            self.returns_oh_reverse = np.zeros_like(self.returns)
        else:
            self.ratios = np.zeros((self.n_agents_in_env - 1, T, N, 1), dtype=np.float32)
            self.returns_oa = np.zeros((self.n_agents_in_env - 1, T + 1, N, 1), dtype=np.float32)
            self.returns_oh = np.zeros_like(self.returns_oa)
            self.returns_oh_reverse = np.zeros_like(self.returns_oa)
        self.hdd_advantages = np.zeros((T, N, 1), dtype=np.float32)
        
    def set_all_novels(self, all_novels):
        super().set_all_novels(all_novels)
        if self.disc is not None:
            for agent_id in range(self.n_agents_in_env):
                dnovel = self.disc[agent_id].update(self.novels[agent_id])
                self.dnovels[agent_id] = dnovel.reshape(self.dnovels[agent_id].shape)
    
    def compute_reduced_other_novel_returns(self):
        reduced_novels = np.concatenate([self.novels[:self.agent_id], self.novels[self.agent_id+1:]], axis=0).sum(axis=0)
        if self.dnovels is not None:
            reduced_dnovels = np.concatenate([self.dnovels[:self.agent_id], self.dnovels[self.agent_id+1:]], axis=0).sum(axis=0)
        reduced_na = reduced_dnovels if self.disc_na else reduced_novels
        reduced_nh = reduced_dnovels if self.disc_nh else reduced_novels
        
        if self.hdd_ret_padding:
            # self.returns_oa[-1] = reduced_na[-1] / (1 - self.gamma)
            # self.returns_oh[-1] = reduced_nh[-1] / (1 - self.hdd_gamma)
            # average over last 10 steps
            self.returns_oa[-1] = reduced_na[-10:].mean(axis=0) / (1 - self.gamma)
            self.returns_oh[-1] = reduced_nh[-10:].mean(axis=0) / (1 - self.hdd_gamma)
        
        for step in reversed(range(self.episode_length)):
            # returns of others in adv
            proxy_reward = reduced_na[step + 1] * self.masks[step + 1]
            ret = proxy_reward + self.gamma * self.returns_oa[step + 1] * self.masks[step + 1]
            if self.hdd_ret_padding:
                pad_indices = self.masks[step + 1] == 0
                if pad_indices.sum() > 0:
                    # ret[pad_indices] = reduced_na[step, pad_indices] / (1 - self.gamma)
                    ret[pad_indices] = reduced_na[max(0, step - 9): step + 1, pad_indices].mean(axis=0) / (1 - self.gamma)
            self.returns_oa[step] = ret
            
            # returns of others in ratio
            proxy_reward = reduced_nh[step + 1] * self.masks[step + 1]
            ret = proxy_reward + self.hdd_gamma * self.returns_oh[step + 1] * self.masks[step + 1]
            if self.hdd_ret_padding:
                pad_indices = self.masks[step + 1] == 0
                if pad_indices.sum() > 0:
                    # ret[pad_indices] = reduced_nh[step, pad_indices] / (1 - self.hdd_gamma)
                    ret[pad_indices] = reduced_nh[max(0, step - 9): step + 1, pad_indices].mean(axis=0) / (1 - self.hdd_gamma)
            self.returns_oh[step] = ret
        
        # computer reversed returns
        for step in range(self.episode_length):
            proxy_reward = reduced_nh[step].copy()
            if step == 0:
                ret = proxy_reward
                if self.hdd_ret_padding:
                    ret = proxy_reward / (1 - self.hdd_gamma)
            else:
                ret = proxy_reward + self.hdd_gamma * self.returns_oh_reverse[step - 1] * self.masks[step]
                if self.hdd_ret_padding:
                    pad_indices = self.masks[step] == 0
                    if pad_indices.sum() > 0:
                        ret[pad_indices] = proxy_reward[pad_indices] / (1 - self.hdd_gamma)
                        
            self.returns_oh_reverse[step] = ret
            
    def compute_separate_other_novel_returns(self):
        other_novels = np.concatenate([self.novels[:self.agent_id], self.novels[self.agent_id+1:]], axis=0)
        if self.dnovels is not None:
            other_dnovels = np.concatenate([self.dnovels[:self.agent_id], self.dnovels[self.agent_id+1:]], axis=0)
        other_na = other_dnovels if self.disc_na else other_novels
        other_nh = other_dnovels if self.disc_nh else other_novels
        
        if self.hdd_ret_padding:
            # self.returns_oa[:, -1] = other_na[:, -1] / (1 - self.gamma)
            # self.returns_oh[:, -1] = other_nh[:, -1] / (1 - self.hdd_gamma)
            self.returns_oa[:, -1] = other_na[:, -10:].mean(axis=1) / (1 - self.gamma)
            self.returns_oh[:, -1] = other_nh[:, -10:].mean(axis=1) / (1 - self.hdd_gamma)
        
        for step in reversed(range(self.episode_length)):
            proxy_reward = other_na[:, step + 1] * self.masks[step + 1]
            ret = proxy_reward + self.gamma * self.returns_oa[:, step + 1] * self.masks[step + 1]
            if self.hdd_ret_padding:
                pad_indices = self.masks[step + 1] == 0
                if pad_indices.sum() > 0:
                    # ret[:, pad_indices] = other_na[:, step, pad_indices] / (1 - self.gamma)
                    ret[:, pad_indices] = other_na[:, max(0, step - 9): step + 1, pad_indices].mean(axis=1) / (1 - self.gamma)
            self.returns_oa[:, step] = ret
            
            proxy_reward = other_nh[:, step + 1] * self.masks[step + 1]
            ret = proxy_reward + self.hdd_gamma * self.returns_oh[:, step + 1] * self.masks[step + 1]
            if self.hdd_ret_padding:
                pad_indices = self.masks[step + 1] == 0
                if pad_indices.sum() > 0:
                    # ret[:, pad_indices] = other_nh[:, step, pad_indices] / (1 - self.hdd_gamma)
                    ret[:, pad_indices] = other_nh[:, max(0, step - 9): step + 1, pad_indices].mean(axis=1) / (1 - self.hdd_gamma)
            self.returns_oh[:, step] = ret
            
        for step in range(self.episode_length):
            proxy_reward = other_nh[:, step].copy()
            if step == 0:
                ret = proxy_reward
                if self.hdd_ret_padding:
                    ret = proxy_reward / (1 - self.hdd_gamma)
            else:
                ret = proxy_reward + self.hdd_gamma * self.returns_oh_reverse[:, step - 1] * self.masks[step]
                if self.hdd_ret_padding:
                    pad_indices = self.masks[step] == 0
                    if pad_indices.sum() > 0:
                        ret[:, pad_indices] = proxy_reward[:, pad_indices] / (1 - self.hdd_gamma)
            self.returns_oh_reverse[:, step] = ret
            
    def compute_other_novel_returns(self):
        if self.other_reduce:
            self.compute_reduced_other_novel_returns()
        else:
            self.compute_separate_other_novel_returns()
    
    def compute_returns(self, self_coef=0.0, other_coef=0.0, value_normalizer=None, hdd_coef=0.0, novel_max=False):
        if self._use_gae:
            gae = 0.0
            for step in reversed(range(self.episode_length)):
                proxy_reward = self.ext_rewards[step].copy()
                if novel_max: 
                    proxy_reward += self_coef * self.novels[:, step + 1].max(axis=0) * self.masks[step + 1]
                else:
                    proxy_reward += self_coef * self.novels[self.agent_id, step + 1] * self.masks[step + 1]
                    proxy_reward += other_coef * (self.novels[:, step + 1].sum(axis=0) - 
                                                  self.novels[self.agent_id, step + 1]) * self.masks[step + 1]
                proxy_reward += hdd_coef * self.hdd_advantages[step]
                self.int_rewards[step] = proxy_reward - self.ext_rewards[step]
                    
                if self._use_popart or self._use_valuenorm:
                    delta = proxy_reward + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) \
                        * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    if self._use_proper_time_limits:
                        gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                else:
                    delta = proxy_reward + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] \
                        - self.value_preds[step]
                    gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                    if self._use_proper_time_limits:
                        gae = gae * self.bad_masks[step + 1]
                    self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = self.value_preds[-1].copy()
            for step in reversed(range(self.episode_length)):
                proxy_reward = self.ext_rewards[step].copy()
                if novel_max:
                    proxy_reward += self_coef * self.novels[:, step + 1].max(axis=0) * self.masks[step + 1]
                else:
                    proxy_reward += self_coef * self.novels[self.agent_id, step + 1] * self.masks[step + 1]
                    proxy_reward += other_coef * (self.novels[:, step + 1].sum(axis=0) - 
                                                  self.novels[self.agent_id, step + 1]) * self.masks[step + 1]
                proxy_reward += hdd_coef * self.hdd_advantages[step]
                self.int_rewards[step] = proxy_reward - self.ext_rewards[step]
                
                if self._use_proper_time_limits:
                    if self._use_popart:
                        self.returns[step] = (proxy_reward + self.gamma * self.returns[step + 1]) * \
                            self.bad_masks[step + 1] + value_normalizer.denormalize(self.value_preds[step]) * \
                            (1 - self.bad_masks[step + 1])
                    else:
                        self.returns[step] = (proxy_reward + self.gamma * self.returns[step + 1]) * \
                            self.bad_masks[step + 1] + self.value_preds[step] * (1 - self.bad_masks[step + 1])
                else:
                    self.returns[step] = proxy_reward + self.gamma * self.returns[step + 1] * self.masks[step + 1]
        
    def save_disc_bound(self, save_dir):
        assert self.disc is not None
        for agent_id in range(self.n_agents_in_env):
            np.save(str(save_dir) + f"/nbound_agent{agent_id}.npy", self.disc[agent_id].bounds)
    
    def restore_disc_bound(self, model_dir):
        assert self.disc is not None
        for agent_id in range(self.n_agents_in_env):
            bounds = np.load(str(model_dir) + f"/nbound_agent{agent_id}.npy")
            self.disc[agent_id].bounds = bounds
            self.disc[agent_id].init_flag = False


class HindsightBuffer(object):
    def __init__(self, args, obs_space, act_space):
        self.n_rollout_threads = args.n_rollout_threads
        self.length = args.hdd_buffer_size // self.n_rollout_threads
        self.size = self.length * self.n_rollout_threads
        
        obs_shape = get_shape_from_obs_space(obs_space)
        act_shape = get_shape_from_act_space(act_space)
        self.obs = np.zeros((self.length, self.n_rollout_threads, obs_shape[0]), dtype=np.float32)
        self.actions = np.zeros((self.length, self.n_rollout_threads, act_shape), dtype=np.float32)
        self.returns = np.zeros((self.length, self.n_rollout_threads, 1), dtype=np.float32)
        self.last_rewards = np.zeros((self.length, self.n_rollout_threads, 1), dtype=np.float32)
        self.returns_reverse = np.zeros((self.length, self.n_rollout_threads, 1), dtype=np.float32)
        
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = np.ones((self.length, self.n_rollout_threads, act_space.n), dtype=np.float32)
        else:
            self.available_actions = None
        self.active_masks = np.ones((self.length, self.n_rollout_threads, 1), dtype=np.float32)
        
        self.step = 0
        self.len = 0
        
    def __len__(self):
        return self.len
    
    def insert(self, obs, actions, returns, last_rewards, returns_reverse,
               available_actions=None, active_masks=None):
        for i in range(len(obs)):
            self.obs[self.step] = obs[i].copy()
            self.actions[self.step] = actions[i].copy()
            self.returns[self.step] = returns[i].copy()
            self.last_rewards[self.step] = last_rewards[i].copy()
            self.returns_reverse[self.step] = returns_reverse[i].copy()
            if available_actions is not None:
                self.available_actions[self.step] = available_actions[i].copy()
            if active_masks is not None:
                self.active_masks[self.step] = active_masks[i].copy()
            
            self.step = (self.step + 1) % self.length
            self.len = min(self.len + 1, self.length)
        
    def generator(self, num_mini_batch=None, mini_batch_size=None):
        batch_size = self.len * self.n_rollout_threads
        if mini_batch_size is None:
            assert batch_size >= num_mini_batch
            mini_batch_size = batch_size // num_mini_batch
        else:
            num_mini_batch = batch_size // mini_batch_size
        
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]
        
        obs = self.obs[:self.len].reshape(-1, *self.obs.shape[2:])
        actions = self.actions[:self.len].reshape(-1, self.actions.shape[-1])
        returns = self.returns[:self.len].reshape(-1, 1)
        last_rewards = self.last_rewards[:self.len].reshape(-1, 1)
        returns_reverse = self.returns_reverse[:self.len].reshape(-1, 1)
        if self.available_actions is not None:
            available_actions = self.available_actions[:self.len].reshape(-1, self.available_actions.shape[-1])
        else:
            available_actions = None
        active_masks = self.active_masks[:self.len].reshape(-1, 1)
        
        for indices in sampler:
            obs_batch = obs[indices]
            actions_batch = actions[indices]
            returns_batch = returns[indices]
            last_rewards_batch = last_rewards[indices]
            returns_reverse_batch = returns_reverse[indices]
            if available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            active_masks_batch = active_masks[indices]
            
            yield obs_batch, actions_batch, returns_batch, last_rewards_batch, returns_reverse_batch, \
                  available_actions_batch, active_masks_batch
