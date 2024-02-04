"""Based on https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/runner/separated/base_runner.py. """
import datetime
import json
import os

import numpy as np
import torch
import wandb
from src.algo.r_mappo.r_mappo import R_MAPPO
from src.algo.r_mappo.rMAPPOPolicy import R_MAPPOPolicy
from src.utils.buffer import (HindsightBuffer, HindsightReplayBuffer,
                              ReplayBuffer)
from tensorboardX import SummaryWriter


def _t2n(x):
    return x.detach().cpu().numpy()


class Base(object):
    def __init__(self, config):
        self.all_args = config['all_args']
        self.envs = config['envs']
        self.eval_envs = config['eval_envs']
        self.device = config['device']
        self.num_agents = config['num_agents']

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_wandb = self.all_args.use_wandb
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        self.novel_type = self.all_args.novel_type
        self.use_novel_normalization = self.all_args.use_novel_normalization
        self.novel_cen = self.all_args.joint_count
        
        self.discrete_novel_in_adv = self.all_args.discrete_novel_in_adv
        self.discrete_novel_in_hd = self.all_args.discrete_novel_in_hd
        self.discrete_nbins = self.all_args.discrete_nbins
        self.discrete_momentum = self.all_args.discrete_momentum
        # novel 3: count-based
        try:
            self.count_lookup_index = self.all_args.count_lookup_index
        except AttributeError:
            self.count_lookup_index = None
        self.count_pow = self.all_args.count_pow
        self.count_prev_epoch = self.all_args.count_prev_epoch
        self.count_decay_coef = self.all_args.count_decay_coef
        
        self.save_hdd_count = self.all_args.save_hdd_count

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        if self.use_render:
            if self.all_args.save_gifs:
                self.run_dir = "renders"
                self.gif_dir = self.run_dir + '/' + datetime.datetime.now().strftime("%m%d_%H%M%S")
                if not os.path.exists(self.gif_dir):
                    os.makedirs(self.gif_dir)
        elif config["run_dir"] is not None:
            if self.use_wandb:
                self.save_dir = str(wandb.run.dir)
                self.run_dir = str(wandb.run.dir)
            else:
                self.run_dir = config["run_dir"]
                self.log_dir = str(self.run_dir / 'logs')
                if not os.path.exists(self.log_dir):
                    os.makedirs(self.log_dir)
                self.writter = SummaryWriter(self.log_dir)
                self.save_dir = str(self.run_dir / 'models')
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
        
            config_dir = str(self.run_dir / 'params.json')
            with open(config_dir, 'w', newline='\n') as f:
                f.write(json.dumps(vars(self.all_args), indent=4))

        self.policy = []
        for agent_id in range(self.num_agents):
            if self.use_centralized_V:
                share_observation_space = self.envs.share_observation_space[agent_id]
            else:
                share_observation_space = self.envs.observation_space[agent_id]
            # policy network
            po = R_MAPPOPolicy(self.all_args,
                               self.envs.observation_space[agent_id],
                               share_observation_space,
                               self.envs.action_space[agent_id],
                               device = self.device)
            self.policy.append(po)

        self.trainer, self.buffer, self.novel_rms = [], [], []
        for agent_id in range(self.num_agents):
            if self.use_centralized_V:
                share_observation_space = self.envs.share_observation_space[agent_id]
            else:
                share_observation_space = self.envs.observation_space[agent_id]

            tr = R_MAPPO(self.all_args, self.policy[agent_id], device=self.device)
            if self.policy[agent_id].use_hdd:
                bu = HindsightReplayBuffer(agent_id, self.all_args, self.envs.observation_space[agent_id], 
                                           share_observation_space, self.envs.action_space[agent_id])
            else:
                bu = ReplayBuffer(agent_id, self.all_args, self.envs.observation_space[agent_id],
                                  share_observation_space, self.envs.action_space[agent_id])
            
            self.buffer.append(bu)
            self.trainer.append(tr)
            self.novel_rms.append(RunningMeanStd() if self.use_novel_normalization else None)
        
        self.buffer_hdd = []
        for agent_id in range(self.num_agents):
            if self.policy[agent_id].use_hdd and not self.policy[agent_id].hdd_count:
                if self.all_args.hdd_reduce:
                    self.buffer_hdd.append(HindsightBuffer(self.all_args, 
                                                           self.envs.observation_space[agent_id], 
                                                           self.envs.action_space[agent_id]))
                else:
                    self.buffer_hdd.append([HindsightBuffer(self.all_args,
                                                            self.envs.observation_space[agent_id],
                                                            self.envs.action_space[agent_id])
                                            for _ in range(self.num_agents - 1)])
            else:
                self.buffer_hdd.append(None)
        
        if self.model_dir is not None:
            self.restore()
        
    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    def compute(self):
        if self.novel_type == 1:
            raise NotImplementedError
        elif self.novel_type == 2:
            raise NotImplementedError
        elif self.novel_type == 3:  # count-based novelty
            if self.count_prev_epoch:
                visit_counts = self.old_visit_counts.copy()
            else:
                visit_counts = self.visit_counts.copy()
            
            if self.novel_cen:
                compute_cen_novel(self.buffer, visit_counts, self.count_lookup_index, self.count_pow)
            else:
                for agent_id in range(self.num_agents):
                    self.buffer[agent_id].compute_novel(visit_counts[agent_id], 
                                                        self.count_lookup_index,
                                                        self.count_pow)
        elif self.novel_type == 4:  # RND novelty
            for agent_id in range(self.num_agents):
                self.buffer[agent_id].novels[agent_id] = self.policy[agent_id].get_rnd_results(self.buffer[agent_id].obs)
                
        if self.use_novel_normalization:
            for agent_id in range(self.num_agents):
                self.novel_rms[agent_id].update(self.buffer[agent_id].novels[agent_id].reshape(-1, 1))
                self.buffer[agent_id].novels[agent_id] /= np.sqrt(self.novel_rms[agent_id].var)
        
        all_novels = np.stack([self.buffer[i].novels[i] for i in range(self.num_agents)], axis=0)
        
        infos = [{} for _ in range(self.num_agents)]
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].set_all_novels(all_novels)
            self.buffer[agent_id].compute_other_novel_returns()
            
            info = self.trainer[agent_id].hdd_update(self.buffer[agent_id], self.buffer_hdd[agent_id])
            for k, v in info.items():
                if len(v) > 0:
                    infos[agent_id][k] = np.mean(v)
            self.trainer[agent_id].cal_hdd_advantage(self.buffer[agent_id])
            
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].share_obs[-1], self.buffer[agent_id].rnn_states_critic[-1], 
                self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            self.buffer[agent_id].value_preds[-1] = next_value.copy()
            self.buffer[agent_id].compute_returns(self.trainer[agent_id].self_coef,
                                                  self.trainer[agent_id].other_coef,
                                                  self.trainer[agent_id].value_normalizer,
                                                  hdd_coef=self.trainer[agent_id].ir_coef,
                                                  novel_max=self.trainer[agent_id].novel_max)

        return infos
    
    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)       
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self, episode=None):
        if episode is not None:
            checkpoint_dir = str(self.save_dir) + "/cp_" + str(episode)
            os.makedirs(checkpoint_dir)
            save_dir = str(checkpoint_dir)
        else:
            save_dir = str(self.save_dir)
        
        for agent_id in range(self.num_agents):
            actor = self.trainer[agent_id].policy.actor
            torch.save(actor.state_dict(), str(save_dir) + "/actor_agent" + str(agent_id) + ".pt")
            critic = self.trainer[agent_id].policy.critic
            torch.save(critic.state_dict(), str(save_dir) + "/critic_agent" + str(agent_id) + ".pt")
            if self.trainer[agent_id].use_hdd:
                if not self.trainer[agent_id].hdd_count or self.save_hdd_count:
                    self.trainer[agent_id].policy.hdd.save(save_dir, agent_id)  
                    
            if self.novel_type == 3:
                np.save(str(save_dir) + "/count_agent" + str(agent_id) + ".npy", self.visit_counts[agent_id])

        if self.discrete_novel_in_adv or self.discrete_novel_in_hd:
            self.buffer[0].save_disc_bound(save_dir)

    def restore(self):
        for agent_id in range(self.num_agents):
            actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].actor.load_state_dict(actor_state_dict)
            critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
            self.policy[agent_id].critic.load_state_dict(critic_state_dict)
            if self.trainer[agent_id].use_hdd:
                self.trainer[agent_id].policy.hdd.restore(self.model_dir, agent_id)
                if self.discrete_novel_in_adv or self.discrete_novel_in_hd:
                    self.buffer[agent_id].restore_disc_bound(self.model_dir)
            
            if self.novel_type == 3:
                visit_count = np.load(str(self.model_dir) + '/count_agent' + str(agent_id) + '.npy')
                self.envs.set_visit_counts(visit_count, agent_id)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    # self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
                    self.writter.add_scalar(agent_k, v, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    # self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)
                    self.writter.add_scalar(k, np.mean(v), total_num_steps)


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean, batch_std, batch_count = np.mean(x, axis=0), np.std(x, axis=0), x.shape[0]
        batch_var = np.square(batch_std)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count


def compute_cen_novel(buffers, visit_count, index, pow):
    if (visit_count == 0).all():
        return
    
    visit_count_lookup = []
    row, col = visit_count.shape[0], visit_count.shape[1]
    for buf in buffers:
        visit_count_lookup.append(buf.obs[..., index: index+2].copy())
        visit_count_lookup[-1][..., 0] *= row
        visit_count_lookup[-1][..., 1] *= col
        visit_count_lookup[-1] = visit_count_lookup[-1].astype(np.int32)
    visit_count_lookup = np.concatenate(visit_count_lookup, axis=-1)
    visit_count_lookup = tuple([
        visit_count_lookup[..., i:i+1] for i in range(visit_count_lookup.shape[-1])])
    novel = np.power(np.maximum(visit_count[visit_count_lookup], 1), -pow)
    for agent_id, buf in enumerate(buffers):
        buf.novels[agent_id] = novel.copy()