import time
from itertools import chain

import imageio
import numpy as np
import torch
import wandb
from prettytable import PrettyTable
from src.runner.base import Base


def _t2n(x):
    if isinstance(x, np.ndarray):
        return x
    return x.detach().cpu().numpy()


class Runner(Base):
    def __init__(self, config):
        super(Runner, self).__init__(config)
        
    def run(self):
        self.warmup()
        epoches = int(self.num_env_steps) // (self.episode_length * self.n_rollout_threads)
        rewards_log_dir = str(self.run_dir / "rewards.txt")
        
        eval_rewards, best_eval_reward, best_epcoh = [], -999999, 0
        episode_rewards = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        episode_lengths = np.zeros((self.n_rollout_threads,), dtype=np.float32)
        
        for epoch in range(epoches):
            start = time.time()
        
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(epoch, epoches)
                
            train_episode_rewards, train_episode_lengths = [], []
            train_episode_game_stats = [{} for _ in range(self.num_agents)]
            
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards += rewards[:, :, 0]
                episode_lengths += 1
                
                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic
                self.insert(data)
                
                if dones.any():
                    indices = np.where(dones[:, 0] == True)[0]
                    train_episode_rewards.append(episode_rewards[indices])
                    train_episode_lengths.append(episode_lengths[indices])
                    episode_rewards[indices] *= 0.0
                    episode_lengths[indices] = 0.0
                    
                    self._append_game_stats(train_episode_game_stats, infos[indices])
            
            train_infos = self.compute()
            train_infos_ = self.train()
            for agent_id in range(self.num_agents):
                train_infos[agent_id].update(train_infos_[agent_id])
                
                for k in list(train_episode_game_stats[agent_id].keys()):
                    train_episode_game_stats[agent_id]["events/" + k] = np.mean(train_episode_game_stats[agent_id].pop(k))
                train_infos[agent_id].update(train_episode_game_stats[agent_id])
            
            total_num_steps = (epoch + 1) * self.episode_length * self.n_rollout_threads
            num_steps = self.episode_length * self.n_rollout_threads
        
            if epoch % self.save_interval == 0:
                self.save(epoch)
                
            if epoch % self.log_interval == 0:
                end = time.time()
                print("\n Algo {} Exp {} updates {}/{} epoches, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.algorithm_name,
                                self.experiment_name,
                                epoch,
                                epoches,
                                total_num_steps,
                                int(self.num_env_steps),
                                int(num_steps / (end - start))))

                train_episode_rewards = np.concatenate(train_episode_rewards, axis=0)
                train_episode_lengths = np.concatenate(train_episode_lengths, axis=0)
                for agent_id in range(self.num_agents):
                    train_infos[agent_id].update({"episode_rewards": np.mean(train_episode_rewards[:, agent_id])})
                    train_infos[agent_id].update({"episode_lengths": np.mean(train_episode_lengths)})
                    train_infos[agent_id].update({"intrinsic_rewards": np.mean(self.buffer[agent_id].int_rewards)})
                    train_infos[agent_id].update({"intrinsic_rewards_max": np.max(self.buffer[agent_id].int_rewards)})

                with open(rewards_log_dir, "a") as f:
                    f.write("%f\n" % train_infos[0]["episode_rewards"])
                
                self._print_table(train_infos)
                self.log_train(train_infos, total_num_steps)
            
            if epoch % self.eval_interval == 0 and self.use_eval:
                eval_reward = self.eval(total_num_steps)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_epcoh = epoch
                    self.save()
                print(f"best episode reward: {best_eval_reward:.4f} at epoch {best_epcoh}")
                eval_rewards.append(eval_reward)
    
    def warmup(self):
        obs = self.envs.reset()
        
        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)
        
        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()
    
    @torch.no_grad()
    def collect(self, step, deterministic=False):
        values, rnn_states_critic = [], []
        actions, temp_actions_env, action_log_probs, rnn_states = [], [], [], []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step], 
                                                            self.buffer[agent_id].masks[step],
                                                            deterministic=deterministic)
            # [agents, envs, dim]
            values.append(_t2n(value))
            
            # rearrange action
            action = _t2n(action)
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Box':
                action_env = np.clip(action, self.envs.action_space[agent_id].low, self.envs.action_space[agent_id].high)
            else:
                raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.stack(values, axis=1)
        actions = np.stack(actions, axis=1)
        action_log_probs = np.stack(action_log_probs, axis=1)
        rnn_states = np.stack(rnn_states, axis=1)
        rnn_states_critic = np.stack(rnn_states_critic, axis=1)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env
               
    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        bad_masks = np.ones_like(masks)
        if np.any(dones):
            for i in range(self.n_rollout_threads):
                for j in range(self.num_agents):
                    if infos[i, j].get("TimeLimit.truncated", False):
                        bad_masks[i, j] = 0.0

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                         np.array(list(obs[:, agent_id])),
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id],
                                         bad_masks=bad_masks[:, agent_id])
    
    @torch.no_grad()
    def eval(self, total_num_steps=None, render=0.0):
        eval_episode_rewards, eval_episode_lengths, all_frames = [], [], []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        episode_rewards = np.zeros((self.n_eval_rollout_threads, self.num_agents), dtype=np.float32)
        episode_lengths = np.zeros((self.n_eval_rollout_threads,), dtype=np.float32)
        for _ in range(self.episode_length):
            if self.all_args.save_gifs:
                img = self.eval_envs.render("rgb_array")[0]
                all_frames.append(img)
            if render:
                self.eval_envs.render()
                time.sleep(render)
            eval_actions, eval_temp_actions_env = [], []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()

                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                deterministic=True)
                eval_action = eval_action.detach().cpu().numpy()
                eval_actions.append(eval_action)
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Box':
                    eval_action_env = np.clip(eval_action, self.eval_envs.action_space[agent_id].low, self.eval_envs.action_space[agent_id].high)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            eval_actions = np.stack(eval_actions, axis=1)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            episode_rewards += eval_rewards[:, :, 0]
            episode_lengths += 1

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)
            
            if eval_dones.any():
                indices = np.where(eval_dones[:, 0] == True)[0]
                eval_episode_rewards.append(episode_rewards[indices])
                eval_episode_lengths.append(episode_lengths[indices])
                episode_rewards[indices] *= 0.0
                episode_lengths[indices] = 0.0

        eval_episode_rewards = np.concatenate(eval_episode_rewards, axis=0)
        eval_episode_lengths = np.concatenate(eval_episode_lengths, axis=0)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(eval_episode_rewards[:, agent_id], axis=0)
            eval_std_episode_rewards = np.std(eval_episode_rewards[:, agent_id], axis=0)
            eval_train_infos.append({'eval_episode_rewards': eval_average_episode_rewards,
                                     'eval_episode_lengths': eval_episode_lengths.mean()})
            print(f"eval episode rewards of agent{agent_id}: {eval_average_episode_rewards:.4f} ± {eval_std_episode_rewards:.4f}")

        if not self.use_render:
            self.log_train(eval_train_infos, total_num_steps)
            
        if self.all_args.save_gifs:
            imageio.mimsave(self.gif_dir + '/render.gif', all_frames, duration=self.all_args.ifi)
        
        return eval_average_episode_rewards

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                if k[:6] == "events":
                    agent_k = "event%i/" % agent_id + k[7:]
                else:
                    agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalar(agent_k, v, total_num_steps)

    def _append_game_stats(self, train_episode_game_stats, infos):
        for info in infos:
            for gs_i, info_i in zip(train_episode_game_stats, info):
                for k, v in info_i['game_stats'].items():
                    if k not in gs_i:
                        gs_i[k] = []
                    gs_i[k].append(v)

    def _print_table(self, train_infos):
        col_name = ['agent', 'rew', 'len', 'pi_loss', 'vf_loss', 'ent']
        if "hdd_loss" in train_infos[0]:
            col_name.append('hdd_loss')
        if "hdd0_loss" in train_infos[0]:
            for i in range(self.num_agents - 1):
                col_name.append(f'hdd{i}_loss')
        if "rnd_loss" in train_infos[0]:
            col_name.append('rnd_loss')
        table = PrettyTable(col_name)
        for agent_id in range(self.num_agents):
            r = [str(agent_id), 
                 "{:.4f}".format(train_infos[agent_id]['episode_rewards']),
                 "{:.4f}".format(train_infos[agent_id]['episode_lengths']),
                 "{:.3e}".format(train_infos[agent_id]['policy_loss']), 
                 "{:.3e}".format(train_infos[agent_id]['value_loss']),
                 "{:.4f}".format(train_infos[agent_id]['dist_entropy'])]
            if "hdd_loss" in train_infos[agent_id]:
                r.append("{:.4f}".format(train_infos[agent_id]['hdd_loss']))
            if "hdd0_loss" in train_infos[agent_id]:
                for i in range(self.num_agents - 1):
                    r.append("{:.4f}".format(train_infos[agent_id][f'hdd{i}_loss']))
            if "rnd_loss" in train_infos[agent_id]:
                r.append("{:.3e}".format(train_infos[agent_id]['rnd_loss']))
            table.add_row(r)
        print(table)
