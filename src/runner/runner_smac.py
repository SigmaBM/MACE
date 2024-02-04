"""Based on https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/runner/shared/smac_runner.py. """
import time
from functools import reduce

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
        train_wr_log_dir = str(self.run_dir / "train_wr.txt")
        eval_wr_log_dir = str(self.run_dir / "eval_wr.txt")
        eval_rew_log_dir = str(self.run_dir / "eval_rew.txt")
        
        eval_rewards, best_eval_reward, best_epcoh = [], -999999, 0
        episode_rewards = np.zeros((self.n_rollout_threads, self.num_agents), dtype=np.float32)
        episode_lengths = np.zeros((self.n_rollout_threads,), dtype=np.float32)

        last_battles_game = np.zeros(self.n_rollout_threads, dtype=np.float32)
        last_battles_won = np.zeros(self.n_rollout_threads, dtype=np.float32)

        for epoch in range(epoches):
            start = time.time()
        
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(epoch, epoches)
                
            train_episode_rewards, train_episode_lengths = [], []
            
            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic = self.collect(step)
                
                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)
                episode_rewards += rewards[:, :, 0]
                episode_lengths += 1
                
                data = obs, share_obs, rewards, dones, infos, available_actions, values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic 
                
                self.insert(data)
                
                dones_env = np.all(dones, axis=1)
                if dones_env.any():
                    indices = np.where(dones_env == True)[0]
                    train_episode_rewards.append(episode_rewards[indices])
                    train_episode_lengths.append(episode_lengths[indices])
                    episode_rewards[indices] *= 0.0
                    episode_lengths[indices] = 0.0

            train_infos = self.compute()
            train_infos_ = self.train()
            for agent_id in range(self.num_agents):
                train_infos[agent_id].update(train_infos_[agent_id])
            
            total_num_steps = (epoch + 1) * self.episode_length * self.n_rollout_threads
            num_steps = self.episode_length * self.n_rollout_threads
        
            if epoch % self.save_interval == 0:
                self.save(epoch)
                
            if epoch % self.log_interval == 0:
                end = time.time()
                print("\nMap {} Algo {} Exp {} updates {}/{} epoches, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.map_name,
                                self.algorithm_name,
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

                battles_won = []
                battles_game = []
                incre_battles_won = []
                incre_battles_game = []                    

                for i, info in enumerate(infos):
                    if 'battles_won' in info[0].keys():
                        battles_won.append(info[0]['battles_won'])
                        incre_battles_won.append(info[0]['battles_won']-last_battles_won[i])
                    if 'battles_game' in info[0].keys():
                        battles_game.append(info[0]['battles_game'])
                        incre_battles_game.append(info[0]['battles_game']-last_battles_game[i])

                incre_win_rate = np.sum(incre_battles_won) / np.sum(incre_battles_game) if np.sum(incre_battles_game)>0 else 0.0
                if self.use_wandb:
                    wandb.log({"incre_win_rate": incre_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalar("main/incre_win_rate", incre_win_rate, total_num_steps)
                
                last_battles_game = battles_game
                last_battles_won = battles_won

                for agent_id in range(self.num_agents):
                    dead_ratio = 1 - \
                        self.buffer[agent_id].active_masks.sum() / reduce(lambda x, y: x*y, list(self.buffer[agent_id].active_masks.shape))
                    train_infos[agent_id].update({"dead_ratio": dead_ratio})
                
                print("- Episode rewards: {}".format(train_infos[0]["episode_rewards"]))
                print("- Win rate: {}".format(incre_win_rate))
                # print("- Dead ratio: {}".format([info["dead_ratio"] for info in train_infos]))
                
                with open(rewards_log_dir, "a") as f:
                    f.write("%f\n" % train_infos[0]["episode_rewards"])
                with open(train_wr_log_dir, "a") as f:
                    f.write("%f\n" % incre_win_rate)
                
                self.log_train(train_infos, total_num_steps)
                self._print_table(train_infos)
            
            if epoch % self.eval_interval == 0 and self.use_eval:
                eval_reward, eval_win_rate = self.eval(total_num_steps)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_epcoh = epoch
                    self.save()
                print(f"best episode reward: {best_eval_reward:.4f} at epoch {best_epcoh}")
                eval_rewards.append(eval_reward)
                
                with open(eval_wr_log_dir, "a") as f:
                    f.write("%f\n" % eval_win_rate)
                with open(eval_rew_log_dir, "a") as f:
                    f.write("%f\n" % eval_reward)
    
    def warmup(self):
        obs, share_obs, available_actions = self.envs.reset()

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step, deterministic=False):
        values, rnn_states_critic = [], []
        actions, action_log_probs, rnn_states = [], [], []
        
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step],
                                                            self.buffer[agent_id].available_actions[step],
                                                            deterministic=deterministic)

            values.append(_t2n(value))
            actions.append(_t2n(action))
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))
        
        values = np.stack(values, axis=1)
        actions = np.stack(actions, axis=1)
        action_log_probs = np.stack(action_log_probs, axis=1)
        rnn_states = np.stack(rnn_states, axis=1)
        rnn_states_critic = np.stack(rnn_states_critic, axis=1)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), 
                                                  self.num_agents, 
                                                  self.recurrent_N, 
                                                  self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), 
                                                         self.num_agents, 
                                                         self.recurrent_N, 
                                                         self.hidden_size), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        bad_masks = np.array([[
            [0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)
            ] for info in infos])
        
        if not self.use_centralized_V:
            share_obs = obs
        
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id],
                                         obs[:, agent_id],
                                         rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id],
                                         rewards[:, agent_id],
                                         masks[:, agent_id],
                                         bad_masks[:, agent_id],
                                         active_masks[:, agent_id],
                                         available_actions[:, agent_id])

    def compute(self):
        if self.novel_type == 4:
            for agent_id in range(self.num_agents):
                self.buffer[agent_id].novels[agent_id] = self.policy[agent_id].get_rnd_results(self.buffer[agent_id].obs)
                if self.trainer[agent_id]._use_novel_active_masks:
                    self.buffer[agent_id].novels[agent_id] *= self.buffer[agent_id].active_masks
                
        if self.use_novel_normalization:
            for agent_id in range(self.num_agents):
                if self.trainer[agent_id]._use_novel_active_masks:
                    self.novel_rms[agent_id].update(
                        self.buffer[agent_id].novels[agent_id][self.buffer[agent_id].active_masks == 1].reshape(-1, 1))
                else:
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
    
    def log_train(self, train_infos, total_num_steps):
        super().log_train(train_infos, total_num_steps)
        self.writter.add_scalar("main/average_step_rewards", np.mean(self.buffer[0].ext_rewards), total_num_steps)
    
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_battles_won = 0
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = []

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        while True:
            eval_actions = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                                                eval_rnn_states[:, agent_id],
                                                                                eval_masks[:, agent_id],
                                                                                eval_available_actions[:, agent_id],
                                                                                deterministic=True)
                eval_actions.append(_t2n(eval_action))
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
            
            eval_actions = np.stack(eval_actions, axis=1)
            # Obser reward and next obs
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos, eval_available_actions = self.eval_envs.step(eval_actions)
            one_episode_rewards.append(eval_rewards)

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), 
                                                                self.num_agents, 
                                                                self.recurrent_N, 
                                                                self.hidden_size), dtype=np.float32)

            eval_masks = np.ones((self.all_args.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(np.sum(one_episode_rewards, axis=0))
                    one_episode_rewards = []
                    if eval_infos[eval_i][0]['won']:
                        eval_battles_won += 1

            if eval_episode >= self.all_args.eval_episodes:
                eval_episode_rewards = np.array(eval_episode_rewards)
                eval_env_infos = {'main/eval_average_episode_rewards': eval_episode_rewards}                
                self.log_env(eval_env_infos, total_num_steps)
                eval_win_rate = eval_battles_won / eval_episode
                print("eval win rate is {}.".format(eval_win_rate))
                print("eval average reward is {}.".format(eval_episode_rewards.mean()))
                if self.use_wandb:
                    wandb.log({"eval_win_rate": eval_win_rate}, step=total_num_steps)
                else:
                    self.writter.add_scalar("main/eval_win_rate", eval_win_rate, total_num_steps)
                break
            
        return eval_episode_rewards.mean(), eval_win_rate
    
    def _print_table(self, train_infos):
        col_name = ['agent', 'dead_ratio', 'pi_loss', 'vf_loss', 'ent']
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
                 "{:.3f}".format(train_infos[agent_id]['dead_ratio']),
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
