"""Based on https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/algorithms/r_mappo/algorithm/rMAPPOPolicy.py. """
import torch
from src.algo.r_mappo.hdd_table import HDDTables
from src.algo.r_mappo.hdd_net import HDDNetworks
from src.algo.r_mappo.r_actor_critic import R_Actor, R_Critic
from src.algo.r_mappo.rnd import RND
from src.utils.util import update_linear_schedule


def _t2n(x):
    return x.detach().cpu().numpy()


class R_MAPPOPolicy:
    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.actor_lr, self.critic_lr = args.actor_lr, args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay
        
        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space
        
        self.actor = R_Actor(args, obs_space, act_space, device)
        self.critic = R_Critic(args, cent_obs_space, device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.actor_lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr, eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)

        self.novel_type = args.novel_type   # 1: prediction error; 2: TD error; 3: count-based; 4. RND
        assert self.novel_type in [0, 1, 2, 3, 4], "novelty type should be 0, 1, 2, 3 or 4"

        if self.novel_type == 4:
            self.rnd = RND(args, cent_obs_space, device)
            self.rnd_lr = args.rnd_lr
            self.rnd_optimizer = torch.optim.Adam(self.rnd.predictor.parameters(),
                                                  lr=self.rnd_lr, eps=self.opti_eps,
                                                  weight_decay=self.weight_decay)
            
        self.use_hdd = args.use_hdd
        self.hdd_count = args.hdd_count
        
        if self.use_hdd:
            if self.hdd_count:
                self.hdd = HDDTables(self.act_space, args)
            else:
                self.hdd = HDDNetworks(self.act_space, self.obs_space, args, device)
        else:
            self.hdd = None
        
    def lr_decay(self, episode, episodes):
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        
    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None, 
                    deterministic=False):
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)

        return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
            
    def get_values(self, cent_obs, rnn_states_critic, masks):
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values
            
    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks, 
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        
        return values, action_log_probs, dist_entropy
    
    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
    
    def get_values_pred(self, cent_obs, rnn_states_critic_pred, masks):
        values_pred, rnn_states_critic_pred = self.critic_pred(cent_obs, rnn_states_critic_pred, masks)
        return values_pred, rnn_states_critic_pred
    
    def get_rnd_results(self, obs):
        predict_features, target_features = self.rnd(obs)
        return _t2n(torch.mean(torch.square(predict_features - target_features), dim=-1, keepdim=True))
    
    def get_hdd_probs(self, obs, actions, returns, last_rewards=None, reverse_returns=None):        
        if self.hdd_count:
            return self.hdd.predict(obs, actions, returns, last_rewards, reverse_returns)
        else:
            return self.hdd.evaluate_actions(obs, actions, returns, last_rewards, reverse_returns)
