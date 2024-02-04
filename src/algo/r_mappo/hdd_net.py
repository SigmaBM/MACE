import torch
import torch.nn as nn
from src.algo.utils.act import ACTLayer
from src.algo.utils.cnn import CNNBase
from src.algo.utils.mlp import MLPBase
from src.algo.utils.util import check
from src.utils.util import get_grad_norm, get_shape_from_obs_space


class Predictor(nn.Module):
    def __init__(self, args, obs_space, device=torch.device("cpu")):
        super(Predictor, self).__init__()
        self.hidden_size = args.hidden_size
        self.rep_size = args.rep_size
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base = CNNBase if len(obs_shape) == 3 else MLPBase
        self.base = base(args, obs_shape)
        self.out = nn.Linear(self.hidden_size, self.rep_size)

        self.to(device)

    def forward(self, obs):
        obs = check(obs).to(**self.tpdv)
        return self.out(self.base(obs))
    

class HDD(nn.Module):
    def __init__(self, act_space, obs_space, args, device=torch.device("cpu")):
        super(HDD, self).__init__()
        self.hidden_size = args.hidden_size
        self.hdd_last_rew = args.hdd_last_rew
        self.hdd_reverse_ret = args.hdd_reverse_ret
        self.hdd_return_scale = args.hdd_ret_ub
        self.tpdv = dict(dtype=torch.float32, device=device)
        
        obs_shape = get_shape_from_obs_space(obs_space)
        if len(obs_shape) == 3: # image input
            self.vis_base = CNNBase(args, obs_shape)
            input_size = self.hidden_size
        else:
            self.vis_base = None
            input_size = obs_shape[0]
        
        assert not (self.hdd_last_rew and self.hdd_reverse_ret), "last_rew and reverse_ret cannot be both True."
        if self.hdd_last_rew or self.hdd_reverse_ret:
            input_size += 1
        input_size += 1 # extra dimension for return
        
        self.base = MLPBase(args, (input_size,))   
        self.act = ACTLayer(act_space, self.hidden_size, args.use_orthogonal, args.gain)
        self.to(device)

    def forward(self, obs, ret, last_rew=None, rev_ret=None, available_actions=None):
        obs = check(obs).to(**self.tpdv)
        ret = check(ret).to(**self.tpdv) / self.hdd_return_scale
        if self.vis_base is not None:
            obs_features = self.vis_base(obs)
        else:
            obs_features = obs
            
        if self.hdd_last_rew:
            assert last_rew is not None, "Nonr for last rewards."
            last_rew = check(last_rew).to(**self.tpdv)
            obs_features = torch.cat([obs_features, last_rew], dim=-1)
        if self.hdd_reverse_ret:
            assert rev_ret is not None, "None for reverse returns."
            rev_ret = check(rev_ret).to(**self.tpdv) / self.hdd_return_scale
            obs_features = torch.cat([obs_features, rev_ret], dim=-1)
        obs_features = torch.cat([obs_features, ret], dim=-1)
            
        return self.act.get_dists(self.base(obs_features), available_actions)

    def evaluate_actions(self, obs, action, ret, last_rew=None, rev_ret=None, available_actions=None):
        action_dists = self(obs, ret, last_rew, rev_ret, available_actions)
        
        action = check(action).to(**self.tpdv)
        return action_dists.log_prob(action.squeeze(-1)).unsqueeze(-1)
    

class HDDNetworks(object):
    def __init__(self, act_space, obs_space, args, device=torch.device("cpu")):
        self.reduce = args.hdd_reduce
        self.n_other_agent = args.n_agents - 1
        
        self.hdd_epoch = args.hdd_epoch
        self.hdd_batch_size = args.hdd_batch_size
        self.max_grad_norm = args.max_grad_norm
        self.hdd_lr = args.hdd_lr
        self.opti_eps = args.opti_eps
        
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_hdd_active_masks = args.use_hdd_active_masks
        
        if self.reduce:
            self.hdd = HDD(act_space, obs_space, args, device)
            self.hdd_optim = torch.optim.Adam(self.hdd.parameters(), 
                                              lr=self.hdd_lr, eps=self.opti_eps)
        else:
            self.hdd, self.hdd_optim = [], []
            for _ in range(self.n_other_agent):
                hdd = HDD(act_space, obs_space, args, device)
                hdd_optim = torch.optim.Adam(hdd.parameters(), 
                                             lr=self.hdd_lr, eps=self.opti_eps)
                self.hdd.append(hdd)
                self.hdd_optim.append(hdd_optim)

        self.tpdv = dict(dtype=torch.float32, device=device)
            
    def update(self, buffer_hdd, obs, actions, returns, last_rewards, reverse_returns, 
               available_actions=None, active_masks=None):
        if self.reduce:
            buffer_hdd.insert(obs, actions, returns, last_rewards, reverse_returns, 
                              available_actions, active_masks)
            return self._update(self.hdd, self.hdd_optim, buffer_hdd)
        
        infos = {}
        for i in range(self.n_other_agent):
            buffer_hdd[i].insert(obs, actions, returns[i], last_rewards[i], reverse_returns[i],
                                 None if available_actions is None else available_actions,
                                 None if active_masks is None else active_masks)
            info = self._update(self.hdd[i], self.hdd_optim[i], buffer_hdd[i])
            infos[f"hdd{i}_loss"] = info["hdd_loss"]
            infos[f"hdd{i}_grad_norm"] = info["hdd_grad_norm"]
        
        return infos
    
    def _update(self, hdd, hdd_optim, buffer_hdd):
        infos = {}
        infos["hdd_loss"] = []
        infos["hdd_grad_norm"] = []
        
        for _  in range(self.hdd_epoch):
            data_generator = buffer_hdd.generator(mini_batch_size=self.hdd_batch_size)
            for sample in data_generator:
                obs_batch, actions_batch, returns_oh_batch, last_rewards_oh, \
                returns_oh_reverse_batch, available_actions_batch, active_masks_batch = sample
                active_masks_batch = check(active_masks_batch).to(**self.tpdv)

                hdd_loss = -hdd.evaluate_actions(obs_batch, actions_batch, returns_oh_batch,
                                                 last_rewards_oh, returns_oh_reverse_batch,
                                                 available_actions_batch)

                if self._use_hdd_active_masks:
                    hdd_loss = (hdd_loss * active_masks_batch).sum() / active_masks_batch.sum()
                else:
                    hdd_loss = hdd_loss.mean()

                hdd_optim.zero_grad()
                hdd_loss.backward()
                if self._use_max_grad_norm:
                    hdd_grad_norm = nn.utils.clip_grad_norm_(hdd.parameters(), self.max_grad_norm)
                else:
                    hdd_grad_norm = get_grad_norm(hdd.parameters())
                hdd_optim.step()
                
                infos["hdd_loss"].append(hdd_loss.item())
                infos["hdd_grad_norm"].append(hdd_grad_norm.item())
        
        return infos
    
    def evaluate_actions(self, obs, actions, returns, last_rewards=None, reverse_returns=None, 
                         available_actions=None):
        if self.reduce:
            log_probs = self.hdd.evaluate_actions(
                obs, actions, returns, last_rewards, reverse_returns, available_actions)
            return torch.exp(log_probs).detach().cpu().numpy()
        
        log_probs = []
        for i in range(self.n_other_agent):
            log_probs.append(self.hdd[i].evaluate_actions(obs, actions, returns[i], 
                                                          None if last_rewards is None else last_rewards[i], 
                                                          None if reverse_returns is None else reverse_returns[i],
                                                          available_actions))
        return torch.exp(torch.stack(log_probs, dim=0)).detach().cpu().numpy()
    
    def save(self, save_dir, index):
        if self.reduce:
            torch.save(self.hdd.state_dict(), str(save_dir) + f"/hdd_agent{index}.pt")
            return
        
        for i, h in enumerate(self.hdd):
            torch.save(h.state_dict(), str(save_dir) + f"/hdd_agent{index}_{i}.pt")
            
    def restore(self, model_dir, index):
        if self.reduce:
            hdd_state_dict = torch.load(str(model_dir) + f"/hdd_agent{index}.pt")
            self.hdd.load_state_dict(hdd_state_dict)
            return

        for i, h in enumerate(self.hdd):
            hdd_state_dict = torch.load(str(model_dir) + f"/hdd_agent{index}_{i}.pt")
            h.load_state_dict(hdd_state_dict)
