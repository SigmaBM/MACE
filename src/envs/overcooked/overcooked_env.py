import gym
import numpy as np
import pygame
from gym.spaces import Box, Discrete
from overcooked_ai_py.mdp.actions import Action
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from pygame.locals import DOUBLEBUF, HWSURFACE, RESIZABLE
from src.envs.overcooked.overcooked_mdp import OvercookedGridworld_

BASE_REW_SHAPING_PARAMS = {
    "OBJECT_PICKUP_REW": 3,
    "OBJECT_DROP_REW": 3,
    "PLACEMENT_IN_POT_REW": 3,
    "DISH_PICKUP_REWARD": 3,
    "SOUP_PICKUP_REWARD": 5,
    "DISH_DISP_DISTANCE_REW": 0,
    "POT_DISTANCE_REW": 0,
    "SOUP_DISTANCE_REW": 0,
}


EVENTS = [
    # pickup anything
    "tomato_pickup",
    "onion_pickup",
    "dish_pickup",
    "soup_pickup",
    # drop anything
    "tomato_drop",
    "onion_drop",
    "dish_drop",
    "soup_drop",
    # potting anything
    "potting_tomato",
    "potting_onion",
]


class OvercookedEnv_(OvercookedEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def from_mdp(mdp, **kwargs):
        """
        Create an OvercookedEnv directly from a OvercookedGridworld mdp
        rather than a mdp generating function.
        """
        assert isinstance(mdp, OvercookedGridworld_)
        mdp_generator_fn = lambda _ignored: mdp
        return OvercookedEnv_(mdp_generator_fn=mdp_generator_fn, **kwargs)
    
    def step(
        self, joint_action, joint_agent_action_info=None, display_phi=False
    ):
        """Performs a joint action, updating the environment state
        and providing a reward.

        On being done, stats about the episode are added to info:
            ep_sparse_r: the environment sparse reward, given only at soup delivery
            ep_shaped_r: the component of the reward that is due to reward shaped (excluding sparse rewards)
            ep_length: length of rollout
        """
        assert not self.is_done()
        if joint_agent_action_info is None:
            joint_agent_action_info = [{}, {}]
        next_state, mdp_infos = self.mdp.get_state_transition(
            self.state, joint_action, display_phi, None
            # self.mp   # we do not need motion planner here
        )

        # Update game_stats
        self._update_game_stats(mdp_infos)

        # Update state and done
        self.state = next_state
        done = self.is_done()
        env_info = self._prepare_info_dict(joint_agent_action_info, mdp_infos)

        if done:
            self._add_episode_info(env_info)

        timestep_sparse_reward = sum(mdp_infos["sparse_reward_by_agent"])
        return (next_state, timestep_sparse_reward, done, env_info)


class Overcooked(gym.Env):
    env_name = "Overcooked-v0"
    
    def __init__(
        self, 
        layout_name, 
        max_timesteps, 
        rew_shaping_params=None, 
        obs_type="vector",
        multi_round=False,
    ):
        rew_shaping_params = rew_shaping_params or BASE_REW_SHAPING_PARAMS
        self.mdp = OvercookedGridworld_.from_layout_name(layout_name=layout_name,
                                                         rew_shaping_params=rew_shaping_params)
        self.env = OvercookedEnv_.from_mdp(self.mdp, horizon=max_timesteps)
        self.num_players = self.mdp.num_players
        self.multi_round = multi_round
        
        if obs_type == "vector":
            self.featurize_fn = self.mdp.featurize_state
        elif obs_type == "image":
            self.featurize_fn = self.mdp.lossless_state_encoding
        else:
            raise ValueError("Invalid obs_type: {}".format(obs_type))
        
        self.observation_space = self._setup_observation_space()
        self.share_observation_space = self.observation_space   # for compatibility with code
        self.action_space = [Discrete(len(Action.ALL_ACTIONS)) for _ in range(self.num_players)]
        
        self._render, self._window = None, None
        self.reset()
        
    @property
    def unwrapped(self):
        return self.env
        
    def _setup_observation_space(self):
        dummy_mdp = self.env.mdp
        dummy_state = dummy_mdp.get_standard_start_state()
        dummy_obs = self.featurize_fn(dummy_state)
        obs_shape = [obs.shape for obs in dummy_obs]
        
        observation_space = []
        for player_id in range(self.num_players):
            observation_space.append(Box(low=-1, high=+1, shape=obs_shape[player_id], dtype=np.float32))

        return observation_space
    
    def _prepare_events_info(self, infos):
        game_stats = self.env.game_stats
        for i in range(self.num_players):
            infos[i]['game_stats'] = {}
            for k in EVENTS:
                infos[i]['game_stats'][k] = len(game_stats[k][i])
    
    def reset(self):
        self.env.reset()
        self.mdp = self.env.mdp
        
        return self.featurize_fn(self.env.state)
    
    def step(self, actions):
        if not all(type(a) is int for a in actions):
            actions = [a.argmax() for a in actions]
            
        assert all(
            self.action_space[i].contains(actions[i]) for i in range(self.num_players)
        ), "%r (%s) invalid" % (
            actions,
            type(actions),
        )
        actions = [Action.INDEX_TO_ACTION[a] for a in actions]
        
        next_state, reward, done, env_info = self.env.step(actions)
        next_obs = self.featurize_fn(next_state)
        
        # rewrite done
        if reward > 0.0 and not self.multi_round:
            # done after a successful delivery
            done = True
        elif done:
            env_info["TimeLimit.truncated"] = True
            
        reward_n = [[reward]] * self.num_players
        done_n = [done] * self.num_players
        info_n = [env_info.copy() for _ in range(self.num_players)]
        
        if done:
            self._prepare_events_info(info_n)
        
        return next_obs, reward_n, done_n, info_n
    
    def render(self, mode="human"):
        if self._render is None:
            self._render = StateVisualizer()
            pygame.init()
        
        surface = self._render.render_state(self.env.state, self.mdp.terrain_mtx)
        
        if mode == "human":
            if self._window is None:
                self._window = pygame.display.set_mode(
                    surface.get_size(), HWSURFACE | DOUBLEBUF | RESIZABLE)
        
            self._window.blit(surface, (0, 0))
            pygame.display.flip()
        elif mode == "rgb_array":
            return pygame.surfarray.array3d(surface).transpose(1, 0, 2)
        else:
            raise NotImplementedError("Unknown render mode: {}".format(mode))


if __name__ == "__main__":
    env = Overcooked(layout_name='small_coor_hard_pass_2_v3', max_timesteps=400)
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    
    obs = env.reset()
    for i in range(len(obs)):
        print("Obs {}: {}".format(i, obs[i]))
    env.render(mode="human")
    
    act_dict = {pygame.K_w: [0, 4],
                pygame.K_a: [3, 4],
                pygame.K_s: [1, 4],
                pygame.K_d: [2, 4],
                pygame.K_SPACE: [5, 4],
                pygame.K_UP:    [4, 0],
                pygame.K_LEFT:  [4, 3],
                pygame.K_DOWN:  [4, 1],
                pygame.K_RIGHT: [4, 2],
                pygame.K_RETURN: [4, 5],
                }
    
    done = [False, False]
    while not all(done):
        while True:
            # poll for actions (one agent at a time)
            events = pygame.event.get()
            actions = None
            for event in events:
                if event.type == pygame.KEYDOWN:
                    actions = act_dict.get(event.key, [4, 4])[:env.num_players]
                    break
            if actions is not None:
                break
        
        obs, reward, done, info = env.step(actions)
        # print("Reward:", reward[0][0])
        for i in range(len(obs)):
            print("Obs {}: {}".format(i, obs[i]))
        
        env.render("human")
