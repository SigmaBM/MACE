import copy

import gym
import numpy as np
from gym.spaces import Box, Discrete
from gym.utils import seeding


class Entity():
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        
class Agent(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.movable = True


class Landmark(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.movable = False
        

class Door(Entity):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.open = False


class Pass(gym.Env):
    def __init__(self, map_ind=0, max_timesteps=300, door_in_obs=False, full_obs=False, joint_count=False, 
                 activate_radius=None, grid_size=30, **kwargs):
        self.num_agents = 2
        self.max_timesteps = max_timesteps
        self.joint_count = joint_count
        self.grid_size = grid_size
        self.door_in_obs = door_in_obs
        self.full_obs = full_obs
        
        self._init_entity(map_ind, grid_size)
        
        self.init_wall_map = np.zeros((grid_size, grid_size))
        self.init_wall_map[:, self.grid_size // 2] = 1
        self.time = 0
        self.agents, self.wall_map, self.door = None, None, None
        self.door_radius = self.grid_size // 10
        self.activate_radius = 1.5 * self.door_radius if activate_radius is None else activate_radius

        self._init_space()
        
        if self.joint_count:
            self.visit_counts = np.zeros(self.num_agents * [self.grid_size, self.grid_size])
        else:
            self.visit_counts = np.zeros((self.num_agents, self.grid_size, self.grid_size))
        self.reset()
    
    def _init_entity(self, map_ind, grid_size):
        if map_ind == 0:
            self.init_agents = [Agent(1 + grid_size // 10, 1 + grid_size // 10),
                                Agent(grid_size // 10, grid_size // 10)]
            self.init_door = Door(grid_size // 2, grid_size // 2)
            self.switches = [Landmark(grid_size // 10, int(grid_size * 0.8)),
                            Landmark(int(grid_size * 0.8), grid_size // 10)]
        elif map_ind == 1:
            self.init_agents = [Agent(1 + grid_size // 10, 1 + grid_size // 10),
                                Agent(grid_size // 10, grid_size // 10)]
            self.init_door = Door(grid_size // 10, grid_size // 2)
            self.switches = [Landmark(grid_size // 10, int(grid_size * 0.8)),
                             Landmark(int(grid_size * 0.8), grid_size // 10)]
        elif map_ind == 2:
            self.init_agents = [Agent(1 + grid_size // 10, 1 + grid_size // 10),
                                Agent(grid_size // 10, grid_size // 10)]
            self.init_door = Door(grid_size // 2, grid_size // 2)
            self.switches = [Landmark(grid_size // 10, int(grid_size * 0.4)),
                             Landmark(grid_size // 2, int(grid_size * 0.8))]
        else:
            raise ValueError("Invalid map index.")
        
    def _init_space(self):
        if self.door_in_obs:
            if self.full_obs:
                self.observation_space = [Box(low=-1, high=1, shape=(5,), dtype=np.float32),
                                          Box(low=-1, high=1, shape=(5,), dtype=np.float32)]
                self.share_observation_space = [Box(low=-1, high=1, shape=(10,), dtype=np.float32),
                                                Box(low=-1, high=1, shape=(10,), dtype=np.float32)]
            else:
                self.observation_space = [Box(low=-1, high=1, shape=(3,), dtype=np.float32),
                                          Box(low=-1, high=1, shape=(3,), dtype=np.float32)]
                self.share_observation_space = [Box(low=-1, high=1, shape=(6,), dtype=np.float32),
                                                Box(low=-1, high=1, shape=(6,), dtype=np.float32)]
        else:
            if self.full_obs:
                self.observation_space = [Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                                          Box(low=-1, high=1, shape=(4,), dtype=np.float32)]
                self.share_observation_space = [Box(low=-1, high=1, shape=(8,), dtype=np.float32),
                                                Box(low=-1, high=1, shape=(8,), dtype=np.float32)]
            else:
                self.observation_space = [Box(low=-1, high=1, shape=(2,), dtype=np.float32),
                                          Box(low=-1, high=1, shape=(2,), dtype=np.float32)]
                self.share_observation_space = [Box(low=-1, high=1, shape=(4,), dtype=np.float32),
                                                Box(low=-1, high=1, shape=(4,), dtype=np.float32)]
        self.action_space = [Discrete(4), Discrete(4)]
    
    @staticmethod
    def ind2ndoor(ind):
        return 1
        
    def seed(self, seed=None):
        self.random, seed = seeding.np_random(seed)
        return seed 
        
    def reset(self):
        self.agents = copy.deepcopy(self.init_agents)
        self.wall_map = copy.deepcopy(self.init_wall_map)
        self.door = copy.deepcopy(self.init_door)
        self.time = 0
        
        if self.joint_count:
            visit_indices = tuple(sum([[a.x, a.y] for a in self.agents], []))
            self.visit_counts[visit_indices] += 1
        else:
            for idx, agent in enumerate(self.agents):
                self.visit_counts[idx, agent.x, agent.y] += 1
        
        if self.full_obs:
            return self._get_full_obs()
        return self._get_obs()
    
    def step(self, actions):
        if not all(type(a) is int for a in actions):
            actions = [a.argmax() for a in actions]
        
        # update agents' position
        for idx, agent in enumerate(self.agents):
            action = actions[idx]
            
            x, y = agent.x, agent.y
            if action == 0: # UP
                if x > 0 and self.wall_map[x - 1, y] == 0:
                    agent.x -= 1
            elif action == 1:   # DOWN
                if x < self.grid_size - 1 and self.wall_map[x + 1, y] == 0:
                    agent.x += 1
            elif action == 2:   # LEFT
                if y > 0 and self.wall_map[x, y - 1] == 0:
                    agent.y -= 1
            else:   # RIGHT
                if y < self.grid_size - 1 and self.wall_map[x, y + 1] == 0:
                    agent.y += 1
        
        # update status of door
        open = False
        for switch in self.switches:
            for agent in self.agents:
                if self._dist(agent, switch) <= self.activate_radius:
                    open = True
        if open:
            self.door.open = True
            self.wall_map[max(0, self.door.x - self.door_radius): 
                          min(self.grid_size, self.door.x + self.door_radius + 1),
                          self.door.y] = 0
        else:
            self.door.open = False
            self.wall_map[max(0, self.door.x - self.door_radius): 
                          min(self.grid_size, self.door.x + self.door_radius + 1), 
                          self.door.y] = 1
        
        # update visit counts
        if self.joint_count:
            visit_indices = tuple(sum([[a.x, a.y] for a in self.agents], []))
            self.visit_counts[visit_indices] += 1
        else:
            for idx, agent in enumerate(self.agents):
                self.visit_counts[idx, agent.x, agent.y] += 1
        
        self.time += 1
        reward, done, info = 0.0, False, {}
        if all([agent.y > self.grid_size // 2 for agent in self.agents]):
            reward = 100.
            done = True
        if self.time >= self.max_timesteps:
            done = True
            info["TimeLimit.truncated"] = True
            
        reward_n = [[reward]] * self.num_agents
        done_n = [done] * self.num_agents
        info_n = [info] * self.num_agents
        
        if self.full_obs:
            return self._get_full_obs(), reward_n, done_n, info_n
        return self._get_obs(), reward_n, done_n, info_n
        
    def _get_obs(self):
        obs_n = [np.array([self.agents[0].x, self.agents[0].y]) / self.grid_size,
                 np.array([self.agents[1].x, self.agents[1].y]) / self.grid_size]
        if self.door_in_obs:
            return [np.concatenate([obs_n[0], [self.door.open]]),
                    np.concatenate([obs_n[1], [self.door.open]])]
        return obs_n
    
    def _get_full_obs(self):
        obs_n = [np.array([self.agents[0].x, self.agents[0].y,
                           self.agents[1].x, self.agents[1].y]) / self.grid_size,
                 np.array([self.agents[1].x, self.agents[1].y,
                           self.agents[0].x, self.agents[0].y]) / self.grid_size]
        if self.door_in_obs:
            return [np.concatenate([obs_n[0], [self.door.open]]),
                    np.concatenate([obs_n[1], [self.door.open]])]
        return obs_n
    
    def _dist(self, e1, e2):
        return np.sqrt((e1.x - e2.x) ** 2 + (e1.y - e2.y) ** 2)    

    def get_visit_counts(self, agent_id=None):
        if agent_id is not None and not self.joint_count:
            return self.visit_counts[agent_id]
        return self.visit_counts
    
    def set_visit_counts(self, visit_counts, agent_id):
        if agent_id is not None and not self.joint_count:
            assert self.visit_counts[agent_id].shape == visit_counts.shape
            self.visit_counts[agent_id] = copy.deepcopy(visit_counts)
        else:
            assert self.visit_counts.shape == visit_counts.shape
            self.visit_counts = copy.deepcopy(visit_counts)
    
    def reset_visit_counts(self):
        self.visit_counts *= 0
        
    def visit_counts_decay(self, decay_coef):
        self.visit_counts *= decay_coef
        
    def render(self, **kwargs):
        map = copy.deepcopy(self.wall_map)
        map[self.switches[0].x][self.switches[0].y] = 2
        map[self.switches[1].x][self.switches[1].y] = 2
        map[self.agents[0].x][self.agents[0].y] = 3
        map[self.agents[1].x][self.agents[1].y] = 3
        print('#' * (self.grid_size + 2))
        for r in range(self.grid_size):
            print('#', end='')
            for c in range(self.grid_size):
                if map[r][c] == 0:
                    print(' ', end='')
                elif map[r][c] == 1:
                    print('#', end='')
                elif map[r][c] == 2:
                    print('□', end='')
                elif map[r][c] == 3:
                    print('o', end='')
            print('#')
        print('#' * (self.grid_size + 2))


if __name__ == "__main__":
    env = Pass(map_ind=2)
    done = [False, False]
    a0_dict = {'w': 0, 's': 1, 'a': 2, 'd': 3}
    a1_dict = {'i': 0, 'k': 1, 'j': 2, 'l': 3}
    while not any(done):
        env.render()
        a0 = None
        while not (a0 in ['w', 's', 'a', 'd']):
            print("Input agent 0's action (w: UP, s: DOWN, a: LEFT, d: RIGHT): ", end='')
            a0 = input()
        a0 = a0_dict[a0]
        a1 = None
        while not (a1 in ['i', 'k', 'j', 'l']):
            print("Input agent 1's action (i: UP, k: DOWN, j: LEFT, l: RIGHT): ", end='')
            a1 = input()
        a1 = a1_dict[a1]
        obs, reward, done, info = env.step([a0, a1])
