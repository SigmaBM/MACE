import copy

import gym
import numpy as np
from gym.spaces import Box, Discrete
from gym.utils import seeding


class Entity(object):
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


class Switch(Landmark):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.on = False
        
    def _dist(self, agent):
        return np.sqrt((self.x - agent.x) ** 2 + (self.y - agent.y) ** 2)
    
    def update(self, agents, activate_radius):
        self.on = False
        for agent in agents:
            if self._dist(agent) <= activate_radius:
                self.on = True
                break
        

class Door(Entity):
    def __init__(self, x, y, direction, switches_to_open):
        super().__init__(x, y)
        self.d = direction  # 0: vertical, 1: horizontal
        self.s = switches_to_open
        self.open = False

    def update(self, switches):
        self.open = False
        for indices in self.s:
            if all([switches[i].on for i in indices]):
                self.open = True
                break
        
        
class SecretRooms(gym.Env):
    def __init__(self, map_ind=20, max_timesteps=300, door_in_obs=False, full_obs=False, joint_count=False, 
                 activate_radius=None, grid_size=25, **kwargs):
        self.num_agents = map_ind // 10
        self.map_ind = map_ind
        self.max_timesteps = max_timesteps
        self.joint_count = joint_count
        self.grid_size = grid_size
        self.door_in_obs = door_in_obs
        self.full_obs = full_obs
        
        self._init_entity(map_ind, grid_size)
        self._init_wall(grid_size)
        self._init_space()
        
        self.time = 0
        self.agents, self.wall_map, self.doors = None, None, None
        self.door_radius = 1
        self.activate_radius = activate_radius or 1.5 * self.door_radius
        
        if self.joint_count:
            self.visit_counts = np.zeros(self.num_agents * [self.grid_size, self.grid_size])
        else:
            self.visit_counts = np.zeros((self.num_agents, self.grid_size, self.grid_size))
        self.reset()
    
    def _init_entity(self, map_ind, grid_size):
        ot_grid_size = grid_size // 3       # 1/3 grid_size
        tt_grid_size = grid_size * 2 // 3   # 2/3 grid_size
        
        if map_ind // 10 == 2:
            self.init_agents = [Agent(1 + grid_size // 10, 1 + grid_size // 10),
                                Agent(grid_size // 10, grid_size // 10)]
            self.init_doors = [Door(ot_grid_size // 2, grid_size // 2, 0, [[0], [1]]),
                               Door((ot_grid_size + tt_grid_size) // 2, grid_size // 2, 0, [[0], [2]]),
                               Door((tt_grid_size + grid_size) // 2, grid_size // 2, 0, [[0], [3]])]
            self.switches = [Switch(int(grid_size * 0.8), int(grid_size * 0.2)),          # main switch
                             Switch(self.init_doors[0].x, int(self.grid_size * 0.8)),     # switch of room 1
                             Switch(self.init_doors[1].x, int(self.grid_size * 0.8)),     # switch of room 2
                             Switch(self.init_doors[2].x, int(self.grid_size * 0.8))]     # switch of room 3
            
            if map_ind % 10 == 0:
                self.target_room = 1
            elif map_ind % 10 == 1:
                self.target_room = 2
            elif map_ind % 10 == 2:
                self.target_room = 3
            else:
                raise NotImplementedError(f"Not support map_ind {map_ind}.")
        elif map_ind // 10 == 3:
            self.init_agents = [Agent(grid_size // 10, grid_size // 10),
                                Agent(grid_size // 10 + 2, grid_size // 10),
                                Agent(grid_size // 10, grid_size // 10 + 2)]
            self.init_doors = [Door(ot_grid_size // 2, grid_size // 2, 0, [[0], [1]]),
                               Door((ot_grid_size + tt_grid_size) // 2, grid_size // 2, 0, [[0], [2]]),
                               Door((tt_grid_size + grid_size) // 2, grid_size // 2, 0, [[0], [3]])]
            self.switches = [Switch(int(grid_size * 0.8), int(grid_size * 0.2)),          # main switch
                             Switch(self.init_doors[0].x, int(self.grid_size * 0.8)),     # switch of room 1
                             Switch(self.init_doors[1].x, int(self.grid_size * 0.8)),     # switch of room 2
                             Switch(self.init_doors[2].x, int(self.grid_size * 0.8))]     # switch of room 3
            
            if map_ind % 10 == 0:
                # (a0, s0) -> (a1, s1) & (a2, s2) -> (a0, s3)
                self.init_doors[0].s = [[0], [3]]
                self.init_doors[1].s = [[0], [3]]
                self.init_doors[2].s = [[1, 2], [3]]
                self.target_room = 3
            elif map_ind % 10 == 1:
                # (a0, s0) -> (a1, s1) -> (a0, s3) / (a2, s3) -> (a2, s2) / (a0, s2)
                self.init_doors[0].s = [[0], [2]]
                self.init_doors[1].s = [[2], [3]]
                self.init_doors[2].s = [[1], [2]]
                self.target_room = 2
            elif map_ind % 10 == 2:
                # (a0, s0) -> (a1, s1) & (a2, s2) -> (a0, s3)
                self.init_doors[0].s = [[0]]
                self.init_doors[1].s = [[0]]
                self.init_doors[2].s = [[1, 2]]
                self.init_doors.append(Door(ot_grid_size, grid_size * 3 // 4, 1, [[3]]))
                self.init_doors.append(Door(tt_grid_size, grid_size * 3 // 4, 1, [[3]]))
                self.target_room = 3
            elif map_ind % 10 == 3:
                # (a0, s0) -> (a1, s1) -> (a0, s3) / (a2, s3) -> (a2, s2) / (a0, s2)
                self.init_doors[0].s = [[0]]
                self.init_doors[1].s = [[3]]
                self.init_doors[2].s = [[1]]
                self.init_doors.append(Door(ot_grid_size, grid_size * 3 // 4, 1, [[2]]))
                self.init_doors.append(Door(tt_grid_size, grid_size * 3 // 4, 1, [[2]]))
                self.target_room = 2
            elif map_ind % 10 == 4:
                # (a0, s0) -> (a1, s1) -> (a2, s3) -> (a0, s2)
                self.init_doors[0].s = [[0]]
                self.init_doors[1].s = [[3]]
                self.init_doors[2].s = [[0, 1]]
                self.init_doors.append(Door(ot_grid_size, grid_size * 3 // 4, 1, [[2]]))
                self.init_doors.append(Door(tt_grid_size, grid_size * 3 // 4, 1, [[2]]))
                self.target_room = 2
            else:
                raise NotImplementedError(f"Not support map_ind {map_ind}.")
        else:
            raise NotImplementedError(f"Not support map_ind {map_ind}.")
    
    def _init_wall(self, grid_size):
        ot_grid_size = grid_size // 3
        tt_grid_size = grid_size * 2 // 3
        self.init_wall_map = np.zeros((grid_size, grid_size))
        self.init_wall_map[:, grid_size // 2] = 1
        self.init_wall_map[ot_grid_size, grid_size // 2:] = 1
        self.init_wall_map[tt_grid_size, grid_size // 2:] = 1
        
    def _init_space(self):
        L = self.num_agents * 2 if self.full_obs else 2
        if self.door_in_obs:
            L += len(self.init_doors)
            
        self.observation_space = [
            Box(low=0, high=1, shape=(L,), dtype=np.float32) for _ in range(self.num_agents)]
        self.share_observation_space = [
            Box(low=0, high=1, shape=(L * self.num_agents,), dtype=np.float32) for _ in range(self.num_agents)]
        self.action_space = [Discrete(4) for _ in range(self.num_agents)]
    
    def _in_target_room(self, agent, target_room):
        if target_room == 1:
            return (agent.y > self.grid_size // 2) and (agent.x < self.grid_size // 3)
        if target_room == 2:
            return (agent.y > self.grid_size // 2) and (agent.x > self.grid_size // 3) and \
                (agent.x < self.grid_size * 2 // 3)
        if target_room == 3:
            return (agent.y > self.grid_size // 2) and (agent.x > self.grid_size * 2 // 3)
        return False
    
    @staticmethod
    def ind2ndoor(ind):
        if ind // 10 == 2:
            return 3
        if ind // 10 == 3:
            if ind % 10 in [0, 1]:
                return 3
            if ind % 10 in [2, 3, 4]:
                return 5
        raise ValueError(f"Unknown map index {ind}.")

    def seed(self, seed=None):
        self.random, seed = seeding.np_random(seed)
        return seed 
        
    def reset(self):
        self.agents = copy.deepcopy(self.init_agents)
        self.wall_map = copy.deepcopy(self.init_wall_map)
        self.doors = copy.deepcopy(self.init_doors)
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
            if action == 0:     # UP
                if x > 0 and self.wall_map[x - 1, y] == 0:
                    agent.x -= 1
            elif action == 1:   # DOWN
                if x < self.grid_size - 1 and self.wall_map[x + 1, y] == 0:
                    agent.x += 1
            elif action == 2:   # LEFT
                if y > 0 and self.wall_map[x, y - 1] == 0:
                    agent.y -= 1
            else:               # RIGHT
                if y < self.grid_size - 1 and self.wall_map[x, y + 1] == 0:
                    agent.y += 1
        
        # update status of switches and doors
        self.wall_map = copy.deepcopy(self.init_wall_map)
        for switch in self.switches:
            switch.update(self.agents, self.activate_radius)
        for door in self.doors:
            door.update(self.switches)
            if door.open:
                if door.d == 0: # vertical door
                    self.wall_map[door.x - self.door_radius:
                                  door.x + self.door_radius + 1, door.y] = 0
                else:           # horizontal door
                    self.wall_map[door.x, door.y - self.door_radius:
                                  door.y + self.door_radius + 1] = 0
        
        # update visit counts
        if self.joint_count:
            visit_indices = tuple(sum([[a.x, a.y] for a in self.agents], []))
            self.visit_counts[visit_indices] += 1
        else:
            for idx, agent in enumerate(self.agents):
                self.visit_counts[idx, agent.x, agent.y] += 1
        
        self.time += 1
        reward, done, info = 0.0, False, {}
        if all([self._in_target_room(agent, self.target_room) for agent in self.agents]):
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
        obs_n = [np.array([agent.x, agent.y]) / self.grid_size for agent in self.agents]
        if self.door_in_obs:
            return [np.concatenate([obs, [d.open for d in self.doors]]) for obs in obs_n]
        return obs_n
    
    def _get_full_obs(self):
        obs = np.concatenate([np.array([agent.x, agent.y]) / self.grid_size for agent in self.agents])
        if self.door_in_obs:
            obs = np.copncatenate([obs, [d.open for d in self.doors]])
        return [obs for _ in range(self.num_agents)]

    def get_visit_counts(self, agent_id=None):
        if agent_id is not None:
            return self.visit_counts[agent_id]
        return self.visit_counts
    
    def set_visit_counts(self, visit_counts, agent_id):
        if agent_id is None:
            assert self.visit_counts.shape == visit_counts.shape
            self.visit_counts = copy.deepcopy(visit_counts)
        else:
            assert self.visit_counts[agent_id].shape == visit_counts.shape
            self.visit_counts[agent_id] = copy.deepcopy(visit_counts)
    
    def reset_visit_counts(self):
        self.visit_counts *= 0
        
    def visit_counts_decay(self, decay_coef):
        self.visit_counts *= decay_coef
        
    def render(self, **kwargs):
        map = copy.deepcopy(self.wall_map)
        for switch in self.switches:
            map[switch.x, switch.y] = 2
        for agent in self.agents:
            map[agent.x, agent.y] = 3
        for door in self.doors:
            if door.d == 0:
                map[door.x - self.door_radius:door.x + self.door_radius + 1, door.y] = 4
            else:
                map[door.x, door.y - self.door_radius:door.y + self.door_radius + 1] = 5
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
                elif map[r][c] == 4:
                    print('|', end='')
                elif map[r][c] == 5:
                    print('-', end='')
            print('#')
        print('#' * (self.grid_size + 2))
        

if __name__ == "__main__":
    env = SecretRooms(map_ind=31, activate_radius=1, grid_size=15)
    done = [False, False, False]
    a_dict = {'w': 0, 's': 1, 'a': 2, 'd': 3}
    while not any(done):
        env.render()
        a0 = None
        while not (a0 in ['w', 's', 'a', 'd']):
            print("Input agent 0's action: ", end='')
            a0 = input()
        a0 = a_dict[a0]
        a1 = None
        while not (a1 in ['w', 's', 'a', 'd']):
            print("Input agent 1's action: ", end='')
            a1 = input()
        a1 = a_dict[a1]
        if env.num_agents == 2:
            obs, reward, done, info = env.step([a0, a1])
        else:
            a2 = None
            while not (a2 in ['w', 's', 'a', 'd']):
                print("Input agent 2's action: ", end='')
                a2 = input()
            a2 = a_dict[a2]
            obs, reward, done, info = env.step([a0, a1, a2])
        print(obs)
        print(reward)
