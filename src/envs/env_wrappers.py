"""
Modified from OpenAI Baselines code to work with multi-agent envs
"""
import numpy as np
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod
from src.utils.util import tile_images


class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    Used to batch data from multiple copies of an environment, so that
    each observation becomes an batch of observations, and expected action is a batch of actions to
    be applied per-environment.
    """
    closed = False
    viewer = None

    metadata = {
        'render.modes': ['human', 'rgb_array']
    }

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.share_observation_space = share_observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob = env.reset()
            else:
                if np.all(done):
                    ob = env.reset()

            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send((ob))
        elif cmd == 'render':
            if data == "rgb_array":
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == "human":
                env.render(mode=data)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.share_observation_space, env.action_space))
        # 4 commands for gridworld
        elif cmd == 'get_visit_counts':
            if hasattr(env, 'visit_counts'):
                remote.send(env.get_visit_counts(agent_id=data))
            elif hasattr(env.unwrapped, 'visit_counts'):
                remote.send(env.unwrapped.get_visit_counts(agent_id=data))
            else:
                raise NotImplementedError
        elif cmd == 'set_visit_counts':
            if hasattr(env, 'visit_counts'):
                env.set_visit_counts(*data)
            elif hasattr(env.unwrapped, 'visit_counts'):
                env.unwrapped.set_visit_counts(*data)
            else:
                raise NotImplementedError
        elif cmd == 'reset_visit_counts':
            if hasattr(env, 'visit_counts'):
                env.reset_visit_counts()
            elif hasattr(env.unwrapped, 'visit_counts'):
                env.unwrapped.reset_visit_counts()
            else:
                raise NotImplementedError
        elif cmd == 'visit_counts_decay':
            if hasattr(env, 'visit_counts'):
                env.visit_counts_decay(data)
            elif hasattr(env.unwrapped, 'visit_counts'):
                env.unwrapped.visit_counts_decay(data)
            else:
                raise NotImplementedError
        elif cmd == 'get_attr':
            if hasattr(env, data):
                remote.send(getattr(env, data))
            elif hasattr(env.unwrapped, data):
                remote.send(getattr(env.unwrapped, data))
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError


class SubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(self, len(env_fns), observation_space,
                             share_observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos)

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        obs = [remote.recv() for remote in self.remotes]
        return np.stack(obs)

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

    def render(self, mode="rgb_array"):
        for remote in self.remotes:
            remote.send(('render', mode))
        if mode == "rgb_array":   
            frame = [remote.recv() for remote in self.remotes]
            return np.stack(frame) 
        
    def get_visit_counts(self, agent_id=None):
        for remote in self.remotes:
            remote.send(('get_visit_counts', agent_id))
        return sum(remote.recv() for remote in self.remotes)
    
    def set_visit_counts(self, visit_counts, agent_id=None):
        self.remotes[0].send(('set_visit_counts', (visit_counts, agent_id)))

    def reset_visit_counts(self):
        for remote in self.remotes:
            remote.send(('reset_visit_counts', None))
            
    def visit_counts_decay(self, decay_coef):
        for remote in self.remotes:
            remote.send(('visit_counts_decay', decay_coef))

    def get_attr(self, attr_name):
        for remote in self.remotes:
            remote.send(('get_attr', attr_name))
        return [remote.recv() for remote in self.remotes]


# single env
class DummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(self, len(
            env_fns), env.observation_space, env.share_observation_space, env.action_space)
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, rews, dones, infos = map(np.array, zip(*results))

        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            else:
                if np.all(done):
                    obs[i] = self.envs[i].reset()

        self.actions = None
        return obs, rews, dones, infos

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode="human"):
        if mode == "rgb_array":
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == "human":
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

    def get_visit_counts(self, agent_id=None):
        if hasattr(self.envs[0], 'visit_counts'):
            return sum(env.get_visit_counts(agent_id) for env in self.envs)
        else:
            return sum(env.unwrapped.get_visit_counts(agent_id) for env in self.envs)
    
    def set_visit_counts(self, visit_counts, agent_id=None):
        if hasattr(self.envs[0], 'visit_counts'):
            self.envs[0].set_visit_counts(visit_counts, agent_id)
        else:
            self.envs[0].unwrapped.set_visit_counts(visit_counts, agent_id)
    
    def reset_visit_counts(self):
        if hasattr(self.envs[0], 'visit_counts'):
            for env in self.envs:
                env.reset_visit_counts()
        else:
            for env in self.envs:
                env.unwrapped.reset_visit_counts()
    
    def visit_counts_decay(self, decay_coef):
        if hasattr(self.envs[0], 'visit_counts'):
            for env in self.envs:
                env.visit_counts_decay(decay_coef)
        else:
            for env in self.envs:
                env.unwrapped.visit_counts_decay(decay_coef)

    def get_attr(self, attr_name):
        if hasattr(self.envs[0], attr_name):
            return [getattr(env, attr_name) for env in self.envs]
        return [getattr(env.unwrapped, attr_name) for env in self.envs]