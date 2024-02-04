"""Based on https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/scripts/train/train_mpe.py. """
import datetime
import os
import socket
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch
import wandb

from src.config import get_config
from src.envs.env_wrappers import DummyVecEnv, SubprocVecEnv
from src.envs.grid_world import Pass, SecretRooms
from src.envs.overcooked.overcooked_env import Overcooked
from src.utils.logger import Logger

GRID_ENV = {
    "Pass-v0": Pass,
    "SecretRoom-v0": SecretRooms,
}


def make_grid_env(all_args):
    config = {
        "map_ind": all_args.map_ind,
        "max_timesteps": all_args.max_timesteps,
        "door_in_obs": all_args.door_in_obs,
        "full_obs": all_args.full_obs,
        "joint_count": all_args.joint_count,
        "activate_radius": all_args.activate_radius,
        "grid_size": all_args.grid_size,
    }
    
    try:
        env = GRID_ENV[all_args.env_name](**config)
    except KeyError:
        raise KeyError(f"Env {all_args.env_name} not found.")
    all_args.count_lookup_index = 0

    return env


def make_overcook_env(all_args):
    config = {
        "layout_name": all_args.layout_name, 
        "max_timesteps": all_args.max_timesteps,
        "obs_type": all_args.obs_type,
        "multi_round": all_args.multi_round,
    }
    
    env = Overcooked(**config)
    return env


def get_obs_range(all_args):
    L = all_args.grid_size
    
    obs_range, obs_scale = [L, L], [L, L]
    if all_args.full_obs:
        obs_range = [L, L] * all_args.num_agents
        obs_scale = [L, L] * all_args.num_agents
    if all_args.door_in_obs:
        num_doors = GRID_ENV[all_args.env_name].ind2ndoor(all_args.map_ind)
        if all_args.hdd_count_door:
            obs_range += [2] * num_doors
            obs_scale += [1] * num_doors
        else:
            obs_range += [0] * num_doors
            obs_scale += [0] * num_doors
    
    all_args.obs_range = obs_range
    all_args.obs_scale = obs_scale


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name in GRID_ENV:
                env = make_grid_env(all_args)
            elif all_args.env_name == "Overcooked-v0":
                env = make_overcook_env(all_args)
            else:
                raise ValueError(f"Env {all_args.env_name} not found.")
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    if all_args.n_rollout_threads == 1 or not all_args.use_parallel:
        return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name in GRID_ENV:
                env = make_grid_env(all_args)
            elif all_args.env_name == "Overcooked-v0":
                env = make_overcook_env(all_args)
            else:
                raise ValueError(f"Env {all_args.env_name} not found.")
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1 or not all_args.use_parallel:
        return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    # Common arguments
    parser.add_argument("--n_agents", type=int, default=2)
    parser.add_argument("--max_timesteps", type=int, default=300)
    parser.add_argument("--map_ind", type=int, default=0)
    parser.add_argument("--layout_name", type=str, default="small_1room_1")
    
    # GridWorld
    parser.add_argument("--grid_size", type=int, default=30)
    parser.add_argument("--door_in_obs", default=True, action="store_false")
    parser.add_argument("--full_obs", action="store_true")
    parser.add_argument("--activate_radius", type=float, default=1.0)
        
    parser.add_argument("--save_visit", "-sc", action="store_true")
    parser.add_argument("--save_int", "-si", action="store_true")
    parser.add_argument("--save_value", "-sv", action="store_true")
    parser.add_argument("--save_ratio", "-sr", action="store_true")
    
    # Overcooked
    parser.add_argument("--obs_type", type=str, default="vector", choices=["vector", "image"])
    parser.add_argument("--multi_round", action="store_true")
    
    return parser.parse_args(args)


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # check recurrent setting
    if all_args.algorithm_name == "rmappo":
        assert (all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy), ("check recurrent policy!")
    elif all_args.algorithm_name == "mappo":
        assert (all_args.use_recurrent_policy == False and all_args.use_naive_recurrent_policy == False), (
            "check recurrent policy!")
    else:
        raise NotImplementedError

    # run dir
    if all_args.run_dir is None:
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
    else:
        run_dir = Path(all_args.run_dir + "/results")
    t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")
    all_args.experiment_name += "-" + t0 + "-seed" + str(all_args.seed)
    map_name = str(all_args.map_ind) if all_args.env_name in GRID_ENV else all_args.layout_name
    run_dir = run_dir / all_args.env_name / map_name / all_args.algorithm_name / all_args.experiment_name
    
    index = 1
    while run_dir.exists():
        run_dir = run_dir.parent / (all_args.experiment_name + "-" + str(index))
        index += 1
    os.makedirs(str(run_dir))
    
    Logger(run_dir, all_args.stdout)

    # wandb
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                         str(all_args.experiment_name) +
                         "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)
    
    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" + \
        str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(all_args.user_name))

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.n_agents
    if all_args.env_name in GRID_ENV.keys():
        get_obs_range(all_args)
    
    for obs_space, act_space in zip(envs.observation_space, envs.action_space):
        print("obs space:", obs_space, "; act space:", act_space)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }
    
    if all_args.novel_type == 3:  # count
        from src.runner.runner_grid import Runner
    elif all_args.novel_type == 4:  # continous
        from src.runner.runner_cont import Runner
    else:
        raise NotImplementedError

    # run experiments
    runner = Runner(config)
    runner.run()
    
    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
