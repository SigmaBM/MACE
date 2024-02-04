"""Based on https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/scripts/train/train_smac.py. """
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
from src.envs.env_wrappers_smac import DummyVecEnv, SubprocVecEnv
from src.envs.starcraft2.smac_maps import get_map_params
from src.envs.starcraft2.StarCraft2_Env import StarCraft2Env
from src.runner.runner_smac import Runner
from src.utils.logger import Logger


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            if all_args.env_name == "StarCraft2":
                extra_configs = {
                    "reward_sparse": all_args.reward_sparse,
                    "reward_semi_sparse": all_args.reward_semi_sparse,
                    "reward_only_positive": all_args.reward_only_positive,
                    "reward_scale": all_args.reward_scale
                }
                env = StarCraft2Env(all_args, **extra_configs)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
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
            if all_args.env_name == "StarCraft2":
                extra_configs = {
                    # "reward_sparse": all_args.reward_sparse,
                    # "reward_semi_sparse": all_args.reward_semi_sparse,
                    # "reward_only_positive": all_args.reward_only_positive,
                    # "reward_scale": all_args.reward_scale
                }
                env = StarCraft2Env(all_args, **extra_configs)
            else:
                print("Can not support the " + all_args.env_name + "environment.")
                raise NotImplementedError
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    if all_args.n_eval_rollout_threads == 1 or not all_args.use_parallel:
        return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument("--n_agents", type=int, default=3)
    parser.add_argument("--map_name", type=str, default='3m', help="Which smac map to run on")
    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=False)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_false', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_false', default=True)

    parser.add_argument("--reward_sparse", action='store_true', default=False)
    parser.add_argument("--reward_semi_sparse", action='store_true', default=False)
    parser.add_argument("--reward_only_positive", action='store_true', default=False)
    parser.add_argument("--reward_scale", action='store_true', default=False)

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
    run_dir = run_dir / all_args.env_name / all_args.map_name / all_args.algorithm_name / all_args.experiment_name
    
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
                         group=all_args.map_name,
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

    assert not (all_args.reward_sparse and all_args.reward_semi_sparse), \
        "reward_sparse and reward_semi_sparse can not be True at the same time"
    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = get_map_params(all_args.map_name)["n_agents"]
    assert num_agents == all_args.num_agents, "num_agents should be equal to all_args.num_agents"
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
