"""Based on https://github.com/marlbenchmark/on-policy/blob/main/onpolicy/config.py. """
import argparse


def get_config():
    parser = argparse.ArgumentParser(
        description='onpolicy', formatter_class=argparse.RawDescriptionHelpFormatter)

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default='mappo', choices=["rmappo", "mappo"])
 
    parser.add_argument("--experiment_name", type=str, default="check", help="an identifier to distinguish different experiment.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument("--cuda", action='store_false', default=True, help="by default True, will use GPU to train; or else will use CPU;")
    parser.add_argument("--cuda_deterministic",
                        action='store_false', default=True, help="by default, make sure random seed effective. if set, bypass such function.")
    parser.add_argument("--n_training_threads", type=int,
                        default=1, help="Number of torch threads for training")
    parser.add_argument("--n_rollout_threads", type=int, default=128,
                        help="Number of parallel envs for training rollouts")
    parser.add_argument("--n_eval_rollout_threads", type=int, default=50,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument("--n_render_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for rendering rollouts")
    parser.add_argument("--num_env_steps", type=int, default=2e7,
                        help='Number of environment steps to train (default: 2e7)')
    parser.add_argument("--user_name", type=str, default='marl',help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--use_wandb", action='store_true', help="[for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.")
    parser.add_argument("--use_parallel", action='store_true', help="by default False, will use single process to run env. or else will use multi-processing to run env.")

    # env parameters
    parser.add_argument("--env_name", type=str, default='Pass-v0', help="specify the name of environment")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")
    parser.add_argument("--episode_length", type=int, default=300, help="Max length for any episode")

    # network parameters
    parser.add_argument("--share_policy", action='store_true', help='Whether agent share the same policy')
    parser.add_argument("--use_centralized_V", action='store_true', help="Whether to use centralized V function")
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Dimension of hidden layers for actor/critic networks") 
    parser.add_argument("--layer_N", type=int, default=1,
                        help="Number of layers for actor/critic networks")
    parser.add_argument("--use_ReLU", action='store_true', help="Whether to use ReLU")
    parser.add_argument("--use_popart", action='store_true', default=False, help="by default False, use PopArt to normalize rewards.")
    parser.add_argument("--use_valuenorm", action='store_true', default=False, help="by default True, use running mean and std to normalize rewards.")
    parser.add_argument("--use_layernorm", action='store_true', help="Whether to use layer norm to normalize hidden states.")
    parser.add_argument("--use_feature_normalization", action='store_true', help="Whether to apply layernorm to the inputs")
    parser.add_argument("--use_orthogonal", action='store_false', default=True,
                        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases")
    parser.add_argument("--gain", type=float, default=0.01,
                        help="The gain # of last action layer")
    
    # recurrent parameters
    parser.add_argument("--use_naive_recurrent_policy", action='store_true',
                        default=False, help='Whether to use a naive recurrent policy')
    parser.add_argument("--use_recurrent_policy", action='store_true',
                        default=False, help='use a recurrent policy')
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument("--data_chunk_length", type=int, default=10,
                        help="Time length of chunks used to train a recurrent_policy")

    # optimizer parameters
    parser.add_argument("--actor_lr", type=float, default=7e-4,
                        help='learning rate (default: 7e-4)')
    parser.add_argument("--critic_lr", type=float, default=7e-4,
                        help='critic learning rate (default: 7e-4)')
    parser.add_argument("--opti_eps", type=float, default=1e-5,
                        help='RMSprop optimizer epsilon (default: 1e-5)')
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=10, help='number of ppo epochs (default: 10)')
    parser.add_argument("--use_clipped_value_loss",
                        action='store_false', default=True, help="by default, clip loss value. If set, do not clip loss value.")
    parser.add_argument("--clip_param", type=float, default=0.2,
                        help='ppo clip parameter (default: 0.2)')
    parser.add_argument("--num_mini_batch", type=int, default=1,
                        help='number of batches for ppo (default: 1)')
    parser.add_argument("--entropy_coef", type=float, default=0.01,
                        help='entropy term coefficient (default: 0.01)')
    parser.add_argument("--value_loss_coef", type=float,
                        default=1, help='value loss coefficient (default: 0.5)')
    parser.add_argument("--use_max_grad_norm",
                        action='store_false', default=True, help="by default, use max norm of gradients. If set, do not use.")
    parser.add_argument("--max_grad_norm", type=float, default=10.0,
                        help='max norm of gradients (default: 0.5)')
    parser.add_argument("--use_gae", action='store_false',
                        default=True, help='use generalized advantage estimation')
    parser.add_argument("--gamma", type=float, default=0.99,
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument("--gae_lambda", type=float, default=0.97,
                        help='gae lambda parameter (default: 0.97)')
    parser.add_argument("--use_proper_time_limits", action='store_false', default=True,
                        help='compute returns taking into account time limits')
    parser.add_argument("--use_huber_loss", action='store_false', default=True, help="by default, use huber loss. If set, do not use huber loss.")
    parser.add_argument("--use_value_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in value loss.")
    parser.add_argument("--use_policy_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in policy loss.")
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")
    
    # novel parameters
    parser.add_argument("--novel_type", type=int, default=0, choices=[0, 1, 2, 3, 4],
                        help="0: no novelty;"
                             "1: prediction error (unused);"
                             "2: TD error (unused);"
                             "3: count-based novelty;"
                             "4: RND.")
    parser.add_argument("--use_novel_normalization", action='store_true', help="Whether to use novel normalization")
    parser.add_argument("--self_coef", type=float, default=0.0, help="own novelty coefficient (default: 0.0)")
    parser.add_argument("--other_coef", type=float, default=0.0, help="other novelty coefficient (default: 0.0)")
    parser.add_argument("--novel_max", action="store_true", help="use maximum novelty, instead of summation")
    parser.add_argument("--use_novel_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in novelty.")

    # novel 3 - count-based novelty
    parser.add_argument("--count_pow", type=float, default=0.5, 
                        help="novelty = 1. / count ^ count_pow (default: 0.5)")
    parser.add_argument("--count_prev_epoch", action="store_true",
                        help="by default False, whether to use count excluding current epoch.")
    parser.add_argument("--count_decay_coef", type=float, default=1.0,
                        help="count = count ^ count_decay_coef + current_count (default: 1.0)")
    parser.add_argument("--joint_count", action="store_true", help="whether to use joint count, i.e., global novelty.")
    # novel 4 - RND
    parser.add_argument("--rnd_lr", type=float, default=3e-4, help="Learning rate for RND (default: 3e-4).")
    parser.add_argument("--rnd_rep_size", type=int, default=32, help="Representation size for RND (default: 32).")
    
    # hindsight parameter
    parser.add_argument("--use_hdd", action="store_true", help="whether to use hindsight distribution (HDD)")
    parser.add_argument("--hdd_reduce", action="store_true", help="whether to use reduced HDD")
    parser.add_argument("--hdd_log_weight", default=True, action="store_false", 
                        help="by default, use form: log (h / pi). If set, use form: 1 - pi / h")
    parser.add_argument("--hdd_weight_only", action="store_true", 
                        help="If set, use mutual information between action and return")
    parser.add_argument("--hdd_weight_one", action="store_true", help="If set, use return of others only")
    parser.add_argument("--hdd_norm_adv", action="store_true", help="whether to normalize hindsight advantage")
    parser.add_argument("--ir_coef", type=float, default=0.0, help="hindsight term as intrinsic reward coefficient (default: 0.0)")
    parser.add_argument("--ad_coef", type=float, default=0.0, help="hindsight term as advantage coefficient (default: 0.0)")
    parser.add_argument("--use_hdd_active_masks",
                        action='store_false', default=True, help="by default True, whether to mask useless data in HDD.")
    # hindsight distribution - count-based
    parser.add_argument("--hdd_count", action="store_true", 
                        help="by default, use network-based HDD."
                             "If set, use count-based HDD.")
    parser.add_argument("--hdd_count_init1", action="store_true",
                        help="by default, use zero to initialize count-based HDD." 
                             "If set, use one to initialize count-based HDD.")
    parser.add_argument("--hdd_count_window", type=int, default=0, help="window size of count-based HDD")
    parser.add_argument("--hdd_count_decay", type=float, default=1.0, help="decay rate of count-based HDD")
    parser.add_argument("--hdd_ret_nbins", type=int, default=30, help="number of bins of return")
    parser.add_argument("--hdd_ret_ub", type=float, default=90., help="upper bound of return")
    parser.add_argument("--hdd_reverse_ret", action="store_true", help="whether to use reversed return in HDD")
    parser.add_argument("--hdd_last_rew", action="store_true", help="whether to use last step reward in HDD")
    parser.add_argument("--hdd_ret_padding", action="store_true", help="whether to use padding return at last step each episode")
    parser.add_argument("--hdd_ratio_clip", type=float, default=None, help="clip the ratio between pi and h")
    parser.add_argument("--save_hdd_count", action="store_false", default=True, 
                        help="by default, save hindsight count matrix. If set, do not save.")
    # discretize novel
    parser.add_argument("--discrete_novel_in_adv", action="store_true", 
                        help="whether to use stabilized discreted novel to compute other novelty return")
    parser.add_argument("--discrete_novel_in_hd", action="store_true",
                        help="whether to use stabilized discreted novel to compute other novelty return in hindsight dsitribution")
    parser.add_argument("--discrete_nbins", type=int, default=5, help="number of bins of discretized novel")
    parser.add_argument("--discrete_momentum", type=float, default=0.9, help="momentum of discretized novel bound update")
    parser.add_argument("--hdd_gamma", type=float, default=0.99, help="gamma used in HDD")
    # misc
    parser.add_argument("--hdd_count_door", action="store_false", default=True, 
                        help="by default, use door in hindsight count. If set, do not use door in hindsight count.")
    # hindsight distribution - network-based
    parser.add_argument("--hdd_lr", type=float, default=3e-4, help="learning rate of HDD network")
    parser.add_argument("--hdd_buffer_size", type=int, default=1000000, help="buffer size for learning HDD")
    parser.add_argument("--hdd_epoch", type=int, default=40, help="number of epochs for learning HDD")
    parser.add_argument("--hdd_batch_size", type=int, default=50000, help="batch size for learning HDD")

    # run parameters
    parser.add_argument("--use_linear_lr_decay", action='store_true',
                        default=False, help='use a linear schedule on the learning rate')
    # save parameters
    parser.add_argument("--save_interval", type=int, default=50, help="time duration between contiunous twice models saving.")

    # log parameters
    parser.add_argument("--run_dir", type=str, default=None, help="by default None. set the path to save log")
    parser.add_argument("--stdout", action="store_false", default=True, help="whether to print log to stdout")
    parser.add_argument("--log_interval", type=int, default=5, help="time duration between contiunous twice log printing.")

    # eval parameters
    parser.add_argument("--use_eval", action='store_true', default=False, help="by default, do not start evaluation. If set`, start evaluation alongside with training.")
    parser.add_argument("--use_run", action='store_true', default=False)
    parser.add_argument("--eval_interval", type=int, default=25, help="time duration between contiunous twice evaluation progress.")
    parser.add_argument("--eval_episodes", type=int, default=32, help="number of episodes of a single evaluation.")

    # render parameters
    parser.add_argument("--save_gifs", action='store_true', default=False, help="by default, do not save render video. If set, save video.")
    parser.add_argument("--use_render", action='store_true', default=False, help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.")
    parser.add_argument("--render_episodes", type=int, default=5, help="the number of episodes to render a given env")
    parser.add_argument("--ifi", type=float, default=0.1, help="the play interval of each rendered image in saved video.")

    # pretrained parameters
    parser.add_argument("--model_dir", type=str, default=None, help="by default None. set the path to pretrained model.")

    return parser
