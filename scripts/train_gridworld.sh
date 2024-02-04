# Pass
env="Pass-v0"
ind=0
nagt=2

# SecretRoom
# env="SecretRoom-v0"
# ind=20
# nagt=2

# MultiRoom
# env="SecretRoom-v0"
# ind=33
# nagt=3

python -m src.train --env_name ${env} --map_ind ${ind} --n_agents ${nagt} \
--num_env_steps 80000000 --use_eval --n_eval_rollout_threads 1 --eval_interval 100 --save_interval 500 \
--algorithm_name rmappo --use_recurrent_policy --entropy_coef 0.05 \
--novel_type 3 --self_coef 10.0 --other_coef 10.0 \
--use_hdd --ir_coef 0.1 --hdd_count --hdd_count_window 10 --discrete_novel_in_hd \
--save_hdd_count --run_dir logs --experiment_name MACE --seed 0