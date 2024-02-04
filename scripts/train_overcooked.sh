# Base
name="small_coor_2"
steps=10000000
bsz=100000

# Narrow
# name="small_coor_hard_pass_2"
# steps=20000000
# bsz=100000

# Large
# name="middle_coor_fo_2"
# steps=40000000
# bsz=200000

python -m src.train --env_name Overcooked-v0 --layout_name ${name} \
--num_env_steps ${steps} --use_parallel --n_rollout_threads 32 \
--use_eval --n_eval_rollout_threads 1 --eval_interval 100 --save_interval 100 \
--algorithm_name rmappo --use_recurrent_policy --entropy_coef 0.05 \
--novel_type 4 --self_coef 1.0 --other_coef 1.0 \
--use_hdd --ir_coef 0.1 --hdd_ret_ub 0.5 \
--hdd_buffer_size $((${bsz}+200)) --hdd_batch_size ${bsz} \
--run_dir logs --experiment_name MACE --seed 0