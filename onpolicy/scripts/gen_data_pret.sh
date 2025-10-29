#!/bin/sh
env="aps"
algo="gnnmappol"
seed=1
python gen_data_pret.py --env_name ${env} --algorithm_name ${algo} --k 4 \
 --seed ${seed} --n_training_threads 16 --n_rollout_threads 16 --episode_length 100 --num_env_steps 100000
