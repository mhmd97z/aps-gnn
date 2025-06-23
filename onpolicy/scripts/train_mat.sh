#!/bin/sh
env="aps"
algo="mat"
exp="mrt/ap20_ue6_sinr0_lastg_8env/veh_10step_50ms/mat/localpsum1_sumcost0_conncost1_pcoef5" # 001
seed=1
python train_mat.py --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 8 --gamma 0.01 --use_wandb False \
 --n_rollout_threads 8 --num_mini_batch 1 --episode_length 100 --use_valuenorm \
 --ppo_epoch 5 --clip_param 0.2 --max_grad_norm 0.5 \
 --lr 0.0001 --critic_lr 0.0001  --use_linear_lr_decay \
 --num_env_steps 300000 --entropy_coef 0.0005 --log_interval 1
