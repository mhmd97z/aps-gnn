#!/bin/sh
env="aps"
algo="matl"
exp="test/los_hexaps_mrt/ap20_ue6_se1.5_8env/veh_1step_50ms/matlf1-lagrcoef2-ppoepoch10-ppoclip0.1"
seed=1
python train_matl.py --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 1 --n_rollout_threads 1 \
 --gamma 0.01 --use_wandb False \
 --num_mini_batch 1 --episode_length 32 --use_valuenorm \
 --ppo_epoch 10 --clip_param 0.1 --max_grad_norm 1 \
 --lr 0.0001 --critic_lr 0.0001  --use_linear_lr_decay \
 --num_env_steps 512000 --entropy_coef 0.01 --log_interval 1 \
 --lagrangian_coef_rate 2.0 --lambda_lagr_max 10000.0

echo ${exp} ${algo} trained!
