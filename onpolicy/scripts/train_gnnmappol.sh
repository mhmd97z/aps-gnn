#!/bin/sh
env="aps"
algo="gnnmappol"
exp="los_hexaps_mrt/ap20_ue6_se1.5_8env/ped2_1step_50ms/gnnmappolf1-samelagr2-4strongest-lambdamax30-ppoepoch10"
seed=1
python train_gnnmappol.py --use_valuenorm --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 8 --n_rollout_threads 8 \
 --num_mini_batch 1 --episode_length 32 --num_env_steps 512000 \
 --ppo_epoch 10 --use_ReLU --lr 7e-4 --critic_lr 7e-4 \
 --user_name "marl" --use_recurrent_policy False --max_grad_norm 1 \
 --gamma 0.01 --use_linear_lr_decay --log_interval 1 \
 --entropy_coef 0.1  --if_update_lagr_per_ue False --if_pid_lagr_update False --if_rnn_gnn False \
 --lagrangian_coef_rate 2.0 --lambda_lagr_max 30.0 --lagr_pid_kp 3 --lagr_pid_ki 3 --lagr_pid_kd 3 \
 --model_dir pretrained_models/los_hexaps_ap20_ue6_ped_1history_f1/4strongest

#  --use_eval