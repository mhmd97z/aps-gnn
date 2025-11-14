#!/bin/sh
env="aps"
algo="gnnmappo"
exp="los_hexaps_mrt/ap20_ue6_se1.5_8env/ped_1step_50ms/gnnmappof1r10-4strongest-secoef15-ppoepoch10-ppoclip0.1"
seed=1
python train_gnnmappo.py --use_valuenorm --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 8 --n_rollout_threads 8 \
 --num_mini_batch 1 --episode_length 32 --num_env_steps 512000 \
 --ppo_epoch 10 --use_ReLU --lr 7e-4 --critic_lr 7e-4 \
 --user_name "marl" --use_recurrent_policy False --max_grad_norm 1 \
 --gamma 0.01 --use_linear_lr_decay --log_interval 1 --clip_param 0.1 \
 --entropy_coef 0.1 --if_rnn_gnn True \
 --model_dir pretrained_models/los_hexaps_ap20_ue6_ped_10history_f1/4strongest/

echo ${exp} ${algo} trained!
