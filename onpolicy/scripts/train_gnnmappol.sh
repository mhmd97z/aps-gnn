#!/bin/sh
env="aps"
algo="gnnmappol"
exp="mrt_allsinr_los_se_hexaps/ap20_ue6_se1_8env/ped2_1step_50ms/gnnmappol"
seed=1
python train_gnnmappol.py --use_valuenorm --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 8 --n_rollout_threads 8 \
 --num_mini_batch 1 --episode_length 100 --num_env_steps 300000 \
 --ppo_epoch 5 --use_ReLU --lr 7e-4 --critic_lr 7e-4 \
 --user_name "marl" --use_recurrent_policy False --max_grad_norm 1 \
 --gamma 0.01 --use_linear_lr_decay --log_interval 1 \
 --entropy_coef 0.1 \
#  --model_dir /home/mzi/aps-gnn/onpolicy/results/aps/gnnmappo/pretraining/los_hexaps_ap20_ue6_ped/4strongest/run2/models
#  --use_eval \

#  --model_dir /home/mzi/aps-gnn/onpolicy/results/aps/gnnmappo/pretraining/los_hexaps_ap20_ue6_ped/4strongest/run2/models
#  --model_dir /home/mzi/aps-gnn/onpolicy/results/aps/gnnmappo/mrt_allsinr_los_se_hexaps/ap20_ue6_se1_lastg_8env/ped2_1step_50ms/gnnmappo00110-pret-4strongest/run1/models \
#  --model_dir /home/mzi/aps-gnn/onpolicy/results/aps/gnnmappo/mrt_allsinr_los_se_hexaps/ap20_ue6_se1_lastg_8env/ped2_1step_500ms/gnnmappo00110-eval/run2/models \
# --model_dir /home/mzi/aps-gnn/saved_results/mrt_allsinr_los_se_1env.ap20_ue6_se1.ped2_1step_500ms/gnnmappo00010-pret-4strongest/run1/models \
# --model_dir pretrained_models/los_ap20_ue6_ped/4strongest/run1/models \ 
# --model_dir /home/mzi/aps-gnn/saved_results/mrt_allsinr_los_se.ap20_ue6_se1.ped2_1step_500ms/gnnmappo00110-pret-4strongest/run1/models    \
# --model_dir /home/mzi/aps-gnn/saved_results/mrt_allsinr_los_se.ap20_ue6_se1.veh2_1step_500ms/gnnmappo00110-pret-4strongest/run1/models    \
