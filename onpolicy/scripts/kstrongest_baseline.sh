#!/bin/sh
env="aps"
algo="kstrongest"
seed=1
values="2"
for k in $values; do
    exp="mrt_allsinr_los_se_hexaps_new/ap20_ue6_8env/ped2_1step_50ms/episode32/${k}strongest"
    python baseline.py --env_name ${env} --algorithm_name ${algo} \
    --n_rollout_threads 8 --seed ${seed} \
    --episode_length 32 --num_env_steps 512000 \
    --experiment_name ${exp} --K ${k} --largest True \
    --log_interval 1
done
