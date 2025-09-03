#!/bin/sh
env="aps"
algo="gnnmappo"
exp="pretraining/los_ap20_ue6_ped/2strongest"
seed=1
python pret_gnnmappo.py --use_valuenorm --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 2 --n_rollout_threads 2 \
 --use_recurrent_policy False --if_supervised_learning True \
 --pickled_data_dir /home/mzi/aps-gnn/onpolicy/scripts/pret_dataset/2strongest_los_20aps_6ues_ped.pickle
