#!/bin/sh
env="aps"
algo="gnnmappo"
exp="pretraining/los_hexap20_ue6_ped_history4/2strongest"
seed=1
python pret_gnnmappo.py --use_valuenorm --env_name ${env} --algorithm_name ${algo} \
 --experiment_name ${exp} --seed ${seed} --n_training_threads 2 --n_rollout_threads 2 \
 --use_recurrent_policy False --if_supervised_learning True --if_rnn_gnn True \
 --pickled_data_dir /home/mzi/aps-gnn/onpolicy/scripts/pret_dataset/2strongest_los_20hexaps_6pedues_4history.pickle
