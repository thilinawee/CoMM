#!/bin/bash


#SBATCH --job-name=partial_tta
#SBATCH --output=tta_out.txt
#SBATCH --error=tta_error.err
#SBATCH --partition=SCT

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G


source_dataset="CIFAR-10"
target_dataset="CIFAR-10-C"
data_path="/opt/home/s4079488/datasets/cifar/"
model_path="models/Hendrycks2020AugMixWRN_c10.pt"
lr=0.005
tta_batchsize=128
severity=5
criterion="entropy"
network="wrn-40x2"
eval_before=1
drop_classes="0 8"
seed=(123 456)

for seed in ${seed[@]}; do
    python -u main.py --model_path $model_path \
                      --data_path $data_path \
                      --source_dataset $source_dataset \
                      --target_dataset $target_dataset \
                      --lr $lr \
                      --tta_batchsize $tta_batchsize \
                      --severity $severity \
                      --criterion $criterion \
                      --network $network \
                      --eval_before $eval_before \
                      --drop_classes $drop_classes \
                      --seed $seed
done


