#!/bin/bash


#SBATCH --job-name=partial_tta
#SBATCH --output=tta_out.txt
#SBATCH --error=tta_error.err
#SBATCH --partition=SCT

#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G

# Set the GPU ID to use
# Replace '0' with the ID of the GPU you want to use
export CUDA_VISIBLE_DEVICES=0

python -u main.py --model_path models/Hendrycks2020AugMixWRN_c10.pt \
                  --data_path /opt/home/s4079488/datasets/cifar/ \
                  --source_dataset CIFAR-10 \
                  --target_dataset CIFAR-10-C \
                  --lr 0.005 \
                  --tta_batchsize 128 \
                  --severity 5 \
                  --criterion entropy \
                  --network wrn-40x2 \
                  --eval_before 1 \
                  --drop_classes 0 8 1 9 \