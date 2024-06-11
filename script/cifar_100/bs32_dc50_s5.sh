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

python -u main.py --model_path models/Hendrycks2020AugMixWRN_c100.pt \
                  --data_path /opt/home/s4079488/datasets/cifar/ \
                  --source_dataset CIFAR-100 \
                  --target_dataset CIFAR-100-C \
                  --lr 0.005 \
                  --tta_batchsize 32 \
                  --severity 5 \
                  --criterion entropy \
                  --network wrn-40x2 \
                  --drop_classes  72 4 95 30 55 \
                                73 32 67 91 1 \
                                14 24 6 7 18 \
                                43 97 42 3 88 \
                                15 21 19 31 38 \
                                75 63 66 64 34 \
                                77 26 45 99 79 \
                                11 2 35 46 98 \
                                29 93 27 78 44 \
                                65 50 74 36 80 \
                  --eval_before 1