#!/bin/bash

# Set the GPU ID to use
# Replace '0' with the ID of the GPU you want to use
export CUDA_VISIBLE_DEVICES=0

python -u main.py --model_path models/Hendrycks2020AugMixWRN_c10.pt \
                  --data_path /home/thilina/SSD2/datasets/cifar/ \
                  --source_dataset CIFAR-10 \
                  --target_dataset CIFAR-10-C \
                  --lr 0.005 \
                  --tta_batchsize 128 \
                  --severity 5 \
                  --criterion entropy \
                  --network wrn-40x2 \
                  --drop_classes 0 2 5
