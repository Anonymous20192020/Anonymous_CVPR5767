#!/usr/bin/env sh
set -e

cd /flops-counter.pytorch
sudo python3 setup.py install
pip3 install --extra-index-url https://developer.download.nvidia.com/compute/redist/cuda/9.0 nvidia-dali
pip3 install torch==1.1.0
pip3 install tensorboardX


CUDA_VISIBLE_DEVICES=0 python3 finetune_resnet.py --refine /mnt/cephfs_new_wj/cv/xiaxin/GAL_ImageNet/GAL_KD3/experiment_cp_sign/checkpoint/model_45.pt

