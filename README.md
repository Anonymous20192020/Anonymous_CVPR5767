# Cogradient Descent in Bilinear Model

## CSC

### Environment

The code has been tested by Matlab R2014b. This code is based on [FFCSC](https://github.com/baopingli/FFCSC2015).

### Reconstruction

```shell
learn_kernels_2D.m
```

The results after 20 epochs.

| Model | Fruit_PSNR (dB) | Fruit_SSIM | City_PSNR (dB) | City_SSIM |
| ----- | ------- | ------- | ------- | ------- |
| FFCSC | 26.70 | 0.9602 | 26.98 | 0.9618 | 
| CoGD  | 27.60 | 0.9635 | 28.10 | 0.9642 | 

### Inpainting

```shell
learn_kernels_2D_sparse.m
```

The results after 20 epochs.

| Model | Fruit_PSNR (dB) | Fruit_SSIM | City_PSNR (dB) | City_SSIM |
| ----- | ------- | ------- | ------- | ------- |
| FFCSC | 23.65 | 0.9000 | 24.31 | 0.9199 |
| CoGD  | 25.31 | 0.9211 | 25.58 | 0.9314 |


## CNNs pruning

### Environment

In this code, you can run our model on ImageNet dataset. The code has been tested by Python 3.6, [Pytorch 0.4.1](https://pytorch.org/) and CUDA 9.0 on Ubuntu 16.04.

### Run examples
First, you need install ptflops to calculate flops
```shell
cd flops-counter.pytorch
sudo python3 setup.py install
```

#### Resnet50
We provide our pre-trained, pruned and fine-tuned models below.

| Model                                                        | FLOPs.   | Top-1 Acc/+FT. |
| ------------------------------------------------------------ | ------- | ---------- |
| [Baseline](https://download.pytorch.org/models/resnet50-19c8e357.pth) | 4.09B |  75.24%      |
| [CoGD-0.35](https://drive.google.com/drive/folders/1fFzuL5xvK5Hk-njc2F-4JrRn_qfmdqSs) | 1.46B |  70.27/72.07%      | 
| [CoGD-0.50](https://drive.google.com/drive/folders/1uqeFgj5fI0yPLArLDxu2_ei_8inCM1dV) | 2.24B |  73.20/74.38%      | 
| [CoGD-0.75](https://drive.google.com/drive/folders/1qh77frxEeSwScZlBCYrIXi7cFSZTyTyi) | 2.53B |  73.64/74.89%      | 
| [CoGD-0.0.95](https://drive.google.com/drive/folders/1qAGxjC_A0r0bw-Zgjkk0-IAn7CNY5mPI) | 3.31B |  74.74/75.78%      | 


Then you can test our trained model:
```shell
cd RLP_ImageNet
CUDA_VISIBLE_DEVICES=0 python3 finetune_resnet.py --refine model.pt --pruned_dir model_pruned.pt
```









