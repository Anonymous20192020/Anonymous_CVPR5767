# Tailored Pruning via Rollback Learning(RLP)


## Running Code

In this code, you can run our model on ImageNet dataset. The code has been tested by Python 3.6, [Pytorch 0.4.1](https://pytorch.org/) and CUDA 9.0 on Ubuntu 16.04.



### Run examples
First, you need install ptflops to calculate flops
```shell
cd flops-counter.pytorch
sudo python3 setup.py install
```

### Resnet56
We provide our pre-trained, pruned and fine-tuned models below.

| Model                                                        | FLOPs.   | Top-1 Acc/+FT. |
| ------------------------------------------------------------ | ------- | ---------- |
| [Baseline](https://download.pytorch.org/models/resnet50-19c8e357.pth) | 4.09B |  75.24%      |
| [RLP-0.35](https://drive.google.com/drive/folders/1fFzuL5xvK5Hk-njc2F-4JrRn_qfmdqSs) | 1.46B |  70.27/72.07%      | 
| [RLP-0.50](https://drive.google.com/drive/folders/1uqeFgj5fI0yPLArLDxu2_ei_8inCM1dV) | 2.24B |  73.20/74.38%      | 
| [RLP-0.75](https://drive.google.com/drive/folders/1qh77frxEeSwScZlBCYrIXi7cFSZTyTyi) | 2.53B |  73.64/74.89%      | 
| [RLP-0.0.95](https://drive.google.com/drive/folders/1qAGxjC_A0r0bw-Zgjkk0-IAn7CNY5mPI) | 3.31B |  74.74/75.78%      | 


Then you can test our trained model:
```shell
cd RLP_ImageNet
CUDA_VISIBLE_DEVICES=0 python3 finetune_resnet.py --refine model.pt --pruned_dir model_pruned.pt
```
