import sys
sys.path.append("..")

import re
import numpy as np
from collections import OrderedDict
from model import resnet_56, resnet_56_sparse
from sprase_mask_mobilenetv2 import SpraseMobileNetV2
from sprase_mask_resnet_kse import SpraseResNet,BasicBlock
from sprase_resnet_kse import SpraseResNet18
import torch

def prune_resnet(args, state_dict):
    thre = args.thre
    num_layers = int(args.student_model.split('_')[1])
    n = (num_layers - 2) // 6
    layers = np.arange(0, 3*n ,n)
 
    mask_block = []
    #for name, weight in state_dict.items():
        #if 'mask' in name:
            #mask_block.append(weight.item())

    #pruned_num = sum(m <= thre for m in mask_block)
    #pruned_blocks = [int(m) for m in np.argwhere(np.array(mask_block) <= thre)]

    old_block = 0
    layer = 'layer1'
    layer_num = int(layer[-1])
    new_block = 0
    new_state_dict = OrderedDict()
    
    mask_sum1 = []
    mask_sum2 = []
    for key, value in state_dict.items():
        #print(key)
        #print(value.shape) 
        #print(key)
        if "layer" in key:# and "bn1" in key:
            count_1 = key.split('.')[0][-1:]
            count_2 = key.split('.')[1]
            #print(count_1,count_2,'**************')
            temp1 = 'layer' + count_1 + '.' + count_2 + '.mask1.weight'
            temp2 = 'layer' + count_1 + '.' + count_2 + '.mask2.weight'
            #print(temp1 in state_dict)
            mask1 = state_dict.get(temp1, '')#state_dict['layer' + count_1 + '.' + count_2 + '.mask1.weight']
            mask2 = state_dict.get(temp2, '')#state_dict['layer' + count_1 + '.' + count_2 + '.mask2.weight']
            p_num1 = 0
            p_num2 = 0
            mask_num1 = len(mask1)
            mask_num2 = len(mask2)
            for i in (mask1):
                if i != 0:
                    p_num1 = p_num1 + 1
            for i in (mask2):
                if i != 0:
                    p_num2 = p_num2 + 1

            if "conv1" in key:
                conv1 = 'layer' + count_1 + '.' + count_2 + '.conv1.weight'
                old_conv1 = state_dict[conv1]
                if mask_num1 == 0:
                    new_state_dict[conv1] = old_conv1 
                    continue                   

                new_conv1 = torch.rand(p_num1,old_conv1.shape[1],3,3)
                j = 0
                for i in range(mask_num1):
                    if mask1[i] != 0:
                        new_conv1[j,:,:,:] = old_conv1[i,:,:,:]
                        j = j + 1
                new_state_dict[conv1] = new_conv1
                mask_sum1.append(p_num1)
            elif "conv2" in key:
                conv2 = 'layer' + count_1 + '.' + count_2 + '.conv2.weight'
                old_conv2 = state_dict[conv2]
                if mask_num1 == 0:
                    new_state_dict[conv2] = old_conv2 
                    continue    
                new_conv2 = torch.rand(old_conv2.shape[0],p_num1,3,3)
                j = 0
                for i in range(mask_num1):
                    if mask1[i] != 0:
                        new_conv2[:,j,:,:] = old_conv2[:,i,:,:]
                        j = j + 1
                new_state_dict[conv2] = new_conv2
                #mask_sum2.append(p_num2)
            #elif "mask1" in key:
                #continue
            #    print(value)
            elif "bn1.weight" in key:
                bn1w = 'layer' + count_1 + '.' + count_2 + '.bn1.weight' 
                old_bn1w = state_dict[bn1w]
                new_bn1w = torch.rand(p_num1)
                j = 0
                for i in range(mask_num1):
                    if mask1[i] != 0:
                        new_bn1w[j] = old_bn1w[i]
                        j = j + 1 
                new_state_dict[bn1w] = new_bn1w   
            elif "bn1.bias" in key:
                bn1b = 'layer' + count_1 + '.' + count_2 + '.bn1.bias' 
                old_bn1b = state_dict[bn1b]
                new_bn1b = torch.rand(p_num1)
                j = 0
                for i in range(mask_num1):
                    if mask1[i] != 0:
                        new_bn1b[j] = old_bn1b[i]
                        j = j + 1 
                new_state_dict[bn1b] = new_bn1b   
            elif "bn1.running_mean" in key:
                bn1rm = 'layer' + count_1 + '.' + count_2 + '.bn1.running_mean' 
                old_bn1rm = state_dict[bn1rm]
                new_bn1rm = torch.rand(p_num1)
                j = 0
                for i in range(mask_num1):
                    if mask1[i] != 0:
                        new_bn1rm[j] = old_bn1rm[i]
                        j = j + 1 
                new_state_dict[bn1rm] = new_bn1rm  
            elif "bn1.running_var" in key:
                bn1rv = 'layer' + count_1 + '.' + count_2 + '.bn1.running_var' 
                old_bn1rv = state_dict[bn1rv]
                new_bn1rv = torch.rand(p_num1)
                j = 0
                for i in range(mask_num1):
                    if mask1[i] != 0:
                        new_bn1rv[j] = old_bn1rv[i]
                        j = j + 1 
                new_state_dict[bn1rv] = new_bn1rv  

            else:
                new_state_dict[key] = state_dict[key]

        else:
            new_state_dict[key] = state_dict[key]
    save_state_dict={}
    save_state_dict['state_dict_s'] = new_state_dict

    torch.save(save_state_dict,'./mobilenet_pruned.pt')
    
    print(mask_sum1,'!!!!!!!!!!!!')
    #print(mask_sum2,'@@@@@@@@@@@@@@')
    #for key, value in new_state_dict.items():
        #print(key)
        #print(value.shape)
    # model = SpraseMobileNetV2(has_masks1 = mask_sum1, has_masks2= mask_num2)#.to(args.gpus[0])
    mask_sum1 = None if len(mask_sum1) == 0 else mask_sum1
    mask_sum2 = None if len(mask_sum2) == 0 else mask_sum2
    model = SpraseResNet(BasicBlock, [2, 2, 2, 2], has_mask1 = mask_sum1, has_mask2 = mask_sum2).to(args.gpus[0])

    #for key, value in new_state_dict.items():
        #print(key)
        #print(value.shape)
    print(model)
    # model = torch.nn.DataParallel(model,[0])
    checkpoint = torch.load('./mobilenet_pruned.pt')
    new = checkpoint['state_dict_s']
    # for key, value in new.items():
    #     print(key)
    #     print(value.shape)
    #print(model.state_dict()['module.layers.0.conv1.weight'].shape)
    model.load_state_dict(checkpoint['state_dict_s'])
    #print(model.layer1.0.mask1.weight)

    # model = SpraseResNet18().to(args.gpus[0])
    # checkpoint = torch.load('./model_1.pt')
    # model.load_state_dict(checkpoint['state_dict_s'])

    return model

