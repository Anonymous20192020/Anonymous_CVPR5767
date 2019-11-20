import sys
sys.path.append("..")

import re
import numpy as np
from collections import OrderedDict
from model import resnet_56, resnet_56_sparse
from resnet_sprase import ResNet50_sprase
from sprase_mask_mobilenetv2 import SpraseMobileNetV2
from sprase_mask_resnet_after import SpraseResNet,BasicBlock,Bottleneck
from sprase_resnet_kse import SpraseResNet18
import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

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
    
    p_num11 = 64
    mask11 = torch.ones(p_num11)
    mask_num11 = len(mask11)


    ba_flag = 1

    for key, value in state_dict.items():

        #break
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

            mt1 = torch.nonzero(mask1)
            mt1 = mt1.reshape(-1)
            mt1_1 = (mask1 == 0).nonzero()
            mt1_1 = mt1_1.reshape(-1)
            #mt1 = torch.index_select(mask1, 0, m)

            mt2 = torch.nonzero(mask2)
            mt2 = mt2.reshape(-1)
            mt2_1 = (mask2 == 0).nonzero()
            mt2_1 = mt2_1.reshape(-1)
            #mt2 = torch.index_select(mask2, 0, m)

            p_num1 = 0
            p_num2 = 0
            mask_num1 = len(mask1)
            mask_num2 = len(mask2)
            #print(mask_num1)
            #print(mask_num2)
            for i in (mask1):
                if i != 0:
                    p_num1 = p_num1 + 1
            for i in (mask2):
                if i != 0:
                    p_num2 = p_num2 + 1

################conv1#################
            if "conv1" in key:
                conv1 = 'layer' + count_1 + '.' + count_2 + '.conv1.weight'
                old_conv1 = state_dict[conv1]

                if mask_num1 == 0:
                    new_conv1 = old_conv 
                    p_num1 = old_conv1.shape[1]
                else:
                    new_conv1 = torch.rand(p_num1,old_conv1.shape[1],1,1)
                    j = 0
                    for i in range(mask_num1):
                        if mask1[i] != 0:
                            new_conv1[j,:,:,:] = old_conv1[i,:,:,:]
                            j = j + 1

                new_state_dict[conv1] = new_conv1 
                mask_sum1.append(p_num1)
                #print(new_conv1.shape)
                

################conv2#################
            elif "conv2" in key:
                conv2 = 'layer' + count_1 + '.' + count_2 + '.conv2.weight'
                old_conv2 = state_dict[conv2]

                if mask_num1 == 0:
                    new_conv2 = old_conv2 
                    p_num1 = old_conv2.shape[1]
                else:
                    new_conv2 = torch.rand(old_conv2.shape[0],p_num1,3,3)
                    j = 0
                    for i in range(mask_num1):
                        if mask1[i] != 0:
                            new_conv2[:,j,:,:] = old_conv2[:,i,:,:]
                            j = j + 1

                new_conv22 = torch.rand(p_num2,p_num1,3,3)
                j = 0
                for i in range(mask_num2):
                    if mask2[i] != 0:
                        new_conv22[j,:,:,:] = new_conv2[i,:,:,:]
                        j = j + 1

                new_state_dict[conv2] = new_conv22 
                mask_sum2.append(p_num2)

################conv2#################
            elif "conv3" in key:
                conv3 = 'layer' + count_1 + '.' + count_2 + '.conv3.weight'
                old_conv3 = state_dict[conv3]

                if mask_num2 == 0:
                    new_conv3 = old_conv3 
                    p_num1 = old_conv3.shape[1]
                else:
                    new_conv3 = torch.rand(old_conv3.shape[0],p_num2,1,1)
                    j = 0
                    for i in range(mask_num2):
                        if mask2[i] != 0:
                            new_conv3[:,j,:,:] = old_conv3[:,i,:,:]
                            j = j + 1
                new_state_dict[conv3] = new_conv3 
                #mask_sum2.append(p_num2)   

            #elif "mask" in key:
            	#continue

################bn1#################
            elif "bn1.weight" in key and ba_flag != 0:
                bn1w = 'layer' + count_1 + '.' + count_2 + '.bn1.weight' 
                old_bn1w = state_dict[bn1w]
                #new_bn1w = torch.rand(p_num1)

                bn1w1 = torch.index_select(old_bn1w,0,mt1)
                bn1w2 = torch.index_select(old_bn1w,0,mt1_1)

                new_bn1w = torch.cat((bn1w1,bn1w2),0)

                new_state_dict[bn1w] = new_bn1w   
            elif "bn1.bias" in key and ba_flag != 0:
                bn1b = 'layer' + count_1 + '.' + count_2 + '.bn1.bias' 
                old_bn1b = state_dict[bn1b]

                bn1w1 = torch.index_select(old_bn1b,0,mt1)
                bn1w2 = torch.index_select(old_bn1b,0,mt1_1)

                new_bn1b = torch.cat((bn1w1,bn1w2),0)

                new_state_dict[bn1b] = new_bn1b   
            elif "bn1.running_mean" in key and ba_flag != 0:
                bn1rm = 'layer' + count_1 + '.' + count_2 + '.bn1.running_mean' 
                old_bn1rm = state_dict[bn1rm]

                bn1w1 = torch.index_select(old_bn1rm,0,mt1)
                bn1w2 = torch.index_select(old_bn1rm,0,mt1_1)

                new_bn1rm = torch.cat((bn1w1,bn1w2),0)

                new_state_dict[bn1rm] = new_bn1rm  
            elif "bn1.running_var" in key and ba_flag != 0:
                bn1rv = 'layer' + count_1 + '.' + count_2 + '.bn1.running_var' 
                old_bn1rv = state_dict[bn1rv]

                bn1w1 = torch.index_select(old_bn1rv,0,mt1)
                bn1w2 = torch.index_select(old_bn1rv,0,mt1_1)

                new_bn1rv = torch.cat((bn1w1,bn1w2),0)

                new_state_dict[bn1rv] = new_bn1rv  

################bn2#################
            elif "bn2.weight" in key and ba_flag != 0:
                bn2w = 'layer' + count_1 + '.' + count_2 + '.bn2.weight' 
                old_bn2w = state_dict[bn2w]

                bn1w1 = torch.index_select(old_bn2w,0,mt2)
                bn1w2 = torch.index_select(old_bn2w,0,mt2_1)

                new_bn2w = torch.cat((bn1w1,bn1w2),0)

                new_state_dict[bn2w] = new_bn2w   
            elif "bn2.bias" in key and ba_flag != 0:
                bn2b = 'layer' + count_1 + '.' + count_2 + '.bn2.bias' 
                old_bn2b = state_dict[bn2b]

                bn1w1 = torch.index_select(old_bn2b,0,mt2)
                bn1w2 = torch.index_select(old_bn2b,0,mt2_1)

                new_bn2b = torch.cat((bn1w1,bn1w2),0)

                new_state_dict[bn2b] = new_bn2b   
            elif "bn2.running_mean" in key and ba_flag != 0:
                bn2rm = 'layer' + count_1 + '.' + count_2 + '.bn2.running_mean' 
                old_bn2rm = state_dict[bn2rm]
                bn1w1 = torch.index_select(old_bn2rm,0,mt2)
                bn1w2 = torch.index_select(old_bn2rm,0,mt2_1)

                new_bn2rm = torch.cat((bn1w1,bn1w2),0)
                new_state_dict[bn2rm] = new_bn2rm  
            elif "bn2.running_var" in key and ba_flag != 0:
                bn2rv = 'layer' + count_1 + '.' + count_2 + '.bn2.running_var' 
                old_bn2rv = state_dict[bn2rv]
                bn1w1 = torch.index_select(old_bn2rv,0,mt2)
                bn1w2 = torch.index_select(old_bn2rv,0,mt2_1)

                new_bn2rv = torch.cat((bn1w1,bn1w2),0)
                new_state_dict[bn2rv] = new_bn2rv
            else:
                print(key)
                new_state_dict[key] = state_dict[key]
        else:
            #if "linear.bias" in key:
                #continue
            print(key)
            new_state_dict[key] = state_dict[key]


    save_state_dict={}
    save_state_dict['state_dict_s'] = new_state_dict

    torch.save(save_state_dict,'./pruned.pt')
    
    # for key, value in new_state_dict.items():
    #     if 'conv' in key:
    #         print(key)
    #         print(value.shape)

    mask_sum1 = None if len(mask_sum1) == 0 else mask_sum1
    mask_sum2 = None if len(mask_sum2) == 0 else mask_sum2
    # print(mask_sum1, mask_sum2,'#################')
    model = SpraseResNet(Bottleneck, [3,4,6,3], has_mask1 = mask_sum1, has_mask2 = mask_sum2).to(args.gpus[0])
#ResNet50_sprase().to(args.gpus[0])#

    #for key, value in model.named_parameters():#new_state_dict.items():
        #print(key)
        #print(value.shape)
    #print(model)
    # model = torch.nn.DataParallel(model,[0])
    checkpoint = torch.load('./pruned.pt')
    new = checkpoint['state_dict_s']
    model.load_state_dict(new)

    return model

