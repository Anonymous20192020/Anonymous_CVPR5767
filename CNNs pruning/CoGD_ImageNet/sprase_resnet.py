'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.distributions.normal import Normal
#from sklearn.neighbors import NearestNeighbors
#from sklearn.cluster import KMeans
import math
import numpy as np
import time
# import knn_pytorch



def KSE(weight_kse, G=None, T=None):
    # weight = weight_kse.data.cpu().numpy()
    weight = weight_kse.cuda().detach()
    cin = weight.shape[1]
    cout = weight.shape[0]
    weight = weight.transpose(1, 0).reshape(cin, cout, -1)
    ks_weight = np.sum(np.linalg.norm(weight, ord=1, axis=2), 1)
    ks_weight = (ks_weight - np.min(ks_weight)) / (np.max(ks_weight) - np.min(ks_weight))
    indicator = np.sqrt(ks_weight)
    indicator = (indicator - np.min(indicator))/(np.max(indicator) - np.min(indicator))
    # print(indicator,'indicator')
    indicator_index = np.argsort(indicator)
    # print(indicator_index,'indicator_index')
    threshlod = indicator[indicator_index[int(indicator_index.shape[0] * 0.2)]]
    # for i in range(indicator.shape[0]):
    #     if (indicator[i]>= threshlod):
    print(threshlod,'!!!!!!!!!!!!!!!!!')
    return indicator, threshlod


epoch_g = 0
class Mask(nn.Module):
    def __init__(self, init_value=[1], planes=None):
        super().__init__()
        self.planes = planes
        self.weight = Parameter(torch.Tensor(init_value))
        self.index = 0
    def forward(self, input, weight_kse):
        weight = self.weight     ############mask
        if self.planes is not None:
            weight = self.weight[None, :, None, None]
            output = torch.full_like(weight, 1)
            global epoch_g
            epoch_g = epoch_g + 1

            # print(epoch_g, 'eeeeeeeeeeeeeee')

            if (epoch_g > 0) and (epoch_g < 17):##6630
                # print(epoch_g, 'aaaaaaaaaaaaaaa')
                # if (epoch_g == 17):##6647   7056  7856
                #     epoch_g = 0
                indicat, thre = KSE(weight_kse)
                for i in range(indicat.shape[0]):
                    if (indicat[i] > thre) :
                        # print(indicat[i],'indicat')
                        # self.output[:, i, :, :] = input[:, i, :, :] * weight[:, self.index, :, :]
                        # print(output[:, i, :, :].data,'##############')
                        weight.data[:, i , :, :] = output.data[:, i, :, :]
                        # self.index = self.index + 1
            if (epoch_g == 7856):  ##6647
                epoch_g = 0
        return input * weight


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        m1 = torch.randn(int(in_planes))
        self.mask1 = Mask(m1,planes=planes)
        m2 = torch.randn(int(planes))
        self.mask2 = Mask(m2,planes=planes)

    def forward(self, x):

        out = self.mask1(x, weight_kse=self.conv1.weight)
        out = self.conv1(out)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.mask2(out, weight_kse=self.conv2.weight)
        out = self.conv2(out)
        out = self.bn2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        m1 = torch.randn(int(planes))
        self.mask1 = Mask(m1,planes=planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mask1(out, weight_kse=self.conv1.weight)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv3(out)
        out = self.bn3()
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class SpraseResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(SpraseResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        # print(out.shape, 'out.shape')

        return out


def SpraseResNet18():
    return SpraseResNet(BasicBlock, [2,2,2,2])

def SpraseResNet34():
    return SpraseResNet(BasicBlock, [3,4,6,3])

def SpraseResNet50():
    return SpraseResNet(Bottleneck, [3,4,6,3])

def SpraseResNet101():
    return SpraseResNet(Bottleneck, [3,4,23,3])

def SpraseResNet152():
    return SpraseResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
