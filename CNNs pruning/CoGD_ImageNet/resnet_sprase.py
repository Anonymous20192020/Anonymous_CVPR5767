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
import numpy as np
import utils.globalvar as gl

def KSE(weight_kse, G=None, T=None):

    weight = weight_kse.cuda().detach()

    cin = weight.shape[1]
    cout = weight.shape[0]

    weight = weight.transpose(1, 0).reshape(cin, cout, -1)
    weight_temp = weight.cpu()
    ks_weight = np.sum(np.linalg.norm(weight_temp, ord=1, axis=2), 1)

    ks_weight = (ks_weight - np.min(ks_weight)) / (np.max(ks_weight) - np.min(ks_weight))
    ke_weight = 0

    indicator = np.sqrt(ks_weight / (1 + ke_weight))
    indicator = (indicator - np.min(indicator))/(np.max(indicator) - np.min(indicator))

    indicator_index = np.argsort(indicator)
    
    epoch = gl.get_value('epoch')
    temp = 0.5# - epoch / 100.0 * 0.5
    #print(temp)
    threshlod = indicator[indicator_index[int(indicator_index.shape[0] * temp)]]
    return indicator, threshlod
epoch_g = 0

class Mask(nn.Module):
    def __init__(self, init_value=[1], planes=None):
        super().__init__()
        self.planes = planes
        self.weight = Parameter(torch.Tensor(init_value))
        #self.weight = torch.full_like(self.weight, 1)
        self.index = 0

    def forward(self, input, weight_kse):
        weight = self.weight[None, :, None, None]               
        #if 1 == 0: 
        if self.planes is not None:        
            iteration = gl.get_value('iteration') 
            if iteration == 1:
                #weight = self.weight[None, :, None, None]
                output = torch.full_like(weight, 1) * 0.75
                indicat, thre = KSE(weight_kse)
                for i in range(indicat.shape[0]):
                    if (indicat[i] > thre) and weight.data[:, i , :, :] < 0.5:
                        weight.data[:, i , :, :] = output.data[:, i, :, :]
        return input * weight

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:#1==1:#
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        m1 = torch.rand(int(planes))
        #m1 = torch.ones(int(planes)) * 0.5 + mm1
        self.mask1 = Mask(m1,planes=planes)

        #m2 = torch.randn(int(planes))
        #self.mask2 = Mask(m2,planes=planes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.mask1(out, weight_kse=self.conv2.weight)
        #print(self.mask1.weight)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
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

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        m1 = torch.rand(int(planes))
        # m1 = torch.ones(int(planes))
        #m1 = torch.ones(int(planes)) * 0.5 + mm1
        self.mask1 = Mask(m1,planes=planes)
        m2 = torch.rand(int(planes))
        # m1 = torch.ones(int(planes))
        #m1 = torch.ones(int(planes)) * 0.5 + mm1
        self.mask2 = Mask(m2,planes=planes)

        #m2 = torch.randn(int(planes))
        #self.mask2 = Mask(m2,planes=planes)
    def forward(self, x):
        out = self.conv1(x)

        out = self.bn1(out)
        out = F.relu(out)
        out = self.mask1(out, weight_kse=self.conv2.weight)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.mask2(out, weight_kse=self.conv3.weight)
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18_sprase():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet50_sprase():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
