import torch
from torch import nn as nn
from typing import Type, List, Union, Optional



def con3x3(in_ : int, out_ : int, strides : int = 1, initialize : bool = False):
    bn = nn.BatchNorm2d(out_)
    if initialize == True:
        nn.init.constant_(bn.weight, 0)
    x = nn.Sequential(nn.Conv2d(in_, out_, 3, strides, padding=1), bn)
    return x

def con1x1(in_ : int, out_ : int, strides : int = 1, initialize : bool = False):
    bn = nn.BatchNorm2d(out_)
    if initialize == True:
        nn.init.constant_(bn.weight, 0)
    x = nn.Sequential(nn.Conv2d(in_, out_, 1, strides), bn)
    return x

# 残差单元
class residualunit(nn.Module):
    def __init__(self, out_ : int, strides : int = 1, in_ : Optional[int] = None):
        super().__init__()
        self.stride = strides
        if in_ == None:
            if strides != 1:
                in_ = int(out_/2)
            else:
                in_ = out_
        self.fit_ = nn.Sequential(con3x3(in_, out_, strides)
                                 ,nn.ReLU(inplace=True)
                                 ,con3x3(out_, out_, initialize=True))
        self.skipconv = con1x1(in_, out_, strides)
        self.relu = nn.ReLU(inplace=True)
        
        
    def forward(self, x):
        fit_ = self.fit_(x)
        if self.stride != 1:
            x = self.skipconv(x)
        return self.relu(fit_ + x)
        
# 瓶颈块
class bottleneck(nn.Module):
    def __init__(self, middle_out : int, strides : int =1, in_ : Optional[int] = None):
        super().__init__()
        self.stride = strides
        out_ = middle_out*4 
        if in_ == None:
            if strides != 1:
                in_ = middle_out*2
            else:
                in_ = middle_out*4
        self.fit_ = nn.Sequential(con1x1(in_, middle_out, strides)
                                 ,nn.ReLU(inplace=True)
                                 ,con3x3(middle_out, middle_out)
                                 ,nn.ReLU(inplace=True)
                                 ,con1x1(middle_out, out_, initialize=True))
        self.skipconv = con1x1(in_, out_, strides)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        fit_ = self.fit_(x)
        x = self.skipconv(x)
        return self.relu(fit_ + x)

# 生成 Resnet Layer
def make_layers(block: Type[Union[residualunit, bottleneck]]
               ,middle_out : [int]
               ,blocks_num : [int]
               ,forconv1 : [bool] = False):
    layers = []
    if forconv1 == True:
        layers.append(block(middle_out, in_ = 64))
    else:
        layers.append(block(middle_out, 2))
        
    for i in range(blocks_num - 1):
        layers.append(block(middle_out, 1))
    return nn.Sequential(*layers)

class Resnet(nn.Module):
    def __init__(self, block: Type[Union[residualunit, bottleneck]], layers : List[int]):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3)
                                 ,nn.BatchNorm2d(64)
                                 ,nn.ReLU(inplace=True)
                                 ,nn.MaxPool2d(3, 2, 1))
        self.layer1 = make_layers(block, 64, layers[0], forconv1=True)
        self.layer2 = make_layers(block, 128, layers[1])
        self.layer3 = make_layers(block, 256, layers[2])
        self.layer4 = make_layers(block, 512, layers[3])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        if block == residualunit:
            self.softmax_ = nn.Linear(512, 45, device='cuda')
        else:
            self.softmax_ = nn.Linear(2048, 45, device='cuda')
        
    def forward(self, x):
        x = self.conv(x)
        x = self.layer2(self.layer1(x))
        x = self.layer4(self.layer3(x))
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        output_ = self.softmax_(features)
        return features, output_