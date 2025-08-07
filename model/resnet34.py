import warnings

import numpy as np
import torch.nn as nn
import torch
from torchvision import models

class resnet340(nn.Module):
    def __init__(self):
        super().__init__()
        model = models.resnet34()
        modules = list(model.children())
        modules = modules[:5]
        modules[0] = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        del modules[3]
        self.model = nn.Sequential(*modules)
    def forward(self, x):
        return self.model(x)

class resnet341(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = models.resnet34()
        modules = list(model.children())
        modules = modules[5:6]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class resnet342(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = models.resnet34()
        modules = list(model.children())
        modules = modules[6:7]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class resnet343(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = models.resnet34()
        modules = list(model.children())
        modules = modules[7:8]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)


class Res34backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.nc = 1
        self.res0 =resnet340()
        self.res1 = resnet341()
        self.res2 = resnet342()
        self.res3 = resnet343()
    
    # self.sppf=SPPF()
    def forward(self, x):
        x1 = self.res0(x)  ##1/2
        x2 = self.res1(x1)  ##1/4
        x3 = self.res2(x2)  ##1/8
        x4 = self.res3(x3)  ##1/16
        return [[x],[x1], [x2], [x3], [x4]]
if __name__ == '__main__':
    # 10分类
    res50 = Res34backbone().to('cuda:0')
    # summary(res50, (3, 224, 224))
    x=torch.ones(3,224,224).unsqueeze(0).to('cuda')
    y=res50(x)
