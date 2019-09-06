import torch
import torch.nn as nn
import os
import math

import torchvision.models as models

from models.utils import *
from misc.utils import convert_state_dict


# ResNet: https://github.com/Jongchan/attention-module/blob/master/MODELS/model_resnet.py
def conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=1, dilation=1, bias=False):
    "kxk convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, dilation=dilation, bias=bias)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_type='bn', num_groups=32, use_cbam=False):
        super(BasicBlock, self).__init__()

        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv2d(inplanes, planes, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes) if norm_type == 'bn' else nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        self.conv2 = conv2d(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes) if norm_type == 'bn' else nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.cbam = CBAM( planes, 16 ) if use_cbam else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out = out + identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, dilation=1, norm_type='bn', num_groups=32, use_cbam=False):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv2d(inplanes, width, kernel_size=1, padding=0)
        self.bn1 = nn.BatchNorm2d(width) if norm_type == 'bn' else nn.GroupNorm(num_groups=num_groups, num_channels=width)
        self.conv2 = conv2d(width, width, stride=stride, groups=groups, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(width) if norm_type == 'bn' else nn.GroupNorm(num_groups=num_groups, num_channels=width)
        self.conv3 = conv2d(width, planes * self.expansion, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion) if norm_type == 'bn' else nn.GroupNorm(num_groups=num_groups, num_channels=planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.cbam = CBAM( planes * self.expansion, 16 ) if use_cbam else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        if self.cbam is not None:
            out = self.cbam(out)

        out = out + identity
        out = self.relu(out)

        return out


pretrained = True
model_dict = {'resnet18':  {'pretrained': models.resnet18(pretrained=pretrained),  'block': BasicBlock, 'layers': [2, 2, 2, 2],  'groups': 1, 'width_per_group': 64},
              'resnet34':  {'pretrained': models.resnet34(pretrained=pretrained),  'block': BasicBlock, 'layers': [3, 4, 6, 3],  'groups': 1, 'width_per_group': 64},
              'resnet50':  {'pretrained': models.resnet50(pretrained=pretrained),  'block': Bottleneck, 'layers': [3, 4, 6, 3],  'groups': 1, 'width_per_group': 64},
              'resnet101': {'pretrained': models.resnet101(pretrained=pretrained), 'block': Bottleneck, 'layers': [3, 4, 23, 3], 'groups': 1, 'width_per_group': 64},
              'resnet152': {'pretrained': models.resnet152(pretrained=pretrained), 'block': Bottleneck, 'layers': [3, 8, 36, 3], 'groups': 1, 'width_per_group': 64},
              'resnet38':  {'pretrained': None,                                    'block': Bottleneck, 'layers': [3, 3, 3, 3],  'groups': 1, 'width_per_group': 64},}
              #'resnext50-32x4d':  {'pretrained': models.resnext50_32x4d(pretrained=pretrained),  'block': Bottleneck, 'layers': [3, 4, 6, 3],  'groups': 32, 'width_per_group': 4},
              #'resnext101-32x8d': {'pretrained': models.resnext101_32x8d(pretrained=pretrained), 'block': Bottleneck, 'layers': [3, 4, 23, 3], 'groups': 32, 'width_per_group': 8},}

class resnet(nn.Module):
    def __init__(self, name='resnet18', n_classes=1000, in_channels=3, inplanes=64, zero_init_residual=True, load_pretrained=True, norm_type='bn', num_groups=32, use_cbam=False):
        super(resnet, self).__init__()

        self.name = name
        self.norm_type = norm_type
        self.n_classes = n_classes
        self.use_cbam = use_cbam

        self.block = model_dict[self.name]['block']
        self.layers = model_dict[self.name]['layers']

        self.inplanes = inplanes
        self.dilation = 1
        self.groups = model_dict[self.name]['groups']
        self.base_width = model_dict[self.name]['width_per_group']
        self.conv1 = nn.Conv2d(in_channels, self.inplanes, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes) if norm_type == 'bn' else nn.GroupNorm(num_groups=num_groups, num_channels=self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(self.block, inplanes,   self.layers[0], norm_type=norm_type, num_groups=num_groups, use_cbam=use_cbam)
        self.layer2 = self._make_layer(self.block, inplanes*2, self.layers[1], stride=2, norm_type=norm_type, num_groups=num_groups, use_cbam=use_cbam)
        self.layer3 = self._make_layer(self.block, inplanes*4, self.layers[2], stride=2, norm_type=norm_type, num_groups=num_groups, use_cbam=use_cbam)
        self.layer4 = self._make_layer(self.block, inplanes*8, self.layers[3], stride=2, norm_type=norm_type, num_groups=num_groups, use_cbam=use_cbam)

        self.fc = nn.Linear(512*self.block.expansion, self.n_classes, bias=True)

        self.dropout = nn.Dropout(p=0.5)

        self._init_weights(zero_init_residual=zero_init_residual, load_pretrained=load_pretrained)

    def _init_weights(self, zero_init_residual, load_pretrained):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                if 'cbam.SpatialGate.spatial' in name:
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                ##nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        if load_pretrained:
            if self.norm_type == 'bn':
                pretrained_model = model_dict[self.name]['pretrained']
                pretrained_model_state_dict = pretrained_model.state_dict()
            else:
                pretrained_model = torch.load(os.path.join('pretrained', self.name + '-GN.pth'))#, encoding="latin1")
                pretrained_model_state_dict = convert_state_dict(pretrained_model['model_state'])
            state_dict = self.state_dict()
            for k in state_dict.keys():
                if 'cbam' not in k and 'fc' not in k:
                    state_dict[k] = pretrained_model_state_dict[k]
            self.load_state_dict(state_dict)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, norm_type='bn', num_groups=32, use_cbam=False):
        downsample = None
        norm_layer = nn.BatchNorm2d(planes * block.expansion) if norm_type == 'bn' else nn.GroupNorm(num_groups=num_groups, num_channels=planes * block.expansion)
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                norm_layer,
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, groups=self.groups, base_width=self.base_width, dilation=previous_dilation, norm_type=norm_type, num_groups=num_groups, use_cbam=use_cbam))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups, base_width=self.base_width, dilation=self.dilation, norm_type=norm_type, num_groups=num_groups, use_cbam=use_cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, output_size=(1, 1)) # Global Average Pooling
        x = x.view(x.size(0), -1) # flatten
        x = self.dropout(x)
        x = self.fc(x)
        return x
