import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=1, groups=1, norm_type='bn', momentum=0.1, num_groups=32):
        super(ConvBatchNorm, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                             padding=padding, stride=stride, bias=bias, dilation=dilation, groups=groups)

        if norm_type == 'bn':
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.BatchNorm2d(int(out_channels), momentum=momentum),)
        elif norm_type == 'gn':
            self.cb_unit = nn.Sequential(conv_mod,
                                         nn.GroupNorm(num_groups=num_groups, num_channels=int(out_channels)),)
        else:
            self.cb_unit = nn.Sequential(conv_mod,)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class ConvBatchNormRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=True, dilation=1, groups=1, negative_slope=0.0, norm_type='bn', momentum=0.1, num_groups=32):
        super(ConvBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(int(in_channels), int(out_channels), kernel_size=kernel_size,
                             padding=padding, stride=stride, bias=bias, dilation=dilation, groups=groups)
        relu_mod = nn.LeakyReLU(negative_slope=negative_slope, inplace=True) if negative_slope > 0.0 else nn.ReLU(inplace=True)

        if norm_type == 'bn':
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.BatchNorm2d(int(out_channels), momentum=momentum),
                                          relu_mod,)
        elif norm_type == 'gn':
            self.cbr_unit = nn.Sequential(conv_mod,
                                          nn.GroupNorm(num_groups=num_groups, num_channels=int(out_channels)),
                                          relu_mod,)
        else:
            self.cbr_unit = nn.Sequential(conv_mod,
                                          relu_mod,)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class BottleneckConv(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, 
                 stride, dilation=1, groups=1, norm_type='bn'):
        super(BottleneckConv, self).__init__()

        bias = not (norm_type == 'bn' or norm_type == 'gn')

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.cbr1 = ConvBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, norm_type=norm_type)
        self.cbr2 = ConvBatchNormRelu(mid_channels, mid_channels, 3, stride=stride, padding=dilation, bias=bias, dilation=dilation, groups=groups, norm_type=norm_type)
        self.cb3 = ConvBatchNorm(mid_channels, out_channels, 1, stride=1, padding=0, bias=bias, norm_type=norm_type)
        self.cb4 = ConvBatchNorm(in_channels, out_channels, 1, stride=stride, padding=0, bias=bias, norm_type=norm_type)

    def forward(self, x):
        conv = self.cb3(self.cbr2(self.cbr1(x)))
        residual = self.cb4(x) if self.in_channels != self.out_channels else x
        return F.relu(conv+residual, inplace=True)


class BottleneckIdentify(nn.Module):
    def __init__(self, in_channels, mid_channels, stride, dilation=1, groups=1, norm_type='bn'):
        super(BottleneckIdentify, self).__init__()

        bias = not (norm_type == 'bn' or norm_type == 'gn')

        self.cbr1 = ConvBatchNormRelu(in_channels, mid_channels, 1, stride=1, padding=0, bias=bias, norm_type=norm_type)
        self.cbr2 = ConvBatchNormRelu(mid_channels, mid_channels, 3, stride=1, padding=dilation, bias=bias, dilation=dilation, groups=groups, norm_type=norm_type)
        self.cb3 = ConvBatchNorm(mid_channels, in_channels, 1, stride=1, padding=0, bias=bias, norm_type=norm_type)
        
    def forward(self, x):
        residual = x
        x = self.cb3(self.cbr2(self.cbr1(x)))
        return F.relu(x+residual, inplace=True)


class ResidualBlock(nn.Module):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation=1, groups=1, norm_type='bn'):
        super(ResidualBlock, self).__init__()

        if dilation > 1:
            stride = 1

        layers = []
        layers.append(BottleneckConv(in_channels, mid_channels, out_channels, stride, dilation=dilation, groups=groups, norm_type=norm_type))
        for i in range(n_blocks-1):
            layers.append(BottleneckIdentify(out_channels, mid_channels, stride, dilation=dilation, groups=groups, norm_type=norm_type))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class PPM(nn.Module): # Pyramid Pooling Module
    def __init__(self, in_channels, pool_sizes, mode='cat', norm_type='bn'):
        super(PPM, self).__init__()

        bias = not (norm_type == 'bn' or norm_type == 'gn')

        self.mode = mode
        self.pool_sizes = pool_sizes

        if mode == 'cat':
            self.paths = []
            for i in range(len(pool_sizes)):
                self.paths.append(ConvBatchNormRelu(in_channels, int(in_channels / len(pool_sizes)), 1, 1, 0, bias=bias, norm_type=norm_type))
            self.path_module_list = nn.ModuleList(self.paths)

    def forward(self, x):
        h, w = x.shape[2:]

        strides = [(int(h/pool_size), int(w/pool_size)) for pool_size in self.pool_sizes]
        kernel_sizes = [(int(h - strides[i][0]*(pool_size-1)), int(w - strides[i][1]*(pool_size-1))) for i, pool_size in enumerate(self.pool_sizes)]

        if self.mode == 'cat':
            output_slices = [x]
            for i, (module, pool_size) in enumerate(zip(self.path_module_list, self.pool_sizes)):
                out = F.avg_pool2d(x, kernel_sizes[i], stride=strides[i], padding=0)
                out = module(out)
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
                output_slices.append(out)
            return torch.cat(output_slices, dim=1)
        else: #self.mode == 'sum'
            output = 0.0
            for i, pool_size in enumerate(self.pool_sizes):
                out = F.avg_pool2d(x, kernel_sizes[i], stride=strides[i], padding=0)
                out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=False)
                output = output + out
            return output


class ARM(nn.Module): # Attention Refinement Module
    def __init__(self, in_channels, out_channels, norm_type='bn'):
        super(ARM, self).__init__()

        bias = not (norm_type == 'bn' or norm_type == 'gn')

        self.cbr = ConvBatchNormRelu(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type)
        self.arm = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
                                 ConvBatchNorm(out_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias, dilation=1, norm_type=norm_type),
                                 nn.Sigmoid())

    def forward(self, x):
        x = self.cbr(x)
        x = x * self.arm(x)
        return x


class FFM(nn.Module): # Feature Fusion Module
    def __init__(self, in_channels, out_channels, reduction_ratio=2, norm_type='bn'):
        super(FFM, self).__init__()

        bias = not (norm_type == 'bn' or norm_type == 'gn')

        self.cbr = ConvBatchNormRelu(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=bias, norm_type=norm_type)
        self.cse = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
                                 ConvBatchNormRelu(out_channels, out_channels // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=True, dilation=1, norm_type='none'),
                                 ConvBatchNorm(out_channels // reduction_ratio, out_channels, kernel_size=1, stride=1, padding=0, bias=True, dilation=1, norm_type='none'),
                                 nn.Sigmoid())

    def forward(self, x_s, x_c):
        x = torch.cat([x_s, x_c], dim=1)
        x = self.cbr(x)
        x = x + x * self.cse(x)
        return x


# https://github.com/junfu1115/DANet/blob/master/encoding/nn/attention.py ; https://github.com/PkuRainBow/OCNet.pytorch/blob/master/oc_module/base_oc_block.py
class PAM(nn.Module): # Position Attention Module
    def __init__(self, in_channels, reduction_ratio=8, norm_type='bn', scale=1):
        super(PAM, self).__init__()

        ##bias = not (norm_type == 'bn' or norm_type == 'gn')

        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=True)
        ##self.query_conv = ConvBatchNorm(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=bias, dilation=1, norm_type=norm_type)
        self.key_conv = self.query_conv ##ConvBatchNorm(in_channels=in_channels, out_channels=in_channels // reduction_ratio, kernel_size=1, stride=1, padding=0, bias=bias, dilation=1, norm_type=norm_type)
        self.value_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        self.maxpool = nn.MaxPool2d(kernel_size=(scale, scale)) if scale > 1 else None
        self.gamma = nn.Parameter(torch.zeros(1)) #nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        ##nn.init.constant(self.gamma.weight, 0)

    def forward(self, x):
        ori_h, ori_w = x.shape[2:]
        if self.maxpool is not None:
            x = self.maxpool(x)

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        if self.maxpool is not None:
            out = F.interpolate(out, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
        return out


class CAM(nn.Module): # Channel Attention Module
    def __init__(self, scale=1):
        super(CAM, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size=(scale, scale)) if scale > 1 else None
        self.gamma = nn.Parameter(torch.zeros(1)) #nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, bias=False)
        ##nn.init.constant(self.gamma.weight, 0)

    def forward(self, x):
        ori_h, ori_w = x.shape[2:]
        if self.maxpool is not None:
            x = self.maxpool(x)

        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = F.softmax(energy_new, dim=-1)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        if self.maxpool is not None:
            out = F.interpolate(out, size=(ori_h, ori_w), mode='bilinear', align_corners=False)
        return out


# CBAM: https://github.com/Jongchan/attention-module
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()

        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
                Flatten(),
                nn.Linear(gate_channels, gate_channels // reduction_ratio),
                nn.ReLU(),
                nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types

    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type == 'max':
                max_pool = F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp( max_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat( (torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1 )

class SpatialGate(nn.Module):
    def __init__(self, kernel_size=7, norm_type='bn'):
        super(SpatialGate, self).__init__()

        bias = not (norm_type == 'bn' or norm_type == 'gn')

        self.compress = ChannelPool()
        self.spatial = ConvBatchNorm(in_channels=2, out_channels=1, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), bias=bias, norm_type=norm_type, momentum=0.01)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False, kernel_size=7, norm_type='bn'):
        super(CBAM, self).__init__()

        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate(kernel_size, norm_type)

    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out
