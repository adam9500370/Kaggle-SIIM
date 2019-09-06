import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import *
from models.utils import *


class ContextPath(nn.Module):
    def __init__(self, backbone_name='resnet18', in_channels=3, sc_channels=128, load_pretrained=True, norm_type='bn', use_cbam=False):
        super(ContextPath, self).__init__()

        bias = not (norm_type == 'bn' or norm_type == 'gn')

        self.backbone = resnet(name=backbone_name, in_channels=in_channels, load_pretrained=load_pretrained, norm_type=norm_type, use_cbam=use_cbam)
        expansion = self.backbone.block.expansion
        ##self.ppm = PPM(in_channels=512*expansion, pool_sizes=[6, 3, 2, 1], mode='sum', norm_type=norm_type)

        self.arm_32x = ARM(in_channels=512*expansion, out_channels=sc_channels, norm_type=norm_type)
        self.arm_16x = ARM(in_channels=256*expansion, out_channels=sc_channels, norm_type=norm_type)
        self.arm_8x  = ARM(in_channels=128*expansion, out_channels=sc_channels, norm_type=norm_type)

        self.cbr_gap = ConvBatchNormRelu(in_channels=512*expansion, out_channels=sc_channels, kernel_size=1, stride=1, padding=0, bias=bias, norm_type=norm_type)
        ##self.cbr_ppm = ConvBatchNormRelu(in_channels=512*expansion, out_channels=sc_channels, kernel_size=1, stride=1, padding=0, bias=bias, norm_type=norm_type)
        self.cbr_32x = ConvBatchNormRelu(in_channels=sc_channels,   out_channels=sc_channels, kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type)
        self.cbr_16x = ConvBatchNormRelu(in_channels=sc_channels,   out_channels=sc_channels, kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type)
        self.cbr_8x  = ConvBatchNormRelu(in_channels=sc_channels,   out_channels=sc_channels, kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type)

    def forward(self, x):
        feat_2x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        feat_4x = self.backbone.maxpool(feat_2x)
        feat_4x = self.backbone.layer1(feat_4x)
        feat_8x = self.backbone.layer2(feat_4x)
        feat_16x = self.backbone.layer3(feat_8x)
        feat_32x = self.backbone.layer4(feat_16x)
        feat_gap = F.adaptive_avg_pool2d(feat_32x, output_size=(1, 1)) # Global Average Pooling
        feat_gap = self.cbr_gap(feat_gap)
        ##feat_ppm = self.ppm(feat_32x)
        ##feat_ppm = self.cbr_ppm(feat_ppm)
        ##feat_gap = F.adaptive_avg_pool2d(feat_ppm, output_size=(1, 1)) # Global Average Pooling

        feat_32x = self.arm_32x(feat_32x)
        feat_32x = feat_32x + feat_gap
        ##feat_32x = feat_32x + feat_ppm
        feat_32x = self.cbr_32x(feat_32x)
        up_feat_32x = F.interpolate(feat_32x, size=feat_16x.shape[2:], mode='bilinear', align_corners=False)

        feat_16x = self.arm_16x(feat_16x)
        feat_16x = feat_16x + up_feat_32x
        feat_16x = self.cbr_16x(feat_16x)
        up_feat_16x = F.interpolate(feat_16x, size=feat_8x.shape[2:], mode='bilinear', align_corners=False)

        feat_8x = self.arm_8x(feat_8x)
        feat_8x = feat_8x + up_feat_16x
        feat_8x = self.cbr_8x(feat_8x)
        return feat_8x, feat_16x, feat_32x, feat_gap


class SpatialPath(nn.Module):
    def __init__(self, in_channels=3, sc_channels=128, norm_type='bn'):
        super(SpatialPath, self).__init__()

        bias = not (norm_type == 'bn' or norm_type == 'gn')

        self.cbr1 = ConvBatchNormRelu(in_channels=in_channels,        out_channels=int(sc_channels/2), kernel_size=3, stride=2, padding=1, bias=bias, norm_type=norm_type)
        self.cbr2 = ConvBatchNormRelu(in_channels=int(sc_channels/2), out_channels=int(sc_channels/2), kernel_size=3, stride=2, padding=1, bias=bias, norm_type=norm_type)
        self.cbr3 = ConvBatchNormRelu(in_channels=int(sc_channels/2), out_channels=sc_channels,        kernel_size=3, stride=2, padding=1, bias=bias, norm_type=norm_type)

    def forward(self, x):
        feat_2x = self.cbr1(x)
        feat_4x = self.cbr2(feat_2x)
        feat_8x = self.cbr3(feat_4x)
        return feat_8x


class bisenet(nn.Module): # BiSeNet
    def __init__(self, n_classes=1, backbone_name='resnet18', in_channels=3, sc_channels=128, load_pretrained=True, norm_type='bn', use_cbam=False, p=0.1, dam_scale=1):
        super(bisenet, self).__init__()

        bias = not (norm_type == 'bn' or norm_type == 'gn')

        backbone_name = 'resnet18' if backbone_name is None else backbone_name

        self.spatial_path = SpatialPath(in_channels=in_channels, sc_channels=sc_channels, norm_type=norm_type)
        self.context_path = ContextPath(backbone_name=backbone_name, in_channels=in_channels, sc_channels=sc_channels, load_pretrained=load_pretrained, norm_type=norm_type, use_cbam=use_cbam)
        self.ffm = FFM(in_channels=sc_channels*2, out_channels=sc_channels, reduction_ratio=2, norm_type=norm_type)
        self.pam = PAM(in_channels=sc_channels, reduction_ratio=2, norm_type=norm_type, scale=dam_scale)
        self.cam = CAM(scale=dam_scale)
        self.cbr_pam = ConvBatchNormRelu(in_channels=sc_channels, out_channels=sc_channels, kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type)
        self.cbr_cam = ConvBatchNormRelu(in_channels=sc_channels, out_channels=sc_channels, kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type)

        self.dropout = nn.Dropout2d(p=p)

        # (Auxiliary) Classifier
        self.gamma_gap = nn.Parameter(torch.zeros(1))
        self.aux_cls_gap = nn.Sequential(ConvBatchNormRelu(in_channels=sc_channels, out_channels=int(sc_channels/4), kernel_size=1, stride=1, padding=0, bias=bias, norm_type=norm_type),
                                         self.dropout,
                                         nn.Conv2d(int(sc_channels/4), n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.aux_cls_8x  = nn.Sequential(ConvBatchNormRelu(in_channels=sc_channels, out_channels=int(sc_channels/4), kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                         self.dropout,
                                         nn.Conv2d(int(sc_channels/4), n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.aux_cls_16x = nn.Sequential(ConvBatchNormRelu(in_channels=sc_channels, out_channels=int(sc_channels/4), kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                         self.dropout,
                                         nn.Conv2d(int(sc_channels/4), n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.aux_cls_32x = nn.Sequential(ConvBatchNormRelu(in_channels=sc_channels, out_channels=int(sc_channels/4), kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                         self.dropout,
                                         nn.Conv2d(int(sc_channels/4), n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.cls         = nn.Sequential(ConvBatchNormRelu(in_channels=sc_channels, out_channels=int(sc_channels/4), kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                         self.dropout,
                                         nn.Conv2d(int(sc_channels/4), n_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self._init_weights(load_pretrained=load_pretrained)

    def _init_weights(self, load_pretrained=True):
        for name, m in self.named_modules():
            if 'backbone' in name and load_pretrained:
                continue

            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                ##nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, output_size=(1024, 1024), return_aux=True, interp_aux=True):
        x_s_8x = self.spatial_path(x)
        x_c_8x, x_c_16x, x_c_32x, aux_gap = self.context_path(x)
        out_aux_gap = self.aux_cls_gap(aux_gap)
        if return_aux: #self.training:
            out_aux_32x = self.aux_cls_32x(x_c_32x)
            out_aux_16x = self.aux_cls_16x(x_c_16x)
            out_aux_8x  = self.aux_cls_8x(x_c_8x)
            if interp_aux:
                out_aux_32x = F.interpolate(out_aux_32x, size=output_size, mode='bilinear', align_corners=False)
                out_aux_16x = F.interpolate(out_aux_16x, size=output_size, mode='bilinear', align_corners=False)
                out_aux_8x  = F.interpolate(out_aux_8x,  size=output_size, mode='bilinear', align_corners=False)

        x_ff = self.ffm(x_s_8x, x_c_8x)
        x_pam = self.cbr_pam(self.pam(x_ff))
        x_cam = self.cbr_cam(self.cam(x_ff))
        x_dam = x_pam + x_cam
        out = self.cls(x_dam)
        out = out + self.gamma_gap * out_aux_gap
        if out.shape[2:] != output_size:
            out = F.interpolate(out, size=output_size, mode='bilinear', align_corners=False)

        if return_aux: #self.training:
            return (out, out_aux_8x, out_aux_16x, out_aux_32x), out_aux_gap
        else: # eval mode
            return out
