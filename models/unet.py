import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import *
from models.utils import *


class unet(nn.Module):
    def __init__(self, n_classes=1, backbone_name='resnet18', in_channels=3, load_pretrained=True, norm_type='bn', use_cbam=False, p=0.1, dam_scale=1):
        super(unet, self).__init__()

        bias = not (norm_type == 'bn' or norm_type == 'gn')

        backbone_name = 'resnet18' if backbone_name is None else backbone_name

        self.dropout = nn.Dropout2d(p=p)

        self.backbone = resnet(name=backbone_name, in_channels=in_channels, load_pretrained=load_pretrained, norm_type=norm_type, use_cbam=use_cbam)
        expansion = self.backbone.block.expansion
        self.ppm = PPM(in_channels=512*expansion, pool_sizes=[6, 3, 2, 1], mode='sum', norm_type=norm_type)

        self.c1br_16x = ConvBatchNormRelu(in_channels=256*expansion, out_channels=128*expansion, kernel_size=1, stride=1, padding=0, bias=bias, norm_type=norm_type)
        self.c1br_8x  = ConvBatchNormRelu(in_channels=128*expansion, out_channels=64*expansion,  kernel_size=1, stride=1, padding=0, bias=bias, norm_type=norm_type)
        self.c1br_4x  = ConvBatchNormRelu(in_channels=64*expansion,  out_channels=32*expansion,  kernel_size=1, stride=1, padding=0, bias=bias, norm_type=norm_type)
        self.c1br_2x  = ConvBatchNormRelu(in_channels=64*expansion,  out_channels=16*expansion,  kernel_size=1, stride=1, padding=0, bias=bias, norm_type=norm_type)

        self.c3br_ppm  = ConvBatchNormRelu(in_channels=512*expansion, out_channels=128*expansion, kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type)
        self.c3br2_16x = nn.Sequential(
                                ConvBatchNormRelu(in_channels=256*expansion, out_channels=64*expansion,  kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                ConvBatchNormRelu(in_channels=64*expansion,  out_channels=64*expansion,  kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                            )
        self.c3br2_8x  = nn.Sequential(
                                ConvBatchNormRelu(in_channels=128*expansion, out_channels=32*expansion,  kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                ConvBatchNormRelu(in_channels=32*expansion,  out_channels=32*expansion,  kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                            )
        self.c3br2_4x  = nn.Sequential(
                                ConvBatchNormRelu(in_channels=64*expansion,  out_channels=16*expansion,  kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                ConvBatchNormRelu(in_channels=16*expansion,  out_channels=16*expansion,  kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                            )
        self.c3br2_2x  = nn.Sequential(
                                ConvBatchNormRelu(in_channels=32*expansion,  out_channels=8*expansion,   kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                ConvBatchNormRelu(in_channels=8*expansion,   out_channels=8*expansion,   kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                            )

        #"""
        self.pam = PAM(in_channels=32*expansion, reduction_ratio=2, norm_type=norm_type, scale=dam_scale)
        self.cam = CAM(scale=dam_scale)
        self.c3br_pam = ConvBatchNormRelu(in_channels=32*expansion, out_channels=32*expansion, kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type)
        self.c3br_cam = ConvBatchNormRelu(in_channels=32*expansion, out_channels=32*expansion, kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type)
        #"""

        # (Auxiliary) Classifier
        self.gamma_gap = nn.Parameter(torch.zeros(1))
        self.aux_cls_gap = nn.Sequential(ConvBatchNormRelu(in_channels=128*expansion, out_channels=32*expansion, kernel_size=1, stride=1, padding=0, bias=bias, norm_type=norm_type),
                                         self.dropout,
                                         nn.Conv2d(32*expansion, n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.aux_cls_ppm = nn.Sequential(ConvBatchNormRelu(in_channels=128*expansion, out_channels=32*expansion, kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                         self.dropout,
                                         nn.Conv2d(32*expansion, n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.aux_cls_4x  = nn.Sequential(ConvBatchNormRelu(in_channels=32*expansion,  out_channels=8*expansion,  kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                         self.dropout,
                                         nn.Conv2d(8*expansion,  n_classes, kernel_size=1, stride=1, padding=0, bias=True))
        self.cls         = nn.Sequential(ConvBatchNormRelu(in_channels=8*expansion,   out_channels=8*expansion,  kernel_size=3, stride=1, padding=1, bias=bias, norm_type=norm_type),
                                         self.dropout,
                                         nn.Conv2d(8*expansion,  n_classes, kernel_size=1, stride=1, padding=0, bias=True))

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
        feat_2x = self.backbone.relu(self.backbone.bn1(self.backbone.conv1(x)))
        feat_4x = self.backbone.maxpool(feat_2x)
        feat_4x = self.backbone.layer1(feat_4x)
        feat_8x = self.backbone.layer2(feat_4x)
        feat_16x = self.backbone.layer3(feat_8x)
        feat_32x = self.backbone.layer4(feat_16x)
        feat_ppm = self.ppm(feat_32x)
        feat_ppm = self.c3br_ppm(feat_ppm)
        feat_gap = F.adaptive_avg_pool2d(feat_ppm, output_size=(1, 1)) # Global Average Pooling

        feat_16x = self.c1br_16x(feat_16x)
        up_feat_ppm = F.interpolate(feat_ppm, size=feat_16x.shape[2:], mode='bilinear', align_corners=False)
        feat_16x = torch.cat([feat_16x, up_feat_ppm], dim=1)
        feat_16x = self.c3br2_16x(feat_16x)

        feat_8x = self.c1br_8x(feat_8x)
        up_feat_16x = F.interpolate(feat_16x, size=feat_8x.shape[2:], mode='bilinear', align_corners=False)
        feat_8x = torch.cat([feat_8x, up_feat_16x], dim=1)
        feat_8x = self.c3br2_8x(feat_8x)

        #"""
        feat_8x_pam = self.c3br_pam(self.pam(feat_8x))
        feat_8x_cam = self.c3br_cam(self.cam(feat_8x))
        feat_8x_dam = feat_8x_pam + feat_8x_cam
        #"""

        feat_4x = self.c1br_4x(feat_4x)
        up_feat_8x = F.interpolate(feat_8x_dam, size=feat_4x.shape[2:], mode='bilinear', align_corners=False)
        feat_4x = torch.cat([feat_4x, up_feat_8x], dim=1)
        feat_4x = self.c3br2_4x(feat_4x)

        feat_2x = self.c1br_2x(feat_2x)
        up_feat_4x = F.interpolate(feat_4x, size=feat_2x.shape[2:], mode='bilinear', align_corners=False)
        feat_2x = torch.cat([feat_2x, up_feat_4x], dim=1)
        feat_2x = self.c3br2_2x(feat_2x)

        out_aux_gap = self.aux_cls_gap(feat_gap)
        if return_aux: #self.training:
            out_aux_ppm = self.aux_cls_ppm(feat_ppm)
            out_aux_4x = self.aux_cls_4x(up_feat_8x)
            if interp_aux:
                out_aux_ppm = F.interpolate(out_aux_ppm, size=output_size, mode='bilinear', align_corners=False)
                out_aux_4x  = F.interpolate(out_aux_4x,  size=output_size, mode='bilinear', align_corners=False)

        up_feat_2x = F.interpolate(feat_2x, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = self.cls(up_feat_2x)
        out = out + self.gamma_gap * out_aux_gap
        if out.shape[2:] != output_size:
            out = F.interpolate(out, size=output_size, mode='bilinear', align_corners=False)

        if return_aux: #self.training:
            return (out, out_aux_4x, out_aux_ppm), out_aux_gap
        else: # eval mode
            return out
