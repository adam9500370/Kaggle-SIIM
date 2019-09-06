from models.resnet import *
from models.unet import *
from models.bisenet import *


def get_model(name, n_classes, in_channels=3, load_pretrained=True, norm_type='bn', use_cbam=False):
    model = _get_model_instance(name)

    if name.startswith('resnet') or name.startswith('resnext'):
        model = model(name=name, n_classes=n_classes, in_channels=in_channels, load_pretrained=load_pretrained, norm_type=norm_type, use_cbam=use_cbam)

    if name.startswith('unet') or name.startswith('bisenet'):
        model = model(backbone_name=name.split('-')[1] if '-' in name else None, n_classes=n_classes, in_channels=in_channels, load_pretrained=load_pretrained, norm_type=norm_type, use_cbam=use_cbam)

    return model

def _get_model_instance(name):
    try:
        return {
            'resnet18': resnet,
            'resnet34': resnet,
            'resnet50': resnet,
            'resnet101': resnet,
            'resnet152': resnet,
            'resnet38': resnet,
            'resnext50-32x4d': resnet,
            'resnext101-32x8d': resnet,
            'unet': unet,
            'unet-resnet18': unet,
            'unet-resnet34': unet,
            'unet-resnet50': unet,
            'bisenet': bisenet,
            'bisenet-resnet18': bisenet,
            'bisenet-resnet34': bisenet,
            'bisenet-resnet50': bisenet,
        }[name]
    except:
        print('Model {} not available'.format(name))
