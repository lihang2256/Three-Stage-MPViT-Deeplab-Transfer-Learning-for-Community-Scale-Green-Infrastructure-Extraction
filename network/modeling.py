from .utils import IntermediateLayerGetter
from ._deeplab import DeepLabHead, DeepLabHeadV3Plus, DeepLabV3
from ._deeplab_fz import DeepLabHeadV3PlusFz, DeepLabV3Fz
from .backbone import resnet
from .backbone import mobilenetv2
from .backbone import xception
from .backbone import MPViT
from .backbone import ViT
from .backbone import MPViT_fz
from torchsummary import summary
import torch


def _segm_resnet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        replace_stride_with_dilation = [False, True, True]
        aspp_dilate = [12, 24, 36]
    else:
        replace_stride_with_dilation = [False, False, True]
        aspp_dilate = [6, 12, 18]

    backbone = resnet.__dict__[backbone_name](
        pretrained=pretrained_backbone,
        replace_stride_with_dilation=replace_stride_with_dilation)

    inplanes = 2048
    low_level_planes = 256

    if name == 'deeplabv3plus':
        return_layers = {'layer4': 'out', 'layer1': 'low_level'}  #
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'layer4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    # 提取网络的第几层输出结果并给一个别名
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_mobilenet(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = mobilenetv2.mobilenet_v2(pretrained=pretrained_backbone, output_stride=output_stride)

    # rename layers
    backbone.low_level_features = backbone.features[0:4]
    backbone.high_level_features = backbone.features[4:-1]
    backbone.features = None
    backbone.classifier = None

    inplanes = 320
    low_level_planes = 24

    if name == 'deeplabv3plus':
        return_layers = {'high_level_features': 'out', 'low_level_features': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'high_level_features': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    model = DeepLabV3(backbone, classifier)
    return model


def _segm_xception(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = xception.xception(num_classes=num_classes, pretrained=pretrained_backbone, output_stride=output_stride)

    inplanes = 2048
    low_level_planes = 128

    if name == 'deeplabv3plus':
        return_layers = {'conv4': 'out', 'block1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'conv4': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    # print(backbone.items())
    model = DeepLabV3(backbone, classifier)
    return model


def _seg_mpvit(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = MPViT.mpvit_base(pretrained=pretrained_backbone, output_stride=output_stride)

    inplanes = 480
    low_level_planes = 224

    if name == 'deeplabv3plus':
        return_layers = {'mhca4': 'out', 'mhca1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3plus_fz':
        return_layers = {'mhca4': 'out', 'mhca1': 'low_level'}
        classifier = DeepLabHeadV3PlusFz(inplanes, low_level_planes, num_classes, aspp_dilate)
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        model = DeepLabV3Fz(backbone, classifier)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    # print(backbone.items())
    model = DeepLabV3(backbone, classifier)
    return model


def _seg_mpvit_fz(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = MPViT_fz.mpvit_fz_base(pretrained=pretrained_backbone, output_stride=output_stride)

    inplanes = 480
    low_level_planes = 224

    if name == 'deeplabv3plus':
        return_layers = {'mhca4': 'out', 'mhca1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        model = DeepLabV3(backbone, classifier)
    elif name == 'deeplabv3plus_fz':
        return_layers = {'mhca4': 'out', 'mhca1': 'low_level'}
        classifier = DeepLabHeadV3PlusFz(inplanes, low_level_planes, num_classes, aspp_dilate)
        backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
        model = DeepLabV3Fz(backbone, classifier)
    return model


def _seg_vit(name, backbone_name, num_classes, output_stride, pretrained_backbone):
    if output_stride == 8:
        aspp_dilate = [12, 24, 36]
    else:
        aspp_dilate = [6, 12, 18]

    backbone = ViT.vit_b(pretrained=pretrained_backbone, output_stride=output_stride)

    inplanes = 768
    low_level_planes = 768

    if name == 'deeplabv3plus':
        # return_layers = {'encoder': 'out', 'patch_embedding2': 'low_level'}
        return_layers = {'patch_embedding4': 'out', 'patch_embedding1': 'low_level'}
        classifier = DeepLabHeadV3Plus(inplanes, low_level_planes, num_classes, aspp_dilate)
    elif name == 'deeplabv3':
        return_layers = {'out1': 'out'}
        classifier = DeepLabHead(inplanes, num_classes, aspp_dilate)
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)
    # print(backbone.items())
    model = DeepLabV3(backbone, classifier)
    return model


def _load_model(arch_type, backbone, num_classes, output_stride, pretrained_backbone):
    if backbone == 'mobilenetv2':
        model = _segm_mobilenet(arch_type, backbone, num_classes, output_stride=output_stride,
                                pretrained_backbone=pretrained_backbone)
    elif backbone.startswith('resnet'):
        model = _segm_resnet(arch_type, backbone, num_classes, output_stride=output_stride,
                             pretrained_backbone=pretrained_backbone)
    elif backbone == 'xception':
        model = _segm_xception(arch_type, backbone, num_classes, output_stride=output_stride,
                               pretrained_backbone=pretrained_backbone)
    elif backbone == 'mpvit':
        model = _seg_mpvit(arch_type, backbone, num_classes, output_stride=output_stride,
                           pretrained_backbone=pretrained_backbone)
    elif backbone == 'mpvit_fz':
        model = _seg_mpvit_fz(arch_type, backbone, num_classes, output_stride=output_stride,
                           pretrained_backbone=pretrained_backbone)
    elif backbone == 'vit1':
        model = _seg_vit(arch_type, backbone, num_classes, output_stride=output_stride,
                         pretrained_backbone=pretrained_backbone)
    else:
        raise NotImplementedError
    return model


# Deeplab v3

def deeplabv3_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'resnet101', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True, **kwargs):
    """Constructs a DeepLabV3 model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3', 'mobilenetv2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


# Deeplab v3+

def deeplabv3plus_resnet50(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3 model with a ResNet-50 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet50', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_resnet101(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a ResNet-101 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'resnet101', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mobilenet(num_classes=21, output_stride=8, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a MobileNetv2 backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mobilenetv2', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


"""
author: Li Hang
"""


def deeplabv3plus_xception(num_classes=6, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a Xception backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'xception', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


"""
author: Li Hang
"""


def deeplabv3plus_mpvit(num_classes=6, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a mpvit backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mpvit', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_mpvit_fz(num_classes=6, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a mpvit_fz backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'mpvit_fz', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)

def deeplabv3plus_fz_mpvit(num_classes=6, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a mpvit_fz backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus_fz', 'mpvit', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)

def deeplabv3plus_fz_mpvit_fz(num_classes=6, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a mpvit_fz backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus_fz', 'mpvit_fz', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)


def deeplabv3plus_vit(num_classes=6, output_stride=16, pretrained_backbone=True):
    """Constructs a DeepLabV3+ model with a vit backbone.

    Args:
        num_classes (int): number of classes.
        output_stride (int): output stride for deeplab.
        pretrained_backbone (bool): If True, use the pretrained backbone.
    """
    return _load_model('deeplabv3plus', 'vit1', num_classes, output_stride=output_stride,
                       pretrained_backbone=pretrained_backbone)
