from collections import OrderedDict
from typing import Dict
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from backbone import resnet50


class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: Dict[str, str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
        origin_return_layers = return_layers
        return_layers = {str(k): str(v) for k, v in return_layers.items()}

        # 重新构建backbone，将没用的层去掉 目的只是为了获取backbone提取的特征
        layers = OrderedDict()
        for name, model in list(model.named_children()):
            layers[name] = model
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        self.return_layers = origin_return_layers

        super(IntermediateLayerGetter, self).__init__(layers)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        out = OrderedDict()
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                out_name = self.return_layers[name]
                out[out_name] = x
        return out


class FCNHead(nn.Module):
    def __init__(self, in_channel, out_channel, dropout):
        super(FCNHead, self).__init__()
        # 过度的通道数
        inter_channel = in_channel // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, inter_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(inter_channel),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(inter_channel, out_channel, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.block(x)
        x = self.dropout(x)
        x = self.conv(x)

        return x


class FCN(nn.Module):
    def __init__(self, backbone, classifier, aux_classifier=None):
        super(FCN, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x):
        input_size = x.shape[-2:]
        # contract: features is a dict of tensors
        features = self.backbone(x)

        result = OrderedDict()
        x = features["out"]
        x = self.classifier(x)
        # 双线性插值还原回原先大小
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
        result["out"] = x

        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 双线性插值还原回原先大小
            x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=True)
            result["aux"] = x

        return result


def fcn_resnet50(aux, num_classes=21, pretrained_backbone=False):
    backbone = resnet50(replace_strid_dilation=[False, True, True])

    if pretrained_backbone:
        print('Do Something')
        # load weight

    out_planes = 2048
    aux_planes = 1024

    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"
    backbone = IntermediateLayerGetter(backbone, return_layers)

    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_planes, num_classes)

    classifier = FCNHead(out_planes, num_classes)

    model = FCN(backbone, classifier, aux_classifier)

    return model