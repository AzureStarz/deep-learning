from collections import OrderedDict
from typing import Dict, List

import torch
from torch import nn, Tensor
from torch.nn import functional as F
from resnet_backbone import resnet50


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


class DeepLabV3(nn.Module):
    """
    Implements DeepLabV3 model from
    """
    def __init__(self, backbone, classifier, aux_classifier):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.aux_classifier = aux_classifier

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # get input shape
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        result = OrderedDict()
        # 主分支上的输出
        x = features["out"]
        x = self.classifier(x)
        # 使用双线性插值还原回原图尺寸
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
        result["out"] = x

        # Aux分支上的输出
        if self.aux_classifier is not None:
            x = features["aux"]
            x = self.aux_classifier(x)
            # 使用双线性插值还原回原图尺寸
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)
            result["aux"] = x

        return  result


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        super(ASPPConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        # module就是上面super传过去的结构
        for module in self:
            x = module(x)
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, dilation_rates, out_channels=256):
        super(ASPP, self).__init__()
        _modules = [
            # 对应的第一个1×1大小的卷积核
            nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU())
        ]
        # 对应的三个膨胀卷积操作
        rates = tuple(dilation_rates)
        for rate in rates:
            _modules.append(ASPPConv(in_channels, out_channels, rate))
        # 对应最后一个image pooling分支
        _modules.append(ASPPPooling(in_channels, out_channels))
        # 将列表转换成ModuleList
        self.convs = nn.ModuleList(_modules)
        # 对应ASPP末尾的投影层 也就是1×1卷积将channel数降低
        self.project = nn.Sequential(
            nn.Conv2d(len(self.convs) * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        _res = []
        for conv in self.convs:
            _res.append(conv(x))
        res = torch.cat(_res, dim=1)
        return self.project(res)


class DeepLabHead(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(DeepLabHead, self).__init__(
            ASPP(in_channels, [12, 24, 36]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )


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


def deeplabv3_resnet50(aux, num_classes=21, pretrain_backbone=False):
    backbone = resnet50(replace_stride_with_dilation=[False, True, True])

    if pretrain_backbone:
        print("Do Something")

    out_planes = 2048
    aux_planes = 1024

    return_layers = {'layer4': 'out'}
    if aux:
        return_layers = {'layer3': 'aux'}
    backbone = IntermediateLayerGetter(backbone, return_layers)

    aux_classifier = None
    if aux:
        aux_classifier = FCNHead(aux_planes, num_classes)

    classifier = DeepLabHead(out_planes, num_classes)

    model = DeepLabV3(backbone, classifier, aux_classifier)

    return model