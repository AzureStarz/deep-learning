import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 卷积操作（带padding）"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                        padding=dilation, groups=groups, dilation=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积操作"""
    return nn.Conv2d(in_planes, out_planes,
                        stride=stride, kernel_size=1, bias=False)


class Bottleneck(nn.Module):
    # 特征扩大倍数
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                    base_width=64, groups=1, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        # 定义中间Bottleneck特征通道数
        width = int(out_channel * (base_width / 64.)) * groups
        # 定义归一化层 不写死是BatchNorm原因 我估计是方便更改模型的结构 更加灵活
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # 第一个模块
        self.block1 = nn.Sequential(
            conv1x1(in_channel, width),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        # 第二个模块
        self.block2 = nn.Sequential(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width),
            nn.ReLU(inplace=True)
        )
        # 第三个模块
        self.block3 = nn.Sequential(
            conv1x1(width, out_channel * self.expansion),
            norm_layer(out_channel)
        )
        # 捷径分支
        self.downsample = downsample
        # 最后一层激活函数
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        # 如果输入输出通道数不一致才需要降采样处理
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, blocks_num, num_classes=1000,
                    groups=1, width_per_group=64,
                    norm_layer=None, replace_stride_with_dilation=None):
        super(ResNet, self).__init__()
        # 设立归一化层
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False * 3]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                                "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        # 初始化参数
        self.groups = groups
        self.base_width = width_per_group
        # 第一层的通道数
        self.inplanes = 64
        self.dilation = 1
        # 模型刚开始输入时的模块
        self.inputBlock = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2,
                        padding=3, bias=False),
            self._norm_layer(self.inplanes),
            nn.ReLU(inplace=True)
        )
        # 池化层下采样
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 开始构建每一层残差块 每一层都有多个残差块
        self.layer1 = self._make_layer(Bottleneck, 64, blocks_num[0])
        self.layer2 = self._make_layer(Bottleneck, 128, blocks_num[1], stride=2,
                                        dilation=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(Bottleneck, 256, blocks_num[2], stride=2,
                                        dilation=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(Bottleneck, 512, blocks_num[3], stride=2,
                                        dilation=replace_stride_with_dilation[2])
        # 最后使用平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, block_num, stride=1, dilation=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilation:
            self.dilation *= stride
            stride = 1
        # 如果输入进残差块的通道数与最后输出的通道数不同 就需要降采样 使得通道相同才能够残差链接
        # 且只会出现在不同的layer之间 因为同一层layer输入输出是一样的 所以不需要降采样
        # 而不同layer之间特征维度不同
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion)
            )

        layers = []
        # 第一层与其他层不一样的原因 是要衔接上下不同的残差块
        # 其他层只需要衔接相同的相同的残差块
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        # 更新此层的out_channel数
        self.inplanes = planes * block.expansion
        # 循环迭代生成剩下的残差层
        for _ in range(1, block_num):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.inputBlock(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, **kwargs):
    return ResNet(block, layers, **kwargs)


def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], ** kwargs)
