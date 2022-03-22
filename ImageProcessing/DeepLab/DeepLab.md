# DeepLab系列学习

## DeepLab V1

- **原文**：Semantic image segmentation with deep convolutional nets and fully connected CRFs
- **收录**：ICLR 2015 (International Conference on Learning Representations)
- **Backbone**：VGG16
- **Contributions**：
  - **Atrous convolution**（Dilation convolution）
  - CRF

这里特别学习一些空洞卷积（膨胀卷积）

### Dilation Convolution

### ![img](https://upload-images.jianshu.io/upload_images/2540794-f4b4be2a9128729a.png?imageMogr2/auto-orient/strip|imageView2/2/w/395/format/webp)

Dilated Convolution with a 3 x 3 kernel and dilation rate 2

*为什么语义分割问题 直接使用传统的Deep CNN不好？*

原因有二

- up-sampling层是双线性插值 是不可学习的
- pooling层会导致位置信息丢失 而分割是pixel-wised的任务 因此pooling层会大大损失分割精度  同时因此也会导致小物体的信息无法重新构建

基于以上原因，膨胀卷积就被发明出来

因此膨胀卷积的诞生就是为了能够在增大感受野的同时还能保证原特征图的长和宽

**语义分割由于需要获得较大的分辨率图，因此经常在网络的最后两个stage，取消降采样操作，之后采用膨胀卷积弥补丢失的感受野。**

但是如果直接使用空洞卷积一般效果又会很差 是因为会有Griding Effect的问题

#### Griding Effect

![image-20220322153922559](C:\Users\ZhangHongbin\AppData\Roaming\Typora\typora-user-images\image-20220322153922559.png)

该图分别展示的是 不同膨胀系数策略时  **不同层对于第一层像素的利用情况**

- a图展示的是 三个3×3的膨胀系数为2的卷积核的情况

- b图展示的是 三个3×3的膨胀系数分别为1 2 3 的卷积核的情况 这也就是Hybrid Dilated Convolution（HDC）

（值得一提的是，一开始没太看懂这是啥意思 后来才想明白 后一层的计算建立在前一层的基础之上 比如 在第二层进行卷积运算（此时涉及到第一层的像素的卷积运算就会有九个即围绕着一圈的九次卷积运算）这样的感受野就相当是以第一层九个像素分别为中心在进行九次卷积运算（但数值肯定不是一样的，这里指的是感受野的情况或者说是像素利用次数 颜色深浅就代表利用次数的多少） 同理第三层就是以第二层的二十五个像素为中心在进行二十五次卷积运算 其实是模拟来着  真正的是第一层九个像素已经变成了第二层的一个像素 这个像素再参与到其他的卷积运算就相当于第一层九个像素分别参与到卷积运算）

##### HDC策略

使用HDC的方案解决该问题，不同于采用相同的空洞率的deeplab方案，**该方案将一定数量的layer形成一个组，然后每个组使用连续增加的空洞率，其他组重复。**如deeplab使用rate=2,而HDC采用r=1,r=2,r=3三个空洞率组合，这两种方案感受野都是13。但HDC方案可以从更广阔的像素范围获取信息，避免了grid问题。同时该方案也可以通过修改rate任意调整感受野。

### DeepLab V1结构图

![image-20220322160104085](C:\Users\ZhangHongbin\AppData\Roaming\Typora\typora-user-images\image-20220322160104085.png)

## DeepLab V2

- **原文**：DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs
- **收录**：TPAMI2017 (IEEE Transactions on Pattern Analysis and Machine Intelligence, 2017)
- **Backbone**：ResNet-101
- **Contributions：ASPP**

ASPP是啥？

全称：Atrous Spatial Pyramid Pooling

为什么会提出来？

主要是为了提取多尺度信息，而多尺度信息又是为了能够解决目标在图像中表现为不同大小时仍然能够拥有很好的分割结果。比如同样的物体，在近处拍摄时物体较大，在远处拍摄时物体较小。

具体做法是？

并行地采用多个不同的膨胀系数地膨胀卷积来提取特征，最后将特征融合在一起

![image-20220322161423027](C:\Users\ZhangHongbin\AppData\Roaming\Typora\typora-user-images\image-20220322161423027.png)

![image-20220322161443284](C:\Users\ZhangHongbin\AppData\Roaming\Typora\typora-user-images\image-20220322161443284.png)

DeepLab V2网络结构于V1类似 就是在DCNN那块多了ASPP 来提取多尺度特征

## DeepLab V3

- 原文：Rethinking Atrous Convolution for Semantic Image Segmentation
- 发表会议：CVPR 2017
- Backbone：ResNet-101
- Contributions：
  - ASPP （中间加入了BN层，ASPP分支加入了image pooling和1×1卷积）
  - Going deeper with atrous convolution
  - Remove CRF

文章提出了两种改进结构

其一改进了ASPP的结构 相当于是拓展了网络的宽度

![image-20220322162148689](C:\Users\ZhangHongbin\AppData\Roaming\Typora\typora-user-images\image-20220322162148689.png)

V3中的ASPP与V1不同之处在于

在分支中加入了BN层和非线性激活层

另外是增加了分支的机构

一是使用1×1的卷积，也就是当rate增大以后3×3卷积的退化形式，替代3×3卷积，减少参数个数；另一点就是增加image pooling，可以叫做全局池化，来补充全局特征。具体做法是对每一个通道的像素取平均，之后再上采样到原来的分辨率。



其二利用膨胀卷积加深网络 即增加网络深度

![image-20220322162138395](C:\Users\ZhangHongbin\AppData\Roaming\Typora\typora-user-images\image-20220322162138395.png)