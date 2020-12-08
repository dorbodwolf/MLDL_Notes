# 1 语义分割综述

## 1.1 评价指标-miou/fwiou
基于混淆矩阵计算

<img src="./images/miou.jpg " alt="miou" title="miou" width="500" height="500" />

## 1.2 模型原理

语义分割发展史：

>1. 深度学习方法碾压传统方法
>2. 在全卷机网络FCN之前，深度学习处理语义分割的思路是patch classification，即对图像切块来做像素分类
>3. FCN是具有里程碑意义，卷积后接反卷积（上采样）而不是全连接层来实现逐像素分类。上采样损失细节信息，可以通过跳跃连接来有效还原部分细节信息。
>4. 归属于encoder-decoder的unet是在fcn的思想上建立的网络；用于医学影像分割；特征提取用的不是vgg等预训练网络，而是自由扩展深度的cnn；unet还有一个特点是捷径连接。
>5. 空洞卷积取代pooling来扩大感受野且能够保持图像尺寸。

PSPNet：
>1. 场景分析
>2. 基于fcn的方法缺乏恰当的手段来利用全局场景类别线索。
>3. 空间金字塔池化对于捕捉全局信息效果很好，空间金字塔网络进一步加强了这一能力。
>4. 除了结合全局和局部的线索，该论文提出结合【深度监督损失】的优化策略。

deeplab v3+：
>1. modified xception
>2. xception骨干网络 input flow调用三次block函数
>3. 深度可分离卷积的实现用到了pytorch的pad函数
>4. 网络设计理念：模块化设计
>5. encoder-decoder 有助于提取锐利边缘？（deeplabv3+摘要中描述，理解有误）
>6. Unet和fcn都属于编码-解码结构
>7. pytorch实现：sequential 定义块
>8. pytorch forward 函数是模型定义的逻辑流

注意力语义分割（信息融合）：
>1. nonlocal 保边滤波，保持类间方差的同时，减少类内方差的效果。
>2. PSANET 两路attention，相当于transformer中的两个head，两路分别起到collect和distribute的作用。
>3. Senet 通道信息融合
>4. Cbam 使用了压缩+激活的思想。同时使用通道+空间信息融合。
<img src="./images/cbam.png " alt="miou" title="miou" width="800" height="300" />
>5. danet是cbam和non-local的结合
>6. gcnet优化danet，优化no-local的时间复杂度
>7. Emanet是期望最大值的注意力机制

# 2 优化算法

2.1 梯度下降

important kind of search in current AI

based on concepts from multivariable calculus

search in continuous state space

```
gradientDescent(L){  //损失函数输入
    S = [s1, s2, ..., sn];  // 初始变量
    repeat{  
        G = gradient(L,S); //求L在S处的梯度G
        S = S - e*G; //更新S，速率是e
    } until(termination condition)
    return S;
}
```

# 3 损失函数

3.1 交叉熵损失函数

<img src="./images/crossentropyloss.png " alt="miou" title="miou" width="600" height="400" />


# 4 卷积神经网络

## 4.1 卷积

卷积运算的本质：

<img src="./images/conv.png " alt="miou" title="miou" width="600" height="600" />





# 5 目标检测RCNN家族

## Faster-RCNN：

> 1. cnn提取feature map，接rpn网络进行目标框定位，接roi pooling将目标框对应特征图resize到相同的尺度，两层全连接层后分别进行目标的softmax分类和基于回归的目标框精修。

RoI和RoI Pooling
> 1. 在Faster R-CNN中，RoI是Regions of Interest的意思，又叫Region Proposals，是区域候选网络Region Proposal Network, RPN的输出结果。
> 2. RoI Pooling的计算原理可以参考以下动图:
> 
> ![RoI Pooling](https://img-blog.csdn.net/20180511113933913?watermark/2/text/aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3UwMTE0MzY0Mjk=/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)

## 实现细节
超参数
> 1. 如果RoI和GD的IoU大于0.5则该RoI被认为是正样本，反之亦然；
> 2. 按照正负样本1:3的比例抽取RoIs进行训练
> 3. 学习率随着迭代次数增加而减小

## 实例分割是个什么

> 何凯明大神所言，实例分割有两种思路，一是segmentation-first，二是instance-first，Mask R-CNN是后者

RoI Pooling的缺陷和RoIAlign的引入

> 分块池化导致RoI和Feaure Map错位(misalignments)，这对分类来说带来误差可以忽略不记，但是对于逐像素的分割掩膜来说影响就大了。

Mask-RCNN：

> 目标检测用了faster r-cnn，特征提取用了fpn

实验提升技巧
> 1. 端到端训练相比先训练RPN再训练Mask R-CNN会提升精度
> 2. 基于ImageNet预训练可以提升精度
> 3. 训练时增强可以提升精度
> 4. 把ResNeXt从101层提升到152层可以提升精度
> 5. 测试时增强可以提升精度

# 6 YOLO

## YOLOv1

### 原理介绍

把目标检测当作一个回归问题求解，inference只需一次回归，用速度换精度。  

yolov1把输入图像划分为S*S的格栅，每个格栅负责检测中心落在该格栅的目标，包括目标的坐标值和confidence scores。另外，每个格栅预测一个类别概率值，输出一张类别概率图。

> confidence = P(obj) * IoU(truth,pred)

上式中，当格栅中心点没有目标时P(obj)为0，此时confidence为0；否则为IOU的值。

![yolov1](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3vKjfbgonxppibe0PQ9Y92IVTiaS8mia6ssakgicBeUpEROxsbrFqwD9sS85RpSQp8gIwicWxpzc0FSHEQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 网络结构

仔细阅读[YOLOv1模型](./kms/YOLOv1.km)

![network_of_yolov1](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3vKjfbgonxppibe0PQ9Y92IVMD3ebL12UYt4qw9qQVGHMJIfDMPej162rdFelO3GcEv8GHA04b2Phw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

### 训练阶段

激活函数：Leaky RELU

损失函数设计：损失函数设计的目标是将坐标 (x, y, w, h)，confidence，classification这三个方面达到很好的平衡，所以不能简单的用平方根误差来对所有变量进行回归。

> ![YOLOv1损失函数设计](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3vKjfbgonxppibe0PQ9Y92IVRjDeibblt4YofDTVjjxfkLUFfqqgPIelHefNfXvD1hLicSia8Wt2UjdJw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
> 
> 上图可见损失函数包含五项的和。
>   
> p1 ————第一项+第二项是坐标的偏移，赋予很高的loss权重，实验中取5。第一项是x和y坐标的偏移，第二项是w和h坐标的偏移。
> 
> p2 ————针对第二项，作者用w和h的平方根偏差来计算loss是为了凸显小目标发生偏差的损失。
> 在yolo的样本标签中，w和h是bbox相对图像的宽高大小，目标越大，w和h就越大，目标越小，w和h就越小。小的目标发生Δw和Δh的偏差反映在平方根上要大于大目标的值，以此来放大小目标的损失，来提升小目标的检测效果，具体原理放一张图就容易理解了：
> 
> ![yolov1小目标loss放大数学原理](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3vKjfbgonxppibe0PQ9Y92IVqrdUwlhFCCgJJnyBmyS8V6uMib6M0kicIXI1NMH7v6CRuBeicIOCgj9VA/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
> 
> p3 ————第三项+第四项是bbox的confidence损失，但是分为含有object和不含object，不含object的confidence损失给了0.5的权重。我是这样理解的，不含object的bbox要比含有object的bbox多的多，为了防止其挟持梯度下降过程导致模型难收敛或发散而给予小的权重。
> 
> p4 ————第五项是类别图的损失，权重为1.

### 推理阶段

将bbox的IoU和格栅的类别概率相乘得到class-special confidence score，这个score就是衡量预测的精度。每个格栅的每个bbox都要计算这个score，每个格栅包含S*S*B个bbox的预测，如果S=7 B=2，则每个格栅有98个bbox预测结果，对其score进行阈值过滤来剩下高分数的bbox，再进行NMS过滤来输出最终的预测结果。


## YOLOv2

### 提升

anchor boxes

> 相比yolov1划分S*S大小的格栅，每个格栅预测B个目标，yolov2引入anchor boxes，在13*13的特征图上的每个cell上预测5个anchor boxes，对于每个anchor box预测bbox的的坐标、置信度和分类概率，可以预测13*13*5=845个bbox，大大提升模型召回率（漏检少了）。
> 
> 同样是ancher-based的检测器，不同于faster rcnn在每个cell手动设置9个anchor boxes，yolov2的作者通过在COCO和VOC数据集上做kmeans聚类来确定anchor boxes的个数、分布和形态。如下图：
> 
> ![yolov2_anchor_kmeans](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3sNDQPd93l6hP2fRHsia8cGOsFxkAjmHnyMAkV435JBPJk1pZwzyDR4bDaPFqpD7WaDWs5GOK3VANw/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)
> 
> **思考**：**对于我们的遥感数据集也按这样的方式设计anchor是不是可以提升检测效果**？
> 

新的backbone

> Darknet-19

多尺度训练

> 不固定训练图像尺寸

损失函数

> 沿用yolov1的训练方式，将损失分为坐标、IoU和类别概率三部分，但是做了不少改进，变得更复杂了。

## YOLOv3

看下图就够了，相比v2也就增加了残差模型Darknet-53和FPN结构。

![network_of_yolov3](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3udtqwPpm1NLnLRNTabbFjLmkhuheSQErWAdibHrXAC1TzZ6LmspOb0OoGWku801jiazYracIlNBW6w/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

解释上图：
> 左侧是包含五个残差块的Darknet-53 backbone
> 
> 右侧为FPN结构及在其上进行的三个anchor-base的检测路. FPN就是一种金字塔结构，在不同尺度上输出特征图
> 
> yolov3在三个尺度上输出特征图进行预测，算是多尺度预测，三个输出分别是(batch, 52, 52, 75), (batch, 26, 26, 75), (batch, 13, 13, 75)。每个预测尺度上基于维度聚类的方法在每个cell上生成3个anchor boxes，和faster rcnn一样总共也是9个anchor boxes。 
> 
> yolov3-tiny是更加轻量级的yolov3，层数大大约减，但是依然保留了两个尺度的独立检测，输出分别为(, 13, 13, 255)和(, 26, 26, 255),网络结构见下图：
> 
> ![yolo_v3_tiny](https://mmbiz.qpic.cn/mmbiz_png/SdQCib1UzF3udtqwPpm1NLnLRNTabbFjLJL4ymY0Ltgv3wbVGnVZrtLJBYgohUXlYm9t7RzW3u3CSSGoRwLMqYQ/640?wx_fmt=png&tp=webp&wxfrom=5&wx_lazy=1&wx_co=1)

## YOLOv4

模型结构：

仔细阅读[YOLOv4模型](./kms/YOLOv4.km)

## 通过聚类得到anchor boxes

这是yolo和faster rcnn的一个显著的区别，聚类代码可参考GiantPandaCV《我们是如何改进YOLOv3进行红外小目标检测的？》一文。

这么做是否可以提升检测呢？

> 路人甲：聚类完更差
> 
> 路人乙：聚类了收敛比较快，性能提升一丢丢


# 7 目标检测思维导图

仔细阅读[检测器分类](./kms/detector.km)

仔细阅读[目标检测提升方法](./kms/detector_improve.km)
