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