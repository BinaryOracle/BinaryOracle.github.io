---
title: API记录之框架篇
icon: file
category:
  - tools
tag:
  - 已发布
footer: 技术共建，知识共享
date: 2025-08-22
author:
  - BinaryOracle
---

`API记录之框架篇` 

<!-- more -->

## timm 库

`timm` 是 **PyTorch Image Models** 的缩写，是 Ross Wightman 开发和维护的一个 **PyTorch 视觉模型库**，在计算机视觉领域非常常用。它在科研与工业界都很受欢迎，因为它集合了大量常见与前沿的图像模型，同时提供了高质量的实现和训练权重。

**特点**:

1. **模型丰富**

   * 收录了数百种视觉模型，包括：

     * 经典模型：ResNet、DenseNet、EfficientNet、MobileNet

     * Transformer 系列：ViT、DeiT、Swin Transformer、ConvNeXt

     * 最新论文模型：EVA、ConvNeXt V2、MaxViT 等

   * 你几乎可以把它当成 **视觉模型的“模型仓库”**。

2. **预训练权重**

   * 提供了大量在 **ImageNet-1k / ImageNet-21k** 上训练好的权重，开箱即用。

   * 可以直接加载预训练模型用于 **迁移学习 / finetune**。

3. **统一接口**

   * 使用简单，几乎所有模型都能通过同样的方式调用：

     ```python
     import timm
     model = timm.create_model('resnet50', pretrained=True)
     x = torch.randn(1, 3, 224, 224)
     y = model(x)
     ```

   * API 统一，降低了不同架构之间的切换成本。

4. **实用工具**

   * `timm.data`：包含数据增强（RandAugment、Mixup、CutMix 等）。
 
   * `timm.optim`：包含优化器（AdamP、RAdam、Lookahead 等）。
 
   * `timm.scheduler`：学习率调度器（CosineAnnealing、OneCycle、TanhDecay 等）。
 
   * `timm.loss`：封装了多种损失函数（Label Smoothing、SoftTarget CrossEntropy 等）。
 
   * 这些设计让训练流程非常完整。

5. **高效实现**

   * 很多模型在 `timm` 里做了 **速度和显存优化**，常常比官方实现更高效。
 
   * 支持混合精度训练、channels-last 等特性。

### create_model 与 @register_model 装饰器

**`create_model`**：timm 提供的统一入口，用于按名字实例化模型。

```python
model = timm.create_model('resnet50', pretrained=True)
```

**`@register_model`**：用于将自定义模型注册到 timm 模型库，才能通过 `create_model` 调用。

```python
@register_model
def my_model(pretrained=False, **kwargs):
   return MyModel(**kwargs)
```

前者是**用**模型，后者是**加**模型。

