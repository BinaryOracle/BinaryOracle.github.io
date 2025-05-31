---
icon: file
category:
  - MMLLM
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-05-25
cover: assets/cover/BLIP2.png
author:
  - BinaryOracle
---

`庖丁解牛BLIP2` 

<!-- more -->

# 庖丁解牛BLIP2

> 论文: [https://arxiv.org/abs/2301.12597](https://arxiv.org/abs/2301.12597)
> 代码: [https://github.com/salesforce/LAVIS/tree/main/projects/blip2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)

## 背景

多模态模型在过往发展的过程中，曾有一段时期一直在追求更大的网络架构（image encoder 和 text encoder/decoder）和 数据集，从而导致更大的训练代价。例如CLIP，400M数据，需要数百个GPU训练数十天，如何降低模型训练成本，同时具有很好的性能？

这就是BLIP-2的起因，回顾下之前的多模态网络设计，三个模块（图像分支、文本分支、融合模块）:

![多模态网络设计](庖丁解牛BLIP2/1.png)

(a) 早期的图文多模态：图像分支依赖目标检测器，模态融合比较弱，如VSE++。

(b) 重点训练图像和文本特征提取，模态融合比较轻量，如CLIP。

(c) 图像特征提取和模态融合都很重。

(d) 侧重模态融合，特征提取网络相对轻量，如ViLT。

| 模块         | (a) | (b) | (c) | (d) | 理想情况 |
|--------------|-----|-----|-----|-----|----------|
| 视觉分支     | 重  | 重  | 重  | 轻  | 重       |
| 文本分支     | 轻  | 重  | 轻  | 轻  | 重       |
| 融合模块     | 轻  | 轻  | 重  | 重  | 轻       |
| 性能         | 一般| 好  | 好  | 一般| 好       |
| 训练代价     | 中  | 非常高 | 非常高 | 高 | 中   |

BLIP-2 基于 BLIP 架构，利用已有的ViT 和 LLM（均冻结）+ 一个的轻量Q-Former模块做模态融合，大幅降低训练成本。具有很强的zero-shot image-to-text generation能力，同时因LLM而具有了视觉推理能力。

## 模型结构

BLIP-2 框架按照 Two-Stage 策略预训练轻量级查询 Transformer 以弥合模态差距。

Stage 1: 不同模态数据的提取与融合。       Stage 2: 把数据转换成LLM能识别的格式。

![Two-Stage流程](庖丁解牛BLIP2/2.png)

从冻结的Image Encoder引到Vision-Language表征学习。   从冻结的LLM引到Vision-Language生成学习，实现Zero Shot图文生成。


