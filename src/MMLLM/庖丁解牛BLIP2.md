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

