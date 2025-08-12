---
title: VLMo 模型代码解读
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-12
author:
  - BinaryOracle
---

`VLMO 模型代码解读` 

<!-- more -->

> 论文链接: [VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://arxiv.org/abs/2111.02358)
> 代码链接: [https://github.com/microsoft/unilm/tree/master/vlmo](https://github.com/microsoft/unilm/tree/master/vlmo)

## 前置知识

VLMO 模型的代码实现中主要使用了以下两个库，如果不提前了解一下库的基本用法，可能会导致读不懂代码实现：

1. [Sacred 实验管理框架](https://sacred.readthedocs.io/en/stable/quickstart.html)

2. [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

## 数据集



