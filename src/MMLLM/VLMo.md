---
title: VLMo 论文
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-16
author:
  - BinaryOracle
---

`VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts 论文简析` 

<!-- more -->

> 论文链接: [VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://arxiv.org/abs/2111.02358)
> 代码链接: [https://github.com/microsoft/unilm/tree/master/vlmo](https://github.com/microsoft/unilm/tree/master/vlmo)

## Introduction

视觉-语言（VL）预训练旨在从大规模图文对中学习通用的跨模态表示。现有模型通常通过图文匹配、图文对比学习、掩码区域分类/特征回归、词-区域/块对齐以及掩码语言建模等方法来聚合和对齐视觉与语言信息，然后在下游任务如图文检索、视觉问答（VQA）、视觉推理等进行微调。

现有两类主流架构各有优缺点：

**双编码器架构**（Dual-Encoder，如 CLIP、ALIGN）：

* 图像和文本分别编码，模态间交互通过特征向量的余弦相似度进行。

* 优点：检索任务高效，特征向量可提前计算存储，线性复杂度。

* 缺点：交互浅，对复杂视觉-语言分类任务表现有限，如 CLIP 在视觉推理任务上准确率偏低。

**融合编码器架构**（Fusion-Encoder）：

* 使用多层 Transformer 通过跨模态注意力融合图像和文本表示。

* 优点：在视觉-语言分类任务上性能优异。

* 缺点：检索任务需对所有图文对联合编码，时间复杂度为二次方，推理速度慢。

---

**VLMO 的提出**

为兼顾双编码器和融合编码器的优势，论文提出了 **统一视觉-语言预训练模型 VLMO**，其特点如下：

* 可作为双编码器用于图文检索，也可作为融合编码器处理图文对分类任务。

* 核心组件为 **Mixture-of-Modality-Experts (MOME) Transformer**，一个 Transformer 块内可编码图像、文本及图文对。

* MOME 替换标准 Transformer 的前馈网络为模态专家池，捕获模态特定信息，同时共享自注意力层进行跨模态对齐。

* 三类模态专家：视觉专家（图像编码）、语言专家（文本编码）、视觉-语言专家（图文融合）。

* 模型灵活性高，可复用共享参数实现文本编码器、图像编码器和图文融合编码器。

---

**预训练任务与策略**

VLMO 采用三种联合预训练任务：

* 图文对比学习（image-text contrastive learning）

* 图文匹配（image-text matching）

* 掩码语言建模（masked language modeling）

同时提出 **分阶段预训练策略**，充分利用大规模图像单模态和文本单模态数据：

1. 在图像单模态数据上预训练视觉专家和自注意力模块，采用 BEIT 的掩码图像建模方法。

2. 在文本单模态数据上预训练语言专家，采用掩码语言建模方法。

3. 最终初始化视觉-语言预训练模型，解决图文对数量有限、描述短小的问题，从而学习更泛化的表示。

---

**实验结果与贡献**

* 在图文检索任务中，VLMO 作为双编码器比融合编码器更快，并且性能优于其他融合编码器模型。

* 在视觉问答（VQA）和自然语言视觉推理（NLVR2）任务中，作为融合编码器的 VLMO 达到最先进性能。

**主要贡献**：

* 提出统一视觉-语言预训练模型 VLMO，可灵活用作融合编码器或双编码器。

* 引入通用多模态 Transformer（MOME Transformer），通过模态专家捕获模态特定信息，并通过共享自注意力实现跨模态对齐。

* 分阶段预训练策略利用大规模图像单模态和文本单模态数据，显著提升模型性能。

## Related Work



