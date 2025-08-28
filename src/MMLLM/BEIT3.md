---
title: BEIT3 论文
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-22
author:
  - BinaryOracle
---

`Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks 论文解读` 

<!-- more -->

> 论文链接: [Image as a Foreign Language: BEiT Pretraining for All Vision and Vision-Language Tasks](https://arxiv.org/abs/2208.10442)
> 代码链接: [https://github.com/microsoft/unilm/tree/master/beit3](https://github.com/microsoft/unilm/tree/master/beit3)

## 引言

近年来，**语言、视觉与多模态预训练**正在出现“大融合”的趋势。研究者发现，只要在**海量数据**上进行大规模预训练，就可以把模型轻松迁移到各种下游任务中。一个理想的方向是：**预训练一个通用基础模型，能够同时处理多种模态**。

BEiT-3 正是顺应这一趋势提出的，它在 **视觉任务**和 **视觉-语言任务**上都取得了最新的迁移性能；BEiT-3 核心贡献如下:

**1. 统一的骨干架构**

Transformer 的成功已经从 **语言** 扩展到了 **视觉** 和 **多模态**任务，这使得用统一网络结构来处理不同模态成为可能。不过，不同下游任务常常需要不同架构：

* **双编码器 (dual-encoder)**：用于高效检索（如跨模态检索）。

* **编码器-解码器 (encoder-decoder)**：用于生成任务（如图像描述）。

* **融合编码器 (fusion-encoder)**：用于图文联合表示学习。

问题在于：大多数基础模型需要针对不同任务手动调整网络格式，且不同模态之间的参数往往难以有效共享。

BEiT-3 引入了 **Multiway Transformer**（多路 Transformer），作为通用建模框架。它既能做模态特定的编码，也能实现跨模态深度融合，**做到“一套架构适配所有下游任务”**。

---

**2. 统一的预训练任务**

掩码建模（Masked Data Modeling）已在多种模态上取得成功：

* 文本（Masked Language Modeling, MLM）

* 图像（Masked Image Modeling, MIM）

* 图文对（Masked Multimodal Modeling）

现有视觉-语言基础模型通常需要 **多任务训练**（如图文匹配、对比学习），但这会导致扩展到大规模数据时效率低。

BEiT-3 的做法是：只保留 **单一任务**——**mask-then-predict（掩码预测）**。

* 把图像当作外语（Imglish），和文本用相同的方式建模。

* 图文对被看作“平行句子”，用来学习模态间的对齐关系。

这种方法虽然简单，却能学习到很强的可迁移表征，并在视觉与视觉-语言任务中取得了最新结果。

---

**3. 模型与数据的规模化**

扩大模型规模与数据规模，可以显著提高基础模型的泛化能力。

* BEiT-3 将模型扩展到了 **数十亿参数**级别。

* 预训练数据规模也被扩大，但仅使用 **公开数据集**，保证学术可复现性。

即使没有依赖私有数据，BEiT-3 依然超过了许多依赖私有大数据的基础模型。

此外，将图像当作外语的方式还能直接复用大规模语言模型的训练管线，从而在规模化上进一步受益。

---

BEiT-3 使用 Multiway Transformer，在 **图像、文本和图文对**上进行统一的掩码建模。

* 在训练中，会随机掩码部分文本 token 或图像 patch。

* 学习目标是恢复原始 token（文本 token 或视觉 token）。

这是一个标准的自监督学习任务，使模型在预训练阶段就能获得通用性。 BEiT-3 在多种任务上都取得了最新性能，包括：

* **视觉任务**：目标检测（COCO）、实例分割（COCO）、语义分割（ADE20K）、图像分类（ImageNet）

* **视觉-语言任务**：视觉推理（NLVR2）、视觉问答（VQAv2）、图像描述（COCO）、跨模态检索（Flickr30K、COCO）

结果显示：

* 即使只使用公开数据，BEiT-3 依然超越了许多依赖私有数据的强大模型。

* 它不仅在多模态任务上表现优异，在**纯视觉任务**中也能达到甚至超过专用模型的效果。
