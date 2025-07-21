---
title: BLIP 论文
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-07-20
author:
  - BinaryOracle
---

`BLIP: Bootstrapping Language-Image Pre-training for  Unified Vision-Language Understanding and Generation 论文解读` 

<!-- more -->

> 论文链接: [BLIP: Bootstrapping Language-Image Pre-training for  Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)
> 代码链接: [https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP)

## Introduction


当前视觉-语言预训练（VLP）方法虽然在多模态任务上取得进展，但普遍存在两个问题：

1. **模型限制**：编码器模型不适合文本生成任务；编码器-解码器模型难以用于图文检索。
  
2. **数据质量差**：大多使用从网络收集的嘈杂图文对作为训练数据，监督信号不理想。


BLIP（Bootstrapping Language-Image Pre-training）是一个新颖的 VLP 框架，兼顾理解与生成能力。其两大创新点：

1. **MED 模型结构（Multimodal Mixture of Encoder-Decoder）**：

   * 同时支持编码器、图像条件编码器、图像条件解码器三种模式。

   * 联合训练三种任务：图文对比学习、图文匹配、图像条件语言建模。
   
   * 实现多任务预训练与灵活迁移。

2. **CapFilt 数据自举方法（Captioning and Filtering）**：

   * 使用训练好的 MED 模型构建两个模块：

     * 描述器（captioner）生成图像的合成描述；
   
     * 过滤器（filter）剔除原始和生成的低质量描述。
   
   * 在保留信息的同时提升训练数据质量。

实验结果与表现:

* BLIP 在多个任务（图文检索、图像描述、VQA 等）上取得**最先进性能**。

* 同时，在两个视频-语言任务上以**零样本方式**迁移也表现优异。

* 实验证明：描述器与过滤器的组合能显著提升性能，多样化描述更有利于学习。

## Related Work

### 视觉-语言预训练（VLP）

* **现状问题：**

  * 主流 VLP 方法依赖从网络抓取的图文对数据，虽然规模大，但包含大量噪声文本。
  
  * 尽管使用简单的过滤规则，噪声仍广泛存在。
  
  * 编码器模型适合理解类任务但难以生成文本；编码器-解码器适合生成任务但不适用于检索。

* **BLIP 的改进：**

  * 提出 **CapFilt**：通过“生成 + 过滤”的方式优化数据质量。
  
  * 提出 **MED 模型结构**：在保持预训练高效的前提下，同时兼顾理解与生成任务，提升泛化能力。

### 知识蒸馏（Knowledge Distillation）

* **现有做法：**

  * 知识蒸馏让小模型（学生）学习大模型（教师）的预测结果。
  
  * 自蒸馏也取得了不错效果，尤其在图像分类与部分 VLP 方法中已开始尝试。

* **BLIP 的新视角：**

  * CapFilt 可视为一种结构化的知识蒸馏方式：

    * **Captioner 模块**用生成的语义丰富描述进行蒸馏；
  
    * **Filter 模块**通过剔除噪声文本完成隐式知识过滤。

### 数据增强（Data Augmentation）

* **现有做法：**

  * 图像任务中数据增强广泛应用，但语言任务的数据增强较困难。
 
  * 近年来生成模型被用于文本任务的样本合成，但多用于低资源语言场景。

* **BLIP 的贡献：**

  * 展示了在**大规模视觉-语言预训练中**使用合成图像描述的独特优势，提升了多模态学习效果。


