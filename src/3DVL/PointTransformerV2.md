---
title: Point Transformer V2 论文
icon: file
category:
  - 3D-VL
tag:
  - 3D-VL
  - 编辑中
footer: 技术共建，知识共享
date: 2025-09-07
author:
  - BinaryOracle
---

`Point Transformer V2 论文` 

<!-- more -->

> 论文: [Point Transformer V2: Grouped Vector Attention and Partition-based Pooling](https://arxiv.org/abs/2210.05666)
> 代码: [https://github.com/Pointcept/PointTransformerV2](https://github.com/Pointcept/PointTransformerV2)

## 引言

Point Transformer (PTv1) 首次将自注意力网络引入三维点云理解，并结合 **向量注意力** 与 **U-Net 风格的编码器-解码器框架**，在分类、分割等任务上取得了显著成绩。但其仍存在一些不足：

* 向量注意力的权重编码依赖 MLP，当模型加深、通道数增加时，参数量急剧膨胀，容易导致严重过拟合并限制模型深度。

* 三维点云的位置信息比二维像素更关键，但已有方法大多借鉴二维的编码方式，未能充分利用三维坐标中的几何特性。

* 点云的不规则分布给池化带来挑战，以往方法依赖采样（如最远点采样、网格采样）与邻域查询（如 kNN、半径查询）的结合，既耗时又缺乏良好的空间对齐。

---

**提出的方法**

作者提出了新的 **Point Transformer V2 (PTv2)**，在多个方面改进了 PTv1：

* **分组向量注意力（Grouped Vector Attention, GVA）**

  将向量注意力划分为多个组，每组共享注意力权重，从而减少参数量，提升效率。
  GVA 同时包含了 **多头注意力** 与 **向量注意力** 的优势，并且二者都可以看作是 GVA 的特例。

* **改进的位置编码机制**

  在关系向量中额外引入 **位置编码乘子**，强化三维点的空间关系，使模型更好地利用点云的几何信息。

* **基于分区的池化策略**

  将点云划分为 **互不重叠的分区**，并直接在同一区域内融合点信息，避免了传统方法对采样和邻域查询的依赖，实现了更高效、更精准的空间对齐。


