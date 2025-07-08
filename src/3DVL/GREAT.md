---
title: GREAT 论文解读 
icon: file
category:
  - 3D-VL
  - 3D Affordance
tag:
  - 3D-VL
  - 3D Affordance
  - 编辑中
footer: 技术共建，知识共享
date: 2025-06-15
cover: assets/cover/GREAT.png
author:
  - BinaryOracle
---

`GREAT: Geometry-Intention Collaborative Inference for Open-Vocabulary 3D Object Affordance Grounding 论文解读` 

<!-- more -->

> 论文: [https://arxiv.org/abs/2411.19626](https://arxiv.org/abs/2411.19626)
> 代码: [https://github.com/yawen-shao/GREAT_code](https://github.com/yawen-shao/GREAT_code)
> 数据集: [https://drive.google.com/drive/folders/1n_L_mSmVpAM-1ASoW2T2MltYkaiA_X9X](https://drive.google.com/drive/folders/1n_L_mSmVpAM-1ASoW2T2MltYkaiA_X9X)


## 摘要

GREAT（Geometry-Intention Collaborative Inference）是一种新颖的框架，旨在通过挖掘物体的不变几何属性和潜在交互意图，以开放词汇的方式定位3D物体的功能区域（affordance）。该框架结合了多模态大语言模型（MLLMs）的推理能力，设计了多头部功能链式思维（MHACoT）策略，逐步分析交互图像中的几何属性和交互意图，并通过跨模态自适应融合模块（CMAFM）将这些知识与点云和图像特征结合，实现精准的3D功能定位。此外，研究还提出了目前最大的3D功能数据集PIADv2，包含15K交互图像和38K标注的3D物体实例。实验证明了GREAT在开放词汇场景下的有效性和优越性。

## 简介

Open-Vocabulary 3D对象功能定位（OVAG）旨在通过任意指令定位物体上支持特定交互的“动作可能性”区域，对机器人感知与操作至关重要。现有方法（如[IAGNet](https://arxiv.org/abs/2303.10437)、[LASO](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_LASO_Language-guided_Affordance_Segmentation_on_3D_Object_CVPR_2024_paper.pdf)）通过结合描述交互的图像或语言与3D几何结构引入外部先验，但存在以下局限性（如图1(b)所示）：  

- **语义空间受限**：依赖预定义类别，难以泛化到未见过的功能（如将“pour”错误分类为“grasp”）。  
               
- **几何与意图利用不足**：未充分挖掘物体间共享的几何不变性（如手柄的抓握属性）和同一物体的多交互意图关联。  

![](GREAT/1.png)

**人类认知启发**:

研究表明（[Gick & Holyoak, 1980](https://www.sciencedirect.com/science/article/abs/pii/0010028580900134)），人类通过多步推理和类比思维解决复杂任务。例如，观察倒水场景时（图1(c)），人类会：  

1. 识别交互部件（壶嘴）  

2. 提取几何属性（倾斜曲面）  

3. 推理潜在意图（倒水/注水）  

**方法创新**:  

GREAT框架通过以下设计模拟这一过程（图1(d)）：  

1. **MHACoT推理链**：基于微调的MLLM（如[InternVL](https://arxiv.org/abs/2404.16821)）分步推理：  

   - **Object-Head**：定位交互部件并分析几何结构（如“为什么壶嘴适合倒水”）  

   - **Affordance-Head**：描述实际交互（如“握柄倒水”）并联想潜在意图（如“注水/清洗”）  

2. **跨模态融合**：通过CMAFM模块将几何属性（$\mathbf{\hat{T}}_o$）与交互意图（$\mathbf{\hat{T}}_a$）注入点云（$\mathbf{F}_{tp}$）和图像特征（$\mathbf{F}_{ti}$），最终解码为3D功能热图 $\phi = \sigma(f_\phi(\mathbf{F}_\alpha))$。  

**数据集贡献**:

扩展构建了**PIADv2**（对比见表1）：  

- **规模**：15K交互图像（×3）和38K 3D实例（×5）  

- **多样性**：43类物体、24类功能，覆盖多对多关联（图3(c)）  

![](GREAT/2.png)

![](GREAT/3.png)

## 相关工作

**1. Affordance Grounding**  

现有研究主要从2D数据（如图像、视频）和自然语言理解出发，定位“动作可能性”区域。例如，部分工作通过语言理解在2D数据中定位功能区域（[3](https://arxiv.org/abs/2405.12461), [21](https://arxiv.org/abs/2311.17776)），但机器人操作需要3D信息，2D方法难以直接迁移。随着3D数据集（如[5](https://arxiv.org/abs/2212.08051), [6](https://arxiv.org/abs/2103.16397)）的出现，部分研究开始映射语义功能到3D结构，但受限于预定义类别，无法处理开放词汇场景。  

**2. Open-Vocabulary 3D Affordance Grounding (OVAG)**  

OVAG旨在通过额外指令（如文本或图像）引入交互先验，提升泛化能力。例如：  

- IAGNet  利用2D交互语义指导3D功能定位；  

- LASO 通过文本条件查询分割功能区域；  

- OpenAD 和 OpenKD 利用CLIP编码器实现文本-点云关联。  

这些方法仍受限于训练语义空间，而GREAT通过几何-意图协同推理（CoT）解决此问题（如表2所示）。 

![](GREAT/4.png)

**3. Chain-of-Thought (CoT) 与多模态大模型 (MLLMs)**  

CoT及其变体通过多步推理增强MLLMs能力。例如：  

- 视觉任务中，MLLMs（如InternVL）结合CoT在目标检测、机器人操作等任务中表现优异；  

- 但动态功能特性使得MLLMs难以直接从交互图像推理3D功能，GREAT通过微调MLLMs并设计MHACoT策略解决这一问题。  

**关键问题**（如图1所示）：  

- 现有方法依赖数据对齐，泛化性不足（如将“pour”误分类为“grasp”）；  

- GREAT通过模拟人类多步推理（几何属性提取+意图类比）实现开放词汇功能定位。

## 方法

**1. 框架概述**  

![](GREAT/5.png)

GREAT 的输入为点云 $P \in \mathbb{R}^{N \times 4}$（含坐标 $P_c$ 和功能标注 $P_{label}$）和图像 $I \in \mathbb{R}^{3 \times H \times W}$，输出为3D功能区域 $\phi = f_\theta(P_c, I)$。整体流程（如图2所示）包括：  

- 通过 ResNet [9](https://arxiv.org/abs/1512.03385) 和 PointNet++ [43](https://arxiv.org/abs/1706.02413) 提取特征 $\mathbf{F}_i$ 和 $\mathbf{F}_p$；  

- 利用微调的 MLLM（InternVL [4](https://arxiv.org/abs/2404.16821)）进行多步推理（MHACoT）；  

- 通过 Cross-Modal Adaptive Fusion Module (CMAFM) 融合几何与意图知识；  

- 解码器联合预测功能区域 $\phi$，损失函数为 $\mathcal{L}_{total} = \mathcal{L}_{focal} + \mathcal{L}_{dice}$。  

**2. 多步推理（MHACoT）**  

分为两部分：  

- **Object-Head Reasoning**：  

  - 交互部件定位（提示：“*Point out which part interacts...*”）；  

  - 几何属性推理（提示：“*Explain why this part can interact...*”），生成特征 $\mathbf{T}_o$。  

- **Affordance-Head Reasoning**：  

  - 交互过程描述（提示：“*Describe the interaction...*”）；  

  - 潜在意图类比（提示：“*List two additional interactions...*”），生成特征 $\mathbf{T}_a$。  

通过 Roberta [28](https://arxiv.org/abs/1907.11692) 编码后，交叉注意力对齐特征：  

$$\bar{\mathbf{T}}_o = f_\delta(f_m(\mathbf{T}_o, \mathbf{T}_a)), \quad \bar{\mathbf{T}}_a = f_\delta(f_m(\mathbf{T}_a, \mathbf{T}_o))$$  

**3. 跨模态融合（CMAFM）**  

- 将几何知识 $\bar{\mathbf{T}}_o$ 注入点云特征 $\mathbf{F}_p$：  

  - 投影为 $\mathbf{Q}, \mathbf{K}, \mathbf{V}$，计算交叉注意力 $\mathbf{F}'_p$（公式2）；  

  - 通过全连接层和卷积融合，得到 $\mathbf{F}_{tp}$。  

- 将意图知识 $\bar{\mathbf{T}}_a$ 直接与图像特征 $\mathbf{F}_i$ 拼接，得到 $\mathbf{F}_{ti}$。  

**4. 解码与输出**  

融合特征 $\mathbf{F}_\alpha = f[\Gamma(\mathbf{F}_{ti}), \mathbf{F}_{tp}]$ 通过解码器生成功能热图 $\phi = \sigma(f_\phi(\mathbf{F}_\alpha))$，其中 $\sigma$ 为 Sigmoid 函数。  

**关键设计**（如图5所示）：  

- **几何-意图协同**：MHACoT 同时建模物体属性和交互意图，提升开放词汇泛化性；  

- **动态融合**：CMAFM 自适应对齐点云与图像模态，避免特征偏差（如表3消融实验所示）。