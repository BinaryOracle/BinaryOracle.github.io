---
title: 3D Affordance Grounding 方向复盘
icon: file
category:
  - 3D-VL
tag:
  - 3D-VL
  - 编辑中
footer: 技术共建，知识共享
date: 2025-09-15
author:
  - BinaryOracle
---

`3D Affordance Grounding 方向复盘` 

<!-- more -->

## 点云 + 文本

### [Affogato (Arxiv 2025.06)](https://arxiv.org/abs/2506.12009)

特点:

1. AFFOrdance Grounding All aT Once

2. a large-scale dataset for 3D and 2D affordance grounding

3. minimalistic architecture

![](affordance_grounding复盘/1.png)

损失函数:

1. Focal Loss to handle class imbalance

2. Dice Loss to improve region-level alignment.

现状:

1. wait for code release

2. dataset available

### [SeqAfford (CVPR 2025)](https://arxiv.org/abs/2412.01550)

特点:

1. Propose a 3D multimodal large language model (referring to the LLaVA model architecture)

2. Feed the `<SEG>` segmentation tokens output by the 3D MMLLM into the multi-granularity language-point cloud combination module to complete 3D dense prediction

3. Support sequential instruction execution

4. Large-scale instruction-point cloud pair dataset: A dataset with 180,000 instruction-point cloud pairs, covering single and sequential operability reasoning tasks

![](affordance_grounding复盘/4.png)

损失函数:

1. Autoregressive Cross-Entropy Loss

2. Dice Loss

3. Binary Cross-Entropy Loss

现状:

1. code available

2. dataset available


## 点云 + 图像

## 点云 + 文本 + 图像

### [GREAT (CVPR 2025)](https://arxiv.org/abs/2411.19626)

特点:

1. grounding 3D object affordance in an Open-Vocabulary fashion

2. Multi-Head Affordance Chain-of-Thought

> Data preparation stage: 
> 
>  1. Use prompts to generate descriptions of the object interaction area, the morphology(形态学) of the interaction area, the interaction behavior, and other common interaction behaviors of the object.
>
>  2. Geometric structure knowledge = Answers to Prompt 1 + Prompt 2 = Interaction parts + Inference of geometric properties of these parts
> 
>  3. Interaction knowledge = Answers to Prompt 3 + Prompt 4 = Current interaction + Analogous(类似的)/supplementary(补充) interaction methods

3. PIADv2 dataset 

> 24 affordance ,  43 object categories, 15K interaction images , 38K 3D objects with annotations.

![](affordance_grounding复盘/2.png)

损失函数:

1. Focal Loss to handle class imbalance

2. Dice Loss to improve region-level alignment.

现状:

1. code available

2. dataset available

### [LMAffordance3D (CVPR 2025)](https://arxiv.org/abs/2504.04744)

特点:

1. Combine language instructions, visual observations, and interaction information to locate the affordance of manipulable objects in 3D space.

2. AGPIL（Affordance Grounding dataset with Points, Images and Language instructions）

> This dataset includes estimations of object affordances observed from full-view, partial-view, and rotated perspectives, taking into account factors such as real-world observation angles, object rotation, and spatial occlusion (遮挡).

![](affordance_grounding复盘/5.png)

损失函数:

1. focal loss

2. dice loss


现状:

1. The code and data are closed-source

## 3D Gaussian Splatting (3DGS)

### [GEAL (CVPR 2025)](https://arxiv.org/abs/2412.09511)

特点:

1. "Knowledge Distillation" from 2D to 3D: Transfer the semantic capabilities of pre-trained 2D models to the 3D affordance prediction model through Gaussian splat mapping, cross-modal consistency alignment, and multi-scale fusion.

2. Noisy Dataset: Construct a new benchmark with multiple types of noise/damage to evaluate the generalization and robustness of the model under real/harsh conditions.

![](affordance_grounding复盘/3.png)

损失函数:

1. BCE

2. Dice Loss

3. Consistency Loss（MSE 损失）

现状:

1. wait for code release

2. wait for dataset release

### [3DAffordSplat (Arxiv 2025.04)](https://arxiv.org/abs/2504.11218)

### [IAAO (CVPR 2025)](https://arxiv.org/abs/2504.06827)


## idea

Momentum Encoder 生成伪标签应对噪声问题，实现更加稳健的学习 ？(参考: MoCo , ALBEF , DINO)



