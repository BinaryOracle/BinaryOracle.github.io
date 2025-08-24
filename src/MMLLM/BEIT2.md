---
title: BEIT2 论文
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-17
author:
  - BinaryOracle
---

`BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers 论文解读` 

<!-- more -->

> 论文链接: [BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/abs/2208.06366)
> 代码链接: [https://github.com/microsoft/unilm/tree/master/beit2](https://github.com/microsoft/unilm/tree/master/beit2)

## 引言

掩码图像建模（MIM）通过恢复被掩码的图像块，能够在自监督学习中捕捉丰富的上下文信息，但大多数方法仅在低层像素上操作。

现有重建目标可以分为三类：

* 低层图像元素（如原始像素）

* 手工特征（如 HOG 特征）

* 视觉 token

这些方法大多忽略了高层语义信息，而语言模型中的掩码词都是高层语义，这启发了 MIM 可以借助语义感知监督进行改进。

![](beit2/1.png)

**BEIT V2** 提出 **向量量化知识蒸馏（VQ-KD）**，将连续的语义空间离散化为紧凑的视觉 token。VQ-KD 训练过程：

1. 编码器将输入图像转为离散 token，基于可学习码本（codebook）。

2. 解码器根据教师模型编码的语义特征重建图像特征。

训练完成后，VQ-KD 的编码器被用作 BEIT V2 的语义视觉分词器，离散 token 作为监督信号进行 MIM 预训练。引入 **图像块聚合策略**，让 \[CLS] token 聚合全局信息，解决传统 MIM 过度关注局部块重建而忽略全局表示的问题。

## 代码解读

### 预训练阶段一: VQ-VAE 模型训练

![](beit2/1.png)

#### 码本_EMA



#### 向量量化器

向量量化器负责将连续的视觉特征映射到离散的视觉 `token`，该过程借助内部维护的 `cookbook` 完成，本节我们先来详细解析一下它的实现逻辑:

```python
class NormEMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5, 
                statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        super().__init__()
        
        # codebook 向量的维度（即每个 embedding 的维数）
        self.codebook_dim = embedding_dim
        # codebook 的大小（有多少个离散 token）
        self.num_tokens = n_embed
        # commitment loss 的权重系数
        self.beta = beta
        # EMA 更新的衰减系数
        self.decay = decay
        
        # codebook，使用 EMA 更新（非梯度更新）
        # 这里的 EmbeddingEMA 类负责存储和更新 codebook 向量
        # 参数：
        # - num_tokens: codebook 的大小
        # - codebook_dim: 每个向量的维度
        # - decay, eps: EMA 更新超参
        # - kmeans_init: 是否用 k-means 初始化 codebook
        # - codebook_init_path: 是否从文件加载已有的 codebook
        self.embedding = EmbeddingEMA(
            self.num_tokens, 
            self.codebook_dim, 
            decay, 
            eps, 
            kmeans_init, 
            codebook_init_path
        )
        
        # 是否统计每个 code 的使用频率（防止 dead code）
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            # cluster_size 用来存储每个 code 的使用计数，注册为 buffer，随模型保存
            self.register_buffer('cluster_size', torch.zeros(n_embed))
```

```python
    @torch.jit.ignore
    def init_embed_(self, data):
        """
        使用 k-means 对 codebook 进行初始化。
        只会执行一次，之后 self.initted 会标记为 True。
        """
        if self.initted:
            return
        print("Performing Kmeans init for codebook")

        # 在输入数据 data 上运行 k-means
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim = True)

        # 把 k-means 得到的聚类中心赋值给 codebook
        self.weight.data.copy_(embed)
        # 把每个簇的样本数存下来
        self.cluster_size.data.copy_(cluster_size)
        # 标记为已初始化
        self.initted.data.copy_(torch.Tensor([True]))
```
