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

## 方法

### 预训练阶段一:  向量量化知识蒸馏算法用于d-VAE预训练

![](beit2/1.png)

BEIT V2 继承了 BEIT 的掩码图像建模（Masked Image Modeling）框架，其核心思想是将每张图像通过视觉 tokenizer 转换为一组离散的视觉 token，然后训练模型去恢复被遮挡的 token。每个 token 对应图像中的一个 patch，从而实现对局部图像信息的建模（如图 2 所示）。训练过程中，引入了向量量化知识蒸馏（VQ-KD）算法，用于训练视觉 tokenizer，使其能够有效将图像映射到离散编码。

图像表示部分，输入图像 $x \in R^{H \times W \times C}$ 会被划分为 $N = HW / P^2$ 个 patch ${x_p^i}_{i=1}^N$，每个 patch 大小为 $(P, P)$，在实验中 224 × 224 图像被划分为 14 × 14 个 patch，每个 patch 16 × 16。所有 patch 展平并线性映射得到 Transformer 的输入嵌入 ${h_i}_{i=1}^N$，用于后续编码。

在 VQ-KD 训练中，视觉 tokenizer 由编码器和量化器组成：

* 编码器将图像转换为 patch 表征 $h_i$；

* 量化器在代码本 $V \in R^{K \times D}$ 中查找每个 $h_i$ 的最近邻进行量化，得到离散 token $z_i$，公式为：

$$
z_i = \arg\min_j ||\hat{h}_i - \hat{v}_j||_2, \quad j \in \{1,2,\dots,K\}
$$

其中 $\hat{\cdot}$ 表示 $L_2$ 归一化，等价于基于余弦相似度查找最近代码。量化后的 $L_2$ 归一化代码 ${\hat{v}_{z_i}}$ 输入解码器，解码器输出 ${o_i}$ 尝试重建教师模型（如 DINO 或 CLIP）的语义特征 $t_i$。训练目标最大化 decoder 输出与教师特征的余弦相似度，同时通过 stop-gradient 机制处理量化不可导问题，梯度从 decoder 输入传递到 encoder 输出。训练目标公式为：

$$
\max \sum_{x \in D} \sum_{i=1}^N \cos(o_i, t_i) - ||sg[\hat{h}_i] - \hat{v}_{z_i}||_2^2 - ||\hat{h}_i - sg[\hat{v}_{z_i}]||_2^2
$$

其中 $sg[\cdot]$ 表示停止梯度操作，$D$ 为训练图像数据集。

向量量化训练中常见问题是代码本塌陷（codebook collapse），即只使用少量编码。为缓解此问题，VQ-KD 使用经验策略：

* 查找最近邻时对代码本嵌入进行 $L_2$ 归一化，并将维度降至 32；

* 在输入 decoder 前将低维嵌入映射回高维空间；

* 代码本嵌入使用指数移动平均（EMA）更新，EMA 能更稳定地追踪模型训练动态。

整体而言，BEIT V2 结合视觉 tokenizer、VQ-KD 和 Transformer 架构，通过 patch 级别的离散表示学习与教师特征对齐，实现对图像语义信息的高效编码与预训练。

## 代码解读

### 预训练阶段一: d-VAE 模型训练

![](beit2/1.png)

#### 码本_EMA

```python
class EmbeddingEMA(nn.Module):
    def __init__(self, num_tokens, codebook_dim, decay=0.99, eps=1e-5, 
                 kmeans_init=True, codebook_init_path=''):
        """
        向量量化的 codebook（码本）管理类，采用 EMA（指数滑动平均）进行更新。

        参数:
        - num_tokens: 码本的向量个数（即字典大小）
        - codebook_dim: 每个向量的维度
        - decay: EMA 的衰减系数
        - eps: 避免数值错误的小常数
        - kmeans_init: 是否使用 k-means 初始化
        - codebook_init_path: 若提供，则从已有 checkpoint 加载初始化码本
        """
        super().__init__()
        self.num_tokens = num_tokens
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps 
        
        # ========== 初始化权重 ==========
        if codebook_init_path == '':   # 如果没有提供预训练的 codebook
            if not kmeans_init:
                # 随机初始化，并做 L2 归一化，保证每个 embedding 向量长度为 1
                weight = torch.randn(num_tokens, codebook_dim)
                weight = l2norm(weight)
            else:
                # 若选择 kmeans_init，则先用全零矩阵占位，稍后再通过 k-means 初始化
                weight = torch.zeros(num_tokens, codebook_dim)

            # 标记是否完成初始化（True=已初始化，False=未初始化）
            self.register_buffer('initted', torch.Tensor([not kmeans_init]))
        else:
            # 如果给定路径，则直接加载预训练的 codebook 权重
            print(f"load init codebook weight from {codebook_init_path}")
            codebook_ckpt_weight = torch.load(codebook_init_path, map_location='cpu')
            weight = codebook_ckpt_weight.clone()
            self.register_buffer('initted', torch.Tensor([True]))
            
        # ========== 需要维护的参数 ==========
        # codebook 权重（不参与梯度更新，使用 EMA 更新）
        self.weight = nn.Parameter(weight, requires_grad = False)

        # 每个 cluster 的大小（计数），用来做 EMA 更新
        self.cluster_size = nn.Parameter(torch.zeros(num_tokens), requires_grad = False)

        # 每个 cluster embedding 的均值，用于 EMA 更新
        self.embed_avg = nn.Parameter(weight.clone(), requires_grad = False)

        # 是否启用更新
        self.update = True

    @torch.jit.ignore
    def init_embed_(self, data):
        """
        用 k-means 对码本进行初始化。
        - data: encoder 输出的样本数据 (N, D)
        """
        if self.initted:   # 若已初始化，则跳过
            return
        
        print("Performing K-means init for codebook")

        # 调用 kmeans 获取初始的 cluster 中心和 cluster 大小
        embed, cluster_size = kmeans(data, self.num_tokens, 10, use_cosine_sim = True)

        # 更新权重和 cluster_size
        self.weight.data.copy_(embed)
        self.cluster_size.data.copy_(cluster_size)

        # 设置为已初始化状态
        self.initted.data.copy_(torch.Tensor([True]))
```

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

k-means 的计算步骤可以总结为以下几个核心环节：

1. **初始化簇中心**

   * 从样本中随机选取 `num_clusters` 个向量作为初始中心，或者使用其他方法（如 k-means++）。

2. **计算样本与中心的距离/相似度**

   * 对每个样本计算它与所有簇中心的距离（欧氏距离）或相似度（余弦相似度）。

3. **样本分配**

   * 将每个样本分配到最近的簇（或相似度最高的簇），形成簇成员集合。

4. **统计簇信息**

   * 统计每个簇的样本数量（用于更新中心和处理空簇）。

5. **更新簇中心**

   * 对每个簇，将簇内样本向量求平均，得到新的中心。

   * 若某簇为空，则保留原中心不变。
   
   * 如果使用余弦相似度，更新后的中心需要做 L2 归一化。

6. **迭代**

   * 重复步骤 2–5，直到达到预定迭代次数或收敛条件。

7. **输出结果**

   * 返回最终的簇中心和每个簇的样本数。

这整个过程就是 k-means 聚类的标准迭代流程：**分配 → 更新 → 循环**。

```python
def kmeans(samples, num_clusters, num_iters = 10, use_cosine_sim = False):
    # samples: 输入样本，形状 (N, D)，N 是样本数，D 是维度
    # num_clusters: 聚类簇数，即要分成多少类
    # num_iters: k-means 的迭代次数
    # use_cosine_sim: 是否用余弦相似度（默认用欧氏距离）

    # 提取样本维度、数据类型和设备
    dim, dtype, device = samples.shape[-1], samples.dtype, samples.device

    # 从样本中随机选取 num_clusters 个向量作为初始中心
    means = sample_vectors(samples, num_clusters)

    # 重复迭代更新聚类中心
    for _ in range(num_iters):
        if use_cosine_sim:
            # 使用余弦相似度：直接点积即可（因为向量一般做过 l2norm）
            # 结果 shape: (N, K)，表示每个样本和每个中心的相似度
            dists = samples @ means.t()
        else:
            # 使用欧氏距离： (x - μ)^2
            # diffs: (N, 1, D) - (1, K, D) = (N, K, D)
            diffs = rearrange(samples, 'n d -> n () d') \
                    - rearrange(means, 'c d -> () c d')
            # 计算平方距离并取负号（因为后面要用 max 来找最近中心）
            dists = -(diffs ** 2).sum(dim = -1)   # shape: (N, K)

        # 找到每个样本最近的中心（或相似度最大的中心）
        # buckets: (N,) 每个样本对应的簇编号
        buckets = dists.max(dim = -1).indices

        # 统计每个簇的样本数量
        bins = torch.bincount(buckets, minlength = num_clusters)  # (K,)
        # 标记哪些簇没有分配到样本（空簇）
        zero_mask = bins == 0
        # 防止除以 0，把空簇的计数临时设为 1
        bins_min_clamped = bins.masked_fill(zero_mask, 1)

        # 初始化新的簇中心 (K, D)，全部为 0
        new_means = buckets.new_zeros(num_clusters, dim, dtype = dtype)
        # 把属于同一簇的样本向量加到对应的中心上
        # repeat(buckets, 'n -> n d', d = dim): 把 (N,) 扩展成 (N, D)，方便 scatter_add
        new_means.scatter_add_(0, repeat(buckets, 'n -> n d', d = dim), samples)
        # 除以该簇的样本数，得到新的簇中心
        new_means = new_means / bins_min_clamped[..., None]

        # 如果用余弦相似度，记得对中心做 l2norm 归一化
        if use_cosine_sim:
            new_means = l2norm(new_means)

        # 更新簇中心：
        # - 如果该簇是空簇（zero_mask=True），保留旧的中心
        # - 否则更新为新的中心
        means = torch.where(zero_mask[..., None], means, new_means)

    # 返回最终的簇中心和每个簇的样本数
    return means, bins
```

在 **量化前做 L2 归一化**，主要有几个原因：

1. **避免数值尺度差异**

   * 原始特征向量 $z$ 可能不同维度、不同样本之间的数值范围差异很大。
  
   * 如果直接计算欧氏距离，大的数值范围会主导距离计算，导致不公平。
  
   * L2 归一化后，每个向量都被缩放到单位长度（模长 = 1），使得比较时主要依赖 **方向差异** 而不是数值大小。

2. **提高稳定性**

   * 在训练过程中，如果向量的范数变化剧烈，会导致距离计算不稳定，进而影响 codebook 的更新。
  
   * 归一化可以避免过大的梯度和数值爆炸，稳定 EMA 更新。

```python
class NormEMAVectorQuantizer(nn.Module):
    def __init__(self, n_embed, embedding_dim, beta, decay=0.99, eps=1e-5, 
                 statistic_code_usage=True, kmeans_init=False, codebook_init_path=''):
        """
        基于 EMA（Exponential Moving Average）的向量量化器（Vector Quantizer）
        用于 VQ-VAE 或类似模型。
        
        参数：
        - n_embed: 码本中向量的数量（token 数量）
        - embedding_dim: 每个码本向量的维度
        - beta: 重构损失中的量化损失系数
        - decay: EMA 更新衰减系数
        - eps: 防止除零的小常数
        - statistic_code_usage: 是否统计每个码本向量的使用频率
        - kmeans_init: 是否使用 K-means 初始化码本
        - codebook_init_path: 初始化码本的路径
        """
        super().__init__()
        
        # 保存码本的维度和 token 数量
        self.codebook_dim = embedding_dim
        self.num_tokens = n_embed
        self.beta = beta
        self.decay = decay
        
        # 使用 EMA 的可学习嵌入表
        # EmbeddingEMA 内部会实现 EMA 更新和量化逻辑
        self.embedding = EmbeddingEMA(
            self.num_tokens, 
            self.codebook_dim, 
            decay, 
            eps, 
            kmeans_init, 
            codebook_init_path
        )
        
        # 是否统计码本向量的使用频率
        self.statistic_code_usage = statistic_code_usage
        if statistic_code_usage:
            # cluster_size 用于记录每个码本向量被使用的次数
            # register_buffer 不会被认为是可训练参数，但会随模型一起保存/加载
            self.register_buffer('cluster_size', torch.zeros(n_embed))
        
        # 分布式训练支持
        # 如果当前环境支持分布式训练并已初始化，则使用 all_reduce 同步各 GPU 的码本使用统计
        if distributed.is_available() and distributed.is_initialized():
            print("ddp is enable, so use ddp_reduce to sync the statistic_code_usage for each gpu!")
            self.all_reduce_fn = distributed.all_reduce
        else:
            # 单 GPU 或未初始化分布式训练时，直接使用 Identity（不做任何操作）
            self.all_reduce_fn = nn.Identity()
```

```python
def forward(self, z):
    """
    前向传播函数，实现向量量化（Vector Quantization）和 EMA 更新

    参数:
    - z: 输入特征图, shape (batch, channel, height, width)
    
    返回:
    - z_q: 量化后的特征图，shape 同输入
    - loss: 量化损失
    - encoding_indices: 每个向量对应的码本索引
    """
    
    # 将输入从 (B, C, H, W) 转换为 (B, H, W, C) 以便处理通道维
    z = rearrange(z, 'b c h w -> b h w c')
    
    # L2 归一化
    z = l2norm(z)
    
    # 展平特征图，每一行对应一个向量 (num_vectors, embedding_dim)
    z_flattened = z.reshape(-1, self.codebook_dim)
    
    # 初始化码本（如果需要）
    self.embedding.init_embed_(z_flattened)
    
    # 计算每个向量与码本中所有向量的欧氏距离平方
    d = z_flattened.pow(2).sum(dim=1, keepdim=True) + \
        self.embedding.weight.pow(2).sum(dim=1) - 2 * \
        torch.einsum('bd,nd->bn', z_flattened, self.embedding.weight)  # 'n d -> d n'
    
    # 为每个向量找到最近的码本索引
    encoding_indices = torch.argmin(d, dim=1)

    # 将编码索引映射回码本向量并 reshape 成原来的特征图形状
    z_q = self.embedding(encoding_indices).view(z.shape)
    
    # one-hot 编码
    encodings = F.one_hot(encoding_indices, self.num_tokens).type(z.dtype)     
    
    # 非训练模式下统计码本使用情况
    if not self.training:
        with torch.no_grad():
            cluster_size = encodings.sum(0)
            self.all_reduce_fn(cluster_size)  # 分布式同步
            ema_inplace(self.cluster_size, cluster_size, self.decay)
    
    # 训练模式下更新 EMA 码本
    if self.training and self.embedding.update:
        bins = encodings.sum(0)
        self.all_reduce_fn(bins)

        # 更新 cluster_size 的 EMA
        ema_inplace(self.cluster_size, bins, self.decay)

        # 避免除零
        zero_mask = (bins == 0)
        bins = bins.masked_fill(zero_mask, 1.)

        # 计算每个码本向量的累加特征
        embed_sum = z_flattened.t() @ encodings
        self.all_reduce_fn(embed_sum)

        # 归一化并 L2 正则化
        embed_normalized = (embed_sum / bins.unsqueeze(0)).t()
        embed_normalized = l2norm(embed_normalized)

        # 对未使用的码本向量保持原值
        embed_normalized = torch.where(zero_mask[..., None], self.embedding.weight,
                                       embed_normalized)
        # 更新 EMA 码本权重
        norm_ema_inplace(self.embedding.weight, embed_normalized, self.decay)

    # 量化损失
    loss = self.beta * F.mse_loss(z_q.detach(), z) 
    
    # 保留梯度
    z_q = z + (z_q - z).detach()

    # reshape 回原始输入形状 (B, C, H, W)
    z_q = rearrange(z_q, 'b h w c -> b c h w')
    
    return z_q, loss, encoding_indices
```