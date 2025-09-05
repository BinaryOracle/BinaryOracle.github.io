---
title: Point Transformer 论文
icon: file
category:
  - 3D-VL
tag:
  - 3D-VL
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-31
author:
  - BinaryOracle
---

`Point Transformer 论文` 

<!-- more -->

> 论文: [Point Transformer](https://arxiv.org/abs/2012.09164)
> 代码: [https://github.com/POSTECH-CVLab/point-transformer](https://github.com/POSTECH-CVLab/point-transformer)

## 引言


## 代码实现

`PointTransformerLayer` 类实现了Point Transformer论文中提出的向量自注意力机制，该机制是点云处理领域的重要创新。与传统的标量注意力不同，向量注意力能够更好地捕获3D空间中的几何关系。

**核心设计理念**：

- **向量化注意力**：注意力权重不再是标量，而是向量形式，能够编码更丰富的空间信息

- **位置感知**：通过位置编码函数θ将3D相对坐标信息融入注意力计算

- **局部邻域处理**：基于KNN构建的局部邻域，实现高效的点云特征聚合

**算法流程概述**：

1. **特征变换**：将输入特征通过线性层生成Q、K、V三元组

2. **邻域构建**：利用KNN算法为每个点构建局部邻域

3. **位置编码**：将相对坐标通过MLP网络映射到高维特征空间

4. **注意力计算**：结合特征差值和位置编码生成向量化注意力权重

5. **特征聚合**：基于注意力权重对邻域特征进行加权融合

完整的向量自注意力计算代码实现如下：

```python
class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
        # 中间通道数，简化处理（这里直接等于 out_planes）
        self.mid_planes = mid_planes = out_planes // 1
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample

        # Q, K, V 的线性变换
        self.linear_q = nn.Linear(in_planes, mid_planes)   # 查询向量 (query)
        self.linear_k = nn.Linear(in_planes, mid_planes)   # 键向量 (key)
        self.linear_v = nn.Linear(in_planes, out_planes)   # 值向量 (value)

        # 位置编码 δ (论文 Eq.(4): δ = θ(pi − pj))
        # 输入是相对坐标 (3D)，输出是与 out_planes 对齐的特征
        self.linear_p = nn.Sequential(
            nn.Linear(3, 3),
            nn.BatchNorm1d(3),
            nn.ReLU(inplace=True),
            nn.Linear(3, out_planes)
        )

        # 权重生成函数 γ (MLP)，作用在 (q - k + δ) 上
        # 注意这里做了“通道分组”（share_planes），减少计算量
        self.linear_w = nn.Sequential(
            nn.BatchNorm1d(mid_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes, mid_planes // share_planes),
            nn.BatchNorm1d(mid_planes // share_planes),
            nn.ReLU(inplace=True),
            nn.Linear(mid_planes // share_planes, out_planes // share_planes)
        )

        # softmax 用来对注意力权重归一化
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo) -> torch.Tensor:
        # 输入:
        # p: 点的坐标 (n, 3)
        # x: 点的特征 (n, c)
        # o: batch 索引 (b)
        p, x, o = pxo

        # 得到 Q, K, V
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)  # (n, c)

        # 构建邻域 (kNN)，并返回局部邻域的特征
        # x_k: (n, nsample, 3+c)，包含相对坐标和 K 特征
        # x_v: (n, nsample, c)，邻域内的 V 特征
        x_k = pointops.queryandgroup(self.nsample, p, p, x_k, None, o, o, use_xyz=True)
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, None, o, o, use_xyz=False)

        # 分离相对坐标 p_r 和邻域内的 K 特征
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]

        # 将相对坐标 p_r 输入位置编码 MLP θ
        # 这里因为 BatchNorm 的维度问题，需要转置 (n, nsample, 3) ↔ (n, 3, nsample)
        for i, layer in enumerate(self.linear_p):
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)
        # 经过 MLP 后: (n, nsample, out_planes)

        # 根据 Eq.(3): w = γ(φ(xi) − ψ(xj) + δ)
        # x_q.unsqueeze(1): (n, 1, c)，与邻域对齐
        # p_r reshape 后与 x_k 对齐做相加
        w = x_k - x_q.unsqueeze(1) + p_r.view(
            p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes
        ).sum(2)  # (n, nsample, c)

        # 将 w 输入 γ MLP (linear_w)，得到注意力权重
        for i, layer in enumerate(self.linear_w):
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)

        # softmax 归一化注意力权重
        w = self.softmax(w)  # (n, nsample, c)

        # 最终聚合 (Eq.(3) 中 ρ(...)*α(xj+δ))
        n, nsample, c = x_v.shape
        s = self.share_planes
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)

        return x
```

下面针对上面部分代码进行进一步说明:

1. 计算注意力权重: 领域内最近邻键特征 - 领域所在中心点查询特征 + 相对位置编码(巧妙的view方式，个人理解是为了确保维度对齐)

```python 
        w = x_k - x_q.unsqueeze(1) + p_r.view(
            p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes, self.mid_planes
        ).sum(2)  # (n, nsample, c)
```

2. 聚合: 对每个中心点的所有邻居点的特征在特征维度上进行分组，做通道分组(类似多头注意力，但是作用不完全相同) + 利用广播后做逐元素相乘，完成对同一个邻居点的所有通道分组应用相同权重分配的过程 + 所有邻居点特征进行求和，完成领域值信息聚合过程 + 多头重组回原貌

```python
    # （200，8，8，4） * （200，8，1，4） -> (200, 8, 8, 4) -> (200,8,4) -> (200,32)
    x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
```

分组计算过程可参考如下这个简化版例子:

```python
# 分组后的特征 (1个点，1个邻居，2组，每组2个通道)
grouped_features = torch.tensor([[
    [[1.0, 2.0],    # 组0: 通道0,1
     [3.0, 4.0]]    # 组1: 通道2,3
]])  # shape: (1, 1, 2, 2)

# 注意力权重 (16维权重，这里简化为2维)
attention_weights = torch.tensor([[
    [[0.5, 1.0]]    # 权重向量
]])  # shape: (1, 1, 1, 2)

# 逐元素相乘
result = grouped_features * attention_weights
# [[[[1.0*0.5, 2.0*1.0],   # 组0: [0.5, 2.0]
#    [3.0*0.5, 4.0*1.0]]]] # 组1: [1.5, 4.0]
```

**这实际上是一种分组通道注意力（Grouped Channel Attention**）：

1. 不是在序列维度上做注意力（token-token）

2. 而是在通道维度上做注意力（channel-channel）

3. 通过分组实现参数共享

---

`queryandgroup` 方法实现了点云的邻域查询和特征分组功能。具体流程如下：

1. **邻域查询**：对于查询点集合中的每个点，利用 `KNN` 算法在所有点集合中寻找最近的 `nsample` 个邻居点，并返回这些邻居点的索引；

2. **相对坐标计算**：将每个查询点的邻居点坐标减去查询点自身坐标，得到以查询点为原点的局部相对坐标系；

3. **特征分组**：根据邻居点索引，提取对应的特征向量，形成每个查询点的邻域特征集合。

该方法的核心作用是将无序的点云数据转换为有序的局部邻域结构，为后续的注意力计算提供空间上下文信息。完整代码实现如下所示:

```python
def queryandgroup(nsample, xyz, new_xyz, feat, idx, offset, new_offset, use_xyz=True):
    """
    查询并分组函数：为每个查询点找到最近邻并分组其特征
    input:
        nsample: 最近邻数量
        xyz: 所有点的坐标 (n, 3)
        new_xyz: 查询点的坐标 (m, 3)
        feat: 所有点的特征 (n, c)
        idx: 预计算的最近邻索引，如果为None则重新计算
        offset: 每个batch的点的结束索引 (b)
        new_offset: 每个batch的查询点的结束索引 (b)
        use_xyz: 是否在输出中包含相对坐标信息
    output:
        new_feat: 分组后的特征，如果use_xyz=True则为(m, nsample, 3+c)，否则为(m, nsample, c)
        grouped_idx: 分组后的索引 (m, nsample)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()

    # 如果没有指定查询点，则使用所有点作为查询点
    if new_xyz is None:
        new_xyz = xyz

    # 如果没有提供预计算的索引，则调用KNN查询函数计算
    if idx is None:
        idx, _ = knnquery(nsample, xyz, new_xyz, offset, new_offset)  # (m, nsample)

    n, m, c = xyz.shape[0], new_xyz.shape[0], feat.shape[1]

    # 根据索引分组坐标：获取每个查询点的邻居坐标
    grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3)  # (m, nsample, 3)

    # 计算相对坐标：邻居坐标减去查询点坐标（局部坐标系）： （200，8，3） -  （200，1，3）= （ 200,8,3 ）
    grouped_xyz -= new_xyz.unsqueeze(1)  # (m, nsample, 3)

    # 根据索引分组特征：获取每个查询点的邻居特征
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)  # (m, nsample, c)

    # 根据use_xyz标志决定输出格式
    if use_xyz:
        # 拼接相对坐标和特征：输出形状为(m, nsample, 3+c)
        return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        # 只返回特征：输出形状为(m, nsample, c)
        return grouped_feat
```

`KNNQuery` 类实现了K近邻查询算法，其主要功能是为每个查询点寻找最近的邻居点。具体实现包含以下几个关键步骤：

**1. 问题背景**：在批处理点云数据时，不同样本的点云可能包含不同数量的点（如第一个点云1024个点，第二个点云2048个点），因此需要使用 `offset` 和 `new_offset` 参数来标记每个batch中点云的边界范围。

**2. 算法流程**：

- 对于每个查询点，计算其与当前batch内所有候选点的欧几里得距离

- 使用 `torch.topk` 函数选出距离最小的 `nsample` 个点

- 返回最近邻点的索引和对应的距离值

**3. 实现特点**：采用批处理方式提高计算效率，同时处理点云数量不一致的情况。完整代码实现如下:

```python
class KNNQuery(Function):
    @staticmethod
    def forward(ctx, nsample, xyz, new_xyz, offset, new_offset):
        """
        KNN查询的前向传播函数
        input:
            nsample: 需要查询的最近邻数量
            xyz: 所有点的坐标 (n, 3)
            new_xyz: 查询点的坐标 (m, 3)，如果为None则使用xyz
            offset: 每个batch的点的结束索引 (b)
            new_offset: 每个batch的查询点的结束索引 (b)
        output:
            idx: 每个查询点的最近邻点索引 (m, nsample)
            dist2: 每个查询点到最近邻点的平方距离 (m, nsample)
        """
        if new_xyz is None:
            new_xyz = xyz  # 如果没有指定查询点，则对所有点进行自查询
        assert xyz.is_contiguous() and new_xyz.is_contiguous()
        m = new_xyz.shape[0]  # 查询点的数量

        # 初始化输出张量：索引矩阵和距离矩阵
        idx = torch.zeros((m, nsample), dtype=torch.long)
        dist2 = torch.zeros((m, nsample))

        # 按batch处理数据
        start_idx, new_start_idx = 0, 0  # 当前batch的起始索引
        for i in range(len(offset)):
            # 计算当前batch的结束索引
            end_idx = offset[i] if i < len(offset) else xyz.shape[0]
            new_end_idx = new_offset[i] if i < len(new_offset) else m

            # 确保当前batch有数据需要处理
            if end_idx > start_idx and new_end_idx > new_start_idx:
                # 提取当前batch的点坐标和查询点坐标
                batch_xyz = xyz[start_idx:end_idx]
                batch_new_xyz = new_xyz[new_start_idx:new_end_idx]

                # 计算查询点与所有点之间的欧几里得距离平方
                # 使用广播机制计算坐标差: (1,n,3) - (m,1,3) = (m,n,3) - (m,n,3) = (m_batch, n_batch, 3)
                diff = batch_xyz.unsqueeze(0) - batch_new_xyz.unsqueeze(1)  
                # (m_batch, n_batch) - 平方距离矩阵
                distances = torch.sum(diff ** 2, dim=-1)

                # 获取k个最近邻的索引和距离
                actual_nsample = min(nsample, distances.shape[1])  # 实际可用的最近邻数量
                # torch.topk返回最小的k个值及其索引: (m_batch,actual_nsample)
                knn_dist, knn_idx = torch.topk(distances, actual_nsample, dim=1, largest=False)

                # 如果实际邻居数量小于要求的nsample，进行填充
                if actual_nsample < nsample:
                    # 使用0填充索引和距离矩阵
                    padding = torch.zeros((knn_idx.shape[0], nsample - actual_nsample), dtype=knn_idx.dtype)
                    knn_idx = torch.cat([knn_idx, padding], dim=1)
                    knn_dist = torch.cat(
                        [knn_dist, torch.zeros((knn_dist.shape[0], nsample - actual_nsample), dtype=knn_dist.dtype)],
                        dim=1)

                # 将当前batch的结果存入总输出中，注意加上全局偏移量
                idx[new_start_idx:new_end_idx] = knn_idx + start_idx
                dist2[new_start_idx:new_end_idx] = knn_dist

            # 更新下一个batch的起始索引
            start_idx, new_start_idx = end_idx, new_end_idx

        # 返回最近邻索引和实际距离（加上小常数避免数值不稳定）
        return idx, torch.sqrt(dist2 + 1e-8)


# 定义KNN查询的apply函数
knnquery = KNNQuery.apply
```