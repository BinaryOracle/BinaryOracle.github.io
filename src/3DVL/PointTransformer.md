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

Point Transformer 中提出的向量自注意力计算代码逻辑实现如下:

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

```python

```