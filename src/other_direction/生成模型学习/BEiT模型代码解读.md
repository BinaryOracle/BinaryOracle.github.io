---
title: BEiT 模型代码解读
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-03
author:
  - BinaryOracle
---

`BEiT 模型代码解读` 

<!-- more -->

> 论文解读: [BEiT 论文解读](https://binaryoracle.github.io/other_direction/%E7%94%9F%E6%88%90%E6%A8%A1%E5%9E%8B%E5%AD%A6%E4%B9%A0/BEiT.html)

## dVAE 预训练

BEiT 模型的代码实现中定义了 两种变分自编码器（VAE），分别对应：

1. DiscreteVAE：自研的 VQ-VAE 实现，基于 lucidrains/DALLE-pytorch 改写。

2. Dalle_VAE：封装了 OpenAI 提供的预训练好的 DALL·E 编码器和解码器，直接加载 encoder.pkl 和 decoder.pkl。

###  DiscreteVAE 初始化

本节我们会针对 `DiscreteVAE` 的代码实现展开讲解 ，首先是 `DiscreteVAE` 类的 `__init__` 方法：

```python
class DiscreteVAE(BasicVAE):
    def __init__(
        self,
        image_size = 256,           # 输入图像尺寸，假设为正方形，如 256x256
        num_tokens = 512,           # codebook 中 token 的种类数（即离散编码类别数）
        codebook_dim = 512,         # 每个 token 的嵌入维度
        num_layers = 3,             # 编码器/解码器的下采样/上采样层数
        hidden_dim = 64,            # 每层卷积的通道数
        channels = 3,               # 输入图像通道数，通常为3（RGB图像）
        smooth_l1_loss = False,     # 是否使用 smooth_l1 作为重建损失（否则用 mse）
        temperature = 0.9,          # Gumbel Softmax 的温度参数
        straight_through = False,   # 是否启用 straight-through 近似采样（硬采样但梯度可导）
        kl_div_loss_weight = 0.     # KL 散度项的损失权重（用于保持 token 使用的均匀性）
    ):
        # 保存超参数
        self.image_size = image_size
        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.temperature = temperature
        self.straight_through = straight_through

        # 创建 codebook：token 编号 → 向量（[num_tokens, codebook_dim]）
        self.codebook = nn.Embedding(num_tokens, codebook_dim)

        enc_layers = []  # 编码器层列表
        dec_layers = []  # 解码器层列表

        enc_in = channels         # 编码器初始输入通道（RGB图像为3）
        dec_in = codebook_dim     # 解码器初始输入通道（token 嵌入维度）

        # 构建多层编码器和解码器（对称结构）
        for layer_id in range(num_layers):
            # 编码器：4x4卷积下采样 + ReLU
            enc_layers.append(
                nn.Sequential(
                    nn.Conv2d(enc_in, hidden_dim, kernel_size=4, stride=2, padding=1),
                    nn.ReLU()
                )
            )
            # 编码器残差块（增强特征提取）
            enc_layers.append(
                ResBlock(
                    chan_in=hidden_dim,
                    hidden_size=hidden_dim,
                    chan_out=hidden_dim
                )
            )
            enc_in = hidden_dim  # 下次输入通道设为隐藏通道

            # 解码器：反卷积上采样 + ReLU
            dec_layers.append(
                nn.Sequential(
                    nn.ConvTranspose2d(dec_in, hidden_dim, kernel_size=4, stride=2, padding=1),
                    nn.ReLU()
                )
            )
            # 解码器残差块
            dec_layers.append(
                ResBlock(
                    chan_in=hidden_dim,
                    hidden_size=hidden_dim,
                    chan_out=hidden_dim
                )
            )
            dec_in = hidden_dim  # 下次输入通道设为隐藏通道

        # 编码器最终输出层：将通道映射到 num_tokens，得到 token 分类 logits
        enc_layers.append(nn.Conv2d(hidden_dim, num_tokens, kernel_size=1))

        # 解码器最终输出层：映射回原图像通道数（通常为 3）
        dec_layers.append(nn.Conv2d(hidden_dim, channels, kernel_size=1))

        # 将所有子模块组合为完整的 encoder 和 decoder 网络
        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        # 重建损失函数：使用 Smooth L1 或 MSE
        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

        # KL 散度正则项的权重（默认 0，不启用）
        self.kl_div_loss_weight = kl_div_loss_weight
```

> 残差块实现如下:
>
> ```python
> class ResBlock(nn.Module):
>     def __init__(self, chan_in, hidden_size, chan_out):
>         super().__init__()
>         self.net = nn.Sequential(
>             nn.Conv2d(chan_in, hidden_size, 3, padding=1),
>             nn.ReLU(),
>             nn.Conv2d(hidden_size, hidden_size, 3, padding=1),
>             nn.ReLU(),
>             nn.Conv2d(hidden_size, chan_out, 1)
>         )
> 
>     def forward(self, x):
>         return self.net(x) + x
> ```

### DiscreteVAE 前向传播

下面给出的是 `DiscreteVAE` 类的 `forward` 方法实现:

```python
def forward(
        self,
        img,                        # 输入图像 [B, C, H, W]
        return_loss = False,        # 是否返回 loss
        return_recons = False,      # 是否返回重建图像
        return_logits = False,      # 是否仅返回 logits
        temp = None                 # 覆盖默认温度参数
    ):
        device = img.device
        num_tokens = self.num_tokens
        image_size = self.image_size
        kl_div_loss_weight = self.kl_div_loss_weight

        # 编码器前向传播，得到每个像素位置的 logits
        logits = self.encoder(img)  # shape: [B, num_tokens, H', W']

        # 若仅获取 logits（例如用于 DALL-E 中提取离散 token 索引），则直接返回
        if return_logits:
            return logits

        # 使用 Gumbel Softmax 对 logits 进行采样，得到 soft one-hot 编码
        temp = temp if temp is not None else self.temperature
        soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=self.straight_through)
        # shape: [B, num_tokens, H', W']

        # 查找 codebook 向量，获得连续隐变量
        # einsum 相当于 soft_one_hot @ codebook.weight
        sampled = einsum('b n h w, n d -> b d h w', soft_one_hot, self.codebook.weight)
        # shape: [B, codebook_dim, H', W']

        # 解码器重建图像
        out = self.decoder(sampled)  # shape: [B, C, H, W]

        # 如果不需要 loss，只返回重建图像
        if not return_loss:
            return out

        # 计算重建损失
        recon_loss = self.loss_fn(img, out)

        # KL 散度损失（可选，衡量 q(y) 与 uniform 的差异）
        logits = rearrange(logits, 'b n h w -> b (h w) n')  # 展平空间维度
        qy = F.softmax(logits, dim=-1)                      # 每个位置的 token 概率分布
        log_qy = torch.log(qy + 1e-10)                      # 避免 log(0)

        log_uniform = torch.log(torch.tensor([1. / num_tokens], device=device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, reduction='batchmean', log_target=True)

        # 加权总损失
        loss = recon_loss + (kl_div * kl_div_loss_weight)

        # 如果不需要重建图像，仅返回 loss
        if not return_recons:
            return loss

        # 否则返回 loss 和重建图像
        return loss, out
```

#### Gumbel Softmax

传统的离散变量采样（比如从多分类分布中采样一个类别）是非可微的，不能直接用反向传播训练神经网络。**Gumbel Softmax**（也叫 **Concrete distribution**）是一种连续的可微近似方法，允许在训练时对离散随机变量进行“软采样”，实现端到端的梯度传播。 训练时用“软采样”表示类别概率的加权和；推理时可以用硬采样（one-hot）恢复离散的类别。

Gumbel Softmax的采样过程分为两步：

* 对每个类别的 logits（未归一化的对数概率）加上 Gumbel 噪声（用来模拟采样的随机性）

* 对加噪声后的 logits 使用 softmax，并用温度参数控制“分布的平滑度”

数学表达式：

$$
y_i = \frac{\exp((\log(\pi_i) + g_i)/\tau)}{\sum_{j=1}^K \exp((\log(\pi_j) + g_j)/\tau)}
$$

其中：

* $\pi_i$ 是第 $i$ 类的概率（或 logits 经过 softmax 的概率）

* $g_i$ 是从 Gumbel(0,1) 分布采样的噪声，定义为 $g_i = -\log(-\log(u_i))$，其中 $u_i \sim \text{Uniform}(0,1)$

* $\tau$ 是温度参数，控制分布的“尖锐度”。温度越低，采样越接近 one-hot；温度高时更平滑。


PyTorch 提供了 `F.gumbel_softmax` 函数来实现上述过程：

```python
soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=self.straight_through)
```

* `logits`：输入的未归一化的 logits 张量。

* `tau`：温度参数 `temp`，控制分布的平滑度。

* `dim=1`：在类别维度上执行 softmax。

* `hard`：布尔值，是否使用硬采样 + 直通梯度。

  * `hard=False` 返回的是软概率分布（连续值，可微）。

  * `hard=True` 返回 one-hot 编码，但梯度仍由软样本近似（Straight-Through Estimator）。

Gumbel Softmax 使模型在训练时可以对这些 logits 进行采样，得到一个“软”的 one-hot 向量，代表隐空间的离散编码。这个 soft one-hot 向量乘以 codebook 的 embedding 权重，得到连续的隐向量表示，用于解码器重建图像。

> 当 `hard=True` 时，可以模拟硬采样（one-hot向量），方便离散索引的推断，同时仍保证梯度流通。

##### hard=True时，如何实现的？

当 `hard=self.straight_through` 为 `True` 时，**Gumbel-Softmax 采样**过程使用了一种称为 **Straight-Through Gumbel-Softmax** 的技巧，它使得：

* **前向传播**时是**one-hot 向量**（离散），

* **反向传播**时仍保持**连续可导**（通过 softmax）

在 VAE 或 BEiT 的离散编码器中，我们希望：

* 对图像进行**离散 token 编码**（便于 Transformer 训练）

* 但同时又希望这个采样过程能**反向传播梯度**

这就引出了 Straight-Through Gumbel-Softmax：

---

采样过程详解:

```python
soft_one_hot = F.gumbel_softmax(logits, tau=temp, dim=1, hard=True)
```

等价于：

```python
y_soft = softmax((logits + GumbelNoise) / tau)     # 连续分布，用于反向传播
index = argmax(y_soft)                             # 找到最大概率的 one-hot 索引
y_hard = one_hot(index)                            # 得到一个离散的 one-hot 向量

# 关键一步：straight-through trick
y = y_hard.detach() - y_soft.detach() + y_soft
```

---

```python
y = y_hard.detach() - y_soft.detach() + y_soft
```

* `y_hard.detach()`：将 one-hot 向量从计算图中**分离出来**（不可导）

* `y_soft.detach()`：也分离出来，表示不在反向传播中参与梯度计算

* `+ y_soft`：把 soft 向量加入回来，用于**反向传播**

➡️ 整体效果：

| 方向       | 数据流                 | 梯度流       |
| -------- | ------------------- | --------- |
| forward  | 使用 one-hot 离散 token | ——        |
| backward | 使用 softmax 的连续梯度    | 保持可导，稳定训练 |

#### smooth_l1_loss

`smooth_l1_loss` 是 PyTorch 中的一种 **回归损失函数**，也被称为 **Huber Loss** 的一种变体，它结合了均方误差（MSE）和平均绝对误差（MAE）的优点，在处理 **异常值/离群点鲁棒性更强**。

对于预测值 $x$、目标值 $y$，以及误差 $\delta = x - y$：

$$
\text{smooth\_l1\_loss}(\delta) =
\begin{cases}
0.5 \cdot \delta^2 & \text{if } |\delta| < 1 \\
|\delta| - 0.5 & \text{otherwise}
\end{cases}
$$

这个函数在误差较小时近似于 `MSELoss`，误差较大时退化为 `L1Loss`，从而避免了 MSE 对异常值特别敏感的问题。

相比 MSE，使用 `smooth_l1_loss` 能让 VAE：

* 对单个像素偏差较大的情况更加宽容（防止训练不稳定）

* 更容易在训练中收敛，因为梯度变化更平滑

> **`smooth_l1_loss` 是一个融合了 MSE 的平滑性与 L1 的鲁棒性的损失函数，常用于图像回归与重建任务中，能更好处理异常误差。**

#### KL散度计算

```python
        # KL 散度损失（可选，衡量 q(y) 与 uniform 的差异）
        logits = rearrange(logits, 'b n h w -> b (h w) n')  # 展平空间维度: [B, num_tokens, H, W] 变为 [B, H*W, num_tokens]
        qy = F.softmax(logits, dim=-1)                      # 每个位置的 token 概率分布
        log_qy = torch.log(qy + 1e-10)                      # 避免 log(0)

        log_uniform = torch.log(torch.tensor([1. / num_tokens], device=device))
        kl_div = F.kl_div(log_uniform, log_qy, None, None, reduction='batchmean', log_target=True)
```
在这段代码中，我们计算的是**编码器输出分布** $q(y)$ 与一个**先验分布** $p(y)$ 之间的 Kullback-Leibler 散度，记作：

$$
\mathrm{KL}(p(y) \,\|\, q(y)) = \sum_{i=1}^{N} p(y_i) \cdot \left[ \log p(y_i) - \log q(y_i) \right]
$$

其中：

* $p(y)$：先验分布（理想中我们希望 encoder 生成的分布接近它）

* $q(y)$：encoder 对图像每个 patch 给出的 softmax 分布

* $N$：codebook 中的 token 数，即 `num_tokens`

在本代码中：

* $p(y_i) = \frac{1}{N}$，即是**均匀分布**

* 所以 $\log p(y_i) = \log \left(\frac{1}{N}\right) = -\log N$

带入公式得到：

$$
\mathrm{KL}(p(y) \,\|\, q(y)) = \sum_{i=1}^{N} \frac{1}{N} \cdot \left[ -\log N - \log q(y_i) \right]
= -\log N - \frac{1}{N} \sum_{i=1}^N \log q(y_i)
$$

也就是：

$$
\mathrm{KL}(p(y) \,\|\, q(y)) = \text{常数} - \mathbb{E}_{i \sim \text{uniform}}[\log q(y_i)]
$$

这个损失鼓励 q 趋近于均匀，从而避免编码器只用很少几个 token。

---

##### log_target 参数

```python
F.kl_div(input, target, log_target=False)
```

此时，**`input` 是 log 概率** $\log q_i$，**`target` 是概率** $p_i$，计算公式为：

$$
\text{KL}(p \| q) = \sum_i p_i \cdot (\log p_i - \log q_i)
$$

即：

```python
F.kl_div(log_q, p, log_target=False)
```

----

```python
F.kl_div(log_p, log_q, log_target=True)
```

此时认为 **两个参数都是 log 概率**，底层计算公式变为：

$$
\text{KL}(p \| q) = \sum_i \exp(\log p_i) \cdot (\log p_i - \log q_i)
$$

也就是 PyTorch 自动执行：

```python
KL = (p * (log_p - log_q)).sum()
```

其中：

* $\exp(\log p_i) = p_i$

* $\exp(\log q_i) = q_i$

⚠️ 注意：这种方式需要我们手动把两个分布都以 `log` 形式传进去。

----

##### 为什么先验分布设置为均匀分布？

这是为了满足 **信息瓶颈** 或 **高效利用 codebook** 的目标：

1. **VQ-VAE 的典型问题：code collapse**

* 编码器如果训练不当，可能会只偏好极少数几个 code（比如 512 个 code 中只用 10 个），这是 **codebook collapse**。

* 结果就是：虽然理论上有 512 种可能的图像 patch 表达，但实际只用了极少数，模型表达能力受限。

2. **使用均匀先验的好处**

* 均匀分布意味着我们希望所有 token 被“平等地使用”。

* 加上 KL 散度约束后，编码器会被正则化为“尽可能平均地使用每个 token”。

* 这样可以**提高 codebook 的使用率**，提升模型的表达多样性。


总结为一句话：

> 使用均匀先验是为了鼓励编码器生成的离散 token 分布更加均衡，避免 code collapse，从而充分利用整个 codebook 的表示能力。

## 块状遮挡（blockwise masking）策略

块状遮挡通过遮盖图像中的连续 patch 区域，更真实地模拟自然场景中的遮挡，增强模型对上下文的理解能力，避免信息泄漏，同时实现简单且与基于patch的模型结构高度契合，因此比像素级别遮挡更有效和实用。

遮挡方式：

1. 将图像划分为 $H \times W$ 个 patch

2. 每次遮挡一个矩形块区域，如 4×4 或 6×3 的 patch 区域

3. 最终总共遮掉 num_masking_patches 个 patch

相比于逐 patch 独立遮挡（如 random token masking），这种遮挡方式：

| 遮挡方式 | 特点 | 对比优势 |
| --- | --- | --- |
| 随机单 patch 遮挡 | 每个 patch 独立被遮或不遮 | 遮挡区域零碎 |
| ✅ 块状遮挡 | 遮一片连续矩形区域 | 更符合图像结构、语义连续 |

块状掩码策略的具体实现代码如下所示，首先给出的是掩码生成器的初始化方法，重点注意各个参数的含义:

```python
class MaskingGenerator:
    def __init__(
            self, input_size,                 # 输入图像的 patch 网格大小（如 14 表示 14x14 patch）
            num_masking_patches,             # 最终要 mask 掉的 patch 总数量
            min_num_patches=4,               # 每次生成一个遮挡块时，最小 patch 数
            max_num_patches=None,            # 每次遮挡块最多的 patch 数；默认等于 num_masking_patches
            min_aspect=0.3,                  # 遮挡块的最小宽高比（例如 h/w = 0.3）
            max_aspect=None):               # 最大宽高比，默认取 1 / min_aspect（对称处理）
        
        # 如果输入是整数，则构造成正方形大小的 patch 网格
        if not isinstance(input_size, tuple):
            input_size = (input_size, ) * 2
        self.height, self.width = input_size  # patch 网格的高和宽（例如 14x14）

        self.num_patches = self.height * self.width  # 总共可用 patch 数量
        self.num_masking_patches = num_masking_patches  # 需要被 mask 的 patch 总数

        self.min_num_patches = min_num_patches  # 单个遮挡块的最小 patch 数
        # 若 max_num_patches 未指定，则设为总遮挡目标数（不限制）
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches

        # 如果未指定最大宽高比，默认与 min_aspect 互为倒数，保持对称性
        max_aspect = max_aspect or 1 / min_aspect

        # 记录宽高比范围的对数形式，便于采样（log 均匀采样 → 平滑控制长宽比例分布）
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))
```
下面将展示正式执行块状遮挡策略前的准备工作:

```python
def __call__(self):
    # 初始化一个全零的遮挡掩码，大小为输入图像的patch数目（height x width）
    mask = np.zeros(shape=self.get_shape(), dtype=np.int)
    mask_count = 0  # 当前已经遮挡的patch数量

    # 循环直到遮挡的patch数量达到指定的遮挡总数
    while mask_count < self.num_masking_patches:
        # 计算本轮最多还能遮挡的patch数
        max_mask_patches = self.num_masking_patches - mask_count
        # 限制本次遮挡的patch数量不超过最大遮挡数
        max_mask_patches = min(max_mask_patches, self.max_num_patches)

        # 尝试生成一个遮挡块，返回本次新增遮挡的patch数量
        delta = self._mask(mask, max_mask_patches)
        if delta == 0:
            # 如果没有新增遮挡（即无法再生成有效遮挡块），跳出循环
            break
        else:
            # 更新已遮挡patch数量
            mask_count += delta

    # 返回最终生成的遮挡掩码（0表示未遮挡，1表示遮挡）
    return mask
```

执行块状遮挡策略的核心代码实现如下:

```python
def _mask(self, mask, max_mask_patches):
    delta = 0  # 记录本次新增遮挡的patch数量
    for attempt in range(10):  # 最多尝试10次生成遮挡块
        # 随机采样目标遮挡面积（patch数量），范围在[min_num_patches, max_mask_patches]之间
        # random.uniform 是均匀采样，即每一个值被采样到的可能性完全相等
        target_area = random.uniform(self.min_num_patches, max_mask_patches)
        # 随机采样遮挡块的长宽比（在log空间均匀采样后exp还原）
        aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
        
        # 根据面积和长宽比计算遮挡块的高度和宽度（向最近整数取整）
        # h * w ≈ target_area 
        # h / w ≈ aspect_ratio
        # h^2 =  target_area * aspect_ratio;  w^2 = target_area / aspect_ratio
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        
        # 检查遮挡块尺寸是否小于输入图像patch尺寸，确保遮挡块可放入图像范围内
        if w < self.width and h < self.height:
            # 随机采样遮挡块在图像上的左上角位置，确保遮挡块不会越界
            top = random.randint(0, self.height - h)
            left = random.randint(0, self.width - w)

            # 计算遮挡块区域内已被遮挡的patch数
            num_masked = mask[top: top + h, left: left + w].sum()

            # 判断当前遮挡块的有效新增遮挡数量
            # 必须新增遮挡patch数>0且不超过最大允许遮挡数
            if 0 < h * w - num_masked <= max_mask_patches:
                # 遍历遮挡块区域，将未遮挡的patch设置为遮挡（1），累计新增遮挡数量
                for i in range(top, top + h):
                    for j in range(left, left + w):
                        if mask[i, j] == 0:
                            mask[i, j] = 1
                            delta += 1

            # 如果本次成功新增了遮挡patch，跳出尝试循环
            if delta > 0:
                break
    # 返回本次新增的遮挡patch数量
    return delta
```

