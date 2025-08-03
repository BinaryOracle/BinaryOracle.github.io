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
