---
title: DALL·E 论文
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-04
author:
  - BinaryOracle
---

`DALL·E 论文` 

<!-- more -->

> 论文链接: [Zero-Shot Text-to-Image Generation](https://arxiv.org/abs/2102.12092)
> 第三方代码实现: [DALL-E](https://github.com/lucidrains/DALLE-pytorch/tree/main)

## 代码实现

![DALL-E 模型前向传播整体流程](DALL-E/3.png)

### DALL-E 主模型

`DALL·E` 将 **文本-图像生成** 问题建模为一个**自回归语言建模任务**，即将**文本 token 和图像 token 拼接**起来，作为一个**统一的序列**进行训练，从而学会生成图像的离散表示。

#### 初始化方法

```python
class DALLE(nn.Module):
    def __init__(
        self,
        *,
        dim,                            # Transformer 的隐藏层维度
        vae,                            # 图像离散化使用的 VAE（如 DiscreteVAE、OpenAIDiscreteVAE、VQGanVAE）
        num_text_tokens = 10000,        # 文本词表大小
        text_seq_len = 256,             # 文本最大序列长度
        depth,                          # Transformer 层数
        heads = 8,                      # Attention 头数
        dim_head = 64,                  # 每个 attention head 的维度
        reversible = False,             # 是否使用 reversible transformer（节省显存）
        attn_dropout = 0.,              # attention dropout
        ff_dropout = 0,                 # feedforward dropout
        sparse_attn = False,            # 是否使用稀疏 attention
        attn_types = None,              # 支持混合 attention 类型
        loss_img_weight = 7,            # 图像 token 损失权重（论文中为 1:7）
        stable = False,                 # 是否使用稳定化的 Transformer 训练技巧
        sandwich_norm = False,          # 是否使用 sandwich LayerNorm（LayerNorm前后各一次）
        shift_tokens = True,            # 是否进行 token shifting（提升模型稳定性）
        rotary_emb = True,              # 是否使用旋转位置编码
        shared_attn_ids = None,         # 是否共享注意力模块
        shared_ff_ids = None,           # 是否共享 feedforward 模块
        share_input_output_emb = False, # 是否共享输入输出 embedding（减少参数）
        optimize_for_inference = False, # 推理优化（不使用 Dropout 等）
    ):
        # 图像尺寸和 token 数量来自 VAE
        image_size = vae.image_size                        # 输入图像的尺寸，如 256
        num_image_tokens = vae.num_tokens                  # 图像 token 的词表大小，如 8192
        image_fmap_size = image_size // (2 ** vae.num_layers)  # 编码后图像的 feature map 尺寸（例如 16x16）
        image_seq_len = image_fmap_size ** 2               # 图像 token 序列长度（如 256）

        # 为每个文本位置保留一个唯一 padding token（提高 mask 效果）
        num_text_tokens = num_text_tokens + text_seq_len

        # 定义文本和图像的位置编码
        self.text_pos_emb = nn.Embedding(text_seq_len + 1, dim) if not rotary_emb else always(0)  # +1 for <BOS>
        self.image_pos_emb = AxialPositionalEmbedding(dim, axial_shape=(image_fmap_size, image_fmap_size)) if not rotary_emb else always(0)

        self.num_text_tokens = num_text_tokens  # 用于后续 logits 偏移与 loss 计算
        self.num_image_tokens = num_image_tokens

        self.text_seq_len = text_seq_len
        self.image_seq_len = image_seq_len

        seq_len = text_seq_len + image_seq_len            # 总序列长度（文本 + 图像）
        total_tokens = num_text_tokens + num_image_tokens # 总 token 数（用于 logits 输出）
        self.total_tokens = total_tokens
        self.total_seq_len = seq_len

        self.vae = vae
        set_requires_grad(self.vae, False)                # 冻结 VAE（不参与训练）

        # 构建 Transformer，用于联合建模文本和图像 token
        self.transformer = Transformer(
            dim=dim,
            causal=True,                  # 自回归建模
            seq_len=seq_len,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            reversible=reversible,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            attn_types=attn_types,
            image_fmap_size=image_fmap_size,
            sparse_attn=sparse_attn,
            stable=stable,
            sandwich_norm=sandwich_norm,
            shift_tokens=shift_tokens,
            rotary_emb=rotary_emb,
            shared_attn_ids=shared_attn_ids,
            shared_ff_ids=shared_ff_ids,
            optimize_for_inference=optimize_for_inference,
        )

        self.stable = stable
        if stable:
            self.norm_by_max = DivideMax(dim=-1)          # 稳定化处理

        # 最后的输出映射到 logits（用于 cross entropy loss）
        self.to_logits = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, self.total_tokens),            # 输出维度 = 全部 token 数（文本+图像）
        )

        # Embedding 层：可以选择共享输入输出权重
        if share_input_output_emb:
            self.text_emb = SharedEmbedding(self.to_logits[1], 0, num_text_tokens)
            self.image_emb = SharedEmbedding(self.to_logits[1], num_text_tokens, total_tokens)
        else:
            self.text_emb = nn.Embedding(num_text_tokens, dim)
            self.image_emb = nn.Embedding(num_image_tokens, dim)

        # 构造 logits mask（防止文本位置预测图像 token，反之亦然）
        seq_range = torch.arange(seq_len)
        logits_range = torch.arange(total_tokens)

        seq_range = rearrange(seq_range, 'n -> () n ()')         # shape: (1, seq_len, 1)
        logits_range = rearrange(logits_range, 'd -> () () d')   # shape: (1, 1, total_tokens)

        logits_mask = (
            ((seq_range >= text_seq_len) & (logits_range < num_text_tokens)) |     # 图像位置不能预测文本 token
            ((seq_range < text_seq_len) & (logits_range >= num_text_tokens))       # 文本位置不能预测图像 token
        )

        self.register_buffer('logits_mask', logits_mask, persistent=False)  # 注册为 buffer，防止被当作参数保存

        self.loss_img_weight = loss_img_weight  # 图像 loss 的权重，用于 loss 加权（论文中为 7 倍）
```

#### 前向传播流程

```python
def forward(
    self,
    text,
    image=None,
    return_loss=False,
    null_cond_prob=0.,
    cache=None,
):
    # 获取 batch size、device 和 transformer 的最大序列长度
    batch, device, total_seq_len = text.shape[0], text.device, self.total_seq_len

    # 以一定概率随机删除文本条件（用于训练时的条件 dropout）
    if null_cond_prob > 0:
        null_mask = prob_mask_like((batch,), null_cond_prob, device=device)
        text *= rearrange(~null_mask, 'b -> b 1')  # 如果 null_mask=True，则整条 text 设为 0（即无条件）

    # 将 padding token（0）替换为唯一的 token ID，避免 embedding 冲突
    text_range = torch.arange(self.text_seq_len, device=device) + (self.num_text_tokens - self.text_seq_len)
    text = torch.where(text == 0, text_range, text)

    # 在文本序列开头加上 <bos> token（值为0）
    text = F.pad(text, (1, 0), value=0)

    # 文本 token embedding 与位置编码
    tokens = self.text_emb(text)
    tokens += self.text_pos_emb(torch.arange(text.shape[1], device=device))

    seq_len = tokens.shape[1]  # 当前 token 序列长度（仅包含文本部分）

    # 如果输入了图像（且非空），处理图像 embedding
    if exists(image) and not is_empty(image):
        is_raw_image = len(image.shape) == 4  # 如果是原始图像（B, C, H, W）

        if is_raw_image:
            image_size = self.vae.image_size
            channels = self.vae.channels
            # 确保图像尺寸正确
            assert tuple(image.shape[1:]) == (channels, image_size, image_size), \
                f'invalid image of dimensions {image.shape} passed in during training'

            # 使用 VAE 将原始图像编码为离散 codebook indices
            image = self.vae.get_codebook_indices(image)

        image_len = image.shape[1]
        image_emb = self.image_emb(image)  # 图像 token embedding
        image_emb += self.image_pos_emb(image_emb)  # 图像位置编码

        # 将文本和图像的 embedding 拼接
        tokens = torch.cat((tokens, image_emb), dim=1)
        seq_len += image_len  # 更新总长度

    # 如果 token 总长度超过模型最大长度，则裁剪掉最后一个 token（训练时末尾 token 不需要预测）
    if tokens.shape[1] > total_seq_len:
        seq_len -= 1
        tokens = tokens[:, :-1]

    # 如果启用了稳定训练策略（stabilization trick）
    if self.stable:
        alpha = 0.1
        tokens = tokens * alpha + tokens.detach() * (1 - alpha)

    # 如果使用了 KV Cache（用于推理阶段），只保留最后一个 token
    if exists(cache) and cache.get('offset'):
        tokens = tokens[:, -1:]

    # 送入 transformer 主体
    out = self.transformer(tokens, cache=cache)

    # 如果启用了稳定策略，对输出做归一化
    if self.stable:
        out = self.norm_by_max(out)

    # 得到每个位置上的分类 logits（预测 token）
    logits = self.to_logits(out)

    # 构造 logits mask：限制哪些位置可以预测哪些 token（防止跨模态预测）
    logits_mask = self.logits_mask[:, :seq_len]
    if exists(cache) and cache.get('offset'):
        logits_mask = logits_mask[:, -1:]
    max_neg_value = -torch.finfo(logits.dtype).max  # -inf 替代值
    logits.masked_fill_(logits_mask, max_neg_value)  # 用 -inf 屏蔽不合法预测

    # 更新 KV Cache 的偏移量（用于增量推理）
    if exists(cache):
        cache['offset'] = cache.get('offset', 0) + logits.shape[1]

    # 如果不要求计算损失，直接返回 logits
    if not return_loss:
        return logits

    # 训练时必须提供图像（否则无法计算图像 token 的预测损失）
    assert exists(image), 'when training, image must be supplied'

    # 将图像 token 的索引整体加偏移（让图像 token ID 与文本 token 不重叠）
    offsetted_image = image + self.num_text_tokens

    # 构造预测标签：文本去掉 <bos>（text[:, 1:]），接上图像 token
    labels = torch.cat((text[:, 1:], offsetted_image), dim=1)

    # logits 维度从 [B, N, C] 变成 [B, C, N]，以匹配 cross_entropy 的输入格式
    logits = rearrange(logits, 'b n c -> b c n')

    # 计算文本部分的 cross-entropy loss（前 self.text_seq_len 个 token）
    loss_text = F.cross_entropy(logits[:, :, :self.text_seq_len], labels[:, :self.text_seq_len])

    # 计算图像部分的 cross-entropy loss
    loss_img = F.cross_entropy(logits[:, :, self.text_seq_len:], labels[:, self.text_seq_len:])

    # 按照权重加权融合 loss（图像损失通常占更大比例）
    loss = (loss_text + self.loss_img_weight * loss_img) / (self.loss_img_weight + 1)

    return loss
```
