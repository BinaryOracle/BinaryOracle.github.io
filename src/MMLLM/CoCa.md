---
title: CoCa 论文
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-28
author:
  - BinaryOracle
---

`CoCa: Contrastive Captioners are Image-Text Foundation Models 论文简析` 

<!-- more -->

> 论文链接: [CoCa: Contrastive Captioners are Image-Text Foundation Models](https://arxiv.org/abs/2205.01917)
> 代码链接: [https://github.com/lucidrains/CoCa-pytorch](https://github.com/lucidrains/CoCa-pytorch)

## 引言

近年来，计算机视觉领域对**大规模预训练基础模型**的探索越来越重要，因为这些模型能够快速迁移到各种下游任务上。本论文提出了一种极简设计的 **Contrastive Captioner (CoCa)** 模型，它是一种**图文编码-解码结构的基础模型**，在训练时同时使用**对比损失**和**生成式的描述损失**。这样一来，它既能继承 CLIP 这类**对比方法**的能力，又能结合 SimVLM 这类**生成方法**的优点。

与传统的 encoder-decoder 架构（解码器所有层都对编码器输出做 cross-attention）不同，CoCa 的解码器被一分为二：

* 前半部分为**单模态解码器（unimodal decoder）**，没有 cross-attention，只学习纯文本的表示；

* 后半部分为**多模态解码器（multimodal decoder）**，通过 cross-attention 融合图像和文本，得到跨模态的联合表示。

在训练目标上，CoCa 同时引入：

* **对比损失**：用于图像编码器输出与文本单模态表示之间的对齐；

* **描述损失（captioning loss）**：作用在多模态解码器的输出上，要求其自回归地预测文本 token。

这种**共享计算图**的方式，让两个目标可以在计算上高效结合，几乎没有额外开销。训练时，所有的标签（包括人工标注和网络噪声数据）都被统一当作文本，从而自然地融合了不同来源的监督信号。

> 共享计算图: 一次forward完成两个损失目标值的计算。

---

深度学习的发展，已经在语言领域涌现出 **BERT、T5、GPT-3** 等基础模型，它们通过大规模预训练展示出**零样本、多任务、迁移学习**的能力。相比专用模型，基础模型能在 amortized（摊销）成本上覆盖更多下游任务，推动规模化智能的发展。

在视觉和视觉-语言任务中，已有三条研究路径：

**(1) 单编码器（Single-encoder）**

* 代表性工作在 ImageNet 等图像分类数据集上用交叉熵损失预训练。

* 优点：提供通用的视觉特征，可迁移到图像和视频理解任务。

* 缺点：仅依赖图像标注（类别标签），无法利用自然语言知识，因此在涉及图文结合的任务（如 VQA）上受限。

**(2) 双编码器（Dual-encoder，对比学习）**

* 通过图像编码器和文本编码器分别编码图像与文本，再用对比损失在共享的潜在空间对齐。

* 优点：不仅能服务视觉任务，还能进行跨模态任务（如图文检索、零样本分类）。

* 缺点：缺乏图文融合的联合表示，因此无法直接应用于复杂的多模态理解任务（如 VQA）。

**(3) 编码-解码（Encoder-decoder，生成式预训练）**

* 采用图像输入到编码器，解码器侧使用语言建模损失（LM loss 或 PrefixLM）进行训练。

* 优点：能学到跨模态的联合表示，在多模态理解任务上表现突出。

* 缺点：不能同时得到和图像对齐的纯文本表示，因此在跨模态对齐与检索方面不足。

---

![](coca/1.png)

CoCa **融合并统一了以上三类范式**，提出了一种改进的 encoder-decoder 架构：

* 将解码器拆分为 **单模态部分**（仅学习文本特征）和 **多模态部分**（跨模态融合）。

* 在单模态文本表示与图像表示之间施加**对比目标**，同时在多模态解码器输出上施加**生成目标**。

* 在训练数据上，把所有标注（类别标签、自然描述、网络噪声文本）都视作文本，从而无缝整合了不同监督。

这样一来，CoCa 的训练目标兼顾了：

* **对比学习的优势**（学习全局语义表征，适合检索和零样本分类）；

* **生成学习的优势**（对细粒度的区域特征建模，适合描述和理解任务）。

---

CoCa 在多种任务上展现了强大的零样本和迁移能力：

* **ImageNet 分类**：

  * 零样本准确率 **86.3%**

  * 冻结编码器 + 学分类头：**90.6%**

  * 全模型微调：**91.0%（SOTA）**

* **视频理解**：

  * Kinetics400/600/700：88.0% / 88.5% / 81.1%

  * Moments-in-Time：47.4%

* **跨模态检索**：

  * MSCOCO、Flickr30k：显著优于现有方法

* **多模态理解**：

  * VQA：82.3%

  * SNLI-VE、NLVR2：同样有优异表现

* **图像描述生成**：

  * NoCaps：CIDEr 得分 **120.6**

这些结果表明：**一个统一的 CoCa 模型，能在无需大量任务特定微调的前提下，超越多个专用模型的性能**。

## 方法

## 代码实现

```python
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from einops import rearrange, repeat, einsum

# -------------------------
# 一些辅助模块 (假设已定义)
# - CrossAttention: 跨注意力层
# - LayerNorm: 层归一化
# - Residual: 残差连接包装
# - ParallelTransformerBlock: Transformer 基础块 (自注意力 + FFN)
# - EmbedToLatents: 将表征映射到潜在空间 (用于对比学习)
# - default, exists, all_gather: 常用工具函数
# -------------------------

class CoCa(nn.Module):
    def __init__(
        self,
        *,
        dim,                       # 模型隐空间维度
        num_tokens,                # 词表大小
        unimodal_depth,            # 文本单模态解码层数
        multimodal_depth,          # 图文多模态解码层数
        dim_latents = None,        # 对比学习时潜在空间维度
        image_dim = None,          # 图像编码器输出维度
        num_img_queries=256,       # 图像查询向量数量（额外 +1 作为 CLS token）
        dim_head=64,               # 注意力头的维度
        heads=8,                   # 注意力头数
        ff_mult=4,                 # 前馈层扩展倍数
        img_encoder=None,          # 图像编码器 (例如 ViT)
        caption_loss_weight=1.,    # caption 任务损失权重
        contrastive_loss_weight=1.,# 对比学习损失权重
        pad_id=0                   # padding token ID
    ):
        super().__init__()
        self.dim = dim
        self.pad_id = pad_id
        self.caption_loss_weight = caption_loss_weight
        self.contrastive_loss_weight = contrastive_loss_weight

        # ----------- 文本 embedding -----------
        self.token_emb = nn.Embedding(num_tokens, dim)        # token embedding
        self.text_cls_token = nn.Parameter(torch.randn(dim))  # 文本 CLS token，用于对比学习

        # ----------- 图像编码器 -----------
        self.img_encoder = img_encoder

        # ----------- 图像注意力池化 -----------
        # img_queries: 查询向量 (包含 num_img_queries 个 + 1 个 CLS)
        self.img_queries = nn.Parameter(torch.randn(num_img_queries + 1, dim))
        self.img_attn_pool = CrossAttention(
            dim=dim,
            context_dim=image_dim,
            dim_head=dim_head,
            heads=heads,
            norm_context=True
        )
        self.img_attn_pool_norm = LayerNorm(dim)
        self.text_cls_norm = LayerNorm(dim)

        # ----------- 映射到潜在空间 (contrastive learning 用) -----------
        dim_latents = default(dim_latents, dim)
        self.img_to_latents = EmbedToLatents(dim, dim_latents)
        self.text_to_latents = EmbedToLatents(dim, dim_latents)

        # ----------- 对比学习温度参数 -----------
        self.temperature = nn.Parameter(torch.Tensor([1.]))

        # ----------- 单模态解码层 (仅处理文本) -----------
        self.unimodal_layers = nn.ModuleList([])
        for ind in range(unimodal_depth):
            self.unimodal_layers.append(
                Residual(
                    ParallelTransformerBlock(
                        dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult
                    )
                ),
            )

        # ----------- 多模态解码层 (融合图文) -----------
        self.multimodal_layers = nn.ModuleList([])
        for ind in range(multimodal_depth):
            self.multimodal_layers.append(nn.ModuleList([
                Residual(ParallelTransformerBlock(dim=dim, dim_head=dim_head, heads=heads, ff_mult=ff_mult)),   # 自注意力
                Residual(CrossAttention(dim=dim, dim_head=dim_head, heads=heads, parallel_ff=True, ff_mult=ff_mult))  # 跨模态注意力
            ]))

        # ----------- 输出到 logits -----------
        self.to_logits = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, num_tokens, bias=False)  # 输出词表 logits
        )

        # 权重共享: 输出层的权重和 embedding 权重共享
        self.to_logits[-1].weight = self.token_emb.weight
        nn.init.normal_(self.token_emb.weight, std=0.02)

        # ----------- 分布式训练判断 -----------
        self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

    # -------------------------
    # 文本嵌入
    # -------------------------
    def embed_text(self, text):
        batch, device = text.shape[0], text.device
        seq = text.shape[1]

        text_tokens = self.token_emb(text)

        # 在末尾拼接 text CLS token
        text_cls_tokens = repeat(self.text_cls_token, 'd -> b 1 d', b=batch)
        text_tokens = torch.cat((text_tokens, text_cls_tokens), dim=-2)

        # 构造注意力 mask (避免 CLS token attends 到 padding)
        cls_mask = rearrange(text != self.pad_id, 'b j -> b 1 j')
        attn_mask = F.pad(cls_mask, (0, 1, seq, 0), value=True)

        # 经过单模态层 (仅文本自注意力)
        for attn_ff in self.unimodal_layers:
            text_tokens = attn_ff(text_tokens, attn_mask=attn_mask)

        # 分离 token 表示和 CLS 表示
        text_tokens, text_cls_tokens = text_tokens[:, :-1], text_tokens[:, -1]
        text_embeds = self.text_cls_norm(text_cls_tokens)

        return text_embeds, text_tokens

    # -------------------------
    # 图像嵌入
    # -------------------------
    def embed_image(self, images=None, image_tokens=None):
        # 如果输入的是图像，先通过 img_encoder 得到图像 token
        assert not (exists(images) and exists(image_tokens))
        if exists(images):
            assert exists(self.img_encoder), '必须传入 img_encoder 才能处理原始图像'
            image_tokens = self.img_encoder(images)

        # attention pool 图像 token
        img_queries = repeat(self.img_queries, 'n d -> b n d', b=image_tokens.shape[0])
        img_queries = self.img_attn_pool(img_queries, image_tokens)
        img_queries = self.img_attn_pool_norm(img_queries)

        # 返回 CLS (全局 embedding) + 其余 token (用于 cross attention)
        return img_queries[:, 0], img_queries[:, 1:]

    # -------------------------
    # 前向传播
    # -------------------------
    def forward(
        self,
        text,
        images=None,
        image_tokens=None,
        labels=None,
        return_loss=False,
        return_embeddings=False
    ):
        batch, device = text.shape[0], text.device

        # 如果需要计算 caption loss 且未提供标签，自动构造 labels
        if return_loss and not exists(labels):
            text, labels = text[:, :-1], text[:, 1:]

        # 获取文本嵌入 (CLS + token)
        text_embeds, text_tokens = self.embed_text(text)

        # 获取图像嵌入 (CLS + token)
        image_embeds, image_tokens = self.embed_image(images=images, image_tokens=image_tokens)

        # 如果只需要 embedding，则直接返回
        if return_embeddings:
            return text_embeds, image_embeds

        # 经过多模态层 (自注意力 + 跨模态注意力)
        for attn_ff, cross_attn in self.multimodal_layers:
            text_tokens = attn_ff(text_tokens)                    # 文本自注意力
            text_tokens = cross_attn(text_tokens, image_tokens)   # 文本 attends 图像

        # 输出预测 logits
        logits = self.to_logits(text_tokens)

        # 如果不需要 loss，直接返回 logits
        if not return_loss:
            return logits

        # ----------- Caption Loss (交叉熵损失) -----------
        ce = F.cross_entropy
        logits = rearrange(logits, 'b n c -> b c n')  # [batch, vocab, seq]
        caption_loss = ce(logits, labels, ignore_index=self.pad_id)
        caption_loss = caption_loss * self.caption_loss_weight

        # ----------- Contrastive Loss (对比学习) -----------
        text_latents = self.text_to_latents(text_embeds)
        image_latents = self.img_to_latents(image_embeds)

        # 分布式训练下做 all_gather
        if self.is_distributed:
            latents = torch.stack((text_latents, image_latents), dim=1)
            latents = all_gather(latents)
            text_latents, image_latents = latents.unbind(dim=1)

        # 相似度计算
        sim = einsum('i d, j d -> i j', text_latents, image_latents)
        sim = sim * self.temperature.exp()
        contrastive_labels = torch.arange(batch, device=device)

        # 对称 InfoNCE 损失
        contrastive_loss = (ce(sim, contrastive_labels) + ce(sim.t(), contrastive_labels)) * 0.5
        contrastive_loss = contrastive_loss * self.contrastive_loss_weight

        # 返回总损失 (caption + contrastive)
        return caption_loss + contrastive_losss
```
