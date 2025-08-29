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

## 相关工作

**视觉预训练**

早期的视觉模型大多依赖于在大规模标注数据（如 ImageNet、Instagram、JFT）上对卷积网络或 Transformer 进行预训练，从而解决分类、定位、分割、视频识别、跟踪等视觉识别任务。
近年来，自监督视觉预训练逐渐兴起：

* **BEiT** 借鉴 BERT 思路，提出了基于掩码图像建模的任务，并用量化后的视觉 token id 作为预测目标。

* **MAE 和 SimMIM** 移除了图像 tokenizer，直接使用轻量级解码器或投影层回归像素值。

但这些方法的局限在于：它们只学习视觉模态模型，无法应用到需要图像与文本 **联合推理** 的任务。

---

**视觉-语言预训练（VLP）**

VLP 的目标是让模型能够在融合框架中联合建模视觉和语言。

* 早期方法（LXMERT、UNITER、VinVL）依赖目标检测器（如 Faster R-CNN）提取图像特征。

* 后续方法（ViLT、VLMo）则直接将视觉和语言 Transformer 统一起来，从零开始训练一个多模态 Transformer。

---

**图文基础模型**

最近的研究进一步提出了 **图文基础模型**，它们统一了视觉预训练和视觉-语言预训练：

* **CLIP 和 ALIGN**：利用噪声图文对数据，通过对比学习目标训练双编码器，学习到跨模态对齐能力，并能实现零样本图像分类。

* **Florence**：在 CLIP/ALIGN 的思路上提出统一的对比目标，并训练能适配于更广泛基准的基础模型。

* **LiT 和 BASIC**：先在大规模图像标注数据上用交叉熵训练，再在噪声图文对数据集上用对比损失微调，从而提升零样本图像分类性能。

* **生成式方法（如 \[16, 17, 34]）**：采用编码器-解码器架构并引入生成式损失，在视觉-语言基准任务上表现优异，同时视觉编码器在图像分类上依然具备竞争力。

---

**与现有方法的对比**

本研究提出的 **CoCa**，专注于从零开始，在单一预训练阶段完成图文统一，从而避免多阶段训练（如 ALBEF 那样的先单模态、再多模态流程）。已有一些方法尝试过类似思路（如 ALBEF），但它们存在复杂的训练需求：

* **计算开销**：CoCa 在一批图文对上只需一次前向与反向传播，而 ALBEF 需要两次（一次使用被扰动输入，一次使用未扰动输入）。

* **训练方式**：CoCa 直接在两个目标函数下从零开始训练，而 ALBEF 依赖预训练的视觉和文本编码器，并需要额外的训练机制（如动量模块）。

* **生成式优势**：CoCa 的解码器架构结合生成式损失，更自然地支持图像描述任务，同时还能直接实现零样本学习。

## 方法

我们首先回顾三类利用自然语言监督的基础模型家族：**单编码器分类预训练、双编码器对比学习、编码器-解码器图像描述生成**。随后介绍 **Contrastive Captioners (CoCa)**，它在一个简单的架构下融合了对比学习和图像到文本生成的优势。最后讨论 CoCa 模型如何通过零样本迁移或最小任务微调快速应用到下游任务。

### 自然语言监督

**单编码器分类（Single-Encoder Classification）**

经典的单编码器方法通过在大规模人工标注图像数据集（如 ImageNet、Instagram 或 JFT）上进行图像分类来预训练视觉编码器。注释文本的词汇通常是固定的，图像标注一般被映射为离散类别向量，并使用交叉熵损失训练：

$$
L_{Cls} = -p(y) \log q_\theta(x)
$$

其中 $p(y)$ 是从真实标签 $y$ 得到的 one-hot、多-hot 或平滑标签分布。训练完成后，学习到的图像编码器可作为通用视觉表示提取器，用于下游任务。

---

**双编码器对比学习（Dual-Encoder Contrastive Learning）**

相比单编码器分类需要人工标注和数据清理，双编码器方法利用**大规模噪声文本描述**，并引入可学习的文本编码器来编码自由形式文本。两个编码器通过对比目标联合优化：

$$
L_{Con} = - \frac{1}{N} \sum_{i=1}^{N} \Big( \log \frac{\exp(x_i^\top y_i / \sigma)}{\sum_{j=1}^{N} \exp(x_i^\top y_j / \sigma)} \ + \ \log \frac{\exp(y_i^\top x_i / \sigma)}{\sum_{j=1}^{N} \exp(y_i^\top x_j / \sigma)} \Big)
$$

其中 $x_i$ 和 $y_j$ 分别是第 $i$ 对图像和第 $j$ 对文本的归一化嵌入，$N$ 是批量大小，$\sigma$ 是温度参数。

除了图像编码器，双编码器方法还学习了对齐的文本编码器，使得模型可以进行**跨模态对齐应用**，如图文检索和零样本图像分类。实证结果显示，这种零样本分类在受损或分布外图像上更加稳健。

---

**编码器-解码器图像描述生成（Encoder-Decoder Captioning）**

与双编码器整体编码文本不同，生成式方法（也称 captioner）追求更细粒度的表示，需要模型自回归地预测文本 $y$ 的每个 token。遵循标准的编码器-解码器架构：

* 图像编码器提供潜在编码特征（如使用 Vision Transformer 或卷积网络）。

* 文本解码器通过最大化条件概率来学习配对文本的生成：

$$
L_{Cap} = - \sum_{t=1}^{T} \log P_\theta(y_t | y_{<t}, x)
$$

编码器-解码器训练使用 **teacher-forcing**，以并行化计算并提高学习效率。

与前述方法不同，captioner 不仅提供**联合图文表示**以用于视觉-语言理解任务，还可以直接应用于自然语言生成的图像描述任务。


好的，我来帮你把 **3.2 Contrastive Captioners Pretraining** 这一节完整翻译成中文，保持通俗易懂，但同时尽量保留论文中的全部信息。

---
### 对比描述预训练（Contrastive Captioners Pretraining）

![](coca/2.png)

**整体架构**（如图2所示）：我们提出的 CoCa（Contrastive Captioner）是一种简单的编码器-解码器方法，能够自然地融合三种训练范式。与标准的图文编码器-解码器模型类似，CoCa 使用神经网络编码器（默认采用 Vision Transformer (ViT) ，当然也可以是其他图像编码器，如 ConvNets ）将图像编码为潜在表示，并用带因果掩码的 Transformer 解码器对文本进行解码。

与标准的 Transformer 解码器不同，CoCa 在解码器的前半部分省略了 cross-attention（交叉注意力），只保留单模态的文本表示学习；而在后半部分，则引入 cross-attention，与图像编码器交互，生成跨模态的图文联合表示。这样，CoCa 解码器可以**同时产生单模态和多模态的文本表示**，从而在同一架构下联合优化 **对比损失** 和 **生成损失**：

$$
L_{CoCa} = \lambda_{Con} \cdot L_{Con} + \lambda_{Cap} \cdot L_{Cap}
$$

其中，$\lambda_{Con}$ 和 $\lambda_{Cap}$ 是损失函数的加权系数。值得注意的是，单编码器的交叉熵分类目标其实可以看作是一种特殊的生成方法——只不过词汇表是标签名称集合。

---

**解耦的文本解码器与 CoCa 架构**：

描述生成任务要求模型优化条件概率 $P(y|x)$，而对比学习任务则依赖无条件的文本表示。为了兼顾这两者，我们提出了一种**解耦的解码器设计**，将解码器分为单模态部分和多模态部分：

* **底部 $n_{uni}$ 层（单模态解码器层）**：只用因果掩码的自注意力机制来编码输入文本，不使用 cross-attention，得到单模态文本向量表示。

* **顶部 $n_{multi}$ 层（多模态解码器层）**：在因果掩码自注意力的基础上，再结合 cross-attention 与图像编码器输出交互，生成多模态表示。

所有解码器层都禁止 token 关注未来的 token，因此自然适用于自回归式的生成目标 $L_{Cap}$。而对于对比学习目标 $L_{Con}$，我们在输入句子末尾附加一个可学习的 [CLS] token，并将其在**单模态解码器输出**中的向量作为文本嵌入。实验中我们将解码器平分为两部分，即 $n_{uni} = n_{multi}$。

在图像输入方面，我们遵循 ALIGN 的设定，使用 $288 \times 288$ 的图像分辨率和 $18 \times 18$ 的 patch size，得到 256 个图像 token。我们的最大模型 CoCa（简称 “CoCa”）采用 ViT-giant 结构，图像编码器有 10 亿参数，连同文本解码器在内总参数量为 21 亿。此外，我们还探索了两个更小的变体：“CoCa-Base”和“CoCa-Large”（见表1）。

![](coca/3.png)

---

**注意力池化（Attentional Poolers）**：

需要强调的是，对比损失只使用一个图像嵌入，而标准的图文编码-解码模型通常会利用整段图像 token 序列进行 cross-attention。我们的初步实验发现：

* 使用单个池化的全局图像表示，有利于**视觉识别类任务**。

* 使用更多图像 token（更细粒度表示），更适合**多模态理解类任务**，因为这类任务需要区域级别的特征。

因此，CoCa 引入了**任务特定的注意力池化（task-specific attentional pooling）**，为不同目标和下游任务定制图像表示。具体来说，pooler 是一个单层多头注意力结构，包含 $n_{query}$ 个可学习查询，图像编码器的输出作为 key 和 value。通过这种机制，模型能够学习如何根据不同任务的需求，聚合出不同长度的图像嵌入（如图2所示）。

在预训练中：

* **生成损失**：使用 $n_{query} = 256$，保留更多图像 token。

* **对比损失**：使用 $n_{query} = 1$，只提取全局嵌入。

这样不仅能满足不同任务的需求，还让 pooler 自然地成为任务适配器。

---

**预训练效率**：

解耦的自回归解码器设计的一个关键优势在于：它能高效计算两种损失。因为单向语言模型在完整句子上用因果掩码训练，所以只需**一次前向传播**，就能同时获得对比损失和生成损失（相比双向方法需要两次前向传播）。这意味着两种损失的大部分计算是共享的，CoCa 相比标准的编码器-解码器模型只增加了极小的计算开销。

另一方面，许多现有方法通常分阶段训练模型组件，且需要在不同数据源/模态之间切换。而 CoCa 则是**从零开始进行端到端预训练**，数据来源既包括人工标注的图像，也包括带噪声的网页 alt-text 图像，并且统一地将所有标签视为文本，兼顾对比和生成两类目标。
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
