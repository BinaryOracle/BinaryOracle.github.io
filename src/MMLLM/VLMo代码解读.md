---
title: VLMo 模型代码解读
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-14
author:
  - BinaryOracle
---

`VLMO 模型代码解读` 

<!-- more -->

> 论文链接: [VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts](https://arxiv.org/abs/2111.02358)
> 代码链接: [https://github.com/microsoft/unilm/tree/master/vlmo](https://github.com/microsoft/unilm/tree/master/vlmo)

## 前置知识

VLMO 模型的代码实现中主要使用了以下两个库，如果不提前了解一下库的基本用法，可能会导致读不懂代码实现：

1. [Sacred 实验管理框架](https://sacred.readthedocs.io/en/stable/quickstart.html)

2. [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/starter/introduction.html)

## 数据处理

## MOME（Mixture of Multimodal Experts）Transformer

VLMO 论文中所提到的 `MOME Transformer` 的代码实现对应的类是 `MultiWayTransformer` , 本节我们将一点点完成该类代码的拆解; 首先，既然是 `混合多模态专家模型`, 那么它就需要具有同时处理图像和文本的能力；对于输入的图像，第一步需要完成图像的切片和嵌入，该功能由 `PatchEmbed` 类负责完成，具体代码实现如下:

```python
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    将输入的图像切分成小 patch，并通过卷积映射到指定的 embedding 维度。
    这是 Vision Transformer (ViT) 中常用的图像嵌入方法。
    """

    def __init__(
        self,
        img_size=224,          # 输入图像的高度和宽度（默认224x224）
        patch_size=16,         # 每个patch的高度和宽度（默认16x16）
        in_chans=3,            # 输入通道数，彩色图像通常为3
        embed_dim=768,         # 输出 embedding 的维度
        no_patch_embed_bias=False,  # 是否在卷积层中去掉偏置
    ):
        super().__init__()

        # 将 img_size 和 patch_size 转换为 (height, width) 的 tuple
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        # 计算图像能切分成多少个 patch
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])

        # 保存 patch 行列数，用于位置编码或其他处理
        self.patch_shape = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        # 定义卷积层，将图像切分成 patch 并映射到 embedding 维度
        # 注意：kernel_size = patch_size, stride = patch_size，这样每个卷积核对应一个 patch
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=False if no_patch_embed_bias else True,
        )

    def forward(self, x):
        B, C, H, W = x.shape
        # 检查输入图像尺寸是否与初始化尺寸匹配
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # 卷积映射，将图像切分成 patch 并生成 embedding
        x = self.proj(x)
        # 输出 shape: [B, embed_dim, H_patch, W_patch]
        return x
```

`PatchEmbed` 只完成了借助卷积对图像进行前置处理的步骤，`MultiWayTransformer` 类额外提供了 `visual_embed` 方法来完成与 `文本Token` 统一形式的 `视觉Token` 的构建:

```python
class MultiWayTransformer(nn.Module):

    def visual_embed(self, _x):
        """
        将输入的图像张量 _x 转换为视觉 token embedding。
        步骤包括：
        1. patch embedding
        2. 展平并调整维度
        3. 添加 cls token
        4. 添加位置编码（可选）
        5. 添加 dropout
        """

        # 1. 将图像切分成 patch 并映射到 embedding 维度
        x = self.patch_embed(_x)  # shape: [B, embed_dim, H_patch, W_patch]

        # 2. 展平 patch 并调整维度，使其变为序列形式 [B, num_patches, embed_dim]
        x = x.flatten(2).transpose(1, 2)  # flatten 从 H*W -> L，transpose 调整维度

        B, L, _ = x.shape  # B: batch size, L: patch 数量, _: embedding 维度

        # 3. 扩展 cls_token 到 batch 大小，并与 patch embedding 拼接
        cls_tokens = self.cls_token.expand(B, -1, -1)  # shape: [B, 1, embed_dim]
        x = torch.cat((cls_tokens, x), dim=1)          # 拼接后 shape: [B, L+1, embed_dim]

        # 4. 如果有位置编码，则加上
        if self.pos_embed is not None:
            x = x + self.pos_embed  # shape: [B, L+1, embed_dim]

        # 5. 添加 dropout，增加模型鲁棒性
        x = self.pos_drop(x)

        # 6. 构建 mask，这里全 1 表示所有 token 都有效
        x_mask = torch.ones(x.shape[0], x.shape[1])  # shape: [B, L+1]

        return x, x_mask  # 返回 token embedding 和 mask
```

`MultiWayTransformer` 类没有直接对外提供现成的 `forward` 方法实现，而是由调用方 `VLMo` 类负责完成前向传播流程的组织，所以下面我们将首先对其 `init` 方法进行分析，看看它内部包含哪些重要组件:

```python
class MultiWayTransformer(nn.Module):
    def __init__(
        self,
        img_size=224,                 # 输入图像尺寸
        patch_size=16,                # patch 大小
        in_chans=3,                   # 输入通道数（例如 RGB 图像为 3）
        embed_dim=768,                # embedding 维度
        depth=12,                     # transformer block 层数
        num_heads=12,                 # attention 头数
        mlp_ratio=4.0,                # MLP 隐层维度与 embedding 维度的比例
        qkv_bias=True,                # 是否在 QKV 上使用偏置
        qk_scale=None,                # 可手动设置 QK 缩放值
        drop_rate=0.0,                # dropout 概率
        attn_drop_rate=0.0,           # attention dropout 概率
        drop_path_rate=0.0,           # stochastic depth 概率
        norm_layer=None,              # normalization 层类型
        need_relative_position_embed=True,  # 是否使用相对位置编码
        use_abs_pos_emb=False,        # 是否使用绝对位置编码
        layer_scale_init_values=0.1,  # LayerScale 初始化值
        vlffn_start_layer_index=10,   # 从第几层开始使用 VL-FFN
        config=None,                  # 其他配置（如从 pytorch-lightning 传入）
        **kwargs,                     # 接收 timm 或其他传入的额外参数
    ):
        """
        MultiWayTransformer 构造函数，初始化视觉与文本 transformer 的参数。
        """

        super().__init__()

        # 如果传入 config，则覆盖 drop_path_rate
        drop_path_rate = drop_path_rate if config is None else config["drop_path_rate"]

        # 保存是否使用绝对位置编码和相对位置编码的标志
        self.use_abs_pos_emb = use_abs_pos_emb
        self.need_relative_position_embed = need_relative_position_embed

        # 记录 embedding 特征维度
        self.num_features = (self.embed_dim) = embed_dim  # num_features 与 embed_dim 保持一致

        # 默认归一化层，如果未指定则使用 LayerNorm
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # PatchEmbedding，将图像切分为 patch 并映射到 embed_dim
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        # 记录 patch 数量和 patch 尺寸
        num_patches = self.patch_embed.num_patches
        self.patch_size = patch_size
        self.num_heads = num_heads

        # VL-FFN 从哪一层开始
        self.vlffn_start_layer_index = vlffn_start_layer_index

        # 针对 text-only pretraining，如果 textmlm loss 大于 0，则从最后一层开始使用 VL-FFN
        if config["loss_names"]["textmlm"] > 0:
            self.vlffn_start_layer_index = depth

        # 类别 token 参数（用于全局聚合）
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # 绝对位置编码参数（可选）
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) if self.use_abs_pos_emb else None

        # dropout 层
        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth，每层的 drop_path 概率线性增长
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # 构建 transformer block 列表
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                    with_vlffn=(i >= self.vlffn_start_layer_index),  # 超过起始层索引才启用 VL-FFN
                    layer_scale_init_values=layer_scale_init_values,
                    max_text_len=config["max_text_len"],
                )
                for i in range(depth)
            ]
        )

        # transformer 最后的归一化层
        self.norm = norm_layer(embed_dim)

        # 参数初始化
        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)  # 初始化所有权重
```

`MultiWayTransformer` 支持同时处理文本和图像模态，这个功能具体实现在其内部的 `Transformer Block` 中:

```python
class Block(nn.Module):
    def __init__(
        self,
        dim,                # 输入特征维度
        num_heads,          # 多头注意力的头数
        mlp_ratio=4.0,      # MLP 隐藏层维度与输入维度的比例
        qkv_bias=False,     # QKV 是否使用偏置
        qk_scale=None,      # QK 缩放因子（覆盖默认 head_dim ** -0.5）
        drop=0.0,           # Dropout 概率
        attn_drop=0.0,      # 注意力权重的 Dropout 概率
        drop_path=0.0,      # Stochastic Depth 概率
        act_layer=nn.GELU,  # 激活函数类型
        norm_layer=nn.LayerNorm, # 归一化层类型
        with_vlffn=False,   # 是否使用跨模态 MLP（Vision-Language Feed-Forward Network）
        layer_scale_init_values=0.1, # LayerScale 初始化值
        max_text_len=40,    # 最大文本序列长度
    ):
        super().__init__()

        # 第一个 LayerNorm（作用于注意力之前）
        self.norm1 = norm_layer(dim)

        # 多头注意力机制
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )

        # DropPath（随机丢弃整个残差分支）或恒等映射
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        # 第二阶段的 LayerNorm（针对文本和图像分别有独立的归一化层）
        self.norm2_text = norm_layer(dim)
        self.norm2_imag = norm_layer(dim)

        # MLP 隐藏层维度
        mlp_hidden_dim = int(dim * mlp_ratio)

        # 文本模态的 MLP
        self.mlp_text = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # 图像模态的 MLP
        self.mlp_imag = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

        # 跨模态 MLP（仅在 with_vlffn=True 时使用）
        self.mlp_vl = None
        if with_vlffn:
            self.mlp_vl = Mlp(
                in_features=dim,
                hidden_features=mlp_hidden_dim,
                act_layer=act_layer,
                drop=drop,
            )
            self.norm2_vl = norm_layer(dim)

        # LayerScale 参数（gamma_1 作用于注意力分支，gamma_2 作用于 MLP 分支）
        self.gamma_1 = (
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_values is not None else 1.0
        )
        self.gamma_2 = (
            nn.Parameter(layer_scale_init_values * torch.ones((dim)), requires_grad=True)
            if layer_scale_init_values is not None else 1.0
        )

        # 最大文本长度（在拆分多模态输入时使用）
        self.max_text_len = max_text_len

    def forward(self, x, mask=None, modality_type=None, relative_position_bias=None):
        """
        Args:
            x: 输入特征 [B, L, C]
            mask: 注意力掩码（可选）
            modality_type: 输入模态类型 ("image", "text", 或 None 表示多模态）
            relative_position_bias: 相对位置编码偏置
        """

        # ====== 注意力子层（带残差连接 + LayerScale + DropPath）======
        x = x + self.drop_path(
            self.gamma_1 * self.attn(
                self.norm1(x),
                mask=mask,
                relative_position_bias=relative_position_bias
            )
        )

        # ====== 前馈网络子层（根据模态类型选择不同 MLP）======
        if modality_type == "image":
            # 仅图像模态
            x = x + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x)))
        elif modality_type == "text":
            # 仅文本模态
            x = x + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x)))
        else:
            # 多模态情况
            if self.mlp_vl is None:
                # 分开处理文本和图像序列
                x_text = x[:, : self.max_text_len]   # 前 max_text_len 为文本
                x_imag = x[:, self.max_text_len :]   # 剩余部分为图像
                x_text = x_text + self.drop_path(self.gamma_2 * self.mlp_text(self.norm2_text(x_text)))
                x_imag = x_imag + self.drop_path(self.gamma_2 * self.mlp_imag(self.norm2_imag(x_imag)))
                # 合并回一个序列
                x = torch.cat([x_text, x_imag], dim=1)
            else:
                # 跨模态 MLP
                x = x + self.drop_path(self.gamma_2 * self.mlp_vl(self.norm2_vl(x)))

        return x
```
**LayerScale 技术:**  在深层 Transformer 中，如果直接把残差相加，可能导致梯度爆炸或梯度消失;  LayerScale 允许网络自己调节每一层的残差输出强度，从而改善训练稳定性。提高深层网络可训练性； 对深层 ViT（几十甚至上百层）非常有效，减少了训练前期的收敛难度。

简单理解：

```python
x = x + γ1 * Attention(x)
x = x + γ2 * MLP(x)
```

* γ1、γ2 = 可学习缩放因子

* 作用 = 控制残差贡献，稳定训练

* 为什么分开？因为 Attention 和 MLP 输出的统计特性不同，需要不同的缩放系数


`Attention` 模块的代码属于模版代码，不涉及新技术的引入，代码实现如下所示:

```python
class Attention(nn.Module):
    def __init__(
        self,
        dim,                # 输入特征的维度 (embedding dimension)
        num_heads=8,        # 多头注意力的头数
        qkv_bias=False,     # 是否为 Q、K、V 添加可学习偏置
        qk_scale=None,      # QK 点积的缩放因子（可覆盖默认值）
        attn_drop=0.0,      # 注意力权重的 Dropout 概率
        proj_drop=0.0,      # 输出投影层的 Dropout 概率
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads  # 每个注意力头的维度
        # QK 缩放因子，默认为 1/sqrt(head_dim)，防止点积结果过大
        self.scale = qk_scale or head_dim ** -0.5

        # 线性层生成 Q、K、V（一次性计算 dim → 3*dim）
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        
        # 如果需要 Q、V 偏置，则单独为 Q 和 V 创建可学习参数
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(dim))
            self.v_bias = nn.Parameter(torch.zeros(dim))
        else:
            self.q_bias = None
            self.v_bias = None
        
        # 注意力权重的 Dropout
        self.attn_drop = nn.Dropout(attn_drop)
        # 注意力结果的输出投影层
        self.proj = nn.Linear(dim, dim)
        # 输出投影后的 Dropout
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x, mask=None, relative_position_bias=None):
        """
        Args:
            x: 输入张量 (B, N, C)，
               B=批大小，N=序列长度，C=通道数(embedding dim)
            mask: 注意力掩码 (B, N)，用于屏蔽无效位置
            relative_position_bias: 相对位置编码 (num_heads, N, N)
        """
        B, N, C = x.shape  # 取出批大小、序列长度、通道数

        # 处理 Q、K、V 偏置
        qkv_bias = None
        if self.q_bias is not None:
            # 拼接 Q 偏置、K 偏置(全0)、V 偏置
            qkv_bias = torch.cat((
                self.q_bias,
                torch.zeros_like(self.v_bias, requires_grad=False),
                self.v_bias
            ))

        # 线性映射得到 Q、K、V（在这里一次性计算）
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        # 变形为 (3, B, num_heads, N, head_dim)，并调整维度顺序
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)

        # 拆分 Q、K、V
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 缩放 Q
        q = q * self.scale
        # QK^T 得到注意力分数矩阵
        attn = (q.float() @ k.float().transpose(-2, -1))
        
        # 如果有相对位置偏置，则加上
        if relative_position_bias is not None:
            attn = attn + relative_position_bias.unsqueeze(0)

        # 如果有 mask（如解码器中的自回归屏蔽）
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))

        # 对最后一维做 softmax 得到注意力权重
        attn = attn.softmax(dim=-1).type_as(x)
        # 对注意力权重做 Dropout
        attn = self.attn_drop(attn)

        # 注意力加权 V，然后还原维度为 (B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 输出投影
        x = self.proj(x)
        # 投影结果 Dropout
        x = self.proj_drop(x)
        return x
```

## VLMo 

主模型 `VLMo` 由于使用了 `PyTorch Lightning` 实验全流程管理框架，使得其代码看起来并不常规，但是其本质还是借助 `模版方法设计模型` 抽取出一套通用的模版流程，并通过在各个模版节点预留钩子函数的方式，使得用户可以在不改变模版流程的情况下，自定义模型的行为; 

因此，我们首先用一幅图理清楚 `PyTorch Lightning` 预留的这套模版流程是怎么设计的:

![图片取至 [PytorchLightning : Model calls order](https://stackoverflow.com/questions/73985576/pytorchlightning-model-calls-order?utm_source=chatgpt.com)](VLMO/1.png)




