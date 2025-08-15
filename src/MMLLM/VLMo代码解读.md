---
title: VLMo 模型代码解读
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-15
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

> `VLMo` 模型代码实现是基于 `ViLT` 模型代码进行修改的，因此如果研究过 `ViLT` 代码实现的同学，对 `VLMo` 模型的代码实现应该比较亲切。

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

```python
1. 初始化阶段
   ├─ 用户创建 LightningModule 和 Trainer                   (用户代码)
   ├─ LightningModule.configure_optimizers()                 (LightningModule)
   ├─ Trainer 配置 logger、callbacks、accelerator、分布式   (Trainer 内部)

2. 数据准备阶段
   ├─ LightningDataModule.prepare_data()                     (LightningDataModule, global_rank=0)
   └─ LightningDataModule.setup(stage)                       (LightningDataModule, 每个进程, stage ∈ {'fit','validate','test','predict'})

3. 数据加载阶段
   ├─ LightningDataModule.train_dataloader()                (LightningDataModule)
   ├─ LightningDataModule.val_dataloader()                  (LightningDataModule)
   └─ LightningDataModule.test_dataloader()                 (LightningDataModule)

4. 训练阶段（fit）
   ├─ Trainer.on_fit_start()                                  (Trainer 调用所有 callbacks.on_fit_start)
   └─ Epoch 循环 (for epoch in max_epochs)
       ├─ Trainer.on_train_epoch_start()                     (Trainer callbacks)
       └─ Batch 循环 (for batch in train_dataloader)
            ├─ Trainer.on_train_batch_start(batch, batch_idx)   (Trainer callbacks)
            ├─ LightningModule.training_step(batch, batch_idx)  (LightningModule)
            ├─ Trainer.on_before_zero_grad(optimizer)           (Trainer callbacks)
            ├─ optimizer.zero_grad()                             (PyTorch)
            ├─ loss.backward()                                   (PyTorch)
            ├─ Trainer.on_after_backward()                        (Trainer callbacks)
            ├─ Trainer.on_before_optimizer_step(optimizer)       (Trainer callbacks)
            ├─ optimizer.step()                                   (PyTorch)
            └─ Trainer.on_train_batch_end(output, batch, batch_idx)(Trainer callbacks)
       ├─ LightningModule.training_epoch_end(outputs)         (LightningModule)
       └─ Trainer.on_train_epoch_end()                        (Trainer callbacks)

       └─ 验证阶段（每个 epoch 后可选）
            ├─ Trainer.on_validation_start()                  (Trainer callbacks)
            ├─ model.eval(), torch.no_grad()                  (Trainer 内部)
            └─ 循环 val_dataloader
                 ├─ LightningModule.validation_step(batch, batch_idx)     (LightningModule)
                 ├─ LightningModule.validation_step_end(output)           (LightningModule)
                 └─ 汇总 outputs
            ├─ LightningModule.validation_epoch_end(outputs)               (LightningModule)
            └─ Trainer.on_validation_epoch_end()                            (Trainer callbacks)
   └─ Trainer.on_fit_end()                                       (Trainer callbacks)

5. 测试阶段（test）
   ├─ Trainer.on_test_start()                                     (Trainer callbacks)
   ├─ model.eval(), torch.no_grad()                               (Trainer 内部)
   └─ 循环 test_dataloader
        ├─ LightningModule.test_step(batch, batch_idx)            (LightningModule)
        ├─ LightningModule.test_step_end(output)                  (LightningModule)
        └─ 汇总 outputs
   ├─ LightningModule.test_epoch_end(outputs)                     (LightningModule)
   └─ Trainer.on_test_end()                                        (Trainer callbacks)

6. 预测阶段（predict）
   ├─ Trainer.on_predict_start()                                  (Trainer callbacks)
   ├─ model.eval(), torch.no_grad()                               (Trainer 内部)
   └─ 循环 predict_dataloader
        ├─ LightningModule.predict_step(batch, batch_idx)         (LightningModule)
        ├─ LightningModule.predict_step_end(output)               (LightningModule)
        └─ 汇总 outputs
   ├─ LightningModule.predict_epoch_end(outputs)                  (LightningModule)
   └─ Trainer.on_predict_end()                                     (Trainer callbacks)
```

下面我们将结合上面的模版流程，分析一下 `VLMo` 在模版流程的各种阶段都做了什么:

### 数据模块

一般模型训练都会加载多个来源不同的开源或私有数据集，`VLMo` 也不例外，因此 `VLMo` 提供了 `MTDataModule` 类用于完成多数据源加载的任务:

```python
class MTDataModule(LightningDataModule):
    def __init__(self, _config, dist=False):
        """
        多任务/多数据集 DataModule，负责管理多个子数据集
        Args:
            _config: 配置字典，包含数据集 key 和其他参数
            dist: 是否使用分布式采样
        """
        datamodule_keys = _config["datasets"]
        assert len(datamodule_keys) > 0

        super().__init__()

        # 保存数据集 key 和对应的数据模块实例
        self.dm_keys = datamodule_keys
        self.dm_dicts = {key: _datamodules[key](_config) for key in datamodule_keys}
        self.dms = [v for k, v in self.dm_dicts.items()]

        # 从第一个数据模块读取通用配置
        self.batch_size = self.dms[0].batch_size
        self.vocab_size = self.dms[0].vocab_size
        self.num_workers = self.dms[0].num_workers

        self.dist = dist  # 是否使用分布式采样

    def prepare_data(self):
        """
        数据准备阶段（只在主进程调用一次）
        生命周期阶段: Trainer 调用 prepare_data()
        """
        for dm in self.dms:
            dm.prepare_data()  # 调用每个子数据模块的 prepare_data

    def setup(self, stage):
        """
        数据集构建阶段，每个进程都会调用
        Args:
            stage: 'fit', 'validate', 'test', 'predict' 等
        """
        for dm in self.dms:
            dm.setup(stage)  # 调用子数据模块的 setup

        # 合并各个子数据集
        self.train_dataset = ConcatDataset([dm.train_dataset for dm in self.dms])
        self.val_dataset = ConcatDataset([dm.val_dataset for dm in self.dms])
        self.test_dataset = ConcatDataset([dm.test_dataset for dm in self.dms])

        # 保存 tokenizer 和 collate 函数
        self.tokenizer = self.dms[0].tokenizer
        self.collate = functools.partial(
            self.dms[0].train_dataset.collate,
            mlm_collator=self.dms[0].mlm_collator,
        )

        # 分布式采样器
        if self.dist and torch.distributed.is_initialized():
            self.train_sampler = DistributedSampler(self.train_dataset, shuffle=True)
            self.val_sampler = DistributedSampler(self.val_dataset, shuffle=True)
            self.test_sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            self.train_sampler = None
            self.val_sampler = None
            self.test_sampler = None

    def train_dataloader(self):
        """
        返回训练 DataLoader
        生命周期阶段: Trainer.fit() 内部调用
        """
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.train_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader

    def val_dataloader(self, batch_size=None):
        """
        返回验证 DataLoader
        生命周期阶段: Trainer.validate() 或 Trainer.fit() 内部验证调用
        """
        loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size if batch_size is not None else self.batch_size,
            sampler=self.val_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader

    def test_dataloader(self):
        """
        返回测试 DataLoader
        生命周期阶段: Trainer.test() 内部调用
        """
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            sampler=self.test_sampler,
            num_workers=self.num_workers,
            collate_fn=self.collate,
        )
        return loader
```

`_datamodules` 字典中保存了 `VLMo` 所使用到的所有数据集对应的 `DataModule` 实现类:

```python
_datamodules = {
    "vg": VisualGenomeCaptionDataModule,
    "f30k": F30KCaptionKarpathyDataModule,
    "coco": CocoCaptionKarpathyDataModule,
    "gcc": ConceptualCaptionDataModule,
    "sbu": SBUCaptionDataModule,
    "wikibk": WikibkDataModule,
    "vqa": VQAv2DataModule,
    "nlvr2": NLVR2DataModule,
}
```
当子实现类比较多的时候，自然会存在一些重复性操作，因此 `VLMo` 模型的代码实现中额外抽取了一个抽象类 `BaseDataModule` 用于定义重复性的模版流程，以此来简化子实现类需要做的操作:

```python
class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        """
        基础 DataModule 类，支持图文/文本数据集
        Args:
            _config: 配置字典，包含数据路径、batch_size、tokenizer 等信息
        """
        super().__init__()

        # 数据目录
        self.data_dir = _config["data_root"]

        # DataLoader 参数
        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size

        # 数据处理参数
        self.image_size = _config["image_size"]
        self.max_text_len = _config["max_text_len"]
        self.draw_false_image = _config["draw_false_image"]
        self.draw_false_text = _config["draw_false_text"]
        self.image_only = _config["image_only"]
        self.text_only = _config["text_only"]

        # 数据增强/transform 配置
        self.train_transform_keys = (
            ["default_train"]
            if len(_config["train_transform_keys"]) == 0
            else _config["train_transform_keys"]
        )
        self.val_transform_keys = (
            ["default_val"]
            if len(_config["val_transform_keys"]) == 0
            else _config["val_transform_keys"]
        )

        # tokenizer
        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        # collator: 用于 MLM（mask language model）训练
        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )
        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )

        # setup 状态标志，确保 setup 只执行一次
        self.setup_flag = False

    @property
    def dataset_cls(self):
        """
        子类必须实现
        返回 dataset 类（通常是 Dataset 子类）
        """
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        """
        子类必须实现
        返回数据集名称
        """
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        """
        构建训练数据集
        生命周期阶段: setup() 调用
        """
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            self.train_transform_keys,
            split="train",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def set_val_dataset(self):
        """
        构建验证数据集
        生命周期阶段: setup() 调用
        """
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

        # 如果存在“无干扰”验证数据集类，额外构建
        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                self.val_transform_keys,
                split="val",
                image_size=self.image_size,
                max_text_len=self.max_text_len,
                draw_false_image=0,
                draw_false_text=0,
                image_only=self.image_only,
            )

    def make_no_false_val_dset(self, image_only=False):
        """
        构建无干扰验证数据集（用于评估）
        """
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="val",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
        )

    def make_no_false_test_dset(self, image_only=False):
        """
        构建无干扰测试数据集（用于评估）
        """
        return self.dataset_cls_no_false(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=0,
            draw_false_text=0,
            image_only=image_only,
        )

    def set_test_dataset(self):
        """
        构建测试数据集
        生命周期阶段: setup() 调用
        """
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            self.val_transform_keys,
            split="test",
            image_size=self.image_size,
            max_text_len=self.max_text_len,
            draw_false_image=self.draw_false_image,
            draw_false_text=self.draw_false_text,
            image_only=self.image_only,
        )

    def setup(self, stage):
        """
        数据集构建钩子
        生命周期阶段: Trainer.fit(), Trainer.validate(), Trainer.test() 内部调用
        """
        if not self.setup_flag:
            # 构建 train/val/test 数据集
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            # 给 dataset 注入 tokenizer
            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True  # 标记 setup 已完成

    def train_dataloader(self):
        """
        构建训练 DataLoader
        生命周期阶段: Trainer.fit() 内部调用
        """
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,  # 训练集通常打乱
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        """
        构建验证 DataLoader
        生命周期阶段: Trainer.validate() 或 Trainer.fit() 内部验证调用
        """
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        """
        构建测试 DataLoader
        生命周期阶段: Trainer.test() 内部调用
        """
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
```
`VLMo` 在训练或验证数据集可能会加入一些“干扰样本”：

* `draw_false_image=1`：给文本配上错误图像
 
* `draw_false_text=1`：给图像配上错误文本

这种策略有助于模型学习**跨模态对齐能力**，增强鲁棒性，但它会让数据本身有“噪声”。

**为什么需要无干扰数据集？**
   
* 在训练中，你希望模型看到“有干扰”的数据，提高判别能力；

* 在评估阶段，你希望衡量模型在**真实匹配样本**上的性能，这时候就要去掉干扰，即 `draw_false_image=0`、`draw_false_text=0`；

* 这保证了评估指标（如准确率、召回率等）反映的是模型对正确样本的能力，而不是对抗干扰样本的能力。


**具体实现:**

* `make_no_false_val_dset` → 构建无干扰的验证集，保证验证指标真实可靠；

* `make_no_false_test_dset` → 构建无干扰的测试集，用于最终评估模型效果；

* 可以选择 `image_only=True` 或 `False` 来控制是否只用图像作为输入。


有了 `BaseDataModule` 类负责完成通用模版流程的抽取，子类需要做的事情就非常简单了，只需要告知父类自己的数据集名和数据集类的具体实现即可:

```python
class CocoCaptionKarpathyDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return CocoCaptionKarpathyDataset

    @property
    def dataset_cls_no_false(self):
        return CocoCaptionKarpathyDataset

    @property
    def dataset_name(self):
        return "coco"
```
当 `VLMo` 通过 `LightningDataModule` 完成 `DataSet` 的 `prepare` 和 `set_up` 后，下一步便可以通过 `DataLoader` 来正常获取一个批次的数据了，这里以 `CocoCaptionKarpathyDataset` 子实现类为例，看一下数据的形式:

```python
class CocoCaptionKarpathyDataset(BaseDataset):
    def __init__(self, *args, split="", **kwargs):
        assert split in ["train", "val", "test"]
        self.split = split

        if split == "train":
            names = ["coco_caption_karpathy_train", "coco_caption_karpathy_restval"]
        elif split == "val":
            names = ["coco_caption_karpathy_val"]
        elif split == "test":
            names = ["coco_caption_karpathy_test"]

        super().__init__(*args, **kwargs, names=names, text_column_name="caption")

    def __getitem__(self, index):
        suite = self.get_suite(index)

        if "test" in self.split:
            _index, _question_index = self.index_mapper[index]
            iid = self.table["image_id"][_index].as_py()
            iid = int(iid.split(".")[0].split("_")[-1])
            suite.update({"iid": iid})

        return suite
```
通过 `CocoCaptionKarpathyDataset` 的 `__getitem__` 方法，每次可以获取一条样本数据，具体形式如下:

![](VLMo/2.png)


基类 `BaseDataset` 中提供了 `collate` 方法，用于 `DataLoader` 积攒起一批样本数据后，回调该钩子方法完成合适的批量数据格式组织:

```python
class BaseDataset(torch.utils.data.Dataset):

    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        ...
        return dict_batch
```
该方法实现过程比较复杂，但其主要负责将输入的 `batch` 数据按 `key` 进行聚合 , 同时对输入的文本数据回调 `mlm_collator` 钩子方法，完成 `Masked Language Modeling（MLM）` 任务 , 生成两个新的 `key` : `text_ids_mlm` 和 `text_labels_mlm` 用于表示 `MLM` 后的 `input_ids` 和 `mask标签`。

![batch中只有一条数据](VLMo/3.png)

![按key进行聚合](VLMo/4.png)
