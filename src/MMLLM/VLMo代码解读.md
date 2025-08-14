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
