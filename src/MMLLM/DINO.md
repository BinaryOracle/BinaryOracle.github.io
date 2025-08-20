---
title: DINO 论文
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-08-17
author:
  - BinaryOracle
---

`Emerging Properties in Self-Supervised Vision Transformers 论文解读` 

<!-- more -->

> 论文链接: [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294)
> 代码链接: [https://github.com/facebookresearch/dino](https://github.com/facebookresearch/dino)


## 代码解析

从本节开始我们将对官方开源的 `DINO` 模型代码实现进行详细讲解，下图给出的是 `DINO` 模型的运行完整流程图:

![](dino/1.gif)

按照 `DINO` 模型的训练流程，第一步首先是对 输入图像 进行数据增强，生成 `两张全局视角图像 + 若干局部视角图像`， 该过程由 `DataAugmentationDINO` 类实现，代码如下:

```python
class DataAugmentationDINO(object):
    """
    数据增强类，用于 DINO 训练。
    主要思想：从一张输入图像生成多个视角（multi-crop），
    包括 2 张全局裁剪图像（224x224）和若干张局部裁剪图像（96x96）。
    """

    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        # 基础增强：随机翻转 + 颜色抖动 + 随机灰度化
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),  # 以 50% 概率水平翻转
            transforms.RandomApply(  # 以 80% 概率执行颜色抖动
                [transforms.ColorJitter(
                    brightness=0.4,  # 亮度变化
                    contrast=0.4,    # 对比度变化
                    saturation=0.2,  # 饱和度变化
                    hue=0.1          # 色调变化
                )],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),  # 以 20% 概率转为灰度图
        ])

        # 标准化（Imagenet 预训练均值/方差）
        normalize = transforms.Compose([
            transforms.ToTensor(),  # 转换为 Tensor
            transforms.Normalize(
                (0.485, 0.456, 0.406),  # mean
                (0.229, 0.224, 0.225)   # std
            ),
        ])

        # ----------- 全局裁剪 1 -----------
        self.global_transfo1 = transforms.Compose([
            # 随机裁剪并缩放到 224x224
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),  # 高斯模糊（概率 100%）
            normalize,
        ])

        # ----------- 全局裁剪 2 -----------
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),   # 高斯模糊（概率 10%）
            utils.Solarization(0.2),  # 以 20% 概率进行太阳化增强（反转亮部）
            normalize,
        ])

        # ----------- 局部裁剪 -----------
        self.local_crops_number = local_crops_number  # 局部裁剪数量
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),  # 高斯模糊（概率 50%）
            normalize,
        ])

    def __call__(self, image):
        """
        输入一张图像，返回多个增强后的 crop。
        输出顺序：
        [全局视角1, 全局视角2, 局部视角1, 局部视角2, ...]
        """
        # 先生成两张全局 crop
        crops = [self.global_transfo1(image), self.global_transfo2(image)]
        # 再生成若干张局部 crop
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops
```
特别注意初始化方法中传入的 `global_crops_scale` 和 `local_crops_scale` 参数 :

```python
    # Multi-crop
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.0),
        help="Scale range for global crops.")
    parser.add_argument('--local_crops_number', type=int, default=8,
        help="Number of local crops (0 disables multi-crop).")
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="Scale range for local crops.")

```

`*_crops_scale` 是 **随机裁剪区域的相对尺度范围**，即原始图像面积的比例。

* 参数形式：`*_crops_scale = (min_scale, max_scale)`

* 作用：控制 **随机裁剪区域的最小和最大面积比例**。

* 举例：

  * 如果 `global_crops_scale = (0.4, 1.0)`

    * 随机裁剪的区域面积 ∈ \[40%, 100%] 原图面积之间。
   
    * 也就是说，有时裁掉少量边缘（接近原图），有时只取 40% 的图像内容（更聚焦）。
  
  * 裁剪后再统一缩放到 `224 × 224`，作为模型输入。

---



```python
class MultiCropWrapper(nn.Module):
    """
    一个封装类，用于处理多视角输入（multi-crop inputs）。
    不同分辨率的输入会被分组，每一组在 backbone 中分别进行一次前向计算，
    得到的特征拼接后再送入 head 中处理。
    """

    def __init__(self, backbone, head):
        super(MultiCropWrapper, self).__init__()
        # 去掉 backbone 中原本为 ImageNet 分类准备的全连接层（fc, head）
        # 因为这里是自监督学习，不需要类别分类器
        backbone.fc, backbone.head = nn.Identity(), nn.Identity()
        self.backbone = backbone  # 特征提取网络
        self.head = head          # 投影头（projection head），用于后续对比学习

    def forward(self, x):
        # 保证输入是 list（多 crop 的场景可能传进来的是多个张量）
        if not isinstance(x, list):
            x = [x]

        # 获取每个输入 crop 的分辨率（最后一个维度 size）
        # torch.unique_consecutive 会返回连续相同值的唯一值及计数
        # return_counts=True 表示返回每个唯一值的计数
        # torch.cumsum 累积求和，得到每组 crop 的结束索引
        idx_crops = torch.cumsum(torch.unique_consecutive(
            torch.tensor([inp.shape[-1] for inp in x]),  # 取每个 crop 的宽度
            return_counts=True,
        )[1], 0)

        # 初始化
        start_idx = 0
        # 创建一个空 tensor 用于保存所有特征，放到和输入相同的 device 上
        output = torch.empty(0).to(x[0].device)

        # 遍历不同分辨率的分组
        for end_idx in idx_crops:
            # 将相同分辨率的 crop 拼接（batch 化），一起送入 backbone
            _out = self.backbone(torch.cat(x[start_idx: end_idx]))

            # 有些 backbone（如 XCiT）返回的是 tuple，这里只取第一个元素（主特征）
            if isinstance(_out, tuple):
                _out = _out[0]

            # 将这一批特征拼接到输出中
            output = torch.cat((output, _out))

            # 更新下一个起始位置
            start_idx = end_idx

        # 所有特征拼接完成后，送入 head 得到最终表示
        return self.head(output)
```