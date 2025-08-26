---
title: API记录之框架篇
icon: file
category:
  - tools
tag:
  - 已发布
footer: 技术共建，知识共享
date: 2025-08-22
author:
  - BinaryOracle
---

`API记录之框架篇` 

<!-- more -->

## timm 库

`timm` 是 **PyTorch Image Models** 的缩写，是 Ross Wightman 开发和维护的一个 **PyTorch 视觉模型库**，在计算机视觉领域非常常用。它在科研与工业界都很受欢迎，因为它集合了大量常见与前沿的图像模型，同时提供了高质量的实现和训练权重。

**特点**:

1. **模型丰富**

   * 收录了数百种视觉模型，包括：

     * 经典模型：ResNet、DenseNet、EfficientNet、MobileNet

     * Transformer 系列：ViT、DeiT、Swin Transformer、ConvNeXt

     * 最新论文模型：EVA、ConvNeXt V2、MaxViT 等

   * 你几乎可以把它当成 **视觉模型的“模型仓库”**。

2. **预训练权重**

   * 提供了大量在 **ImageNet-1k / ImageNet-21k** 上训练好的权重，开箱即用。

   * 可以直接加载预训练模型用于 **迁移学习 / finetune**。

3. **统一接口**

   * 使用简单，几乎所有模型都能通过同样的方式调用：

     ```python
     import timm
     model = timm.create_model('resnet50', pretrained=True)
     x = torch.randn(1, 3, 224, 224)
     y = model(x)
     ```

   * API 统一，降低了不同架构之间的切换成本。

4. **实用工具**

   * `timm.data`：包含数据增强（RandAugment、Mixup、CutMix 等）。
 
   * `timm.optim`：包含优化器（AdamP、RAdam、Lookahead 等）。
 
   * `timm.scheduler`：学习率调度器（CosineAnnealing、OneCycle、TanhDecay 等）。
 
   * `timm.loss`：封装了多种损失函数（Label Smoothing、SoftTarget CrossEntropy 等）。
 
   * 这些设计让训练流程非常完整。

5. **高效实现**

   * 很多模型在 `timm` 里做了 **速度和显存优化**，常常比官方实现更高效。
 
   * 支持混合精度训练、channels-last 等特性。

### create_model 与 @register_model 装饰器

**`create_model`**：timm 提供的统一入口，用于按名字实例化模型。

```python
model = timm.create_model('resnet50', pretrained=True)
```

**`@register_model`**：用于将自定义模型注册到 timm 模型库，才能通过 `create_model` 调用。

```python
@register_model
def my_model(pretrained=False, **kwargs):
   return MyModel(**kwargs)
```

前者是**用**模型，后者是**加**模型。

## scikit-learn 库

### train_test_split

`train_test_split`（来自 `sklearn.model_selection`）用于把一个或多个并行数组按比例切分成训练集和测试集，常用于机器学习的数据准备。

```py
def train_test_split(
    *arrays,
    test_size=None,
    train_size=None,
    random_state=None,
    shuffle=True,
    stratify=None,
)
```

* `*arrays`：一个或多个数组（如 `X, y`），长度必须相同。返回值是按输入顺序交错的切分结果：`X_train, X_test, y_train, y_test, ...`。

* `test_size`：`float`（0\~1，表示比例）或 `int`（样本数）或 `None`。若都为 `None`，默认 `test_size=0.25`。

* `train_size`：同 `test_size`，可用来显式指定训练集大小（优先级低于 `test_size`）。

* `random_state`：整数或 `RandomState`，用于可重复的随机化（只在 `shuffle=True` 时生效）。

* `shuffle`：是否先打乱样本（默认 `True`）。设为 `False` 时按原序切分。

* `stratify`：用于分层采样的标签数组（与输入长度相同），保证切分后各类比例与原始数据一致；如果提供了 `stratify`，必须 `shuffle=True`。

简单例子:

```py
from sklearn.model_selection import train_test_split
import numpy as np

X = np.arange(10).reshape(10,1)       # 10 个样本特征
y = np.array([0,0,0,0,1,1,1,1,1,1])   # 不平衡标签

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(X_train.shape, X_test.shape)  # (7,1) (3,1)
print(np.bincount(y_train), np.bincount(y_test))
# 输出会显示训练/测试集中 0/1 类的比例与原始近似一致
```

**注意点**:

* 给多个数组（如 `X, y, z`）时，返回对应数量的切分结果。

* `stratify` 用于类别任务，能避免切分导致某类在测试集中缺失。

* 若需固定切分可用 `random_state`；想保留原序列则 `shuffle=False`。

### compute_class_weight

`compute_class_weight` 是 **scikit-learn** 提供的一个函数，用于根据样本分布计算每个类别的权重，常用于 **类别不平衡** 的分类任务。

```python
sklearn.utils.class_weight.compute_class_weight(
    class_weight,
    classes,
    y
)
```

* **class\_weight**

  * `'balanced'`：自动计算权重，和类别频率成反比。
  
  * `dict`：手动指定某些类别的权重，如 `{0: 1.0, 1: 5.0}`。
  
  * `None`：不计算，所有类别权重为 1。

* **classes**
  
  * 所有类别的 **唯一标签数组**（如 `[0, 1, 2]`）。

* **y**
  
  * 训练数据的标签数组（如 `[0,0,1,2,2,2]`）。


返回值:

* **`weights`**：一维数组，长度与 `classes` 相同，表示每个类别的权重。

计算公式（`balanced` 模式）：

$$
w_j = \frac{n_{samples}}{n_{classes} \times n_j}
$$

其中：

* $n_{samples}$：样本总数

* $n_{classes}$：类别数

* $n_j$：第 j 类的样本数

示例:

```python
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

y = np.array([0, 0, 1, 2, 2, 2])  # 样本标签
classes = np.unique(y)

weights = compute_class_weight('balanced', classes=classes, y=y)
print("类别权重:", dict(zip(classes, weights)))
```

输出：

```
类别权重: {0: 1.0, 1: 3.0, 2: 0.67}
```

说明：

* 类别 `0` 有 2 个样本 → 权重较低

* 类别 `1` 只有 1 个样本 → 权重最高

* 类别 `2` 有 3 个样本 → 权重最低


## python 内置 collections 库

### Counter

`Counter` 是 Python 内置库 **collections** 提供的一个计数器类，用于统计可迭代对象中各元素出现的次数。

```python
from collections import Counter
Counter(iterable)      # 输入一个可迭代对象
Counter(mapping)       # 输入一个字典
Counter(a=2, b=3, ...) # 输入关键字参数
```

* 返回的是一个字典的子类，键为元素，值为出现次数。

* 查询一个未出现的元素时，计数为 0。

* 支持常见的字典操作，还扩展了计数相关方法。

常用方法:

* `most_common(n)`：返回出现次数最多的前 `n` 个元素及其频数。

* `elements()`：按出现次数依次返回元素（迭代器）。

* `update(iterable)`：更新计数。

* `subtract(iterable)`：减少计数。

例子:

```python
from collections import Counter

all_labels = [0, 1, 0, 2, 1, 0, 2, 2, 2]
label_counts = Counter(all_labels)

print(label_counts)           # Counter({2: 4, 0: 3, 1: 2})
print(label_counts[2])        # 4
print(label_counts.most_common(1))  # [(2, 4)]
print(list(label_counts.elements())) # [0, 0, 0, 1, 1, 2, 2, 2, 2]
```

## pytorch 内置 采样库

### WeightedRandomSampler

`WeightedRandomSampler` 是 **PyTorch** 提供的一个采样器，用于在构建 `DataLoader` 时 **按权重采样样本**，常用于类别不平衡的数据集。

```python
torch.utils.data.WeightedRandomSampler(
    weights,
    num_samples,
    replacement=True
)
```
* **weights**

  * 一维数组/列表，长度等于样本数。
  
  * 每个元素表示对应样本被采样的概率权重。
  
  * 权重越大，被抽到的概率越高。

* **num\_samples**

  * 采样的样本数（即每个 epoch 中从数据集中抽多少个样本）。
  
  * 通常设为 `len(dataset)` 或 `len(train_labels)`。

* **replacement**

  * 是否有放回采样：

    * `True`：可以重复采样同一样本。
  
    * `False`：无放回采样（但这时 `num_samples` 不能超过数据集大小）。


举例:

```python
train_label_counts = Counter(train_labels)
# 计算每个样本的权重：类别样本越少，权重越高
train_sample_weights = [1.0 / train_label_counts[label] for label in train_labels]

# 构建加权随机采样器
train_sampler = WeightedRandomSampler(
    weights=train_sample_weights,
    num_samples=len(train_labels),  # 每个epoch采样样本数=总样本数
    replacement=True                # 允许重复采样
)
```

* **思路**：类别数量少 → 权重大 → 更容易被采到。

* **目的**：让每个类别在训练过程中被抽到的机会接近均衡，从而缓解类别不平衡问题。

