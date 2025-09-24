---
title: API记录之训练细节篇
icon: file
category:
  - tools
tag:
  - 已发布
footer: 技术共建，知识共享
date: 2025-09-02
author:
  - BinaryOracle
---

`API记录之训练细节篇` 

<!-- more -->

## os.environ["CUDA_VISIBLE_DEVICES"]

* **作用**：限制进程可见的 GPU，隐藏未列出的设备。

* **继承性**：主进程设置后，所有子进程会继承同样的可见 GPU 配置。

* **效果**：

  * `torch.cuda.device_count()` 返回的是“可见 GPU 数”，而非物理 GPU 总数。

  * 子进程无法访问未分配的 GPU。

* **注意**：子进程里修改只对该子进程生效，不会影响主进程或其他进程。

**设置方式**: 你可以在运行程序前用命令行设置

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py
```

也可以在代码里设置：

```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

**一句话总结**：`CUDA_VISIBLE_DEVICES` 用来指定程序和其子进程能看到哪些 GPU，相当于给进程建立一个“GPU 白名单”。

## 随机数种子与确定性

1. **作用**：

   * 随机数种子（random seed）用于固定伪随机数生成器的初始状态。

   * 同样的种子保证每次生成的随机数序列**完全相同**，从而实现实验可重复性。

   * 随机数本质上仍是伪随机，只是**可预测**，不是一个常数。

2. **深度学习中通常固定的随机源**：

   * Python 内置 `random` 模块（数据增强等）
  
   * NumPy 随机数生成器（数据预处理、初始化等）
  
   * PyTorch CPU 随机数生成器（权重初始化、dropout 等）
  
   * PyTorch GPU 随机数生成器（CUDA 上的操作，如 dropout、卷积初始化等）
  
   * cuDNN 的随机算法（`cudnn.deterministic=True`）保证卷积等操作的可重复性

**常用设置如下**:

1. **设置 Python 内置随机数种子**：

```python
random.seed(args.manual_seed)
```

* 控制 Python `random` 模块产生的随机数序列，确保每次运行生成相同的随机数。

2. **设置 NumPy 随机数种子**：

```python
np.random.seed(args.manual_seed)
```

* 控制 NumPy 的随机数生成，比如初始化权重、打乱数据顺序等。

3. **设置 PyTorch CPU 随机数种子**：

```python
torch.manual_seed(args.manual_seed)
```

* 控制 PyTorch 在 CPU 上的随机操作，例如权重初始化、dropout 等。

4. **设置 PyTorch CUDA 随机数种子**：

```python
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.manual_seed)
    torch.cuda.manual_seed_all(args.manual_seed)
```

* `manual_seed` 只设置当前 GPU 的随机种子，`manual_seed_all` 则设置所有可见 GPU，保证多 GPU 情况下结果一致。

5. **控制 cuDNN 行为**：

```python
cudnn.benchmark = False
cudnn.deterministic = True
```

* `cudnn.benchmark = False`：禁止 cuDNN 自动寻找最优卷积算法（这个算法搜索过程会引入随机性）。

* `cudnn.deterministic = True`：强制使用确定性的卷积算法，保证每次运行结果一致。


## 偏置与归一化层的关系

**BatchNorm 会重新中心化数据**

```python
# BatchNorm 的数学公式：
y = (x - mean) / sqrt(var + eps) * gamma + beta

# 其中：
# - gamma: 缩放参数（可学习）
# - beta: 偏置参数（可学习）
# - mean, var: 批次的均值和方差
```

**Linear 层也有偏置**

```python
# Linear 层的计算：
z = x @ W.T + b  # b是偏置项
```

**为什么此时 Linear 层的偏置是冗余的** ？

```
输入 → Linear → BatchNorm 的计算链：

output = BN(Linear(x))
        = BN(x @ W.T + b)
        = [(x @ W.T + b - mean) / std] * gamma + beta
        = [x @ W.T / std + (b - mean)/std] * gamma + beta
        = (x @ W.T) * (gamma/std) + [gamma*(b-mean)/std + beta]
```

**可以看到:**

- `(x @ W.T) * (gamma/std)`：有效的权重变换

- `[gamma*(b-mean)/std + beta]`：**常数偏置项**

| 组件 | 作用 | 是否冗余 |
|------|------|----------|
| **Linear.bias** | 添加常数偏移 | ✅ 冗余 |
| **BatchNorm.beta** | 添加常数偏移 | ✅ 唯一需要的 |
| **BatchNorm.gamma** | 缩放特征 | ❌ 必要 |
| **BatchNorm** 的均值归一化 | 中心化数据 | ❌ 必要 |

**实际代码对比:**

1. 错误做法（冗余）：

```python
# 浪费参数和计算
self.linear = nn.Linear(in_dim, out_dim, bias=True)  # 有偏置
self.bn = nn.BatchNorm1d(out_dim)                    # 也有偏置beta
```

2. 正确做法（优化后）：

```python
# 参数和计算更高效
self.linear = nn.Linear(in_dim, out_dim, bias=False)  # 无偏置
self.bn = nn.BatchNorm1d(out_dim)                     # 用BN的beta作为偏置
```

3. 如果**不使用 BatchNorm**，那么应该保留偏置：

```python
# 只有Linear层，没有BN
self.linear = nn.Linear(in_dim, out_dim, bias=True)  # 需要偏置

# 或者使用LayerNorm等其他归一化
self.linear = nn.Linear(in_dim, out_dim, bias=False)
self.norm = nn.LayerNorm(out_dim)  # LayerNorm也有偏置参数
```

**最佳实践总结**:

| 场景 | 建议 | 原因 |
|------|------|------|
| **Linear + BatchNorm** | `bias=False` | 避免冗余，BN的beta足够 |
| **只有Linear** | `bias=True` | 需要偏置来增加模型表达能力 |
| **Linear + LayerNorm** | `bias=False` | LayerNorm也有可学习的偏置 |
| **Linear + InstanceNorm** | `bias=False` | InstanceNorm有可学习的参数 |

深度学习中的**标准实践**：

1. **CNN中**：`Conv2d + BatchNorm` 时，`conv.bias=False`

2. **Transformer中**：`Linear + LayerNorm` 时，`linear.bias=False`  

3. **点云网络中**：`Linear + BatchNorm` 时，`linear.bias=False`

## 交叉折叠验证

在机器学习中，我们训练模型后，通常需要评估模型在新数据上的表现。如果我们直接用训练集来评估，可能会出现 **过拟合**，导致评估结果过于乐观；如果只留出一小部分数据作为验证集，评估结果可能会受随机划分影响，波动很大。

交叉验证就是为了解决这个问题，让模型的评估更稳定、更可靠。

> k 折交叉验证的步骤

假设我们有一个数据集 $D$，想做 5 折交叉验证（k=5）：

1. **划分数据**
   
   将数据集平均分成 5 份（折），记作 $D_1, D_2, D_3, D_4, D_5$。

2. **循环训练和验证**
   
   每次用 4 份数据训练模型，剩下 1 份作为验证集，进行评估。例如：

   * 第1次：训练集 $D_2+D_3+D_4+D_5$，验证集 $D_1$
   
   * 第2次：训练集 $D_1+D_3+D_4+D_5$，验证集 $D_2$
   
   * ……
   
   * 第5次：训练集 $D_1+D_2+D_3+D_4$，验证集 $D_5$

3. **计算平均性能**
   
   每次验证会得到一个性能指标（如准确率、MSE 等），最终取 5 次的平均值，作为模型的整体性能评估。

> 优点

* **充分利用数据**：每个样本都既作为训练数据也作为验证数据。

* **稳定可靠**：评估结果减少了偶然性波动。

* **适合小数据集**：即便数据集不大，也能得到合理的性能估计。

> 变体

* **留一交叉验证（LOOCV）**：每次只留一个样本作为验证集，其余全部训练。

* **分层 k 折交叉验证（Stratified k-Fold）**：在分类问题中，保证每一折中各类别样本比例与整体数据集相同。
