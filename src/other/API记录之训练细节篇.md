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

