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
