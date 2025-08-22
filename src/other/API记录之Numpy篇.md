---
title: API记录之Numpy篇
icon: file
category:
  - tools
tag:
  - 已发布
footer: 技术共建，知识共享
date: 2025-08-10
author:
  - BinaryOracle
---

`API记录之Numpy篇` 

<!-- more -->

## Numpy

### np.linspace

`np.linspace` 是 NumPy 中生成等间隔数列的函数。基本用法：

```python
import numpy as np

# 在 0 到 1 之间生成 5 个等间隔数
arr = np.linspace(0, 1, 5)
print(arr)  # 输出: [0.   0.25 0.5  0.75 1.  ]
```

参数说明：

* `start`：起始值

* `stop`：结束值

* `num`：生成的样本数量（默认 50）

* `endpoint`：是否包含 stop（默认 True）

* `retstep`：是否返回步长（True 返回 `(array, step)`）

### np.concatenate

`np.concatenate` 用于沿指定轴将多个数组拼接在一起。

```python
import numpy as np

a = np.array([1, 2])
b = np.array([3, 4])

# 沿默认轴（axis=0）拼接
c = np.concatenate([a, b])
print(c)  # 输出: [1 2 3 4]

# 对二维数组沿不同轴拼接
a2 = np.array([[1, 2], [3, 4]])
b2 = np.array([[5, 6], [7, 8]])

# 沿行拼接（axis=0）
c2 = np.concatenate([a2, b2], axis=0)
# [[1 2]
#  [3 4]
#  [5 6]
#  [7 8]]

# 沿列拼接（axis=1）
c3 = np.concatenate([a2, b2], axis=1)
# [[1 2 5 6]
#  [3 4 7 8]]
```
