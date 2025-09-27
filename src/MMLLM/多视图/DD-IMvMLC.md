---
title: Deep Double Incomplete Multi-View Multi-Label Learning With Incomplete Labels and Missing Views 论文
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-09-25
author:
  - BinaryOracle
---

`Deep Double Incomplete Multi-View Multi-Label Learning With Incomplete Labels and Missing Views 论文` 

<!-- more -->

> 论文链接: [https://ieeexplore.ieee.org/document/10086538](https://ieeexplore.ieee.org/document/10086538)
> 代码链接: [https://github.com/justsmart/DIMC](https://github.com/justsmart/DIMC)

### 3 提出的方法总结

本文提出的方法基于一个**信息理论框架**，在该框架下一致性学习和数据恢复被统一处理，并可学习到**充分且最小的多视角表示**（如图1所示）。该方法主要包括两部分：理论结果和实际实现方法 DCP。

#### 信息理论框架与基本定义

* 我们用 $X_1$ 和 $X_2$ 表示同一实例的两个视角，其标签为 $Y$，视角表示通过 $Z_i = f^{(i)}(X_i)$ 获得，其中 $i=1,2$。
* 互信息 $I(A;B)$、熵 $H(A)$、条件熵 $H(A|B)$ 和条件互信息 $I(A;B|C)$ 用于衡量信息量和共享信息。

**跨视角一致性定义**：

* $Z_i$ 和 $Z_j$ 被认为一致，如果对任意 $Z_0 \in T(X_j)$ 和 $Z'' \in T(X_i)$ 有
  $I(Z_i; Z_j) \ge I(Z_i; Z_0)$ 且 $I(Z_i; Z_j) \ge I(Z''; Z_j)$
* 图1红色区域表示 $I(Z_i; Z_j)$，通过最大化它可以获得一致表示。

**跨视角可恢复性定义**：

* $Z_i$ 相对于 $Z_j$ 可恢复，如果 $H(Z_i|Z_j) \le H(Z_i|Z_0)$ 对任意 $Z_0 \in T(X_j)$ 成立。
* 当 $H(Z_i|Z_j)=0$ 时，表示可完全恢复。
* 图1灰色区域表示 $H(Z_i|Z_j)$，最小化条件熵可以恢复视角共享信息，同时丢弃视角不一致信息。

**定理1**（一致性与可恢复性的等价性）：

* $Z_i$ 和 $Z_j$ 跨视角一致，当且仅当 $Z_i$ 可被 $Z_j$ 恢复，且 $Z_j$ 可被 $Z_i$ 恢复。
* 这说明一致性学习和数据恢复是同一问题的两个方面，互相促进：最大化互信息增加共享信息，最小化条件熵丢弃不一致信息。

**通用目标函数**：

$$
\max I(Z_1; Z_2) \quad \text{s.t.} \quad \min H(Z_1|Z_2), \min H(Z_2|Z_1)
$$

* 在该目标下，学习到的表示可以兼顾一致性和可恢复性。

#### 多视角数据假设与充分最小表示

* **假设1（多视角数据等效充分性）**：
  每个视角对下游任务的充分性近似相等
  $I(X_1; Y) = I(X_2; Y) = I(X_1, X_2; Y)$

* **命题1**：
  在假设1下，可得 $I(X_1; Y|X_2) = I(X_2; Y|X_1) = 0$，即额外视角不提供额外信息。

**充分表示定义**：

* 若 $I(Z_1; Y) = I(Z_2; Y) = I(X_1, X_2; Y)$，表示为充分。
* 充分表示保证输入中含有下游任务所需的完整信息（图1中的A3区域）。

**最小表示定义**：

* 若 $I(Z_1; X_1|Y) = I(Z_2; X_2|Y) = I(X_1, X_2|Y)$，表示为最小。
* 最小表示去除任务无关信息（图1中A1∪A5区域），保留固定间隙 $I(X_1; X_2|Y)$（图1中A2区域）。

**定理2**（充分且最小的多视角表示）：

* 方程目标的优化结果 $Z_1^{sm}$ 和 $Z_2^{sm}$ 是充分且最小表示。
* 证明利用马尔可夫链和数据处理不等式：最大化 $I(Z_1; Z_2)$ 可确保充分性，最小化 $H(Z_1|Z_2)$ 可确保最小性。

