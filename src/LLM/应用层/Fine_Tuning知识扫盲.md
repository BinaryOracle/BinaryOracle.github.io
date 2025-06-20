---
title: 大模型微调(Fine Tuning)知识扫盲
icon: file
category:
  - 大模型应用层
tag:
  - 大模型应用层
  - 已发布
footer: 技术共建，知识共享
date: 2025-06-17
order: 2
author:
  - BinaryOracle
---

`大模型微调(Fine Tuning)知识扫盲`
 
<!-- more -->

## 什么是大模型 ？

开始之前，为了方便大家理解，我们先对大模型做一个直观的抽象。

本质上，现在的大模型要解决的问题，就是一个序列数据转换的问题：

- 输入序列 X = [x1, x2, ..., xm]

- 输出序列Y = [y1, y2, …, yn]

- X和Y之间的关系是：Y = WX。

我们所说的“大模型”这个词：“大”是指用于训练模型的参数非常多，多达千亿、万亿；而“模型”指的就是上述公式中的矩阵W。

在这里，矩阵W就是通过机器学习，得出的用来将X序列，转换成Y序列的权重参数组成的矩阵。

> 需要特别说明：这里为了方便理解，做了大量的简化。在实际的模型中，会有多个用于不同目的的权重参数矩阵，也还有一些其它参数。


## 为什么要对大模型进行微调 ？

通常，要对大模型进行微调，有以下一些原因：

1. 因为大模型的参数量非常大，训练成本非常高，每家公司都去从头训练一个自己的大模型，这个事情的性价比非常低；

2. Prompt Engineering的方式是一种相对来说容易上手的使用大模型的方式，但是它的缺点也非常明显。因为通常大模型的实现原理，都会对输入序列的长度有限制，Prompt Engineering 的方式会把Prompt搞得很长。

> - 越长的Prompt，大模型的推理成本越高，因为推理成本是跟Prompt长度的平方正向相关的。
>
> - 另外，Prompt太长会因超过限制而被截断，进而导致大模型的输出质量打折口，这也是一个非常严重的问题。
>
> - 对于个人使用者而言，如果是解决自己日常生活、工作中的一些问题，直接用Prompt Engineering的方式，通常问题不大。
>
> - 但对于对外提供服务的企业来说，要想在自己的服务中接入大模型的能力，推理成本是不得不要考虑的一个因素，微调相对来说就是一个更优的方案。

3. Prompt Engineering的效果达不到要求，企业又有比较好的自有数据，能够通过自有数据，更好的提升大模型在特定领域的能力。这时候微调就非常适用。

4. 要在个性化的服务中使用大模型的能力，这时候针对每个用户的数据，训练一个轻量级的微调模型，就是一个不错的方案。

5. 数据安全的问题。如果数据是不能传递给第三方大模型服务的，那么搭建自己的大模型就非常必要。通常这些开源的大模型都是需要用自有数据进行微调，才能够满足业务的需求，这时候也需要对大模型进行微调。

## 如何对大模型进行微调 ？

从参数规模的角度，大模型的微调分成两条技术路线：

- 一条是对全量的参数，进行全量的训练，这条路径叫全量微调FFT(Full Fine Tuning)。

- 一条是只对部分的参数进行训练，这条路径叫PEFT(Parameter-Efficient Fine Tuning)。

FFT的原理，就是用特定的数据，对大模型进行训练，将W变成$W'$，$W'$相比W ，最大的优点就是上述特定数据领域的表现会好很多。

但FFT也会带来一些问题，影响比较大的问题，主要有以下两个：

- 一个是训练的成本会比较高，因为微调的参数量跟预训练的是一样的多的；

- 一个是叫灾难性遗忘(Catastrophic Forgetting)，用特定训练数据去微调可能会把这个领域的表现变好，但也可能会把原来表现好的别的领域的能力变差。

PEFT主要想解决的问题，就是FFT存在的上述两个问题，PEFT也是目前比较主流的微调方案。

从训练数据的来源、以及训练的方法的角度，大模型的微调有以下几条技术路线：

1. 监督式微调SFT(Supervised Fine Tuning) :  用人工标注的数据，用传统机器学习中监督学习的方法，对大模型进行微调；

2. 基于人类反馈的强化学习微调RLHF(Reinforcement Learning with Human Feedback) : 把人类的反馈，通过强化学习的方式，引入到对大模型的微调中去，让大模型生成的结果，更加符合人类的一些期望；

3. 基于AI反馈的强化学习微调RLAIF(Reinforcement Learning with AI Feedback) :  原理大致跟RLHF类似，但是反馈的来源是AI。这里是想解决反馈系统的效率问题，因为收集人类反馈，相对来说成本会比较高、效率比较低。

不同的分类角度，只是侧重点不一样，对同一个大模型的微调，也不局限于某一个方案，可以多个方案一起。

微调的最终目的，是能够在可控成本的前提下，尽可能地提升大模型在特定领域的能力。

## 常用的PEFT方案

从成本和效果的角度综合考虑，PEFT是目前业界比较流行的微调方案。接下来介绍几种比较流行的PEFT微调方案。

### Prompt Tuning

Prompt Tuning的出发点，是基座模型(Foundation Model)的参数不变，**为每个特定任务，训练一个少量参数的小模型，在具体执行特定任务的时候按需调用**。

Prompt Tuning的基本原理是***在输入序列X之前，增加一些特定长度的特殊Token，以增大生成期望序列的概率***。

具体来说，就是将$X = [x1, x2, ..., xm]$变成，$X' = [x'1, x'2, ..., x'k; x1, x2, ..., xm], Y = WX'$。

Prompt Tuning是发生在Embedding这个环节的。如果将大模型比做一个函数：$Y=f(X)$，那么Prompt Tuning就是在保证函数本身不变的前提下，在X前面加上了一些特定的内容，而这些内容可以影响X生成期望中Y的概率。

> Prompt Tuning的具体细节,可以参见：[The Power of Scale for Parameter-Efficient Prompt Tuning](https://arxiv.org/abs/2104.08691)。

### Prefix Tuning

Prefix Tuning的灵感来源是，基于Prompt Engineering的实践表明，在不改变大模型的前提下，在Prompt上下文中添加适当的条件，可以引导大模型有更加出色的表现。

Prefix Tuning的出发点，跟Prompt Tuning的是类似的，只不过它们的具体实现上有一些差异。

Prompt Tuning是在Embedding环节，往输入序列X前面加特定的Token。而Prefix Tuning是在Transformer的Encoder和Decoder的网络中都加了一些特定的前缀。

具体来说，就是将Y=WX中的W，变成$W' = [Wp; W]，Y=W'X$。

Prefix Tuning也保证了基座模型本身是没有变的，只是在推理的过程中，按需要在W前面拼接一些参数。

> Prefix Tuning的具体细节,可以参见：[Prefix-Tuning: Optimizing Continuous Prompts for Generation](https://arxiv.org/abs/2101.00190)。

### LoRA

LoRA是跟Prompt Tuning和Prefix Tuning完全不相同的另一条技术路线。

LoRA背后有一个假设：我们现在看到的这些大语言模型，它们都是被过度参数化的。而过度参数化的大模型背后，都有一个低维的本质模型。

通俗讲人话：大模型参数很多，但并不是所有的参数都是发挥同样作用的；大模型中有其中一部分参数，是非常重要的，是影响大模型生成结果的关键参数，这部分关键参数就是上面提到的低维的本质模型。

LoRA的基本思路，包括以下几步：

1. 首先, 要适配特定的下游任务，要训练一个特定的模型，将Y=WX变成Y=(W+∆W)X，这里面∆W主是我们要微调得到的结果；

2. 其次，将∆W进行低维分解∆W=AB (∆W为m * n维，A为m * r维，B为r * n维，r就是上述假设中的低维)；

3. 接下来，用特定的训练数据，训练出A和B即可得到∆W，在推理的过程中直接将∆W加到W上去，再没有额外的成本。

4. 另外，如果要用LoRA适配不同的场景，切换也非常方便，做简单的矩阵加法即可：(W + ∆W) - ∆W + ∆W'。

> 关于LoRA的具体细节,可以参见LoRA: [Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)。


### QLoRA

LoRA 效果已经非常好了，可以媲美全量微调的效果了，那为什么还要有个QLoRA呢？

这里先简单介绍一下，量化（Quantization）。

量化，是一种在保证模型效果基本不降低的前提下，通过降低参数的精度，来减少模型对于计算资源的需求的方法。

量化的核心目标是降成本，降训练成本，特别是降后期的推理成本。

QLoRA就是量化版的LoRA，它是在LoRA的基础上，进行了进一步的量化，将原本用16bit表示的参数，降为用4bit来表示，可以在保证模型效果的同时，极大地降低成本。

论文中举的例子，65B的LLaMA的微调要780GB的GPU内存；而用了QLoRA之后，只需要48GB。效果相当惊人！

> 关于QLoRA的具体细节,可以参见：[QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314)。



PEFT 的微调方法，还有很多种，限于篇幅原因，不再这里一一介绍。感兴趣的朋友，可以阅读这篇论文：[Scaling Down to Scale Up: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2306.02511)。


相关阅读资料:

0. [近代自然语言处理技术发展的“第四范式”](https://zhuanlan.zhihu.com/p/395115779)

1. [Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing](https://arxiv.org/abs/2107.13586)
