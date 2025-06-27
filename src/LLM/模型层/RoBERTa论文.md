---
title: RoBERTa 论文
icon: file
category:
  - NLP
tag:
  - 预训练语言模型
  - 编辑中
footer: 技术共建，知识共享
date: 2025-06-27
order: 2
author:
  - BinaryOracle
---

`RoBERTa 论文`
 
<!-- more -->

> 论文链接: [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692)

## 摘要

RoBERTa是一项针对BERT预训练方法的优化研究，通过系统性的实验发现BERT存在训练不足的问题，并提出了一系列改进措施。这些改进包括更长的训练时间、更大的批次规模、更多的数据、移除下一句预测（NSP）目标、使用更长的序列以及动态调整掩码模式。实验结果表明，优化后的RoBERTa在多个基准测试（如GLUE、RACE和SQuAD）上达到了最先进的性能，甚至超越了后续提出的模型。研究强调了预训练中设计选择和数据规模的重要性，同时表明BERT的掩码语言模型目标在优化后仍具有竞争力。相关模型和代码已公开供进一步研究。