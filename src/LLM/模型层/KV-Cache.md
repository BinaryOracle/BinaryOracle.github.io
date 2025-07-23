---
title: KV-Cache 详解
icon: file
category:
  - NLP
tag:
  - Trick
  - 编辑中
footer: 技术共建，知识共享
date: 2025-07-22
order: 2
author:
  - BinaryOracle
---

`大模型加速技术之KV Cache详解`
 
<!-- more -->

## Why we need KV Cache ？

生成式generative模型的推理过程很有特点，我们给一个输入文本，模型会输出一个回答（长度为N），其实该过程中执行了N次推理过程。即GPT类模型一次推理只输出一个token，输出token会与输入tokens 拼接在一起，然后作为下一次推理的输入，这样不断反复直到遇到终止符。

如上描述是我们通常认知的GPT推理过程。代码描述如下：

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def main():
    # 加载模型和 tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # 初始输入
    in_text = "Open AI is a"
    in_tokens = torch.tensor(tokenizer.encode(in_text)).unsqueeze(0)  # [1, seq_len]
    token_eos = torch.tensor([198])  # line break symbol
    out_token = None
    i = 0

    with torch.no_grad():
        while out_token != token_eos:
            outputs = model(in_tokens)
            logits = outputs.logits
            out_token = torch.argmax(logits[0, -1, :], dim=-1, keepdim=True).unsqueeze(0)  # [1, 1]
            in_tokens = torch.cat((in_tokens, out_token), dim=1)
            text = tokenizer.decode(in_tokens[0])
            print(f'step {i} input: {text}', flush=True)
            i += 1

    out_text = tokenizer.decode(in_tokens[0])
    print(f'\nInput: {in_text}')
    print(f'Output: {out_text}')

if __name__ == "__main__":
    main()
```

输出:

```python
step 0 input: Open AI is a new
step 1 input: Open AI is a new way
step 2 input: Open AI is a new way to
step 3 input: Open AI is a new way to build
step 4 input: Open AI is a new way to build AI
step 5 input: Open AI is a new way to build AI that
step 6 input: Open AI is a new way to build AI that is
step 7 input: Open AI is a new way to build AI that is more
step 8 input: Open AI is a new way to build AI that is more efficient
step 9 input: Open AI is a new way to build AI that is more efficient and
step 10 input: Open AI is a new way to build AI that is more efficient and more
step 11 input: Open AI is a new way to build AI that is more efficient and more efficient
step 12 input: Open AI is a new way to build AI that is more efficient and more efficient than
step 13 input: Open AI is a new way to build AI that is more efficient and more efficient than traditional
step 14 input: Open AI is a new way to build AI that is more efficient and more efficient than traditional AI
step 15 input: Open AI is a new way to build AI that is more efficient and more efficient than traditional AI.
step 16 input: Open AI is a new way to build AI that is more efficient and more efficient than traditional AI.


Input: Open AI is a
Output: Open AI is a new way to build AI that is more efficient and more efficient than traditional AI.
```

在上面的推理过程中，每 step 内，输入一个 token序列，经过Embedding层将输入token序列变为一个三维张量 [b, s, h]，经过一通计算，最后经 logits 层将计算结果映射至词表空间，输出张量维度为 [b, s, vocab_size]。

当前轮输出token与输入tokens拼接，并作为下一轮的输入tokens，反复多次。可以看出第 i+1 轮输入数据只比第 i 轮输入数据新增了一个 token，其他全部相同！

因此第 i+1 轮推理时必然包含了第 i 轮的部分计算。KV Cache 的出发点就在这里，缓存当前轮可重复利用的计算结果，下一轮计算时直接读取缓存结果。

> 上面所举例子并没有使用KV Cache进行推理,请注意。

## Self-Attention Without Cache

下图给出了无 Cache 情况下，类GPT式生成式模型进行推理的过程:

![](KV-Cache/1.png)

这种方式的问题是: **每生成一个 token，就要重新计算所有之前 token 的 Q/K/V + Attention + FFN** 。


## Self-Attention With Cache


下图给出了有 Cache 情况下，类GPT式生成式模型进行推理的过程:

![](KV-Cache/2.png)

## Huggingface 官方代码实现

本节将根据 Huggingface 官方代码实现进行 KV Cache 实现讲解 (只展示核心代码，移除了大量与本文无关的逻辑)。

> 官方代码链接: [https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py](https://github.com/huggingface/transformers/blob/main/src/transformers/models/gpt2/modeling_gpt2.py)

下面将给出使用了 KV Cache 进行推理的代码:

```python
import torch
from transformers import GPT2Tokenizer, GPT2Config
from modeling_gpt2 import GPT2LMHeadModel  # copy from huggingface , 删除了大量无关代码

def generate_text(model, tokenizer, prompt, max_new_tokens=50, eos_token_id=198):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    past_key_values = None
    output_ids = input_ids.clone()

    with torch.no_grad():
        for step in range(max_new_tokens):
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )
            logits = outputs.logits
            past_key_values = outputs.past_key_values

            next_token_logits = logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            output_ids = torch.cat([output_ids, next_token], dim=-1)

            if next_token.item() == eos_token_id:
                break

            input_ids = next_token  # 采用KV Cache后，推理过程修改的关键: 下一步只送入新 token

            print(f"step {step}: {tokenizer.decode(output_ids[0])}", flush=True)

    return tokenizer.decode(output_ids[0])

def main():
    config = GPT2Config()
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel(config)

    prompt = "Once upon a time"
    output = generate_text(model, tokenizer, prompt)
    print("\nFinal output:")
    print(output)

if __name__ == "__main__":
    main()
```

KV Cache 的引入是为了加速自回归模型的推理速度，具体体现在:

1. 每轮推理时，只需要计算当前轮新增 token 的 Q/K/V，而不需要重新计算所有之前 token 的 Q/K/V。

2. 缓存当前轮计算结果，下一轮推理时直接读取缓存结果。

在首轮推理的过程中，我们传入的是 promt 提示词列表，并且 KV Cache 此时为空，还未进行初始化。因此首轮推理过程需要完成 promt 提示词列表的 keys 和 values 的缓存；由于 GPT2 由多层 GPT2Block 堆叠而成，而每一层 GPT2Block 都有一个 GPT2Attention 模块， 因此 KV Cache 需要准备好每一层 GPT2Attention 模块的 keys 和 values 缓存 (分层Cache - legacy_cache)。

