---
title: 多模态常用改编Bert代码实现
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-07-22
author:
  - BinaryOracle
---

`多模态论文中常用的改编版本的Bert代码实现记录` 

<!-- more -->

> 本文改编Bert代码讲解基于BLIP项目展开，代码链接: [BLIP/models/med.py](https://github.com/salesforce/BLIP/blob/main/models/med.py)

## 多模态 Bert 前向传播流程

本节我们将对多模态Bert的前向传播基本流程进行讲解，所给代码删除了大量非核心逻辑，如需了解各类优化手段，请阅读源码进行学习。

### 1. 整体流程总览（BertModel）

```python
class BertModel(BertPreTrainedModel):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states=None,         # 图像模态特征
        encoder_attention_mask=None,        # 图像掩码
        is_decoder=False,
        mode='multimodal',                  # 控制是否启用 cross-attention
    ):
        # 1. 词嵌入 + 位置编码
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids)
        
        # 2. 编码阶段（Text-only 或 Cross-modal）
        sequence_output = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask, # 可用于多头自注意力的文本 padding mask
            encoder_hidden_states=encoder_hidden_states, 
            encoder_attention_mask=encoder_extended_attention_mask, # 可用于多头自注意力的图像 padding mask
            mode=mode,
        )

        # 3. 池化输出（用于分类任务）
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
        )
```

> 池化输出实现:
>
> ```python
> class BertPooler(nn.Module):
>     def __init__(self, config):
>         super().__init__()
>         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
>        self.activation = nn.Tanh()
>
>    def forward(self, hidden_states):
>        # 1. 拿到能够代表整段文本或者整个多模态表示的 CLS Token
>        first_token_tensor = hidden_states[:, 0]
>        # 2. 非线性变换
>        pooled_output = self.dense(first_token_tensor)
>        pooled_output = self.activation(pooled_output)
>       return pooled_output
>
> ```

---

### 2. 编码器：BertEncoder

```python
class BertEncoder(nn.Module):
    def __init__(self, config):
        self.layer = nn.ModuleList([BertLayer(config, i) for i in range(config.num_hidden_layers)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        mode='multimodal',
    ):
        for i in range(self.config.num_hidden_layers):
            layer_module = self.layer[i]
            hidden_states = layer_module(
                hidden_states,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                mode=mode,
            )
        return hidden_states
```

多模态关键点：

* 多模态时，每个 Layer 都有机会执行 cross-attention。

* `encoder_hidden_states` 来自视觉模型（如 ViT 的输出），将图像特征注入到文本流中。

---

### 3. Transformer 层：BertLayer

```python
class BertLayer(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        mode=None,
    ):
        # 1. 自注意力（Self-Attention）
        attention_output = self.attention(hidden_states, attention_mask)

        # 2. 多模态交叉注意力（Cross-Attention）
        if mode == 'multimodal':
            attention_output = self.crossattention(
                attention_output,
                attention_mask,
                encoder_hidden_states,
                encoder_attention_mask,
            )

        return attention_output
```

多模态关键点：

* **自注意力**捕捉文本内部的依赖；

* 跨模态注意力（CrossAttention）让文本 Query 关注图像 Key 和 Value，实现信息融合。

---

### 4. Attention 模块：BertAttention

```python
class BertAttention(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        # attention 后应用一个 MLP
        return self.output(self_outputs, hidden_states)
```

> MLP 实现:
> ```python
> class BertSelfOutput(nn.Module):
>     def __init__(self, config):
>         super().__init__()
>         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
>         self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
>         self.dropout = nn.Dropout(config.hidden_dropout_prob)
>
>     def forward(self, hidden_states, input_tensor):
>         hidden_states = self.dense(hidden_states)
>         hidden_states = self.dropout(hidden_states)
>         hidden_states = self.LayerNorm(hidden_states + input_tensor)
>         return hidden_states
> ```

---

### 5. 核心计算：BertSelfAttention

```python
class BertSelfAttention(nn.Module):
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        # 获取 Query
        mixed_query_layer = self.query(hidden_states)

        # 判断是否为 Cross Attention
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算 Attention 分数（缩放点积注意力）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 加 Mask
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Softmax 归一化为权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # Dropout（来自 Transformer 原始实现）
        attention_probs_dropped = self.dropout(attention_probs)

        # 应用注意力权重
        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        # Reshape
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)

        return context_layer.view(*new_context_layer_shape)
```

### 6. 小结

多模态交互核心(Cross Attention):

| 项目        | 说明                                 |
| --------- | ---------------------------------- |
| Query     | 来自文本（`attention_output`）           |
| Key/Value | 来自图像（`encoder_hidden_states`）      |
| 作用        | 让文本动态关注图像区域，建立 Token 与视觉 Patch 的对齐 |
| 应用        | 文本问图（VQA）、图文检索、图文生成等多模态任务          |

总结:

```text
     +--------------------------+
     |      Text Embeddings     |
     +-----------+--------------+
                 |
        [Transformer Encoder]
                 |
        ┌────────┴───────────┐
        │ Self-Attention      │
        │ (Text <-> Text)     │
        └────────┬───────────┘
                 │
        ┌────────▼───────────┐
        │ Cross-Attention     │ <--- 图像特征作为 Key / Value
        │ (Text <-> Image)    │
        └────────┬───────────┘
                 │
         FeedForward + LayerNorm + Residual
```

## 自回归语言建模

BertLMHeadModel 是基于 BERT 构建的 语言建模头（Language Modeling Head）模型，其主要用于 自回归语言建模（Causal Language Modeling, CLM），尤其是在 多模态生成任务中充当解码器。它通常用于像 UNITER、VLBERT、MiniGPT-4、BLIP 等多模态架构中的文本生成部分。

```python
class BertLMHeadModel(BertPreTrainedModel):
    def __init__(self, config):
        self.bert = BertModel(config, add_pooling_layer=False)
        self.cls = BertOnlyMLMHead(config)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,          
        is_decoder=True,
        reduction='mean',
        mode='multimodal', 
    ):
        # 1. 调用BertModel
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            is_decoder=is_decoder,
            mode=mode,
        )
        
        # 2. 解码
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)
        
        if return_logits:
            return prediction_scores[:, :-1, :].contiguous()  


        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            shifted_prediction_scores = prediction_scores[:, :-1, :].contiguous()
            labels = labels[:, 1:].contiguous()
            loss_fct = CrossEntropyLoss(reduction=reduction, label_smoothing=0.1) 
            lm_loss = loss_fct(shifted_prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            if reduction=='none':
                lm_loss = lm_loss.view(prediction_scores.size(0),-1).sum(1)               

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=lm_loss,
            logits=prediction_scores,
        )
```

```python
#  对输入进行非线性变换: 投影 + 激活 + 归一化
class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 默认采用GELU激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class BertLMPredictionHead(nn.Module):
    def __init__(self, config):
        self.transform = BertPredictionHeadTransform(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, hidden_states):
        # 1. 非线性变换
        hidden_states = self.transform(hidden_states)
        # 2. 解码: 将(seq_len,hidden_size)中每个word映射到词空间
        hidden_states = self.decoder(hidden_states)
        return hidden_states
```
