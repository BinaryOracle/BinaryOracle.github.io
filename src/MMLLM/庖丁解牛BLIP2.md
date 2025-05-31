---
icon: file
category:
  - MMLLM
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-05-25
cover: assets/cover/BLIP2.png
author:
  - BinaryOracle
---

`庖丁解牛BLIP2` 

<!-- more -->

# 庖丁解牛BLIP2

> 论文: [https://arxiv.org/abs/2301.12597](https://arxiv.org/abs/2301.12597)
> 代码: [https://github.com/salesforce/LAVIS/tree/main/projects/blip2](https://github.com/salesforce/LAVIS/tree/main/projects/blip2)

## 背景

多模态模型在过往发展的过程中，曾有一段时期一直在追求更大的网络架构（image encoder 和 text encoder/decoder）和 数据集，从而导致更大的训练代价。例如CLIP，400M数据，需要数百个GPU训练数十天，如何降低模型训练成本，同时具有很好的性能？

这就是BLIP-2的起因，回顾下之前的多模态网络设计，三个模块（图像分支、文本分支、融合模块）:

![多模态网络设计](庖丁解牛BLIP2/1.png)

(a) 早期的图文多模态：图像分支依赖目标检测器，模态融合比较弱，如VSE++。

(b) 重点训练图像和文本特征提取，模态融合比较轻量，如CLIP。

(c) 图像特征提取和模态融合都很重。

(d) 侧重模态融合，特征提取网络相对轻量，如ViLT。

| 模块         | (a) | (b) | (c) | (d) | 理想情况 |
|--------------|-----|-----|-----|-----|----------|
| 视觉分支     | 重  | 重  | 重  | 轻  | 重       |
| 文本分支     | 轻  | 重  | 轻  | 轻  | 重       |
| 融合模块     | 轻  | 轻  | 重  | 重  | 轻       |
| 性能         | 一般| 好  | 好  | 一般| 好       |
| 训练代价     | 中  | 非常高 | 非常高 | 高 | 中   |

BLIP-2 基于 BLIP 架构，利用已有的ViT 和 LLM（均冻结）+ 一个的轻量Q-Former模块做模态融合，大幅降低训练成本。具有很强的zero-shot image-to-text generation能力，同时因LLM而具有了视觉推理能力。

## 模型结构

BLIP-2 框架按照 Two-Stage 策略预训练轻量级查询 Transformer 以弥合模态差距。

Stage 1: 不同模态数据的提取与融合。       

Stage 2: 把数据转换成LLM能识别的格式。

![Two-Stage流程](庖丁解牛BLIP2/2.png)

从冻结的Image Encoder引到Vision-Language表征学习。   

从冻结的LLM引到Vision-Language生成学习，实现Zero Shot图文生成。

### Stage 1: Representation Learning （表征学习）

![tage 1: Representation Learning （表征学习）](庖丁解牛BLIP2/3.png)

Q-Former 由两个transformer模块组成，输入包含三部分：

1. 冻结参数的Image Encoder提取的图像embeddings
2. Learned Queries

>  - Queries是一组可学习的embeddings，是第一个transformer模块的input，可认为是模型参数一部分
>  - 推理时，Queries被用来从image encoder输出的embeddings里提取与input text最相关的视觉信息

3. Input Text

Stage 1 使用 图像-文本对 进行预训练，目标是训练好 Q-Former，**以便 Queries 可以学习到如何更好地结合文本提取图片信息**。

对于Q-Former，一种比较好理解的方式：把Q-Former类比为一个Self-attention模块

- Q：learned queries
- K：input text
- V：image embeddings from Image Encoder

```python
class Blip2Qformer(Blip2Base):
    ...    
   
    def forward(self, samples):
        image = samples["image"]
        text = samples["text_input"]
        # (2,257,1408) --> attn: (2,257) ，(batch_size , seq_len , hidden_size)
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )
        # (1,32,768) --> (2,32,768) --> 共享内存
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        #  BertEncoder 的 squence_output
        image_feats = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(image.device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask, # padding mask
            return_dict=True,
        )
        text_feat = F.normalize( # 取CLS TOKEN ？
            self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1
        )

        ###============== Image-text Contrastive ===================###
        ...                   
        loss_itc = (
            F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
            + F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
        ) / 2

        ###============== Image-text Matching ===================###
        ...
        loss_itm = F.cross_entropy(logits, itm_labels)

        ##================= Image Captioning ========================##
        ...
        loss_lm = lm_output.loss

        return BlipOutput(
            loss=loss_itc + loss_itm + loss_lm,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
        )
```
