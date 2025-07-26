---
title: BLIP 论文
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-07-26
author:
  - BinaryOracle
---

`BLIP: Bootstrapping Language-Image Pre-training for  Unified Vision-Language Understanding and Generation 论文解读` 

<!-- more -->

> 论文链接: [BLIP: Bootstrapping Language-Image Pre-training for  Unified Vision-Language Understanding and Generation](https://arxiv.org/abs/2201.12086)
> 代码链接: [https://github.com/salesforce/BLIP](https://github.com/salesforce/BLIP)

## Introduction


当前视觉-语言预训练（VLP）方法虽然在多模态任务上取得进展，但普遍存在两个问题：

1. **模型限制**：编码器模型不适合文本生成任务；编码器-解码器模型难以用于图文检索。
  
2. **数据质量差**：大多使用从网络收集的嘈杂图文对作为训练数据，监督信号不理想。


BLIP（Bootstrapping Language-Image Pre-training）是一个新颖的 VLP 框架，兼顾理解与生成能力。其两大创新点：

1. **MED 模型结构（Multimodal Mixture of Encoder-Decoder）**：

   * 同时支持编码器、图像条件编码器、图像条件解码器三种模式。

   * 联合训练三种任务：图文对比学习、图文匹配、图像条件语言建模。
   
   * 实现多任务预训练与灵活迁移。

2. **CapFilt 数据自举方法（Captioning and Filtering）**：

   * 使用训练好的 MED 模型构建两个模块：

     * 描述器（captioner）生成图像的合成描述；
   
     * 过滤器（filter）剔除原始和生成的低质量描述。
   
   * 在保留信息的同时提升训练数据质量。

实验结果与表现:

* BLIP 在多个任务（图文检索、图像描述、VQA 等）上取得**最先进性能**。

* 同时，在两个视频-语言任务上以**零样本方式**迁移也表现优异。

* 实验证明：描述器与过滤器的组合能显著提升性能，多样化描述更有利于学习。

## Related Work

### 视觉-语言预训练（VLP）

* **现状问题：**

  * 主流 VLP 方法依赖从网络抓取的图文对数据，虽然规模大，但包含大量噪声文本。
  
  * 尽管使用简单的过滤规则，噪声仍广泛存在。
  
  * 编码器模型适合理解类任务但难以生成文本；编码器-解码器适合生成任务但不适用于检索。

* **BLIP 的改进：**

  * 提出 **CapFilt**：通过“生成 + 过滤”的方式优化数据质量。
  
  * 提出 **MED 模型结构**：在保持预训练高效的前提下，同时兼顾理解与生成任务，提升泛化能力。

### 知识蒸馏（Knowledge Distillation）

* **现有做法：**

  * 知识蒸馏让小模型（学生）学习大模型（教师）的预测结果。
  
  * 自蒸馏也取得了不错效果，尤其在图像分类与部分 VLP 方法中已开始尝试。

* **BLIP 的新视角：**

  * CapFilt 可视为一种结构化的知识蒸馏方式：

    * **Captioner 模块**用生成的语义丰富描述进行蒸馏；
  
    * **Filter 模块**通过剔除噪声文本完成隐式知识过滤。

### 数据增强（Data Augmentation）

* **现有做法：**

  * 图像任务中数据增强广泛应用，但语言任务的数据增强较困难。
 
  * 近年来生成模型被用于文本任务的样本合成，但多用于低资源语言场景。

* **BLIP 的贡献：**

  * 展示了在**大规模视觉-语言预训练中**使用合成图像描述的独特优势，提升了多模态学习效果。

## Code Implementation

### CapFilt 模块实现

BLIP 使用 CapFilt 对多个大规模噪声网页图文数据集（包括 CC12M、CC3M 和 SBU Captions）进行增强，首先通过 captioner 为图像生成合成文本，再通过 filter 过滤掉与图像不匹配的原始和合成文本，最终构建出高质量的自举数据集（bootstrapped dataset），用于预训练新模型。

在 CapFilt 模块微调阶段，BLIP 则基于高质量人工标注的数据集如 COCO Captions、Visual Genome 和 Flickr30K 进行训练和评估。

经过 CapFilt 处理后，输出的数据集是经过图文对齐质量优化的图文对集合，有效提升了下游任务中的表现。

>  官方代码库并没有非常清晰指明CapFilt模块的实现代码位置，但是官方仓库的ISSUE给出了明确答复: [https://github.com/salesforce/BLIP/issues/86?utm_source=chatgpt.com](https://github.com/salesforce/BLIP/issues/86?utm_source=chatgpt.com)

#### Captioner 模块

##### 微调阶段

Captioner 基于 Coco 数据集进行微调:

```python
# 训练函数：执行一个 epoch 的训练流程
def train(model, data_loader, optimizer, device):
    for i, (image, caption, _) in data_loader:
        loss = model(image, caption)  # 前向传播，计算语言建模损失        
        optimizer.zero_grad()         # 清除旧梯度
        loss.backward()               # 反向传播
        optimizer.step()              # 更新模型参数

# 主流程：加载数据、初始化模型和优化器、执行多轮训练
def main(args, config):
    #### Dataset #### 
    # 加载 COCO Caption 数据集的训练/验证/测试划分
    train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config)  
    
    # 构造对应的数据加载器
    train_loader, val_loader, test_loader = create_loader(
        [train_dataset, val_dataset, test_dataset], samplers,
        batch_size=[config['batch_size']]*3, num_workers=[4, 4, 4],
        is_trains=[True, False, False], collate_fns=[None, None, None]
    )

    #### Model #### 
    # 初始化 BLIP 解码器模型（用于图像字幕生成），加载预训练视觉编码器与文本解码器
    model = blip_decoder(
        pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
        vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
        prompt=config['prompt']
    )
    
    # 初始化优化器（AdamW）
    optimizer = torch.optim.AdamW(
        params=model.parameters(), 
        lr=config['init_lr'], 
        weight_decay=config['weight_decay']
    )

    #### Train ####         
    for epoch in range(0, config['max_epoch']):
        # 每个 epoch 执行一次训练
        train_stats = train(model, train_loader, optimizer, epoch, device)
```

训练阶段唯一需要注意的一点就是数据集的构造过程中，会在Coco数据集每个样本原有Caption的基础上添加一个Prompt:

```python
class coco_karpathy_train(Dataset):
    def __getitem__(self, index):    
        ann = self.annotation[index]
        
        image_path = os.path.join(self.image_root,ann['image'])        
        image = Image.open(image_path).convert('RGB')   
        image = self.transform(image)

        # 在caption前添加prompt , prompt 默认为 'a picture of '
        caption = self.prompt+pre_caption(ann['caption'], self.max_words) 
        
        # self.img_ids[ann['image_id']]: 取出图像 ID 对应的 索引编号
        return image, caption, self.img_ids[ann['image_id']] 
```

下面将给出Captioner模块基于Coco数据集，采用 `next token predict` 方法进行训练的代码实现：

```python
class BLIP_Decoder(nn.Module):
    def __init__(self,                 
                 med_config = 'configs/med_config.json',  
                 image_size = 384,
                 vit = 'base',
                 vit_grad_ckpt = False,
                 vit_ckpt_layer = 0,
                 prompt = 'a picture of ',
                 ):
        """
        BLIP Captioner模块初始化，实现论文中提出的图像-文本跨模态编码器-解码器架构

        Args:
            med_config (str): 混合编码器-解码器模型配置文件路径，对应论文3.1节中提到的多模态融合模块配置
            image_size (int): 输入图像尺寸，论文4.1节实验设置中使用384x384
            vit (str): 视觉Transformer模型大小，论文中采用ViT-Base作为默认视觉编码器
            vit_grad_ckpt (bool): 是否使用梯度检查点优化ViT显存占用，论文附录A中提到的训练优化策略
            vit_ckpt_layer (int): ViT梯度检查点层数，用于平衡训练效率与显存使用
            prompt (str): 图像描述生成的引导提示词，对应论文3.2节中使用的prompt engineering技术
        """
        super().__init__()
        
        self.visual_encoder, vision_width = create_vit(vit,image_size, vit_grad_ckpt, vit_ckpt_layer)  # 初始化视觉编码器，对应论文图1中的视觉Transformer
        self.tokenizer = init_tokenizer()   # 初始化文本分词器，采用BERT分词器实现论文中的文本预处理
        med_config = BertConfig.from_json_file(med_config)
        med_config.encoder_width = vision_width
        self.text_decoder = BertLMHeadModel(config=med_config)    # 初始化文本解码器，实现论文3.1节中的跨模态解码器
        
        self.prompt = prompt  # 存储图像描述引导提示词，用于论文3.3节中的条件生成任务
        self.prompt_length = len(self.tokenizer(self.prompt).input_ids)-1  # 计算提示词token长度，用于后续解码时区分提示与生成文本
        
    def forward(self, image, caption):
        # 提取图像特征表示
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)

        # 编码文本输入，并替换开头 token 为 [BOS]
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device)
        text.input_ids[:, 0] = self.tokenizer.bos_token_id

        # 构建语言建模标签：屏蔽掉 padding 和 prompt 部分
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:, :self.prompt_length] = -100

        # 调用跨模态解码器，执行语言建模训练
        decoder_output = self.text_decoder(
            text.input_ids,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            labels=decoder_targets,
            return_dict=True
        )   

        loss_lm = decoder_output.loss  # 提取语言建模损失
        return loss_lm
```

> [BertLMHeadModel自回归语言建模实现](http://localhost:8080/MMLLM/%E5%A4%9A%E6%A8%A1%E6%80%81%E5%B8%B8%E7%94%A8%E6%94%B9%E7%BC%96Bert%E5%AE%9E%E7%8E%B0.html#%E8%87%AA%E5%9B%9E%E5%BD%92%E8%AF%AD%E8%A8%80%E5%BB%BA%E6%A8%A1)

##### 生成阶段

当 Captioner 模块在 Coco 数据集上进行训练后，即可用于生成图像描述。

```python
def generate(self, image, sample=False, num_beams=3, max_length=30, min_length=10, top_p=0.9, repetition_penalty=1.0):
    # 通过视觉编码器提取图像的视觉特征（image embeddings）
    image_embeds = self.visual_encoder(image)
      
    if not sample:
        # 如果使用 beam search 生成，则需要将图像特征复制 num_beams 份
        # 这是为了每个 beam 都能接收相同的图像信息
        image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        
    # 构造图像 attention mask，全为 1，表示图像特征没有被 mask
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    
    # 构造 encoder-decoder 所需的关键词参数（即图像特征作为 cross-attention 的条件输入）
    model_kwargs = {
        "encoder_hidden_states": image_embeds,
        "encoder_attention_mask": image_atts
    }
    
    # 构造输入的 prompt，格式为 ["a picture of ", "a picture of ", ...]
    prompt = [self.prompt] * image.size(0)
    
    # 对 prompt 进行分词并转为 tensor，作为 decoder 的输入起点（input_ids）
    input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
    
    # 强制将每个样本开头的 token 设为 [BOS]（beginning-of-sentence）
    input_ids[:, 0] = self.tokenizer.bos_token_id
    
    # 移除最后一个 token（保持 prompt 是 decoder 的 prefix）
    input_ids = input_ids[:, :-1]

    if sample:
        # ---------- 使用 nucleus sampling（核采样）生成 ----------
        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            do_sample=True,                 # 启用采样
            top_p=top_p,                    # nucleus 采样的阈值
            num_return_sequences=1,        # 每张图像生成一个序列
            eos_token_id=self.tokenizer.sep_token_id,  # 使用 [SEP] 作为结束标记
            pad_token_id=self.tokenizer.pad_token_id,  # 使用 [PAD] 作为 padding
            repetition_penalty=1.1,        # 防止重复生成
            **model_kwargs                  # 传入图像编码信息
        )
    else:
        # ---------- 使用 beam search（束搜索）生成 ----------
        outputs = self.text_decoder.generate(
            input_ids=input_ids,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,           # beam 数量
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            repetition_penalty=repetition_penalty,  # 重复惩罚项
            **model_kwargs
        )

    # ---------- 解码生成的 token 序列为文本 ----------
    captions = []
    for output in outputs:
        caption = self.tokenizer.decode(output, skip_special_tokens=True)
        # 去掉 prompt 的前缀，只保留生成部分
        captions.append(caption[len(self.prompt):])

    return captions
```

#### Filter 模块

##### 微调阶段

Filter 模块同样也是基于 Coco 数据集进行微调，但采用图文检索和图文匹配作为训练目标:

```python
def train(model, data_loader, optimizer, epoch, device, config):
    for i,(image, caption, idx) in data_loader:
        # 采用Moco的动量慢更新策略进行学习
        if epoch>0:
            alpha = config['alpha']
        else:
            alpha = config['alpha']*min(1,i/len(data_loader))
        
        # idx 是每个图像对应的索引编号
        loss_ita, loss_itm = model(image, caption, alpha=alpha, idx=idx)                  
        loss = loss_ita + loss_itm
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()    

def main(args, config):
    #### Dataset #### 
    # 数据集处理方面，同样会为Coco数据集中每个样本的Caption前添加固定长度的Prompt: 'a picture of'
    train_dataset, val_dataset, test_dataset = create_dataset('retrieval_%s'%config['dataset'], config)  
   
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size_train']]+[config['batch_size_test']]*2,
                                                          num_workers=[4,4,4],
                                                          is_trains=[True, False, False], 
                                                          collate_fns=[None,None,None])   
   

    #### Model #### 
    model = blip_retrieval(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                             vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                             queue_size=config['queue_size'], negative_all_rank=config['negative_all_rank'])

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay']) 
    
    for epoch in range(0, config['max_epoch']):    
        train(model, train_loader, optimizer, epoch, device, config)  
```

训练过程代码实现基本遵循Moco论文中所提出的动量慢更新对比学习代码实现，下面先给出 `BLIP_Retrieval` 模型 `init` 方法实现:

```python
class BLIP_Retrieval(nn.Module):
    def __init__(self,
                 med_config='configs/med_config.json',
                 image_size=384,
                 vit='base',
                 vit_grad_ckpt=False,
                 vit_ckpt_layer=0,
                 embed_dim=256,
                 queue_size=57600,
                 momentum=0.995,
                 negative_all_rank=False):

        # 初始化视觉编码器
        self.visual_encoder, vision_width = create_vit(vit, image_size, vit_grad_ckpt, vit_ckpt_layer)

        # 初始化文本编码器和分词器
        self.tokenizer = init_tokenizer()
        self.text_encoder = BertModel(config=med_config, add_pooling_layer=False)

        # 特征投影层（用于对比学习）
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        # ITM分类头（判断图文是否匹配）
        self.itm_head = nn.Linear(text_width, 2)

        # 创建动量编码器（momentum encoder）
        self.visual_encoder_m, vision_width = create_vit(vit, image_size)
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertModel(config=med_config, add_pooling_layer=False)
        self.text_proj_m = nn.Linear(text_width, embed_dim)

        # 配对编码器用于同步参数
        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m]]
        self.copy_params()

        # 初始化对比学习的负样本队列
        self.register_buffer("image_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, queue_size))
        self.register_buffer("idx_queue", torch.full((1, queue_size), -100))
        self.register_buffer("ptr_queue", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

        self.queue_size = queue_size
        self.momentum = momentum
        self.temp = nn.Parameter(0.07 * torch.ones([]))  # 对比学习温度参数

        self.negative_all_rank = negative_all_rank
```

前向传播过程主要是为了计算两个训练目标的损失:

- 图文对比目标（ITC）

- 图文匹配目标（ITM）

代码整体流程比较长，我们切分为多个步骤进行解析:

1. 提取图像和文本特征

```python
    def forward(self, image, caption, alpha, idx):
        # 图像特征提取和投影
        # image: (B, 3, H, W) -> image_embeds: (B, N, D)
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)  # (B, N)
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)  # 只取CLS Token做投影: (B, D_proj)

        # 文本特征提取和投影
        text = self.tokenizer(caption, padding='max_length', truncation=True, max_length=35,
                              return_tensors="pt").to(image.device)  # text.input_ids: (B, L)
        text_output = self.text_encoder(text.input_ids, attention_mask=text.attention_mask,
                                        return_dict=True, mode='text')
        text_feat = F.normalize(self.text_proj(text_output.last_hidden_state[:, 0, :]), dim=-1)  # 只取CLS Token做投影: (B, D_proj)
```

2. 构造图文匹配矩阵 (Target)

```python
        # 构造图文匹配矩阵
        idx = idx.view(-1, 1)  # (B, 1)
        idx_all = torch.cat([idx.t(), self.idx_queue.clone().detach()], dim=1)  # (1, B + Q)
        pos_idx = torch.eq(idx, idx_all).float()  # (B, B + Q)
        sim_targets = pos_idx / pos_idx.sum(1, keepdim=True)  # (B, B + Q)
```
这里需要和 MoCo 论文实现进行区分，MoCo 中采用的是每个 query（图像）只对应一个 positive（key），因此正负样本是 one-hot 编码，contrastive loss 是严格的一对一；而在 BLIP 中，由于图文对来自自然语言描述，可能存在多个正样本（即同一个图像可以有多个 caption），而且动量队列可能重复包含同一个样本（multi-hot），因此这里构造 `sim_targets` 时不是用 one-hot，而是通过 `pos_idx` 判断当前图文对与队列中哪些样本是正对（`idx` 相等），然后用行归一化将多个正样本平均分配权重，形成 soft target 分布，从而使对比学习更加稳健。

举例:

```python
# 1. 环境
idx = [[7], [13], [20]]  # B=3
idx_queue = [1, 7, 5, 13, 9, 30]  # Q=6
idx_all = [7, 13, 20, 1, 7, 5, 13, 9, 30]  # shape: (1, 9)

# 2. 构造图文匹配矩阵
pos_idx = torch.eq(idx, idx_all)
# 第1行: [1, 0, 0, 0, 1, 0, 0, 0, 0]
# 第2行: [0, 1, 0, 0, 0, 0, 1, 0, 0]
# 第3行: [0, 0, 1, 0, 0, 0, 0, 0, 0]

# 3. 进行归一化
# 第1行: [0.5, 0, 0, 0, 0.5, 0, 0, 0, 0]
# 第2行: [0, 0.5, 0, 0, 0, 0, 0.5, 0, 0]
# 第3行: [0, 0, 1.0, 0, 0, 0, 0, 0, 0]
```
3.  动量慢更新 + 软标签计算

```python
       # 使用动量编码器获取特征
        with torch.no_grad():
            self._momentum_update()
            image_embeds_m = self.visual_encoder_m(image)  # (B, N, D)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)  # (B, D_proj)
            image_feat_m_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)  # (D_proj, B + Q)

            text_output_m = self.text_encoder_m(text.input_ids, attention_mask=text.attention_mask,return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)  # (B, D_proj)
            text_feat_m_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)  # (D_proj, B + Q)
            
            sim_i2t_m = image_feat_m @ text_feat_m_all / self.temp  # (B, B + Q)
            sim_t2i_m = text_feat_m @ image_feat_m_all / self.temp  # (B, B + Q)
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets  # (B, B + Q)
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets  # (B, B + Q)
```

4.  计算ITC损失 + 更新动量队列

```python
        # 当前 batch 与动量队列的相似度
        sim_i2t = image_feat @ text_feat_m_all / self.temp  # (B, B + Q)
        sim_t2i = text_feat @ image_feat_m_all / self.temp  # (B, B + Q)

        # 对比损失（InfoNCE）
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()
        loss_ita = (loss_i2t + loss_t2i) / 2

        # 更新队列
        idxs = concat_all_gather(idx)  # (B*, 1)
        self._dequeue_and_enqueue(image_feat_m, text_feat_m, idxs)
```

5. 正样本编码 + 难负样本采样

```python
   ### 图文匹配任务 (ITM) ###

        encoder_input_ids = text.input_ids.clone()  # (B, L)
        encoder_input_ids[:, 0] = self.tokenizer.enc_token_id
        bs = image.size(0)

        # 正样本编码
        output_pos = self.text_encoder(encoder_input_ids,
                                       attention_mask=text.attention_mask,
                                       encoder_hidden_states=image_embeds,
                                       encoder_attention_mask=image_atts,
                                       return_dict=True)  # last_hidden_state: (B, L, D)

        # 采样难负样本（是否跨 GPU）
        if self.negative_all_rank:
            with torch.no_grad():
                mask = torch.eq(idx, idxs.t())  # (B, B*)
                image_feat_world = concat_all_gather(image_feat)  # (B*, D_proj)
                text_feat_world = concat_all_gather(text_feat)  # (B*, D_proj)
                sim_i2t = image_feat @ text_feat_world.t() / self.temp  # (B, B*)
                sim_t2i = text_feat @ image_feat_world.t() / self.temp  # (B, B*)
                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            image_embeds_world = all_gather_with_grad(image_embeds)  # (B*, N, D)
            image_embeds_neg = [image_embeds_world[torch.multinomial(weights_t2i[b], 1).item()] for b in range(bs)]  # list[(N, D)]
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)  # (B, N, D)

            input_ids_world = concat_all_gather(encoder_input_ids)  # (B*, L)
            att_mask_world = concat_all_gather(text.attention_mask)  # (B*, L)
            text_ids_neg = [input_ids_world[torch.multinomial(weights_i2t[b], 1).item()] for b in range(bs)]  # list[(L,)]
            text_atts_neg = [att_mask_world[torch.multinomial(weights_i2t[b], 1).item()] for b in range(bs)]  # list[(L,)]
        else:
            with torch.no_grad():
                mask = torch.eq(idx, idx.t())
                sim_i2t = image_feat @ text_feat.t() / self.temp
                sim_t2i = text_feat @ image_feat.t() / self.temp
                weights_i2t = F.softmax(sim_i2t, dim=1)
                weights_i2t.masked_fill_(mask, 0)
                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_t2i.masked_fill_(mask, 0)

            image_embeds_neg = [image_embeds[torch.multinomial(weights_t2i[b], 1).item()] for b in range(bs)]  # list[(D,)]
            image_embeds_neg = torch.stack(image_embeds_neg, dim=0)  # (B, D)
            text_ids_neg = [encoder_input_ids[torch.multinomial(weights_i2t[b], 1).item()] for b in range(bs)]  # list[(L,)]
            text_atts_neg = [text.attention_mask[torch.multinomial(weights_i2t[b], 1).item()] for b in range(bs)]  # list[(L,)]
        
        text_ids_neg = torch.stack(text_ids_neg, dim=0)  # (B, L)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)  # (B, L)
```

> `idx` 是当前 `GPU` 上 本地批次（batch）的样本索引（`shape: (B, 1)`）。

> `idxs` 是通过 `concat_all_gather(idx)` 得到的 所有 `GPU` 上所有样本索引的集合（`shape: (total_batch_size, 1)`）。


6. 构造两组负样本

```python
        # [正样本，负样本]
        text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)  # (2B, L)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)  # (2B, L)
        # [负样本，正样本]
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)  # (2B, N, D)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)  # (2B, N)

        # 两组负样本编码
        output_neg = self.text_encoder(text_ids_all,
                                       attention_mask=text_atts_all,
                                       encoder_hidden_states=image_embeds_all,
                                       encoder_attention_mask=image_atts_all,
                                       return_dict=True)
```

7. 计算ITM损失 + 返回ITC损失和ITM损失

```python
        # ITM 分类损失
        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)  # (3B, D)
        vl_output = self.itm_head(vl_embeddings)  # (3B, 2)
        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)], dim=0).to(image.device)  # (3B,)
        loss_itm = F.cross_entropy(vl_output, itm_labels)

        return loss_ita, loss_itm
```

##### 过滤阶段

```python

```