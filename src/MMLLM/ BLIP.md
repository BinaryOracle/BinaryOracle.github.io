---
title: BLIP 论文
icon: file
category:
  - 多模态
tag:
  - 多模态
  - 编辑中
footer: 技术共建，知识共享
date: 2025-07-20
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

#### 微调阶段

```python
def train(model, data_loader, optimizer, device):
    for i, (image, caption, _) in data_loader:
        loss = model(image, caption)      
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main(args, config):
    #### Dataset #### 
    train_dataset, val_dataset, test_dataset = create_dataset('caption_coco', config)  
    
    train_loader, val_loader, test_loader = create_loader([train_dataset, val_dataset, test_dataset],samplers,
                                                          batch_size=[config['batch_size']]*3,num_workers=[4,4,4],
                                                          is_trains=[True, False, False], collate_fns=[None,None,None])         

    #### Model #### 
    model = blip_decoder(pretrained=config['pretrained'], image_size=config['image_size'], vit=config['vit'], 
                           vit_grad_ckpt=config['vit_grad_ckpt'], vit_ckpt_layer=config['vit_ckpt_layer'], 
                           prompt=config['prompt'])
    
    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])

    #### Train ####         
    for epoch in range(0, config['max_epoch']):     
       train_stats = train(model, train_loader, optimizer, epoch, device)
```

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
        
        image_embeds = self.visual_encoder(image) 
        image_atts = torch.ones(image_embeds.size()[:-1],dtype=torch.long).to(image.device)
        
        text = self.tokenizer(caption, padding='longest', truncation=True, max_length=40, return_tensors="pt").to(image.device) 
        
        text.input_ids[:,0] = self.tokenizer.bos_token_id
        
        decoder_targets = text.input_ids.masked_fill(text.input_ids == self.tokenizer.pad_token_id, -100)         
        decoder_targets[:,:self.prompt_length] = -100
     
        decoder_output = self.text_decoder(text.input_ids, 
                                           attention_mask = text.attention_mask, 
                                           encoder_hidden_states = image_embeds,
                                           encoder_attention_mask = image_atts,                  
                                           labels = decoder_targets,
                                           return_dict = True,   
                                          )   
        loss_lm = decoder_output.loss
        
        return loss_lm
```