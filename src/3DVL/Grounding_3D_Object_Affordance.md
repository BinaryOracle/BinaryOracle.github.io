---
icon: file
category:
  - 3D-VL
tag:
  - 3D-VL
  - point cloud
  - 编辑中
footer: 技术共建，知识共享
date: 2025-05-30
cover: assets/cover/G3OA.png
author:
  - BinaryOracle
---

`Grounding 3D Object Affordance with Language Instructions, Visual  Observations and Interactions 论文代码解读与复现` 

<!-- more -->

# LMAffordance3D 模型代码解读与复现

> 论文: [https://arxiv.org/abs/2504.04744](https://arxiv.org/abs/2504.04744)
> 代码: [https://github.com/cn-hezhu/LMAffordance3D](https://github.com/cn-hezhu/LMAffordance3D)

## 环境配置 (待完善)

> 建议用Linux或者Windows系统进行测试，MacOS系统某些包的加载和依赖关系上存在问题，不方便进行处理。


## 模型结构

![模型结构图](Grounding_3D_Object_Affordance/1.png)

### LMAffordance3D 

```python
class LMAffordance3D(Blip2Base):
    ...
    def forward(self, img, point, description, label, inference_mode=False):
        '''
        img: [B, 3, H, W] -> 输入图像 (batch_size, channels, height, width)
        point: [B, 3, 2048] -> 点云数据 (batch_size, dimensions, num_points)
        description: 自然语言指令 (e.g., "Grasp the bottle")
        label: 真实标签，即每个点对应的 affordance 概率分布 (B, 2048, 1)
        inference_mode: 是否为推理模式（True/False）
        '''

        # 获取输入维度信息
        B, C, H, W = img.size()
        B, D, N = point.size()
        device = img.device  # 获取设备信息（CPU/GPU）

        # Step 1: 提取图像和点云的特征
        # --------------------------------------------------
        # 图像编码器：ResNet18 提取 2D 特征 F2D ∈ RB×CI×H×W
        img_feature = self.img_encoder(img)  # shape: [B, CI, H', W']

        # 点云编码器：PointNet++ 提取 3D 特征 F3D ∈ RB×CP×NP
        point_feature = self.point_encoder(point)  # shape: [B, CP, NP]

        # Step 2: 融合多模态空间特征
        # --------------------------------------------------
        # 使用 MLP 和自注意力机制融合图像与点云特征
        spatial_feature = self.fusion(img_feature, point_feature)  # shape: [B, NS, CS]
        
        # Step 3: 多模态特征投影到语言语义空间
        # --------------------------------------------------
        # 将融合后的空间特征通过适配器上采样到与语言模型匹配的维度
        if self.has_qformer:
            ...  # 如果使用 Q-Former，则进行额外处理
        else:
            multi_embeds = self.adapter_up(spatial_feature)  # shape: [B, NS, CL]
            image_atts = None  # 默认图像注意力掩码为空

        # Step 4: 对自然语言指令进行 Tokenization
        # --------------------------------------------------
        # 设置 tokenizer 的 padding 和 truncation 方向
        self.llm_tokenizer.padding_side = "right"
        self.llm_tokenizer.truncation_side = 'left'

        # 对语言指令进行分词，转换为 token ID 并生成 attention mask
        text_input_tokens = self.llm_tokenizer(
            description,
            return_tensors="pt",
            padding="longest",         # 填充至最长序列长度
            truncation=True,           # 截断过长文本
            max_length=self.max_txt_len,  # 最大文本长度
        ).to(device)

        # Step 5: 获取语言嵌入
        # --------------------------------------------------
        # 使用 LLM 的 embedding 层将 token ID 转换为嵌入向量
        inputs_embeds = self.llm_model.get_input_embeddings()(text_input_tokens.input_ids)
        # shape: [B, NL, CL] （NL=token数，CL=语言嵌入维度）

        # Step 6: 拼接多模态嵌入与语言嵌入
        # --------------------------------------------------
        # 调用 concat_input 函数，将图像+点云特征插入语言嵌入中
        llm_inputs, llm_attention_mask = self.concat_input(
            inputs_embeds, 
            text_input_tokens.attention_mask, 
            multi_embeds, 
            image_atts
        )
        # llm_inputs: [B, NL + NS, CL]
        # llm_attention_mask: [B, NL + NS]

        # Step 7: 使用 Vision-Language Model 进行联合推理
        # --------------------------------------------------
        # 在混合精度下运行 LLM，融合语言与视觉特征
        with self.maybe_autocast():
            hidden_states = self.llm_model(
                inputs_embeds=llm_inputs,
                attention_mask=llm_attention_mask,
                return_dict=False,  # 返回 tuple 格式输出
            )

        # Step 8: 降维适配器
        # --------------------------------------------------
        # 通过适配器层将 LLM 输出映射回合适维度
        hidden_states = self.adapter_down(hidden_states)  # shape: [B, NS + NL, CS]

        # 分割出 instructional feature 和 semantic feature
        semantic_feature, instructional_feature = torch.split(
            hidden_states, 
            split_size_or_sections=spatial_feature.size(1), 
            dim=1
        )

        # Step 9: 解码器融合所有特征以预测可操作性特征
        # --------------------------------------------------
        # 使用 cross-attention 融合 instruction, semantic, spatial features
        affordance_feature = self.affordance_decoder(
            spatial_feature, 
            instructional_feature, 
            semantic_feature
        )  # shape: [B, NA, CA]

        # Step 10: 使用分割头预测最终的 3D 可操作性热图
        # --------------------------------------------------
        out = self.head(spatial_feature, affordance_feature, point_feature)
        # 输出 shape: [B, 2048, 1]，表示每个点是否具有特定可操作性的概率

        # Step 11: 推理或训练分支
        # --------------------------------------------------
        if inference_mode == True:
            return out  # 仅返回预测结果
        else:
            loss_hm = self.loss_hm(out, label)  # 计算 heatmap 的损失（focal + dice）
            loss = loss_hm * self.w_hm  # 加权总损失
            return {
                "out": out, 
                "loss": loss, 
                "loss_hm": loss_hm
            }
```

### Step 2: 融合多模态空间特征

```python
class Fusion(nn.Module):
    def __init__(self, emb_dim = 512, num_heads = 4):
        super().__init__()
        self.emb_dim = emb_dim
        # 对点积结果进行缩放，防止 softmax 梯度消失或爆炸。
        self.div_scale = self.emb_dim ** (-0.5)
        self.num_heads = num_heads
       
        # 对图像和点云特征进行 非线性增强和空间对齐 ，使得它们能够在统一的语义空间中进行有效的跨模态交互。
        self.mlp = nn.Sequential(
            nn.Conv1d(self.emb_dim, 2*self.emb_dim, 1, 1),
            nn.BatchNorm1d(2*self.emb_dim),
            nn.ReLU(),
            nn.Conv1d(2*self.emb_dim, self.emb_dim, 1, 1),
            nn.BatchNorm1d(self.emb_dim),
            nn.ReLU()         
        )

        self.img_attention = Self_Attention(self.emb_dim, self.num_heads)
        self.point_attention = Self_Attention(self.emb_dim, self.num_heads)
        self.joint_attention = Self_Attention(self.emb_dim, self.num_heads)

    def forward(self, img_feature, point_feature):
        '''
        i_feature: [B, C, H, W]
        p_feature: [B, C, N_p]
        HW = N_i
        '''
        B, C, H, W = img_feature.size()
        img_feature = img_feature.view(B, self.emb_dim, -1)                            #[B, C, N_i]
        point_feature = point_feature[-1][1]
        
        # 对图像和点云特征进行 非线性增强和空间对齐 ，使得它们能够在统一的语义空间中进行有效的跨模态交互。
        p_feature = self.mlp(point_feature)
        i_feature = self.mlp(img_feature)
        
        # 跨模态注意力矩阵: 每个点云点与图像中每个位置之间的相似度得分
        phi = torch.bmm(p_feature.permute(0, 2, 1), i_feature)*self.div_scale          #[B, N_p, N_i]
       
        # 每列是一个 softmax 分布（每个图像位置对应的所有点云点）, 表示：“对于图像中的每一个位置，应该关注哪些点云点？”
        phi_p = F.softmax(phi,dim=1)
        # 每行是一个 softmax 分布（每个点云点对应的所有图像位置）, 表示：“对于点云中的每一个点，应该关注图像中的哪些位置？”
        phi_i = F.softmax(phi,dim=-1)  
       
        # I_enhance 是图像 patch 引导下提取的点云信息增强后的图像特征
        # 它不是直接包含原始图像 patch 的语义
        # 而是通过“点云中相关点”的方式重构图像 patch 的语义
        I_enhance = torch.bmm(p_feature, phi_p)                                        #[B, C, N_i]
        # P_enhance 是每个点云局部区域关键点引导下提取的图像信息增强后的点云关键点局部区域特征
        P_enhance = torch.bmm(i_feature, phi_i.permute(0,2,1))                         #[B, C, N_p]
       
        # 在跨模态融合后，进一步提取各自模态内部的语义一致性与结构关系，形成更稳定的联合表示。
        I = self.img_attention(I_enhance.mT)                                           #[B, N_i, C]
        P = self.point_attention(P_enhance.mT)                                         #[B, N_p, C]
        
        # 将图像patch和点云点拼接成一个统一的token序列
        # 使用自注意力机制提炼两个模态之间的语义一致性
        joint_patch = torch.cat((P, I), dim=1)                                       
        multi_feature = self.joint_attention(joint_patch)                              #[B, N_p+N_i, C]

        return multi_feature
```

### Step 3: 多模态特征投影到语言语义空间

```python
        # 将融合后的 3D 和 2D 特征从原始嵌入维度 (self.emb_dim) 映射到 LLM（语言模型）所使用的隐藏状态空间维度 （self.llm_model.config.hidden_size）。
        self.adapter_up = nn.Sequential(
            nn.Linear(self.emb_dim, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.llm_model.config.hidden_size)
        )
```

### Step 6: 拼接多模态嵌入与语言嵌入


```python
def concat_input(self, input_embeds, input_atts, multi_embeds, image_atts=None):
    '''
    将语言嵌入（text embeddings）与多模态嵌入（如图像、点云等）拼接在一起，
    构建 Vision-Language Model (VLM) 所需的输入格式。

    Args:
        input_embeds: (batch_size, sequence_length, hidden_size)
                      - 语言 token 经过 embedding 层后的结果。
        input_atts:   (batch_size, sequence_length)
                      - 语言部分的 attention mask（1 表示有效，0 表示填充）。
        multi_embeds: (batch_size, n, hidden_size)
                      - 多模态嵌入（如图像或点云特征），形状为 [B, n, H]。
        image_atts:   (batch_size, n), optional
                      - 多模态数据的 attention mask，默认为全 1（即所有 token 都有效）。

    Returns:
        llm_inputs:       (batch_size, total_length, hidden_size)
                        - 拼接后的输入嵌入，供 LLM 使用。
        llm_attention_mask: (batch_size, total_length)
                            - 对应的注意力掩码。
    '''

    # 初始化用于存储每个样本拼接后输入和 attention mask 的列表
    llm_inputs = []
    llm_attention_mask = []

    # 获取 batch size
    bs = multi_embeds.size()[0]

    # 对每个样本单独处理（逐个拼接）
    for i in range(bs):

        # 获取当前样本中多模态嵌入的维度信息：(n, dim)
        _, n, dim = multi_embeds.size()

        # 计算当前语言输入中有多少个有效 token（非 padding）
        this_input_ones = input_atts[i].sum()

        # 拼接嵌入向量：
        # 语言前半段（有效的部分）+ 多模态嵌入 + 语言后半段（padding 部分）
        llm_inputs.append(
            torch.cat([
                input_embeds[i][:this_input_ones],   # 有效语言部分
                multi_embeds[i],                     # 插入的多模态嵌入
                input_embeds[i][this_input_ones:]    # 剩余的语言 padding 部分
            ])
        )

        # 构建 attention mask：
        if image_atts is None:
            # 如果没有提供 image_atts，则默认多模态 token 都是有效的（mask 全为 1）
            llm_attention_mask.append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    torch.ones((n), device=multi_embeds.device, dtype=torch.long),
                    input_atts[i][this_input_ones:]
                ])
            )
        else:
            # 否则使用给定的 image_atts 来标记哪些多模态 token 是有效的
            llm_attention_mask.append(
                torch.cat([
                    input_atts[i][:this_input_ones],
                    image_atts[i],
                    input_atts[i][this_input_ones:]
                ])
            )

    # 将 list 转换为 batched tensor
    llm_inputs = torch.stack(llm_inputs, 0)
    llm_attention_mask = torch.stack(llm_attention_mask, 0)

    # 返回拼接好的输入和 attention mask
    return llm_inputs, llm_attention_mask
```

### Step 8: 降维适配器

```python
        # 降维适配器：将 LLM 输出的隐藏状态映射回原始嵌入维度（self.emb_dim）
        self.adapter_down = nn.Sequential(
            nn.Linear(self.llm_model.config.hidden_size, self.llm_model.config.hidden_size),
            nn.ReLU(),
            nn.Linear(self.llm_model.config.hidden_size, self.emb_dim)
        )
```









