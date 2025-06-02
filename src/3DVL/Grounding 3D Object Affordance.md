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

### Fusion 多模态特征融合模块

``python
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
        # P_enhance 是每个点云点引导下提取的图像信息增强后的点云点特征
        P_enhance = torch.bmm(i_feature, phi_i.permute(0,2,1))                         #[B, C, N_p]
        I = self.img_attention(I_enhance.mT)                                           #[B, N_i, C]
        P = self.point_attention(P_enhance.mT)                                         #[B, N_p, C]

        joint_patch = torch.cat((P, I), dim=1)                                       
        multi_feature = self.joint_attention(joint_patch)                              #[B, N_p+N_i, C]

        return multi_feature
```






