---
icon: file
category:
  - 3D-VL
  - 3D Affordance
tag:
  - 3D-VL
  - 3D Affordance
  - 编辑中
footer: 技术共建，知识共享
date: 2025-05-30
cover: assets/cover/LASO.png
author:
  - BinaryOracle
---

`LASO: Language-guided Affordance Segmentation on 3D Object 论文代码解读与复现` 

<!-- more -->

# LASO 模型代码解读与复现

> 论文: [https://openaccess.thecvf.com/content/CVPR2024/papers/Li_LASO_Language-guided_Affordance_Segmentation_on_3D_Object_CVPR_2024_paper.pdf](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_LASO_Language-guided_Affordance_Segmentation_on_3D_Object_CVPR_2024_paper.pdf)
> 代码: [https://github.com/yl3800/LASO](https://github.com/yl3800/LASO)


这篇论文提出了一项新的任务和一个配套的数据集，旨在推动 **语言引导下的** 3D对象功能区域分割（Language-guided Affordance Segmentation on 3D Object, 简称 LASO）。

## 数据集

### 1. 基础数据来源

数据集基于 **3D-AffordanceNet** 提供的点云和功能区域标注构建：

- 每个物体都以点云形式表示；
- 点云中的每个点被标注为支持一个或多个功能类型（multi-class affordance labels），例如 grasp、open、lift、move 等；
- 这些功能标注是人工标注的，具有语义意义；

> **为什么使用 3D-AffordanceNet？** 
> - 因为它提供了高质量的点云和功能标注，能够很好地支持 LASO 的目标：根据自然语言问题找出与之相关的功能区域。


### 2. 构建问题（Question Crafting）

1. **选取物体-功能组合**：
   - 从 3D-AffordanceNet 中选取了 **58 种物体-功能组合**（如 mug-grasp、door-open 等）；
2. **手工设计问题**：
   - 对每种组合手工编写 **5 个代表性问题**；
3. **使用 GPT-4 扩展生成更多问题**：
   - 使用 GPT-4 为每个组合额外生成 **10 个问题**；
   - 总共得到 **870 个专家设计的问题**（58 × 15 = 870）；


![Affordance-Question数据可视化](LASO/1.png)   



在扩展过程中，GPT-4 生成的问题遵循以下三个关键原则，以确保问题多样性和语义丰富性：

| 原则 | 描述 |
|------|------|
| **Contextual Enrichment（上下文丰富化）** | 添加更多上下文细节，使问题更具体地连接目标对象的功能；<br>例：将 “Grasping scissors: top choice?” 改为 “Identify the key points on the scissors that ensure successful grasping.” |
| **Concise Phrasing（简洁表达）** | 提炼问题本质，使其简短但仍有意义； |
| **Structural Diversity（结构多样性）** | 使用不同句式结构（疑问句、陈述句等），防止模型偏向特定句式或长度； |

### 3. 标注 GT Mask（Ground Truth Mask）

对于每个问题，结合其对应的功能类型和原始点云标注信息，构造出对应的二值掩码 `gt_mask`：

- 每个点是否属于当前问题描述的功能区域；
- `gt_mask` 是 `(N,)` 形状的一维数组，其中 N 是点数；
- 数值可以是 0/1（binary mask），也可以是**软标签（soft label）**，表示点属于该功能区域的概率；
- 软标签通常用于边界模糊区域，反映点与功能核心区域的距离远近；

> 💡 注意：这些功能标签仅用于构造问题和定位正确功能区域，在训练和测试中不作为显式监督信号。

### 4. 数据集组织方式

数据总量：

- **总样本数**：19,751 个点云-问题配对；
- **物体类别数**：23 类；
- **功能类型数**：17 类；
- **问题总数**：870 个专家设计的问题；
- **每个物体类别可有多个形状实例**；
- **一个问题可以作用于多个物体类别**（泛化能力）；

数据集设置（两种模式）：

🔹 Seen（见过）

- 训练和测试阶段共享相似的物体类别和功能类型的分布；
- 目的是评估模型在熟悉场景下的表现；

🔹 Unseen（未见）

- 某些功能类型在特定物体类别下会从训练集中省略，但在测试集中保留；
- **目的是测试模型对新组合的泛化能力；**
- 例如：模型在训练期间学会了抓取包和杯子，但测试时要求“抓取耳机”——这是训练中未曾遇到过的功能-物体组合；

数据划分方式：

| 分区 | 物体类别数 | 问题数 | 样本数 |
|------|-------------|--------|---------|
| Train | 6883 | 638 | 16,120 |
| Val | 516 | 58 | 1,215 |
| Test | 1035 | 174 | 2,416 |

### 5. 数据增强与配对策略

训练阶段：

- 每次迭代中，每个形状实例随机匹配一个与其功能类型一致的问题；
- 随机配对使模型暴露于各种语义上下文中，提升泛化能力；

推理阶段（验证 & 测试）：

- 问题配对是固定的；
- 所有问题专属于评估阶段，不在训练中透露；
- 确保推理一致性，保持评估完整性；


### 6. 数据集统计信息（来自论文图3）

| 维度 | 内容 |
|------|------|
| 功能类型 | 17 类，如 grasp、open、lift、move 等 |
| 物体类别 | 23 类，如 mug、microwave、chair、door 等 |
| 物体-功能组合 | 58 种唯一组合（object-affordance pairs） |
| 问题总数 | 870 个定制化问题 |
| 点云-问题配对 | 19,751 对 |
| 点云来源 | 来自 3D-AffordanceNet，每个点云约 2048 个点 |

### 7. 代码实现

数据集加载的核心代码实现如下:

```python
class AffordQ(Dataset):

    def __init__(self,
                 split='train',
                 **kwargs
                 ):
        # 数据集存放目录         
        data_root='LASO_dataset'
        # 数据集类型: 训练集，评估集，测试集
        self.split = split
        # 所支持的23种物体类型和17种功能类型 
        classes = ["Bag", "Bed", "Bowl","Clock", "Dishwasher", "Display", "Door", "Earphone", "Faucet",
            "Hat", "StorageFurniture", "Keyboard", "Knife", "Laptop", "Microwave", "Mug",
            "Refrigerator", "Chair", "Scissors", "Table", "TrashCan", "Vase", "Bottle"]
        
        afford_cl = ['lay','sit','support','grasp','lift','contain','open','wrap_grasp','pour', 
                     'move','display','push','pull','listen','wear','press','cut','stab']
        
        self.cls2idx = {cls.lower():np.array(i).astype(np.int64) for i, cls in enumerate(classes)}
        self.aff2idx = {cls:np.array(i).astype(np.int64) for i, cls in enumerate(afford_cl)}

        with open(os.path.join(data_root, f'anno_{split}.pkl'), 'rb') as f:
            self.anno = pickle.load(f)
        
        with open(os.path.join(data_root, f'objects_{split}.pkl'), 'rb') as f:
            self.objects = pickle.load(f)

        # Load the CSV file, and use lower case
        self.question_df = pd.read_csv(os.path.join(data_root, 'Affordance-Question.csv'))
    
        self.len = len(self.anno)
       
        print(f"load {split} set successfully, lenth {len(self.anno)}")

        # sort anno by object class and affordance type
        self.sort_anno ={}
        for item in sorted(self.anno, key=lambda x: x['class']):
            key = item['class']
            value = {'shape_id': item['shape_id'], 'mask': item['mask'], 'affordance': item['affordance']}
            
            if key not in self.sort_anno:
                self.sort_anno[key] = [value]
            else:
                self.sort_anno[key].append(value)


    def find_rephrase(self, df, object_name, affordance):
        qid = str(np.random.randint(1, 15)) if self.split == 'train' else '0'
        qid = 'Question'+qid
        result = df.loc[(df['Object'] == object_name) & (df['Affordance'] == affordance), [qid]]
        if not result.empty:
            # return result.index[0], result.iloc[0]['Rephrase']
            return result.iloc[0][qid]
        else:
            raise NotImplementedError
            
         
    def __getitem__(self, index):
        data = self.anno[index]            
        shape_id = data['shape_id']
        cls = data['class']
        affordance = data['affordance']
        gt_mask = data['mask']
        point_set = self.objects[str(shape_id)]
        point_set,_,_ = pc_normalize(point_set)
        point_set = point_set.transpose()
            
        question = self.find_rephrase(self.question_df, cls, affordance)
        affordance = self.aff2idx[affordance]

        return point_set, self.cls2idx[cls], gt_mask, question, affordance     
```

### 8. 总结

LASO 数据集基于 3D-AffordanceNet 的点云和功能标注，结合人工+GPT-4 生成的多样化问题，构造出 19,751 个点云-问题配对，旨在实现语言引导下的 3D 功能区域分割，推动 3D 视觉与大语言模型（LLM）的深度融合。