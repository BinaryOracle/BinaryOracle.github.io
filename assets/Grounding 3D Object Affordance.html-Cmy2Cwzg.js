import{_ as t}from"./plugin-vue_export-helper-DlAUqK2U.js";import{c as a,a as n,e as r,d as i,o}from"./app-BXntV8XD.js";const s={};function c(d,e){return o(),a("div",null,[e[0]||(e[0]=n("p",null,[n("code",null,"Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions 论文代码解读与复现")],-1)),r(" more "),e[1]||(e[1]=i(`<h1 id="lmaffordance3d-模型代码解读与复现" tabindex="-1"><a class="header-anchor" href="#lmaffordance3d-模型代码解读与复现"><span>LMAffordance3D 模型代码解读与复现</span></a></h1><blockquote><p>论文: <a href="https://arxiv.org/abs/2504.04744" target="_blank" rel="noopener noreferrer">https://arxiv.org/abs/2504.04744</a><br> 代码: <a href="https://github.com/cn-hezhu/LMAffordance3D" target="_blank" rel="noopener noreferrer">https://github.com/cn-hezhu/LMAffordance3D</a></p></blockquote><h2 id="环境配置-待完善" tabindex="-1"><a class="header-anchor" href="#环境配置-待完善"><span>环境配置 (待完善)</span></a></h2><blockquote><p>建议用Linux或者Windows系统进行测试，MacOS系统某些包的加载和依赖关系上存在问题，不方便进行处理。</p></blockquote><h2 id="模型结构" tabindex="-1"><a class="header-anchor" href="#模型结构"><span>模型结构</span></a></h2><h3 id="fusion-多模态特征融合模块" tabindex="-1"><a class="header-anchor" href="#fusion-多模态特征融合模块"><span>Fusion 多模态特征融合模块</span></a></h3><p>\`\`python<br> class Fusion(nn.Module):<br> def <strong>init</strong>(self, emb_dim = 512, num_heads = 4):<br> super().<strong>init</strong>()<br> self.emb_dim = emb_dim<br> # 对点积结果进行缩放，防止 softmax 梯度消失或爆炸。<br> self.div_scale = self.emb_dim ** (-0.5)<br> self.num_heads = num_heads</p><pre><code>    # 对图像和点云特征进行 非线性增强和空间对齐 ，使得它们能够在统一的语义空间中进行有效的跨模态交互。
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
    &#39;&#39;&#39;
    i_feature: [B, C, H, W]
    p_feature: [B, C, N_p]
    HW = N_i
    &#39;&#39;&#39;
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
</code></pre><div class="language- line-numbers-mode" data-highlighter="shiki" data-ext="" style="--shiki-light:#383A42;--shiki-dark:#abb2bf;--shiki-light-bg:#FAFAFA;--shiki-dark-bg:#282c34;"><pre class="shiki shiki-themes one-light one-dark-pro vp-code"><code><span class="line"><span></span></span></code></pre><div class="line-numbers" aria-hidden="true" style="counter-reset:line-number 0;"><div class="line-number"></div></div></div>`,9))])}const l=t(s,[["render",c]]),f=JSON.parse('{"path":"/3DVL/Grounding%203D%20Object%20Affordance.html","title":"LMAffordance3D 模型代码解读与复现","lang":"zh-CN","frontmatter":{"icon":"file","category":["3D-VL"],"tag":["3D-VL","point cloud","编辑中"],"footer":"技术共建，知识共享","date":"2025-05-30T00:00:00.000Z","cover":"assets/cover/G3OA.png","author":["BinaryOracle"],"description":"Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions 论文代码解读与复现","head":[["script",{"type":"application/ld+json"},"{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"Article\\",\\"headline\\":\\"LMAffordance3D 模型代码解读与复现\\",\\"image\\":[\\"\\"],\\"datePublished\\":\\"2025-05-30T00:00:00.000Z\\",\\"dateModified\\":\\"2025-06-02T01:39:26.000Z\\",\\"author\\":[{\\"@type\\":\\"Person\\",\\"name\\":\\"BinaryOracle\\"}]}"],["meta",{"property":"og:url","content":"https://mister-hope.github.io/3DVL/Grounding%203D%20Object%20Affordance.html"}],["meta",{"property":"og:site_name","content":"MetaMind"}],["meta",{"property":"og:title","content":"LMAffordance3D 模型代码解读与复现"}],["meta",{"property":"og:description","content":"Grounding 3D Object Affordance with Language Instructions, Visual Observations and Interactions 论文代码解读与复现"}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:locale","content":"zh-CN"}],["meta",{"property":"og:updated_time","content":"2025-06-02T01:39:26.000Z"}],["meta",{"property":"article:author","content":"BinaryOracle"}],["meta",{"property":"article:tag","content":"编辑中"}],["meta",{"property":"article:tag","content":"point cloud"}],["meta",{"property":"article:tag","content":"3D-VL"}],["meta",{"property":"article:published_time","content":"2025-05-30T00:00:00.000Z"}],["meta",{"property":"article:modified_time","content":"2025-06-02T01:39:26.000Z"}]]},"git":{"createdTime":1748594622000,"updatedTime":1748828366000,"contributors":[{"name":"BinaryOracle","username":"BinaryOracle","email":"3076679680@qq.com","commits":1,"url":"https://github.com/BinaryOracle"},{"name":"大忽悠","username":"","email":"3076679680@qq.com","commits":3}]},"readingTime":{"minutes":2.02,"words":605},"filePathRelative":"3DVL/Grounding 3D Object Affordance.md","excerpt":"<p><code>Grounding 3D Object Affordance with Language Instructions, Visual  Observations and Interactions 论文代码解读与复现</code></p>\\n","autoDesc":true}');export{l as comp,f as data};
