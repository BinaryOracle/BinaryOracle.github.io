---
title: conda虚拟环境管理
icon: file
category:
  - tools
tag:
  - 已发布
footer: 技术共建，知识共享
date: 2025-06-11
author:
  - BinaryOracle
---

`conda虚拟环境管理` 

<!-- more -->

## 一、创建新环境

基本语法：

```bash
conda create --name <环境名> [包名]
```

> 可使用 -name（或 n）来命名环境。

示例1：创建一个空环境（只包含 Python）

```bash
conda create --name myenv
```
示例2：创建环境时指定 Python 版本

```bash
conda create --name myenv python=3.9
```
示例3：创建环境并安装一些常用包

```bash
conda create --name myenv python=3.8 numpy pandas
```

## 二、激活（切换）环境

激活环境的命令：

```bash
conda activate <环境名>

```

示例：

```bash
conda activate lmaffordance3d

```

激活后，你的终端提示符通常会显示当前环境的名字，例如：

```bash
(myenv) user@machine:~$

```

## 三、退出当前环境

要退出当前激活的环境，返回 base 环境：

```bash
conda deactivate
```

## 四、查看所有已创建的环境

你可以使用以下命令查看你所有的 conda 环境：

```bash
conda env list
# 或者
conda info --envs
```

输出示例：

```
# conda environments:
#
base                  *  /home/user/anaconda3
myenv                    /home/user/anaconda3/envs/myenv
testenv                  /home/user/anaconda3/envs/testenv
```

> 注：带星号 * 的表示当前激活的环境。
> 

## 五、删除已创建的环境

如果你想删除某个环境，可以使用：
    
```bash
conda env remove -n myenv
```

如需进一步帮助，可使用：

```bash
conda create --help
conda activate --help
```

## 六、查看当前激活的环境

查看当前conda激活的环境:

```bash
conda info
```

## 七、查看当前环境已安装的包

查看当前环境已安装的包：

```bash
conda list
```
## 八、在当前环境下安装包

根据 requirements.txt 安装所需要的依赖包:

```bash
conda activate 你的环境名  # 先激活你的conda环境
pip install -r requirements.txt
```
**重要说明：**

1. 在激活的 Conda 环境中使用 pip install，包会安装到该环境的 site-packages 中，不会影响其他环境或系统 Python

2. 如果未激活任何环境时使用 pip install，包可能会安装到基础环境或系统 Python 中

3. 建议总是先激活 Conda 环境再使用 pip，以避免安装到错误的位置

4. 可以使用 which pip 或 where pip (Windows) 确认你使用的是 Conda 环境中的 pip


## 九、常见错误

1. CondaError: Run 'conda init' before 'conda activate’

```bash
conda init

如果是 bash：
source ~/.bashrc

如果是 zsh：
bash

conda activate lavis
```