---
icon: file
category:
  - NLP
tag:
  - 预训练语言模型
  - 编辑中
footer: 探索AI边界，拥抱智能未来
date: 2025-05-19
cover: assets/cover/BERT.jpg
order: 2
author:
  - BinaryOracle
---

`图解Bert & Bert文本分类实战` 
<!-- more -->

# 图解 Bert

## 环境搭建

按序执行以下命令完成环境搭建:

```bash
git clone https://github.com/DA-southampton/Read_Bert_Code.git
cd Read_Bert_Code
conda create -n Read_Bert_Code python=3.9.22
conda activate Read_Bert_Code
```
本文使用的是谷歌的中文预训练模型：chinese_L-12_H-768_A-12.zip，模型有点大，我就不上传了，如果本地不存在，就点击[这里](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)直接下载,或者直接命令行运行

```shell
wget https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip
```

预训练模型下载下来之后，进行解压，然后将tf模型转为对应的pytorch版本即可。对应代码如下:

```shell
export BERT_BASE_DIR=/Users/zhandaohong/Read_Bert_Code/chinese_L-12_H-768_A-12

python convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path $BERT_BASE_DIR/bert_model.ckpt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --pytorch_dump_path $BERT_BASE_DIR/pytorch_model.bin
```

转化成功之后，将模型放入到仓库对应位置：

```shell
Read_Bert_Code/bert_read_step_to_step/prev_trained_model/
```

并重新命名为：

```shell
 bert-base-chinese
```
其次是准备训练数据，这里我准备做一个文本分类任务，使用的是Tnews数据集，这个数据集来源是[这里](https://github.com/ChineseGLUE/ChineseGLUE/tree/master/baselines/models_pytorch/classifier_pytorch/chineseGLUEdatasets)，分为训练，测试和开发集，我已经上传到了仓库中，具体位置在

```shell
Read_Bert_Code/bert_read_step_to_step/chineseGLUEdatasets/tnews
```
需要注意的一点是，因为我只是为了了解内部代码情况，所以准确度不是在我的考虑范围之内，所以我只是取其中的一部分数据，其中训练数据使用1k，测试数据使用1k，开发数据1k。

准备就绪，使用pycharm导入项目，准备调试，我的调试文件是`run_classifier.py`文件，对应的参数为

```shell
--model_type=bert --model_name_or_path=prev_trained_model/bert-base-chinese --task_name="tnews" --do_train --do_eval --do_lower_case --data_dir=./chineseGLUEdatasets/tnews --max_seq_length=128 --per_gpu_train_batch_size=16 --per_gpu_eval_batch_size=16 --learning_rate=2e-5 --num_train_epochs=4.0 --logging_steps=100 --save_steps=100 --output_dir=./outputs/tnews_output/ --overwrite_output_dir
```
然后启动 run_classifier.py 文件进行调试即可 , 所参考源仓库未提供requirements.txt文件，因此需要大家自行完成运行时缺失依赖包的安装。


## 数据预处理

1. 输入数据格式

```json
{
  "guid": "train-0",
  "label": "104",              // 文本分类任务: 文本对应的标签
  "text_a": "股票中的突破形态", 
  "text_b": null               // NSP任务: 用于判断给出的两个句子是否连续
}
```
> NSP (Next Sentence Prediction)

2. 文本分词 & 借助字典映射为word id

```json
"股票中的突破形态" --> ['股', '票', '中', '的', '突', '破', '形', '态'] --> [5500, 4873, 704, 4638, 4960, 4788, 2501, 2578]
```
> 对于字典中不存在的词 , 用 `[UNK]` 表示, 对应的id为 100

3. 过长截断策略

4. 添加特殊Token标记

![原序列添加特殊Token标记图](图解BERT/1.png)

```json
[101, 5500, 4873, 704, 4638, 4960, 4788, 2501, 2578, 102]
```

> BertTokenizer中的特殊token id:
> - `[CLS]`: 101
> - `[SEP]`: 102
> - `[MASK]`: 103
> - `[UNK]`: 100
> - `[PAD]`: 0

```python
    # BertTokenizer
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep
```
5. 创建句子辨识列表，用以区分不同的句子

![token_type_ids作用图解](图解BERT/2.png)

```python
     # BertTokenizer
     def create_token_type_ids_from_sequences(self, token_ids_0, token_ids_1=None):
        """
        Creates a mask from the two sequences passed to be used in a sequence-pair classification task.
        A BERT sequence pair mask has the following format:
        0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence

        if token_ids_1 is None, only returns the first portion of the mask (0's).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]
```
6. 创建用以区分special tokens部分的mask列表

![special_tokens_mask作用图解](图解BERT/3.png)

```python
    # BertTokenizer
    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
```
7. 超长截断

```python
       # PreTrainedTokenizer
       if max_length and len(encoded_inputs["input_ids"]) > max_length:
            encoded_inputs["input_ids"] = encoded_inputs["input_ids"][:max_length]
            encoded_inputs["token_type_ids"] = encoded_inputs["token_type_ids"][:max_length]
            encoded_inputs["special_tokens_mask"] = encoded_inputs["special_tokens_mask"][:max_length]
```

8. 生成padding部分的mask列表

![attention_mask作用图解](图解BERT/4.png)
```python
        # 生成注意力掩码，真实token对应1，填充token对应0
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)
 ```

 9. 所有序列都填充到max_length长度,不足长度用padding填充

![填充过程图](图解BERT/5.png)

```python
        # 记录输入长度
        input_len = len(input_ids)
        # 计算需要填充的长度 --- 所有输入序列等长，都等于max_length
        padding_length = max_length - len(input_ids)
        # 右填充
        input_ids = input_ids + ([pad_token] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
```

10. 数据集中每一个样本最终都会解析得到一个InputFeatures

![InputFeatures组成图解](图解BERT/6.png)

```python
features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label,
                          input_len=input_len))
```
> label 是当前文本对应的类别标签
> input_len 是序列实际长度(含special tokens)

11. 数据集预处理完后，将InputFeatures List列表组装起来得到需要的DataSet

```python
dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_lens,all_labels)
```

## 模型架构