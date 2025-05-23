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

3. 

特殊token id:
- `[CLS]`: 101
- `[SEP]`: 102
- `[MASK]`: 103
- `[UNK]`: 100
- `[PAD]`: 0

```python
class BertTokenizer(PreTrainedTokenizer):
    ...
    
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A BERT sequence has the following format:
            single sequence: [CLS] X [SEP]
            pair of sequences: [CLS] A [SEP] B [SEP]
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls = [self.cls_token_id]
        sep = [self.sep_token_id]
        return cls + token_ids_0 + sep + token_ids_1 + sep

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

    def get_special_tokens_mask(self, token_ids_0, token_ids_1=None, already_has_special_tokens=False):
        """
        Retrieves sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer ``prepare_for_model`` or ``encode_plus`` methods.

        Args:
            token_ids_0: list of ids (must not contain special tokens)
            token_ids_1: Optional list of ids (must not contain special tokens), necessary when fetching sequence ids
                for sequence pairs
            already_has_special_tokens: (default False) Set to True if the token list is already formated with
                special tokens for the model

        Returns:
            A list of integers in the range [0, 1]: 0 for a special token, 1 for a sequence token.
        """

        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError("You should not supply a second sequence if the provided sequence of "
                                 "ids is already formated with special tokens for the model.")
            return list(map(lambda x: 1 if x in [self.sep_token_id, self.cls_token_id] else 0, token_ids_0))

        if token_ids_1 is not None:
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1]
```