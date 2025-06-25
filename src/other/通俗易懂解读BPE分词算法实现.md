---
title: 通俗易懂解读BPE分词算法实现
icon: file
category:
  - tokenizer
tag:
  - 已发布
footer: 技术共建，知识共享
date: 2025-06-22
author:
  - BinaryOracle
---

`通俗易懂解读BPE分词算法实现` 

<!-- more -->

# BPE (Byte Pair Encoding)

**BPE（Byte Pair Encoding，字节对编码）**是一种基于频率统计的子词分词算法 ，广泛用于现代自然语言处理任务中，特别是在像 BERT、GPT 和 LLaMA 这样的大模型中。它的核心思想是通过**不断合并最常见的字符对来构建一个高效的词汇表**。

**BPE 的核心思想**:

- 从字符级别开始，逐步合并高频的字符对。

- 最终生成一个既能表示常见单词，又能拆解未知词的子词词汇表 。

- 可以有效控制词汇表大小，同时避免“未登录词”问题（OOV, Out-of-Vocabulary）。

## 预训练过程

**BPE 算法预训练工作流程**:

> 训练语料为: Hello World , Hey Wow

**1. 读取训练语料，同时完成断句分词任务**

```python
# filepaths: 训练语料所在的文件列表
def create_vocab(filepaths: List[str]) -> Dict[str, int]:
    # 获取所有单词和每个单词的出现次数词典
    vocab = defaultdict(int)
    for path in tqdm(filepaths, desc='Creating vocabulary'):
        text = open(path, 'r', encoding='utf-8-sig').read()
        # 利用NLTK库提供的sent_tokenize方法完成断句功能，即将原文本按照空格，句号等标点符号结合语义进行断句。
        sentences = sent_tokenize(text)
        # 遍历句子列表
        for sentence in sentences:
            #  利用NLTK库提供的wordpunct_tokenize方法完成分词功能
            tokens = wordpunct_tokenize(sentence)
            #  记录每个词的出现次数 
            for token in tokens:
                vocab[token] += 1
    # vocab: 记录每个词的出现次数的词典
    return vocab
```
![](BPE/1.png)

**2. 过滤掉vocab中的低频词**

```python
def truncate_vocab(vocab: Dict[str, int], mincount: int) -> None:
    tokens = list(vocab.keys())
    for token in tokens:
        if vocab[token] < mincount:
            del(vocab[token])
```
> 示例中设置为了1，不会过滤掉任何词。

**3. 数据预处理**

- 将训练语料中的每个单词按字符拆分，并在结尾加上特殊标记 `</w>` 表示单词结束。

```python
def prepare_bpe_vocab(vocab: Dict[str, int]) -> Dict[str, int]:
    bpe_vocab = {}
    # 遍历vocab中所有词
    for token in vocab:
        # 每个词的每个字符后都加上空格，同时末尾加上 </w> 表示单词结束
        ntoken = ' '.join(list(token)) + ' </w>'
        bpe_vocab[ntoken] = vocab[token]

    return bpe_vocab
```

![](BPE/2.png)

**4. 经历N次迭代，合并前N个最频繁的字符对**

```python
        # 一共合并merges个高频字符对后,才结束词汇表的构建
        for i in trange(merges, desc='Merging'):
            # 1. 获取每个相邻字符对的出现次数
            pairs = get_stats(vocab)
            # 2. 获取当前最高频的字符对
            best = max(pairs, key=pairs.get)
            # 3. 合并当前最高频的字符对
            vocab = merge_vocab(best, vocab)
```

**4.1 获取每个相邻字符对的出现次数**

```python
def get_stats(vocab: Dict[str, int]) -> Dict[Tuple[str, str], int]:
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        # 对经过预处理的vocab中的每个词按空格进行切分
        symbols = word.split()
        # 统计每个相邻字符对的出现次数
        for i in range(len(symbols)-1):
            pairs[symbols[i],symbols[i+1]] += freq

    return pairs
```

![首轮统计展示](BPE/3.png)

**4.2 获取当前最高频的字符对**

![](BPE/4.png)

**4.3 合并当前最高频的字符对**

```python
def merge_vocab(pair: Tuple[str, str], v_in: Dict[str, int]) -> Dict[str, int]:
    # 1. 将传入的最高频字符对中的两个字符用空格拼接起来，如: "H e"
    bigram = re.escape(' '.join(pair))
    v_out = {}
    # 2. 正则匹配含有“H e”的所有单词，并且“H”和“e”必须为两个独立的词，而不能为"HH e"或者"H ee"形式
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    # 3. 遍历vocab中所有词
    for word in v_in:
        # 3.1 用正则匹配并替换匹配上的 "H e" 为 “He”
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    # 4. 返回合并最高频字符对后的vocab
    return v_out
```
![](BPE/5.png)

**5.根据N轮迭代合并后的Vocab来构建最终的频次表(每个子词的出现次数)**

```python
def count_byte_freqs(vocab: Dict[str, int]) -> Dict[str, int]:
    freqs = defaultdict(int)
    for word in vocab:
        # 1. 按空格切分
        bytes_ = word.split(' ')
        # 2. 每个子词出现次数加1
        for byte in bytes_:
            freqs[byte] += 1
   # 3. 添加一些特殊词 
    for token in ['<line/>', '</line>', '<pad>', '<unk>']:
        freqs[token] += 1

    return freqs
```
![](BPE/6.png)

**6.根据频次表构建最终的词汇表**

```python
def create_vocab_maps(freqs: Dict[str, int]) -> (Dict[str, int], Dict[int, str]):
    # 1. 按照 词频从高到低 的顺序排序
    ordered_freqs = sorted(freqs.items(), key=lambda x: x[1], reverse=True)
    vocab_to_idx, idx_to_vocab = {}, {}
    for i in range(len(ordered_freqs)):
        # 2. 构建词汇表
        word, freq = ordered_freqs[i]
        vocab_to_idx[word] = i
        idx_to_vocab[i] = word

    return vocab_to_idx, idx_to_vocab
```
![](BPE/7.png)

> BPE 算法预训练过程完整代码如下

```python
    def train_bpe(filepaths: List[str], mincount: int, merges: int) -> 'BytePairtokenizer':
        vocab = create_vocab(filepaths)
        truncate_vocab(vocab, mincount)
        vocab = prepare_bpe_vocab(vocab)
        for i in trange(merges, desc='Merging'):
            pairs = get_stats(vocab)
            best = max(pairs, key=pairs.get)
            vocab = merge_vocab(best, vocab)

        freqs = count_byte_freqs(vocab)
        vocab_to_idx, idx_to_vocab = create_vocab_maps(freqs)
        return BytePairTokenizer(freqs, vocab_to_idx, idx_to_vocab)
```

## 分词过程

**1.对输入的文本进行断句加分词**

```python
    # 使用NLTK库提供的sent_tokenize方法进行分词
    lines = sent_tokenize(open(filepath, encoding='utf-8-sig').read())

    tokens = []
    # 遍历所有句子
    for line in lines:
        if len(line) > 1:
            tokens += get_line_ids(line, tokenizer)
```
```python
def get_line_ids(line: str, tokenizer: BytePairTokenizer) -> List[int]:
    # 对每个句子进行分词
    tokens = wordpunct_tokenize(line)
    # 将每个词从str转换为list列表形式，同时列表末尾追加</w>
    tokens = [list(t) + ['</w>'] for t in tokens]
    ...
```
**2. 对当前句子中每个词进行子词合并加词ID映射，最后得到当前句子对应的Token列表**

```python
def get_line_ids(line: str, tokenizer: BytePairTokenizer) -> List[int]:
    ...
    lineids = []
    for token in tokens:
        # 2.1 对每个词进行子词合并，直到无法合并为止
        token = tokenizer.merge_bytes(token)
        # 2.2 将当前词列表中每个子词映射为字典中对于的词ID
        ids = tokenizer.get_byte_ids(token)
        lineids += ids
    
    sol_id = tokenizer.get_byte_id('<line/>')
    eol_id = tokenizer.get_byte_id('</line>')
    lineids = [sol_id] + lineids + [eol_id]
    return lineids
```

**2.1 对每个词进行子词合并，直到无法合并为止**

```python
    # 对当前词的子词进行合并，直到无法合并为止
    def merge_bytes(self, bytes_: List[str]) -> List[str]:
        bytes_, merged = self.merge_max_pair(bytes_)
        while merged:
            bytes_, merged = self.merge_max_pair(bytes_)

        return bytes_ 


    def merge_max_pair(self, bytes_: List[str]) -> (List[str], bool):
        # 1. 取出出现次数最多的字符对
        max_pair = self.get_max_pair_idxs(bytes_)
        merged = True if max_pair is not None else False
        
        if merged:
            # 2. 合并该字符对
            bytes_ = bytes_[:max_pair[0]] + \
                    [''.join(bytes_[max_pair[0]:max_pair[1]+1])] + \
                    bytes_[max_pair[1]+1:]

        return bytes_, merged

    def get_max_pair_idxs(self, bytes_) -> Tuple[int, int]:
        pairs = {}
        # 1. 遍历所有相邻字符对的组合
        for i in range(1, len(bytes_)):
            pair = ''.join(bytes_[i-1:i+1])
            # 2. 判断没饿过字符对是否存在于频次表中，如果存在记录出现次数
            if pair in self.freqs:
                pairs[(i-1, i)] = self.freqs[pair]
        # 3. 取出出现次数最多的字符对
        return None if len(pairs) == 0 else max(pairs, key=pairs.get) 
```

**2.2 将当前词列表中每个子词映射为字典中对于的词ID**

```python
    def get_byte_ids(self, bytes_):
        ids = []
        for byte in bytes_:
            if byte in self.vocab_to_idx:
                ids.append(self.vocab_to_idx[byte])

            else:
                ids.append(self.vocab_to_idx[self.unk])

        return ids
```
