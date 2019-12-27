# -*- coding:utf-8 -*- 

import json
import math
import os
import pickle
from collections import Counter
from datetime import datetime

import numpy as np
from gensim.models import KeyedVectors

UNK_CHAR = "<UNK>"
PAD_CHAR = "<PAD>"


class NNData(object):
    """
    将文本数据转换为适合神经网络的数据格式
    """
    def __init__(self, config=None):
        self.config = config
        if self.config is not None:
            self.vocab_dir = os.path.join("/".join(self.config.common.model_path.split("/")[:2]), "vocab")
            if not os.path.exists(self.vocab_dir):
                os.mkdir(self.vocab_dir)
            self.word_to_index_path = os.path.join(
                self.vocab_dir, "word_to_index.json")
            self.index_to_word_path = os.path.join(
                self.vocab_dir, "index_to_word.json")
            self.index_to_label_path = os.path.join(
                self.vocab_dir, "index_to_label.json")
            self.label_to_index_path = os.path.join(
                self.vocab_dir, "label_to_index.json")
            self.w2v_lookup_path = os.path.join(
                self.vocab_dir, "w2v_lookup.pkl")
        self._word_to_index = {}
        self._index_to_word = {}
        self._index_to_label = {}
        self._label_to_index = {}
        self.word_embedding = None

    def _save_file(self, data, out_path):
        """
        将文件保存为json或者pickle
        """
        if out_path.endswith("json"):
            with open(out_path, "w", encoding="utf-8") as fw:
                json.dump(data, fw)
        else:
            with open(out_path, "wb") as fw:
                pickle.dump(data, fw)

    def _load_file(self, in_path):
        """
        加载json或者pickle文件
        """
        if in_path.endswith("json"):
            with open(in_path, "r", encoding="utf-8") as fr:
                return json.load(fr)
        else:
            with open(in_path, "rb") as fr:
                return pickle.load(fr)

    def _get_word_embedding(self, words):
        """
        获取词表和对应的词向量矩阵
        Args:
            words -- 过滤后全量单词列表
        Returns:
            vocab(list) --词汇列表: 相对words多出补全项<PAD>和未知项<UNK>，少了不存在词向量的单词
            word_embedding(ndarray or None) -- 词表词汇对应的向量
        """
        vocab = []
        vocab.extend([PAD_CHAR, UNK_CHAR])

        st = datetime.now().isoformat()
        pre_w2v = self.config.common.w2v_dict
        pre_w2v_dim = self.config.common.w2v_dim
        if pre_w2v:
            word_embedding = []
            pad_init_vec = np.zeros(pre_w2v_dim)
            unk_init_vec = np.random.randn(pre_w2v_dim)
            word_embedding.append(pad_init_vec)
            word_embedding.append(unk_init_vec)
            print("%s: Start load embedding from `%s`." % (st, pre_w2v))
            if pre_w2v.endswith(".bin"):
                word_vec = KeyedVectors.load_word2vec_format(
                    pre_w2v,
                    binary=True,
                    unicode_errors="ignore")
                for word in words:
                    try:
                        vector = word_vec.wv[word]
                        vocab.append(word)
                        word_embedding.append(vector)
                    except:
                        print("`%s`: not exist in the word embedding." % word)
            elif pre_w2v.endswith(".txt"):  # 只获取训练集的词汇向量
                words_set = set(words)
                with open(pre_w2v, "r") as fr:
                    each_line_len = pre_w2v_dim + 1
                    for row, line in enumerate(fr):
                        if row % 1000000 == 0:
                            print("-->> w2v line-%s" % row)
                        word_info = line.strip().split()
                        if len(word_info) == each_line_len:
                            _word, vector = word_info[0], word_info[1:]
                            if _word in words_set:
                                vocab.append(_word)
                                word_embedding.append(
                                    np.array(vector, dtype=np.float))
            else:
                # TODO: 兼容db格式
                w2v_f = pre_w2v.split(".")[-1]
                raise ValueError("`%s` is not a supported w2v_format" % w2v_f)
            et = datetime.now().isoformat()
            print("%s: End load embedding." % et)
            return vocab, np.asarray(word_embedding, dtype="float32")
        else:
            vocab.extend(words)
            word_embedding = None
            return vocab, word_embedding

    def _gen_sequence_vocab(self, sequences):
        """
        生成`词向量`、`词汇-索引`和`索引-词汇`映射字典
        Args:
            sequences -- 文本序列列表，如[["我", "来自", "中国"],]
        """
        all_words = [word for sequence in sequences for word in sequence]

        word_count = Counter(all_words)
        sort_word_count = sorted(word_count.items(), key=lambda x: -x[1])
        words = [item[0] for item in sort_word_count
                    if item[1] >= self.config.common.min_freq]

        vocab, word_embedding = self._get_word_embedding(words)
        self.word_embedding = word_embedding
        self.vocab_size = len(vocab)

        print("The number of vocab is %s." % self.vocab_size)
        self._word_to_index = dict(zip(vocab, range(self.vocab_size)))
        self._index_to_word = dict(zip(range(self.vocab_size), vocab))
        
        self._save_file(self._word_to_index, self.word_to_index_path)
        self._save_file(self._index_to_word, self.index_to_word_path)
        self._save_file(self.word_embedding, self.w2v_lookup_path)

    def _gen_label_vocab(self, labels):
        """
        生成`标签-索引`和`索引-标签`映射字典
        Args:
            labels -- 标签列表，["label1", "label2",]
        """
        labels = list(set(labels))
        self._label_to_index = dict(zip(labels, range(len(labels))))
        self._index_to_label = dict(zip(range(len(labels)), labels))
        self._save_file(self._label_to_index, self.label_to_index_path)
        self._save_file(self._index_to_label, self.index_to_label_path)
        self.num_class = len(labels)

    def _process_sequences(self, sequences):
        """
        原始文本转换为词库下标索引序列
        (1) 文本序列截断
        (2) 将每个文本序列用词表中词汇索引表示
        """
        sequences_vec = []
        seq_max_len = self.config.common.sequence_length
        word_to_index = self._word_to_index
        for sequence in sequences:
            sequence_vec = np.zeros(seq_max_len)  # 序列截断

            seq_len = len(sequence)
            seq_len = seq_len if seq_len < seq_max_len else seq_max_len
            
            for i in range(seq_len):
                if sequence[i] in word_to_index:
                    sequence_vec[i] = word_to_index[sequence[i]]
                else:
                    sequence_vec[i] = word_to_index[UNK_CHAR]
            sequences_vec.append(sequence_vec)
        return np.asarray(sequences_vec, dtype="int64")

    def _process_labels(self, labels_list):
        """
        原始标签集转换为one-hot向量
        Args:
            labels_list -- 类别列表，如[["类别1", "类别3", "类别5",],]
        Returns:
            labels_vec -- 类别one-hot编码，如[[1, 0, 1, 0, 1, 0,],]
        """
        labels_vec = []
        for labels in labels_list:
            label_vec = [0] * self.num_class
            for label in labels:
                label_vec[self._label_to_index[label]] = 1
            labels_vec.append(label_vec)
        return np.asarray(labels_vec, dtype="int64")

    def _vectorize_data(self, sequences, labels=None):
        """
        数据向量化
        Args:
            sequences -- 文本序列列表，如[["我", "来自", "中国"],]
            labels    -- ["label1",]
        Returns:
            sequences_vec -- 向量化的文本序列矩阵
            labels_vec    -- 向量化的标签序列矩阵
        """
        sequences_vec = self._process_sequences(sequences)
        if labels is not None:
            labels_list = [[label] for label in labels]  # 预留兼容多标签
            labels_vec = self._process_labels(labels_list)
            return sequences_vec, labels_vec
        else:
            return sequences_vec

    def _split_data(self, sequences_vec, labels_vec, shuffle=True):
        """
        切分为训练集和测试集
        """
        if shuffle:
            perm = np.arange(len(sequences_vec))
            np.random.shuffle(perm)
            sequences_vec = sequences_vec[perm]
            labels_vec = labels_vec[perm]
        train_idx = int(len(sequences_vec) * \
            (1-self.config.common.val_size))
        if train_idx == len(sequences_vec):
            return sequences_vec, labels_vec, None, None
        else:
            train_sequences = sequences_vec[:train_idx]
            train_labels = labels_vec[:train_idx]
            val_sequences = sequences_vec[train_idx:]
            val_labels = labels_vec[train_idx:]
            return train_sequences, train_labels, val_sequences, val_labels

    def load_vocab(self):
        """
        加载词典数据
        """
        self._word_to_index = self._load_file(self.word_to_index_path)
        self._index_to_word = self._load_file(self.index_to_word_path)
        self._label_to_index = self._load_file(self.label_to_index_path)
        self._index_to_label = self._load_file(self.index_to_label_path)
        self.word_embedding = self._load_file(self.w2v_lookup_path)
        self.vocab_size = len(self._word_to_index)
        self.num_class = len(self._label_to_index)
        print("Complete data initialization!")

    def get_train_data(self, sequences, labels):
        """
        获取训练数据
        Args:
            sequences -- 原始文本分词后的列表，如[["sequence1_token1", "sequence1_token2",],]
            labels    -- 标签列表，["label1", "label2",]
        Returns:
            train_sequences(np.array) -- 训练X矩阵
            train_labels(np.array)    -- 训练Y矩阵
            val_sequences(np.array)   -- 验证X矩阵
            val_labels(np.array)      -- 验证Y矩阵
        """
        self._gen_sequence_vocab(sequences)
        self._gen_label_vocab(labels)
        return self._vectorize_data(sequences, labels=labels)

    def get_test_data(self, sequences, labels):
        """
        获取测试数据
        """
        return self._vectorize_data(sequences, labels=labels)

    def get_predict_data(self, sequences):
        """
        获取预测数据
        """
        return self._vectorize_data(sequences)

    def load_embedding(self):
        """
        加载词向量
        """
        return self._load_file(self.w2v_lookup_path)
