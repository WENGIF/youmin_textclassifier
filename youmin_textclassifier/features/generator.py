# -*- coding: utf-8 -*-
""" 特征生成 """

import pickle

import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from ..utils.db import Word2VecDb


def _word2vec_bin(text_data,
                  w2v_dict,
                  w2v_dim,
                  is_lower):
    """
    文本词向量表征（二进制[bin]文件版）
    异常处理：当句子中所有词汇在词向量库中均不存在，用0向量代替
    Args:
        text_data -- 需要转换的用空格隔开的文本数据，["我 来自 广州",...]
        w2v_dict  -- 二进制词向量模型文件（w2v_dim维）
        w2v_dim   -- 词向量维度
        is_lower  -- 是否将词汇转换为小写
    returns:
        word2vec_list -- numpy.matrix
    """
    w2v_model = KeyedVectors.load_word2vec_format(w2v_dict,
                                                  binary=True,
                                                  unicode_errors="ignore")
    word2vec_list = []
    for each_sen in text_data:
        sum_array = np.zeros(w2v_dim)
        cnt = 0
        for _word in each_sen.split():
            _word = _word.lower() if is_lower else _word
            if _word in w2v_model:
                sum_array += np.array(w2v_model[_word])
                cnt += 1
        if cnt == 0:
            word2vec_list.append(np.zeros(w2v_dim))
        else:
            word2vec_list.append(sum_array / float(cnt))
    return np.matrix(word2vec_list)


def _word2vec_db(text_data,
                 w2v_dict,
                 w2v_dim,
                 is_lower):
    """
    文本词向量表征（文件SQLite版）
    异常处理：当句子中词汇在词向量库中均不存在，用0向量代替
    Args:
        text_data -- 需要转换的用空格隔开的文本数据，["我 来自 广州",...]
        w2v_dict  -- 词向量模型文件（w2v_dim维）
        w2v_dim   -- 词向量维度
        is_lower  -- 是否将词汇转换为小写
    returns:
        word2vec_list -- numpy.matrix
    """
    word2vec_list = []
    cli = Word2VecDb(db_path=w2v_dict)
    for each_sen in text_data:
        sum_array = cli.get_vec_batch(
            [_.lower() if is_lower else _ for _ in each_sen.split()])
        if sum_array is not None:
            word2vec_list.append(np.array(sum_array).mean(axis=0))
        else:
            word2vec_list.append(np.zeros(w2v_dim))
    cli.destroy()
    return np.matrix(word2vec_list)


def token_to_vec(token_texts,
                 feature,
                 w2v_dict,
                 w2v_dim,
                 is_lower,
                 vectorizer_path=None,
                 mode=None):
    """
    将文本表征为向量
    Args:
        token_texts -- 切词后空格隔开列表["w1 w2 ... w_n", ..., ]
        feature     -- optional: 文档-词项矩阵(dtm，基于bow), 词频-逆向文档频率(tf-idf), 词向量(w2v)
        w2v_dict    -- 词向量模型文件（w2v_dim维）
        w2v_dim     -- 词向量维度
        is_lower    -- 是否将词汇转换为小写
    Kwargs:
        vectorizer_path -- vectorizer 存储路径
        mode            -- 模型模式，optional: train/test/predict
    returns:
        pred_vec -- 文本特征向量，类型为: scipy.sparse.csr_matrix 或 numpy.matrix
    """
    if feature == "w2v":
        if w2v_dict:
            if w2v_dict.endswith("db"):
                pred_vec = _word2vec_db(token_texts, w2v_dict,
                                        w2v_dim, is_lower)
            elif w2v_dict.endswith("bin"):
                pred_vec = _word2vec_bin(token_texts, w2v_dict,
                                        w2v_dim, is_lower)
            else:
                # TODO: 兼容txt格式的词向量
                raise ValueError(
                    "`%s` is not a supported w2v_dict" % w2v_dict)
        else:
            raise ValueError("Please input the w2v_dict path!")
    elif feature in ("bow", "bow_l2", "tfidf"):
        if vectorizer_path is None or mode is None:
            raise ValueError("Please input the vectorizer_path or mode!")
        if mode == "train":
            # `token_pattern`默认过滤一个字符长度的词，在此设置保留
            if feature == "bow":
                vectorizer = CountVectorizer(token_pattern=r"(?u)\b\w+\b")
            elif feature == "bow_l2":
                # 等价: CountVectorizer + normalize("l2")
                vectorizer = TfidfVectorizer(use_idf=False,
                                             token_pattern=r"(?u)\b\w+\b")
            else:
                vectorizer = TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")
            pred_vec = vectorizer.fit_transform(token_texts)
            with open(vectorizer_path, "wb") as fw:
                pickle.dump(vectorizer, fw)
        else:
            vectorizer = pickle.load(open(vectorizer_path, "rb"))
            pred_vec = vectorizer.transform(token_texts)
    else:
        raise ValueError("`%s` is not a supported feature" % feature)
    return pred_vec


def token_to_file(label_token_texts, outpath, sep="__label__"):
    """
    将列表转为文件，用于规整fasttext所需输入格式
    Args:
        label_token_texts -- 训练数据，如[(label, "token1 token2"),...]
        outpath           -- 导出文件路径
    Kwargs:
        sep -- 分割符
    returns:
        outpath -- 数据行格式如: `__label__<y> <text>`
    """
    try:
        with open(outpath, "w", encoding="utf-8") as fw:
            for _label, _text in label_token_texts:
                fw.write("{}{} {}\n".format(sep, _label, _text))
    except FileNotFoundError:
        raise FileNotFoundError("Can't write the file(%s)." % outpath)
    return outpath
