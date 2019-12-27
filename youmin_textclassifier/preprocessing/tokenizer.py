# -*- coding: utf-8 -*-
""" 中文分词 """

import jieba
from jieba.analyse import extract_tags


def get_stop_words(stop_word):
    """
    过滤词汇
    Args:
        stop_word -- 停用词文件路径
    """
    if stop_word:
        with open(stop_word, "r") as fr:
            stop_list = [_line.rstrip("\n") for _line in fr]
        glb_stop_words = set(stop_list)
    else:
        glb_stop_words = set()
    return glb_stop_words


def text_to_token(text_list,
                  cut_type,
                  user_dict,
                  stop_word=None,
                  top_k=10):
    """
    切词工具
    Args:
        text_list -- 待处理文本列表，如["xxx",...]
        cut_type  -- optional: user(用户自定义), jieba, jieba_extract
        user_dict -- 用户自定义切词字典路径
        stop_word -- 用户自定义停用词路径
    Kwargs:
    returns:
        cut_data -- 切词之后用空格隔开，["x xx xxx",...]
    """
    stop_words = get_stop_words(stop_word)
    if user_dict:
        jieba.load_userdict(user_dict)
    if cut_type == "user":
        cut_data = [(" ".join([k for k in i.split()
                                 if k not in stop_words]))
                    for i in text_list]
    elif cut_type == "jieba":
        cut_data = [" ".join([k for k in jieba.cut(i)
                                if k not in stop_words])
                    for i in text_list]
    elif cut_type == "jieba_extract":
        cut_data = [(" ".join([k for k in extract_tags(i, topK=top_k)
                                 if k not in stop_words]))
                    for i in text_list]
    else:
        raise ValueError("Only supported cut_type: user/jieba/jieba_extract!")
    return cut_data
