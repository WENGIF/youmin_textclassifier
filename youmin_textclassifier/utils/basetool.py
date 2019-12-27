# -*- coding: utf-8 -*-

import os
from collections import Counter

from prettytable import PrettyTable


def data_distribution(data):
    """
    展示样本分布
    Args:
        data -- 样本: [[label, text], ] OR [(label, text), ] OR [label, ]
    Returns:
        x    -- 样本分布统计报告
    """
    if not isinstance(data, list):
        raise TypeError("Data is not a supported type")
    if isinstance(data[0], list) or isinstance(data[0], tuple):
        data_count = Counter([i[0] for i in data])
    elif isinstance(data[0], str):
        data_count = Counter(data)
    else:
        raise TypeError("Data is not a supported type")
    data_cnt_sort = sorted(data_count.items(),
                           key=lambda k: k[1],
                           reverse=True)
    x = PrettyTable(["类别", "样本数", "占比"])
    x.align["类别"] = "l"
    all_cnt = len(data)
    for i, j in data_cnt_sort:
        x.add_row([i, j, round(j / float(all_cnt) * 100, 2)])
    x.add_row(["总计", all_cnt, "100%"])
    return x


def data_feature_word(x_tokens, out_path, level="word"):
    """
    词(字)频统计
    Args:
        x_tokens -- 格式: `["token1 token2",]`
        out_path -- 导出路径
    Kwargs:
        level -- 统计级别
    """
    if level not in ("word", "char"):
        raise ValueError("`{}` is not a supported level".format(level))
    fs = []
    for xt in x_tokens:
        fs.extend(xt.split())
    fs_cnt = Counter(fs)
    fs_cnt_sort = sorted(fs_cnt.items(),
                         key=lambda k: k[1],
                         reverse=True)
    if os.path.exists(out_path):
        with open(out_path, "w") as fw:
            for fea, cnt in fs_cnt_sort:
                fw.write("{}\t{}\n".format(fea, cnt))
    else:
        raise FileExistsError("File {} not exists!".format(out_path))


class ObjectDict(dict):
    """
    像访问属性一样访问字典对象
    """
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def pydict_to_objdict(pydict):
    """
    将py内置字典转换为类对象字典
    """
    obj_dict = ObjectDict()
    for k, v in pydict.items():
        if isinstance(v, dict):
            obj_dict[k] = pydict_to_objdict(v)
        else:
            obj_dict[k] = v
    return obj_dict


def objdict_to_pydict(objdict):
    """
    将类对象字典转换为py内置字典
    """
    obj_dict = {}
    for k, v in objdict.items():
        if isinstance(v, dict):
            obj_dict[k] = objdict_to_pydict(v)
        else:
            obj_dict[k] = v
    return obj_dict