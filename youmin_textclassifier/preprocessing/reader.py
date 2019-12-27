# -*- coding: utf-8 -*-

import os


def file_to_text(data_path, sep, file_format):
    """
    读取所需数据，支持文件和文件夹格式
    Args:
        data_path   -- 数据文件（夹）路径
        sep         -- 风格符
        file_format -- 文件格式，如"txt"
    Returns:
        data -- 如[(label, text),...]
    """
    def load_file(file_path):
        """
        读取每个文件
        """
        with open(file_path, "r") as fr:
            data = [_line.rstrip("\n").split(sep, 1) for _line in fr]
        return data

    if os.path.isdir(data_path):
        files = [f for f in os.listdir(data_path) if not os.path.isdir(f) \
                    and data_path.endswith(".%s" % file_format)]
        data = []
        for f in files:
            data_e = load_file(os.path.join(data_path, f))
            data.extend(data_e)
        return data
    elif data_path.endswith(".%s" % file_format):
        return load_file(data_path)
    else:
        raise TypeError("File type is not `{}`!".format(file_format))