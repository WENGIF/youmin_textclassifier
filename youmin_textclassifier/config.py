# -*- coding: utf-8 -*-


class Config(object):
    """
    根据需求获取所需参数
    """
    # ----------- 数据输入 --------------
    sep = "\t"           # 模型文本与标签的分隔符
    file_format = "txt"  # 模型文本格式

    # ----------- 缓存文件路径 --------------
    cache_home = "./data/cache/"  # 缓存fastText中间数据等
    log_name = "pipeline"         # `pipeline.py`的日志名称
    log_path = None               # 日志路径，可选项: None表示输出日志到控制台；文件路径则输出日志到文件
    log_level = "debug"           # 日志级别，可选项: error/warning/info/debug

    # ----------- 数据处理 --------------
    ############## 切词
    cut_type = "jieba"  # 切词方法，可选项: user(表示用户自己切好词汇并采用空格分隔), jieba, jieba_extract
    user_dict = None    # 用户分词自定义词汇地址，可选项: None表示不加载，文件路径表示该文件为自定义词汇（行格式: “词汇 词频 词性”）
    stop_word = None    # 用户分类自定义停用词地址，可选项: None表示不加载，文件路径表示该文件为自定义停用词（行格式: “词汇”）
    feature_word = None # 文本特征词汇统计导出路径，可选项: None表示不加载，文件路径表示该文件为特征词汇输出路径（行格式: “词汇\t频次”）

    # ----------- 表征配置 --------------
    ##############
    feature = "w2v"     # 文本表征，可选项: tfidf/bow(词袋)/bow_l2(词袋+l2正则化)
    w2v_dim = 200       # 词向量维度
    is_lower = True     # 是否将词汇转为小写，如腾讯词向量是将词汇统一为小写，且英文中间没有空格，如: iPad Pro ==>> ipadpro
    w2v_dict = None     # 词向量，若为文件，则文件可选项: bin(二进制词向量)
                        #                            txt(文件格式"词汇 维度1值 维度2值 ... 维度n值"): 只适用神经网络模型
                        #                            db(sqlite格式): 只使用非神经网络模型
    # ----------- 模型配置 -----------
    ############## 存储
    model = "lr"             # 模型名称，如lr: 逻辑回归; fr: fastText
    model_output = "./data"  # 模型储存路径
    ############## 评估
    min_proba = 0.0          # 测试置信度阈值，如设为0.4，则只会获取类别最大概率>=该值的样本作为测试样本
    ############## 参数
    general_params = {       # sklearn系列模型参数配置，若为空则表示采用默认参数
        "lr": {
            "class_weight": "balanced",
            "n_jobs": -1,
        },
        "svm": {
            "probability": True,
        },
    }
    ft_params = {            # fastText模型参数配置
        "label_prefix": "__label__",
        "epoch": 15,
        "thread": 4,
    }
    nn_params = {
        "common": {                 # #### 神经网络公用参数
            "min_freq": 5,          # 词汇最小词频
            "sequence_length": 14,  # 序列最大长度，经验法则: (1)文本长度均值(+-2); (2)文本长度3/4分位数; (3)文本长度最大值
            "val_size": 0,          # 验证集比例
            "w2v_dict": w2v_dict if w2v_dict else "",  # 词向量，同上(保证命令行训练可重置)
            "w2v_dim": w2v_dim,                        # 词向量维度
            "w2v_trainable": False, # 词向量是否参与训练
            "summaries": True,      # 是否产生TensorBoard
            "early_stopping": True, # 是否根据迭代情况提前终止训练
        },
        "training": {               # #### 神经网络训练参数
            "epoches": 5,
            "batch_size": 128,
            "evaluate_every": 100,
            "checkpoint_every": 100,
            "learning_rate": 0.001,
        },
        "textcnn":{
            "num_filters": 128,
            "filter_sizes": [2, 3, 4, 5],  # 原论文只用到[3,4,5]
            "dropout_keep_prob": 0.5,
            "l2_reg_lambda": 0.0,          # 原论文为3
            "fnn_num": 1,
        },
        "textrnn": {
            "core_layer": "gru",       # lstm/bi-lstm/gru/bi-gru
            "core_layer_units": 256,
            "dropout": 0.2,
            "recurrent_dropout": 0.1,
            "add_attention": False,    # 是否添加attention层
        },
        "ensemble_nn_avg": {
            "train": False,            # 融合后是否二次训练
            "weights": None,           # 融合模型权重，可选项: None表示采用均值权重代替; 非None表示各个模型的权重，其长度需和融合模型一致，如双模型可设置为[1/2, 1/2]
        },
        "ensemble_nn_concat": {
            "train": True,
            "output_num": 18,   # 分类类别总数
        },
        "ensemble_nn_stacking": {
            "train": True,
            "kflod": 3,
            "stage1_models": ["textcnn", "textrnn"],
            "stage2_model": "lr",
            "stage2_model_params": {},
        },
    }


def parse(self, kwargs):
    """
    根据字典kwargs 更新 config参数
    """
    def _update_dict(dic, usr_dic):
        """
        解析共享参数
        """
        for k, v in dic.items():
            if isinstance(v, dict):
                _update_dict(v, usr_dic)
            else:
                if k in usr_dic:
                    dic[k] = usr_dic[k]
        return dic

    self.print_conf = kwargs.get("print_conf", False)
    share_kv = {}
    for k, v in kwargs.items():
        if hasattr(self, k) and v:
            print(k, v)
            setattr(self, k, v)
            if k in set(["w2v_dict", "w2v_dim"]):
                share_kv[k] = v

    if share_kv:
        self.nn_params = _update_dict(self.nn_params, share_kv)

    if self.print_conf:
        print("user config:")
        print("-" * 20)
        for k in dir(self):
            if not k.startswith("_") \
                and k != "parse" and k != "state_dict":
                print("{} = {}".format(k, getattr(self, k)))
        print("-" * 20)
    return self


def state_dict(self):
    """
    获取配置字典
    """
    return  {k: getattr(self, k)
                for k in dir(self)
                    if not k.startswith("_")
                        and k!= "parse"
                        and k!= "state_dict" }


Config.parse = parse
Config.state_dict = state_dict
opt = Config()
