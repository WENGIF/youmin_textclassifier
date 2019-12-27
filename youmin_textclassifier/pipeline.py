# -*- coding: utf-8 -*-

import os
from json import dumps as json_format

from .config import opt
from .features import token_to_file, token_to_vec
from .metrics import YmPredictResult, YmTestResult
from .models import (AbstractNN, EnsembleNNBaseline, EnsembleNNStacking,
                     FasttextModel, GeneralModel)
from .preprocessing import NNData, file_to_text, text_to_token
from .utils.basetool import (data_distribution, data_feature_word,
                             pydict_to_objdict)
from .utils.log import get_logger


class YmTextClassifier:
    def __init__(self, name, model=None, **kwargs):
        """
        Args:
            name -- 模型名字，如"my_model"
        Kwargs:
            model -- 所选模型，如"lr"，默认使用逻辑回归:lr，模型导出唯一id为——"name_model.m[model.bin]"
        """
        self._general_models = ("lr", "svm", "rf", "mnb", "gnb",)
        self._ft_models = ("ft",)
        self._nn_models = ("textcnn", "textrnn",)
        self._ensemble_baseline = ("ensemble_nn_avg", "ensemble_nn_concat",)
        self._ensemble_stacking = ("ensemble_nn_stacking",)
        self._nn_models = self._nn_models \
                            + self._ensemble_baseline \
                            + self._ensemble_stacking
        self.name = name
        self.model = model if model else opt.model
        self.opt = opt  # 用户配置
        self.opt.parse(kwargs)
        self.logger = get_logger(name=opt.log_name,
                                 level=opt.log_level,
                                 log_path=opt.log_path)
        self._get_model_client()

    def _get_model_client(self):
        """
        获取模型客户端
        """
        uid = "{}_{}".format(self.name, self.model)
        if self.model in self._general_models:
            vectorizer = "{}_vectorizer.pkl".format(uid)
            self.vectorizer_path = os.path.join(self.opt.model_output,
                                                vectorizer)
            if self.model == "mnb" and self.opt.feature == "w2v":
                # 贝叶斯多项式--适合dtm格式化
                raise ValueError("Please format by `bow`!")
            self.model_path = os.path.join(self.opt.model_output,
                                           "{}.m".format(uid))
            model_params = self.opt.general_params.get(self.model, {})
            self.model_cli = GeneralModel(name=self.model,
                                          model_params=model_params)
        elif self.model in self._ft_models:
            self.model_path = os.path.join(self.opt.model_output,
                                           "{}.model".format(uid))
            self.model_cli = FasttextModel(model_params=self.opt.ft_params)
        elif self.model in self._nn_models:
            self.model_path = os.path.join(self.opt.model_output, uid)
            if self.model in self._ensemble_baseline:
                self.model_cli = EnsembleNNBaseline(name=self.model,
                                                    model_params=None)
            elif self.model in self._ensemble_stacking:
                conf = pydict_to_objdict(self.opt.nn_params)
                conf.common.model_path = self.model_path
                conf.common.model_name = self.model
                self.model_cli = EnsembleNNStacking(
                    name=self.model,
                    model_params=conf)
            else:
                self.model_cli = AbstractNN(name=self.model,
                                            model_params=None)
        else:
            raise ValueError(
                "`{}` is not a supported model".format(self.model))

    def _parse_nn_data(self, mode, input_x_token, input_y=None):
        """
        获取神经网络数据处理对象，将数据转换为神经网络的数据格式
        包括计算类别个数和词汇个数参数等
        """
        common_conf = self.opt.nn_params.get("common", {})
        training_conf = self.opt.nn_params.get("training", {})
        model_conf = self.opt.nn_params.get(self.model, {})
        self.nn_conf = pydict_to_objdict({
            "common": common_conf,
            "training": training_conf,
            "model": model_conf
        })
        self.nn_conf.common.model_path = self.model_path
        self.nn_conf.common.model_name = self.model
        self.nn_data = NNData(self.nn_conf)
        if mode != "train":
            self.nn_data.load_vocab()

        input_x = [_.split() for _ in input_x_token]
        if mode == "train":
            return self.nn_data.get_train_data(input_x, input_y)
        elif mode == "test":
            return self.nn_data.get_test_data(input_x, input_y)
        else:
            return self.nn_data.get_predict_data(input_x)

    def _format_data_to_model(self, data, mode="test"):
        """
        将文本数据转换为模型输入数据
        Args:
            data -- list数据、文本文件路径或文本文件夹路径，如[("text1","label1"), ("text2","label1")]
        Kwargs:
            mode -- 模型阶段，可选项: train/test/predict
        Returns:
            模型所需数据
              --训练测试阶段: 返回X([array(dim), array(dim)]或文件路径)和Y(["label1", "label2"])
              --预测阶段: 返回X
        """
        if isinstance(data, str):
            data = file_to_text(data,
                                sep=self.opt.sep,
                                file_format=self.opt.file_format)
        elif isinstance(data, list):
            pass
        else:
            raise TypeError("data type should be list or str")
        
        if mode in ("train", "test"):
            self.logger.debug("Data distribution:\n{}".format(
                data_distribution(data)))
            input_y, input_x = zip(*data)
            input_x_token = text_to_token(input_x,
                                          cut_type=self.opt.cut_type,
                                          user_dict=self.opt.user_dict,
                                          stop_word=self.opt.stop_word)
            if mode == "train" and self.opt.feature_word:  # 导出特征词汇字典
                data_feature_word(input_x_token, self.opt.feature_word)
            # ######## sklearn系列模型 ########
            if self.model in self._general_models:
                input_x = token_to_vec(input_x_token,
                                       feature=self.opt.feature,
                                       w2v_dict=self.opt.w2v_dict,
                                       w2v_dim=self.opt.w2v_dim,
                                       is_lower=self.opt.is_lower,
                                       vectorizer_path=self.vectorizer_path,
                                       mode=mode)
            # ######## fastText系列模型 ########
            elif self.model in self._ft_models:
                if mode == "train":
                    file_ = "{}_{}_{}.txt".format(self.name,
                                                       self.model,
                                                       mode)
                    file_path = os.path.join(self.opt.cache_home, file_)
                    label_token_texts = zip(input_y, input_x_token)
                    input_x = token_to_file(label_token_texts, file_path)
                else:
                    # 无法表征，默认采用空格代替
                    input_x = [_xt if _xt else " " for _xt in input_x_token]
            # ######## 神经网络系列模型 ########
            elif self.model in self._nn_models:
                input_x, input_y = self._parse_nn_data(mode, input_x, input_y)
                # 初始化神经网络结构
                self.nn_conf.common.vocab_size = self.nn_data.vocab_size
                self.nn_conf.common.num_class = self.nn_data.num_class
                self.nn_conf.common.word_embedding = self.nn_data.word_embedding
                if mode == "train":
                    if self.model in self._ensemble_baseline:
                        self.model_cli = EnsembleNNBaseline(
                            name=self.model,
                            model_params=self.nn_conf)
                    elif self.model in self._ensemble_stacking:
                        self.model_cli = EnsembleNNStacking(
                            name=self.model,
                            model_params=self.nn_conf)
                    else:
                        self.model_cli = AbstractNN(
                            name=self.model,
                            model_params=self.nn_conf)
            else:
                raise ValueError(
                    "`{}` is not a supported model".format(self.model))
            return input_x, input_y
        else:
            self.predict_data = list(
                map(lambda x : x[-1] if isinstance(x, list) else x, data))
            input_x_token = text_to_token(self.predict_data,
                                          cut_type=self.opt.cut_type,
                                          user_dict=self.opt.user_dict,
                                          stop_word=self.opt.stop_word)
            if self.model in self._general_models:
                input_x = token_to_vec(input_x_token,
                                       feature=self.opt.feature,
                                       w2v_dict=self.opt.w2v_dict,
                                       w2v_dim=self.opt.w2v_dim,
                                       is_lower=self.opt.is_lower,
                                       vectorizer_path=self.vectorizer_path,
                                       mode=mode)
            elif self.model in self._ft_models:
                # 无法表征，默认采用空格代替
                input_x = [_xt if _xt else " " for _xt in input_x_token]
            elif self.model in self._nn_models:
                input_x = self._parse_nn_data(mode, input_x_token)
            else:
                raise ValueError(
                    "`{}` is not a supported model".format(self.model))
            return input_x

    def train(self, data):
        """
        Args:
            data -- list数据、文本文件路径或文本文件夹路径，如[("text1","label1"), ("text2","label1")]
        Returns:
            clf_log -- 分类器参数字典
        """
        self.logger.debug("====== [Train] ======")
        input_x, input_y = self._format_data_to_model(data, mode="train")
        clf_log = self.model_cli.train(input_x,
                                       input_y,
                                       model_path=self.model_path)
        self.logger.debug("**** Model params:\n{}".format(
            json_format(clf_log,
                        indent=4,
                        sort_keys=True,
                        ensure_ascii=False)))
        self.logger.debug("**** Model path: {}".format(self.model_path))
        self.logger.debug("====== [END] ======")
        return clf_log

    def test(self, data):
        """
        Args:
            data -- list数据、文本文件路径或文本文件夹路径，如[("text1","label1"), ("text2","label1")]
        Returns:
            YmTestResult -- 测试结果类对象
        """
        self.logger.debug("====== [Test] ======")
        input_x, yt = self._format_data_to_model(data, mode="test")
        yt, yp = self.model_cli.test(input_x, yt, self.opt.min_proba)
        self.logger.debug("====== [END] ======")
        return YmTestResult(yt, yp)

    def load(self):
        """
        加载已训练好的模型
        """
        self.logger.debug("**** Loading `{}`...".format(self.model_path))
        self.model_cli.load(model_path=self.model_path)

    def predict(self, text):
        """
        Args:
            text -- list数据、文本文件路径或文本文件夹路径，如["text1", "text2"]
        Returns:
            YmPredictResult -- 预测结果类对象
        """
        self.logger.debug("====== [Predict] ======")
        if not isinstance(text, list)\
            and not isinstance(text, tuple)\
                and not os.path.exists(text):
            text = [text]
        input_x = self._format_data_to_model(text, mode="predict")
        predict_result = self.model_cli.predict(input_x)
        self.logger.debug("====== [END] ======")
        return YmPredictResult(self.predict_data, predict_result, opt.sep)
