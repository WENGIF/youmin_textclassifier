# -*- coding: utf-8 -*-

import fasttext


class FasttextModel:
    def __init__(self, model_params=None):
        self.model_params = model_params if model_params else {}

    def train(self, input_x, input_y, model_path):
        """
        Args:
            input_x    -- 训练文件路径
            input_y    -- 统一预留参数，["label1", "label2", ...]
            model_path -- 模型文件导出路径
        Returns:
            classifier_log -- 分类器训练参数字典
        """
        self.classifier = fasttext.supervised(input_x,
                                              model_path,
                                              **self.model_params)
        classifier_log = {
            # "_model": classifier
            # "labels": classifier.labels,
            "dim": self.classifier.dim,
            "ws": self.classifier.ws,
            "epoch": self.classifier.epoch,
            "neg": self.classifier.neg,
            "bucket": self.classifier.bucket,
            "minn": self.classifier.minn,
            "maxn": self.classifier.maxn,
            "t": self.classifier.t,
            "label_prefix": self.classifier.label_prefix,
            "encoding": self.classifier.encoding
        }
        return classifier_log

    def test(self, input_x, input_y, min_proba=0):
        """
        Args:
            input_x -- list, ["token1 token2", ...]
            input_y -- list, ["label1", "label2", ...]
        Kwargs:
            min_proba -- 测试概率阈值
        Returns:
            yt -- 真实标签列表，["label1", "label2",]
            yp -- 预测标签列表，["label1", "label2",]
        """
        pred = self.classifier.predict_proba(input_x, k=1)  # [[(lable1, proba1), (lable2, proba2)],]
        yt, yp = [], []
        for _yt, _yp_info in zip(input_y, pred):
            _yp, _proba = _yp_info[0]
            if _proba >= min_proba:
                yt.append(_yt)
                yp.append(_yp)
        return yt, yp

    def load(self, model_path):
        self.classifier = fasttext.load_model(
            model_path + ".bin",
            label_prefix=self.model_params["label_prefix"])

    def predict(self, input_x):
        """
        Args:
            input_x -- list, ["token1 token2", ...]
            input_y -- list, ["label1", "label2", ...]
        Returns:
            predict_result -- [(label, proba),]
        """
        pred = self.classifier.predict_proba(input_x, k=1)
        predict_result = [_label[0] for _label in pred]
        return predict_result
