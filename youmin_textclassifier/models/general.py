# -*- coding: utf-8 -*-
""" 常规的算法，如lr、bayes等 """

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC


class GeneralModel:
    def __init__(self, name="lr", model_params=None):
        self.model_params = model_params if model_params else {}
        self._name = name
        self.models = {
            "mnb": MultinomialNB, 
            "gnb": GaussianNB, 
            "svm": SVC,
            "rf": RandomForestClassifier,
            "lr": LogisticRegression,
        }

    def train(self, input_x, input_y, model_path):
        """
        Args:
            input_x    -- scipy.sparse.csr_matrix 或 numpy.matrix，如[array(100), array(100), ]
            input_y    -- 统一预留参数，["label1", "label2", ...]
            model_path -- 模型文件导出路径
        Returns:
            clf.get_params() -- 分类器训练参数字典
        """
        self.clf = self.models[self._name](**self.model_params)
        self.clf.fit(input_x, input_y)
        joblib.dump(self.clf, model_path)
        return self.clf.get_params()

    def test(self, input_x, input_y, min_proba=0):
        """
        Args:
            input_x, input_y -- 同self.train
        Kwargs:
            min_proba -- 测试概率阈值
        Returns:
            yt -- 真实标签列表，["label1", "label2",]
            yp -- 预测标签列表，["label1", "label2",]
        """
        y_proba = self.clf.predict_proba(input_x)
        test_result = []
        for _r, _yp in enumerate(y_proba):
            maxv_ix = np.argmax(_yp)
            if _yp[maxv_ix] >= min_proba:
                test_result.append((input_y[_r], self.clf.classes_[maxv_ix]))
        return zip(*test_result)

    def load(self, model_path):
        self.clf = joblib.load(model_path)

    def predict(self, input_x):
        """
        Args:
            input_x -- 同self.train
        Returns:
            predict_result -- [(label, proba),]
        """
        y_val_proba = self.clf.predict_proba(input_x)
        maxv_ixs = np.argmax(y_val_proba, axis=1)
        predict_result = list(zip(self.clf.classes_[maxv_ixs],
                                  np.max(y_val_proba, axis=1)))
        return predict_result
