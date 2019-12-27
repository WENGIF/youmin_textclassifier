# -*- coding: utf-8 -*-

import os

from numpy import dstack
from tensorflow.keras.layers import Add, Average, Dense, Lambda, concatenate

from .general import GeneralModel
from .nn import AbstractNN, AttentionLayer, Model, load_model


class EnsembleNNBaseline(AbstractNN):
    def __init__(self, name, model_params=None):
        """
        Args:
            name -- 模型名称
        Kwargs:
            model_params -- 模型配置字典
        """
        self.config = model_params if model_params else {}
        self._name = name
        self.models = {
            "ensemble_nn_avg": NNAvgModel,
            "ensemble_nn_concat": NNConcatModel,
        }

    def train(self, input_x, input_y, model_path):
        """
        Args:
            input_x -- np.array([[11,3],[1,10]])
            input_y -- np.array([[0,0,0,1],[1,0,0,0]])
            model_path -- 模型路径
        Returns:
            config -- 模型日志
        """
        self.clf = self.models[self._name](config=self.config).model
        # 转换为多通道输入
        if hasattr(self.clf, "input") and isinstance(self.clf.input, list):
            input_x = [input_x] * len(self.clf.input)
        super().__init__(name=self._name, model_params=self.config)
        model_log = super().train(input_x=input_x,
                                  input_y=input_y,
                                  model_path=model_path,
                                  clf=self.clf)
        return model_log

    def load(self, model_path):
        super().__init__(name=self._name, model_params=self.config)
        super().load(model_path)

    def test(self, input_x, input_y, min_proba=0):
        if hasattr(self.clf, "input") and isinstance(self.clf.input, list):
            input_x = [input_x] * len(self.clf.input)
        test_result = super().test(input_x=input_x,
                                   input_y=input_y,
                                   min_proba=min_proba)
        return test_result

    def predict(self, input_x):
        if hasattr(self.clf, "input") and isinstance(self.clf.input, list):
            input_x = [input_x] * len(self.clf.input)
        predict_result = super().predict(input_x=input_x)
        return predict_result


def load_all_models(base_model_dir, ensemble_model_name, train_layer=False):
    """
    加载keras模型
    Args:
        base_model_dir -- stage1 基础模型所在路径
        ensemble_model_name -- 集成后的模型名字，防止加载集成后的模型
    Kwargs:
        train_layer -- [预留参数] 是否冻结stage1模型层
    Returns:
        models -- 模型对象
    """
    models = []
    if os.path.isdir(base_model_dir):
        for i, filename in enumerate(os.listdir(base_model_dir)):
            if not os.path.isdir(filename) \
                and filename.endswith(".h5") \
                    and not filename.startswith(ensemble_model_name):
                model = load_model(
                    base_model_dir + "/" + filename,
                    custom_objects={"AttentionLayer": AttentionLayer})
                model._name = filename[:-3]
                for layer in model.layers:
                    layer._trainable = train_layer
                    layer._name = "ensemble_{}_{}".format(i+1, layer.name)
                models.append(model)
                print(">> Loaded %s" % filename)
    else:
        raise ValueError("Please input `%s` as dir!" % base_model_dir)
    return models


class NNAvgModel:
    """（加权）平均集成（同理投票）"""
    def __init__(self, config):
        self.config = config
        self.model_name = self.config.common.model_name
        model_dir = self.config.common.model_path
        weights = self.config.model.weights  # 模型权重，默认为均值
        models = load_all_models(model_dir, self.model_name)
        self.model_num = len(models)

        input_x = [model.input for model in models]

        if weights:
            if not isinstance(weights, list) \
                and not isinstance(weights, tuple):
                raise TypeError("The `weights` must be `list` or `tuple`")
            elif len(weights) != self.model_num:
                raise ValueError(
                    "The length of `weights` is %s" % self.model_num)
        else:
            weights = [1 / self.model_num] * self.model_num

        def set_weight(last_layer_output, weight):
            return weight * last_layer_output

        outputs = [
            Lambda(set_weight, arguments={"weight": _w})(_m.output) \
            for _w, _m in zip(weights, models)]
        output_y = Add()(outputs)

        model_ens = Model(inputs=input_x,
                          outputs=output_y,
                          name="ensemble_avg")
        self.model = model_ens


class NNConcatModel:
    """ 直接拼接集成 """
    def __init__(self, config):
        self.config = config
        self.model_name = self.config.common.model_name
        model_dir = self.config.common.model_path
        self.output_num = self.config.model.output_num
        models = load_all_models(model_dir, self.model_name)
        self.model_num = len(models)

        inputs_x = [model.input for model in models]
        ensemble_outputs = [model.output for model in models]
        merge = concatenate(ensemble_outputs)
        output = Dense(self.output_num, activation="softmax")(merge)
        model_ens = Model(inputs=inputs_x,
                          outputs=output,
                          name="ensemble_concat")
        self.model = model_ens


class EnsembleNNStacking(GeneralModel):
    """ Stacking线性集成 
    训练数据: 为防止过拟合，最好选择与基模型不同的训练集
    """
    def __init__(self, name, model_params=None):
        """
        Args:
            name -- 模型名称
        Kwargs:
            model_params -- 模型配置字典
        """
        self.config = model_params if model_params else {}
        self._name = name
        model_dir = self.config.common.model_path
        self.models = load_all_models(model_dir, self._name)
        self.model_num = len(self.models)

    def get_stacking_x(self, input_x):
        """ 将堆叠第一阶段模型输出作为第二阶段的模型输入
        Args:
            input_x -- 需要预测特征数据
        Returns:
            stacking_x -- stage2 模型输入
        """
        stacking_x = None
        for model in self.models:
            yhat = model.predict(input_x)
            if stacking_x is None:
                stacking_x = yhat
            else:
                # 模型预测列拼接起来，[rows, len(models), probabilities]
                stacking_x = dstack((stacking_x, yhat))
        # 展开为: [rows, len(models) * probabilities]
        stacking_x = stacking_x.reshape(
            (stacking_x.shape[0],
             stacking_x.shape[1]*stacking_x.shape[2]))
        print("get stacking data successful~")
        return stacking_x

    def train(self, input_x, input_y, model_path):
        """ stage 2 模型训练
        Args:
            input_x/input_y -- 通常采用验证集数据
        """
        stacking_x = self.get_stacking_x(input_x)
        input_y = input_y.argmax(axis=1)

        stage2_model = self.config.model.stage2_model
        stage2_model_params = self.config.model.stage2_model_params
        model_save_path = os.path.join(model_path, "%s.m" % (self._name))
        super().__init__(name=stage2_model, model_params=stage2_model_params)
        model_log = super().train(input_x=stacking_x,
                                  input_y=input_y,
                                  model_path=model_save_path)
        return model_log

    def load(self, model_path):
        model_save_path = os.path.join(model_path, "%s.m" % (self._name))
        super().load(model_save_path)

    def test(self, input_x, input_y, min_proba=0):
        stacking_x = self.get_stacking_x(input_x)
        input_y = input_y.argmax(axis=1)
        test_result = super().test(input_x=stacking_x,
                                   input_y=input_y,
                                   min_proba=min_proba)
        return test_result

    def predict(self, input_x):
        stacking_x = self.get_stacking_x(input_x)
        predict_result = super().predict(input_x=stacking_x)
        return predict_result
