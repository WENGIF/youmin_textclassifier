# -*- coding:utf-8 -*- 

import json
import os

import numpy as np
from tensorflow import TensorShape
from tensorflow.keras import Input, Model, Sequential
from tensorflow.keras import backend as K
from tensorflow.keras import constraints, initializers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.layers import (GRU, LSTM, Bidirectional,
                                     Conv1D, Dense, Dropout, Embedding,
                                     Flatten, GlobalMaxPooling1D,
                                     Layer, concatenate)
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical


class AbstractNN:
    """ 神经网络系列模型
    """
    def __init__(self, name, model_params=None):
        self.config = model_params if model_params else {}
        self._name = name
        self.models = {
            "textcnn": TextCNN,
            "textrnn": TextRNN,
        }

    def train(self, input_x, input_y, model_path, clf=None):
        """
        Args:
            input_x -- np.array([[11,3],[1,10]])
            input_y -- np.array([[0,0,0,1],[1,0,0,0]])
            model_path -- 模型路径
        Returns:
            config -- 模型日志
        """
        if not os.path.exists(model_path):
            print("Make dir `%s`." % model_path)
            os.mkdir(model_path)
        self.config["model"]["model_path"] = model_path
        self.config["model"]["name"] = self._name
        callbacks = []
        if self.config["common"].get("early_stopping", False):
            callbacks.append(EarlyStopping(monitor="loss"))
        if self.config["common"].get("summaries", False):
            log_dir = os.path.join(model_path, "summaries") 
            tb_callback = TensorBoard(log_dir=log_dir,
                                      write_graph=True,
                                      write_grads=True,
                                      write_images=True)
            callbacks.append(tb_callback)
        callbacks = callbacks if callbacks else None
        self.word_embedding = self.config.common.word_embedding
        self.config["common"].pop("word_embedding")  # 节省内存
        if clf:
            self.clf = clf
        else:
            self.clf = self.models[self._name](
                config=self.config,
                word_embedding=self.word_embedding).model
        self.clf.compile(optimizer="adam",
                         loss="categorical_crossentropy",
                         metrics=["accuracy"])
        if self.config["model"].get("train", True):  # 部分融合模型无需训练
            self.clf.fit(input_x, input_y,
                         validation_split=self.config.common.val_size,
                         batch_size=self.config.training.batch_size,
                         epochs=self.config.training.epoches,
                         class_weight="auto",
                         callbacks=callbacks)
        self.clf.save(os.path.join(model_path, "%s.h5" % self._name))
        return json.loads(self.clf.to_json())

    def load(self, model_path):
        model_path = os.path.join(model_path, "%s.h5" % self._name)
        self.clf = load_model(
            model_path,
            custom_objects={"AttentionLayer": AttentionLayer})

    def test(self, input_x, input_y, min_proba=0):
        test_result = []
        y_proba = self.clf.predict(input_x)
        for _r, _yp in enumerate(y_proba):
            maxv_ix_t = np.argmax(input_y[_r])
            maxv_ix_p = np.argmax(_yp)
            if _yp[maxv_ix_p] >= min_proba:
                test_result.append((maxv_ix_t, maxv_ix_p))
        return zip(*test_result)

    def predict(self, input_x):
        y_proba = self.clf.predict(input_x)
        maxv_ixs = np.argmax(y_proba, axis=1)
        predict_result = list(zip(maxv_ixs, np.max(y_proba, axis=1)))
        return predict_result


class TextCNN:
    """
    模型结构：词嵌入--卷积池化*x--concat_and_flat--dropout--全连接
    """
    def __init__(self, config, word_embedding=None):
        """
        Args:
            config -- 配置字典
        Kwargs:
            word_embedding(np.array) -- 词向量矩阵，当为None时表示没有使用词向量
        """
        self.config = config
        vocab_size = self.config.common.vocab_size
        num_class = self.config.common.num_class
        embed_dim = self.config.common.w2v_dim
        seq_len = self.config.common.sequence_length
        w2v_trainable = self.config.common.w2v_trainable
        num_filters = self.config.model.num_filters
        filter_sizes = self.config.model.filter_sizes
        dropout_keep_prob = self.config.model.dropout_keep_prob
        if word_embedding is not None:
            weights = [word_embedding]
        else:
            weights = word_embedding

        input_x = Input(shape=(seq_len,), dtype="int64", name="input_x")
        embedding = Embedding(vocab_size,
                              embed_dim,
                              weights=weights,
                              input_length=seq_len,
                              trainable=w2v_trainable,
                              name="embedding")(input_x)
        pooled_outputs = []
        for filter_size in filter_sizes:
            conv = Conv1D(num_filters,
                          filter_size,
                          padding="same",
                          strides=1,
                          name="conv_%s" % filter_size)(embedding)
            pool = GlobalMaxPooling1D(name="max_pool_%s" % filter_size)(conv)
            pooled_outputs.append(pool)

        h_pool = concatenate(pooled_outputs, axis=-1)
        h_pool_flat = Flatten()(h_pool)
        h_drop = Dropout(dropout_keep_prob, name="dropout")(h_pool_flat)
        output_y = Dense(num_class,
                         activation="softmax",
                         name="output_y")(h_drop)
        self.model = Model(inputs=input_x, outputs=output_y)


class AttentionLayer(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionLayer())
    """
    def __init__(self,
                 W_regularizer=None,
                 u_regularizer=None,
                 b_regularizer=None,
                 W_constraint=None,
                 u_constraint=None,
                 b_constraint=None,
                 bias=True,
                 **kwargs):
        self.supports_masking = True
        self.init = initializers.get("glorot_uniform")

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        定义权重，添加可训练参数
        """
        assert len(input_shape) == 3

        self.W = self.add_weight(
            name="%s_W" % self.name,
            shape=TensorShape((input_shape[-1], input_shape[-1])).as_list(),
            initializer=self.init,
            regularizer=self.W_regularizer,
            constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(
                name="%s_b" % self.name,
                shape=TensorShape((input_shape[-1],)).as_list(),
                initializer="zero",
                regularizer=self.b_regularizer,
                constraint=self.b_constraint)

        self.u = self.add_weight(
            name="%s_u" % self.name,
            shape=TensorShape((input_shape[-1],)).as_list(),
            initializer=self.init,
            regularizer=self.u_regularizer,
            constraint=self.u_constraint)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x, mask=None):
        """
        定义层功能
        """
        v = K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1)
        v = K.dot(x, self.W)

        if self.bias:
            v += self.b
        v = K.tanh(v)

        vu = K.squeeze(K.dot(v, K.expand_dims(self.u)), axis=-1)
        exps = K.exp(vu)

        if mask is not None:
            exps *= K.cast(mask, K.floatx())

        # 添加极小常数`epsilon`以避免`NAN`的出现
        at = exps / K.cast(
            K.sum(exps, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        at = K.expand_dims(at)

        return K.sum(x * at, axis=1)

    def compute_mask(self, input, input_mask=None):
        """
        不将mask传入一下层
        """
        return None

    def compute_output_shape(self, input_shape):
        """
        定义输出大小
        """
        return input_shape[0], input_shape[-1]


class TextRNN:
    """
    模型结构: 词嵌入--双向LSTM--全连接 或 词嵌入--双向GRU--全连接
    """
    def __init__(self, config, word_embedding=None):
        """
        Args:
            config -- 配置字典
        Kwargs:
            word_embedding(np.array) -- 词向量矩阵，当为None时表示没有使用词向量
        """
        self.config = config
        vocab_size = self.config.common.vocab_size
        num_class = self.config.common.num_class
        embed_dim = self.config.common.w2v_dim
        seq_len = self.config.common.sequence_length
        w2v_trainable = self.config.common.w2v_trainable

        core_layer = self.config.model.core_layer
        core_layer_units = self.config.model.core_layer_units
        dropout = self.config.model.dropout
        recurrent_dropout = self.config.model.recurrent_dropout
        add_attention = self.config.model.add_attention
        if not add_attention:
            return_sequences = False
        else:
            return_sequences = True

        if word_embedding is not None:
            weights = [word_embedding]
        else:
            weights = word_embedding

        input_x = Input(shape=(seq_len,), dtype="int64", name="input_x")
        embedding = Embedding(vocab_size,
                              embed_dim,
                              weights=weights,
                              input_length=seq_len,
                              trainable=w2v_trainable,
                              name="embedding")(input_x)
        if core_layer == "lstm":
            encoder = LSTM(core_layer_units,
                           dropout=dropout,
                           recurrent_dropout=recurrent_dropout,
                           return_sequences=return_sequences,
                           name=core_layer)(embedding)
        elif core_layer == "bi-lstm":
            encoder = Bidirectional(LSTM(core_layer_units,
                                         dropout=dropout,
                                         recurrent_dropout=recurrent_dropout,
                                         return_sequences=return_sequences),
                                    name=core_layer)(embedding)
        elif core_layer == "gru":
            encoder = GRU(core_layer_units,
                          dropout=dropout,
                          recurrent_dropout=recurrent_dropout,
                          return_sequences=return_sequences,
                          name=core_layer)(embedding)
        elif core_layer == "bi-gru":
            encoder = Bidirectional(GRU(core_layer_units,
                                        dropout=dropout,
                                        recurrent_dropout=recurrent_dropout,
                                        return_sequences=return_sequences),
                                    name=core_layer)(embedding)
        else:
            raise ValueError(
                "`%s` is not a supported encoder" % encoder)
        if add_attention:
            attention = AttentionLayer(name="attention")(encoder)
            output_y = Dense(num_class,
                             activation="softmax",
                             name="output_y")(attention)
        else:
            output_y = Dense(num_class,
                             activation="softmax",
                             name="output_y")(encoder)
        self.model = Model(input_x, output_y)
