# 简介

一个简单高效的中文短文本分类工具。

# 版本

0.1

# 表现

- 数据来源: NLPCC2017
  - 训练集: 包含18个类别的15.6万新闻标题
  - 测试集: 包含18个类别的3.6万新闻标题
- 词向量来源: Tencent AI Lab

## 字级别-基础模型

| 分类器 |  表征  | 精确率 | 召回率 | f1值  | 时间(单位: 秒) |
| :--------: | :-------: | :-------------: | :----------: | :-------: | :----------: |
|  **mnb**   |  **bow**  |    **0.694**    |  **0.688**   | **0.688** |    **8**     |
|    mnb     |  bow_l2   |      0.691      |    0.678     |   0.677   |      8       |
|    mnb     |   tfidf   |      0.688      |    0.651     |   0.649   |      9       |
|    gnb     |    w2v    |      0.517      |    0.506     |   0.505   |      70      |
|     lr     |    w2v    |      0.621      |    0.626     |   0.622   |     393      |
|    svm     |    w2v    |      0.590      |    0.582     |   0.580   |    59364     |
|     rf     |    w2v    |      0.405      |    0.380     |   0.367   |     105      |
|   **ft**   | **ngram** |    **0.715**    |  **0.709**   | **0.710** |    **12**    |

## 词汇级别-基础模型

| 分类器 |  表征  | 精确率 | 召回率 | f1值  | 时间(单位: 秒) |
| :--------: | :-----: | :-------------: | :----------: | :-------: | :----------: |
|    mnb     |   bow   |      0.767      |    0.742     |   0.739   |      9       |
|    mnb     | bow_l2  |      0.754      |    0.721     |   0.715   |      9       |
|    mnb     |  tfidf  |      0.755      |    0.707     |   0.693   |      10      |
|    gnb     |   w2v   |      0.733      |    0.723     |   0.725   |      58      |
|   **lr**   | **w2v** |    **0.792**    |  **0.794**   | **0.793** |   **357**    |
|    svm     |   w2v   |      0.783      |    0.780     |   0.780   |    25014     |
|     rf     |   w2v   |      0.644      |    0.627     |   0.621   |      92      |
|     ft     |  ngram  |      0.753      |    0.745     |   0.747   |      14      |

运行`report.sh`可获取以上报告到具体信息。

## 词汇级别-TextCNN

| 分类器 |  精确率  |  召回率   |   f1值    |
| :--------------: | :-------------: | :----------: | :-------: |
|     CNN-rand     |      0.754      |    0.747     |   0.748   |
|  **CNN-static**  |    **0.781**    |  **0.773**   | **0.774** |
|  CNN-non-static  |      0.779      |    0.772     |   0.773   |
| CNN-multichannel |        -        |      -       |     -     |

## 词汇级别-TextRNN

| 分类器 |  精确率  | 召回率 | f1值 |
| :-------------------: | :-------------: | :----------: | :-------: |
|       RNN-LSTM        |      0.794      |    0.789     |   0.790   |
|      RNN-BiLSTM       |      0.791      |    0.788     |   0.787   |
|      **RNN-GRU**      |    **0.795**    |  **0.792**   | **0.792** |
|       RNN-BiGRU       |      0.795      |    0.789     |   0.790   |
|  RNN-Attention-LSTM   |      0.773      |    0.769     |   0.769   |
| RNN-Attention-BiLSTM  |      0.774      |    0.764     |   0.765   |
| **RNN-Attention-GRU** |    **0.775**    |  **0.772**   | **0.772** |
|  RNN-Attention-BiGRU  |      0.775      |    0.771     |   0.770   |

## 词汇级别-Ensemble

通过原始训练集训练基础模型和集成模型。

| 分类器 |  精确率  | 召回率 | f1值 | 参数 |
| :--------------: | :-------------: | :----------: | :-------: | :--------------: |
| Base1-CNN-static |      0.783      |    0.778     |   0.778   |     epoch: 5     |
| Base2-GRU-static |       0.7       |    0.771     |   0.771   |    epoch: 10     |
|   **Average**    |    **0.797**    |  **0.793**   | **0.794** |        -         |
|      Concat      |      0.783      |    0.779     |   0.779   |     epoch: 5     |
|   Stacking-LR    |      0.796      |    0.791     |   0.792   |        -         |

将数据集按6:4切分训练集和测试集，然后通过训练集训练第一阶段基础模型，通过测试集预测获得结果与原始标签构成欣的训练集训练第二阶段集成模型。

| 分类器 |  精确率  | 召回率 | f1值 | 参数 |
| :--------------: | :-------------: | :----------: | :-------: | :--------------: |
| Base1-CNN-static |      0.783      |    0.778     |   0.778   |     epoch: 5     |
| Base2-GRU-static |      0.789      |    0.785     |   0.785   |    epoch: 10     |
|     Average      |      0.796      |    0.791     |   0.791   |        -         |
|      Concat      |      0.757      |    0.751     |   0.751   |    epoch: 25     |
| **Stacking-LR**  |    **0.798**    |  **0.797**   | **0.797** |      **-**       |

# 安装

## Python版本

Python 3.5+

## 依赖

**方法1**

```shell
pip install -r requirements.txt  # 需要将该程序包导入到项目中
```

**方法2**

```shell
python setup.py install(or develop)  # 可以跟使用其他python程序包一样使用该工具包
```

# 快速开始

## 命令行模式

### 参数列表

| 简写参数 | 完整参数               | 是否必须  | 解释                                       | 格式                                       |
| ---- | ------------------ | ----- | ---------------------------------------- | ---------------------------------------- |
| -h   | --help             | 否     | 显示此帮助信息并退出                               |                                          |
| -n   | --name             | **是** | 模型名字                                     |                                          |
| -m   | --model            | 否     | 分类器的选择，可选值包括{"lr", "ft(fastText)", "svm", "rf(random foreast)", "textcnn", "textrnn", "ensemble_nn_avg", "ensemble_nn_concat", "ensemble_nn_stacking"}，默认"lr" |                                          |
| -f   | --feature          | 否     | 特征提取的选择，取值包括{"bow", "bow_l2", "tfidf", "w2v"}，默认"w2v" |                                          |
| -t   | --train            | **否** | 训练的文本文件或者目录 **（训练时必须输入）**                | 每行包括类的名字和文本，\t分割，utf-8编码                 |
| -e   | --test             | **否** | 测试的文本文件或者目录 **（测试时必须输入）**                | 同训练数据                                    |
| -p   | -predict           | **否** | 预测的文本文件或者目录 **（预测时必须输入）**                |                                          |
| -u   | --user-dict        | 否     | 分词时额外使用的用户字典                             | 每行包括一个词汇、词频（可选）和词性（可选），并用空格隔开，如“广州图书馆 1000 n” |
| -s   | --stop-word        | 否     | 用户自定义的去除词                                | 每行包括一个词                                  |
| -w   | --wordvec-dict     | 否     | 自定义使用的词向量文件或者目录                          | 每行包括一个词和对应的词向量，词和词向量空格分开，词向量每个值之间也是空格分开  |
| -o   | --model-output     | **是** | 模型保存的位置                                  |                                          |
| -d   | --predict-download | 否     | 预测结果保存位置                                 |                                          |
| -V   | --version          | 否     | 显示版本信息并退出                                |                                          |

### 模型的训练和测试

```shell
python youmin_textclassifier_train.py -n="test" -t="./data_sample/train_data.txt" -e="./data_sample/test_data.txt" -o="./data/"
```

### 模型预测

```shell
python youmin_textclassifier_predict.py -n="test" -o="./data/" -p="./data_sample/predict_data.txt" -d="./data/predict.txt"
```

## Python包模式

```shell
cd examples
python3 classify.py base_on_list
python3 classify.py base_on_file
python3 classify.py base_on_dir
```
更多细节，请看到`examples/classify.py`

# 高级应用

请修改`youmin_textclassifier/config.py`

# 引用

[NLPCC2017示例代码以及数据描述](https://github.com/FudanNLP/nlpcc2017_news_headline_categorization)

[Tencent AI Lab Embedding Corpus for Chinese Words and Phrases]([https://ai.tencent.com/ailab/nlp/embedding.html](https://ai.tencent.com/ailab/nlp/embedding.html))

# 感谢

非常感谢我的导师Bertram给予的支持与帮助！

# 许可

Apache 2.0 license

# 联系

微信：whenif

邮箱：yongjin.weng@foxmail.com