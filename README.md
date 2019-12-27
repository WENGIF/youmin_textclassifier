# Introduction

A simple and efficient Chinese short text classification tool.

# Version

0.1

# Performance

- Data origin: NLPCC2017
  - Train set: 156k news titles with 18 labels
  - Test set: 36k news titles with 18 labels
- Embedding origin: Tencent AI Lab

## Char-level-base

| Classifier |  feature  | precision-score | recall-score | f1-score  | Time cost(s) |
| :--------: | :-------: | :-------------: | :----------: | :-------: | :----------: |
|  **mnb**   |  **bow**  |    **0.694**    |  **0.688**   | **0.688** |    **8**     |
|    mnb     |  bow_l2   |      0.691      |    0.678     |   0.677   |      8       |
|    mnb     |   tfidf   |      0.688      |    0.651     |   0.649   |      9       |
|    gnb     |    w2v    |      0.517      |    0.506     |   0.505   |      70      |
|     lr     |    w2v    |      0.621      |    0.626     |   0.622   |     393      |
|    svm     |    w2v    |      0.590      |    0.582     |   0.580   |    59364     |
|     rf     |    w2v    |      0.405      |    0.380     |   0.367   |     105      |
|   **ft**   | **ngram** |    **0.715**    |  **0.709**   | **0.710** |    **12**    |

## Word-level-base

| Classifier | feature | precision-score | recall-score | f1-score  | Time cost(s) |
| :--------: | :-----: | :-------------: | :----------: | :-------: | :----------: |
|    mnb     |   bow   |      0.767      |    0.742     |   0.739   |      9       |
|    mnb     | bow_l2  |      0.754      |    0.721     |   0.715   |      9       |
|    mnb     |  tfidf  |      0.755      |    0.707     |   0.693   |      10      |
|    gnb     |   w2v   |      0.733      |    0.723     |   0.725   |      58      |
|   **lr**   | **w2v** |    **0.792**    |  **0.794**   | **0.793** |   **357**    |
|    svm     |   w2v   |      0.783      |    0.780     |   0.780   |    25014     |
|     rf     |   w2v   |      0.644      |    0.627     |   0.621   |      92      |
|     ft     |  ngram  |      0.753      |    0.745     |   0.747   |      14      |

You can get this report by running the `report.sh`.

## Word-level-TextCNN

|    Classifier    | precision-score | recall-score | f1-score  |
| :--------------: | :-------------: | :----------: | :-------: |
|     CNN-rand     |      0.754      |    0.747     |   0.748   |
|  **CNN-static**  |    **0.781**    |  **0.773**   | **0.774** |
|  CNN-non-static  |      0.779      |    0.772     |   0.773   |
| CNN-multichannel |        -        |      -       |     -     |

## Word-level-TextRNN

|      Classifier       | precision-score | recall-score | f1-score  |
| :-------------------: | :-------------: | :----------: | :-------: |
|       RNN-LSTM        |      0.794      |    0.789     |   0.790   |
|      RNN-BiLSTM       |      0.791      |    0.788     |   0.787   |
|      **RNN-GRU**      |    **0.795**    |  **0.792**   | **0.792** |
|       RNN-BiGRU       |      0.795      |    0.789     |   0.790   |
|  RNN-Attention-LSTM   |      0.773      |    0.769     |   0.769   |
| RNN-Attention-BiLSTM  |      0.774      |    0.764     |   0.765   |
| **RNN-Attention-GRU** |    **0.775**    |  **0.772**   | **0.772** |
|  RNN-Attention-BiGRU  |      0.775      |    0.771     |   0.770   |

## Word-level-Ensemble

Ensemble model is trained with the same origin train dataset.

|    Classifier    | precision-score | recall-score | f1-score  | hyper parameters |
| :--------------: | :-------------: | :----------: | :-------: | :--------------: |
| Base1-CNN-static |      0.783      |    0.778     |   0.778   |     epoch: 5     |
| Base2-GRU-static |       0.7       |    0.771     |   0.771   |    epoch: 10     |
|   **Average**    |    **0.797**    |  **0.793**   | **0.794** |        -         |
|      Concat      |      0.783      |    0.779     |   0.779   |     epoch: 5     |
|   Stacking-LR    |      0.796      |    0.791     |   0.792   |        -         |

Split test dataset by a ratio of 6:4 and get a new train dataset one and test dataset. And then ensemble model is trained with new train dataset.

|    Classifier    | precision-score | recall-score | f1-score  | hyper parameters |
| :--------------: | :-------------: | :----------: | :-------: | :--------------: |
| Base1-CNN-static |      0.783      |    0.778     |   0.778   |     epoch: 5     |
| Base2-GRU-static |      0.789      |    0.785     |   0.785   |    epoch: 10     |
|     Average      |      0.796      |    0.791     |   0.791   |        -         |
|      Concat      |      0.757      |    0.751     |   0.751   |    epoch: 25     |
| **Stacking-LR**  |    **0.798**    |  **0.797**   | **0.797** |      **-**       |

# Installation

## Python version

Python 3.5+

## dependencies

**method 1**

```shell
pip install -r requirements.txt  # must add this package in your project
```

**method 2**

```shell
python setup.py install(or develop)  # You can use like other python modules
```

# Quickstart

## Command

### Params

| Abbreviation | Full               | Necessary  | Explanation                                       | Format                                       |
| ---- | ------------------ | ----- | ---------------------------------------- | ---------------------------------------- |
| -h   | --help             | No     | Show help                               |                                          |
| -n   | --name             | **Yes** | Model name                                    |                                          |
| -m   | --model            | No     | Classifier，Optional:{"lr", "ft(fastText)", "svm", "rf(random foreast)", "textcnn", "textrnn", "ensemble_nn_avg", "ensemble_nn_concat", "ensemble_nn_stacking"}，default "lr" |                                          |
| -f   | --feature          | No     | Optional:{"bow", "bow_l2", "tfidf", "w2v"}，default "w2v" |                                          |
| -t   | --train            | **No** | The path or directory of train data  **（Must be entered during training）**                | each line like "class_name\ttext"       |
| -e   | --test             | **No** | The path or directory of test data **（Must be entered during testing）**                | 同训练数据                                    |
| -p   | -predict           | **No** | The path or directory of predict data **（Must be entered during predicting）**                |                                          |
| -u   | --user-dict        | No     | User dictionary used for word segmentation                            | each line like "广州图书馆 1000 n" |
| -s   | --stop-word        | No     | stop words                                | each line like "word"                      |
| -w   | --wordvec-dict     | No     |  The path or directory of w2v                        | each line like "word dim_1 dim_2 ... dim_n" |
| -o   | --model-output     | **Yes** | The path of model saved.                                  |                                          |
| -d   | --predict-download | No     | The path of predicting result saved.                                |                                          |
| -V   | --version          | No     | Show version                              |                                          |

### Model train and test

```shell
python youmin_textclassifier_train.py -n="test" -t="./data_sample/train_data.txt" -e="./data_sample/test_data.txt" -o="./data/"
```

### Model predict

```shell
python youmin_textclassifier_predict.py -n="test" -o="./data/" -p="./data_sample/predict_data.txt" -d="./data/predict.txt"
```

## Python package

```shell
cd examples
python3 classify.py base_on_list
python3 classify.py base_on_file
python3 classify.py base_on_dir
```
For details, please see the file.

# Advanced Usage

Please see `youmin_textclassifier/config.py`

# References

[NLPCC2017示例代码以及数据描述](https://github.com/FudanNLP/nlpcc2017_news_headline_categorization)

[Tencent AI Lab Embedding Corpus for Chinese Words and Phrases]([https://ai.tencent.com/ailab/nlp/embedding.html](https://ai.tencent.com/ailab/nlp/embedding.html))

# License

All code and models are released under the Apache 2.0 license. See the LICENSE file for more information.

# Thanks

Thanks for the support and help from my mentor Bertram.

# Contact

WeChat: whenif

Mail: yongjin.weng@foxmail.com