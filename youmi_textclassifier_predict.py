# -*- coding: utf-8 -*-
"""
Usage: python youmin_textclassifier_predict.py -n="test" -o="./data/" -p="./data_sample/predict_data.txt" -d="./data/predict.txt"
"""

import argparse
import os
import sys

from youmin_textclassifier.pipeline import YmTextClassifier


def get_args():
    if len(sys.argv) == 1:
        sys.argv.append("--help")
    parser = argparse.ArgumentParser(prog="youmin_textclassifier", description="A library for Text Classifier")
    parser.add_argument("-V", "--version", action="version", version="%(prog)s 0.1.0", help="显示版本信息并退出")
    parser.add_argument("-m", "--model", 
                        type=str, default="lr",
                        choices=[
                            "lr", "ft", "svm", "rf", "mnb", "gnb",
                            "textcnn", "textrnn",
                            "ensemble_nn_avg", "ensemble_nn_concat", "ensemble_nn_stacking"],
                        help="分类器的选择，可选值包括{\"lr\", \"ft(fastText)\"}，默认\"lr\"")
    parser.add_argument("-f", "--feature",
                        type=str, default="w2v",
                        choices=["bow", "tfidf", "w2v"],
                        help="特征提取的选择，取值包括{\"bow\", \"bow_l2\", \"tfidf\", \"w2v\"}，默认\"w2v\"")
    parser.add_argument("-n", "--name", required=True, type=str, help="模型名字")
    parser.add_argument("-p", "--predict", type=str, help="预测的文本文件或者目录")
    parser.add_argument("-u", "--user-dict", type=str, help="分词时额外使用的用户字典")
    parser.add_argument("-s", "--stop-word", type=str, help="用户自定义的过滤词")
    parser.add_argument("-w", "--w2v-dict", type=str, help="分词时额外使用的用户字典")
    parser.add_argument("-o", "--model-output", required=True, type=str, help="模型保存的位置")
    parser.add_argument("-d", "--predict-download", type=str, help="预测结果保存位置")
    # 高级参数
    parser.add_argument("-feature_word", "--feature-word", type=str, help="文本特征词汇统计导出路径")
    parser.add_argument("-min_proba", "--min-proba", type=float, help="测试的最低概率值")
    
    args = parser.parse_args()
    if not os.path.exists(args.model_output):
        raise ValueError("The `%s` not exists!" % args.model_output)

    if args.predict and not os.path.exists(args.predict):
        raise ValueError("The `%s` not exists!" % args.predict)
    if args.user_dict and not os.path.exists(args.user_dict):
        raise ValueError("The `%s` not exists!" % args.user_dict)
    if args.stop_word and not os.path.exists(args.stop_word):
        raise ValueError("The `%s` not exists!" % args.stop_word)
    if args.w2v_dict and not os.path.exists(args.w2v_dict):
        raise ValueError("The `%s` not exists!" % args.w2v_dict)
    if args.predict_download and not os.path.exists(args.predict_download.rsplit("/", 1)[0]):
        raise ValueError("The `%s` not exists!" % args.predict_download)
    return vars(args)


def classifier(kwargs):
    ym_classify = YmTextClassifier(**kwargs)
    ym_classify.load()
    predict_result = ym_classify.predict(kwargs["predict"])
    predict_result.show_result(top=10)
    if kwargs["predict_download"]:
        predict_result.save_result(predict_download=kwargs["predict_download"])


if __name__ == "__main__":
    args = get_args()
    classifier(args)
