# -*- coding: utf-8 -*-
"""
Usage: python youmin_textclassifier_train.py -n="test" -t="./data_sample/train_data.txt" -e="./data_sample/test_data.txt" -o="./data/"
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
                        choices=["bow", "bow_l2", "tfidf", "w2v"],
                        help="特征提取的选择，取值包括{\"bow\", \"bow_l2\", \"tfidf\", \"w2v\"}，默认\"w2v\"")
    parser.add_argument("-n", "--name", required=True, type=str, help="模型名字")
    parser.add_argument("-t", "--train", type=str, help="训练的文本文件或者目录")
    parser.add_argument("-e", "--test", type=str, help="测试的文本文件或者目录")
    parser.add_argument("-u", "--user-dict", type=str, help="分词时额外使用的用户字典")
    parser.add_argument("-s", "--stop-word", type=str, help="用户自定义的过滤词")
    parser.add_argument("-w", "--w2v-dict", type=str, help="分词时额外使用的用户字典")
    parser.add_argument("-o", "--model-output", required=True, type=str, help="模型保存的位置")
    # 高级参数
    parser.add_argument("-ensemble", "--ensemble", type=str, help="模型集成")
    parser.add_argument("-feature_word", "--feature-word", type=str, help="文本特征词汇统计导出路径")
    parser.add_argument("-min_proba", "--min-proba", type=float, help="测试的最低概率值")
    
    args = parser.parse_args()
    if args.train and not os.path.exists(args.train):
        raise ValueError("The `%s` not exists!" % args.train)
    if args.test and not os.path.exists(args.test):
        raise ValueError("The `%s` not exists!" % args.test)
    if not args.train and not args.test:
        raise ValueError("Please input one of the `--train` and `--test`")

    if not os.path.exists(args.model_output):
        raise ValueError("The `%s` not exists!" % args.model_output)
    if args.user_dict and not os.path.exists(args.user_dict):
        raise ValueError("The `%s` not exists!" % args.user_dict)
    if args.stop_word and not os.path.exists(args.stop_word):
        raise ValueError("The `%s` not exists!" % args.stop_word)
    if args.w2v_dict and not os.path.exists(args.w2v_dict):
        raise ValueError("The `%s` not exists!" % args.w2v_dict)
    return vars(args)


def classifier(kwargs):
    ym_classify = YmTextClassifier(**kwargs)
    if kwargs["train"]:
        ym_classify.train(kwargs["train"])
    else:
        ym_classify.load()
    if kwargs["test"]:
        test_result = ym_classify.test(kwargs["test"])
        print("precision-score: {}\nrecall-score: {}\nf1-score: {}"\
            .format(test_result.precision,
                    test_result.recall,
                    test_result.f1_score))
        test_result.show_report()


if __name__ == "__main__":
    args = get_args()
    classifier(args)
