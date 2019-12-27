# -*- coding: utf-8 -*-

from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from prettytable import PrettyTable


class YmTestResult(object):
    def __init__(self, yt, yp, average="weighted"):
        self.yt = yt
        self.yp = yp
        self.precision = precision_score(self.yt, self.yp, average=average)
        self.recall = recall_score(self.yt, self.yp, average=average)
        self.f1_score = f1_score(self.yt, self.yp, average=average)
        self.report = classification_report(self.yt, self.yp)

    def show_report(self):
        print(self.report)

    def __str__(self):
        return self.report


class YmPredictResult(object):
    def __init__(self, texts, predict_result, sep):
        self.predict_result = predict_result
        self.texts = texts
        self.sep = sep

    def result(self):
        return self.predict_result

    def show_result(self, top=None):
        x = PrettyTable(["Num", "Label", "Proba"])
        x.align["Num"] = "l"
        len_result = len(self.predict_result)
        top = min(top, len_result) if top else len_result
        for i in range(top):
            label, proba = self.predict_result[i]
            x.add_row([i + 1, label, proba])
        print(x)

    def save_result(self, predict_download, ids=None):
        with open(predict_download, "w") as fw:
            if ids:
                for _num, (_tx, (_label, _proba)) in \
                        enumerate(zip(self.texts, self.predict_result)):
                    fw.write(self.sep.join(
                        map(str, [ids[_num], _tx, _label, _proba])) + "\n")
            else:
                for _num, (_tx, (_label, _proba)) in \
                        enumerate(zip(self.texts, self.predict_result)):
                    fw.write(self.sep.join(
                        map(str, [_num + 1, _tx, _label, _proba])) + "\n")
