# -*- coding: utf-8 -*-

from sklearn.metrics import classification_report


def model_evaluate(evaluate_info):
    """
    Args:
        evaluate_info -- 需评估信息，[(实际标签, 预测标签), (), ...,()]
    """
    y, y_pred = zip(*evaluate_info)
    print(classification_report(y, y_pred))
