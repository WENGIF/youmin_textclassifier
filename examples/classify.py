# -*- coding: utf-8 -*-

from youmin_textclassifier.pipeline import YmTextClassifier


def base_on_list(name="test_list", model="lr", feature="w2v"):
    """
    框架最小成本测试样例
    """
    name = name if feature == "w2v" else name + "_" + feature
    ym_classify = YmTextClassifier(name, model=model, feature=feature)

    train_data = [
        ("数码家电", "上手华为P20系列 AI三摄首面世"),
        ("数码家电", "关于小米8青春版，来看网友怎么说？"),
        ("汽车", "奔驰宝马保时捷 三大性能车代表直线对决，猜谁赢！"),
        ("汽车", "买车！英菲尼迪QX80新车首付更低，部分热销车型更有0首付惊喜！")
    ]

    ym_classify.train(train_data)
    # ym_classify.load()
    test_result = ym_classify.test([
        ("汽车", "宝马3系，最新优惠10.04万"),
        ("数码家电", "华为mate20，旗舰新品强势来袭，华为徕卡联合设计！"),
    ])
    print("precision-score: {}\nrecall-score: {}\nf1-score: {}"\
        .format(test_result.precision,
                test_result.recall,
                test_result.f1_score))
    test_result.show_report()  # print(test_result)

    ym_classify.load()
    predict_result = ym_classify.predict(["宝马3系，最新优惠10.04万",
                                          "华为mate20，旗舰新品强势来袭，华为徕卡联合设计！"])
    predict_result.show_result()


def base_on_file(name="test_file", model="lr"):
    ym_classify = YmTextClassifier(name, model=model)

    ym_classify.train("./data/train_data.txt")

    # ym_classify.load()
    test_result = ym_classify.test("./data/test_data.txt")
    test_result.show_report()

    ym_classify.load()
    predict_result = ym_classify.predict("./data/predict_data.txt")
    predict_result.show_result()


def base_on_dir(name="test_dir", model="lr"):
    ym_classify = YmTextClassifier(name, model=model)

    ym_classify.train("./data/train")
    test_result = ym_classify.test("./data/test")
    test_result.show_report()

    ym_classify.load()
    predict_result = ym_classify.predict("./data/predict")
    predict_result.show_result()


def min_test_unit():
    """
    最小测试样例
    """
    base_on_list(name="test_list", model="lr", feature="w2v")
    base_on_list(name="test_list", model="svm", feature="w2v")
    base_on_list(name="test_list", model="rf", feature="w2v")
    base_on_list(name="test_list", model="mnb", feature="bow")
    base_on_list(name="test_list", model="mnb", feature="bow_l2")
    base_on_list(name="test_list", model="mnb", feature="tfidf")
    base_on_list(name="test_list", model="gnb", feature="w2v")
    base_on_list(name="test_list", model="ft", feature="w2v")


if __name__ == "__main__":
    import fire
    fire.Fire()
