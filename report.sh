#!/bin/bash

echo "*****[Start]*****"

source ../../.env3/bin/activate

levelVar="char word"

for level in $levelVar

do
    # 多项式贝叶斯
    featureVar="bow bow_l2 tfidf"

    for feature in $featureVar
    do
        start=$(date +%s)
        echo "====mnb-""$level-""$feature"
        python3 youmin_textclassifier_train.py -n="nlpcc2017_$level""_$feature" -m="mnb" -f="$feature" -t="./data/nlpcc2017_$level""_train.txt" -e="./data/nlpcc2017_$level""_test.txt" -o="./data/out/"
        end=$(date +%s)
        echo "Times: $(( $end - $start ))"
    done

    # 其他模型
    nameVar="gnb lr svm rf ft textcnn"

    for name in $nameVar
    do
        start=$(date +%s)
        echo "====$name-""$level-""w2v"
        python3 youmin_textclassifier_train.py -n="nlpcc2017_$level""_w2v" -m="$name" -f="w2v" -t="./data/nlpcc2017_$level""_train.txt" -e="./data/nlpcc2017_$level""_test.txt" -o="./data/out/"
        end=$(date +%s)
        echo "Times: $(( $end - $start ))"
    done

done

echo "*****[End]*****"
