# -*- coding: utf-8 -*-

from setuptools import setup
from setuptools import find_packages


setup(
    name = "youmin_textclassifier",
    version = "0.1.0",
    keywords="Youmin text classifier",
    description = "A library for Text Classifier",
    license = "Apache License, Version 2.0 (the 'License')",
    url = "",
    author = "Kinson",
    author_email = "wengyongjin@foxmail.com",
    packages = ["youmin_textclassifier"],
    include_package_data = True,
    platforms = "any",
    install_requires = [
        "gensim==3.6.0",
        "numpy==1.22.0",
        "jieba==0.39",
        "Cython==0.27.3",
        "scikit-learn>=0.20.2",
        "fasttext==0.8.3",
        "prettytable==0.7.2",
        "fire==0.1.3",
    ],
    extras_require = {},
)
