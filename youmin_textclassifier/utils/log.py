# -*- coding: utf-8 -*-

import logging
import time
from logging.handlers import TimedRotatingFileHandler
from sys import stdout


def get_logger(name, level="info", log_path=None):
    """
    脚本辅助日志文件
    """
    if level == "debug":
        level = logging.DEBUG
    elif level == "info":
        level = logging.INFO
    elif level == "warning":
        level = logging.WARNING
    else:
        level = logging.ERROR
    logger = logging.getLogger(name)
    if not logger.handlers:
        log_formatter = logging.Formatter(
            fmt=("%(levelname)s %(asctime)s %(module)s "
                 "%(funcName)s[line:%(lineno)d]: %(message)s"))
        if log_path:
            log_handler = TimedRotatingFileHandler(filename=log_path,
                                                   when="D",
                                                   interval=1,
                                                   backupCount=30)
        else:
            log_handler = logging.StreamHandler(stream=stdout)
        log_handler.suffix = "%Y%m%d"
        log_handler.setFormatter(log_formatter)
        logger.addHandler(log_handler)
    logger.setLevel(level)
    return logger
