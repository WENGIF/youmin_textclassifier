# -*- coding: utf-8 -*-
""" 数据库相关函数 """

import pickle
import sqlite3

from gensim.models import KeyedVectors

from .log import get_logger


class Word2VecDb:
    def __init__(self, db_path):
        print("Use: `{}`".format(db_path))
        self.db = sqlite3.connect(db_path)
        self.cur = self.db.cursor()

    def get_vec(self, key):
        """
        获取key对应的向量
        Args:
            key -- 词汇，如"我"
        Returns:
            vector -- 如果key存在，则返回对应的向量numpy.array(dim), 否则返回None
        """
        self.cur.execute("SELECT * FROM `model` WHERE `word`=?", (key, ))
        result = self.cur.fetchone()
        if result:
            return pickle.loads(result[1])
        else:
            return None

    def get_vec_batch(self, keys):
        """
        获取key对应的向量
        Args:
            keys -- 词汇列表，如["我", "来自", "广州"]
        Returns:
            vector list -- 如果keys存在，则返回对应的向量列表[numpy.array(dim),...], 否则返回None
        """
        try:
            if keys:
                self.cur.execute("SELECT * FROM `model` WHERE `word` IN ({})".\
                    format("'" + "','".join([k.replace("'", "''") for k in keys]) + "'"))
                res = [pickle.loads(d[1]) for d in self.cur.fetchall()]
                res = res if res else None
            else:
                res = None
        except Exception as er:
            print("Error: {}".format(er))
            res = None
        return res

    def insert_vec(self, key, val):
        try:
            self.cur.execute("INSERT INTO `model` VALUES (?, ?)", (key, pickle.dumps(val)))
            self.db.commit()
        except Exception as er:
            print("Key: `{}`, Value: `{}`\nError: {}!".format(key, val, er))

    def insert_vec_batch(self, table_name, iter_obj, batch):
        """
        Args:
            table_name -- 数据表名
            iter_obj   -- 数据对象，格式：[(?, ?, ..., ?), (), ..., ()]
            batch      -- 每批数量
        """
        each_len = len(iter_obj[0])
        place_holder = ", ".join(["?"] * each_len)
        sql_text = "INSERT INTO %s VALUES (%s)" % (table_name, place_holder)
        for i in range(0, len(iter_obj), batch):
            try:
                print("==>>[{},{})".format(i, i + batch))
                self.cur.executemany(sql_text, iter_obj[i : i + batch])
                self.db.commit()
            except Exception as er:
                print("[{},{})\nError: {}!".format(i, i + batch, er))

    def create(self):
        sql = """
        CREATE TABLE IF NOT EXISTS `model` (
            `word` VARCHAR(128) NOT NULL,
            `value` BLOB NOT NULL,
            PRIMARY KEY (`word`)
        )
        """
        self.cur.execute(sql)
        self.db.commit()

    def drop(self):
        sql = "DROP TABLE IF EXISTS `model`;"
        self.cur.execute(sql)
        self.db.commit()

    def get_size(self):
        self.cur.execute("SELECT COUNT(*) FROM `model`;")
        return self.cur.fetchone()

    def destroy(self):
        self.cur.close()
        self.db.close()


def vec_to_db(vec_path,
              db_path,
              binary=True,
              table_name="model",
              batch=10000):
    logger = get_logger(name="vec2db_log", level="debug")
    logger.info("====[init sqlite]====")
    db = Word2VecDb(db_path=vec_path)
    db.drop()
    db.create()
    logger.info("====[load vector]====")
    model = KeyedVectors.load_word2vec_format(vec_path, binary=binary)
    logger.info("====[insert to the db]====")
    iter_obj = [(w, pickle.dumps(model[w])) for w in model.vocab]
    del model
    db.insert_vec_batch(table_name, iter_obj, batch=batch)
    db.destroy()
    logger.info("====[update `%s`]====" % db_path)
