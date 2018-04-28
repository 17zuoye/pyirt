# encoding:utf-8
import logging
import os


class Logger():
    @staticmethod
    def logger(log_path):
        if log_path is not None:
            log_dir = os.path.dirname(log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(pathname)s[line:%(lineno)d] %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename=log_path,
                filemode='w')
        else:
            logging.basicConfig(
                level=logging.DEBUG,
                format='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

        # 创建一个handler，用于输出到控制台
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)

        logger = logging.getLogger()
        # 给logger添加handler
        logger.addHandler(console)
        return logger
