# encoding:utf-8
import logging
import time

import os

class Logger():
    @staticmethod
    def logger(logger_out_dir, mode='run'):
        
        date_str = time.strftime('%Y-%m-%d', time.localtime(time.time()))
        
        if mode == "debug":
            log_level = logging.DEBUG
        else:
            log_level = logging.INFO

        if logger_out_dir is not None:
            if not os.path.exists(logger_out_dir):
                os.makedirs(logger_out_dir)
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s %(pathname)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S',
                filename=logger_out_dir + '/{date}.log'.format(date=date_str),
                filemode='w')
        else:
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s %(pathname)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')

        # 创建一个handler，用于输出到控制台
        console = logging.StreamHandler()
        console.setLevel(log_level)  

        logger = logging.getLogger()
        # 给logger添加handler
        logger.addHandler(console)
        return logger
