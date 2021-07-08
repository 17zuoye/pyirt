# encoding:utf-8
import logging
import os


class Logger():
    @staticmethod
    def logger(log_path):
        logger = logging.getLogger()

        # initialize logger
        for h in logger.handlers[:]:
            logger.removeHandler(h)
            h.close()
        logger.setLevel(logging.DEBUG)

        # output log to console
        console = logging.StreamHandler()
        console.setLevel(logging.DEBUG)
        console.setFormatter(
            logging.Formatter(
                fmt='%(asctime)s %(levelname)s: %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
        )
        logger.addHandler(console)

        # output log to file
        if log_path is not None:
            log_dir = os.path.dirname(log_path)
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            file_handler = logging.FileHandler(
                filename=log_path,
                mode='w'
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(
                logging.Formatter(
                    fmt='%(asctime)s %(pathname)s[line:%(lineno)d] %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                )
            )
            logger.addHandler(file_handler)

        return logger
