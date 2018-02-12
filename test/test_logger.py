# encoding:utf-8
import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)

import unittest
from pyirt.logger import Logger


class TestLogger(unittest.TestCase):
    def setUp(self):
        if os.path.exists('mock/'):
            os.removedirs('mock/')
        os.mkdir('mock/')

    def test_log(self):
        logger = Logger.logger('mock/test.log')
        logger.debug("debug")
        logger.info("info")
        logger.error("error")
        logger.critical("critical")

        with open('mock/test.log') as f:
            self.assertTrue(len(f.readlines()) == 4)

    def tearDown(self):
        os.remove('mock/test.log')
        os.removedirs('mock/')


if __name__ == "__main__":
    unittest.main()
