# -*- coding: utf-8 -*-
import unittest
import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)
from pyirt import irt
import numpy as np


class TestDataSrc(unittest.TestCase):
    def setUp(self):
        self.data_src = 'test_data.csv'
        with open(self.data_src, 'w') as f:
            for i in range(1000):
                if np.random.uniform() >= 0.5:
                    y = 1.0
                else:
                    y = 0.0
                f.write('%d,%d,%d\n' % (i, 0, y))

    def test_from_file(self):
        src_fp = open(self.data_src, 'r')
        item_param, user_param = irt(src_fp)

    def test_from_memory(self):
        data = []
        with open(self.data_src, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                uid, qid, ans = line.strip().split(',')
                data.append((int(uid), int(qid), int(ans)))
        item_param, user_param = irt(data)

    def test_from_mongo(self):
        pass

    def tearDown(self):
        os.remove(self.data_src)
