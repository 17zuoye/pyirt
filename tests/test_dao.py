# -*- coding: utf-8 -*-
import unittest
import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)
from pyirt import irt
# from pyirt.dao import mongoDAO, mongoDb
import numpy as np


class TestDataSrc(unittest.TestCase):
    def setUp(self):
        self.data_src = 'test_data.csv'
        with open(self.data_src, 'w') as f:
            for i in range(50):
                if np.random.uniform() >= 0.5:
                    y = 1.0
                else:
                    y = 0.0
                f.write('%d,%d,%d\n' % (i, 0, y))

    def test_from_file(self):
        src_fp = open(self.data_src, 'r')
        item_param, user_param = irt(src_fp, max_iter=2)

    def test_from_list(self):
        data = []
        with open(self.data_src, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                uid, qid, ans = line.strip().split(',')
                data.append((int(uid), int(qid), int(ans)))
        item_param, user_param = irt(data, max_iter=2)
    """
    def test_from_mongo(self):
        # setup
        gid = 1001
        db = mongoDb()
        db.item2user_conn.remove({"gid": gid})
        db.user2item_conn.remove({"gid": gid})
        item2user_data = [[], []]
        with open(self.data_src, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                uid, _, ans = line.strip().split(',')
                item2user_data[int(ans)].append(int(uid))
                db.user2item_conn.insert_one({"gid": gid, "data": [("0", int(ans))], "id": int(uid)})
        db.item2user_conn.insert_one({"gid": gid, "data": item2user_data, "id": '0'})

        # usage
        test_dao = mongoDAO(group_id=gid)
        item_param, user_param = irt(test_dao, dao_type='db', max_iter=2)
        # need to test singleton and multiprocess forking
        item_param, user_param = irt(test_dao, dao_type='db', max_iter=2, is_parallel=True, num_cpu=2, check_interval=1)

        # tear down
        db.item2user_conn.remove({"gid": gid})
        db.user2item_conn.remove({"gid": gid})
    """
    def tearDown(self):
        os.remove(self.data_src)
