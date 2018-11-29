# -*- encoding:utf-8 -*-
import io
import time
from collections import defaultdict
import pymongo
from datetime import datetime
import numpy as np
from .util.dao import loadFromHandle, loadFromTuples, construct_ref_dict

from decouple import config
MONGO_USER_NAME = config('MONGO_USER_NAME', default="")
MONGO_PASSWORD = config('MONGO_PASSWORD', default="")
MONGO_ADDRESS = config('MONGO_ADDRESS', default="")
MONGO_AUTH_SOURCE = config('MONGO_AUTH_SOURCE', default="")
MONGO_DB_NAME = config('MONGO_DB_NAME', default="")


class mongoDb(object):
    """ cannot use singleton design, otherwise gets 'Warning: MongoClient opened before fork. Create MongoClient only after forking.'
    """
    def __init__(self):
        mongouri = 'mongodb://{un}:{pw}@{addr}'.format(un=MONGO_USER_NAME, pw=MONGO_PASSWORD, addr=MONGO_ADDRESS)
        if MONGO_AUTH_SOURCE:
            mongouri += '/?authsource={auth_src}'.format(auth_src=MONGO_AUTH_SOURCE)
        # connect
        try:
            self.client = pymongo.MongoClient(mongouri, connect=False, serverSelectionTimeoutMS=10, waitQueueTimeoutMS=100, readPreference='secondaryPreferred')
        except Exception as e:
            raise e
        self.user2item_conn = self.client[MONGO_DB_NAME]['irt_user2item']
        self.item2user_conn = self.client[MONGO_DB_NAME]['irt_item2user']

    def __del__(self):
        self.client.close()


def search_filter(search_id, gid):
    return {'id': search_id, 'gid': gid}


class mongoDAO(object):
    # NOTE: mongoDAO does not use the runtime logger
    # NOTE: The client and the connection is not passed by self because of parallel processing
    def __init__(self, group_id=1, is_msg=False):
        db = mongoDb()
        user_ids = list(set([x['id'] for x in db.user2item_conn.find({'gid': group_id}, {'id': 1})]))
        item_ids = list(set([x['id'] for x in db.item2user_conn.find({'gid': group_id}, {'id': 1})]))

        _, self.user_idx_ref, self.user_reverse_idx_ref = construct_ref_dict(user_ids)
        _, self.item_idx_ref, self.item_reverse_idx_ref = construct_ref_dict(item_ids)

        self.stat = {'user': len(self.user_idx_ref.keys()), 'item': len(self.item_idx_ref.keys())}

        self.gid = group_id
        self.is_msg = is_msg

    def open_conn(self, name):
        if name == "item2user":
            return mongoDb().item2user_conn
        elif name == "user2item":
            return mongoDb().user2item_conn
        else:
            raise ValueError('conn name must be either item2user or user2item')

    def get_num(self, name):
        if name not in ['user', 'item']:
            raise Exception('Unknown stat source %s' % name)
        return self.stat[name]

    def get_log(self, user_idx, user2item_conn):
        user_id = self.translate('user', user_idx)
        # query
        if self.is_msg:
            stime = datetime.now()
            res = user2item_conn.find(search_filter(user_id, self.gid))
            etime = datetime.now()
            search_time = int((etime - stime).microseconds / 1000)
            if search_time > 100:
                print('warning: slow search:%d' % search_time)
        else:
            res = user2item_conn.find(search_filter(user_id, self.gid))
        # parse
        res_num = res.count()
        if res_num == 0:
            return_list = []
        elif res_num > 1:
            raise Exception('duplicate doc for (%s, %d) in user2item' % (user_id, self.gid))
        else:
            log_list = res[0]['data']
            return_list = [(self.item_idx_ref[x[0]], x[1]) for x in log_list]
        return return_list

    def get_map(self, item_idx, ans_key_list, item2user_conn):
        item_id = self.translate('item', item_idx)
        # query
        if self.is_msg:
            stime = datetime.now()
            res = item2user_conn.find(search_filter(item_id, self.gid))
            etime = datetime.now()
            search_time = int((etime - stime).microseconds / 1000)
            if search_time > 100:
                print('warning:slow search:%d' % search_time)
        else:
            res = item2user_conn.find(search_filter(item_id, self.gid))
        # parse
        res_num = res.count()
        if res_num == 0:
            return_list = [[] for ans_key in ans_key_list]
        elif res_num > 1:
            raise Exception('duplicate doc for (%s, %d) in item2user' % (item_id, self.gid))
        else:
            doc = res[0]['data']
            return_list = []
            for ans_key in ans_key_list:
                if str(ans_key) in doc:
                    return_list.append([self.user_idx_ref[x] for x in doc[str(ans_key)]])
                else:
                    return_list.append([])
        return return_list

    def translate(self, data_type, idx):
        if data_type == 'item':
            return self.item_reverse_idx_ref[idx]
        elif data_type == 'user':
            return self.user_reverse_idx_ref[idx]


class localDAO(object):

    def __init__(self, src, logger):

        self.database = localDataBase(src, logger)

        # quasi-bitmap
        user_id_idx_vec, self.user_idx_ref, self.user_reverse_idx_ref = construct_ref_dict(self.database.user_ids)
        item_id_idx_vec, self.item_idx_ref, self.item_reverse_idx_ref = construct_ref_dict(self.database.item_ids)

        self.database.setup(user_id_idx_vec, item_id_idx_vec, self.database.ans_tags)

    def get_num(self, name):
        if name not in ['user', 'item']:
            raise Exception('Unknown stat source %s' % name)
        return self.database.stat[name]

    def get_log(self, user_idx):
        return self.database.user2item[user_idx]

    def get_map(self, item_idx, ans_key_list):
        results = []
        for ans_key in ans_key_list:
            try:
                results.append(self.database.item2user_map[str(ans_key)][item_idx])
            except KeyError:
                results.append([])
        return results

    def close_conn(self):
        pass

    def translate(self, data_type, idx):
        if data_type == 'item':
            return self.item_reverse_idx_ref[idx]
        elif data_type == 'user':
            return self.user_reverse_idx_ref[idx]


class localDataBase(object):
    def __init__(self, src, logger):

        self.logger = logger
        if isinstance(src, io.IOBase):
            # if the src is file handle
            self.user_ids, self.item_ids, self.ans_tags = loadFromHandle(src)
        else:
            # if the src is list of tuples
            self.user_ids, self.item_ids, self.ans_tags = loadFromTuples(src)

    def setup(self, user_idx_vec, item_idx_vec, ans_tags, msg=False):

        start_time = time.time()
        self._process_data(user_idx_vec, item_idx_vec, ans_tags)
        if msg:
            self.logger.debug("--- Process: %f secs ---" % np.round((time.time() - start_time)))

        # initialize some intermediate variables used in the E step
        start_time = time.time()
        self._init_item2user_map()
        if msg:
            self.logger.debug("--- Sparse Mapping: %f secs ---" % np.round((time.time() - start_time)))

    '''
    Need the following dictionary for esitmation routine
    (1) item -> user: key: item_id, value: (user_id, ans_tag)
    (2) user -> item: key: user_id, value: (item_id, ans_tag)
    '''

    def _process_data(self, user_idx_vec, item_idx_vec, ans_tags):
        self.item2user = {}
        self.user2item = defaultdict(list)

        self.stat = {}
        num_log = len(user_idx_vec)
        self.stat['user'] = max(user_idx_vec) + 1  # start count from 0
        self.stat['item'] = max(item_idx_vec) + 1

        for i in range(num_log):
            item_idx = item_idx_vec[i]
            user_idx = user_idx_vec[i]
            ans_tag = ans_tags[i]
            # add to the data dictionary
            if item_idx not in self.item2user:
                self.item2user[item_idx] = defaultdict(list)
            self.item2user[item_idx][ans_tag].append(user_idx)
            self.user2item[user_idx].append((item_idx, ans_tag))

    def _init_item2user_map(self, ans_key_list=['0', '1']):

        self.item2user_map = defaultdict(dict)

        for item_idx, log_result in self.item2user.items():
            for ans_tag, user_idx_vec in log_result.items():
                self.item2user_map[str(ans_tag)][item_idx] = user_idx_vec
