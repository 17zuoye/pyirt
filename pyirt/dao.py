# -*- encoding:utf-8 -*-
import io
import time
from six import string_types
from collections import defaultdict

from .util.dao import loadFromHandle, loadFromTuples, construct_ref_dict

import pymongo

class mongoDAO(object): 
    def __init__(self, connect_config, num_log, group_id=1):
        user_name = connect_config['user']
        password = connect_config['password']
        address = connect_config['address']  # IP:PORT
        db_name = connect_config['db']
        if 'authsource' not in connect_config:
            mongouri = 'mongodb://{un}:{pw}@{addr}'.format(un=user_name, pw=password, addr=address)
        else:
            authsource = connect_config['authsource']
            mongouri = 'mongodb://{un}:{pw}@{addr}/?authsource={auth_src}'.format(un=user_name, pw=password, addr=address, auth_src=authsource)
        try:
            self.client = pymongo.MongoClient(mongouri, serverSelectionTimeoutMS=10)
        except:
            raise 
        
        user2item_collection_name = 'irt_user2item'
        item2user_collection_name = 'irt_item2user'
        
        self.user2item = self.client[db_name][user2item_collection_name]
        self.item2user = self.client[db_name][item2user_collection_name]
        
        # TODO:不能做全量扫描
        user_ids =  list(set([x['id'] for x in self.user2item.find({'gid':group_id}, {'id':1})]))
        item_ids =  list(set([x['id'] for x in self.item2user.find({'gid':group_id}, {'id':1})]))
        
        _, self.user_idx_ref, self.user_reverse_idx_ref = construct_ref_dict(user_ids) 
        _, self.item_idx_ref, self.item_reverse_idx_ref = construct_ref_dict(item_ids)
        
        self.stat = {'log':num_log, 'user':len(self.user_idx_ref.keys()), 'item':len(self.item_idx_ref.keys())}

    def get_num(self, name):
        if name not in ['user','item','log']:
            raise Exception('Unknown stat source %s'%name)
        return self.stat[name]

    def get_log(self, user_idx):
        user_id = self.translate('user', user_idx)
        res = self.user2item.find({'id':user_id})
        log_list = res[0]['data']
        return [(self.item_idx_ref[x[0]], x[1]) for x in log_list]

    def get_right_map(self, item_idx):
        item_id = self.translate('item', item_idx) 
        res = self.item2user.find({'id':item_id})
        return [self.user_idx_ref[x] for x in res[0]['data']['1']]

    def get_wrong_map(self, item_idx):
        item_id = self.translate('item', item_idx) 
        res = self.item2user.find({'id':item_id})
        return [self.user_idx_ref[x] for x in res[0]['data']['0']]

    def translate(self, data_type, idx):
        if data_type == 'item':
            return self.item_reverse_idx_ref[idx]
        elif data_type == 'user':
            return self.user_reverse_idx_ref[idx]
    
    def __del__(self):
        self.client.close()

class localDAO(object):

    def __init__(self, src):
         
        self.database = localDataBase(src)
        
        # quasi-bitmap 
        user_id_idx_vec, self.user_idx_ref, self.user_reverse_idx_ref = construct_ref_dict(self.database.user_ids) 
        item_id_idx_vec, self.item_idx_ref, self.item_reverse_idx_ref = construct_ref_dict(self.database.item_ids)
        
        self.database.setup(user_id_idx_vec, item_id_idx_vec, self.database.ans_tags)

    def get_num(self, name):
        if name not in ['user','item','log']:
            raise Exception('Unknown stat source %s'%name)
        return self.database.stat[name]

    def get_log(self, user_idx):
        return self.database.user2item[user_idx]

    def get_right_map(self, item_idx):
        return self.database.item2user[1][item_idx]
    
    def get_wrong_map(self, item_idx):
        return self.database.item2user[0][item_idx]

    def translate(self, data_type, idx):
        if data_type == 'item':
            return self.item_reverse_idx_ref[idx]
        elif data_type == 'user':
            return self.user_reverse_idx_ref[idx]


class localDataBase(object):
    def __init__(self, src):

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
            print("--- Process: %f secs ---" % np.round((time.time() - start_time)))

       # initialize some intermediate variables used in the E step
        start_time = time.time()
        self._init_right_wrong_map()
        if msg:
            print("--- Sparse Mapping: %f secs ---" % np.round((time.time() - start_time)))

    '''
    Need the following dictionary for esitmation routine
    (1) item -> user: key: item_id, value: (user_id, ans_tag)
    (2) user -> item: key: user_id, value: (item_id, ans_tag)
    '''

    def _process_data(self, user_idx_vec, item_idx_vec, ans_tags):
        self.item2user = defaultdict(defaultdict(list))
        self.user2item = defaultdict(list)
        
        self.stat = {}
        self.stat['log'] = len(user_idx_vec)
        self.stat['user'] =  max(user_idx_vec)+1 # start count from 0
        self.stat['item'] = max(item_idx_vec)+1

        for i in range(self.stat['log']):
            item_idx = item_idx_vec[i]
            user_idx = user_idx_vec[i]
            ans_tag = ans_tags[i]
            # add to the data dictionary
            self.item2user[item_idx][ans_tag].append(user_idx)
            self.user2item[user_idx].append((item_idx, ans_tag))

