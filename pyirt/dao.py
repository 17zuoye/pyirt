# -*- encoding:utf-8 -*-
from util.dao import loadFromHandle, loadFromTuples, construct_ref_dict
import io
import time
from six import string_types
from collections import defaultdict


#TODO: bitmap is a function of DAO. Seperate that with database

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
        return self.database.right_map[item_idx]
    
    def get_wrong_map(self, item_idx):
        return self.database.wrong_map[item_idx]

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
        self.item2user = defaultdict(list)
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
            self.item2user[item_idx].append((user_idx, ans_tag))
            self.user2item[user_idx].append((item_idx, ans_tag))

    def _init_right_wrong_map(self):
        self.right_map = defaultdict(list)
        self.wrong_map = defaultdict(list)

        for item_idx, log_result in self.item2user.items():
            for log in log_result:
                ans_tag = log[1]
                user_idx = log[0]
                if ans_tag == 1:
                    self.right_map[item_idx].append(user_idx)
                else:
                    self.wrong_map[item_idx].append(user_idx)

