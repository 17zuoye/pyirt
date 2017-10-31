# -*- encoding:utf-8 -*-
import io
import time
from six import string_types
from collections import defaultdict

from .util.dao import loadFromHandle, loadFromTuples, construct_ref_dict
import pymongo
from datetime import datetime



#NOTE: mongoDAO不使用runtime的logger体系
class mongoDAO(object): 

    def __init__(self,connect_config, group_id=1, is_msg=False):
        

        self.connect_config = connect_config

        client = self.open_conn()
        self.db_name = connect_config['db']

        self.user2item_collection_name = 'irt_user2item'
        self.item2user_collection_name = 'irt_item2user'
        
        user2item_conn = client[self.db_name][self.user2item_collection_name]
        item2user_conn = client[self.db_name][self.item2user_collection_name]

        
        user_ids = list(set([x['id'] for x in user2item_conn.find({'gid':group_id},{'id':1})]))
        item_ids = list(set([x['id'] for x in item2user_conn.find({'gid':group_id},{'id':1})]))
        
        _, self.user_idx_ref, self.user_reverse_idx_ref = construct_ref_dict(user_ids) 
        _, self.item_idx_ref, self.item_reverse_idx_ref = construct_ref_dict(item_ids)
        
        self.stat = {'user':len(self.user_idx_ref.keys()), 'item':len(self.item_idx_ref.keys())}
         
        self.gid = group_id
        self.is_msg = is_msg
        
        client.close()

    def open_conn(self):
        
        user_name = self.connect_config['user']
        password = self.connect_config['password']
        address = self.connect_config['address']  # IP:PORT
        if 'authsource' not in self.connect_config:
            mongouri = 'mongodb://{un}:{pw}@{addr}'.format(un=user_name, pw=password, addr=address)
        else:
            authsource = self.connect_config['authsource']
            mongouri = 'mongodb://{un}:{pw}@{addr}/?authsource={auth_src}'.format(un=user_name, pw=password, addr=address, auth_src=authsource)
        try:
            client = pymongo.MongoClient(mongouri, connect=False, serverSelectionTimeoutMS=10, waitQueueTimeoutMS=100 ,readPreference='secondaryPreferred')
        except:
            raise 
        return client


    def get_num(self, name):
        if name not in ['user','item']:
            raise Exception('Unknown stat source %s'%name)
        return self.stat[name]

    def get_log(self, user_idx, user2item_conn):
        user_id = self.translate('user', user_idx)
        # query
        if self.is_msg:
            stime = datetime.now()
            res = user2item_conn.find({'id':user_id, 'gid':self.gid})
            etime = datetime.now()
            search_time = int((etime-stime).microseconds/1000)
            if search_time > 100:
                print('warning: slow search:%d' % search_time)
        else:
            res = user2item_conn.find({'id':user_id, 'gid':self.gid})
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
            res = item2user_conn.find({'id':item_id, 'gid':self.gid})
            etime = datetime.now()
            search_time = int((etime-stime).microseconds/1000)
            if search_time > 100:
                print('warning:slow search:%d' % search_time)
        else:
            res = item2user_conn.find({'id':item_id, 'gid':self.gid})
        # parse
        res_num = res.count()
        if res_num == 0:
            return_list =  [[] for ans_key in ans_key_list]
        elif res_num > 1:
            raise Exception('duplicate doc for (%s, %d) in item2user' % (item_id, self.gid))
        else:
            doc = res[0]['data']
            return_list = []
            for ans_key in ans_key_list:
                if str(ans_key) in doc:
                    return_list.append([self.user_idx_ref[x] for x in doc[str(ans_key)]] )
                else:
                    return_list.append([])
        return return_list
    
    def translate(self, data_type, idx):
        if data_type == 'item':
            return self.item_reverse_idx_ref[idx]
        elif data_type == 'user':
            return self.user_reverse_idx_ref[idx]
    
class localDAO(object):

    def __init__(self,  src, logger):
         
        self.database = localDataBase(src, logger)
        
        # quasi-bitmap 
        user_id_idx_vec, self.user_idx_ref, self.user_reverse_idx_ref = construct_ref_dict(self.database.user_ids) 
        item_id_idx_vec, self.item_idx_ref, self.item_reverse_idx_ref = construct_ref_dict(self.database.item_ids)
        
        self.database.setup(user_id_idx_vec, item_id_idx_vec, self.database.ans_tags)

    def get_num(self, name):
        if name not in ['user','item']:
            raise Exception('Unknown stat source %s'%name)
        return self.database.stat[name]

    def get_log(self, user_idx):
        return self.database.user2item[user_idx]

    def get_map(self, item_idx, ans_key_list):
        # NOTE: return empty list for invalid ans key
        return [self.database.item2user_map[str(ans_key)][item_idx] for ans_key in ans_key_list]
   
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
            logger.debug("--- Process: %f secs ---" % np.round((time.time() - start_time)))

       # initialize some intermediate variables used in the E step
        start_time = time.time()
        self._init_item2user_map()
        if msg:
            logger.debug("--- Sparse Mapping: %f secs ---" % np.round((time.time() - start_time)))

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
        self.stat['user'] =  max(user_idx_vec)+1 # start count from 0
        self.stat['item'] = max(item_idx_vec)+1

        for i in range(num_log):
            item_idx = item_idx_vec[i]
            user_idx = user_idx_vec[i]
            ans_tag = ans_tags[i]
            # add to the data dictionary
            if item_idx not in self.item2user:
                self.item2user[item_idx] = defaultdict(list)
            self.item2user[item_idx][ans_tag].append(user_idx)
            self.user2item[user_idx].append((item_idx, ans_tag))

    def _init_item2user_map(self, ans_key_list = ['0','1']):
        
        self.item2user_map = defaultdict(dict)

        for item_idx, log_result in self.item2user.items():
            for ans_tag, user_idx_vec in log_result.items():
                self.item2user_map[str(ans_tag)][item_idx] = user_idx_vec

