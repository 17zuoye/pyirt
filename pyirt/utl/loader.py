'''
This script deals with the data format problem.

The stardard format for pyirt is ( user_id,item_id,result),
where user_id is the idx for test taker, item_id is the idx for items
It is set in this way to deal with the sparsity in the massive dataset.

'''
import numpy as np
import time
import os
import subprocess
from six import string_types
import collections as cos


def parse_item_paramer(item_param_dict, output_file=None):

    if output_file is not None:
        # open the file
        out_fh = open(output_file, 'w')

    sorted_item_ids = sorted(item_param_dict.keys())

    for item_id in sorted_item_ids:
        param = item_param_dict[item_id]
        alpha_val = np.round(param['alpha'], decimals=2)
        beta_val = np.round(param['beta'], decimals=2)
        if output_file is None:
            print(item_id, alpha_val, beta_val)
        else:
            out_fh.write('{},{},{}\n'.format(item_id, alpha_val, beta_val))


def construct_ref_dict(in_list):
    # map the in_list to a numeric variable from 0 to N
    unique_elements = list(set(in_list))
    element_idxs =  range(len(unique_elements))
    idx_ref = dict(zip(unique_elements, element_idxs))
    reverse_idx_ref = dict(zip(element_idxs, unique_elements))
    out_idx_list = [idx_ref[x] for x in in_list]

    return out_idx_list, idx_ref, reverse_idx_ref


'''
Build a data storage facility that allows for memory dict and diskdb dict
'''


class data_storage(object):

    def setup(self, user_idx_vec, item_idx_vec, ans_tags):
        
        start_time = time.time()
        self._process_data(user_idx_vec, item_idx_vec, ans_tags)
        print("--- Process: %f secs ---" % np.round((time.time() - start_time)))

       # initialize some intermediate variables used in the E step
        start_time = time.time()
        self._init_right_wrong_map()
        print("--- Sparse Mapping: %f secs ---" % np.round((time.time() - start_time)))

    '''
    Need the following dictionary for esitmation routine
    (1) item -> user: key: item_id, value: (user_id, ans_tag)
    (2) user -> item: key: user_id, value: (item_id, ans_tag)
    '''

    def _process_data(self, user_idx_vec, item_idx_vec, ans_tags):
        self.num_log = len(user_idx_vec)
        self.item2user = cos.defaultdict(list)
        self.user2item = cos.defaultdict(list)
        self.num_user = max(user_idx_vec)+1 # start count from 0
        self.num_item = max(item_idx_vec)+1

        for i in range(self.num_log):
            item_idx = item_idx_vec[i]
            user_idx = user_idx_vec[i]
            ans_tag = ans_tags[i]
            # add to the data dictionary
            self.item2user[item_idx].append((user_idx, ans_tag))
            self.user2item[user_idx].append((item_idx, ans_tag))

    def _init_right_wrong_map(self):
        self.right_map = cos.defaultdict(list)
        self.wrong_map = cos.defaultdict(list)

        for item_idx, log_result in self.item2user.items():
            for log in log_result:
                ans_tag = log[1]
                user_idx = log[0]
                if ans_tag == 1:
                    self.right_map[item_idx].append(user_idx)
                else:
                    self.wrong_map[item_idx].append(user_idx)

    def get_log(self, user_idx):
        log_list = self.user2item[user_idx]
        return log_list

    def get_rwmap(self, item_idx):
        right_user_idx_vec = self.right_map[item_idx]
        wrong_user_idx_vec = self.wrong_map[item_idx]
        return right_user_idx_vec, wrong_user_idx_vec
