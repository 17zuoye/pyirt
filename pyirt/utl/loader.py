'''
This script deals with the data format problem.

The stardard format for pyirt is ( uid,eid,result),
where uid is the idx for test taker, eid is the idx for items

It is set in this way to deal with the sparsity in the massive dataset.

'''
import numpy as np
import time
import os
import subprocess
from six import string_types


'''
# bsddb3 is hard to install
#
# example to install bsddb3 on OSX
# brew install Berkeley-db
# YES_I_HAVE_THE_RIGHT_TO_USE_THIS_BERKELEY_DB_VERSION=TRUE 
# BERKELEYDB_DIR=/usr/local/Cellar/berkeley-db/6.1.19 
# pip install bsddb3

try:
    import bsddb as diskdb
except:
    try:
        import bsddb3 as diskdb
    except:
        import shelve as diskdb

# Compact with bsddb3
if hasattr(diskdb, "hashopen"):
    diskdb.open = diskdb.hashopen
'''

import collections as cos


def from_matrix_to_list(indata_file, sep=',', header=False, is_uid=False):
    # assume the data takes the following format
    # (uid,) item1, item2, item3
    # (1,)   0,     1,     1
    is_skip = True
    is_init = False
    uid = 0

    result_list = []

    with open(indata_file, 'r') as f:
        for line in f:
            if is_skip and header:
                # if there is a header, skip
                is_skip = False
                continue
            segs = line.strip().split(sep)

            if len(segs) == 0:
                continue

            if not is_init:
                # calibrate item id
                if is_uid:
                    num_item = len(segs) - 1
                else:
                    num_item = len(segs)

            # parse
            for j in range(num_item):
                if is_uid:
                    idx = j + 1
                else:
                    idx = j
                result_list.append((uid, j, int(segs[idx])))

            # TODO: the current code is hard wired to uid starts from 0 to n
            # needs to remove the dependency
            uid += 1

    return result_list


def load_sim_data(sim_data_file):
    # this function loads the simulation data for testing
    # the sim format is
    #(result, uid, eid, theta, beta, alpha)
    test_data = []
    test_param = {}
    # the outputs are [a] solver readable dataset, [b] item parameter
    with open(sim_data_file, 'r') as f:
        for line in f:
            if line.strip() == '':
                continue

            result, uid, eid, theta, beta, alpha = line.strip().split(',')
            test_data.append((int(uid), int(eid), int(result)))
            if eid not in test_param:
                test_param[int(eid)] = {'alpha': float(alpha), 'beta': float(beta)}
    return test_data, test_param


def parse_item_paramer(item_param_dict, output_file=None):

    if output_file is not None:
        # open the file
        out_fh = open(output_file, 'w')

    sorted_eids = sorted(item_param_dict.keys())

    for eid in sorted_eids:
        param = item_param_dict[eid]
        alpha_val = np.round(param['alpha'], decimals=2)
        beta_val = np.round(param['beta'], decimals=2)
        if output_file is None:
            print(eid, alpha_val, beta_val)
        else:
            out_fh.write('{},{},{}\n'.format(eid, alpha_val, beta_val))

'''
Build a data storage facility that allows for memory dict and diskdb dict
'''


class data_storage(object):

    def setup(self, uids, eids, atags, mode='memory',
              tmp_dir=None, is_mount=False, user_name=None):
        
        start_time = time.time()
        self._process_data_memory(uids, eids, atags)
        self._init_data_param()
        print("--- Process: %f secs ---" % np.round((time.time() - start_time)))

       # initialize some intermediate variables used in the E step
        start_time = time.time()
        self._init_right_wrong_map_memory()
        print("--- Sparse Mapping: %f secs ---" % np.round((time.time() - start_time)))

    '''
    Need the following dictionary for esitmation routine
    (1) item -> user: key: eid, value: (uid, atag)
    (2) user -> item: key: uid, value: (eid, atag)
    '''

    def _process_data_memory(self, uids, eids, atags):
        self.num_log = len(uids)
        self.item2user = cos.defaultdict(list)
        self.user2item = cos.defaultdict(list)

        for i in range(self.num_log):
            eid = eids[i]
            uid = uids[i]
            atag = atags[i]
            # add to the data dictionary
            self.item2user[eid].append((uid, atag))
            self.user2item[uid].append((eid, atag))

    def _init_right_wrong_map_memory(self):
        self.right_map = cos.defaultdict(list)
        self.wrong_map = cos.defaultdict(list)

        for eid, log_result in self.item2user.items():
            for log in log_result:
                atag = log[1]
                uid = log[0]
                uid_idx = self.uidx[uid]
                if atag == 1:
                    self.right_map[eid].append(uid_idx)
                else:
                    self.wrong_map[eid].append(uid_idx)

    def _init_data_param(self):
        # system parameter
        self.uid_vec = [int(x) for x in self.user2item.keys()]
        self.num_user = len(self.uid_vec)
        self.eid_vec = [int(x) for x in self.item2user.keys()]
        self.num_item = len(self.eid_vec)

        # build a dictionary for fast uid index, which is used in map
        # Does not require uid or eid to be continuous
        self.uidx = dict(zip(self.uid_vec, range(len(self.uid_vec))))
        self.eidx = dict(zip(self.eid_vec, range(len(self.eid_vec))))

    def get_log(self, uid):
        log_list = self.user2item[uid]

    def get_rwmap(self, eid):
        right_uid_vec = self.right_map[eid]
        wrong_uid_vec = self.wrong_map[eid]
        return right_uid_vec, wrong_uid_vec
