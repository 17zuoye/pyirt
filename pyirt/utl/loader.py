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
try:
    import bsddb
except:
    import bsddb3 as bsddb

import collections as cos

def from_matrix_to_list(indata_file, sep=',',header=False, is_uid=False):
    # assume the data takes the following format
    # (uid,) item1, item2, item3
    # (1,)   0,     1,     1
    is_skip = True
    is_init = False
    uid = 0

    result_list = []

    with open(indata_file,'r') as f:
        for line in f:
            if is_skip and header:
                # if there is a header, skip
                is_skip=False
                continue
            segs = line.strip().split(sep)

            if len(segs)==0:
                continue

            if not is_init:
                # calibrate item id
                if is_uid:
                    num_item = len(segs)-1
                else:
                    num_item = len(segs)

            # parse
            for j in range(num_item):
                if is_uid:
                    idx = j+1
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

            result, uid, eid, theta, beta, alpha= line.strip().split(',')
            test_data.append((int(uid), int(eid), int(result)))
            if eid not in test_param:
                test_param[int(eid)]={'alpha':float(alpha), 'beta':float(beta)}
    return test_data, test_param

def load_dbm(dmb_val):
    # the format is 'id,flag;'*n
    pairs = dmb_val.split(';')
    log_list = []
    # last element is empty
    for i in xrange(len(pairs)-1):
        idstr,flagstr = pairs[i].split(',')
        log_list.append((int(idstr), int(flagstr)))
    return log_list




def parse_item_paramer(item_param_dict, output_file = None):

    if output_file is not None:
        # open the file
        out_fh = open(output_file,'w')

    sorted_eids = sorted(item_param_dict.keys())

    for eid in sorted_eids:
        param = item_param_dict[eid]
        alpha_val = np.round(param['alpha'],decimals=2)
        beta_val = np.round(param['beta'],decimals=2)
        if output_file is None:
            print eid, alpha_val, beta_val
        else:
            out_fh.write('{},{},{}\n'.format(eid, alpha_val, beta_val))

'''
Build a data storage facility that allows for memory dict and bsddb dict
'''

class data_storage(object):

    def setup(self, uids, eids, atags, mode='memory',
              tmp_dir = None, is_mount = False, user_name = None):
        # mode could be 'memory', which uses RAM, or 'dbm', which uses the hard disk.
        # When in 'dbm', set is_mount = False and the user_name to allows for
        # RAMdisk. However, it is not nearly as fast as memory when data is
        # large, unless uses SSD
        self.mode = mode

        if mode == 'dbm':
            # check if the tmp directory is accessible
            if not os.path.isdir(tmp_dir):
                os.mkdir(tmp_dir)
            # by default,do NOT mount the temp dir in memory, unless otherwise specified
            if is_mount:
                # create the directory
                subprocess.call(["sudo", "mount","-t","tmpfs","tmpfs",tmp_dir])
                # transfer ownwership
                subprocess.call(["sudo","chown",user_name+":root",tmp_dir])

            self.tmp_dir = tmp_dir  # passed in for later cache

        # pre processing
        start_time = time.time()
        if mode == 'memory':
            self._process_data_memory(uids, eids, atags)
        elif mode == 'dbm':
            self._process_data_dbm(uids, eids, atags)
        else:
            raise Exception('Unknown mode of data storage.')

        self._init_data_param()
        print("--- Process: %f secs ---" % np.round((time.time()-start_time)))


       # initialize some intermediate variables used in the E step
        start_time = time.time()
        if mode == 'memory':
            self._init_right_wrong_map_memory()
        elif mode == 'dbm':
            self._init_right_wrong_map_dbm()
        else:
            raise Exception('Unknown mode of data storage.')
        print("--- Sparse Mapping: %f secs ---" % np.round((time.time()-start_time)))

    '''
    Need the following dictionary for esitmation routine
    (1) item -> user: key: eid, value: (uid, atag)
    (2) user -> item: key: uid, value: (eid, atag)
    '''

    def _process_data_memory(self, uids, eids, atags):
        self.num_log = len(uids)
        self.item2user = cos.defaultdict(list)
        self.user2item = cos.defaultdict(list)

        for i in xrange(self.num_log):
            eid  = eids[i]
            uid  = uids[i]
            atag = atags[i]
            # add to the data dictionary
            self.item2user[eid].append((uid, atag))
            self.user2item[uid].append((eid, atag))

    def _process_data_dbm(self, uids, eids, atags):
        '''
        Memory efficiency optimization:

        (1) The matrix is sparse, so use three parallel lists for data storage.
        parallel lists are more memory eficient than tuple
        (2) For fast retrieval, turn three parallel list into dictionary by uid and eid
        (3) Python dictionary takes up a lot of memory, so use dbm

        # for more details, see
        http://stackoverflow.com/questions/2211965/python-memory-usage-loading-large-dictionaries-in-memory

        '''

        # always rewrite
        self.item2user = bsddb.hashopen(self.tmp_dir+'/item2user.db', 'n')
        self.user2item = bsddb.hashopen(self.tmp_dir+'/user2item.db', 'n')
        self.num_log = len(uids)

        for i in xrange(self.num_log):
            eid  = eids[i]
            uid  = uids[i]
            atag = atags[i]
            # if not initiated, init with empty str
            if str(eid) not in self.item2user:
                self.item2user['%d' % eid] = ''
            if str(uid) not in self.user2item:
                self.user2item['%d' % uid] = ''

            self.item2user['%d' % eid] += '%d,%d;' % (uid, atag)
            self.user2item['%d' % uid] += '%d,%d;' % (eid, atag)

    def _init_right_wrong_map_bdm(self):
        self.right_map = bsddb.hashopen(self.tmp_dir+'/right_map.db', 'n')
        self.wrong_map = bsddb.hashopen(self.tmp_dir+'/wrong_map.db', 'n')

        for eidstr, log_val_list in self.item2user.iteritems():
            log_result = utl.loader.load_dbm(log_val_list)
            for log in log_result:
                # The E step uses the index of the uid
                uid = log[0]
                atag = log[1]
                uid_idx = self.uidx[uid]
                if atag == 1:
                    if eidstr not in self.right_map:
                        self.right_map[eidstr] = '%d' % uid_idx
                    else:
                        self.right_map[eidstr] += ',%d' % uid_idx
                else:
                    if eidstr not in self.wrong_map:
                        self.wrong_map[eidstr] = '%d' % uid_idx
                    else:
                        self.wrong_map[eidstr] += ',%d' % uid_idx

    def _init_right_wrong_map_memory(self):
        self.right_map = cos.defaultdict(list)
        self.wrong_map = cos.defaultdict(list)

        for eid, log_result in self.item2user.iteritems():
            for log in log_result:
                atag = log[1]
                uid  = log[0]
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
        self.uidx = cos.defaultdict(int)
        for i in xrange(self.num_user):
            self.uidx[self.uid_vec[i]] = i

        self.eidx = cos.defaultdict(int)
        for j in xrange(self.num_item):
            self.eidx[self.eid_vec[j]] = j


    def get_log(self, uid):
        if self.mode == 'bdm':
            log_list = load_dbm(self.user2item_db['%d' % uid])
        elif self.mode == 'memory':
            log_list = self.user2item[uid]
        else:
            raise Exception('Unknown mode of storage.')
        return log_list

    def get_rwmap(self, eid):
        if self.mode == 'bdm':
            right_uid_vec = [int(x) for x in self.right_map[str(eid)].split(',') ]
            wrong_uid_vec = [int(x) for x in self.wrong_map[str(eid)].split(',') ]
        elif self.mode == 'memory':
            right_uid_vec = self.right_map[eid]
            wrong_uid_vec = self.wrong_map[eid]
        else:
            raise Exception('Unknown mode of storage.')
        return right_uid_vec, wrong_uid_vec



