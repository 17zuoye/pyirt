'''
This script deals with the data format problem.

The stardard format for pyirt is ( uid,eid,result),
where uid is the idx for test taker, eid is the idx for items

It is set in this way to deal with the sparsity in the massive dataset.

'''
import numpy as np

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

            #TODO: the current code is hard wired to uid starts from 0 to n
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


