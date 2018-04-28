'''
This script deals with the data format problem.

The stardard format for pyirt is ( user_id,item_id,result),
where user_id is the idx for test taker, item_id is the idx for items
It is set in this way to deal with the sparsity in the massive dataset.

'''
import numpy as np


def loadFromTuples(data):
    user_ids = []
    item_ids = []
    ans_tags = []
    if len(data) == 0:
        raise Exception('data are empty')

    for log in data:
        user_ids.append(log[0])
        item_ids.append(log[1])
        ans_tags.append(int(log[2]))

    return user_ids, item_ids, ans_tags


def loadFromHandle(fp, sep=','):
    # Default format is comma separated files,
    # Only int is allowed within the environment
    user_ids = []
    item_ids = []
    ans_tags = []

    for line in fp:
        if line == '':
            continue
        user_id_str, item_id_str, ans_tagstr = line.strip().split(sep)
        user_ids.append(user_id_str)
        item_ids.append(item_id_str)
        ans_tags.append(int(ans_tagstr))
    return user_ids, item_ids, ans_tags


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
    element_idxs = range(len(unique_elements))
    idx_ref = dict(zip(unique_elements, element_idxs))
    reverse_idx_ref = dict(zip(element_idxs, unique_elements))
    out_idx_list = [idx_ref[x] for x in in_list]

    return out_idx_list, idx_ref, reverse_idx_ref
