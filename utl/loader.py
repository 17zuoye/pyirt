'''
This script deals with the data format problem.

The stardard format for pyirt is ( uid,eid,result),
where uid is the idx for test taker, eid is the idx for items

It is set in this way to deal with the sparsity in the massive dataset.

'''

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






