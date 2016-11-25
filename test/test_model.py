# -*- coding: utf-8 -*-

import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RootDir)

from pyirt._pyirt import irt

"""

#LAST7_data = utl.loader.from_matrix_to_list('data/LAST7.txt',sep='\t',is_uid=True)
src_handle = open('data/sim_data_simple.txt','r')

# load file handle
print('Load file handle.')
item_param,user_param = irt(src_handle)
src_handle.close()
"""

# load tuples
print('Load tuple data.')
src_handle = open(RootDir+'/data/single_param_data.txt','r')
src_data = []
for line in src_handle:
    if line == '':
        continue
    uidstr, eidstr, atagstr = line.strip().split(',')
    src_data.append((int(uidstr),int(eidstr),int(atagstr)))
src_handle.close()

item_param,user_param = irt(src_data)
print(item_param)
