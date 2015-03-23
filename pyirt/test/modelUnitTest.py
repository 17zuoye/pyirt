import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

from pyirt import *
#LAST7_data = utl.loader.from_matrix_to_list('data/LAST7.txt',sep='\t',is_uid=True)

# generate c
src_handle = open('data/sim_data_simple.txt','r')
item_param,user_param = irt(src_handle)

utl.loader.parse_item_paramer(item_param, output_file = 'data/sim_est.txt')

