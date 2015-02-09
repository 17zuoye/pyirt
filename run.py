'''
load the data
'''

#TODO: the data have problems in fiiting multiple data

root_dir = '/home/junchen/git/pyirt/'
import sys

sys.path.insert(0,root_dir)
import solver
import numpy as np
import utl



LAST7_data = utl.loader.from_matrix_to_list(root_dir+'data/LAST7.txt',sep='\t',is_uid=True)
test_data, test_param = utl.loader.load_sim_data(root_dir + 'data/sim_data.txt')

test_model = solver.model.IRT_MMLE_2PL()
test_model.load_data(test_data)
test_model.load_config()
test_model.solve_EM()

# print out the result
utl.tools.parse_item_paramer(test_model.item_param_dict)



