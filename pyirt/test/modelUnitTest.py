import unittest
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import solver, utl

LAST7_data = utl.loader.from_matrix_to_list('data/LAST7.txt',sep='\t',is_uid=True)
test_data, test_param = utl.loader.load_sim_data('data/sim_data.txt')

test_model = solver.model.IRT_MMLE_2PL()
test_model.load_data(test_data)
test_model.load_config()
test_model.solve_EM()

# print out the result
utl.tools.parse_item_paramer(test_param, output_file = 'data/sim_true.txt')
utl.tools.parse_item_paramer(test_model.item_param_dict, output_file = 'data/sim_est.txt')

