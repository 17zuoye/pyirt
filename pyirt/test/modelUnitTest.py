import unittest
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import solver, utl
import ConfigParser
import io
LAST7_data = utl.loader.from_matrix_to_list('data/LAST7.txt',sep='\t',is_uid=True)
test_data, test_param = utl.loader.load_sim_data('data/sim_data.txt')

'''
PICK !
'''
run_data = test_data

#############################
eids = list(set([x[1] for x in run_data]))
guess_param_dict = {}
for eid in eids:
    guess_param_dict[eid] = {'c':0,'update_c':True}

test_model = solver.model.IRT_MMLE_2PL()
test_model.load_data(run_data)

sample_config = open(root_dir+'/config.cfg', 'r').read()
config = ConfigParser.RawConfigParser(allow_no_value=True)
config.readfp(io.BytesIO(sample_config))


test_model.load_config(config)
test_model.load_guess_param(guess_param_dict)

test_model.solve_EM()

# print out the result
utl.tools.parse_item_paramer(test_param, output_file = 'data/sim_true.txt')
utl.tools.parse_item_paramer(test_model.get_item_param(), output_file = 'data/sim_est.txt')

