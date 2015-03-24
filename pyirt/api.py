import os
import sys

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

import solver

def irt(src, theta_bnds=[-4,4],
        alpha_bnds = [0.25,2], beta_bnds=[-2,2], in_guess_param = 'default',
        model_spec='2PL',
        is_mount = False, user_name = None):

    if model_spec == '2PL':
        model = solver.model.IRT_MMLE_2PL()
    else:
        raise Exception('Unknown model specification.')

    # load
    model.load_data(src, is_mount, user_name)
    model.load_param(theta_bnds, alpha_bnds, beta_bnds)
    model.load_guess_param(in_guess_param)

    # solve
    model.solve_EM()

    # post
    item_param_dict = model.get_item_param()
    user_param_dict = model.get_user_param()


    return item_param_dict, user_param_dict

