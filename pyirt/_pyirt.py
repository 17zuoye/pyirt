import os
import sys

from .solver import model

def irt(src, theta_bnds=[-4,4],
        alpha_bnds = [0.25,2], beta_bnds=[-2,2], in_guess_param = 'default',
        model_spec='2PL',
        mode = 'memory',is_mount = False, user_name = None):

    if model_spec == '2PL':
        mod = model.IRT_MMLE_2PL()
    else:
        raise Exception('Unknown model specification.')

    # load
    mod.load_data(src, is_mount, user_name)
    mod.load_param(theta_bnds, alpha_bnds, beta_bnds)
    mod.load_guess_param(in_guess_param)

    # solve
    mod.solve_EM()

    # post
    item_param_dict = mod.get_item_param()
    user_param_dict = mod.get_user_param()


    return item_param_dict, user_param_dict

