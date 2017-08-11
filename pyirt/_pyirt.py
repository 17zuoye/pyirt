# -*-coding:utf-8-*-
from .solver import model
from .dao import localDAO

def irt(data_src,
        dao = 'memory',
        theta_bnds=[-4, 4], num_theta=11,
        alpha_bnds=[0.25, 2], beta_bnds=[-2, 2], in_guess_param='default',
        model_spec='2PL',
        max_iter=10, tol=1e-3, nargout=2,
        is_msg=False, is_parallel=False):


    # load data
    if dao=='memory':
        dao_instance = localDAO(data_src)
    else:
        dao_instance = data_src
    
    # setup the model
    if model_spec == '2PL':
        mod = model.IRT_MMLE_2PL(dao_instance, is_msg=is_msg, is_parallel=is_parallel)
    else:
        raise Exception('Unknown model specification.')

    # specify the irt parameters
    mod.set_options(theta_bnds, num_theta, alpha_bnds, beta_bnds,max_iter, tol)
    mod.set_guess_param(in_guess_param)

    # solve
    mod.solve_EM()

    # post
    item_param_dict = mod.get_item_param()

    if nargout ==1:
        return item_param_dict
    elif nargout ==2:
        user_param_dict = mod.get_user_param()
        return item_param_dict, user_param_dict
    else:
        raise Exception('Invalid number of argument')
