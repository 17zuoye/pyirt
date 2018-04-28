# -*-coding:utf-8-*-
from .solver import model
from .dao import localDAO
from .logger import Logger


def irt(data_src,
        dao_type='memory',
        theta_bnds=[-4, 4], num_theta=11,
        alpha_bnds=[0.25, 2], beta_bnds=[-2, 2], in_guess_param={},
        model_spec='2PL',
        max_iter=10, tol=1e-3, nargout=2,
        is_parallel=False, num_cpu=6, check_interval=60,
        mode='debug', log_path=None):

    # add logging
    logger = Logger.logger(log_path)

    # load data
    logger.info("start loading data")
    if dao_type == 'memory':
        dao_instance = localDAO(data_src, logger)
    elif dao_type == "db":
        dao_instance = data_src
    else:
        raise ValueError("dao type needs to be either memory or db")
    logger.info("data loaded")

    # setup the model
    if model_spec == '2PL':
        mod = model.IRT_MMLE_2PL(dao_instance,
                logger,
                dao_type=dao_type,
                is_parallel=is_parallel,
                num_cpu=num_cpu,
                check_interval=check_interval,
                mode=mode)
    else:
        raise Exception('Unknown model specification.')

    # specify the irt parameters
    mod.set_options(theta_bnds, num_theta, alpha_bnds, beta_bnds, max_iter, tol)
    mod.set_guess_param(in_guess_param)

    # solve
    mod.solve_EM()
    logger.info("parameter estimated")
    # output
    item_param_dict = mod.get_item_param()
    logger.info("parameter retrieved")

    if nargout == 1:
        return item_param_dict
    elif nargout == 2:
        user_param_dict = mod.get_user_param()
        return item_param_dict, user_param_dict
    else:
        raise Exception('Invalid number of argument')
