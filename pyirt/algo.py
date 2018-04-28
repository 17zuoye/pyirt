# -*- coding:utf-8 -*-
from .util import clib, tools
import numpy as np


def update_theta_distribution(data, num_theta, theta_prior_val, theta_density, item_param_dict):
    '''
    data = [(item_idx int, ans_tag binary)]
    '''

    '''
    Basic Math.
        P_t(theta, data |q_param) = p(data|q_param, theta)*p_[t-1](theta)
        p_t(data|q_param) = sum(p_t(theta,data|q_param)) over theta
        p_t(theta|data, q_param) = P_t(theta, data|q_param)/p_t(data|q_param)
    '''
    likelihood_vec = np.zeros(num_theta)

    for k in range(num_theta):
        theta = theta_prior_val[k]
        ell = 0.0
        for log in data:
            item_idx = log[0]
            ans_tag = log[1]
            alpha = item_param_dict[item_idx]['alpha']
            beta = item_param_dict[item_idx]['beta']
            c = item_param_dict[item_idx]['c']
            ell += clib.log_likelihood_2PL(0.0 + ans_tag, 1.0 - ans_tag, theta, alpha, beta, c)
        likelihood_vec[k] = ell

    # posterior
    joint_llk_vec = likelihood_vec + np.log(theta_density)
    marginal = tools.logsum(joint_llk_vec)
    posterior = np.exp(joint_llk_vec - marginal)

    return posterior
