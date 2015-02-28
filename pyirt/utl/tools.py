# -*- coding:utf-8 -*-

'''
Created on 2015/1/2

@author: junchen
'''
import numpy as np

def irt_fnc(theta, beta, alpha=1.0, c=0.0):
    # beta is item difficulty
    # theta is respondent capability

    prob = (1.0-c) / (1 + np.exp(-(alpha*theta+beta)))
    return prob

def log_likelihood_2PL(y1, y0, theta, alpha, beta):

    expComp = np.exp(-(alpha*theta + beta));

    ell =  y1*np.log(1.0/(1.0+expComp)) + y0*np.log(1.0-1.0/(1.0+expComp)) ;

    return ell

def log_likelihood_2PL_gradient(y1, y0, theta, alpha, beta):
    #TODO: could be organized into matrix
    # It is the gradient of the log likelihood, not the NEGATIVE log likelihood
    grad = np.zeros(2)

    temp = np.exp(beta + alpha * theta)
    beta_grad = -(-y1+y0*temp)/(1+temp)
    alpha_grad = theta*beta_grad
    grad[0] = beta_grad
    grad[1] = alpha_grad
    return grad

def log_likelihood_factor_gradient(y1, y0, theta, alpha, beta):

    temp = np.exp(beta + alpha * theta)
    grad = -alpha*(-y1+y0*temp)/(1+temp)

    return grad

def log_likelihood_factor_hessian(y1, y0, theta, alpha, beta):
    temp = np.exp(beta + alpha * theta)
    hessian = - alpha**2*(y1+y0)*temp/(1+temp)**2
    return hessian


def logsum(logp):
    w = max(logp)
    logSump = w+ np.log(sum(np.exp(logp-w)))
    return logSump


def parse_item_paramer(item_param_dict, output_file = None):

    if output_file is not None:
        # open the file
        out_fh = open(output_file,'w')

    for eid, param in item_param_dict.iteritems():
        alpha_val = np.round(param['alpha'],decimals=2)
        beta_val = np.round(param['beta'],decimals=2)
        if output_file is None:
            print eid, alpha_val, beta_val
        else:
            out_fh.write('{},{},{}\n'.format(eid, alpha_val, beta_val))


