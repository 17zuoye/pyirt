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

def log_likelihood_2PL(y1, theta, alpha, beta):
    expComp = np.exp(-(alpha*theta + beta));
    ell =  y1*np.log(1.0/(1.0+expComp)) + (1.0-y1)*np.log(1.0-1.0/(1.0+expComp)) ;
    return ell;

def logsum(logp):
    w = max(logp)
    logSump = w+ np.log(sum(np.exp(logp-w)))
    return logSump


def parse_item_paramer(item_param_dict):
    for eid, param in item_param_dict.iteritems():
        print eid, np.round(param['alpha'],decimals=2), np.round(param['beta'],decimals=2)


