# -*- coding:utf-8 -*-

'''
Created on 2015/1/2

@author: junchen
'''
import numpy as np

def irt_fnc(theta, beta, alpha=1.0, c=0.0):
    # beta is item difficulty
    # theta is respondent capability

    prob = c+ (1.0-c) / (1 + np.exp(-(alpha*theta+beta)))
    return prob




def log_likelihood_factor_gradient(y1, y0, theta, alpha, beta, c=0.0):
    temp = np.exp(beta + alpha * theta)
    grad = alpha/(1.0+temp) *( y1*( (1.0-c)/(c/temp+1.0))-
                                  y0*temp)

    return grad

def log_likelihood_factor_hessian(y1, y0, theta, alpha, beta, c=0.0):
    x = np.exp(beta + alpha * theta)
    #hessian = - alpha**2*(y1+y0)*temp/(1+temp)**2
    hessian  =  alpha**2*x/(1+x)**2*( y1*(1-c)*(c-x**2)/(c+x)**2 -y0)

    return hessian


def logsum(logp):
    w = max(logp)
    logSump = w+ np.log(sum(np.exp(logp-w)))
    return logSump


