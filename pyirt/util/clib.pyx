# -*-coding:utf-8-*-

import cython
import numpy as np

def log_likelihood_2PL(double y1,
                       double y0,
                       double theta,
                       double alpha,
                       double beta,
                       double c=0.0):
    cdef extern from "math.h":
        double exp(double x)
        double log(double x)
    expPos = exp(alpha*theta + beta) ;
    ell =  y1*log((c+expPos)/(1.0+expPos)) + y0*log((1.0-c)/(1.0+expPos)) ;

    return ell


def log_likelihood_2PL_gradient(double y1,
                                double y0,
                                double theta,
                                double alpha,
                                double beta,
                                double c=0.0):
    # It is the gradient of the log likelihood, not the NEGATIVE log likelihood
    grad = np.zeros(2)

    temp = exp(beta + alpha * theta)
    beta_grad = temp /(1.0+temp) *( y1*(1.0-c)/(c+temp)- y0)

    #beta_grad = -(-y1+y0*temp)/(1+temp)
    alpha_grad = theta*beta_grad
    grad[0] = beta_grad
    grad[1] = alpha_grad
    return grad
