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
    expComp = exp(-(alpha*theta + beta));
    ell =  y1*log(c+(1.0-c)/(1.0+expComp)) + y0*log((1.0-c)*expComp/(1.0+expComp)) ;

    return ell


def log_likelihood_2PL_gradient(double y1,
                                double y0,
                                double theta,
                                double alpha,
                                double beta,
                                double c=0.0):
    #TODO: could be organized into matrix
    # It is the gradient of the log likelihood, not the NEGATIVE log likelihood
    cdef extern from "math.h":
        double exp(double x)

    grad = np.zeros(2)

    temp = exp(beta + alpha * theta)
    beta_grad = 1.0/(1.0+temp) *( y1*( (1.0-c)/(c/temp+1.0))-
                                  y0*temp)

    #beta_grad = -(-y1+y0*temp)/(1+temp)
    alpha_grad = theta*beta_grad
    grad[0] = beta_grad
    grad[1] = alpha_grad
    return grad
