import numpy as np

def log_likelihood_2PL(y1, y0, theta, alpha, beta, c=0.0):

    expComp = np.exp(-(alpha*theta + beta));
    ell =  y1*np.log(c+(1.0-c)/(1.0+expComp)) + y0*np.log((1.0-c)*expComp/(1.0+expComp)) ;

    return ell


def log_likelihood_2PL_gradient(y1, y0, theta, alpha, beta, c=0.0):
    #TODO: could be organized into matrix
    # It is the gradient of the log likelihood, not the NEGATIVE log likelihood
    grad = np.zeros(2)

    temp = np.exp(beta + alpha * theta)
    beta_grad = 1.0/(1.0+temp) *( y1*( (1.0-c)/(c/temp+1.0))-
                                  y0*temp)

    #beta_grad = -(-y1+y0*temp)/(1+temp)
    alpha_grad = theta*beta_grad
    grad[0] = beta_grad
    grad[1] = alpha_grad
    return grad
