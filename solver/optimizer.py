import numpy as np
from scipy.optimize import minimize

#TODO: expand to multiple thetas

class irt_2PL(object):

    def load_res_data(self, res_data):
        self.res_data = res_data

    def setparam(self, theta):
        self.theta = theta

    @staticmethod
    def irt_fnc(theta, beta, alpha=1.0, c=0.0):
        # beta is item difficulty
        # theta is respondent capability

        #if not np.issubdtype(theta, float):
        #    raise Exception('Theta is not float data')
        #if not np.issubdtype(beta, float):
        #    raise Exception('Beta is not float data')

        prob = (1.0-c) / (1 + np.exp(-alpha*(theta-beta)))
        return prob

    # generate the likelihood function
    @staticmethod
    def likelihood(res, theta, alpha, beta):
        #TODO: check the input

        # figure out the number of right and wrong
        num_right = sum(res)
        num_wrong = len(res) - num_right
        expComp = np.exp(-(alpha*theta+beta))
        l =  num_right * np.log(1+expComp) - num_wrong * np.log(1-1.0/(1+expComp))
        return l

    @staticmethod
    def gradient(res, theta, alpha, beta):
        # res should be integers
        y1 = np.array(res)
        y0 = 1.0 - y1
        negExpComp = np.exp(beta + alpha * theta)
        temp = sum(y1-y0*negExpComp)
        der = np.zeros(1)
        #der[0] = -(theta*temp)/(negExpComp+1)
        der[0] = -temp/(negExpComp+1)
        return der


    def solve_beta_direct(self):
        x0 = 4.0
        alpha = 1.0
        # for now, temp set alpha to 1
        def target_fnc(beta):
            return self.likelihood(self.res_data, self.theta, alpha, beta)

        res = minimize(target_fnc,x0, method = 'BFGS',options={'xtol':1e-8, 'disp':True})
        self.x = res.x

    def solve_beta_gradient(self):
        x0 = 4.0
        alpha = 1.0
        # for now, temp set alpha to 1
        def target_fnc(beta):
            return self.likelihood(self.res_data, self.theta, alpha, beta)
        def target_der(beta):
            return self.gradient(self.res_data, self.theta, alpha, beta)

        res = minimize(target_fnc,x0, method = 'BFGS', jac= target_der, options={'xtol':1e-8, 'disp':True})
        self.x = res.x



