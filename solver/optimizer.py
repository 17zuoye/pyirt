import numpy as np
from scipy.optimize import minimize



#TODO: The BFGS method is not as precise as the NM method
#TODO: There maybe overflowing issue in data
#TODO: Enable the calibration for two parameter estimation
np.seterr(over='raise')

class irt_2PL(object):
    alpha = 1.0

    def load_res_data(self, res_data):
        self.res_data = np.array(res_data)

    def setparam(self, theta):
        self.theta = theta

    def setBounds(self, bnds):
        self.bnds = bnds

    def setInitialGuess(self, x0):
        self.x0 = x0

    # generate the likelihood function
    @staticmethod
    def likelihood(res_data, theta_vec, alpha, beta):
        #TODO: check the input
        num_data = len(res_data)
        # figure out the number of right and wrong
        y1 = res_data
        y0 = 1.0-res_data
        expComp_vec = []
        for theta in theta_vec:
            try:
                expComp_vec.append(np.exp(-(alpha*theta+beta)))
            except:
                print alpha, theta, beta
                raise Exception('Numerical overflow')

           # expComp_vec = [np.exp(-(alpha*theta+beta)) for theta in theta_vec]
        likelihood_vec = [y1[i]* np.log(1+expComp_vec[i]) -
                          y0[i] * np.log(1-1.0/(1+expComp_vec[i]))
                          for i in range(num_data)]
        return sum(likelihood_vec)


    @staticmethod
    def gradient(res_data, theta_vec, alpha, beta):
        # res should be numpy array
        num_data = len(res_data)
        y1 = res_data
        y0 = 1.0 - y1
        negExpComp_vec = [np.exp(beta + alpha * theta) for theta in theta_vec]
        temp_vec = [y1[i]-y0[i]*negExpComp_vec[i] for i in range(num_data)]
        gradient_vec = [-temp_vec[i]/negExpComp_vec[i] for i in range(num_data)]

        der = np.zeros(1)
        #der[0] = -(theta*temp)/(negExpComp+1)
        der[0] = sum(gradient_vec)
        return der


    def solve_param_linear(self):
        # for now, temp set alpha to 1
        def target_fnc(beta):
            return self.likelihood(self.res_data, self.theta, self.alpha, beta)

        res = minimize(target_fnc, self.x0, method = 'SLSQP',
                       bounds=self.bnds, options={'disp':False})
        return res.x[0]

    def solve_param_gradient(self):
        # for now, temp set alpha to 1
        def target_fnc(beta):
            return self.likelihood(self.res_data, self.theta, self.alpha, beta)
        def target_der(beta):
            return self.gradient(self.res_data, self.theta, self.alpha, beta)
        res = minimize(target_fnc, self.x0, method = 'L-BFGS-B',
                       jac= target_der, bounds = self.bnds,
                       options={'disp':False})
        return res.x[0]
