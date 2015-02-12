import numpy as np
from scipy.optimize import minimize



#TODO: The BFGS method is not as precise as the NM method
#TODO: There maybe overflowing issue in data
#TODO: Enable the calibration for two parameter estimation
np.seterr(over='raise')

class irt_2PL_Optimizer(object):

    def load_res_data(self, res_data):
        self.res_data = np.array(res_data)

    def set_theta(self, theta):
        self.theta = theta

    def set_bounds(self, bnds):
        self.bnds = bnds

    def set_initial_guess(self, x0):
        self.x0 = x0

    # generate the likelihood function
    @staticmethod
    def _likelihood(res_data, theta_vec, alpha, beta):
        #TODO: check the input to be two list
        num_data = len(res_data[0])
        # for MMLE method, y1 and y0 will be expected count
        y1 = res_data[0]
        y0 = res_data[1]
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
    def _gradient(res_data, theta_vec, alpha, beta):
        # res should be numpy array
        num_data = len(res_data[0])
        y1 = res_data[0]
        y0 = res_data[1]
        negExpComp_vec = [np.exp(beta + alpha * theta) for theta in theta_vec]
        temp_vec = [y1[i]-y0[i]*negExpComp_vec[i] for i in range(num_data)]
        beta_gradient_vec = [-temp_vec[i]/negExpComp_vec[i] for i in range(num_data)]
        alpha_gradient_vec = [-(theta_vec[i]*temp_vec[i])/negExpComp_vec[i] for i in range(num_data)]
        der = np.zeros(2)
        der[0] = sum(beta_gradient_vec)
        der[1] = sum(alpha_gradient_vec)
        return der


    def solve_param_linear(self, is_constrained):
        # for now, temp set alpha to 1
        def target_fnc(x):
            beta = x[0]
            alpha = x[1]
            return self._likelihood(self.res_data, self.theta, alpha, beta)
        if is_constrained:
            res = minimize(target_fnc, self.x0, method = 'SLSQP',
                        bounds=self.bnds, options={'disp':False})
        else:
            res = minimize(target_fnc, self.x0, method='nelder-mead',
                           options={'xtol':1e-8, 'disp':False})
        return res.x

    def solve_param_gradient(self, is_constrained):
        # for now, temp set alpha to 1

        def target_fnc(x):
            beta = x[0]
            alpha = x[1]
            return self._likelihood(self.res_data, self.theta, alpha, beta)

        def target_der(x):
            beta = x[0]
            alpha = x[1]
            return self._gradient(self.res_data, self.theta, alpha, beta)

        if is_constrained:
            res = minimize(target_fnc, self.x0, method = 'L-BFGS-B',
                        jac= target_der, bounds = self.bnds,
                        options={'disp':False})
        else:
            res = minimize(target_fnc, self.x0, method = 'BFGS',
                           jac=target_der,
                           options={'disp':False})

        return res.x
