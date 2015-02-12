import numpy as np
from scipy.optimize import minimize

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import utl


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
        # for MMLE method, y1 and y0 will be expected count
        y1 = res_data[0]
        y0 = res_data[1]

        # check for equal length between y1,y0 and theta_vec
        num_data = len(y1)
        if len(y0) != num_data:
            raise ValueError('The response data does not match in length.')
        if len(theta_vec) != num_data:
            raise ValueError('The response data does not match theta vec in length.')

        if sum(y1<0)>0 or  sum(y0<0)>0:
            raise ValueError('y1 or y0 contains negative count.')

        likelihood_vec = [utl.tools.log_likelihood_2PL(y1[i],y0[i],theta_vec[i],
                                                     alpha, beta) \
                          for i in range(num_data)]
        ell = sum(likelihood_vec)

        return ell

    @staticmethod
    def _gradient(res_data, theta_vec, alpha, beta):
        # res should be numpy array
        y1 = res_data[0]
        y0 = res_data[1]
        num_data = len(y1)

        negExpComp_vec = [np.exp(beta + alpha * theta) for theta in theta_vec]

        temp_vec = [y1[i]-y0[i]*negExpComp_vec[i] for i in range(num_data)]

        beta_gradient_vec = [-temp_vec[i]/(1+negExpComp_vec[i]) for i in range(num_data)]

        alpha_gradient_vec = [theta_vec[i]*beta_gradient_vec[i] for i in range(num_data)]

        der = np.zeros(2)
        der[0] = sum(beta_gradient_vec)
        der[1] = sum(alpha_gradient_vec)
        #TODO: This is actually a bit of cheating
        if abs(der[0]) >50 or abs(der[1])>50:
            ratio = max(der/50)
            der = der/ratio
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
                           options={'xtol':1e-4, 'disp':False})
        #if not res.success:
            #import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
            #raise Exception('Algorithm failed, because '+ res.message)
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

        if not res.success:
            raise Exception('Algorithm failed.')

        return res.x
