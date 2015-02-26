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
        # this is the likelihood
        likelihood_vec = [utl.tools.log_likelihood_2PL(y1[i],y0[i],theta_vec[i],
                                                     alpha, beta) \
                          for i in range(num_data)]
        # transform into negative likelihood
        ell = -sum(likelihood_vec)

        return ell

    @staticmethod
    def _gradient(res_data, theta_vec, alpha, beta):
        # res should be numpy array
        y1 = res_data[0]
        y0 = res_data[1]
        num_data = len(y1)


        der = np.zeros(2)
        for i in range(num_data):
            # the toolbox calculate the gradient of the log likelihood,
            # but the algorithm needs that of the negative ll
            der -= utl.tools.log_likelihood_2PL_gradient(y1[i],y0[i],theta_vec[i],alpha,beta)
        #grad = [ -utl.tools.log_likelihood_2PL_gradient(y1[i],y0[i],theta_vec[i],alpha,beta) for i in range(num_data)]
        #TODO: This is actually a bit of cheating
        #if abs(der[0]) >50 or abs(der[1])>50:
        #    ratio = max(abs(der/50))
        #    der = der/ratio
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
                           options={'xtol':1e-3, 'disp':False})

        # deal with expcetions
        if not res.success:
            if not is_constrained and \
                    res.message == 'Maximum number of function evaluations has been exceeded.':
                raise Exception('Optimizer fails to find solution. Try constrained search.')
            else:
                raise Exception('Algorithm failed because '+ res.message)

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
