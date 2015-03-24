import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import utl



# TODO: The BFGS method is not as precise as the NM method
# TODO: There maybe overflowing issue in data
# TODO: Enable the calibration for two parameter estimation
np.seterr(over='raise')

class irt_2PL_Optimizer(object):

    def load_res_data(self, res_data):
        self.res_data = np.array(res_data)

    def set_theta(self, theta):
        self.theta = theta

    def set_c(self, c):
        self.c = c

    def set_bounds(self, bnds):
        self.bnds = bnds

    def set_initial_guess(self, x0):
        self.x0 = x0

    # generate the likelihood function
    @staticmethod
    def _likelihood(res_data, theta_vec, alpha, beta, c):
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
        likelihood_vec = [utl.clib.log_likelihood_2PL(y1[i],y0[i],theta_vec[i],
                                                     alpha, beta,c) \
                          for i in range(num_data)]
        # transform into negative likelihood
        ell = -sum(likelihood_vec)

        return ell

    @staticmethod
    def _gradient(res_data, theta_vec, alpha, beta, c):
        # res should be numpy array
        y1 = res_data[0]
        y0 = res_data[1]
        num_data = len(y1)

        der = np.zeros(2)
        for i in range(num_data):
            # the toolbox calculate the gradient of the log likelihood,
            # but the algorithm needs that of the negative ll
            der -= utl.clib.log_likelihood_2PL_gradient(y1[i],y0[i],theta_vec[i],alpha,beta, c)
        #grad = [ -utl.clib.log_likelihood_2PL_gradient(y1[i],y0[i],theta_vec[i],alpha,beta) for i in range(num_data)]
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
            return self._likelihood(self.res_data, self.theta, alpha, beta, self.c)

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
            return self._likelihood(self.res_data, self.theta, alpha, beta, self.c)

        def target_der(x):
            beta = x[0]
            alpha = x[1]
            return self._gradient(self.res_data, self.theta, alpha, beta, self.c)

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

class irt_factor_optimizer(object):
    def load_res_data(self, res_data):
        self.res_data = np.array(res_data)

    def set_item_parameter(self, alpha_vec, beta_vec, c_vec):
        if len(alpha_vec) != len(beta_vec):
            raise ValueError('The alpha vec and the beta vec does not match in length.')

        self.alpha_vec = alpha_vec
        self.beta_vec  = beta_vec
        self.c_vec = c_vec

    def set_bounds(self, bnds):
        self.bnds = bnds

    def set_initial_guess(self, x0):
        self.x0 = x0

    @staticmethod
    def _likelihood(res_data, theta, alpha_vec, beta_vec, c_vec):
        # for MMLE method, y1 and y0 will be expected count
        y1 = res_data[0]
        y0 = res_data[1]

        # check for equal length between y1,y0 and theta_vec
        num_data = len(y1)
        if len(y0) != num_data:
            raise ValueError('The response data does not match in length.')
        if len(alpha_vec) != num_data:
            raise ValueError('The response data does not match alpha vec in length.')
        if sum(y1<0)>0 or  sum(y0<0)>0:
            raise ValueError('y1 or y0 contains negative count.')
        # this is the likelihood
        likelihood_vec = [utl.clib.log_likelihood_2PL(y1[i],y0[i],theta,
                                                     alpha_vec[i], beta_vec[i],c_vec[i]) \
                          for i in range(num_data)]
        # transform into negative likelihood
        ell = -sum(likelihood_vec)

        return ell

    @staticmethod
    def _gradient(res_data, theta, alpha_vec, beta_vec, c_vec):
        # res should be numpy array
        y1 = res_data[0]
        y0 = res_data[1]
        num_data = len(y1)

        der = 0.0
        for i in range(num_data):
            der -= utl.tools.log_likelihood_factor_gradient(y1[i],y0[i],theta,alpha_vec[i],beta_vec[i], c_vec[i])
        return der

    @staticmethod
    def _hessian(res_data, theta, alpha_vec, beta_vec, c_vec):
        # res should be numpy array
        y1 = res_data[0]
        y0 = res_data[1]
        num_data = len(y1)

        hes = 0.0
        for i in range(num_data):
            hes -= utl.tools.log_likelihood_factor_hessian(y1[i],y0[i],theta,alpha_vec[i],beta_vec[i], c_vec[i])
        return hes


    def solve_param_linear(self, is_constrained):
        # for now, temp set alpha to 1
        def target_fnc(x):
            return self._likelihood(self.res_data, x, self.alpha_vec, self.beta_vec, self.c_vec)

        if is_constrained:
            res = minimize(target_fnc, self.x0, method = 'SLSQP',
                        bounds=self.bnds, options={'disp':False})
        else:
            res = minimize(target_fnc, self.x0, method='nelder-mead',
                           options={'xtol':1e-4, 'disp':False})

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
            return self._likelihood(self.res_data, x, self.alpha_vec, self.beta_vec, self.c_vec)
        def target_der(x):
            return self._gradient(self.res_data, x, self.alpha_vec, self.beta_vec, self.c_vec)

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

    def solve_param_hessian(self):
        def target_fnc(x):
            return self._likelihood(self.res_data, x, self.alpha_vec, self.beta_vec, self.c_vec)
        def target_der(x):
            return self._gradient(self.res_data, x, self.alpha_vec, self.beta_vec, self.c_vec)
        def target_hess(x):
            return self._hessian(self.res_data, x, self.alpha_vec, self.beta_vec, self.c_vec)

        res = minimize(target_fnc, self.x0, method='Newton-CG',
                jac=target_der, hess=target_hess,
                options={'xtol': 1e-8, 'disp': False})
        if not res.success:
            if res.message == 'Desired error not necessarily achieved due to precision loss.':
                # TODO:still returns a result. Something is wrong with the BFGS
                # though
                pass
            else:
                raise Exception('Algorithm failed, because ' + res.message)
        return res.x


    def solve_param_scalar(self):
        def target_fnc(x):
            return self._likelihood(self.res_data, x, self.alpha_vec, self.beta_vec, self.c_vec)
        res = minimize_scalar(target_fnc, bounds=self.bnds, method='bounded')
        return res.x
