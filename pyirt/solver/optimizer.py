# -*-coding:utf-8-*-
import numpy as np
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

from ..utl import clib, tools

# TODO: The BFGS method is not as precise as the NM method
# TODO: There maybe overflowing issue in data
# TODO: Enable the calibration for two parameter estimation
np.seterr(over='raise')


#####################################################################
# new multi-purpose solver
class Mirt_Optimizer(object):
    def load_res_data(self, Ys, Xs, J=None, K=None):
        # Ys are N*1, Xs are N*K
        # Ys and Xs are lists
        N = len(Ys)
        if len(Xs) != N:
            raise ValueError('Xs and Ys are not equal length')
        if not J:
            J = len(set(Ys))
        if not K:
            K = len(Xs[0])

        self.exog = np.array(Xs)
        self.K = K
        self.J = J
        
        # transfrom into wendog
        self.wendog = np.zeros((N,J))
        for i in range(N):
            self.wendog[i,Ys[i]] = 1

    def solve_param(self, x0):
        target_fnc = lambda params: -self._loglikelihood(params, self.wendog, self.exog, self.K)
        res = minimize(target_fnc,
                x0, method='Newton-CG',
                jac = lambda params: -self._score(params, self.wendog, self.exog, self.K),
                hess = lambda params: -self._hessian(params, self.wendog, self.exog, self.K)
                )
        new_params = res.x.reshape(self.K, self.J-1, order='F')
        return new_params

    def _cdf(self, X):
        # X has to be a list or numpy array
        eXb = np.column_stack((np.ones(len(X)),np.exp(X)))
        return eXb/eXb.sum(1)[:,None]

    def _loglikelihood(self, params, wendog,exog,K):
        params = params.reshape(K,-1,order='F')
        logprob = np.log(self._cdf(np.dot(exog,params)))
        return np.sum(wendog*logprob)

    # TODO: bugs: cannot handle single X wihtout constant
    def _score(self,params, wendog, exog, K):
        params = params.reshape(K,-1,order='F')
        first_term = wendog[:,1:] - self._cdf(np.dot(exog,params))[:,1:]
        g =  np.dot(first_term.T, exog).flatten()
        return g

    def _hessian(self,params, wendog, exog, K):
        params = params.reshape(K,-1,order='F')
        pr = self._cdf(np.dot(exog,params))
        partials = []
        J = wendog.shape[1] - 1  # first defaults to be 1
        K = exog.shape[1]
        for i in range(J):
            for j in range(J):
                partials.append( -np.dot( ( (pr[:,i+1]*(int(i==j)-pr[:,j+1]))[:,None]*exog).T, exog ) )
        H = np.array(partials)
        H = np.transpose(H.reshape(J,J,K,K),(0,2,1,3)).reshape(J*K,J*K)
        return H

    




#####################################################################
# old solvers

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
            raise ValueError('The response data does not match in length. y0:%s, y1:%s' % (y0, y1))
        if len(theta_vec) != num_data:
            raise ValueError('The response data does not match theta vec in length. theta_vec:%s, num_data:%s' % (theta_vec, num_data))

        if sum(y1 < 0) > 0 or sum(y0 < 0) > 0:
            raise ValueError('y1 or y0 contains negative count.')
        # this is the likelihood
        likelihood_vec = [clib.log_likelihood_2PL(y1[i], y0[i], theta_vec[i],
                                                  alpha, beta, c)
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
            der -= clib.log_likelihood_2PL_gradient(y1[i], y0[i], theta_vec[i], alpha, beta, c)
        return der

    def solve_param_linear(self, is_constrained):
        # for now, temp set alpha to 1
        def target_fnc(x):
            beta = x[0]
            alpha = x[1]
            return self._likelihood(self.res_data, self.theta, alpha, beta, self.c)

        if is_constrained:
            res = minimize(target_fnc, self.x0, method='SLSQP',
                           bounds=self.bnds, options={'disp': False})
        else:
            res = minimize(target_fnc, self.x0, method='nelder-mead',
                           options={'xtol': 1e-3, 'disp': False})

        # deal with expcetions
        if not res.success:
            if not is_constrained and \
                    res.message == 'Maximum number of function evaluations\
                    has been exceeded.':
                raise Exception('Optimizer fails to find solution.\
                                Try constrained search.')
            else:
                raise Exception('Algorithm failed because: ' + res.message)

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
            res = minimize(target_fnc, self.x0, method='L-BFGS-B',
                           jac=target_der, bounds=self.bnds,
                           options={'disp': False})
        else:
            res = minimize(target_fnc, self.x0, method='BFGS',
                           jac=target_der,
                           options={'disp': False})

        if not res.success:
            raise Exception("Algorithm failed because " + res.message)

        return res.x

    def solve_param_mix(self, is_constrained=True):
        """
        Mix solve_param_gradient and solve_param_linear.
        """
        # solve by L-BFGS-B
        # * linear is more robust than gradient.
        try:
            est_param = self.solve_param_gradient(is_constrained)
        except:
            # if the alogrithm is nelder-mead and the optimization fails to
            # converge, use the constrained version
            #
            # * solve_param_linear with different params in two times.
            try:
                est_param = self.solve_param_linear(is_constrained)
            except Exception as e:
                if str(e) == 'Optimizer fails to find solution. Try constrained search.':
                    est_param = self.solve_param_linear(True)
                else:
                    raise e
        return est_param


class irt_factor_optimizer(object):

    def load_res_data(self, res_data):
        self.res_data = np.array(res_data)

    def set_item_parameter(self, alpha_vec, beta_vec, c_vec):
        if len(alpha_vec) != len(beta_vec):
            raise ValueError('The alpha vec and the beta vec does not match in length.')

        self.alpha_vec = alpha_vec
        self.beta_vec = beta_vec
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
        if sum(y1 < 0) > 0 or sum(y0 < 0) > 0:
            raise ValueError('y1 or y0 contains negative count.')
        # this is the likelihood
        likelihood_vec = [clib.log_likelihood_2PL(y1[i], y0[i], theta,
                                                  alpha_vec[i], beta_vec[i], c_vec[i])
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
            der -= tools.log_likelihood_factor_gradient(y1[i], y0[i], theta, alpha_vec[i], beta_vec[i], c_vec[i])
        return der

    @staticmethod
    def _hessian(res_data, theta, alpha_vec, beta_vec, c_vec):
        # res should be numpy array
        y1 = res_data[0]
        y0 = res_data[1]
        num_data = len(y1)

        hes = 0.0
        for i in range(num_data):
            hes -= tools.log_likelihood_factor_hessian(y1[i], y0[i], theta, alpha_vec[i], beta_vec[i], c_vec[i])
        return hes

    def solve_param_linear(self, is_constrained):
        # for now, temp set alpha to 1
        def target_fnc(x):
            return self._likelihood(self.res_data, x, self.alpha_vec, self.beta_vec, self.c_vec)

        if is_constrained:
            res = minimize(target_fnc, self.x0, method='SLSQP',
                           bounds=self.bnds, options={'disp': False})
        else:
            res = minimize(target_fnc, self.x0, method='nelder-mead',
                           options={'xtol': 1e-4, 'disp': False})

        # deal with expcetions
        if not res.success:
            if not is_constrained and \
                    res.message == 'Maximum number of function evaluations has been exceeded.':
                raise Exception('Optimizer fails to find solution. Try constrained search.')
            else:
                raise Exception('Algorithm failed because ' + res.message)

        return res.x

    def solve_param_gradient(self, is_constrained):
        # for now, temp set alpha to 1
        def target_fnc(x):
            return self._likelihood(self.res_data, x, self.alpha_vec, self.beta_vec, self.c_vec)

        def target_der(x):
            return self._gradient(self.res_data, x, self.alpha_vec, self.beta_vec, self.c_vec)

        if is_constrained:
            res = minimize(target_fnc, self.x0, method='L-BFGS-B',
                           jac=target_der, bounds=self.bnds,
                           options={'disp': False})
        else:
            res = minimize(target_fnc, self.x0, method='BFGS',
                           jac=target_der,
                           options={'disp': False})

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
