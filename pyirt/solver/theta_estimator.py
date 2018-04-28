import numpy as np
from scipy.stats import beta


from ..util import clib, tools

from ..solver import optimizer


class bayesian_estimator(object):

    def set_prior(self, theta_min, theta_max, num_theta, dist_name):
        self.theta_val = np.linspace(theta_min, theta_max, num=num_theta)
        self.num_theta = num_theta
        if dist_name == 'uniform':
            # the density is uniform
            self.theta_density = np.ones(num_theta) / num_theta

        elif dist_name == 'beta':
            # centered beta
            # rescale to move away from the boundary
            self.theta_density = beta.pdf((self.theta_val - theta_min) / (theta_max - theta_min + 0.1), 2, 2)
            # renormalize
            self.theta_density = self.theta_density / sum(self.theta_density)
        else:
            raise Exception('Unknown prior distribution.')

    def update(self, logs):
        # data comes in as
        # tag(0/1), (a, b,c)

        likelihood_vec = np.zeros(self.num_theta)
        # calculate
        for k in range(self.num_theta):
            theta = self.theta_val[k]
            # calculate the likelihood
            ell = 0.0
            for log in logs:
                atag = log[0]
                alpha = log[1][0]
                beta = log[1][1]
                c = log[1][2]
                ell += clib.log_likelihood_2PL(atag, 1.0 - atag, theta, alpha, beta, c)
            # now update the density
            likelihood_vec[k] = ell

        # ell  = p(param|x), full joint = logp(param|x)+log(x)
        # Fix np.log, see http://stackoverflow.com/questions/13497891/python-getting-around-division-by-zero
        log_joint_prob_vec = likelihood_vec + np.log(self.theta_density.clip(min=0.0000000001))
        # calculate the posterior
        # p(x|param) = exp(logp(param,x) - log(sum p(param,x)))
        marginal = tools.logsum(log_joint_prob_vec)
        self.theta_density = np.exp(log_joint_prob_vec - marginal)

    def get_estimator(self):

        # expected value
        theta_mean = np.dot(self.theta_density, self.theta_val)
        # theta_var = np.dot(self.theta_density, self.theta_val**2) - theta_mean**2

        theta_hat = theta_mean
        return theta_hat


class MLE_estimator(object):
    worker = optimizer.irt_factor_optimizer()

    def update(self, logs):
        # log [tag(0/1), (a, b,c)]

        # transform the logs
        y1 = []
        y0 = []
        alphas = []
        betas = []
        cs = []
        for log in logs:
            y1.append(log[0])
            y0.append(1.0 - log[0])
            alphas.append(log[1][0])
            betas.append(log[1][1])
            cs.append(log[1][2])
        self.worker.load_res_data([y1, y0])
        self.worker.set_item_parameter(alphas, betas, cs)
        self.worker.set_bounds([(-4.0, 4.0)])
        self.worker.set_initial_guess(0.0)
        try:
            est_theta = self.worker.solve_param_gradient(is_constrained=True)
        except Exception as e:
            est_theta = self.worker.solve_param_linear(is_constrained=True)

        # the output is an numpy array!
        return est_theta[0]
