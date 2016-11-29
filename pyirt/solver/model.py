'''
The model is an implementation of EM algorithm of IRT


For reference, see:
Brad Hanson, IRT Parameter Estimation using the EM Algorithm, 2000

The current version only deals with unidimension theta

'''
import numpy as np
import time

from ..utl import clib, tools, loader
from ..solver import optimizer
import io
from six import string_types


class IRT_MMLE_2PL(object):

    '''
    Three steps are exposed
    (1) load data
    (2) set parameter
    (3) solve
    '''

    def load_data(self, src):
        # three columns are user_id, item_id, ans_tag
        if isinstance(src, io.IOBase):
            # if the src is file handle
            user_ids, item_ids, ans_tags = self._loadFromHandle(src)
        else:
            # if the src is list of tuples
            user_ids, item_ids, ans_tags = self._loadFromTuples(src)
       
        #  map the arbitrary inputs to continuous idx which used in later idx
        user_id_idx_vec, self.user_idx_ref, self.user_reverse_idx_ref = loader.construct_ref_dict(user_ids) 
        item_id_idx_vec, self.item_idx_ref, self.item_reverse_idx_ref = loader.construct_ref_dict(item_ids)
        
        
        # process it
        print('Data loading is complete.')

        self.data_ref = loader.data_storage()
        self.data_ref.setup(user_id_idx_vec, item_id_idx_vec, ans_tags)

    def load_param(self, theta_bnds, alpha_bnds, beta_bnds,max_iter, tol):
        # TODO: allow for a more flexible parameter setting
        # The config object has to be passed in because hdfs file system does
        # not load target file

        # load user item
        num_theta = 11
        self._init_user_param(theta_bnds[0], theta_bnds[1], num_theta)

        # load the solver
        boundary = {'alpha': alpha_bnds,
                    'beta': beta_bnds}

        solver_type = 'gradient'
        is_constrained = True

        self._init_solver_param(is_constrained, boundary, solver_type, max_iter, tol)

    def load_guess_param(self, in_guess_param):
        self.guess_param_dict = {}
        if isinstance(in_guess_param, string_types):
            # all c are 0
            for item_idx in range(self.data_ref.num_item):
                self.guess_param_dict[item_idx] = {'c': 0.0}
        else:
            for item_idx in range(self.data_ref.num_item):
                item_id = self.item_reverse_idx_ref[item_idx]
                self.guess_param_dict[item_idx] = in_guess_param[item_id]

    def solve_EM(self):
        # create the inner parameters
        # currently item parameter requires no setup
        self._init_item_param()

        self.posterior_theta_distr = np.zeros((self.data_ref.num_user, self.num_theta))

        # TODO: enable the stopping condition
        num_iter = 1
        self.ell_list = []
        avg_prob_t0 = 0

        while True:
            # save the iterations from last time
            if num_iter!=1:
                last_theta_vec = self.theta_vec
                last_item_param_dict = self.item_param_dict
            
            
            iter_start_time = time.time()
            # add in time block
            start_time = time.time()
            self._exp_step()
            print("--- E step: %f secs ---" % np.round((time.time() - start_time)))

            start_time = time.time()
            self._max_step()
            print("--- M step: %f secs ---" % np.round((time.time() - start_time)))

            self.__calc_theta()

            '''
            Exp
            '''
            # self.update_guess_param()

            # the goal is to maximize the "average" probability
            avg_prob = np.exp(self.__calc_data_likelihood() / self.data_ref.num_log)
            self.ell_list.append(avg_prob)
            print("--- all: %f secs ---" % np.round((time.time() - iter_start_time)))
            print(avg_prob)

            # if the algorithm improves, then ell > ell_t0
            if avg_prob_t0 > avg_prob:
                self.theta_vec = last_theta_vec
                self.item_param_dict = last_item_param_dict
                print('Likelihood descrease, stops at iteration %d.' % num_iter)
                break

            if avg_prob_t0 < avg_prob and avg_prob - avg_prob_t0 <= self.tol:
                print('EM converged at iteration %d.' % num_iter)
                break
            # update the stop condition
            avg_prob_t0 = avg_prob
            num_iter += 1

            if (num_iter > self.max_iter):
                print('EM does not converge within max iteration')
                break

    def get_item_param(self):
        output_item_param = {}
        for item_idx in range(self.data_ref.num_item):
            item_id = self.item_reverse_idx_ref[item_idx]
            output_item_param[item_id] = self.item_param_dict[item_idx]  
        return output_item_param

    def get_user_param(self):
        output_user_param = {}
        for user_idx in range(self.data_ref.num_user):
            user_id = self.user_reverse_idx_ref[user_idx]
            output_user_param[user_id] = self.theta_vec[user_idx]
        return output_user_param

    '''
    Main Routine
    '''

    def _exp_step(self):
        '''
        Basic Math:
        In the maximization step, need to use E_[j,k](Y=1),E_[j,k](Y=0)
        E(Y=1|param_j,theta_k) = sum_i(data_[i,j]*P(Y=1|param_j,theta_[i,k]))
        since data_[i,j] = 0/1, it is equivalent to sum all done right users

        E(Y=0|param_j,theta_k) = sum_i(
                                (1-data_[i,j]) *(1-P(Y=1|param_j,theta_[i,k])
                                    )
        By similar logic, it is equivalent to sum (1-p) for all done wrong users

        '''

        # (1) update the posterior distribution of theta
        self.__update_theta_distr()

        # (2) marginalize
        # because of the sparsity, the expected right and wrong may not sum up
        # to the total num of items!
        self.__get_expect_count()

    def _max_step(self):
        '''
        Basic Math
            log likelihood(param_j) = sum_k(log likelihood(param_j, theta_k))
        '''
        # [A] max for item parameter
        opt_worker = optimizer.irt_2PL_Optimizer()
        # the boundary is universal
        # the boundary is set regardless of the constrained option because the
        # constrained search serves as backup for outlier cases
        opt_worker.set_bounds([self.beta_bnds, self.alpha_bnds])

        # theta value is universal
        opt_worker.set_theta(self.theta_prior_val)

        for item_idx in range(self.data_ref.num_item):
            # set the initial guess as a mixture of current value and a new
            # start to avoid trap in local maximum
            initial_guess_val = (self.item_param_dict[item_idx]['beta'],
                                 self.item_param_dict[item_idx]['alpha'])

            opt_worker.set_initial_guess(initial_guess_val)
            opt_worker.set_c(self.item_param_dict[item_idx]['c'])

            # assemble the expected data
            expected_right_count = self.item_expected_right_by_theta[:, item_idx]
            expected_wrong_count = self.item_expected_wrong_by_theta[:, item_idx]
            input_data = [expected_right_count, expected_wrong_count]
            opt_worker.load_res_data(input_data)
            # if one wishes to inspect the model input, print the input data

            est_param = opt_worker.solve_param_mix(self.is_constrained)

            # update
            self.item_param_dict[item_idx]['beta'] = est_param[0]
            self.item_param_dict[item_idx]['alpha'] = est_param[1]

        # [B] max for theta density
        # pi = r_k/(w_k+r_k)
        r_vec = np.sum(self.item_expected_right_by_theta, axis=1)
        w_vec = np.sum(self.item_expected_wrong_by_theta, axis=1)
        self.theta_density = np.divide(r_vec, r_vec + w_vec)

    '''
    Auxuliary function
    '''

    def _loadFromTuples(self, data):
        user_ids = []
        item_ids = []
        ans_tags = []
        if len(data) == 0:
            raise Exception('Data is empty.')

        for log in data:
            user_ids.append(log[0])
            item_ids.append(log[1])
            ans_tags.append(int(log[2]))

        return user_ids, item_ids, ans_tags

    def _loadFromHandle(self, fp, sep=','):
        # Default format is comma separated files,
        # Only int is allowed within the environment
        user_ids = []
        item_ids = []
        ans_tags = []

        for line in fp:
            if line == '':
                continue
            user_id_str, item_id_str, ans_tagstr = line.strip().split(sep)
            user_ids.append(user_id_str)
            item_ids.append(item_id_str)
            ans_tags.append(int(ans_tagstr))
        return user_ids, item_ids, ans_tags

    def _init_solver_param(self, is_constrained, boundary,
                           solver_type, max_iter, tol):
        # initialize bounds
        self.is_constrained = is_constrained
        self.alpha_bnds = boundary['alpha']
        self.beta_bnds = boundary['beta']
        self.solver_type = solver_type
        self.max_iter = max_iter
        self.tol = tol

        if solver_type == 'gradient' and not is_constrained:
            raise Exception('BFGS has to be constrained')

    def _init_item_param(self):
        self.item_param_dict = {}
        for item_idx in range(self.data_ref.num_item):
            # need to call the old item_id
            c = self.guess_param_dict[item_idx]['c']
            self.item_param_dict[item_idx] = {'alpha': 1.0, 'beta': 0.0, 'c': c}

    def _init_user_param(self, theta_min, theta_max, num_theta):
        self.theta_prior_val = np.linspace(theta_min, theta_max, num=num_theta)
        self.num_theta = len(self.theta_prior_val)
        if self.num_theta != num_theta:
            raise Exception('Theta initialization failed')
        # store the prior density
        self.theta_density = np.ones(num_theta) / num_theta

    def __update_theta_distr(self):

        def update(log_list, num_theta, theta_prior_val, theta_density, item_param_dict):
            '''
            Basic Math. Notice that the distribution is user specific
                P_t(theta,data_i,param) = p(data_i,param|theta)*p_[t-1](theta)
                p_t(data_i,param) = sum(p_t(theta,data_i,param)) over theta
                p_t(theta|data_i,param) = P_t(theta,data_i,param)/p_t(data_i,param)
            '''
            # find all the items
            likelihood_vec = np.zeros(num_theta)
            # calculate
            for k in range(num_theta):
                theta     = theta_prior_val[k]
                # calculate the likelihood
                ell       = 0.0
                for log in log_list:
                    item_idx   = log[0]
                    ans_tag  = log[1]
                    alpha = item_param_dict[item_idx]['alpha']
                    beta  = item_param_dict[item_idx]['beta']
                    c     = item_param_dict[item_idx]['c']
                    ell   += clib.log_likelihood_2PL(0.0+ans_tag, 1.0 - ans_tag,
                                                     theta, alpha, beta, c)

                # now update the density
                likelihood_vec[k] = ell
            # ell  = log(p(y|theta,param))
            # full joint|param = log(p(y|theta,param))+log(p(theta))
            log_joint_prob_vec = likelihood_vec + np.log(theta_density)

            # calculate the posterior
            # p(theta|y,param) = exp(logp(y,theta|param) - log(sum p(y,theta|param)))
            marginal = tools.logsum(log_joint_prob_vec)
            posterior = np.exp(log_joint_prob_vec - marginal)
            return posterior


        # [A] calculate p(data,param|theta)
        # TODO: speed it up
        for user_idx in range(self.data_ref.num_user):
            self.posterior_theta_distr[user_idx, :] = update(self.data_ref.get_log(user_idx),
                                                      self.num_theta, self.theta_prior_val, self.theta_density,
                                                      self.item_param_dict)
        '''
        # create temporay variable for the loops
        ntheta = self.num_theta
        theta_prior = self.theta_prior_val
        theta_density = self.theta_density
        item_param = self.item_param_dict
        num_user = self.data_ref.num_user
        logs = [self.data_ref.get_log(self.data_ref.user_id_vec[i]) for i in range(num_user)]
        posterior_vec = parallel_update(logs, ntheta, theta_prior, theta_density, item_param, num_user)

        for i in range(self.data_ref.num_user):
            self.posterior_theta_distr[i,:] = np.exp(posterior_vec[i])
        '''
        # When the loop finish, check if the theta_density adds up to unity for each user
        check_user_distr_marginal = np.sum(self.posterior_theta_distr, axis=1)
        if any(abs(check_user_distr_marginal - 1.0) > 0.0001):
            raise Exception('The posterior distribution of user ability is not proper')

    def __get_expect_count(self):

        self.item_expected_right_by_theta = np.zeros((self.num_theta, self.data_ref.num_item))
        self.item_expected_wrong_by_theta = np.zeros((self.num_theta, self.data_ref.num_item))

        for item_idx in range(self.data_ref.num_item):
            right_user_idx_vec, wrong_user_idx_vec = self.data_ref.get_rwmap(item_idx)
            # condition on the posterior ability, what is the expected count of
            # students get it right
            # TODO: for readability, should specify the rows and columns
            self.item_expected_right_by_theta[:, item_idx] = np.sum(self.posterior_theta_distr[right_user_idx_vec, :], axis=0)
            self.item_expected_wrong_by_theta[:, item_idx] = np.sum(self.posterior_theta_distr[wrong_user_idx_vec, :], axis=0)

    def __calc_data_likelihood(self):
        # calculate the likelihood for the data set

        ell = 0
        for user_idx in range(self.data_ref.num_user):
            theta = self.theta_vec[user_idx]
            # find all the item_id
            logs = self.data_ref.get_log(user_idx)
            for log in logs:
                item_idx = log[0]
                ans_tag = log[1]
                alpha = self.item_param_dict[item_idx]['alpha']
                beta = self.item_param_dict[item_idx]['beta']
                c = self.item_param_dict[item_idx]['c']

                ell += clib.log_likelihood_2PL(0.0+ans_tag, 1.0-ans_tag,
                                               theta, alpha, beta, c)
        return ell

    def __calc_theta(self):
        self.theta_vec = np.dot(self.posterior_theta_distr, self.theta_prior_val)

