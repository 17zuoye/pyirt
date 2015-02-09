'''
The model is an implementation of EM algorithm of IRT


For reference, see:
Brad Hanson, IRT Parameter Estimation using the EM Algorithm, 2000

The current version only deals with unidimension theta

'''
import numpy as np
import collections as cos
import ConfigParser
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import utl
import solver

class IRT_MMLE_2PL(object):

    '''
    Three steps are exposed
    (1) load data
    (2) set parameter
    (3) solve
    '''
    def load_data(self, res_data_list):
        # the input data N*3 array,
        # (uid, eid, atag)
        #TODO: input check

        '''
        Because the algorithm reads a sparse list,
        It is necessary to cache the index methods
        '''
        # parse the data into dictionary, key by item id
        # because the M step is done by
        item2user_dict = cos.defaultdict(list)
        user2item_dict = cos.defaultdict(list)

        # the user
        for log in res_data_list:
            eid = log[1]
            uid = log[0]
            atag = log[2]
            # add to the data dictionary
            item2user_dict[eid].append((uid, atag))
            user2item_dict[uid].append((eid, atag))

        # update the class
        self.user2item_dict = user2item_dict
        self.item2user_dict = item2user_dict


    def load_config(self):
        config = ConfigParser.RawConfigParser()
        config.readfp(open(root_dir+'/config.cfg'))

        # load user item
        theta_min = config.getfloat('user', 'min_theta')
        theta_max = config.getfloat('user', 'max_theta')
        num_theta = config.getint('user','num_theta')
        self._init_user_param(theta_min, theta_max, num_theta)

        # load the solver
        alpha_min = config.getfloat('item', 'min_alpha')
        alpha_max = config.getfloat('item', 'max_alpha')
        beta_min = config.getfloat('item', 'min_beta')
        beta_max = config.getfloat('item', 'max_beta')
        boundary = {'alpha':[alpha_min,alpha_max], 'beta':[beta_min,beta_max]}

        jump_prob = config.getfloat('solver','jump_prob')
        solver_type = config.get('solver','type')
        is_constrained = config.getint('solver','is_constrained')==1
        max_iter = config.getint('solver','max_iter')
        tol = config.getfloat('solver','tol')

        self._init_solver_param(is_constrained, boundary,
                                jump_prob, solver_type, max_iter, tol)


    def solve_EM(self):
        # create the inner parameters
        # currently item parameter requires no setup
        self._init_sys_param()
        self._init_item_param()

        # initialize some intermediate variables used in the E step
        self._init_right_wrong_map()
        self.posterior_theta_distr = np.zeros((self.num_user, self.num_theta))

        #TODO: enable the stopping condition
        num_iter = 1
        self.ell_list = []
        ell_t0 = 0
        while True:
            self._exp_step()
            self._max_step()

            self.theta_vec = np.dot(self.posterior_theta_distr, self.theta_prior_val)

            ell = self.__calc_data_likelihood()
            self.ell_list.append(ell)

            # if the algorithm improves, then ell > ell_t0
            if ell_t0 > ell:
                print('Likelihood descrease, stops.')
                break

            if ell - ell_t0 <= self.tol:
                print('EM converged')
                break
            # update the stop condition
            ell_t0 = ell
            num_iter += 1

            if (num_iter > self.max_iter) :
                print('EM does not converge within max iteration')
                break


    '''
    Main Routine
    '''
    def _exp_step(self):
        '''
        Basic Math:
            In the maximization step, need to use E_[j,k](Y=1),E_[j,k](Y=0)
            E(Y=1|param_j,theta_k) = sum_i(data_[i,j]*P(Y=1|param_j,theta_[i,k]))
            since data_[i,j] takes 0/1, it is equivalent to sum over all done right users

            E(Y=0|param_j,theta_k) = sum_i(
                                    (1-data_[i,j]) *(1-P(Y=1|param_j,theta_[i,k])
                                    )
            By similar logic, it is equivalent to sum over (1-p) for all done wrong users

        '''

        #(1) update the posterior distribution of theta
        self.__update_theta_distr()

        #(2) marginalize
        # because of the sparsity, the expected right and wrong may not sum up
        # to the total num of items!
        self.__get_expect_count()


    def _max_step(self):
        '''
        Basic Math
            log likelihood(param_j) = sum_k(log likelihood(param_j, theta_k))
        '''
        #### [A] max for item parameter
        opt_worker = solver.optimizer.irt_2PL_Optimizer()
        # the boundary is universal
        if self.is_constrained:
            opt_worker.set_bounds([(self.beta_bound[0],  self.beta_bound[1]),
                                (self.alpha_bound[0], self.alpha_bound[1])])

        # theta value is universal
        opt_worker.set_theta(self.theta_prior_val)

        for j in range(self.num_item):
            eid = self.eid_vec[j]
            # set the initial guess as a mixture of current value and a new
            # start to avoid trap in local maximum
            if np.random.uniform() >= self.jump_prob:
                initial_guess_val = (self.item_param_dict[eid]['beta'],
                                    self.item_param_dict[eid]['alpha'])
            else:
                initial_guess_val = (0, 1)
            opt_worker.set_initial_guess(initial_guess_val)

            # assemble the expected data
            expected_right_count = self.item_expected_right_bytheta[:,j]
            expected_wrong_count = self.item_expected_wrong_bytheta[:,j]
            input_data = [expected_right_count,expected_wrong_count]
            opt_worker.load_res_data(input_data)

            # solve by L-BFGS-B
            if self.solver_type == 'BFGS':
                est_param = opt_worker.solve_param_gradient(self.is_constrained)
            elif self.solver_type == 'linear':
                est_param = opt_worker.solve_param_linear(self.is_constrained)
            else:
                raise Exception('Unknown solver type')

            # update
            self.item_param_dict[eid]['beta'] = est_param[0]
            self.item_param_dict[eid]['alpha'] = est_param[1]

        #### [B] max for theta density
        # pi = r_k/(w_k+r_k)
        r_vec = np.sum(self.item_expected_right_bytheta,axis=1)
        w_vec = np.sum(self.item_expected_wrong_bytheta,axis=1)
        self.theta_density = np.divide(r_vec, r_vec+w_vec)



    '''
    Auxuliary function
    '''
    def _init_sys_param(self):
        # system parameter
        self.uid_vec = self.user2item_dict.keys()
        self.num_user = len(self.uid_vec)
        self.eid_vec = self.item2user_dict.keys()
        self.num_item = len(self.eid_vec)

    def _init_solver_param(self, is_constrained, boundary,
                           jump_prob, solver_type, max_iter, tol):
        # initialize bounds
        self.is_constrained = is_constrained
        self.alpha_bound = boundary['alpha']
        self.beta_bound =  boundary['beta']
        self.jump_prob = jump_prob  ## used in max step
        self.solver_type = solver_type
        self.max_iter = max_iter
        self.tol = tol

    def _init_item_param(self):
        self.item_param_dict = {}
        for eid in self.eid_vec:
            self.item_param_dict[eid] = {'alpha':1.0, 'beta':0.0}

    def _init_user_param(self, theta_min, theta_max, num_theta):

        self.theta_prior_val = np.linspace(theta_min, theta_max, num = num_theta)
        self.num_theta = len(self.theta_prior_val)
        if self.num_theta != num_theta:
            raise Exception('Theta initialization failed')
        # store the prior density
        self.theta_density = np.ones(num_theta)/num_theta


    def _init_right_wrong_map(self):
        self.right_wrong_map = {}
        for eid, log_result in self.item2user_dict.iteritems():
            temp = {'right':[], 'wrong':[]}
            for log in log_result:
                atag = log[1]
                uid = log[0]
                # TODO: fix the data type of atag, int or float
                if abs(atag-1.0)<0.001:
                    temp['right'].append(uid)
                else:
                    temp['wrong'].append(uid)
            # update
            self.right_wrong_map[eid] = temp




    def __update_theta_distr(self):

        # TODO: consider shrinkage for the prior update
        '''
        Basic Math. Notice that the distribution is user specific
            P_t(theta,data_i,param) = p(data_i,param|theta)*p_[t-1](theta)
            p_t(data_i,param) = sum(p_t(theta,data_i,param)) over theta
            p_t(theta|data_i,param) = P_t(theta,data_i,param) / p_t(data_i,param)
        '''

        # [A] calculate p(data,param|theta)
        # TODO: speed it up
        for i in range(self.num_user):
            uid = self.uid_vec[i]
            # find all the items
            log_list = self.user2item_dict[uid]
            # create eid list and atag list
            num_log = len(log_list)
            # create temp likelihood vector for each possible value of theta
            likelihood_vec = np.zeros(self.num_theta)
            # calculate
            for k in range(self.num_theta):
                theta = self.theta_prior_val[k]
                # calculate the likelihood
                ell = 0.0
                for m in range(num_log):
                    eid = log_list[m][0]
                    alpha = self.item_param_dict[eid]['alpha']
                    beta = self.item_param_dict[eid]['beta']
                    atag = log_list[m][1]
                    ell += utl.tools.log_likelihood_2PL(atag, theta, alpha, beta)
                # now update the density
                likelihood_vec[k] = ell

            # ell  = p(param|x), full joint = logp(param|x)+log(x)
            log_joint_prob_vec  = likelihood_vec + np.log(self.theta_density)
            # calculate the posterior
            # p(x|param) = exp(logp(param,x) - log(sum p(param,x)))
            marginal = utl.tools.logsum(log_joint_prob_vec)
            self.posterior_theta_distr[i,:] = np.exp(log_joint_prob_vec - marginal)

        # When the loop finish, check if the theta_density adds up to unity for each user
        check_user_distr_marginal = np.sum(self.posterior_theta_distr, axis=1)
        if any(abs(check_user_distr_marginal-1.0)>0.0001):
            raise Exception('The posterior distribution of user ability is not proper')

    def __get_expect_count(self):

        self.item_expected_right_bytheta = np.zeros((self.num_theta, self.num_item))
        self.item_expected_wrong_bytheta = np.zeros((self.num_theta, self.num_item))

        for j in range(self.num_item):
            eid = self.eid_vec[j]
            # get all the users that done it right
            # get all the users that done it wrong
            right_uid_vec = self.right_wrong_map[eid]['right']
            wrong_uid_vec = self.right_wrong_map[eid]['wrong']
            # condition on the posterior ability, what is the expected count of
            # students get it right
            #TODO: for readability, should specify the rows and columns
            self.item_expected_right_bytheta[:,j] = np.sum(self.posterior_theta_distr[right_uid_vec,:], axis = 0)
            self.item_expected_wrong_bytheta[:,j] = np.sum(self.posterior_theta_distr[wrong_uid_vec,:], axis = 0)



    def __calc_data_likelihood(self):
        # calculate the likelihood for the data set

        ell = 0
        for i in range(self.num_user):
            uid = self.uid_vec[i]
            theta = self.theta_vec[i]
            # find all the eid
            for log in self.user2item_dict[uid]:
                eid = log[0]
                atag = log[1]
                alpha = self.item_param_dict[eid]['alpha']
                beta = self.item_param_dict[eid]['beta']

                ell += utl.tools.log_likelihood_2PL(atag, theta, alpha, beta)
        return ell











