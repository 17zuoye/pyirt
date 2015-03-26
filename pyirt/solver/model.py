'''
The model is an implementation of EM algorithm of IRT


For reference, see:
Brad Hanson, IRT Parameter Estimation using the EM Algorithm, 2000

The current version only deals with unidimension theta

'''
import numpy as np
import os
import sys
import time


root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import utl
import solver


# import cython
class IRT_MMLE_2PL(object):

    '''
    Three steps are exposed
    (1) load data
    (2) set parameter
    (3) solve
    '''
    def load_data(self, src, is_mount, user_name, tmp_dir='/tmp/pyirt/'):
        # three columns are uid, eid, atag
        if isinstance(src, file):
            # if the src is file handle
            uids, eids, atags = self._loadFromHandle(src)
        else:
            # if the src is list of tuples
            uids, eids, atags = self._loadFromTuples(src)
        # process it
        print('Data loading is complete.')

        self.data_ref = utl.loader.data_storage()
        self.data_ref.setup(uids,eids,atags)

    def load_param(self, theta_bnds, alpha_bnds, beta_bnds):
        # TODO: allow for a more flexible parameter setting
        # The config object has to be passed in because hdfs file system does
        # not load target file

        # load user item
        num_theta = 11
        self._init_user_param(theta_bnds[0], theta_bnds[1], num_theta)

        # load the solver
        boundary = {'alpha': alpha_bnds,
                    'beta':  beta_bnds}

        solver_type    = 'gradient'
        is_constrained = True
        max_iter       = 10
        tol            = 1e-3

        self._init_solver_param(is_constrained, boundary, solver_type, max_iter, tol)

    def load_guess_param(self, in_guess_param):
        if isinstance(in_guess_param, basestring):
            # all c are 0
            guess_param_dict = {}
            for eid in self.data_ref.eid_vec:
                guess_param_dict[eid] = {'c': 0.0}
        else:
            guess_param_dict = in_guess_param

        self.guess_param_dict = guess_param_dict

    def solve_EM(self):
        # create the inner parameters
        # currently item parameter requires no setup
        self._init_item_param()

        self.posterior_theta_distr = np.zeros((self.data_ref.num_user, self.num_theta))

        # TODO: enable the stopping condition
        num_iter      = 1
        self.ell_list = []
        avg_prob_t0   = 0

        while True:
            iter_start_time = time.time()
            # add in time block
            start_time = time.time()
            self._exp_step()
            print("--- E step: %f secs ---" % np.round((time.time()-start_time)))

            start_time = time.time()
            self._max_step()
            print("--- M step: %f secs ---" % np.round((time.time()-start_time)))

            self.__calc_theta()

            '''
            Exp
            '''
            # self.update_guess_param()

            # the goal is to maximize the "average" probability
            avg_prob = np.exp(self.__calc_data_likelihood()/self.data_ref.num_log)
            self.ell_list.append(avg_prob)
            print("--- all: %f secs ---" % np.round((time.time()-iter_start_time)))
            print(avg_prob)

            # if the algorithm improves, then ell > ell_t0
            if avg_prob_t0 > avg_prob:
                # TODO: needs to roll back if the likelihood decrease
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
        # need to remap the inner id to the outer id
        return self.item_param_dict

    def get_user_param(self):
        user_param_dict = {}
        for i in xrange(self.data_ref.num_user):
            uid = self.data_ref.uid_vec[i]
            user_param_dict[uid] = self.theta_vec[i]

        return user_param_dict

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
        opt_worker = solver.optimizer.irt_2PL_Optimizer()
        # the boundary is universal
        # the boundary is set regardless of the constrained option because the
        # constrained search serves as backup for outlier cases
        opt_worker.set_bounds([self.beta_bnds, self.alpha_bnds])

        # theta value is universal
        opt_worker.set_theta(self.theta_prior_val)

        for eid in self.data_ref.eid_vec:
            # set the initial guess as a mixture of current value and a new
            # start to avoid trap in local maximum
            initial_guess_val = (self.item_param_dict[eid]['beta'],
                                 self.item_param_dict[eid]['alpha'])

            opt_worker.set_initial_guess(initial_guess_val)
            opt_worker.set_c(self.item_param_dict[eid]['c'])

            # assemble the expected data
            j = self.data_ref.eidx[eid]
            expected_right_count = self.item_expected_right_bytheta[:,j]
            expected_wrong_count = self.item_expected_wrong_bytheta[:,j]
            input_data = [expected_right_count, expected_wrong_count]
            opt_worker.load_res_data(input_data)

            # solve by L-BFGS-B
            if self.solver_type == 'gradient':
                est_param = opt_worker.solve_param_gradient(self.is_constrained)
            elif self.solver_type == 'linear':
                # if the alogrithm is nelder-mead and the optimization fails to
                # converge, use the constrained version
                try:
                    est_param = opt_worker.solve_param_linear(self.is_constrained)
                except Exception  as  e:
                    if str(e) == 'Optimizer fails to find solution. Try constrained search.':
                        est_param = opt_worker.solve_param_linear(True)
                    else:
                        raise e
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
    def _loadFromTuples(self, data):
        uids  = []
        eids  = []
        atags = []
        if len(data) == 0:
            raise Exception('Data is empty.')

        for log in data:
            uids.append(int(log[0]))
            eids.append(int(log[1]))
            atags.append(int(log[2]))

        return uids, eids, atags

    def _loadFromHandle(self, fp, sep=','):
        # Default format is comma separated files,
        # Only int is allowed within the environment
        uids  = []
        eids  = []
        atags = []

        for line in fp:
            if line == '':
                continue
            uidstr, eidstr, atagstr = line.strip().split(sep)
            uids.append(int(uidstr))
            eids.append(int(eidstr))
            atags.append(int(atagstr))
        return uids, eids, atags

    def _init_solver_param(self, is_constrained, boundary,
                           solver_type, max_iter, tol):
        # initialize bounds
        self.is_constrained  = is_constrained
        self.alpha_bnds      = boundary['alpha']
        self.beta_bnds       = boundary['beta']
        self.solver_type     = solver_type
        self.max_iter        = max_iter
        self.tol             = tol

        if solver_type == 'gradient' and not is_constrained:
            raise Exception('BFGS has to be constrained')

    def _init_item_param(self):
        self.item_param_dict = {}
        for eid in self.data_ref.eid_vec:
            # need to call the old eid
            c = self.guess_param_dict[eid]['c']

            self.item_param_dict[eid] = {'alpha': 1.0, 'beta': 0.0, 'c': c}

    def _init_user_param(self, theta_min, theta_max, num_theta):
        self.theta_prior_val = np.linspace(theta_min, theta_max, num=num_theta)
        self.num_theta       = len(self.theta_prior_val)
        if self.num_theta   != num_theta:
            raise Exception('Theta initialization failed')
        # store the prior density
        self.theta_density   = np.ones(num_theta)/num_theta


    def __update_theta_distr(self):

        # TODO: consider shrinkage for the prior update
        '''
        Basic Math. Notice that the distribution is user specific
            P_t(theta,data_i,param) = p(data_i,param|theta)*p_[t-1](theta)
            p_t(data_i,param) = sum(p_t(theta,data_i,param)) over theta
            p_t(theta|data_i,param) = P_t(theta,data_i,param)/p_t(data_i,param)
        '''

        # [A] calculate p(data,param|theta)
        # TODO: speed it up
        for i in xrange(self.data_ref.num_user):
            uid = self.data_ref.uid_vec[i]
            # find all the items
            log_list = self.data_ref.get_log(uid)
            # create eid list and atag list
            num_log  = len(log_list)
            # create temp likelihood vector for each possible value of theta
            likelihood_vec= np.zeros(self.num_theta)
            # calculate
            for k in range(self.num_theta):
                theta     = self.theta_prior_val[k]
                # calculate the likelihood
                ell       = 0.0
                for m in range(num_log):
                    eid   = log_list[m][0]
                    alpha = self.item_param_dict[eid]['alpha']
                    beta  = self.item_param_dict[eid]['beta']
                    c     = self.item_param_dict[eid]['c']
                    atag  = log_list[m][1]
                    ell   += utl.clib.log_likelihood_2PL(atag, 1.0-atag,
                                                          theta, alpha, beta, c)
                # now update the density
                likelihood_vec[k] = ell

            # ell  = p(param|x), full joint = logp(param|x)+log(x)
            log_joint_prob_vec = likelihood_vec + np.log(self.theta_density)

            # calculate the posterior
            # p(x|param) = exp(logp(param,x) - log(sum p(param,x)))
            marginal = utl.tools.logsum(log_joint_prob_vec)
            self.posterior_theta_distr[i,:] = np.exp(log_joint_prob_vec - marginal)

        # When the loop finish, check if the theta_density adds up to unity for each user
        check_user_distr_marginal = np.sum(self.posterior_theta_distr, axis=1)
        if any(abs(check_user_distr_marginal - 1.0) > 0.0001):
            raise Exception('The posterior distribution of user ability is not proper')

    def __get_expect_count(self):

        self.item_expected_right_bytheta = np.zeros((self.num_theta, self.data_ref.num_item))
        self.item_expected_wrong_bytheta = np.zeros((self.num_theta, self.data_ref.num_item))

        for j in range(self.data_ref.num_item):
            eid = self.data_ref.eid_vec[j]
            # get all the users that done it right
            # get all the users that done it wrong
            right_uid_vec, wrong_uid_vec = self.data_ref.get_rwmap(eid)
            # condition on the posterior ability, what is the expected count of
            # students get it right
            # TODO: for readability, should specify the rows and columns
            self.item_expected_right_bytheta[:,j] = np.sum(self.posterior_theta_distr[right_uid_vec,:], axis = 0)
            self.item_expected_wrong_bytheta[:,j] = np.sum(self.posterior_theta_distr[wrong_uid_vec,:], axis = 0)


    def __calc_data_likelihood(self):
        # calculate the likelihood for the data set

        ell = 0
        for i in range(self.data_ref.num_user):
            uid = self.data_ref.uid_vec[i]
            theta = self.theta_vec[i]
            # find all the eid
            logs = self.data_ref.get_log(uid)
            for log in logs:
                eid = log[0]
                atag = log[1]
                alpha = self.item_param_dict[eid]['alpha']
                beta = self.item_param_dict[eid]['beta']
                c = self.item_param_dict[eid]['c']

                ell += utl.clib.log_likelihood_2PL(atag, 1-atag,
                                                    theta, alpha, beta, c)
        return ell

    def __calc_theta(self):
        self.theta_vec = np.dot(self.posterior_theta_distr, self.theta_prior_val)


    '''
    Experimental
    def update_guess_param(self):
        # at the end of each repetition, try to update the distribution of c

        #C is only identified at the extreme right tail, which is not possible in the EM environment
        #Use the average performance of the worst ability

        raise Exception('Currently deprecated!')

        # find the user that are in the bottom 5%
        cut_threshold = np.percentile(self.theta_vec, 5)
        bottom_group = [i for i in range(self.data_ref.num_user) if self.theta_vec[i] <= cut_threshold]

        # now loop through all the items
        for eid in self.data_ref.eid_vec:

            if self.item_param_dict[eid]['update_c']:
                # find the user group
                user_group = self.eid2uid_dict[eid]
                guessers = set(user_group).intersection(bottom_group)
                num_guesser = len(guessers)
                if num_guesser>10:
                    # average them
                    rw_list = self.right_wrong_map[eid]
                    right_cnt = 0.0
                    for uid in guessers:
                        if uid in rw_list['right']:
                            right_cnt += 1
                    # update c
                    # cap at 0.5
                    self.item_param_dict[eid]['c'] = min(right_cnt/num_guesser, 0.5)
    '''

