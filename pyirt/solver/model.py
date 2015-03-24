'''
The model is an implementation of EM algorithm of IRT


For reference, see:
Brad Hanson, IRT Parameter Estimation using the EM Algorithm, 2000

The current version only deals with unidimension theta

'''
import numpy as np
import os
import sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import utl
import solver
import bsddb
from utl import clib


class IRT_MMLE_2PL(object):

    '''
    Three steps are exposed
    (1) load data
    (2) set parameter
    (3) solve
    '''

    def loadFromHandle(self, fp, sep=',', tmp_dir='/tmp/pyirt/'):
        # Default format is comma separated files, three columns are uid, eid,
        # atag
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

        # process it
        # check if the tmp directory is accessible
        if not os.path.isdir(tmp_dir):
            os.mkdir(tmp_dir)
        self.tmp_dir = tmp_dir  # passed in for later cache
        self._process_data(uids, eids, atags)
        self._init_data_param()

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

        self._init_solver_param(is_constrained, boundary,
                                solver_type, max_iter, tol)

    def load_guess_param(self, in_guess_param):
        if isinstance(in_guess_param, basestring):
            # all c are 0
            guess_param_dict = {}
            for eid in self.eid_vec:
                guess_param_dict[eid] = {'c': 0.0}
        else:
            guess_param_dict = in_guess_param

        self.guess_param_dict = guess_param_dict

    def solve_EM(self):
        # create the inner parameters
        # currently item parameter requires no setup
        self._init_item_param()

        # initialize some intermediate variables used in the E step
        self._init_right_wrong_map()
        self.posterior_theta_distr = np.zeros((self.num_user, self.num_theta))

        # TODO: enable the stopping condition
        num_iter      = 1
        self.ell_list = []
        avg_prob_t0   = 0

        while True:
            # add in time block
            # start_time = time.time()
            self._exp_step()
            # print("--- E step: %f secs ---" %
            # np.round((time.time()-start_time)))

            # start_time = time.time()
            self._max_step()
            # print("--- M step: %f secs ---" %
            # np.round((time.time()-start_time)))

            self.__calc_theta()

            '''
            Exp
            '''
            # self.update_guess_param()

            # the goal is to maximize the "average" probability
            avg_prob = np.exp(self.__calc_data_likelihood()/self.num_log)
            self.ell_list.append(avg_prob)
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
        for i in xrange(self.num_user):
            uid = self.uid_vec[i]
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

        for eid in self.eid_vec:
            # set the initial guess as a mixture of current value and a new
            # start to avoid trap in local maximum
            initial_guess_val = (self.item_param_dict[eid]['beta'],
                                 self.item_param_dict[eid]['alpha'])

            opt_worker.set_initial_guess(initial_guess_val)
            opt_worker.set_c(self.item_param_dict[eid]['c'])

            # assemble the expected data
            j = self.eid_vec.index(eid)
            expected_right_count = self.item_expected_right_bytheta[:,j]
            expected_wrong_count = self.item_expected_wrong_bytheta[:,j]
            input_data = [expected_right_count,expected_wrong_count]
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
    def _process_data(self, uids, eids, atags):
        '''
        Memory efficiency optimization:

        (1) The matrix is sparse, so use three parallel lists for data storage.
        parallel lists are more memory eficient than tuple
        (2) For fast retrieval, turn three parallel list into dictionary by uid and eid
        (3) Python dictionary takes up a lot of memory, so use dbm

        # for more details, see
        http://stackoverflow.com/questions/2211965/python-memory-usage-loading-large-dictionaries-in-memory

        # The design is originally described by Chaoqun Fu
        '''

        '''
        Need the following dictionary for esitmation routine
        (1) item -> user: key: eid, value: (uid, atag)
        (2) user -> item: key: uid, value: (eid, atag)
        '''
        # always rewrite
        self.item2user_db = bsddb.hashopen(self.tmp_dir+'/item2user.db', 'n')
        self.user2item_db = bsddb.hashopen(self.tmp_dir+'/user2item.db', 'n')
        self.num_log = len(uids)

        for i in xrange(self.num_log):
            eid  = eids[i]
            uid  = uids[i]
            atag = atags[i]
            # if not initiated, init with empty str
            if str(eid) not in self.item2user_db:
                self.item2user_db['%d' % eid] = ''
            if str(uid) not in self.user2item_db:
                self.user2item_db['%d' % uid] = ''

            self.item2user_db['%d' % eid] += '%d,%d;' % (uid, atag)
            self.user2item_db['%d' % uid] += '%d,%d;' % (eid, atag)

    def _init_data_param(self):
        # system parameter
        self.uid_vec = [int(x) for x in self.user2item_db.keys()]
        self.num_user = len(self.uid_vec)
        self.eid_vec = [int(x) for x in self.item2user_db.keys()]
        self.num_item = len(self.eid_vec)


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
        for eid in self.eid_vec:
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

    def _init_right_wrong_map(self):
        self.right_map = bsddb.hashopen(self.tmp_dir+'/right_map.db', 'n')
        self.wrong_map = bsddb.hashopen(self.tmp_dir+'/wrong_map.db', 'n')

        for eidstr, log_val_list in self.item2user_db.iteritems():
            log_result = utl.loader.load_dbm(log_val_list)
            for log in log_result:
                # The E step uses the index of the uid
                uid_idx = self.uid_vec.index(log[0])
                atag = log[1]
                if atag == 1:
                    if eidstr not in self.right_map:
                        self.right_map[eidstr] = '%d' % uid_idx
                    else:
                        self.right_map[eidstr] += ',%d' % uid_idx
                else:
                    if eidstr not in self.wrong_map:
                        self.wrong_map[eidstr] = '%d' % uid_idx
                    else:
                        self.wrong_map[eidstr] += ',%d' % uid_idx

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
        for i in xrange(self.num_user):
            uid = self.uid_vec[i]
            # find all the items
            log_list      = utl.loader.load_dbm(self.user2item_db['%d' % uid])
            # create eid list and atag list
            num_log       = len(log_list)
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
                    ell   += clib.log_likelihood_2PL(atag, 1.0-atag,
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

        self.item_expected_right_bytheta = np.zeros((self.num_theta, self.num_item))
        self.item_expected_wrong_bytheta = np.zeros((self.num_theta, self.num_item))

        for j in range(self.num_item):
            eid = self.eid_vec[j]
            # get all the users that done it right
            # get all the users that done it wrong

            right_uid_vec = [int(x) for x in self.right_map[str(eid)].split(',') ]
            wrong_uid_vec = [int(x) for x in self.wrong_map[str(eid)].split(',') ]
            # condition on the posterior ability, what is the expected count of
            # students get it right
            # TODO: for readability, should specify the rows and columns
            self.item_expected_right_bytheta[:,j] = np.sum(self.posterior_theta_distr[right_uid_vec,:], axis = 0)
            self.item_expected_wrong_bytheta[:,j] = np.sum(self.posterior_theta_distr[wrong_uid_vec,:], axis = 0)

    def __calc_data_likelihood(self):
        # calculate the likelihood for the data set

        ell = 0
        for i in range(self.num_user):
            uid = self.uid_vec[i]
            theta = self.theta_vec[i]
            # find all the eid
            logs = utl.loader.load_dbm(self.user2item_db['%d' % uid])
            for log in logs:
                eid = log[0]
                atag = log[1]
                alpha = self.item_param_dict[eid]['alpha']
                beta = self.item_param_dict[eid]['beta']
                c = self.item_param_dict[eid]['c']

                ell += clib.log_likelihood_2PL(atag, 1-atag,
                                                    theta, alpha, beta, c)
        return ell

    def __calc_theta(self):
        self.theta_vec = np.dot(self.posterior_theta_distr, self.theta_prior_val)


    '''
    Experimental
    '''
    def update_guess_param(self):
        # at the end of each repetition, try to update the distribution of c
        '''
        C is only identified at the extreme right tail, which is not possible in the EM environment
        Use the average performance of the worst ability
        '''

        raise Exception('Currently deprecated!')

        # find the user that are in the bottom 5%
        cut_threshold = np.percentile(self.theta_vec, 5)
        bottom_group = [i for i in range(self.num_user) if self.theta_vec[i] <= cut_threshold]

        # now loop through all the items
        for eid in self.eid_vec:

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
