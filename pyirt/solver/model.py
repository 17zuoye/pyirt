#!/usr/bin/env python -W ignore
'''
The model is an implementation of EM algorithm of IRT


For reference, see:
Brad Hanson, IRT Parameter Estimation using the EM Algorithm, 2000

The current version only deals with unidimension theta

'''
import numpy as np
from scipy.stats import norm 
import time
from copy import deepcopy
from six import string_types


from ..util import clib, tools
from ..solver import optimizer
from ..algo import update_theta_distribution

from datetime import datetime
import multiprocessing as mp
from tqdm import tqdm


class IRT_MMLE_2PL(object):

    '''
    Exposed methods
    (1) set options
    (2) solve
    (3) get esitmated result
    '''
    def __init__(self, dao_instance, is_msg=False, is_parallel=False):
        # interface to data
        self.dao=dao_instance
        self.num_iter = 1
        self.ell_list = []
        self.last_avg_prob = 0

        self.is_msg = is_msg
        self.is_parallel = is_parallel
    def set_options(self, theta_bnds, num_theta, alpha_bnds, beta_bnds, max_iter, tol):
        #  user
        self.num_theta = num_theta
        self._init_user_param(theta_bnds[0], theta_bnds[1], num_theta)
        # item
        boundary = {'alpha': alpha_bnds, 'beta': beta_bnds}
        # solver
        solver_type = 'gradient'
        is_constrained = True

        self._init_solver_param(is_constrained, boundary, solver_type, max_iter, tol)

    def set_guess_param(self, in_guess_param):
        self.guess_param_dict = {}
        if isinstance(in_guess_param, string_types):
            for item_idx in range(self.dao.get_num('item')):
                self.guess_param_dict[item_idx] = {'c': 0.0}  # default set to 0
        else:
            for item_idx in range(self.dao.get_num('item')):
                item_id = self.dao.translate('item', item_idx) 
                self.guess_param_dict[item_idx] = in_guess_param[item_id]

    
    def solve_EM(self):
        # data dependent initialization
        self._init_item_param()

        # main routine
        while True:
            #----- E step -----
            stime = datetime.now()
            self._exp_step()
            etime = datetime.now()
            if self.is_msg: 
                runtime = (etime-stime).microseconds / 1000
                print('E step runs for %s sec' %runtime)
            
            #----- M step -----
            stime = datetime.now()
            self._max_step()
            etime = datetime.now()
            if self.is_msg: 
                runtime = (etime-stime).microseconds / 1000
                print('M step runs for %s sec' %runtime)
            
            # ---- Stop Condition ----
            stime = datetime.now()
            is_stop = self._check_stop()
            etime = datetime.now()
            if self.is_msg: 
                runtime = (etime-stime).microseconds / 1000
                print('stop condition runs for %s sec' %runtime)

            if is_stop:
                break
    
    def get_item_param(self):
        output_item_param = {}
        for item_idx in range(self.dao.get_num('item')):
            item_id = self.dao.translate('item', item_idx)
            output_item_param[item_id] = self.item_param_dict[item_idx]  
        return output_item_param

    def get_user_param(self):
        output_user_param = {}
        theta_vec = self.__calc_theta()
        for user_idx in range(self.dao.get_num('user')):
            user_id = self.dao.translate('user', user_idx)
            output_user_param[user_id] = theta_vec[user_idx]
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
        num_item = self.dao.get_num('item')
        num_chunk = min(6, mp.cpu_count())
 
        if num_item > num_chunk and self.is_parallel: 
            def update(d, start_idx, end_idx):
                for item_idx in tqdm(xrange(start_idx, end_idx)):
                    initial_guess_val = (self.item_param_dict[item_idx]['beta'],
                                        self.item_param_dict[item_idx]['alpha'])
                    opt_worker.set_initial_guess(initial_guess_val)  # the value is a mix of 1/0 and current estimate
                    opt_worker.set_c(self.item_param_dict[item_idx]['c'])
                    
                    # estimate
                    expected_right_count = self.item_expected_right_by_theta[:, item_idx]
                    expected_wrong_count = self.item_expected_wrong_by_theta[:, item_idx]
                    input_data = [expected_right_count, expected_wrong_count]
                    opt_worker.load_res_data(input_data)
                    est_param = opt_worker.solve_param_mix(self.is_constrained)
                    d[item_idx] = est_param
            
            # [A] calculate p(data,param|theta)
            chunk_list = tools.cut_list(num_item, num_chunk)

            pool = []
            manager = mp.Manager()
            pool_repo = manager.dict()
            
            self.dao.close_conn()
            for i in range(num_chunk):
                p = mp.Process(target=update, args=(pool_repo, chunk_list[i][0],chunk_list[i][1],))
                pool.append(p) 
            for p in pool:
                p.start()
            for p in pool:
                p.join()
        
            for item_idx in range(num_item):
                self.item_param_dict[item_idx] = {
                        'beta':pool_repo[item_idx][0],
                        'alpha':pool_repo[item_idx][1],
                        'c':self.item_param_dict[item_idx]['c']
                    }


        else:
            for item_idx in range(num_item):
                initial_guess_val = (self.item_param_dict[item_idx]['beta'],
                                    self.item_param_dict[item_idx]['alpha'])
                opt_worker.set_initial_guess(initial_guess_val)  # the value is a mix of 1/0 and current estimate
                opt_worker.set_c(self.item_param_dict[item_idx]['c'])
                
                # estimate
                expected_right_count = self.item_expected_right_by_theta[:, item_idx]
                expected_wrong_count = self.item_expected_wrong_by_theta[:, item_idx]
                input_data = [expected_right_count, expected_wrong_count]
                opt_worker.load_res_data(input_data)
                est_param = opt_worker.solve_param_mix(self.is_constrained)
                
                # update
                self.item_param_dict[item_idx]['beta'] = est_param[0]
                self.item_param_dict[item_idx]['alpha'] = est_param[1]

        # [B] max for theta density
        self.theta_density = self.posterior_theta_distr.sum(axis=0)/self.posterior_theta_distr.sum()
        self.__check_theta_density()
        
    def _check_stop(self):
        '''
        preserve user and item parameter from last iteration. This is useful in restoring after a declining llk iteration 
        '''
        avg_prob = np.exp(self.__calc_data_likelihood())
        self.ell_list.append(avg_prob)
        if self.is_msg: print(avg_prob)

        if self.last_avg_prob < avg_prob and avg_prob - self.last_avg_prob <= self.tol:
            print('EM converged at iteration %d.' % self.num_iter)
            return True
        
        # if the algorithm improves, then ell > ell_t0
        if self.last_avg_prob > avg_prob:
            self.item_param_dict = self.last_item_param_dict
            print('Likelihood descrease, stops at iteration %d.' % self.num_iter)
            return True

        # update the stop condition
        self.last_avg_prob = avg_prob
        self.num_iter += 1

        if (self.num_iter > self.max_iter):
            print('EM does not converge within max iteration')
            return True 
        if self.num_iter != 1:
            self.last_item_param_dict = self.item_param_dict     
        return False

    def _init_solver_param(self, is_constrained, boundary, solver_type, max_iter, tol):
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
        for item_idx in range(self.dao.get_num('item')):
            # need to call the old item_id
            c = self.guess_param_dict[item_idx]['c']
            self.item_param_dict[item_idx] = {'alpha': 1.0, 'beta': 0.0, 'c': c}

    def _init_user_param(self, theta_min, theta_max, num_theta, dist='normal'):
        # generte value
        self.theta_prior_val = np.linspace(theta_min, theta_max, num=num_theta) 
        if self.num_theta != len(self.theta_prior_val):
            raise Exception('wrong number of inintial theta values')
        # use a normal approximation
        if dist == 'uniform':
            self.theta_density = np.ones(num_theta) / num_theta
        elif dist == 'normal':
            norm_pdf = [norm.pdf(x) for x in self.theta_prior_val]
            normalizer = sum(norm_pdf)
            self.theta_density = np.array([x/normalizer for x in norm_pdf])
        else:
            raise Exception('invalid theta prior distibution %s' % dist)
        self.__check_theta_density()
        # space for each learner 
        self.posterior_theta_distr = np.zeros((self.dao.get_num('user'), num_theta))

    def __update_theta_distr(self):
        # [A] calculate p(data,param|theta)
        num_user = self.dao.get_num('user')
        num_chunk = min(6, mp.cpu_count())
        if num_user>num_chunk and self.is_parallel:
            def update(d, start_idx, end_idx):
                for user_idx in tqdm(xrange(start_idx, end_idx)):
                    logs = self.dao.get_log(user_idx)
                    d[user_idx] = update_theta_distribution(logs, self.num_theta, self.theta_prior_val, self.theta_density, self.item_param_dict)
            # [A] calculate p(data,param|theta)
            chunk_list = tools.cut_list(num_user, num_chunk)
            pool = []
            manager = mp.Manager()
            pool_repo = manager.dict()
            self.dao.close_conn()
            for i in range(num_chunk):
                p = mp.Process(target=update, args=(pool_repo, chunk_list[i][0],chunk_list[i][1],))
                pool.append(p) 
            for p in pool:
                p.start()
            for p in pool:
                p.join() 
            for user_idx in range(num_user):
                self.posterior_theta_distr[user_idx,:] = pool_repo[user_idx]
        else:
            for user_idx in range(num_user):
                logs = self.dao.get_log(user_idx)
                self.posterior_theta_distr[user_idx, :] = update_theta_distribution(logs, self.num_theta, self.theta_prior_val, self.theta_density, self.item_param_dict) 
        # When the loop finish, check if the theta_density adds up to unity for each user
        check_user_distr_marginal = np.sum(self.posterior_theta_distr, axis=1)
        if any(abs(check_user_distr_marginal - 1.0) > 0.0001):
            raise Exception('The posterior distribution of user ability is not proper')

    def __check_theta_density(self):
        if abs(sum(self.theta_density) - 1)> 1e-6:
            raise Exception('theta density does not sum upto 1')

        if self.theta_density.shape != (self.num_theta,):
            raise Exception('theta desnity has wrong shape (%s,%s)'%self.theta_density.shape)

    def __get_expect_count(self):
        self.item_expected_right_by_theta = np.zeros((self.num_theta, self.dao.get_num('item')))
        self.item_expected_wrong_by_theta = np.zeros((self.num_theta, self.dao.get_num('item')))
        num_item = self.dao.get_num('item')
        for item_idx in range(num_item):
            map_user_idx_vec = self.dao.get_map(item_idx, ['1','0'])  # reduce the io time for mongo dao 
            self.item_expected_right_by_theta[:, item_idx] = np.sum(self.posterior_theta_distr[map_user_idx_vec[0], :], axis=0)
            self.item_expected_wrong_by_theta[:, item_idx] = np.sum(self.posterior_theta_distr[map_user_idx_vec[1], :], axis=0)

    def __calc_data_likelihood(self):
        # calculate the likelihood for the data set
        # geometric within learner and across learner
       
        #1/N * sum[i](1/Ni *sum[j] (log pij))
        theta_vec  = self.__calc_theta()
        num_user = self.dao.get_num('user')
        num_chunk = min(6, mp.cpu_count())
        if num_user>num_chunk and self.is_parallel:
            def update(tot_llk, cnt, start_idx, end_idx):
                for user_idx in tqdm(range(start_idx, end_idx)):
                    theta = theta_vec[user_idx]
                    # find all the item_id
                    logs = self.dao.get_log(user_idx)
                    if len(logs) == 0:
                        continue
                    ell = 0
                    for log in logs:
                        item_idx = log[0]
                        ans_tag = log[1]
                        alpha = self.item_param_dict[item_idx]['alpha']
                        beta = self.item_param_dict[item_idx]['beta']
                        c = self.item_param_dict[item_idx]['c']
                        ell += clib.log_likelihood_2PL(0.0+ans_tag, 1.0-ans_tag, theta, alpha, beta, c) 
                    with tot_llk.get_lock():
                        tot_llk.value += ell/len(logs)
                    with cnt.get_lock():
                        cnt.value += 1

            user_ell = mp.Value('d', 0.0)  
            user_cnt = mp.Value('i', 0)

            chunk_list = tools.cut_list(num_user, num_chunk)
            pool = []
            self.dao.close_conn()
            for i in range(num_chunk):
                p = mp.Process(target=update, args=(user_ell, user_cnt, chunk_list[i][0],chunk_list[i][1],))
                pool.append(p) 
            for p in pool:
                p.start()
            for p in pool:
                p.join() 

            avg_ell = user_ell.value/user_cnt.value
        else:
            user_ell = 0
            user_cnt = 0
            for user_idx in range(num_user):
                theta = theta_vec[user_idx]
                # find all the item_id
                logs = self.dao.get_log(user_idx)
                if len(logs) == 0:
                    continue
                ell = 0
                for log in logs:
                    item_idx = log[0]
                    ans_tag = log[1]
                    alpha = self.item_param_dict[item_idx]['alpha']
                    beta = self.item_param_dict[item_idx]['beta']
                    c = self.item_param_dict[item_idx]['c']
                    ell += clib.log_likelihood_2PL(0.0+ans_tag, 1.0-ans_tag, theta, alpha, beta, c) 
                user_ell += ell/len(logs)
                user_cnt += 1
            avg_ell = user_ell / user_cnt
        return avg_ell 

    def __calc_theta(self):
        return np.dot(self.posterior_theta_distr, self.theta_prior_val)

