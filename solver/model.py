'''
The model is an implementation of EM algorithm of IRT
'''
import numpy as np
import collections as cos
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import utl

class IRT_MMLE_2PL(object):

    def set_theta_prior(self, num_theta = 25):
        self.theta_prior_val = np.linspace(-4, 8, num = num_theta)

    def load_response_data(self, res_data_list):
        # the input data N*3 array,
        # (uid, eid, atag)
        #TODO: input check
        # parse the data into dictionary, key by item id
        # because the M step is done by
        item2user_dict = cos.defaultdict(list)
        user2item_dict = cos.defaultdict(list)

        user_param_dict = {}
        item_param_dict = {}

        # the user
        for log in res_data_list:
            eid = log[1]
            uid = log[0]
            atag = log[2]
            # add to the data dictionary
            item2user_dict[eid].append((uid, atag))
            user2item_dict[uid].append((eid, atag))
            # initialize the parameter collector
            if uid not in user_param_dict:
                user_param_dict[uid] = 0.0
            if eid not in item_param_dict:
                item_param_dict[eid] = {'alpha':1.0, 'beta':0.0}

        # update the class
        self.user2item_dict = user2item_dict
        self.item2user_dict = item2user_dict
        self.user_param_dict = user_param_dict
        self.item_param_dict = item_param_dict

        # create the inner parameters
        self._init_parameters()
        self._init_theta_density()
        self._init_item2user_result_dict()  # this is used in the expectation step

    def _init_parameters(self):
        self.num_theta_prior = len(self.theta_prior_val)
        self.uid_vec = self.user_param_dict.keys()
        self.num_user = len(self.uid_vec)
        self.eid_vec = self.item_param_dict.keys()
        self.num_item = len(self.eid_vec)

    def _init_item2user_result_dict(self):
        self.item2user_result_dict = {}
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
            self.item2user_result_dict[eid] = temp

    def _init_theta_density(self):
        # the prior is uniform density
        self.theta_density = np.ones((self.num_user, self.num_theta_prior)) / self.num_theta_prior

    def _update_theta_density(self):
        # for the ease of debugging, loop over user then theta
        for i in range(self.num_user):
            uid = self.uid_vec[i]
            # find all the items
            log_list = self.user2item_dict[uid]
            # create eid list and atag list
            num_log = len(log_list)
            # create temp likelihood vector
            likelihood_vec = np.zeros(self.num_theta_prior)

            # update the posterior of theta distribution
            for j in range(self.num_theta_prior):
                theta = self.theta_prior_val[j]
                # calculate the likelihood
                # TODO: need to cache it, otherwise it will be too slow
                for k in range(num_log):
                    eid = log_list[k][0]
                    alpha = self.item_param_dict[eid]['alpha']
                    beta = self.item_param_dict[eid]['beta']
                    atag = log_list[k][1]
                    ell = 0.0
                    ell += utl.tools.log_likelihood_2PL(atag, theta, alpha, beta)
                # now update the density
                likelihood_vec[j] = ell
            # TODO: consider if dampens it by adding a constant base
            # ell  = p(param|x), full joint = logp(param|x)+log(x)
            log_joint_prob_vec  = np.log(self.theta_density[i,:]) + likelihood_vec
            # calculate the posterior
            # p(x|param) = exp(logp(param,x) - log(sum p(param,x)))
            marginal = utl.tools.logsum(log_joint_prob_vec)
            self.theta_density[i,:] = np.exp(log_joint_prob_vec - marginal)

            # check if the theta_density adds up to unity for each user
            check_user_marginal = np.sum(self.theta_density, axis=1)
            if any(abs(check_user_marginal-1.0)>0.0001):
                raise Exception('The user ability density is not proper')


    def _expectation_step(self):
        '''
        The expectation step calculate E(Y|param, theta, Y_o), which integrates out user
        '''
        '''
        (1) update the posterior prob of theta
        '''
        self._update_theta_density()

        '''
        (2) get the expected count
        '''
        # because of the sparsity, the expected right and wrong may not sum up
        # to the total num of items
        self.item_expected_right_bytheta = np.zeros((self.num_theta_prior, self.num_item))
        self.item_expected_wrong_bytheta = np.zeros((self.num_theta_prior, self.num_item))
        for j in range(self.num_item):
            eid = self.eid_vec[j]
            right_uid_vec = self.item2user_result_dict[eid]['right']
            wrong_uid_vec = self.item2user_result_dict[eid]['wrong']
            # condition on the posterior ability, what is the expected count of
            # students get it right
            #TODO: for readability, should specify the rows and columns
            self.item_expected_right_bytheta[:,j] = np.sum(self.theta_density[right_uid_vec,:], axis = 0)
            self.item_expected_wrong_bytheta[:,j] = np.sum(self.theta_density[wrong_uid_vec,:], axis = 0)

        #TODO: get the global prior(posterior) density of theta distribution
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT








