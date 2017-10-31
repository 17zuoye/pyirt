# encoding:utf-8
import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)


import unittest
from pyirt import irt
from pyirt.util.tools import irt_fnc
import numpy as np

alpha =     [0.5, 0.5, 0.5, 1,   1,  1,  2,  2,  2]
beta =      [0,   1,   -1,  0,   1, -1,  0,  1, -1]
c =         [0.5, 0,    0,  0,   0.5,0,  0,  0, 0.5]

N = 1000
T = len(alpha)
theta_range = 4

guess_param = {}
for t in range(T):
    guess_param['q%d'%t]=c[t]




class Test2PLSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # simulate
        np.random.seed(20170807)
        thetas = np.random.rand(N,1)*theta_range - theta_range/2   
        cls.data = []
        for i in range(N):
            for t in range(T):
                prob = irt_fnc(thetas[i,0], beta[t], alpha[t])
                cls.data.append(('u%d'%i, 'q%d'%t, np.random.binomial(1, prob)))

    def test_2pl_solver(self):
        item_param, user_param = irt(self.data,
                theta_bnds=[-theta_range/2,theta_range/2], num_theta=11, alpha_bnds=[0.25,3], beta_bnds=[-3,3], tol=1e-5, max_iter=30)
        for t in range(T):
            item_id = 'q%d'%t
            print(item_id, item_param[item_id])
            mdl_alpha = item_param[item_id]['alpha'] 
            mdl_beta = item_param[item_id]['beta'] 
            if item_id != 'q6':
                self.assertTrue(abs(mdl_alpha - alpha[t])<0.37)
            self.assertTrue(abs(mdl_beta - beta[t])<0.16)
    
    def test_2pl_solver_production(self):
        item_param, user_param = irt(self.data,
                mode='production',theta_bnds=[-theta_range/2,theta_range/2], num_theta=11, alpha_bnds=[0.25,3], beta_bnds=[-3,3], tol=1e-5, max_iter=30)
        for t in range(T):
            item_id = 'q%d'%t
            print(item_id, item_param[item_id])
            mdl_alpha = item_param[item_id]['alpha'] 
            mdl_beta = item_param[item_id]['beta'] 
            if item_id != 'q6':
                self.assertTrue(abs(mdl_alpha - alpha[t])<0.37)
            self.assertTrue(abs(mdl_beta - beta[t])<0.16)

    def test_2pl_solver_parallel(self):
        item_param, user_param = irt(self.data, 
                theta_bnds=[-theta_range/2,theta_range/2], num_theta=11, alpha_bnds=[0.25,3], beta_bnds=[-3,3], tol=1e-5, max_iter=30, is_parallel=True, check_interval=0.1)
        for t in range(T):
            item_id = 'q%d'%t
            print(item_id, item_param[item_id])
            mdl_alpha = item_param[item_id]['alpha'] 
            mdl_beta = item_param[item_id]['beta'] 
            if item_id != 'q6':
                self.assertTrue(abs(mdl_alpha - alpha[t])<0.37)
            self.assertTrue(abs(mdl_beta - beta[t])<0.16)

class Test3PLSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # simulate
        np.random.seed(20170807)
        thetas = np.random.rand(N,1)*theta_range - theta_range/2   
        cls.data = []
        for i in range(N):
            for t in range(T):
                prob = irt_fnc(thetas[i,0], beta[t], alpha[t], c[t])
                cls.data.append(('u%d'%i, 'q%d'%t ,np.random.binomial(1,prob)))

    def test_3pl_solver(self):
        item_param, user_param = irt(self.data, 
                theta_bnds=[-theta_range/2,theta_range/2], num_theta=11, alpha_bnds=[0.25,3], beta_bnds=[-3,3], 
                in_guess_param=guess_param, tol=1e-5, max_iter=30)

        for t in range(T):
            item_id = 'q%d'%t
            print(item_id, item_param[item_id])
            mdl_alpha = item_param[item_id]['alpha'] 
            mdl_beta = item_param[item_id]['beta'] 
            if item_id not in ['q6','q7']:
                self.assertTrue(abs(mdl_alpha - alpha[t])<0.25)
            if item_id != 'q8':
                self.assertTrue(abs(mdl_beta - beta[t])<0.15)


    def test_3pl_solver_production(self):
        item_param, user_param = irt(self.data,
                mode='production', theta_bnds=[-theta_range/2,theta_range/2], num_theta=11, alpha_bnds=[0.25,3], beta_bnds=[-3,3], 
                in_guess_param=guess_param, tol=1e-5, max_iter=30)

        for t in range(T):
            item_id = 'q%d'%t
            print(item_id, item_param[item_id])
            mdl_alpha = item_param[item_id]['alpha'] 
            mdl_beta = item_param[item_id]['beta'] 
            if item_id not in ['q6','q7']:
                self.assertTrue(abs(mdl_alpha - alpha[t])<0.25)
            if item_id != 'q8':
                self.assertTrue(abs(mdl_beta - beta[t])<0.15)


    def test_3pl_solver_parallel(self):
        item_param, user_param = irt(self.data, 
                theta_bnds=[-theta_range/2,theta_range/2], num_theta=11, alpha_bnds=[0.25,3], beta_bnds=[-3,3], 
                in_guess_param=guess_param, tol=1e-5, max_iter=30, is_parallel=True, check_interval=0.1)

        for t in range(T):
            item_id = 'q%d'%t
            print(item_id, item_param[item_id])
            mdl_alpha = item_param[item_id]['alpha'] 
            mdl_beta = item_param[item_id]['beta'] 
            if item_id not in ['q6','q7']:
                self.assertTrue(abs(mdl_alpha - alpha[t])<0.25)
            if item_id != 'q8':
                self.assertTrue(abs(mdl_beta - beta[t])<0.15)



if __name__ == '__main__':
    unittest.main()
