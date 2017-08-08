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
item_ids =  ['a', 'b', 'c', 'd', 'e','g','h','i','j']

N = 1000
T = len(alpha)
theta_range = 4

guess_param = {}
for t in range(T):
    guess_param[item_ids[t]]={'c':c[t]}
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
                cls.data.append((i, item_ids[t], np.random.binomial(1, prob)))
        

    def test_2pl_solver(self):
        item_param, user_param = irt(self.data, theta_bnds=[-theta_range/2,theta_range/2], num_theta=11, alpha_bnds=[0.25,3], beta_bnds=[-3,3], tol=1e-5, max_iter=30)
        for t in range(T):
            item_id = item_ids[t]
            print(item_id, item_param[item_id])
            mdl_alpha = item_param[item_id]['alpha'] 
            mdl_beta = item_param[item_id]['beta'] 
            if item_id != 'h':
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
                cls.data.append((i, item_ids[t] ,np.random.binomial(1,prob)))

    def test_3pl_solver(self):
        item_param, user_param = irt(self.data, theta_bnds=[-theta_range/2,theta_range/2], num_theta=11, alpha_bnds=[0.25,3], beta_bnds=[-3,3], 
                in_guess_param=guess_param, tol=1e-5, max_iter=30)

        for t in range(T):
            item_id = item_ids[t]
            print(item_id, item_param[item_id])
            mdl_alpha = item_param[item_id]['alpha'] 
            mdl_beta = item_param[item_id]['beta'] 
            if item_id not in ['h','i']:
                self.assertTrue(abs(mdl_alpha - alpha[t])<0.25)
            if item_id != 'j':
                self.assertTrue(abs(mdl_beta - beta[t])<0.15)

if __name__ == '__main__':
    unittest.main()
