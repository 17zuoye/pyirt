# encoding:utf-8
import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)


import unittest
from pyirt import irt
from pyirt.utl.tools import irt_fnc
import numpy as np


class Test_UniDim_Binary_Solver(unittest.TestCase):

    def setUp(self):
        # simulate
        alpha = [1.5,1,1]
        beta = [0,1,-1]
        c = [0.33,0,0.5]
        N = 10000
        T = len(alpha)
        thetas = np.random.rand(N,1)*8-4  
        item_ids = ['a','b','c']

        self.data_2pl = []
        self.data_3pl = []
        for i in range(N):
            for t in range(T):
                prob_2pl = irt_fnc(thetas[i,0],beta[t],alpha[t])
                prob_3pl = irt_fnc(thetas[i,0],beta[t],alpha[t],c[t])
                self.data_2pl.append((i, item_ids[t], np.random.binomial(1,prob_2pl)))
                self.data_3pl.append((i, item_ids[t] ,np.random.binomial(1,prob_3pl)))
        
        self.guess_param = {}
        for t in range(T):
            self.guess_param[item_ids[t]]={'c':c[t]}
        self.alpha = alpha
        self.beta = beta

    def test_2pl_solver(self):
        item_param, user_param = irt(self.data_2pl, alpha_bnds=[0.01,3], beta_bnds=[-3,3], tol=1e-5)
        
        self.assertTrue(abs(item_param['a']['alpha'] - self.alpha[0]) < 0.5)
        self.assertTrue(abs(item_param['a']['beta'] - self.beta[0]) < 1)
    def test_3pl_solver(self):
        item_param, user_param = irt(self.data_3pl, alpha_bnds=[0.01,3], beta_bnds=[-3,3], in_guess_param=self.guess_param, tol=1e-5)

        self.assertTrue(abs(item_param['a']['alpha'] - self.alpha[0]) < 0.5)
        self.assertTrue(abs(item_param['a']['beta'] - self.beta[0]) < 1)

if __name__ == '__main__':
    unittest.main()
