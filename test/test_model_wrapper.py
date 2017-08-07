# encoding:utf-8
import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)


import unittest
from pyirt import irt
from pyirt.util.tools import irt_fnc
import numpy as np

alpha = [1.5, 1,  1 ]
beta =  [0,   1, -1 ]
c =     [0.33,0, 0.5]
N = 10000
T = len(alpha)
item_ids = ['a','b','c']


guess_param = {}
for t in range(T):
    guess_param[item_ids[t]]={'c':c[t]}


class Test2PLSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # simulate
        thetas = np.random.rand(N,1)*8-4  
        np.random.seed(20170807)
        cls.data = []
        for i in range(N):
            for t in range(T):
                prob = irt_fnc(thetas[i,0], beta[t], alpha[t])
                cls.data.append((i, item_ids[t], np.random.binomial(1, prob)))
        

    def test_2pl_solver(self):
        item_param, user_param = irt(self.data, alpha_bnds=[0.25,3], beta_bnds=[-3,3], tol=1e-8, max_iter=3)
        '''  
        mdl_alpha = item_param['a']['alpha'] 
        mdl_beta = item_param['a']['beta'] 
        self.assertAlmostEqual(mdl_alpha, self.alpha[0], places=5)
        self.assertAlmostEqual(mdl_beta, self.beta[0], places=5)
        '''
        for item_id, param in item_param.items():
            print(item_id, param)
'''
class Test3PLSolver(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

                prob_3pl = irt_fnc(thetas[i,0],beta[t],alpha[t],c[t])

                cls.data_3pl.append((i, item_ids[t] ,np.random.binomial(1,prob_3pl)))

    def test_3pl_solver(self):
        item_param, user_param = irt(self.data_3pl, alpha_bnds=[0.01,3], beta_bnds=[-3,3], in_guess_param=self.guess_param, tol=1e-5)
        
        mdl_alpha = item_param['a']['alpha'] 
        mdl_beta = item_param['a']['beta'] 
'''

if __name__ == '__main__':
    unittest.main()
