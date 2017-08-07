# -*- coding:utf-8 -*-
import unittest

import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)

from pyirt.algo import update_theta_distribution


class TestUpdateThetaDistribution(unittest.TestCase):

    def setUp(self):
        self.num_theta = 3
        self.theta_prior_val = [-1, 0, 1]
        self.theta_density = [0.25, 0.5, 0.25]
        self.item_param_dict = {
                0:{'alpha':1,'beta':0,'c':0},
                1:{'alpha':1.5,'beta':1,'c':0} 
            }
        
        # avoid repetive assignment
        self.val_fnc = lambda data: update_theta_distribution(data, self.num_theta, self.theta_prior_val, self.theta_density, self.item_param_dict)
    
    def test_single_log(self):
        log = [(0,0)] 
        posterior = self.val_fnc(log)
        target_posterior = [0.365529, 0.5, 0.134471]
        for x in range(self.num_theta):
            self.assertAlmostEqual(posterior[x], target_posterior[x], places=6)


        log = [(0,1)] 
        posterior = self.val_fnc(log)
        target_posterior = [0.134471, 0.5, 0.365529]
        for x in range(self.num_theta):
            self.assertAlmostEqual(posterior[x], target_posterior[x], places=6)


    def test_multiple_log(self):
        log = [(0,0),(1,0)]
        posterior = self.val_fnc(log)
        target_posterior = [0.611306, 0.361288, 0.027407]
        for x in range(self.num_theta):
            self.assertAlmostEqual(posterior[x], target_posterior[x], places=6)


        log = [(0,0),(1,1)]
        posterior = self.val_fnc(log)
        target_posterior = [0.219818, 0.582237, 0.197945]
        for x in range(self.num_theta):
            self.assertAlmostEqual(posterior[x], target_posterior[x], places=6)

        log = [(0,1),(1,0)]
        posterior = self.val_fnc(log)
        target_posterior = [0.34039, 0.546848, 0.112762]
        for x in range(self.num_theta):
            self.assertAlmostEqual(posterior[x], target_posterior[x], places=6)

        log = [(0,1),(1,1)]
        posterior = self.val_fnc(log)
        target_posterior = [0.067323, 0.484724, 0.447953]
        for x in range(self.num_theta):
            self.assertAlmostEqual(posterior[x], target_posterior[x], places=6)

if __name__ == "__main__":
    unittest.main()
