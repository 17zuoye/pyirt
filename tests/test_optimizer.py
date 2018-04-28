# -*- coding: utf-8 -*-
import os
import sys
RootDir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(RootDir)

import unittest
from pyirt.solver import optimizer
from pyirt.util import tools
import numpy as np


class TestItemSolverNoGuess(unittest.TestCase):
    '''
    题目估计beta误差较小(-1.99)，alpha误差较大(1.488)
    '''
    @classmethod
    def setUp(cls):
        # initialize the data
        n = 10000
        cls.alpha = 1.5
        cls.beta = -2.0
        np.random.seed(20170807)
        theta_vec = np.random.normal(loc=0.0, scale=1.0, size=n)
        y1 = []
        y0 = []
        for i in range(n):
            # generate the two parameter likelihood
            prob = tools.irt_fnc(theta_vec[i], cls.beta, cls.alpha)
            # generate the response sequence
            if prob >= np.random.uniform():
                y1.append(1.0)
                y0.append(0.0)
            else:
                y1.append(0.0)
                y0.append(1.0)
        # the response format follows the solver API
        response_data = [y1, y0]
        cls.init_theta_vec = theta_vec

        cls.solver = optimizer.irt_2PL_Optimizer()
        cls.solver.load_res_data(response_data)
        cls.solver.set_c(0)
        cls.solver.set_bounds([(-4.0, 4.0), (0.05, 2)])

    def test_linear_unconstrained(self):
        self.solver.set_theta(self.init_theta_vec)
        self.solver.set_initial_guess((0.0, 1.0))

        est_param = self.solver.solve_param_linear(is_constrained=False)
        self.assertTrue(abs(est_param[0] - self.beta) < 0.01)
        self.assertTrue(abs(est_param[1] - self.alpha) < 0.02)

    def test_linear_constrained(self):
        self.solver.set_theta(self.init_theta_vec)
        self.solver.set_initial_guess((0.0, 1.0))

        est_param = self.solver.solve_param_linear(is_constrained=True)
        self.assertTrue(abs(est_param[0] - self.beta) < 0.01)
        self.assertTrue(abs(est_param[1] - self.alpha) < 0.02)

    def test_gradient_unconstrained(self):
        self.solver.set_theta(self.init_theta_vec)
        self.solver.set_initial_guess((0.0, 1.0))

        est_param = self.solver.solve_param_gradient(is_constrained=False)
        self.assertTrue(abs(est_param[0] - self.beta) < 0.01)
        self.assertTrue(abs(est_param[1] - self.alpha) < 0.02)

    def test_gradient_constrained(self):
        self.solver.set_theta(self.init_theta_vec)
        self.solver.set_initial_guess((0.0, 1.0))

        est_param = self.solver.solve_param_gradient(is_constrained=True)
        self.assertTrue(abs(est_param[0] - self.beta) < 0.01)
        self.assertTrue(abs(est_param[1] - self.alpha) < 0.02)

    def test_data_for_solve_param_mix(self):
        # 该API用于EM算法
        expected_right_count = np.array([4.05604874e-08, 7.06740321e-06, 6.50532986e-04,
                                         3.18995908e-01, 1.04895900e+01, 1.25667422e+02,
                                         4.77649918e+02, 7.60813810e+02, 5.98748095e+02,
                                         2.85752301e+02, 1.42559210e+02])
        expected_wrong_count = np.array([5.44930352e-03, 8.43617536e-02, 9.54184900e-01,
                                         8.00842890e+00, 9.74576266e+01, 7.48633956e+02,
                                         1.46395898e+03, 1.04005287e+03, 2.71513420e+02,
                                         3.65580835e+01, 6.77263765e+00])
        self.solver.set_initial_guess((0.0, 1.0))
        self.solver.set_theta(np.linspace(-4, 4, num=11))
        self.solver.load_res_data([expected_right_count, expected_wrong_count])
        self.solver.solve_param_mix(is_constrained=True)
        self.assertTrue(bool("solve_param_mix has no exception!"))


class TestUserSolver(unittest.TestCase):
    '''
    用户参数估计相对误差较大，大约是1.57
    '''
    @classmethod
    def setUpClass(cls):
        # initialize the data
        n = 1000
        cls.theta = 1.5
        np.random.seed(20170807)
        alpha_vec = np.random.random(n) + 1  # 1-2
        beta_vec = np.random.normal(loc=0.0, scale=2.0, size=n)
        c_vec = np.zeros(n)
        y1 = []
        y0 = []
        for i in range(n):
            # generate the two parameter likelihood
            prob = tools.irt_fnc(cls.theta, beta_vec[i], alpha_vec[i])
            # generate the response sequence
            if prob >= np.random.uniform():
                y1.append(1.0)
                y0.append(0.0)
            else:
                y1.append(0.0)
                y0.append(1.0)
        # the response format follows the solver API
        response_data = [y1, y0]

        # initialize optimizer
        cls.solver = optimizer.irt_factor_optimizer()
        cls.solver.load_res_data(response_data)
        cls.solver.set_item_parameter(alpha_vec, beta_vec, c_vec)
        cls.solver.set_bounds([(-4, 4)])

    def test_linear_unconstrained(self):
        self.solver.set_initial_guess(0.0)
        est_param = self.solver.solve_param_linear(is_constrained=False)
        self.assertTrue(abs(est_param - self.theta) < 0.1)  # orig is 0.1

    def test_linear_constrained(self):
        self.solver.set_initial_guess(0.0)
        est_param = self.solver.solve_param_linear(is_constrained=True)
        self.assertTrue(abs(est_param - self.theta) < 0.1)  # orig is 0.1

    def test_gradient_unconstrained(self):
        self.solver.set_initial_guess(0.0)
        est_param = self.solver.solve_param_gradient(is_constrained=False)
        self.assertTrue(abs(est_param - self.theta) < 0.1)  # orig is 0.1

    def test_gradient_constrained(self):
        self.solver.set_initial_guess(0.0)
        est_param = self.solver.solve_param_gradient(is_constrained=True)
        self.assertTrue(abs(est_param - self.theta) < 0.1)  # orig is 0.1

    def test_hessian(self):
        # unconstrained
        self.solver.set_initial_guess(0.0)
        est_param = self.solver.solve_param_hessian()
        self.assertTrue(abs(est_param - self.theta) < 0.1)  # orig is 0.1

    def test_scalar(self):
        # constrained
        self.solver.set_initial_guess(0.0)
        self.solver.set_bounds((-4, 4))
        est_param = self.solver.solve_param_scalar()
        self.assertTrue(abs(est_param - self.theta) < 0.1)  # orig is 0.1


if __name__ == '__main__':
    unittest.main()
