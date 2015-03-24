import unittest
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import solver, utl
import math
import numpy as np



class TestItemSolverNoGuess(unittest.TestCase):
    def setUp(self):
        # initialize the data
        n = 10000
        self.alpha = 1.5
        self.beta = -2.0
        theta_vec=np.random.normal(loc = 0.0, scale =1.0, size = n)
        y1 = []
        y0 = []
        for i in range(n):
            # generate the two parameter likelihood
            prob = utl.tools.irt_fnc(theta_vec[i], self.beta, self.alpha)
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
        self.solver = solver.optimizer.irt_2PL_Optimizer()
        self.solver.load_res_data(response_data)
        self.solver.set_theta(theta_vec)
        self.solver.set_c(0)

    def test_linear_unconstrained(self):
        self.solver.set_initial_guess((0.0,1.0))
        est_param = self.solver.solve_param_linear(is_constrained = False)
        self.assertTrue(abs(est_param[0]-self.beta)<0.07 and abs(est_param[1]-self.alpha)<0.05)

    def test_linear_constrained(self):
        self.solver.set_initial_guess((0.0,1.0))
        self.solver.set_bounds([(-4.0,4.0),(0.25,2)])
        est_param = self.solver.solve_param_linear(is_constrained = True)
        self.assertTrue(abs(est_param[0]-self.beta)<0.05 and abs(est_param[1]-self.alpha)<0.05)



    #def test_gradient_unconstrained(self):
    #    self.solver.set_initial_guess((0.0,2.0))
    #    est_param = self.solver.solve_param_gradient(is_constrained = False)
    #    import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

    def test_gradient_constrained(self):
        self.solver.set_initial_guess((0.0,1.0))
        self.solver.set_bounds([(-4.0,4.0),(0.25,2)])
        est_param = self.solver.solve_param_gradient(is_constrained = True)
        self.assertTrue(abs(est_param[0]-self.beta)<0.1 and abs(est_param[1]-self.alpha)<0.1)



class TestUserSolver(unittest.TestCase):
    def setUp(self):
        # initialize the data
        n = 1000
        self.theta = 1.5
        self.alpha_vec = np.random.random(n)+1
        self.beta_vec  = np.random.normal(loc = 0.0, scale =2.0, size = n)
        self.c_vec = np.zeros(n)
        y1 = []
        y0 = []
        for i in range(n):
            # generate the two parameter likelihood
            prob = utl.tools.irt_fnc(self.theta, self.beta_vec[i], self.alpha_vec[i])
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
        self.solver = solver.optimizer.irt_factor_optimizer()
        self.solver.load_res_data(response_data)
        self.solver.set_item_parameter(self.alpha_vec, self.beta_vec, self.c_vec)
        self.solver.set_bounds([(-6,6)])

    def test_linear_unconstrained(self):
        self.solver.set_initial_guess(0.0)
        est_param = self.solver.solve_param_linear(is_constrained = False)
        self.assertTrue(abs(est_param-self.theta)<0.1)

    def test_gradient_constrained(self):
        self.solver.set_initial_guess(0.0)
        self.solver.set_bounds([(-6,6)])
        est_param = self.solver.solve_param_gradient(is_constrained = True)
        self.assertTrue(abs(est_param-self.theta)<0.1)

    def test_hessian_unconstrained(self):
        self.solver.set_initial_guess(0.0)
        est_param = self.solver.solve_param_hessian()
        self.assertTrue(abs(est_param-self.theta)<0.1)


    def test_scalar_constrained(self):
        self.solver.set_initial_guess(0.0)
        self.solver.set_bounds((-6,6))
        est_param = self.solver.solve_param_scalar()
        self.assertTrue(abs(est_param-self.theta)<0.1)


if __name__ == '__main__':
    unittest.main()

