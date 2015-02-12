import unittest
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)
import solver, utl
import math
import numpy as np


class TestLinearSolver(unittest.TestCase):
    def setUp(self):
        # initialize the data
        n = 5000
        self.alpha = 1.5
        self.beta = -2.0
        theta_vec=np.random.normal(loc = 0.0, scale =1.0, size = n)
        y1 = []
        y0 = []
        for i in range(n):
            prob = utl.tools.irt_fnc(theta_vec[i], self.beta, self.alpha)
            if prob>np.random.uniform():
                y1.append(1.0)
                y0.append(0.0)
            else:
                y1.append(0.0)
                y0.append(1.0)
        response_data = [y1, y0]

        # initialize optimizer
        self.solver = solver.optimizer.irt_2PL_Optimizer()
        self.solver.load_res_data(response_data)
        self.solver.set_theta(theta_vec)

    def test_quadratic(self):
        self.solver.set_initial_guess((0.0,1.0))
        est_param_linear = self.solver.solve_param_linear(False)
        #est_param_gradient = self.solver.solve_param_gradient(False)
        import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

if __name__ == '__main__':
    unittest.main()

