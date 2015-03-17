import unittest
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import solver

class TestBayesianEstimator(unittest.TestCase):
    bad_log = [[1,(1,8,0)]] # not very informative
    good_log= [[0,(1,8,0)]] # very informative


    def test_informative_log(self):
        uni_est = solver.theta_estimator.bayesian_estimator()
        uni_est.set_prior(-4,4,9,'uniform')

        beta_est = solver.theta_estimator.bayesian_estimator()
        beta_est.set_prior(-4,4,9,'beta')
        uni_est.update(self.good_log)

        uni_theta_posterior = uni_est.get_estimator()
        self.assertTrue(abs(uni_theta_posterior+3.413)<1e-2)

        beta_est.update(self.good_log)
        beta_theta_posterior = beta_est.get_estimator()
        self.assertTrue(abs(beta_theta_posterior+2.14)<1e-2)




    def test_uninformative_log(self):
        uni_est = solver.theta_estimator.bayesian_estimator()
        uni_est.set_prior(-4,4,9,'uniform')

        beta_est = solver.theta_estimator.bayesian_estimator()
        beta_est.set_prior(-4,4,9,'beta')

        uni_est.update(self.bad_log)
        uni_theta_posterior = uni_est.get_estimator()
        self.assertTrue(abs(uni_theta_posterior-0.0)<1e-2)

        beta_est.update(self.bad_log)
        beta_theta_posterior = beta_est.get_estimator()
        self.assertTrue(abs(beta_theta_posterior-0.0)<1e-2)
if __name__ == '__main__':
    unittest.main()
