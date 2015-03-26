import unittest


from ..solver import theta_estimator

class TestBayesianEstimator(unittest.TestCase):
    bad_log = [[1,(1,8,0)]] # not very informative
    good_log= [[0,(1,8,0)]] # very informative


    def test_bayesian_informative_log(self):
        uni_est = theta_estimator.bayesian_estimator()
        uni_est.set_prior(-4,4,9,'uniform')

        beta_est = theta_estimator.bayesian_estimator()
        beta_est.set_prior(-4,4,9,'beta')
        uni_est.update(self.good_log)

        uni_theta_posterior = uni_est.get_estimator()
        self.assertTrue(abs(uni_theta_posterior+3.413)<1e-2)

        beta_est.update(self.good_log)
        beta_theta_posterior = beta_est.get_estimator()
        self.assertTrue(abs(beta_theta_posterior+2.14)<1e-2)




    def test_bayesian_uninformative_log(self):
        uni_est = theta_estimator.bayesian_estimator()
        uni_est.set_prior(-4,4,9,'uniform')

        beta_est = theta_estimator.bayesian_estimator()
        beta_est.set_prior(-4,4,9,'beta')

        uni_est.update(self.bad_log)
        uni_theta_posterior = uni_est.get_estimator()
        self.assertTrue(abs(uni_theta_posterior-0.0)<1e-1)

        beta_est.update(self.bad_log)
        beta_theta_posterior = beta_est.get_estimator()
        self.assertTrue(abs(beta_theta_posterior-0.0)<1e-1)

    def test_mle_log(self):
        mle_est = theta_estimator.MLE_estimator()
        theta_hat_bad = mle_est.update(self.bad_log)
        theta_hat_good = mle_est.update(self.good_log)

        self.assertTrue(abs(theta_hat_bad-3.69)<1e-2)
        self.assertTrue(abs(theta_hat_good+4.0)<1e-2)


if __name__ == '__main__':
    unittest.main()
