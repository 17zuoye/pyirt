import unittest
import os, sys
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

import utl
import math
import numpy as np

class TestIrtFunctions(unittest.TestCase):


    def test_irt_fnc(self):
        # make sure the shuffled sequence does not lose any elements
        prob = utl.tools.irt_fnc(0.0,0.0,1.0)
        self.assertEqual(prob, 0.5)
        # alpha should play no role
        prob = utl.tools.irt_fnc(0.0,0.0,2.0)
        self.assertEqual(prob, 0.5)
        # higher theta should have higher prob
        prob = utl.tools.irt_fnc(1.0,0.0,1.0)
        self.assertEqual(prob, 1.0/(1.0+math.exp(-1.0)))
        # cancel out by higher beta
        prob = utl.tools.irt_fnc(1.0,-1.0,1.0)
        self.assertEqual(prob, 0.5)
        # test for c as limit situation
        prob = utl.tools.irt_fnc(-99,0.0,1.0, 0.25)
        self.assertTrue(abs(prob-0.25)<1e-5)
        prob = utl.tools.irt_fnc(99,0.0,1.0, 0.25)
        self.assertTrue(abs(prob-1.0)<1e-5)


    def test_log_likelihood(self):
        # raise error
        #with self.assertRaisesRegexp(ValueError,'Slope/Alpha should not be zero or negative.'):
        #     utl.clib.log_likelihood_2PL(0.0, 1.0, 0.0,-1.0,0.0)

        # the default model, log likelihood is log(0.5)
        ll = utl.clib.log_likelihood_2PL(1.0, 0.0, 0.0,1.0,0.0)
        self.assertEqual(ll, math.log(0.5))
        ll = utl.clib.log_likelihood_2PL(0.0, 1.0, 0.0,1.0,0.0)
        self.assertEqual(ll, math.log(0.5))

        # check the different model
        ll = utl.clib.log_likelihood_2PL(1.0,0.0, 1.0,1.0,0.0)
        self.assertEqual(ll, math.log(1.0/(1.0+math.exp(-1.0))))

        ll = utl.clib.log_likelihood_2PL(0.0,1.0, 1.0,1.0,0.0)
        self.assertEqual(ll, math.log(1.0-1.0/(1.0+math.exp(-1.0))))

        # check a real value
        ll = utl.clib.log_likelihood_2PL(0.0,1.0,-1.1617696779178492,1.0,0.0)

        self.assertTrue(abs(ll + 0.27226272946920399)<0.0000000001)

        # check if it handles c correctly
        ll = utl.clib.log_likelihood_2PL(1.0, 0.0, 0.0,1.0,0.0, 0.25)
        self.assertEqual(ll, math.log(0.625))
        ll = utl.clib.log_likelihood_2PL(0.0, 1.0, 0.0,1.0,0.0,0.25)
        self.assertEqual(ll, math.log(0.375))


    def test_log_sum(self):
        # add up a list of small values
        log_prob = np.array([-135,-115,-125,-100])
        approx_sum = utl.tools.logsum(log_prob)
        exact_sum = 0
        for num in log_prob:
            exact_sum += math.exp(num)
        exact_sum = math.log(exact_sum)
        self.assertTrue(abs(approx_sum-exact_sum)<1e-10)

    def test_log_item_gradient(self):
        delta = 0.00001
        y1 = 1.0
        y0 = 2.0
        theta = -2.0
        alpha = 1.0
        beta = 0.0
        # simulate the gradient
        true_gradient_approx_beta = (utl.clib.log_likelihood_2PL(y1,y0,theta,alpha,beta+delta) - \
            utl.clib.log_likelihood_2PL(y1,y0,theta,alpha,beta))/delta
        true_gradient_approx_alpha =  (utl.clib.log_likelihood_2PL(y1,y0,theta,alpha+delta,beta) - \
            utl.clib.log_likelihood_2PL(y1,y0,theta,alpha,beta))/delta
        # calculate
        calc_gradient = utl.clib.log_likelihood_2PL_gradient(y1,y0,theta,alpha,beta)

        self.assertTrue(abs(calc_gradient[0] - true_gradient_approx_beta ) < 1e-4)
        self.assertTrue(abs(calc_gradient[1] - true_gradient_approx_alpha) < 1e-4)

        # simulate the gradient with c
        c = 0.25
        true_gradient_approx_beta = (utl.clib.log_likelihood_2PL(y1,y0,theta,alpha,beta+delta,c) - \
            utl.clib.log_likelihood_2PL(y1,y0,theta,alpha,beta,c))/delta
        true_gradient_approx_alpha =  (utl.clib.log_likelihood_2PL(y1,y0,theta,alpha+delta,beta,c) - \
            utl.clib.log_likelihood_2PL(y1,y0,theta,alpha,beta,c))/delta
        # calculate
        calc_gradient = utl.clib.log_likelihood_2PL_gradient(y1,y0,theta,alpha,beta,c)

        self.assertTrue(abs(calc_gradient[0] - true_gradient_approx_beta ) < 1e-4)
        self.assertTrue(abs(calc_gradient[1] - true_gradient_approx_alpha) < 1e-4)


    def test_log_factor_gradient(self):
        delta = 0.00001
        y1 = 1.0
        y0 = 2.0
        theta = -2.0
        alpha = 1.0
        beta = 0.0
        # simulate the gradient
        true_gradient_approx_theta = (utl.clib.log_likelihood_2PL(y1,y0,theta+delta,alpha,beta) - \
            utl.clib.log_likelihood_2PL(y1,y0,theta,alpha,beta))/delta
       # calculate
        calc_gradient = utl.tools.log_likelihood_factor_gradient(y1,y0,theta,alpha,beta)

        self.assertTrue(abs(calc_gradient - true_gradient_approx_theta ) < 1e-4)


        # simulate the gradient
        c = 0.25
        true_gradient_approx_theta = (utl.clib.log_likelihood_2PL(y1,y0,theta+delta,alpha,beta,c) - \
            utl.clib.log_likelihood_2PL(y1,y0,theta,alpha,beta,c))/delta
       # calculate
        calc_gradient = utl.tools.log_likelihood_factor_gradient(y1,y0,theta,alpha,beta,c)

        self.assertTrue(abs(calc_gradient - true_gradient_approx_theta ) < 1e-4)


    def test_log_factor_hessian(self):
        delta = 0.00001
        y1 = 1.0
        y0 = 2.0
        theta = -2.0
        alpha = 1.0
        beta = 0.0
        c = 0.25
        # simulate the gradient
        true_hessian_approx_theta = (utl.tools.log_likelihood_factor_gradient(y1,y0,theta+delta,alpha,beta,c) - \
            utl.tools.log_likelihood_factor_gradient(y1,y0,theta,alpha,beta,c))/delta
       # calculate
        calc_hessian = utl.tools.log_likelihood_factor_hessian(y1,y0,theta,alpha,beta,c)

        self.assertTrue(abs(calc_hessian - true_hessian_approx_theta ) < 1e-4)

if __name__ == '__main__':
    unittest.main()
