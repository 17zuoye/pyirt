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

    def test_log_likelihood(self):
        # raise error
        with self.assertRaisesRegexp(ValueError,'Slope/Alpha should not be zero or negative.'):
             utl.tools.log_likelihood_2PL(0.0, 1.0, 0.0,-1.0,0.0)

        # the default model, log likelihood is log(0.5)
        ll = utl.tools.log_likelihood_2PL(1.0, 0.0, 0.0,1.0,0.0)
        self.assertEqual(ll, math.log(0.5))
        ll = utl.tools.log_likelihood_2PL(0.0, 1.0, 0.0,1.0,0.0)
        self.assertEqual(ll, math.log(0.5))

        # check the different model
        ll = utl.tools.log_likelihood_2PL(1.0,0.0, 1.0,1.0,0.0)
        self.assertEqual(ll, math.log(1.0/(1.0+math.exp(-1.0))))

        ll = utl.tools.log_likelihood_2PL(0.0,1.0, 1.0,1.0,0.0)
        self.assertEqual(ll, math.log(1.0-1.0/(1.0+math.exp(-1.0))))


    def test_log_sum(self):
        # add up a list of small values
        log_prob = np.array([-135,-115,-125,-100])
        approx_sum = utl.tools.logsum(log_prob)
        exact_sum = 0
        for num in log_prob:
            exact_sum += math.exp(num)
        exact_sum = math.log(exact_sum)
        self.assertTrue(abs(approx_sum-exact_sum)<0.0000000001)



if __name__ == '__main__':
    unittest.main()
