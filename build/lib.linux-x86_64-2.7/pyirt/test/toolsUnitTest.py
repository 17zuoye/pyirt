import unittest
from pyirt import *


class TestSequenceFunctions(unittest.TestCase):


    def test_irt_fnc(self):
        # make sure the shuffled sequence does not lose any elements
        prob = utl.tools.irt_fnc(0.0,0.0,1.0)
        self.assertEqual(prob, 0.5)

if __name__ == '__main__':
    unittest.main()
