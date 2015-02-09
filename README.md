pyirt
=====

A python library of IRT algorithm designed to cope with sparse data structure.

The algorithm is described in details by Bradey Hanson(2000), see in the
literature section. We are grateful to Mr.Hanson's word.

## Current version is in early development stage. Use at your own peril. ## 


Model Specification
===================

The current version only supports MMLE algorithm and unidimension two parameter
IRT model.

The prior distribution of theta is uniform rather than beta.


run
===
from pyirt import *
# load data
data, param = utl.loader.load_sim_data('pyirt/data/sim_data.txt')
# setup solver
model = solver.model.IRT_MMLE_2PL()
model.load_data(data)
model.load_config()
model.solve_EM()

# print out the result
utl.tools.parse_item_paramer(model.item_param_dict)



requirement
===========
numpy,scipy




Performance Check
=======
There are some discrepancy needs to be closed


##LAST7 example:

MIRT estimation: 
Item, slope, intercept

1 0.584 1.093 

2 0.634 0.475 

3 0.993 1.054 

4 0.452 0.286 

5 0.436 1.091 

pyirt estimation:

1 0.41 1.42

2 0.7 0.24

3 0.62 1.03

4 0.34 0.15

5 0.29 1.54



Problem
===========

(1) The unconstrained solver cannot handle students that have all the answers right
or all the answers wrong.

(2) The solver cannot handle polytomous answers

(3) The solver cannot handle multi-dimensional data
