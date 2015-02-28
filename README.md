pyirt
=====

A python library of IRT algorithm designed to cope with sparse data structure.

The algorithm is described in details by Bradey Hanson(2000), see in the
literature section. We are grateful to Mr.Hanson's word.

- Current version is in early development stage. Use at your own peril.


Model Specification
===================

The current version only supports MMLE algorithm and unidimension two parameter
IRT model.

The prior distribution of theta is uniform rather than beta.


What's New
==========

IRT model is developed for offline test that has few missing data. However,
when try to calibrate item parameters for online testing bank, such assumption
breaks down and the algorithm runs into sparse data problem, as well as severe
missing data problem.

This pacakge uses list rather than matrix format to deal with sparse data.

As for now, missing data are assumed to be ignorable.


run
===
from pyirt import *

(1)load data

data, param = utl.loader.load_sim_data('pyirt/data/sim_data.txt')

(2) setup solver

model = solver.model.IRT_MMLE_2PL()

model.load_data(data)

model.load_config()

model.solve_EM()

(3) print out the result

utl.tools.parse_item_paramer(model.item_param_dict)



requirement
===========
numpy,scipy




Performance Check
=======
The likelihood is not stable in terms of converging. 
To use max iteration improve the performance compared to the MIRT/LTM package

Simulation shows that (with proper bounds), the pyirt is more stable in
estimating the 2PL model when the alpha is within 0.25~3 and the beta is within
-2~4. The performance improves because it prevents alpha and beta going too
large when the item is too easy. 



ToDos
===========

(1) The solver cannot handle polytomous answers.

(2) The solver cannot handle multi-dimensional data.

(3) The solver cannot handle group constraints.
