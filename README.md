pyirt
=====

A python library of IRT algorithm designed to cope with sparse data structure.

The algorithm is described in details by Bradey Hanson(2000), see in the
literature section. We are grateful to Mr.Hanson's word.


Model Specification
===================

The current version only supports MMLE algorithm and unidimension two parameter
IRT model.

The prior distribution of theta is uniform rather than beta.


run
===


The parameters are set in the config.cfg.

An example:
model = solver.model.IRT_MMLE_2PL()
model.load_data(data)
model.load_config()
model.solve_EM()

The result is in model.item_param_dict and model.user_param

requirement
===========
numpy,scipy




Example
=======
## Simulated Data:


##LAST7 example:
There are some discrepancy, but qualitatively close.

MIRT estimation:
Item.1 0.584 1.093 
Item.2 0.634 0.475 
Item.3 0.993 1.054 
Item.4 0.452 0.286 
Item.5 0.436 1.091 

pyirt estimation:
Item.1 0.41 1.42
Item.2 0.7 0.24
Item.3 0.62 1.03
Item.4 0.34 0.15
Item.5 0.29 1.54

0 0.53 1.24
1 0.54 0.11
2 0.7 0.77
3 0.47 -0.09
4 0.4 1.39



Problem
===========

(1) The unconstrained solver cannot handle students that have all the answers right
or all the answers wrong.

(2) The solver cannot handle polytomous answers

(3) The solver cannot handle multi-dimensional data
