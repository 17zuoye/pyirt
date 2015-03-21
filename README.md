pyirt
=====

A python library of IRT algorithm designed to cope with sparse data structure.

The algorithm is described in details by Bradey Hanson(2000), see in the
literature section. We are grateful to Mr.Hanson's word.

- Current version is in early development stage. Use at your own peril.


Model Specification
===================

## MMLE
The current version supports MMLE algorithm and unidimension two parameter
IRT model. There is a backdoor method to specify the guess parameter but there
is not active estimation.

The prior distribution of theta is uniform rather than beta.

There is no regularization in alpha and beta estimation. Therefore, the default
algorithm uses boundary on the parameter to prevent over-fitting and deal with
extreme cases where almost all responses to the item is right.

## Theta estimation
The package offers two methods to estimate theta, given item parameters: Bayesian and MLE. <br>
The estimation procedure is quite primitive. For examples, see the test case.  

What's New
==========

IRT model is developed for offline test that has few missing data. However,
when try to calibrate item parameters for online testing bank, such assumption
breaks down and the algorithm runs into sparse data problem, as well as severe
missing data problem.

This pacakge offers a few modifications to accomendate large dataset:
(1) use list and dictionary rather than matrix to represent data. Because the
response data is sparse, list+dict is far more efficient.

As for now, missing data are assumed to be ignorable.

Config File
===========
The pacakge comes with a default config file, which could be modified to suit
specific needs.

*USER* section gives the range of the theta paramter and the grid, [-4,4] and
0.5 = (4-(-4))/(17-1) by default.

*ITEM* section gives the range of alpha and beta paramter.

*SOLVER* sections outlines the configuration for the solver routine. 

is-constraintd: 0/1, whether to use constrained algorithm or not. Constrained
by default.<br>
type: linear/gradient. the optimization type. gradient by default<br>
max-iter: the max times of iteration in MMLE. 10 by default although that may
be too high.<br>
tol: the stop condition threshold, 1e-3 by default. The algorithm computes the
average likelihood per log at the end of the iteration. If the likelihood
increases less than the threshold, stops.

Example
=========
from pyirt import *
import ConfigParser
import io

(1) Load data

data, param = utl.loader.load _sim_data('pyirt/data/sim _data.txt')

(2) Set the guess parameter

eids = list(set([x[1] for x in run_data]))

guess_ param _dict = {}

for eid in eids:<br>
    guess_ param _dict[eid] = {'c':0,'update_c':False}


(2) Setup solver

model = solver.model.IRT _MMLE _2PL()

model.load _data(data)

** the config file has to be read in by io system to accomendate hdfs system** 

config = ConfigParser.RawConfigParser(allow _no _value=True)
config.readfp(io.BytesIO(config-text))

model.load _config(config)

model.load _guess _param(guess _param _dict)

(3) Solve

model.solve _EM()

(4) Get the parameter

item_param _dict = model.get_item _param()

user_param _dict = model.get_user _param()


requirement
===========
numpy,scipy




Performance Check
=======
The likelihood is not stable in terms of converging. 
To use max iteration improve the performance compared to the MIRT/LTM package

Simulation shows that (with proper bounds), the pyirt is more stable in
estimating the 2PL model when the item parameters are constrained.

The performance improves because it prevents alpha and beta going too
large when the item is too easy. 



ToDos
===========

(1) The solver cannot handle polytomous answers.

(2) The solver cannot handle multi-dimensional data.

(3) The solver cannot handle group constraints.
