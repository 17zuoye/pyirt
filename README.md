pyirt
=====
[![Build Status](https://img.shields.io/travis/17zuoye/pyirt/master.svg?style=flat)](https://travis-ci.org/17zuoye/pyirt)
[![Coverage Status](https://coveralls.io/repos/17zuoye/pyirt/badge.svg)](https://coveralls.io/r/17zuoye/pyirt)
[![Health](https://landscape.io/github/17zuoye/pyirt/master/landscape.svg?style=flat)](https://landscape.io/github/17zuoye/pyirt/master)
[![Download](https://img.shields.io/pypi/dm/pyirt.svg?style=flat)](https://pypi.python.org/pypi/pyirt)
[![License](https://img.shields.io/pypi/l/pyirt.svg?style=flat)](https://pypi.python.org/pypi/pyirt)



A python library of IRT algorithm designed to cope with sparse data structure.

- Current version is in early development stage. Use at your own peril.


I.Model Specification
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

II.What's New
==========

IRT model is developed for offline test that has few missing data. However,
when try to calibrate item parameters for online testing bank, such assumption
breaks down and the algorithm runs into sparse data problem, as well as severe
missing data problem.


## Missing Data

As for now, missing data are assumed to be ignorable.

III.Default Config
===========
## Exposed
The theta paramter range from [-4,4] and a step size of 0.8 by default.

Alpha is bounded by [0.25,2] and beta is bounded by [-2,2], which goes with the user ability
specifiation. 

Guess parameter is set to 0 by default, but one could supply a dictionary with eid as key and c as value.


## Hidden
The default solver is L-BFGS-G. 

The default max iteration is 10.

The stop condition threshold is 1e-3 by default. The algorithm computes the
average likelihood per log at the end of the iteration. If the likelihood
increament is less than the threshold, stops.

IV.Data Format
=========
The file is expected to be comma delimited. 

The three columns are uid, eid, result-flag.

Currently the model only works well with 0/1 flag but will NOT raise error for
other types.



V.Example
=========
```python
from pyirt import *

src_fp = open(file_path,'r')

# alternatively, pass in list of tuples in the format of [(uid, eid, atag),...]


# (1)Run by default
item_param, user_param = irt(src_fp)

# (2)Supply bnds
item_param, user-param = irt(src_fp, theta_bnds = [-5,5], beta_bnds = [-3,3])

# (3)Supply guess parameter
guessParamDict = {1:{'c':0.0}, 2:{'c':0.25}}

item_param, user_param = irt(src_fp, in_guess_param = guessParamDict)
```


VI.Performance
=======

## Cython Optimization
The crucial function is log likelihood evaluation, which is implemented in
Cython. At 1 million records scale, it halves the run time.

## Why no parallel
Multi-processing in Python does not accpet class method.

In addition, none of the calculation is particular computation heavy. The
communication cost over-weighs the parallel gain.

## Minimization solver
The scipy minimize is as good as cvxopt.cp and matlab fmincon on item parameter
estimation to the 6th decimal point, which can be viewed as identical for all
practical purposes.

However, the convergence is pretty slow. It requires about 10k obeverations per
item to recover the parameter to the 0.01 precision.


VII.ToDos
===========

## Models
(1) The solver cannot handle polytomous answers.

(2) The solver cannot handle multi-dimensional data.

(3) The solver cannot handle group constraints.


## BIG DATA
bdm is a work around when the data are too much for memory. However,berkeley db
is quite hard to install on operating system. Therefore, although in utl module
there are code snips for dbm trick. It is not standard shipping.



VIII.Acknowledgement
==============
The algorithm is described in details by Bradey Hanson(2000), see in the
literature section. I am grateful to Mr.Hanson's work.

The python implementation is benefited greatly from the comments and suggestions from Chaoqun Fu and Dawei Chen.

