pyirt
=====
[![Build Status](https://img.shields.io/travis/junchenfeng/pyirt/master.svg?style=flat)](https://travis-ci.org/junchenfeng/pyirt)
[![Coverage Status](https://coveralls.io/repos/github/junchenfeng/pyirt/badge.svg?branch=master)](https://coveralls.io/github/junchenfeng/pyirt?branch=master)
[![Code Health](https://landscape.io/github/junchenfeng/pyirt/master/landscape.svg?style=flat)](https://landscape.io/github/junchenfeng/pyirt/master)
[![Download](https://img.shields.io/pypi/dm/pyirt.svg?style=flat)](https://pypi.python.org/pypi/pyirt)
[![License](https://img.shields.io/pypi/l/pyirt.svg?style=flat)](https://pypi.python.org/pypi/pyirt)


A python library of IRT algorithm designed to cope with sparse data structure.

- Current version is in early development stage. Use at your own peril.
- built and test under py3.3. py2.7 compatibility is tested in my own
  environment. 


# Demo
```python
from pyirt import irt

src_fp = open(file_path,'r')

# alternatively, pass in list of tuples in the format of [(user_id, item_id, ans_boolean)]
# ans_boolean is 0/1.


# (1)Run by default
item_param, user_param = irt(src_fp)

# (2)Supply bounds
item_param, user-param = irt(src_fp, theta_bnds = [-5,5], alpha_bnds=[0.1,3], beta_bnds = [-3,3])

# (3)Supply guess parameter
guessParamDict = {1:{'c':0.0}, 2:{'c':0.25}}
item_param, user_param = irt(src_fp, in_guess_param = guessParamDict)
```


I.Model Specification
===================

## MMLE
The current version supports MMLE algorithm and unidimension two parameter
IRT model. There is a backdoor method to specify the guess parameter but there
is not active estimation.

The prior distribution of theta is **uniform**.

There is no regularization in alpha and beta estimation. Therefore, the default
algorithm uses boundary on the parameter to prevent over-fitting and deal with
extreme cases where almost all responses to the item is right.

## Theta estimation
The package offers two methods to estimate theta, given item parameters: Bayesian and MLE. <br>
The estimation procedure is quite primitive. For examples, see the test case.  

II.Sparse Data Structure
==========

In non-test learning dataset, missing data is the common. Not all students
finish all the items. When the number of students and items are large, the data
can be extremely sparse.

The package deals with the sparse structure in two ways:
- Efficient memory storage. Use collapsed list to index data. The memory usage
  is about 3 times of the text data file. If the workstation has 6G free
memory, it can handle 2G data file. Most other IRT package will definitely
break.

- No joint estimation. Under IRT's conditional independence assumption,
  estimate each item's parameter is consistent but inefficient. To avoid
reverting a giant jacobian matrix, the item parameters are estimated seprately. 


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

Currently the model only works well with 0/1 flag but will **NOT** raise error for
other types.


V.Note
=======


## Minimization solver
The scipy minimize is as good as cvxopt.cp and matlab fmincon on item parameter
estimation to the 6th decimal point, which can be viewed as identical for all
practical purposes.

However, the convergence is pretty slow. It requires about 10k obeverations per
item to recover the parameter to the 0.01 precision.


VII.Acknowledgement
==============
The algorithm is described in details by Bradey Hanson(2000), see in the
literature section. I am grateful to Mr.Hanson's work.

[Chaoqun Fu](https://github.com/fuchaoqun)'s comment leads to the (much better) API design. 

[Dawei Chen](https://github.com/mvj3) and [Lei Wang](https://github.com/wlbksy) contributed to the code.

