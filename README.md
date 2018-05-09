pyirt
=====
[![Build Status](https://img.shields.io/travis/junchenfeng/pyirt/master.svg?style=flat)](https://travis-ci.org/junchenfeng/pyirt)
[![Coverage Status](https://coveralls.io/repos/github/junchenfeng/pyirt/badge.svg?branch=master)](https://coveralls.io/github/junchenfeng/pyirt?branch=master)
[![Code Health](https://landscape.io/github/junchenfeng/pyirt/master/landscape.svg?style=flat)](https://landscape.io/github/junchenfeng/pyirt/master)
[![Download](https://img.shields.io/pypi/dm/pyirt.svg?style=flat)](https://pypi.python.org/pypi/pyirt)
[![License](https://img.shields.io/pypi/l/pyirt.svg?style=flat)](https://pypi.python.org/pypi/pyirt)


A python library of IRT algorithm designed to cope with sparse data structure.

- built and test under py3.6. Python 2 compatibility is tested but not guaranteed.

# Installation

When install from github source code

```shell
pipenv --three
make
```

# Demo
```python
from pyirt import irt

src_fp = open(file_path,'r')

# alternatively, pass in list of tuples in the format of [(user_id, item_id, ans_boolean)]
# ans_boolean is 0/1.


# (1)Run by default
item_param, user_param = irt(src_fp)

# (2)Supply bounds
item_param, user-param = irt(src_fp, theta_bnds = [-4,4], alpha_bnds=[0.1,3], beta_bnds = [-3,3])

# (3)Supply guess parameter
guessParamDict = {1:{'c':0.0}, 2:{'c':0.25}}
item_param, user_param = irt(src_fp, in_guess_param = guessParamDict)
```

# MongoDb Integration

When dealing with big data, the memory limit of the single machine is usually the bottle neck.

pyirt ships with a pymongo integration that can handle millions of record (we tried 1 billion).

The mongo db connection config is in "settings.ini", whose format is the same as "settings.ini.example"

For usage, see
```python
python -m unittest tests.test_dao.TestDataSrc.test_from_mongo
```

Technical Documentation
=======================
See [wiki](https://github.com/17zuoye/pyirt/wiki)



Acknowledgement
==============
The algorithm is described in details by Bradey Hanson(2000), see in the literature section. I am grateful to Mr.Hanson's work.

[Chaoqun Fu](https://github.com/fuchaoqun)'s comment leads to the (much better) API design. 

[Dawei Chen](https://github.com/mvj3) and [Lei Wang](https://github.com/wlbksy) contributed to the code.

