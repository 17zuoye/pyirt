# -*- coding:utf-8 -*-

'''
Created on 2015/1/2

@author: junchen
'''
import numpy as np

def irt_fnc(theta, beta, alpha=1.0, c=0.0):
    # beta is item difficulty
    # theta is respondent capability

 
    prob = (1.0-c) / (1 + np.exp(-alpha*(theta-beta)))
    return prob

