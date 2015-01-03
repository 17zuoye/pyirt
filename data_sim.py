root_dir = '/home/junchen/git/pyirt/'
import sys
# import matplotlib.pyplot as plt
sys.path.insert(0,root_dir)
import random
import numpy as np
import utl

num_theta = 1000
num_beta = 5
theta_vec = np.linspace(-4.0, 4.0, num = num_theta)
beta_vec = np.linspace(-2.0, 2.0, num = num_beta)

'''
create the matrix
'''

with open(root_dir + 'data/matrix_full_data.txt','w') as f:
    for i in range(num_theta):
        for j in range(num_beta):
            theta = theta_vec[i]
            beta = beta_vec[j]
            response_prob = utl.tools.irt_fnc(theta, beta)
            response = int(random.random() <= response_prob)
            f.write('{},{},{}\n'.format(response, theta, beta))
