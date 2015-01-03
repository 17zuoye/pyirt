root_dir = '/home/junchen/git/pyirt/'
import sys
# import matplotlib.pyplot as plt
sys.path.insert(0,root_dir)
import random
import numpy as np
import utl

num_theta = 500
num_beta = 5
num_alpha = 4
master_theta = [-2.0,0.0,0.5,1.0,1.5,2.0,2.5,3.0,3.5,4.0]

theta_vec = master_theta*(num_theta/10)
beta_vec = np.linspace(-2.0, 2.0, num = num_beta)
alpha_vec = np.linspace(0.5, 2.0, num= num_alpha)
'''
create the matrix
'''

with open(root_dir + 'data/matrix_full_data.txt','w') as f:
    for i in range(num_theta):
        for j in range(num_beta):
            for k in range(num_alpha):
                theta = theta_vec[i]
                beta = beta_vec[j]
                alpha_val = alpha_vec[k]
                response_prob = utl.tools.irt_fnc(theta, beta, alpha=alpha_val)
                response = int(random.random() <= response_prob)
                f.write('{},{},{},{}\n'.format(response, theta, beta, alpha_val))
