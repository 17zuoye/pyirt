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
response_matrix = np.zeros((num_theta, num_beta))

'''
create the matrix
'''
for i in range(num_theta):
    for j in range(num_beta):
        response_prob = utl.tools.irt_fnc(theta_vec[i], beta_vec[j])
        response_matrix[i,j] = int(random.random() <= response_prob)

with open(root_dir + 'data/matrix_full_data.txt','w') as f:
    for i in range(num_theta):
        for j in range(num_beta):
            f.write('{},{},{}\n'.format(response_matrix[i,j], theta_vec[i], beta_vec[j]))
