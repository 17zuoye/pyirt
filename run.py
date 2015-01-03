'''
load the data
'''

#TODO: the data have problems in fiiting multiple data

root_dir = '/home/junchen/git/pyirt/'
import sys

sys.path.insert(0,root_dir)
import solver
import numpy as np

with open(root_dir + 'data/matrix_full_data.txt','r') as f:
    res_data = []
    theta_vec = []
    beta_vec = []
    alpha_vec = []
    for line in f:
        segs = line.strip().split(',')
        res_data.append(float(segs[0]))
        theta_vec.append(float(segs[1]))
        beta_vec.append(float(segs[2]))
        alpha_vec.append(float(segs[3]))

# reconstruct the matrix
beta_array = np.array(beta_vec)
res_data_array = np.array(res_data)
theta_array = np.array(theta_vec)
alpha_array = np.array(alpha_vec)

unique_beta = np.unique(beta_array)
num_beta = len(unique_beta)

unique_alpha = np.unique(alpha_array)
num_alpha = len(unique_alpha)

# the data is organized by dictionary
data_dict = {}
for beta in unique_beta:
    data_dict[beta] = {}
    for alpha in unique_alpha:

        sample_idx = np.logical_and(beta_array == beta,
                                    alpha_array == alpha)
        sample_res_data = res_data_array[sample_idx, ]
        sample_theta = theta_array[sample_idx, ]

        data_dict[beta][alpha] = {'res_data':sample_res_data, 'theta':sample_theta}
        #print beta, alpha, np.mean(sample_res_data)
'''
Set the environment for maximizer
'''

opt_worker = solver.optimizer.irt_2PL()
opt_worker.setInitialGuess((0.0,1.0))
opt_worker.setBounds([(-4.0,4.0),(0.25,2.5)])

for beta, data_vec in data_dict.iteritems():
    for alpha, data in data_vec.iteritems():

        opt_worker.load_res_data(data['res_data'])
        opt_worker.setparam(data['theta'])

        result_l = opt_worker.solve_param_linear()
        result_g = opt_worker.solve_param_gradient()

        print (beta, alpha), result_l, result_g

