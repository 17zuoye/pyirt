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
    for line in f:
        segs = line.strip().split(',')
        res_data.append(float(segs[0]))
        theta_vec.append(float(segs[1]))
        beta_vec.append(float(segs[2]))

# reconstruct the matrix
beta_array = np.array(beta_vec)
res_data_array = np.array(res_data)
theta_array = np.array(theta_vec)

unique_beta = np.unique(beta_array)
num_beta = len(unique_beta)

# the data is organized by dictionary
data_dict = {}
for beta in beta_vec:
    sample_idx = np.where(beta_array == beta)[0]
    sample_res_data = res_data_array[sample_idx, ]
    sample_theta = theta_array[sample_idx, ]
    data_dict[beta] = {'res_data':sample_res_data, 'theta':sample_theta}

'''
Set the environment for maximizer
'''

opt_worker = solver.optimizer.irt_2PL()
opt_worker.setInitialGuess(0.0)
opt_worker.setBounds([(-4.0,4.0)])

for beta, data in data_dict.iteritems():
    #if abs(beta-2.0) >= 0.001:
    #    continue
    opt_worker.load_res_data(data['res_data'])
    opt_worker.setparam(data['theta'])
    result_l = opt_worker.solve_param_linear()
    result_g = opt_worker.solve_param_gradient()

    print beta, result_l, result_g

