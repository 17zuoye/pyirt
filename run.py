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

unique_theta = np.unique(theta_array)
num_theta = len(unique_theta)

# the data is organized by dictionary
opt_data_dict = {}
for beta in unique_beta:
    opt_data_dict[beta] = {}
    for alpha in unique_alpha:

        sample_idx = np.logical_and(beta_array == beta,
                                    alpha_array == alpha)
        sample_res_data = res_data_array[sample_idx, ]
        sample_theta = theta_array[sample_idx, ]

        opt_data_dict[beta][alpha] = {'res_data':sample_res_data, 'theta':sample_theta}
        #print beta, alpha, np.mean(sample_res_data)
'''
Set the environment for maximizer
'''
'''
opt_worker = solver.optimizer.irt_2PL_Optimizer()
opt_worker.set_initial_guess((0.0,1.0))
opt_worker.set_bounds([(-4.0,4.0),(0.25,2.5)])

for beta, data_vec in opt_data_dict.iteritems():
    for alpha, data in data_vec.iteritems():

        # generate app data where y1 and y0 are two separate list
        y1 = data['res_data']
        y0 = [1-x for x in y1]
        input_data= [y1,y0]

        opt_worker.load_res_data(input_data)
        opt_worker.set_theta(data['theta'])

        result_l = opt_worker.solve_param_linear()
        result_g = opt_worker.solve_param_gradient()

        print [beta, alpha], result_l, result_g
'''

# reset the sample data
sample_idx = alpha_array == alpha_array[1]
test_res_data = res_data_array[sample_idx, ]
test_theta_array = theta_array[sample_idx,]
test_beta_array = beta_array[sample_idx,]

log_data_set = []
for i in range(len(test_res_data)):
    # find the idx for item(beta)
    # find the idx for user(theta)
    item_idx = np.where(unique_beta==test_beta_array[i])[0][0]
    user_idx = np.where(unique_theta==test_theta_array[i])[0][0]
    log_data_set.append((user_idx,item_idx,test_res_data[i]))



'''
Joint estimation
sim_log_data = []
for i in range(num_theta):
    for j in range(num_beta):
       sample_idx = np.logical_and(test_beta_array == unique_beta[j],
                                    test_theta_array == unique_theta[i])
       log_res = test_res_data[sample_idx][0]
       sim_log_data.append((i,j,log_res))

'''

import ipdb; ipdb.set_trace()  # XXX BREAKPOINT
test_model = solver.model.IRT_MMLE_2PL()
test_model.load_response_data(log_data_set)
test_model.set_theta_prior()
test_model.solve_EM()

# print out the result
import ipdb; ipdb.set_trace()  # XXX BREAKPOINT

