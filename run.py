'''
load the data
'''
root_dir = '/home/junchen/git/pyirt/'
import sys

sys.path.insert(0,root_dir)
import solver

with open(root_dir + 'single_param_data.txt','r') as f:
    res_data = []
    for line in f:
        res_data.append(int(line.strip()))

'''
Set the environment for maximizer
'''
opt_worker = solver.optimizer.irt_2PL()

opt_worker.load_res_data(res_data)
theta = 1.0
opt_worker.setparam(theta)
opt_worker.solve_beta_direct()
print opt_worker.x
opt_worker.solve_beta_gradient()
print opt_worker.x

