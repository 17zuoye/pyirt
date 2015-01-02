'''
load the data
'''
root_dir = '/home/junchen/git/pyirt/'
import sys

sys.path.insert(0,root_dir)
import solver

with open(root_dir + 'multi_theta_data.txt','r') as f:
    res_data = []
    theta_vec = []
    for line in f:
        segs = line.strip().split(',')
        res_data.append(int(segs[0]))
        theta_vec.append(float(segs[1]))



'''
Set the environment for maximizer
'''
opt_worker = solver.optimizer.irt_2PL()

opt_worker.load_res_data(res_data)

opt_worker.setparam(theta_vec)
opt_worker.solve_beta_direct()
#print opt_worker.x

opt_worker.solve_beta_gradient()
print opt_worker.x
