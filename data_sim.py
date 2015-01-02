import random
root_dir = '/home/junchen/git/pyirt/'
import sys
# import matplotlib.pyplot as plt
sys.path.insert(0,root_dir)
import numpy as np
import solver
theta_vec = np.linspace(-4.0,4.0,num=30)
beta = np.array([0.0])

#prob_vec = [irt_fnc(theta, beta[0],alpha=2.0) for theta in theta_vec]
#plt.plot(theta_vec, prob_vec)
#plt.show()

'''
The function checks out
Now simulate the data
'''

# simulate one batch
true_beta = 0.0
true_theta = 1.0
true_prob = solver.optimizer.irt_2PL.irt_fnc(true_theta, true_beta)

# generate 100 items
response_seq = [int(random.random() <= true_prob) for i in range(1000)]

root_dir = '/home/junchen/git/pyirt/'
with open(root_dir + 'single_param_data.txt','w') as f:
    for res in response_seq:
        f.write(str(res)+'\n')
