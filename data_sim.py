import random
root_dir = '/home/junchen/git/pyirt/'
import sys
# import matplotlib.pyplot as plt
sys.path.insert(0,root_dir)
import numpy as np
import solver

num_sim = 1000
theta_vec = np.linspace(-4.0,4.0, num = num_sim)
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
# generate 100 items
response_seq = [int(random.random() <= solver.optimizer.irt_2PL.irt_fnc(theta, true_beta)) for theta in theta_vec]

root_dir = '/home/junchen/git/pyirt/'
with open(root_dir + 'multi_theta_data.txt','w') as f:
    for i in range(num_sim):
        f.write('{},{}\n'.format(response_seq[i], theta_vec[i]))
