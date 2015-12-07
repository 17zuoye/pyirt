# generate the data
from math import exp, log

import numpy as np
from scipy.stats import norm
from collections import Counter
'''
 theta are drew from N(0,2)

 parameters are a = 1.0, b0 = 0, b1 = 1
 p(Y=0) = 1/G
 p(Y=1) = exp[a(theta-b0)]/G
 p(Y=2) = exp[a(theta-b0)+a(theta-b1)]/G
 G = 1+ exp[a(theta-b0)] + exp[a(theta-b0)+a(theta-b1)]
'''

def get_grade_prob(theta, As, Bs):
    num_b = len(Bs)
    if num_b <1:
        raise ValueError('beta vector is empty.')
    ps = [1]
    for i in range(num_b):
        p = exp(sum([As[k]*theta+Bs[k] for k in range(i+1)]))
        ps.append(p)
    G = sum(ps)
    prob_vec = [p/G for p in ps]
    return prob_vec

def simulate_response(theta,As,Bs):
    prob_vec = get_grade_prob(theta,As,Bs)
    y = np.random.choice(range(len(Bs)+1), p=prob_vec)
    return y

def get_simulate_sample(As, Bs, N=1000, M=5, sigma=1):
    thetas = []
    ys = []
    
    for i in range(N):
        theta = norm.rvs()*sigma
        thetas.append(theta)
        for j in range(M):
            ys.append([simulate_response(theta,As,Bs),i])
    return ys, thetas


def logsum(logp):
    # this version use list
    w = max(logp)
    logSump = w + log(sum([exp(x-w) for x in logp]))
    return logSump

def parse_params(params):
    n = len(params)
    As = [0.0 for x1 in range(n)]
    Bs = [0.0 for x1 in range(n)]
    for j in range(n):
        if j == 0:
            As[j] = params[1][j]
            Bs[j] = params[0][j]
        else:
            As[j] = params[1][j] - As[j-1]
            Bs[j] = params[0][j] - Bs[j-1]

    return As, Bs

if __name__ == '__main__':

    '''
    Did not converge, because without sufficient Y combinations, cannot pin down posterior very well
    # time to add to the main function
    '''


    num_obs = len(data)

    data_repo = {}
    for pair in data:
        uid = pair[1]
        y = pair[0]
        if uid not in data_repo:
            data_repo[uid]=[]
        data_repo[uid].append(y)

    uids = data_repo.keys()
    num_user = len(uids)

    # fit by MNlogit
    from statsmodels.discrete import discrete_model

    xs = []
    ys = []
    for i in range(num_obs):
        uid = data[i][1]
        y = data[i][0]
        theta = thetas[uid]
        xs.append([1.0, theta])
        ys.append(y)
    X = np.array(xs)
    Y = np.array(ys)
    true_model = discrete_model.MNLogit(Y,X)
    # fit and show
    true_mod_res = true_model.fit(disp=0)
    true_As, true_Bs = parse_params(true_mod_res.params)  # almost recover the parameters
    print(true_As, true_Bs) 



    # initialize theta and (a,bs)
    theta_min = -2.0
    theta_max = 2.0
    num_theta = 10
    step = (theta_max - theta_min)/(num_theta-1)
    theta_val = [theta_min + k0*step for k0 in range(num_theta)]
    theta_dist = [[1.0/num_theta for k1 in range(num_theta)] for i1 in xrange(num_user)]

    num_y = 3
    As = [1.0 for j0 in range(num_y-1)]
    Bs = [0.0 for j1 in range(num_y-1)]

    
    # converged, but not constrained!
    # alpha becomes smaller because the theta posterior is too concentrated
    # Shrink toward the mean may help
    for T in range(10):
        # calculate the expected count
        predicted_prob_matrix = [get_grade_prob(theta, As, Bs) for theta in theta_val]
        expected_y_matrix = [[0.0 for j2 in range(num_y)] for k2 in range(num_theta)]

        # given y, the posterior is 
        for i3 in xrange(num_user):
            ys = data_repo[i3]
            log_likelihood_vec = [ sum([log(predicted_prob_matrix[k3][y]) for y in ys]) for k3 in range(num_theta)]
             
            sum_ll = logsum(log_likelihood_vec)
            theta_posterior = [exp(ll-sum_ll) for ll in log_likelihood_vec]
            theta_dist[i3] = theta_posterior

        # given the posterior, calculate the expected counts
        for k4 in range(num_theta):
            for i4 in xrange(num_user):
                p = theta_dist[i4][k4]
                for y in data_repo[i4]:
                    expected_y_matrix[k4][y] += p

        # generate estimate data
        sim_ys = []
        sim_xs = []
        for k5 in range(num_theta):
            for j3 in range(num_y):
                sim_cnt = int(expected_y_matrix[k5][j3])  # approximation here. Work well in large sample
                for t in range(sim_cnt):
                    sim_ys.append(j3)
                    sim_xs.append([1.0, theta_val[k5]])
        # estimate
        X = np.array(sim_xs)
        Y = np.array(sim_ys)
        model = discrete_model.MNLogit(Y,X)
        mod_res = model.fit(disp=0)

        # update As,Bs
        params = mod_res.params
        As,Bs = parse_params(params)
        
        print(As,Bs)

        # estimate the thetas
        MAP_thetas = [sum([theta_dist[i5][k6]*theta_val[k6] for k6 in range(num_theta)]) for i5 in range(num_user)]
        print(np.corrcoef(np.array(thetas), np.array(MAP_thetas))[0][1])
        # does the parameter becomes more clustered
        print(np.mean([np.var(theta_dist[i6])for i6 in range(num_user)]))



    

    

    



    
