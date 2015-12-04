import numpy as np

# the basic parameter is 

def cdf(X):
    # X has to be a list or numpy array
    eXb = np.column_stack((np.ones(len(X)),np.exp(X)))
    return eXb/eXb.sum(1)[:,None]

def loglikelihood(params, wendog,exog,K):
    params = params.reshape(K,-1,order='F')
    logprob = np.log(cdf(np.dot(exog,params)))
    return np.sum(wendog*logprob)

# TODO: bugs: cannot handle single X wihtout constant
def score(params, wendog,exog,K):
    params = params.reshape(K,-1,order='F')
    first_term = wendog[:,1:] - cdf(np.dot(exog,params))[:,1:]
    g =  np.dot(first_term.T, exog).flatten()
    return g

def hessian(params, wendog, exog, K):
    params = params.reshape(K,-1,order='F')
    pr = cdf(np.dot(exog,params))
    partials = []
    J = wendog.shape[1] - 1  # first defaults to be 1
    K = exog.shape[1]
    for i in range(J):
        for j in range(J):
            partials.append( -np.dot( ( (pr[:,i+1]*(int(i==j)-pr[:,j+1]))[:,None]*exog).T, exog ) )
    H = np.array(partials)
    H = np.transpose(H.reshape(J,J,K,K),(0,2,1,3)).reshape(J*K,J*K)
    return H



if __name__ == '__main__':
    '''
    print(cdf([0]),[0.5,0.5])
    print(cdf(np.log([0.25])), [0.8,0.2])
    '''

    params = np.array([[1,1],[-1,2]]) ## betas, alphas
    wendog = np.array([[1,0,0],[0,1,0],[0,0,1]])
    exog = np.array([[1,-1],[1,0],[1,1]])
    K = 2
    ll = loglikelihood(params, wendog, exog, K)
    g = score(params, wendog, exog, K)
    h = hessian(params,wendog,exog,K)
    print (ll,g,h)








