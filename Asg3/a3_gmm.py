from sklearn.model_selection import train_test_split
import numpy as np
from scipy.special import logsumexp
import os, fnmatch
import random

# dataDir = '/u/cs401/A3/data/'
dataDir = 'data/'
class theta:
    def __init__(self, name, M=8,d=13):
        self.name = name
        self.omega = np.zeros((M,1))
        self.mu = np.zeros((M,d))
        self.Sigma = np.zeros((M,d))


def log_b_m_x( m, x, myTheta, preComputedForM=[]):
    ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
        See equation 1 of the handout

        As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
        If you do this, you pass that precomputed component in preComputedForM

    '''
    # print ( 'TODO' )
    ans = 0.0
    d = myTheta.mu.shape[1]
    mu = myTheta.mu[m].squeeze()
    sigma = myTheta.Sigma[m].squeeze()
    axis = 1
    if (len(x.shape) == 1):
        axis = 0
    else:
        axis = 1

    if len(preComputedForM) == 0:
        ans = np.sum(np.power(x - mu, 2)/sigma, axis = axis) + d * np.log(2.0 * np.pi) + np.sum(np.log(sigma))
    else:
        ans = np.sum(x * (x - 2.0 * mu)/sigma, axis = axis) + preComputedForM[m]

    return -0.5 * ans

def preCompute(myTheta, M):
    preComputedForM = []

    d = myTheta.Sigma.shape[1]

    vector_term = np.sum(np.power(myTheta.mu, 2)/myTheta.Sigma + np.log(myTheta.Sigma), axis = 1)
    constant_term = d * np.log(2.0 * np.pi)

    return constant_term + vector_term

def log_p_m_x( m, x, myTheta):
    ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
        See equation 2 of handout
    '''
    # print ( 'TODO' )
    M = myTheta.omega.shape[0]
    d = len(x)

    preComputedForM = preCompute(myTheta, M)

    log_b_m_x_values = np.array([log_b_m_x(i, x, myTheta, preComputedForM) for i in range(M)])

    log_D_r = logsumexp(log_b_m_x_values, b=myTheta.omega.squeeze())

    return np.log(myTheta.omega.squeeze()[m]) + log_b_m_x_values[m] - log_D_r

    
def logLik( log_Bs, myTheta ):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency. 

        See equation 3 of the handout
    '''
    # print( 'TODO' )
    ans = 0.0
    M = log_Bs.shape[0]
    T = log_Bs.shape[1]

    weights = np.tile(myTheta.omega.reshape(log_Bs.shape[0], 1),(1, log_Bs.shape[1]))
    ans = np.sum(logsumexp(a=log_Bs, axis=0, b=weights))
    
    return ans

def init_theta(theta, X, M):
    theta.mu = X[np.random.choice(X.shape[0], M, replace = False)]
    theta.Sigma[...] = 1
    theta.omega[...] = 1.0/M
    return theta

def proxy_log_b_m_x(x, m, myTheta, preComputedForM):
    return log_b_m_x(m, x, myTheta, preComputedForM)

def proxy_log_p_m_x(x, m, myTheta, preComputedForM):
    return log_p_m_x(m, x, myTheta, preComputedForM)

def compute_intermediate_results(X, M, myTheta, preComputedForM):
    T = X.shape[0]

    log_Bs = np.zeros((M, T))
    log_Ps = np.zeros((M, T))

    for m in range(M):
        log_Bs[m, :] = log_b_m_x(m, X, myTheta, preComputedForM)
    
    log_Ps_N_r = np.add(np.log(myTheta.omega),  log_Bs)
    log_Ps = np.add(log_Ps_N_r, -logsumexp(log_Ps_N_r, axis = 0))

    return log_Bs, log_Ps

def proxy_logsumexp(x, log_ps):
    ans = logsumexp(log_ps, b = x, return_sign = True)
    return ans[0] * ans[1]

def update_params(myTheta, X, log_Ps, L):
    Ps = np.exp(log_Ps)
    M = log_Ps.shape[0]
    T = log_Ps.shape[1]
    myTheta.omega = np.sum(Ps, axis = 1)/T
    myTheta.omega = myTheta.omega.reshape((M, 1))
    
    for m in range(M):
        D_r = logsumexp(log_Ps[m, :])
        mu_N_r = np.apply_along_axis(proxy_logsumexp, 0, X, log_Ps[m, :])
        myTheta.mu[m, :] = np.exp(mu_N_r - D_r)
        sigma_N_r = np.apply_along_axis(proxy_logsumexp, 0, np.power(X, 2), log_Ps[m, :])
        myTheta.Sigma[m, :] = np.exp(sigma_N_r - D_r) - np.power(myTheta.mu[m, :], 2)

    return myTheta  

def train( speaker, X, M=8, epsilon=0.0, maxIter=20 ):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)
        Shape of X: (num_samples, d = 13)
    '''
    myTheta = theta( speaker, M, X.shape[1] )
    myTheta = init_theta(myTheta, X, M)
    # print ('TODO')

    i = 0
    prev_L = -np.inf
    improvement = np.inf
    while (i <= maxIter and improvement >= epsilon):
        print(prev_L, i)
        preComputedForM = preCompute(myTheta, M)
        log_Bs, log_Ps = compute_intermediate_results(X, M, myTheta, preComputedForM)
        L = logLik(log_Bs, myTheta)
        myTheta = update_params(myTheta, X, log_Ps, L)
        improvement = L - prev_L
        prev_L = L
        i += 1

    return myTheta


def test( mfcc, correctID, models, k=5 ):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK] 

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    # print ('TODO')
    model_to_loglik = []
    print(models[correctID].name)
    for model_index in range(len(models)):
        model = models[model_index]

        M = models[model_index].omega.shape[0]
        log_Bs = np.zeros((M, mfcc.shape[0]))
        preComputedForM = preCompute(model, M)
        for m in range(M):
            log_Bs[m, :] = log_b_m_x(m, mfcc, model, preComputedForM)
        model_to_loglik.append([model_index, logLik(log_Bs, model)])

    sorted_model_to_loglik = sorted(model_to_loglik, key = lambda x: x[1], reverse = True)

    for model_index, value in sorted_model_to_loglik[0:5]:
        print(models[model_index].name, value)

    bestModel = sorted_model_to_loglik[0][0]

    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20
    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print( speaker )

            files = fnmatch.filter(os.listdir( os.path.join( dataDir, speaker ) ), '*npy')
            random.shuffle( files )
            
            testMFCC = np.load( os.path.join( dataDir, speaker, files.pop() ) )
            testMFCCs.append( testMFCC )

            X = np.empty((0,d))
            for file in files:
                myMFCC = np.load( os.path.join( dataDir, speaker, file ) )
                X = np.append( X, myMFCC, axis=0)

            trainThetas.append( train(speaker, X, M, epsilon, maxIter) )

    # evaluate 
    numCorrect = 0;
    for i in range(0,len(testMFCCs)):
        numCorrect += test( testMFCCs[i], i, trainThetas, k ) 
    accuracy = 1.0*numCorrect/len(testMFCCs)