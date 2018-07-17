import numpy as np
import scipy.linalg as la
from ggplot import *
np.random.seed(1)

def monte_carlo(mu, Q, startValue, sims, totalTime, periodTime):
    
    # mu is the drift multiplier (eg nx1 expected return vector for financial models)
    # Q is the shock multiplier (eg nxn covariance matrix for financial models)
    # startValue is the current values (nx1 vector)
    # sims is number of simulations (1x1)
    # totalTime is length of simulation, periodTime is time per period (eg for 26 weeks simulated each week, sims = 26 and periodTime = 1)
    lastValue = startValue
    rho = np.corrcoef(Q)
    L = la.cholesky(rho, lower=True) #due to possible correlation between variables
    num_periods = totalTime//periodTime
    for i in range(sims):
        for j in range(num_periods):
            print(j)
            print(mu)
            xi = np.random.normal(0,1,size = mu.shape)
            nextValue = np.multiply(lastValue, np.exp(((mu - 0.5*(np.diag(Q)))*periodTime) + np.multiply(np.sqrt(np.diag(Q)),xi*np.sqrt(periodTime))))
            lastValue = nextValue
            
monte_carlo(np.array([0.1,0.1,0.1]),np.array([[2s,3,4],[1,2,5],[4,3,1]]),np.array([1,1,1]),1,20,1)
    
    