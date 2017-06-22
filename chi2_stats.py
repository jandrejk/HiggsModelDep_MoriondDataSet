import numpy as np

#-----------------------------------------------------------------
def chi2N (N, N_true, invCov) :
    vector = (N - N_true)
    return np.dot(vector.T,np.dot(invCov,vector))
#-----------------------------------------------------------------




#-----------------------------------------------------------------
def chi2 (Delta_sigma, response, N_true, invCov,lumi=1.) :
    #print np.shape(np.dot(response,Delta_sigma))
    #print np.shape(N_true)
    vector = (np.dot(response,Delta_sigma)*lumi - N_true)
    return np.dot(vector.T,np.dot(invCov,vector))
#-----------------------------------------------------------------



#-----------------------------------------------------------------
def chi2_fixOneComp (Delta_sigma, response, N_true, invCov,fixed,lumi=1.) :
    #print fixed
    #print 'before: ', Delta_sigma
    pos = fixed[0]
    value = fixed[1]
    Delta_sigma[pos] = value
    #print 'after: ', Delta_sigma
    #print np.shape(np.dot(response,Delta_sigma))
    #print np.shape(N_true)
    vector = (np.dot(response,Delta_sigma)*lumi - N_true)
    return np.dot(vector.T,np.dot(invCov,vector))
#-----------------------------------------------------------------


#-----------------------------------------------------------------
def chi2Fsolve (N, N_true,N_true_SM, sigma_mu) :
    sigma_N = np.multiply(sigma_mu,N_true_SM)
    D_SigmaN = np.diag(sigma_N)
    
    vector = (N - N_true)
    return np.dot(vector.T,np.dot(invCov,vector))
#-----------------------------------------------------------------



#-----------------------------------------------------------------
def GetInverseCovariance (sigma_mu,N_true_SM,corr) :
    sigma_N = np.multiply(sigma_mu,N_true_SM)
    print N_true_SM
    print sigma_N
    #C = np.tensordot(sigma_N.T,np.dot(corr,sigma_N))
    D_SigmaN = np.diag(sigma_N)
    C = np.dot(D_SigmaN.T,np.dot(corr,D_SigmaN))
    print 'max C: ',np.max(C)
    print 'det C: ',np.linalg.det(C)
    C_inverse = np.linalg.inv(C)
    
    identity = np.dot(C,C_inverse)
    print np.all(identity-np.eye(24)<=1e-15)
    identity = np.dot(C_inverse,C)
    print np.all(identity-np.eye(24)<=1e-15)
    print 'conditional C', np.linalg.cond(C)
    print np.linalg.cond(C) > np.finfo(C.dtype).eps
    return C_inverse 
#-----------------------------------------------------------------