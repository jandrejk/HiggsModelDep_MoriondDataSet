import numpy as np
from scipy.linalg import solve

#-------------------------------------------------------------------------------
def chi2N (N, N_true, invCov) :
    """
    This function implements the the chi2 = (N-N_true)^T C^{-1} (N-N_true)
    params:
            N : 1d-array - being the number of events in each reco bin 
                (free parameter)
       N_true : 1d-array - being the number of events in each reco bin
                given by MC
       invCov : 2d-array - being the inverse of the covariance matrix
    returns:
         chi2 : float - being the chi2 
    """
    vector = (N - N_true)
    return np.dot(vector.T,np.dot(invCov,vector))
#===============================================================================




#-------------------------------------------------------------------------------
def chi2 (Delta_sigma, response, N_true, invCov,lumi=1.) :
    """
    This function returns the chi2 parametrized through the
    fiducial cross section Delta_sigma.
    params:
        Delta_sigma : 1d-array - being the fiducial cross section
           response : 2d-array - being the detector response matrix
             N_true : 1d-array - being the number of events in each reco bin
                      given by MC
             invCov : 2d-array - being the inverse of the covariance matrix
               lumi : float - being the total integrated luminosity of the 
                      sample (default: 1.).
    returns:
               chi2 : float - being the chi2
    """
    vector = (np.dot(response,Delta_sigma)*lumi - N_true)
    return np.dot(vector.T,np.dot(invCov,vector))
#===============================================================================



#-------------------------------------------------------------------------------
def chi2_fixOneComp (Delta_sigma, response, N_true, invCov,fixed,lumi=1.) :
    """
    This function computes the chi2 according to the fucntion chi2() but
    fixes one component of the fiducial cross section.
    params:
        Delta_sigma : 1d-array - being the fiducial cross section
           response : 2d-array - being the detector response matrix
             N_true : 1d-array - being the number of events in each reco bin
                      given by MC
             invCov : 2d-array - being the inverse of the covariance matrix
              fixed : list - first entry being the index of the component of
                      the cross section array which should be fixed. The
                      second entry specifies its value.
               lumi : float - being the total integrated luminosity of the 
                      sample (default: 1.).
    returns:
               chi2 : float - being the chi2
    """
    pos = fixed[0]
    value = fixed[1]
    Delta_sigma[pos] = value
    #return chi2(Delta_sigma=Delta_sigma, response=response, 
    #            N_true=N_true, invCov=invCov,lumi=lumi)
    vector = (np.dot(response,Delta_sigma)*lumi - N_true)
    return np.dot(vector.T,np.dot(invCov,vector))
#===============================================================================


#-------------------------------------------------------------------------------
def chi2Solve_fixOneComp (Delta_sigma, response, N_true, N_true_SM, 
                          sigma_mu, corr, fixed, lumi=1.) :
    """
    This function computes the chi2 according to the fucntion chi2() but
    fixes one component of the fiducial cross section.
    params:
        Delta_sigma : 1d-array - being the fiducial cross section
           response : 2d-array - being the detector response matrix
             N_true : 1d-array - being the number of events in each reco bin
                      given by MC
             invCov : 2d-array - being the inverse of the covariance matrix
              fixed : list - first entry being the index of the component of
                      the cross section array which should be fixed. The
                      second entry specifies its value.
               lumi : float - being the total integrated luminosity of the 
                      sample (default: 1.).
    returns:
               chi2 : float - being the chi2
    """
    pos = fixed[0]
    value = fixed[1]
    Delta_sigma[pos] = value
    #return chi2(Delta_sigma=Delta_sigma, response=response, 
    #            N_true=N_true, invCov=invCov,lumi=lumi)
    result = chi2Fsolve(Delta_sigma, response, N_true, N_true_SM, sigma_mu, corr)
    return result
#===============================================================================




#-------------------------------------------------------------------------------
def chi2Fsolve (Delta_sigma, response, N_true, N_true_SM, sigma_mu, corr, 
                lumi=1.) :
    """
    This function calculates the chi2. Instead of inverting the covariance matrix
    it uses scipy.linalg.solve to find the result of a system of linear equations
    params:
        Delta_sigma : 1d-array - being the fiducial cross section
           response : 2d-array - being the detector response matrix
             N_true : 1d-array - being the number of events in each reco bin
                      given by MC
          N_true_SM : 1d-array - being the number of events in each reco bin
                      given by the **SM** MC
           sigma_mu : 1d-array - being the uncertainties on the signal strengths
                      in each reco bin.
               corr : 2d array - being the correlation matrix
               lumi : float - being the total integrated luminosity of the 
                      sample (default: 1.).
    returns:
               chi2 : float - being the chi2
    """
    sigma_N = np.multiply(sigma_mu,N_true_SM)
    D_SigmaN = np.diag(sigma_N)
    C = np.dot(D_SigmaN.T,np.dot(corr,D_SigmaN))
    vector = (np.dot(response,Delta_sigma)*lumi - N_true)    
    y = solve(C,vector)
    return np.dot(vector.T,y)
#===============================================================================



#-------------------------------------------------------------------------------
def GetCovarianceAndInverse (sigma_mu,N_true_SM,corr) :
    """
    This function computes the inverse of the covariance
    matrix.
    :params
        sigma_mu : 1d-array - being the uncertainties on the signal strengths
                      in each reco bin.
       N_true_SM : 1d-array - being the number of events in each reco bin
                      given by the **SM** MC
            corr : 2d array - being the correlation matrix
    returns:
               C : 2d-array - being the covariance matrix
       C_inverse : 2d-array - being the inverse covariance matrix
    """
    sigma_N = np.multiply(sigma_mu,N_true_SM)
    D_SigmaN = np.diag(sigma_N)
    C = np.dot(D_SigmaN.T,np.dot(corr,D_SigmaN))
    
    C_inverse = np.linalg.inv(C)
        
    return C, C_inverse 
#===============================================================================
