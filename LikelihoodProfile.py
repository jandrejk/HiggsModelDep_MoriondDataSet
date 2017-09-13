import numpy as np
import matplotlib.pyplot as plt
import itertools as it
import plotting as pl
import chi2_stats as chistat 
import scipy.optimize as opt
import os
from scipy.stats import chi2
from scipy.stats import norm




class LikelihoodProfile:
    def __init__(self,LoadPath,Model,Observable, N_ScaningPoints, whichResponse='pred',FastScan=False, N_true_model_MC_toy=None) :
        """
        LoadPath : path where the differential number of events, N^{ij}_l, are stored. N^{ij}_l are the
        number of events in the (ij) reco-level bin coming from the l particle level bin.
        """
        self.path = LoadPath
        self.mode = Model
        self.obs = Observable
        self.N_profiles = N_ScaningPoints
        
        self.N_pred_SM = np.load(self.path+'N_reco_pred_SM'+'.npy')
        self.N_true_SM = np.load(self.path+'N_reco_true_SM'+'.npy')
        
        
        
        self.whichResponse = whichResponse
        self.FastScan = FastScan
        
        self.N_pred_model = np.load(self.path+'N_reco_pred_'+self.mode+'.npy')
        
        N_true_model = np.load(self.path+'N_reco_true_'+self.mode+'.npy')
        #normalize N_true_model to N_true_SM
        factor = np.sum(self.N_true_SM) / np.sum(N_true_model)
        self.N_true_model = factor * N_true_model
        if (N_true_model_MC_toy != None) :
            N_true_model_MC_toy = np.insert(N_true_model_MC_toy,0,0.)
            zer = np.zeros(np.shape(N_true_model))
            zer[:,0] = N_true_model_MC_toy
            self.N_true_model = zer

        #set negative values in any response to zero - this can happen for
        #example in the ttH sample in recoNjets
        self.N_pred_SM[self.N_pred_SM<0] = 0
        self.N_true_SM[self.N_true_SM<0] = 0
        self.N_true_model[self.N_true_model<0] = 0
    
        
        if self.FastScan :
            self.GenerateInvCovMatrix()
        
        
        self.GetBestFitXsec()
        self.GetScanRange()
        self.LikelihoodScan()
        self.ExtractOneSigmaConfIntervall()
    

    #-----------------------------------------------------------------------------------------
    def GenerateInvCovMatrix (self) :
        """
        This function computes the inverse covariance matrix and is called when the class
        instance is initialized. 
        Both, covariance matrix and its inverse become class attributes.
        params  :
        returns :
        """
        self.Cov, self.invCov = chistat.GetCovarianceAndInverse(sigma_mu=self.GetSigmaMu(),
                            N_true_SM = np.sum(self.N_true_SM,axis=1)[1:],
                             corr=self.GetRhoMatrix()
                            )        
    #=========================================================================================

    #-----------------------------------------------------------------------------------------    
    def GetRhoMatrix (self) :
        """
        This function loads the correlation matrix of the signal strengths among different
        reco-bins (out-of-acceptance is not included). The path to the correlation matrices
        is costumized and goes back 3 times back in directories.
        params  :
        returns : rho - 2d-array being the correlation matrix  
        """
        #get the path to correlation matrix by going 3 times back in directories
        path_list = self.path.split(os.sep)
        pathToCorrMatrices = '/'+os.path.join(*path_list[:-3])+'/'
        if ('recoPt' in self.obs) :
            rho = np.load(pathToCorrMatrices+'RHO_recoPt.npy')
            return rho
        if ('recoNjets' in self.obs) :
            rho = np.load(pathToCorrMatrices+'RHO_recoNjets.npy')
            return rho
        
        else :
            print 'No correlation matrix in ', self.path
    #=========================================================================================


    #-----------------------------------------------------------------------------------------    
    def GetSigmaMu (self) :
        """
        This function loads the Asimov uncertainties on the signal strengths in each
        reco-bin (out-of-acceptance is not included). The values are taken from the
        Asimov reco-fit from Thomas Klijnsma.
        params  :
        returns : sigma_mu - 1d-array being the signal strength uncertainties. 
        """
        if ('recoPt' in self.obs) :
            sigma_mu = np.array([7.39e-01, 6.48e-01, 9.52e-01, 6.03e-01, 5.24e-01,
            6.17e-01, 6.88e-01, 6.04e-01, 8.00e-01, 6.30e-01, 
            5.45e-01, 8.00e-01, 6.64e-01, 6.29e-01, 7.03e-01,
            5.40e-01, 5.74e-01, 6.14e-01, 5.19e-01, 6.15e-01, 
            7.24e-01, 5.92e-01, 9.85e-01, 1.22e+00
                                ])
            #do not forget to order per category
            sigma_mu = pl.OrderPerCategory(array=sigma_mu,n=3)
            return sigma_mu 
        
        if ('recoNjets' in self.obs) :
            sigma_mu = np.array([4.93e-01,3.60e-01,8.59e-01,5.40e-01,4.84e-01,6.08e-01,
                                 6.44e-01,7.35e-01,8.90e-01,1.00e+00,1.16e+00,1.41e+00,
                                 1.11e+00,1.28e+00,1.64e+00
                                ])
            #do not forget to order per category
            sigma_mu = pl.OrderPerCategory(array=sigma_mu,n=3)
            return sigma_mu
        else :
            print 'No uncertainties available for recoNjets'
    #=========================================================================================
        
    
    #-----------------------------------------------------------------------------------------    
    def GetDeltaSigmaTrue (self) :
        """
        This function returns the true cross section inferred from the Number-of-Event-matrix
        N_true_model.
        params  :
        returns : true differential fiducial cross section vector.
        """
        return np.sum(self.N_true_model,axis=0)
    #=========================================================================================
    

    #-----------------------------------------------------------------------------------------    
    def GetResponse (self) :
        """
        This function retruns the response matrix by normalizing the number of events N^{ij}_l
        By initializing the class, one can compute the BDT predicted response (default), the SM
        response matrix or the true response matrix.
        params  :
        retruns : K_pred_model - 2d-array being the response matrix, i.e. the element K^{ij}_l
                  is the probability of an event generated in particle level bin l to be 
                  reconstructed in reco-level bin (ij).

        """
        # if not stated differently use the BDT pred response
        K_pred_model = self.N_pred_model / np.sum(self.N_pred_model,axis=0)
        
        if (self.whichResponse == 'true') :
            K_pred_model = self.N_true_model / np.sum(self.N_true_model,axis=0)
        if (self.whichResponse == 'SM') :
            K_pred_model = self.N_true_SM / np.sum(self.N_true_SM,axis=0)
            
        return K_pred_model
    #=========================================================================================
    

    #-----------------------------------------------------------------------------------------    
    def GetBestFitXsec (self) :
        """
        This function computes the best fit fiducial differential cross section vector by 
        minimizing the likelihood function lambda.
        It makes this best fit a class instance as well as the value of lambda when the best fit 
        value is used to compute it.
        params  :
        returns :
        """
        rho = self.GetRhoMatrix()
        sigma_mu = self.GetSigmaMu()
        
        #compute the response matrix
        K_pred_model = self.GetResponse()
        
        # get the best fit fid x-sections
        self.initial_guess = np.ones(len(np.sum(K_pred_model,axis=0)))
        
        if self.FastScan :
            best_fit = opt.minimize(fun=chistat.chi2,
                       x0=self.initial_guess,
                       args=(K_pred_model[1:,:], 
                             np.sum(self.N_true_model,axis=1)[1:], 
                             self.invCov),
                        )
        else :
            best_fit = opt.minimize(fun=chistat.chi2Fsolve,
                       x0=self.initial_guess,
                       args=(K_pred_model[1:,:], 
                             np.sum(self.N_true_model,axis=1)[1:], #take out not-reco row 
                             np.sum(self.N_true_SM,axis=1)[1:],sigma_mu, rho)
                       )
        
        self.lambdaBF = best_fit.fun
        self.DeltaSigma_BF = best_fit.x
    #=========================================================================================

        
    #-----------------------------------------------------------------------------------------    
    def GetProfileLMin (self,fixed_tuple) :
        """
        This function computes the value of the likelihood function lambda if a certain component 
        of the fid differential cross section vector is set to fixed value, i.e. this function
        is used for the profiling.
        params  : fixed_tuple - tuple [int,float], 1st entry specifies the component of the 
                  cross section vector and the 2nd entry its value.
        returns : result - float being the value of the likelihood function evaluated at the
                  minimum for fixing one component of the fid. diff. x-sec vector to specific
                  value.
        """
        rho = self.GetRhoMatrix()
        sigma_mu = self.GetSigmaMu()
        
        #compute the response matrix
        K_pred_model = self.GetResponse()
                
        if self.FastScan :
            optim = opt.minimize(fun=chistat.chi2_fixOneComp,
                       x0=self.initial_guess,
                       args=(K_pred_model[1:,:], 
                             np.sum(self.N_true_model,axis=1)[1:], 
                             self.invCov,
                            fixed_tuple),
                        )
            
        else :
            optim = opt.minimize(fun=chistat.chi2Solve_fixOneComp,
                           x0=self.initial_guess,
                           args=(K_pred_model[1:,:], 
                                 np.sum(self.N_true_model,axis=1)[1:], #take out not-reco row 
                                 np.sum(self.N_true_SM,axis=1)[1:],sigma_mu, rho,fixed_tuple)
                           )        
        
        result = optim.fun
        return result
    #=========================================================================================
    
   
    
    #-----------------------------------------------------------------------------------------    
    def GetScanRange (self) :
        """
        This function perfroms a binary search to probe how much the likelihood function changes
        in the neighborhood of the best fit value.
        It finds the range where the likelihood function reaches 2.5 above its minimum and sets
        this range as the interesting one to do the profile scanning.
        The variable scan_range is made a class range and contains the interesting profile ranges
        for all components of the fid. diff. x-sec vector.
        params  :
        returns :
        """
        self.scan_range = []
        
        for i,s in enumerate(self.DeltaSigma_BF) :
            counter = 0
            step = 1
            result_previous = 0
            nominal = s
            s += step
            while (counter < 50) :
                result = self.GetProfileLMin([i,s]) - self.lambdaBF
              
                if ( abs(result-2.5) < 0.2 ) :
                    self.scan_range.append(s)
                    break

                if (result > 2.5) :
                    
                    if (result>5) :
                        s -= step
                        step = step / np.sqrt(result)
                    else :
                        s -= step/2.
                        step = step/2.
                    
                else :
                    s += step #* (result-result_previous)
             
                result_previous = result
                counter += 1 
    #=========================================================================================

    
    #-----------------------------------------------------------------------------------------    
    def LikelihoodScan (self) :
        """
        Each component of the fid. diff x-sec vector is profiled in a region of interest set 
        by scan_range.
        The values of the likelihood function of these profiles are passed to the class attribute
        profiles.
        params   :
        returns  :
        """
        self.profiles = []
        
        #print self.scan_range
        for i,sigma_k in enumerate(self.DeltaSigma_BF) :
            sigma_k_range = np.linspace(2*sigma_k-self.scan_range[i],self.scan_range[i],self.N_profiles)
            lam_sigma = []
            for s in sigma_k_range :  
                lam_sigma.append(self.GetProfileLMin([i,s]))
                
            self.profiles.append(lam_sigma)
        self.profiles = np.array(self.profiles)
    #=========================================================================================
       
    
    #-----------------------------------------------------------------------------------------    
    def ExtractOneSigmaConfIntervall (self) :
        """
        This function computes the position on the x-axis of each profile where the likelihood
        function crosses 1. 
        This is done by fitting a 2-sided parabola to the descrete scanning points.
        The positions plus_uncert and minus_uncert are made class attribtues.
        params   :
        returns  :
        """
        c = self.lambdaBF
        x_bestFit = self.DeltaSigma_BF
        self.plus_uncert = []
        self.minus_uncert = []
        for i, x_bf in enumerate(x_bestFit) :
            sigma_k_range = np.linspace(x_bf-self.scan_range[i],x_bf+self.scan_range[i],self.N_profiles)
            sigma_k_range = np.linspace(2*x_bf-self.scan_range[i],self.scan_range[i],self.N_profiles)
            xdata = [(sigma_k_range),x_bf]
            ydata = self.profiles[i] - c
            param_opt, pcov = opt.curve_fit(AsymParabola,xdata,ydata)
            
            self.plus_uncert.append(abs(param_opt[1]))
            self.minus_uncert.append(abs(param_opt[0]))
        self.plus_uncert = np.array(self.plus_uncert)
        self.minus_uncert = np.array(self.minus_uncert)
    #=========================================================================================

    
    #-----------------------------------------------------------------------------------------    
    def GetAverageBias (self) :
        """
        This function computes the average bias between best fit and true differential
        cross section.
        params  :
        returns : bias - float being the average bias.
        """
        sigmaBF = self.DeltaSigma_BF
        sigmaTrue = self.GetDeltaSigmaTrue()
        N_genbins = len(sigmaBF)
        print abs(sigmaBF-sigmaTrue) / (0.5*(sigmaBF+sigmaTrue))
        bias = np.sum(abs(sigmaBF-sigmaTrue) / (0.5*(sigmaBF+sigmaTrue))) / N_genbins
        return bias
    #=========================================================================================

    
    #-----------------------------------------------------------------------------------------    
    def GetMedianBias (self) :
        """
        This function computes the median bias between best fit and true differential
        cross section. This quantity is less prone to outliers.
        params  :
        returns : bias - float being the median bias.
        """
        sigmaBF = self.DeltaSigma_BF
        sigmaTrue = self.GetDeltaSigmaTrue()
        N_genbins = len(sigmaBF)
        print abs(sigmaBF-sigmaTrue) / (0.5*(sigmaBF+sigmaTrue))
        bias = np.median(abs(sigmaBF-sigmaTrue) / (0.5*(sigmaBF+sigmaTrue))) 
        return bias
    #=========================================================================================

    #-----------------------------------------------------------------------------------------    
    def DoToyExperiment (self, N_toys) :
        """
        This function generates toys of N_true_model according to a multivariate Gaussian and 
        computes the likelihood function for all these toy MCs.
        When plotted they have to follow a chi2 distribution with N_bins d.o.f. according to
        the large sample limit (Wilk's theorem).
        params  : N_toys - int being the number of toy experiments.
        returns :
        """
        mean_N = np.sum(self.N_true_model,axis=1)[1:] # sum over rows and get rid of non-reco bin
        self.N_MultiGaussToys = np.random.multivariate_normal(mean_N, self.Cov, N_toys)
        
        lambda_toys = []
        lambda_true = []

        for N_i in self.N_MultiGaussToys :
            #compute the response matrix
            K_pred_model = self.GetResponse()


            best_fit = opt.minimize(fun=chistat.chi2,
                               x0=self.initial_guess,
                               args=(K_pred_model[1:,:], 
                                     N_i, 
                                     self.invCov),
                                )
            
            lam_true = chistat.chi2(Delta_sigma=self.GetDeltaSigmaTrue(),
                                   response=self.GetResponse()[1:,:],
                                   N_true=N_i,
                                   invCov=self.invCov 
                                  )

            lambda_toys.append(best_fit.fun)
            lambda_true.append(lam_true)

        self.lambda_true = np.array(lambda_true)
        self.lambda_toys = np.array(lambda_toys)
    #=========================================================================================

        
    #-----------------------------------------------------------------------------------------    
    def GetPulls (self) :
        """
        This function calculates the pulls for each component of the fid. diff x.sec vector
        for all the MC toys.
        params  :
        returns :
        """
        toy_list = []
        for N_i in self.N_MultiGaussToys :
            toy_list.append(LikelihoodProfile(LoadPath=self.path,
                                              Observable=self.obs,
                                     N_true_model_MC_toy=N_i,   
                         Model=self.mode,N_ScaningPoints=10,FastScan=True))
        self.pull_list = []
        for t in toy_list :
            DeltaSigmaTrue = self.GetDeltaSigmaTrue()
            DeltaSigmaBF = t.DeltaSigma_BF
            UncertDeltaSigma = np.where(DeltaSigmaTrue>DeltaSigmaBF, t.plus_uncert,t.minus_uncert)
            
            pull = (DeltaSigmaBF-DeltaSigmaTrue) / UncertDeltaSigma
            self.pull_list.append(pull)
    #=========================================================================================

            
"""
#-----------------------------------------------------------------------------------------    
END OF CLASS LikelihoodProfile
#-----------------------------------------------------------------------------------------    
"""



#-----------------------------------------------------------------------------------------    
def AsymParabola(dependent_variable,sigma_plus,sigma_minus) : 
    """
    This function encodes the functional form of a 2-sided parabola.
    params  : dependent_variable - tuple the 1st component being the x-value of the 
              parabola and the 2nd components its off-set (position of the minimum).
              sigma_plus - float the uncertainty on the right side wrt the 
              minimum
              sigma_minus - float the uncertainty on the left side wrt the 
              minimum
    returns : 
    """
    x = dependent_variable[0]
    x_BF = dependent_variable[1]
    
    y = np.where(x>x_BF, ((x-x_BF)/sigma_plus)**2 , ((x-x_BF)/sigma_minus)**2 )
    return y 
#=========================================================================================

#-----------------------------------------------------------------------------------------    
def GaussianLogLikelihood (x, param) :
    """
    Functional form of a Gaussian log-likelihood
    """
    mu = param[0]
    sigma = param[1]
    return -0.5*np.log(2*np.pi*sigma**2) - np.sum((x-mu)**2)/(2*sigma**2)
#=========================================================================================


#-----------------------------------------------------------------------------------------    
def ensure_dir(file_path):
    """
    This fucntion makes a sure a directory exists and if not creates it
    params  : file_path - string being the full path to the directory
    """
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
#=========================================================================================


#-----------------------------------------------------------------------------------------    
def Chi2Plots (instance, Nbins) :
    """
    This function plots the distribution of the likelihood function obtained by doing
    the MC toy experiments. The distribution should follow a chi^2 distribution with
    Nbins degrees of freedom.
    params  :
              Nbins - int specifying the number of bins, i.e no. of dof.
    returns :
    """
    binning = np.linspace(0,30,20)
    plt.hist(instance.lambda_true-instance.lambda_toys,normed=True,bins=binning,label='toy MC')
    x = np.linspace(0,30,1000)
    plt.plot(x,chi2.pdf(x,df=8),label=r'$\chi^2$ dof$=8$',lw=3,color='r')

    plt.legend(loc='best')
    
    pathToChi2Toys = '/mnt/t3nfs01/data01/shome/jandrejk/higgs_model_dep/MoriondAnalysis/plots/chi2_DistFromToys/'
    study_name = 'BSM1_compareDifferentResponseScenarios'
    pathSave = pathToChi2Toys+'/'+instance.obs+'/'+instance.mode+'/'
    ensure_dir(file_path=pathSave)
    plt.savefig(pathSave+'VHchi2.png')
    plt.show()
#=========================================================================================




#-----------------------------------------------------------------------------------------    
def Chi2ScanPlots (instances,LegendAddition=[],savepath=None,interpolationPoints=500) :
    """
    This function plots the profiles for the chosen class instances and saves them if
    a path is provided.
    params  :
    retruns :
    """
    
    #ensure directory
    ensure_dir(file_path=savepath)
    
    colors = ['black','red','magenta','cyan','purple']
    markers = ['o','^','s']
    
    
    for i in xrange(len(instances[0].initial_guess)) :
        
        for j,instance in enumerate(instances) :
            yMax = 2.5
            line_2p5 = np.zeros(interpolationPoints)+yMax
            line_1 = np.zeros(instance.N_profiles)+1.


            sigma_fid_true = instance.GetDeltaSigmaTrue()
          
            sigma_k = instance.DeltaSigma_BF[i]
            sigma_k_range = np.linspace(2*sigma_k-instance.scan_range[i],instance.scan_range[i],instance.N_profiles) 

            text2 = r"$ \Delta \sigma^{{\mathrm{{(true)}}}}_{{{:}}}  = {:.{p}f} \, \mathrm{{pb}}$".format(i,sigma_fid_true[i],p=2)

            text = r"$ \Delta \hat{{\sigma}}_{{{:}}}  = {:.{p}f} \, {{}}^{{ + {:.{p}f}}}_{{ - {:.{p}f}}} \, \mathrm{{pb}}$".format(str(i),sigma_k,instance.plus_uncert[i],instance.minus_uncert[i],p=2)   
            plt.plot(sigma_k_range,np.array(instance.profiles[i])-instance.lambdaBF,
                     color=colors[j],marker=markers[j],linestyle='',lw=2.,label='Likelihood scan '+LegendAddition[j]+'\n'+text)
            
            x_parabola= np.linspace(sigma_k-instance.scan_range[i],
                                    sigma_k+instance.scan_range[i],interpolationPoints) 

            y_parabola = AsymParabola([x_parabola,sigma_k],instance.plus_uncert[i],instance.minus_uncert[i])
            plt.plot(x_parabola,y_parabola,
                    color=colors[j])
            
            idx2p5 = np.argwhere(np.diff(np.sign(y_parabola - line_2p5)) != 0).reshape(-1) + 0
         
            
            title = 'Profile Likelihood scan '

            plt.title(title,fontsize=18,y=1.1)
            plt.suptitle('model: '+instance.mode,y=0.96)
            
            if ('recoPt' in instance.obs) :
                xlabel = r'$\Delta \sigma_{{{:}}} \, [\mathrm{{pb}}]$ (gen-$p_\mathrm{{T}}^{{\gamma \gamma}}$ bin: {:})'.format(i,pl.GetPtBinRange(str(i)))
            else : 
                xlabel = r'$\Delta \sigma_{{{:}}} \, [\mathrm{{pb}}]$ (gen-$N_\mathrm{{jets}}$ bin: {:})'.format(i,pl.GetNjetsBinRange(str(i)))
            plt.xlabel(xlabel,fontsize=14)
            plt.ylabel(r'$\lambda(\Delta \hat{\hat{\vec{\sigma}}}) - \lambda (\Delta \hat{\vec{\sigma}})$',fontsize=14)

            plt.ylim(0.,2.5)#yMax)

            plt.xlim(*x_parabola[idx2p5])

            plt.hlines(1.0,sigma_k_range[0],sigma_k_range[-1],color='k')

            plt.vlines(sigma_k + instance.plus_uncert[i],0.,1.)
            plt.vlines(sigma_k - instance.minus_uncert[i],0.,1.)
            
            
            if(j==0) :
                plt.vlines(sigma_fid_true[i],0.,yMax,'blue',lw=2.,label='True diff. x-sec'+'\n'+text2)

            plt.legend(bbox_to_anchor=(1.8, 1.),fontsize=13)
        if (savepath != None) :
            print "image saved"
            plt.savefig(savepath+'fidXSbin'+str(i),bbox_inches='tight')
        plt.show()        
#=========================================================================================   