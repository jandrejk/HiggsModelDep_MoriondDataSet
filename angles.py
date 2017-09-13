import numpy as np
import itertools as it


#--------------------------------------------------------------------------------
def AppendGenJetVariableNames (name_list,Njets_recorded = 6, 
                               keywords = ['Pt','Rapidity','Eta','Phi']) :
    """
    This function appends jet variables to list of names.
    params:
             name_list : list - list of names
        Njets_recorded : int - number of jet information available (default : 6)
              keywords : list - list of variables names 
                         (default : ['Pt','Rapidity','Eta','Phi'])
    returns :
             name_list : list - list of names where all the jet information 
                         is added.
    """
    for i in xrange(Njets_recorded) :
        for key in keywords :
            word = 'genJet2p5'+key+str(i)
            name_list.append(word)
    return name_list
#================================================================================

#--------------------------------------------------------------------------------
def AppendPhotonVariableNames (name_list) :
    """
    This function appends the leading/subleading eta and phi to a list of names
     params:
             name_list : list - list of names
    returns :
             name_list : list - list of names where information about photon
                         eta and phi are added.
    """
    name_list.append('genLeadEta')
    name_list.append('genLeadPhi')
    name_list.append('genSubleadEta')
    name_list.append('genSubleadPhi')
    return name_list
#================================================================================



#--------------------------------------------------------------------------------
def absCosDeltaPhi (phi1,phi2) :
    """
    This fucntion returns the absolute cosine of the angular difference in 
    the transverse plane
    """
    if (phi1==-999 or phi2==-999) :
        return -999
    else :
        return abs(np.cos(phi1-phi2))
#================================================================================
    
    
#--------------------------------------------------------------------------------
def absCosDeltaAlpha (px,py,pz,qx,qy,qz) :
    """
    This function returns the absolute cosine of the angular difference between
    two object in 3-d according to their momenta.
    """
    if (px==-999 or qx==-999) :
        return -999
    else :
        scalar_product = px*qx+py*qy+pz*qz
        p = np.sqrt(px**2+py**2+pz**2)
        q = np.sqrt(qx**2+qy**2+qz**2)
        return abs(scalar_product/(p*q))
#================================================================================

#--------------------------------------------------------------------------------
def cot (angle) :   
    """
    This function returns the cotangent
    """
    return 1./np.tan(angle)
#================================================================================
    
    
#--------------------------------------------------------------------------------    
def absCosDeltaAlphaPhotonJet (phi_Gamma,eta_Gamma, phi_Jet, eta_Jet) :
    """
    This function returns the absolute cosine of the angular difference between
    two object in 3-d according to their phi and eta (pseudo-rapidity).
    Despite the name it can be also used for a generic combination of objects,
    not only photon - jet.
    """
    if (phi_Jet == -999) :
        return -999
    else :
        """go from eta to theta!!!"""
        theta_Gamma = 2.* np.arctan(np.exp((-1)*eta_Gamma))
        theta_Jet = 2.* np.arctan(np.exp((-1)*eta_Jet))

        scalar_product = np.cos(phi_Gamma)*np.cos(phi_Jet) + np.sin(phi_Gamma)*np.sin(phi_Jet) + (cot(theta_Gamma)*cot(theta_Jet))
        denominator_inv = np.sin(theta_Gamma)*np.sin(theta_Jet) # np.sqrt((1.+(cot(theta_Gamma)**2))*(1.+(cot(theta_Jet)**2)))

        return abs(scalar_product*denominator_inv)

    
#--------------------------------------------------------------------------------    
def DeltaRinEtaPhiPlane (phi_1,eta_1, phi_2, eta_2) :
    if (phi_2 == -999 or phi_1 == -999) :
        return -999
    else :
        # transform phi properly to [0, 2pi]
        DeltaR2 = (np.arcsin(np.sin(phi_1-phi_2)))**2 + (eta_1 - eta_2)**2
        return np.sqrt(DeltaR2)

#--------------------------------------------------------------------------------    
def AntiKTDist (pT_1,pT_2,DeltaR_12) : 
    if (DeltaR_12 == -999) :
        return -999
    else :
        #cone parameter of the anti-kt algorithm
        R = 0.4
        pT_pick = np.maximum(pT_1,pT_2)
        return (DeltaR_12 / (pT_pick*R))**2

#--------------------------------------------------------------------------------    
def AddAntiKTDistance(df) :
    MaxJetRecorded = 6
    # adding anti_KT distance between up to six jets
    for i,j in it.combinations(np.arange(MaxJetRecorded),2) :
        # adding Delta R between leading photon and jets
        DistVar = 'DistAntiKT'+str(i)+str(j)
        col1_pt = 'genJet2p5Pt'+str(i)
        col2_pt = 'genJet2p5Pt'+str(j)
        Delta_R = 'DeltaR'+str(i)+str(j)
        
        df[DistVar] = df[[col1_pt,col2_pt,Delta_R]].apply(lambda x : AntiKTDist(*x),axis=1) 
    
    return df
    
#--------------------------------------------------------------------------------    
def AddDistanceToDataFrame (df) :
    MaxJetRecorded = 6
    for i in xrange(MaxJetRecorded) :
        # adding Delta R between leading photon and jets
        DistVar = 'DeltaRLeadGamma'+str(i)
        col1_phi = 'genLeadPhi' 
        col1_eta = 'genLeadEta'
        col2_phi = 'genJet2p5Phi'+str(i) 
        col2_eta = 'genJet2p5Eta'+str(i) 

        df[DistVar] = df[[col1_phi,col1_eta,col2_phi,col2_eta]].apply(lambda x : DeltaRinEtaPhiPlane(*x),axis=1)  
        #----------------------------------------------------------------------
        # adding Delta R between subleading photon and jets
        DistVar = 'DeltaRSubleadGamma'+str(i)
        col1_phi = 'genSubleadPhi' 
        col1_eta = 'genSubleadEta'
        col2_phi = 'genJet2p5Phi'+str(i) 
        col2_eta = 'genJet2p5Eta'+str(i) 

        df[DistVar] = df[[col1_phi,col1_eta,col2_phi,col2_eta]].apply(lambda x : DeltaRinEtaPhiPlane(*x),axis=1)  
        #----------------------------------------------------------------------

    # adding Delta R between leading photon and subleading photon
    DistVar = 'DeltaRLeadGammaSubleadGamma'
    col1_phi = 'genLeadPhi' 
    col1_eta = 'genLeadEta'
    col2_phi = 'genSubleadPhi'
    col2_eta = 'genSubleadEta'

    df[DistVar] = df[[col1_phi,col1_eta,col2_phi,col2_eta]].apply(lambda x : DeltaRinEtaPhiPlane(*x),axis=1)  
    #----------------------------------------------------------------------

    # adding Delta R between up to six jets
    for i,j in it.combinations(np.arange(MaxJetRecorded),2) :
        # adding Delta R between leading photon and jets
        DistVar = 'DeltaR'+str(i)+str(j)
        
        col1_phi = 'genJet2p5Phi'+str(i) 
        col1_eta = 'genJet2p5Eta'+str(i)
        col2_phi = 'genJet2p5Phi'+str(j) 
        col2_eta = 'genJet2p5Eta'+str(j)
        
        df[DistVar] = df[[col1_phi,col1_eta,col2_phi,col2_eta]].apply(lambda x : DeltaRinEtaPhiPlane(*x),axis=1) 
        
    return df




#--------------------------------------------------------------------------------    
def AddAngularVariablesToDataFrame (df, JetAngles = True, JetGammaAngles=True, GammaGammaAngles=True) :
    MaxJetRecorded = 6
    
    if GammaGammaAngles :
        # adding abs Delta phi between leading photon and subleading photon
        phiVar = 'absCosDeltaPhiLeadGammaSubleadGamma'
        col1 = 'genLeadPhi'
        col2 = 'genSubleadPhi'        
        
        df[phiVar] = df[[col1,col2]].apply(lambda x : absCosDeltaPhi(*x),axis=1)
        #----------------------------------------------------------------------
        
        # adding abs Delta alpha between leading photon and subleading photon
        alphaVar = 'absCosDeltaAlphaLeadGammaSubleadGamma'

        col1_phi = 'genLeadPhi' 
        col1_eta = 'genLeadEta'

        col2_phi = 'genSubleadPhi' 
        col2_eta = 'genSubleadEta'

        df[alphaVar] = df[[col1_phi,col1_eta,col2_phi,col2_eta]].apply(lambda x : absCosDeltaAlphaPhotonJet(*x),axis=1)  
        #----------------------------------------------------------------------

    
    if JetAngles :
        for i,j in it.combinations(np.arange(MaxJetRecorded),2) :

            # adding abs Delta phi between jets 
            phiVar = 'absCosDeltaPhi'+str(i)+str(j)
            col1 = 'genJet2p5Phi'+str(i)
            col2 = 'genJet2p5Phi'+str(j)        
            #print phiVar
            df[phiVar] = df[[col1,col2]].apply(lambda x : absCosDeltaPhi(*x),axis=1)
            #----------------------------------------------------------------------

            """
            # adding abs Delta alpha between jets
            alphaVar = 'absCosDeltaAlpha'+str(i)+str(j)

            col1_x = 'genJet2p5Px'+str(i) 
            col1_y = 'genJet2p5Py'+str(i) 
            col1_z = 'genJet2p5Pz'+str(i) 

            col2_x = 'genJet2p5Px'+str(j) 
            col2_y = 'genJet2p5Py'+str(j) 
            col2_z = 'genJet2p5Pz'+str(j)

            df[alphaVar] = df[[col1_x,col1_y,col1_z,col2_x,col2_y,col2_z]].apply(lambda x : absCosDeltaAlpha(*x),axis=1)  
            #----------------------------------------------------------------------
            """
            # checked and this way is correct as well
            # adding abs Delta alpha between jets in another way
            alphaVar = 'absCosDeltaAlpha'+str(i)+str(j)

            col1_phi = 'genJet2p5Phi'+str(i) 
            col1_eta = 'genJet2p5Eta'+str(i) 
            
            col2_phi = 'genJet2p5Phi'+str(j) 
            col2_eta = 'genJet2p5Eta'+str(j) 
            
            df[alphaVar] = df[[col1_phi,col1_eta,col2_phi,col2_eta]].apply(lambda x : absCosDeltaAlphaPhotonJet(*x),axis=1) 
            
    if JetGammaAngles :      
        for i in xrange(MaxJetRecorded) :
            # adding abs Delta phi between leading photon and jets
            phiVar = 'absCosDeltaPhiLeadGamma'+str(i)
            col1 = 'genLeadPhi'
            col2 = 'genJet2p5Phi'+str(i)        
            #print phiVar
            df[phiVar] = df[[col1,col2]].apply(lambda x : absCosDeltaPhi(*x),axis=1)
            #----------------------------------------------------------------------
            # adding abs Delta phi between subleading photon and jets
            phiVar = 'absCosDeltaPhiSubleadGamma'+str(i)
            col1 = 'genSubleadPhi'
            col2 = 'genJet2p5Phi'+str(i)        
            #print phiVar
            df[phiVar] = df[[col1,col2]].apply(lambda x : absCosDeltaPhi(*x),axis=1)
            #----------------------------------------------------------------------

            # adding abs Delta alpha between leading photon and jets
            alphaVar = 'absCosDeltaAlphaLeadGamma'+str(i)

            col1_phi = 'genLeadPhi' 
            col1_eta = 'genLeadEta'

            col2_phi = 'genJet2p5Phi'+str(i) 
            col2_eta = 'genJet2p5Eta'+str(i) 

            df[alphaVar] = df[[col1_phi,col1_eta,col2_phi,col2_eta]].apply(lambda x : absCosDeltaAlphaPhotonJet(*x),axis=1)  
            #----------------------------------------------------------------------
            # adding abs Delta alpha between leading photon and jets
            alphaVar = 'absCosDeltaAlphaSubleadGamma'+str(i)

            col1_phi = 'genSubleadPhi' 
            col1_eta = 'genSubleadEta'

            col2_phi = 'genJet2p5Phi'+str(i) 
            col2_eta = 'genJet2p5Eta'+str(i) 

            df[alphaVar] = df[[col1_phi,col1_eta,col2_phi,col2_eta]].apply(lambda x : absCosDeltaAlphaPhotonJet(*x),axis=1)  
            #----------------------------------------------------------------------    
    return df