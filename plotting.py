"""
import matplotlib
matplotlib.use('ps')
from matplotlib import rc

rc('text',usetex=True)
rc('text.latex', preamble='\usepackage{color}')
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import LikelihoodProfile as lp

from util import target_name
from scipy.stats import skellam, ks_2samp

import os 



# ---------------------------------------------------------------------------
def rat_plus_remind(num,den):
    """
    This function returns the value of the ratio num : den rounded up
    to the next integer.
    :params 
            num : int - numerator
            den : int - denominator
    :retruns
            ret : int - num/den rounded up
    """
    ret = num / den
    if num % den > 0: ret+=1
    return ret

# ---------------------------------------------------------------------------
def scatter_hist(df,columns,cmap=plt.cm.Blues,figsize=(14,8),colorbar=False,
                 log=False, **kwargs):
    """
    This function produces a set of scatter plots.
    : params 
             df : pandas dataframe - includes the tabulated properties
                  of the classifier
        columns : list, string - specifies which columns in the dataframe
                  should be plotted against each other.  
           cmap : 
        figsize : tuple, int - specifies the figure size. (default: 
                  figsize=(14,8))
       colorbar : boolean - specifies whether to display the colorbar
                  (True) or not (False). (default=False)
            log : boolean - specifies whether to display the scatter
                  plot in logarithmic scale (True) or not (False). 
                  (default=False)
    : retruns 
                :
    """
    
    #extract the number of columns and initialize a figure with
    #sqaure size ncols x ncols
    ncols = len(columns)
    fig, axarr = plt.subplots(ncols,ncols,figsize=figsize)
    
    
    for ix,xcol in enumerate(columns):        
        xargs = {}
        if type(xcol) == tuple: 
            xcol, xargs = xcol
        xlabel = "prob "+"cat "+xargs.get('xlabel',xcol.split("_")[-1])
        for iy,ycol in enumerate(columns):
            histargs = { "bins" : 20, "edgecolor" : 'black', "color" : "red" } #, "normed" : True, "log" : log }
            histargs.update(xargs)
            yargs = {}
            if type(ycol) == tuple: 
                ycol,yargs = ycol
            ylabel = "prob "+"cat "+yargs.get('ylabel',ycol.split("_")[-1])
            if iy == ix:
                axarr[ix,iy].hist(df[xcol],weights=df['weight'], **histargs)
            else:
                axarr[iy,ix].hexbin(x=df[xcol],y=df[ycol],C=df['weight'],cmap=cmap)
                if colorbar: plt.colorbar(ax=axarr[iy,ix])
                    
            if ix == 0:
                axarr[iy,ix].set_ylabel(ylabel)
            else:
                plt.setp(axarr[iy,ix].get_yticklabels())#, visible=False)                
            if iy == ncols-1:
                axarr[iy,ix].set_xlabel(xlabel)
            else:
                plt.setp(axarr[iy,ix].get_xticklabels(), visible=False)
                
    plt.show()
    
    
    
    figsize = map(lambda x : x/3, figsize)
    fig2, ax = plt.subplots(ncols/2,ncols/2, figsize=figsize)
    
    for i,col in enumerate(columns):        
        args = {}
        if type(col) == tuple: 
            col, args = col
        xlabel = "prob "+"cat "+args.get('xlabel',col.split("_")[-1])
        ylabel = "weighted count" 
        histarg = { "bins" : 20, "edgecolor" : 'black', "color" : "red" }
        histarg.update(args)
        
        #indices in the subplots
        k = i/2
        l = i%2
        
        ax[k,l].hist(df[col],weights=df['weight'], **histarg)
        ax[k,l].set_xlabel(xlabel)
        if l == 0:
            ax[k,l].set_ylabel(ylabel)      
        
    fig2.tight_layout()        
    plt.show()
    
# ---------------------------------------------------------------------------
def efficiency_map(x,y,z,cmap=plt.cm.viridis,layout=None,
                   xlabel=None,ylabel=None,**kwargs):
    """
    This function produces efficiency plots for all the categories.
    : params 
            x : numpy.ndarry - specifies the x bins of the of the
                2d histogram.
            y : numpy.ndarry - specifies the corresponding y bins of the of 
                the 2d histogram.
            z : numpy.ndarray - specifiec the efficiency/probability of each
                category to be ploted.
         cmap : colormap style
       layout : tuple - specifies the number of rows and columns of the plot
                (default: layout=None)
       xlabel : string - specifies the label of the x-axis (default: 
                xlabel=None)
       ylabel : string - specifies the label of the y-axis (default: 
                ylabel=None)
            
    """
    #initilize a figure. kwargs are e.g. figsize
    fig = plt.figure(**kwargs)
    #extract the number of categories
    nplots = z[0].size
    
    #This block of code is used in order to find out the number of rows
    #and columns in case no specific layout is given.
    #-----------------------------------------------
    if not layout:
        for ncols in xrange(1,nplots):
            nrows = rat_plus_remind(nplots,ncols)
            if abs(nrows-ncols) <= 1: break
    else:
        ncols,nrows = layout
        if not nrows:
            nrows = rat_plus_remind(nplots,ncols)
        if not ncols:
            ncols = rat_plus_remind(nplots,nrows)
    ## layout=(nrows,ncols)
    #-----------------------------------------------
    
    for icat in xrange(1,nplots):
        #row and column index
        irow = (icat-1) / ncols
        icol = (icat-1) % ncols

        ## ax = axarr[irow,icol]
        ax = plt.subplot(nrows, ncols, icat)
        plt.hexbin(x=x,y=y,C=z[:,icat],cmap=cmap,vmin=0,vmax=1)
        
        #editing the plots when to show which label
        if icol == 0: 
            if ylabel: plt.ylabel(ylabel)
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        if ((irow+1)*ncols + icol >= nplots):
            if xlabel: plt.xlabel(xlabel)
        else:
            plt.setp(ax.get_xticklabels(), visible=False)
            
        plt.title("efficiency cat %d" % icat)
        ## if icol == ncols - 1: plt.colorbar()
        
    plt.subplot(nrows, ncols, nplots)
    
    #1. - not_reco = efficiency
    plt.hexbin(x=x,y=y,C=1.-z[:,0],cmap=cmap,vmin=0,vmax=1)
    if (nplots % (ncols) == 1): plt.ylabel(ylabel)
    else: 
        plt.setp(ax.get_yticklabels(), visible=False)
    if xlabel: plt.xlabel(xlabel)
    plt.title("total efficiency")
    
    fig.subplots_adjust(top=0.9)
    cbar_ax = fig.add_axes([0.15, 0.97, 0.7, 0.02])
    plt.colorbar(cax=cbar_ax,orientation='horizontal')
    

    
    
    
# ---------------------------------------------------------------------------
def naive_closure(df,column,first=0,logy=False,title=None,absolute=True,
                 savepath = None):
    """
    This function produces 1d histograms ensuring the in this case the
    BDT gives comparable results to simply counting the events.
    :params 
             df : pandas dataframe - includes the tabulated properties
                  of the classifier
         column : list, string - specifies which columns in the dataframe
                  should be plotted against each other.  
          first : int - specifies from which category on the histogram
                  should be produced. This can be used to omit the first
                  class of not reconstructed events by setting first=1
                  (default: first=0)
           logy : boolean - specifies whether the histogram should have a 
                  logarithmic y-axis (True) or not (False).
                  (default: logy=False)
          title : string - specifies the title of the histogram (default: 
                  title = None)
       absolute : boolean - specifies whether the histogram is filled witout
                  weights (True) are with weights (True). Note that for 
                  weighted histograms the skellam distribution gives an 
                  estimate on the uncertainties of the true histogram.
                  (default: absolute=True)
    
    """
    
    #naive_closure(df,column=key,logy=True,title='All')
    
    
    target = target_name(column)
    #extract the number of features that belong to column
    nstats = np.unique(df[target]).size
    print("There are " + str(nstats) + " features of type " + str(target))
    
    pred_cols = map(lambda x: ("%s_prob_%d" % (target, x)), range(nstats) ) 
    
    #print(pred_cols)
    
    
    if absolute :
        # fill the histograms without weights. The error on the number
        # of events in a bin are driven by a Poissonian
        
        #the outcome of trueh is an int
        trueh = np.histogram(df[target],np.arange(-1.5,nstats-0.5))[0].ravel() 
        #the predicted hist sums over all events weighted by their probability
        #hence the result is a float
        predh = np.array((df[pred_cols]).sum(axis=0)).ravel()
        
        draw_data_mc(bins = np.arange(-1.5+first,nstats-0.5),
                    corr = predh[first:],
                    data = trueh[first:],
                    ratio=True,
                    var=[column+' category',''],
                    savepath=savepath,
                    title=title)
        savepath = None 
    else :
        # here the weights are taken into account. In order to estimate the 
        # uncertaintiy on the weighted number of events the skellam dist function
        # from scipy is used.
        
        sum_of_weights = df['weight'].sum()
        #the outcome of trueh is an int
        trueh = np.histogram(df[target],np.arange(-1.5,nstats-0.5),weights=df['weight'])[0].ravel() #/ sum_of_weights
        
        """
        print(trueh)
        print(df[df[target]==-1]['weight'].sum())
        print(df[df[target]==0]['weight'].sum())
        print(df[df[target]==1]['weight'].sum())
        print(df[df[target]==2]['weight'].sum())
        """
        square_weight = np.multiply(df['weight'],df['weight'])
        Var_trueh = np.histogram(df[target],np.arange(-1.5,nstats-0.5),weights=square_weight)[0].ravel() 
        
        # no of positive weight events
        mu_1 = df[df['weight']>0].groupby(target).count()['weight'].values
        # no of negative weight events
        mu_2 = df[df['weight']<0].groupby(target).count()['weight'].values
        
        
        # take the mean of the absolute weights as an approximate multiplication
        # factor for the average detector eff and lumi-factor correction.
        avg_absweight = df[['absweight',target]].groupby(target).agg(np.mean).values.ravel()   
        
        
        N_est_evts = np.multiply((mu_1-mu_2),avg_absweight) 

        """
        print(Var_trueh)
        
        print(mu_1)
        print(mu_2)
        print(N_est_evts)
        print(avg_absweight)
        print('sum of weights: ', sum_of_weights)
        print('sum of absweights: ', df['absweight'].sum())
        """
        
        err_pos = +np.multiply(skellam.ppf(1.-0.16, mu_1, mu_2),avg_absweight) - N_est_evts
        err_neg = -np.multiply(skellam.ppf(0.16, mu_1, mu_2),avg_absweight)    + N_est_evts
        err_pos = err_pos / sum_of_weights
        err_neg = err_neg / sum_of_weights
        
        """
        print('sellam pos', skellam.ppf(1.-0.16, mu_1, mu_2))
        print('sellam neg', skellam.ppf(0.16, mu_1, mu_2))
        
        print('errpos', err_pos )
        print('errneg', err_neg )
        """
        predh = []
        for c in pred_cols :
            #print(c)
            predh.append(weighted_average(df,c,'weight'))
        

    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    ## true = ax.bar(np.arange(0,2*(nstats),2)[first:],trueh[first:],color='black')
    ## pred = ax.bar(np.arange(1,2*(nstats)+1,2)[first:],predh[first:],color='red')

    #list of class features, e.g. [0,1,2,3]
    xp = np.arange(nstats)[first:]
    
    #pred = ax.bar(xp-0.5,predh[first:],color='green',width=1.,alpha=0.5)
    pred = ax.bar(xp-.5,predh[first:],color='green',width=1.,alpha=0.5, edgecolor='black')
    
    if absolute :
        true = ax.errorbar(xp,trueh[first:],ls='None',
                       xerr=np.ones_like(xp)*0.5,
                       yerr=np.sqrt(trueh[first:]),
                       ecolor='black')
        plt.ylabel("No. of events")
        print('sqrt error for poissonian', np.sqrt(trueh[first:]))
    else :
        true = ax.errorbar(xp,trueh[first:],ls='None',
                        xerr=np.ones_like(xp)*0.5,
                        yerr=[abs(err_neg)[first:],err_pos[first:]],#np.sqrt(Var_trueh[first:]),#[abs(err_neg)[first:],err_pos[first:]],#np.sqrt(trueh[first:]),
                        ecolor='black')
        plt.ylabel("No. of events (weighted)")
    
    
    plt.xticks(xp,xp)
    plt.xlabel(column)
    
    plt.ylabel("No. of events (weighted)")
    
    if title:
        plt.title(title)
    
    if logy:
        ax.set_yscale('log')
        
        
    #ax.legend((true,pred),("true","predicted"),bbox_to_anchor=(1.45, 1.))    
    plt.legend((true,pred),("true","predicted"),loc='best')
    
    if (savepath != None) :
        try :
            plt.savefig(savepath+'/'+title)
        except IOError :
            os.mkdir(savepath)
            plt.savefig(savepath+'/'+title)
      
    else :
        plt.show()
    

def weighted_average(df_name, column_name, weight_name=None):
    """
        This function computes the weighted average of the quantity column_name
        stared in the pandas dataframe df_name. In case no weights are given
        or if they sum up to zero, the mean is returned instead.
        :params 
                df_name :
            column_name :
            weight_name :
        :retruns
                        :
        """
    #----------------------------------------------------------------------------
    d = df_name[column_name]
    
    if (weight_name == None) :
        return float(d.mean())
    else :
        try:
            w = df_name[weight_name]
            return (d * w).sum() #/ float(w.sum())
        except ZeroDivisionError:
            return float(d.mean())
    #----------------------------------------------------------------------------
    
    
# ---------------------------------------------------------------------------
def control_plots(key,fitter):
    """
    This function produces a series of plots. First it performs a box plot.
    Then it produces a scatter plot and at the end some histograms with
    different selection cuts.
    
    : params   
              key : string - specifies what type of feature should be 
                    extracted, e.g. key='class'
           fitter : train.EfficiencyFitter - trained classifier
    """
    #goes to util to call target_name. If it is key=class then target=class
    target = target_name(key)
    
    #extract the number of classes. fitter.clfs[key].classes_ yields [-1,0,1,2 and datatype]
    nclasses = len(fitter.clfs[key].classes_)
    
    #map creates new list by applying the inside function to xrange(nclasses)
    #creates a list of [class_prob_0,...,class_prob_3]
    columns = map(lambda x: "%s_prob_%d" % (target,x), xrange(nclasses) ) 
    columns = columns[:1]+columns[1:]
    
    #create pandas data frame
    df = fitter.df
    #if data set was splitten in train and test set then take only the test set.
    #note that the test set is indexed from 0 to first_train_evt
    if fitter.split_frac > 0:
        first_train_evt = int(round(df.index.size*(1.-fitter.split_frac)))
        df = df[:first_train_evt]
    
    #needed for box plot
    nrows = nclasses/3+1 #would not work in python 3
    ncols = 3  
    #perform the box plot
    df.boxplot(by=target,column=columns,figsize=(7*ncols,7*nrows),layout=(nrows,ncols))
       
        
    #perform the scatter plots
    scatter_hist(df,columns,figsize=(28,28))
    
    
    
    naive_closure(df,key,logy=True,title='All')
    
    naive_closure(df,key,first=1,logy=False,title='All')
    
    naive_closure(df[df['genPt'] > 50.],key,first=1,logy=False,title='pT > 50')

    naive_closure(df[df['genPt'] < 50.],key,first=1,logy=False,title='pT < 50')

    naive_closure(df[df['absGenRapidity'] > 1.],key,title='|y| > 1.',
                  first=1,logy=False)
    naive_closure(df[df['absGenRapidity'] < 1.],key,
                  title='|y| < 1.',
                  first=1,logy=False)
    naive_closure(df[(df['absGenRapidity'] > 1.) & (df['genPt'] > 50.)],key,
                  title='|y| > 1. & pT > 50',
                  first=1,logy=False)
    naive_closure(df[(df['absGenRapidity'] > 1.) & (df['genPt'] < 50.)],key,
                  title='|y| > 1. & pT < 50',
                  first=1,logy=False)
    naive_closure(df[(df['absGenRapidity'] > 0.5) & (df['absGenRapidity'] < 1.) ],key,
                  title='0.5 < |y| < 1.',
                  first=1,logy=False)
    naive_closure(df[df['absGenRapidity'] < 0.5],key,
                  title='|y| < 0.5',
                  first=1,logy=False)
    naive_closure(df[(df['absGenRapidity'] > 0.25) & (df['absGenRapidity'] < .5) ],key,
                  title='0.25 < |y| < 0.5',
                  first=1,logy=False)
    naive_closure(df[df['absGenRapidity'] < 0.25],key,
                  title='|y| < 0.25',
                  first=1,logy=False)
# ---------------------------------------------------------------------------
    

# ---------------------------------------------------------------------------
def OrderPerCategory (array,n) :
    """
    This functions re-orders an array such that always
    multiples of n are following each other.
    params: 
            array : 1d-array - being the array which will be reordered 
                n : int - spcifying which n entries are followed
    returns:
       re_ordered : 1d-array being the re-ordered array
    examples:
    OrderPerCategory([1,2,3,4,5,6,7,8,],2) retruns [1,3,5,7,2,4,6,8]
    """
    re_ordered = np.array([])
    
    N = len(array)
    if (N%n == 0) :
        for i in xrange(n) :
            re_ordered = np.append(re_ordered,array[i::n]) 
    else :
        first = array[0]
        new_array = array[1:]
        for i in xrange(n) :
            re_ordered = np.append(re_ordered,new_array[i::n]) 
        re_ordered = np.append(first,re_ordered)
    return re_ordered    
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------    
def draw_data_mc(df,column,first=0,figsize=(8,6),var=None,logy=False,ratio=False,
                savepath = None,title=None,plot_title=[],absolute=True,
                model_processes=[],DiffGenVariable=False):
    
    """
    In this function the closure plots are printed and saved if a path is given.
    Further it returns the differential histograms of number of events in the 
    l-th particle level bin ending up in the (ij)th reco-level bin.
    params:
                          df : pandas dataframe
                      column : string - specifying the classifer, e.g. recoPt or recoNjets2p5
                     first=0 : int - being the first argument to plot in the histograms. Use first=1
                               if you want to omit the category not reconstructed
               figsize=(8,6) : tuple
                    var=None :
                  logy=False :
                 ratio=False : boolean - specifying whether to plot also the ratio
             savepath = None : string - specifying the save directory
                  title=None : string - title for the save-path
               plot_title=[] :
               absolute=True : boolean - should the histograms be plotted absolute or 
                               with weights
          model_processes=[] : list - ints specifying which processes to be included in
                               the histogram
       DiffGenVariable=False : boolean - put True if the histogram is differntially in 
                               a gen-level bin. 
    returns:
            N_reco_pred_ij_l : 2d-array - giving the number of predicted events in each (i,j) reco-level
                               bin coming from the l gen-level bin.
            N_reco_true_ij_l : 2d-array - giving the number of true events in each (i,j) reco-level
                               bin coming from the l gen-level bin.
    """
    
    target = target_name(column)
    #extract the number of features that belong to column
    nstats = np.unique(df[target]).size
    
    #In case DiffGenVariable==True this will be later filled with the BDT 
    #preiction of number reco events in (ij) reco coming from l gen bin
    N_reco_pred_ij_l = []
    N_reco_true_ij_l = []
    N_err_reco_true_ij_l = []
    
    if 'recoPt' in target :
        nstats =25
    
    if 'recoNjets2p5' in target :
        nstats =16
    
    
    print("There are " + str(nstats) + " features of type " + str(target))
    
    pred_cols = map(lambda x: ("%s_prob_%d" % (target, x)), range(nstats) ) 
    
     
    if absolute :    
        #the outcome of trueh is an int
        trueh = np.histogram(df[target],np.arange(-1.5,nstats-0.5))[0].ravel() 
        #the predicted hist sums over all events weighted by their probability
        #hence the result is a float
        predh = np.array((df[pred_cols]).sum(axis=0)).ravel()  

        #reorder the histograms per mass category:
        trueh = OrderPerCategory(trueh,3)
        predh = OrderPerCategory(predh,3)
    
    else :
        predh = []
        for c in pred_cols :
            #print(c)
            predh.append(weighted_average(df,c,'weight'))
        
        trueh = np.histogram(df[target],np.arange(-1.5,nstats-0.5),
                             weights=df['weight'])[0].ravel() 
        
        #reorder the histograms per mass category:
        trueh = OrderPerCategory(trueh,3)
        predh = OrderPerCategory(predh,3)

        
        
    bins = np.arange(-1.5+first,nstats-0.5)
    corr = predh[first:]
    data = trueh[first:]    
    binw=bins[1]-bins[0]
    
    if ratio:
        fig, axes = plt.subplots(2,figsize=figsize,sharex=True,gridspec_kw = {'height_ratios':[3, 1]})
        top = axes[0]
        bottom = axes[1]
    else:
        fig = plt.figure(figsize=figsize)
        axes = None
        top = plt
        
    fig.tight_layout()
    #fig.suptitle(r'response matrix '+ r'$K^{ij}_l$'+'\n',fontsize=20,y=1.03)
    fig.suptitle(plot_title+'\n'+'\n',fontsize=25,y=1.2)
    
    
    xc = bins[1:]-binw*0.5     
    corr_label = 'BDT pred'
    
    top.bar(xc-.5,corr,width=binw,label=corr_label,
            alpha=0.5,color='green',linewidth=0.5, edgecolor='black')
    
    
    
    if absolute :
        top.errorbar( xc, data,ls='None', xerr=np.ones_like(data)*binw*0.5, yerr=np.sqrt(data), color='black', label=r'true $1 \sigma$' )
    else :
        trueh_posw = np.histogram(df[df['weight']>=0][target],np.arange(-1.5,nstats-0.5))[0].ravel() 
        trueh_negw = np.histogram(df[df['weight']<0][target],np.arange(-1.5,nstats-0.5))[0].ravel() 
        
        #reorder the histograms per mass category:
        trueh_posw = OrderPerCategory(trueh_posw,3)
        trueh_negw = OrderPerCategory(trueh_negw,3)
        
            
        hist_mu_0 = trueh_posw - trueh_negw
        
        hist_sigma_0 = np.sqrt(trueh_posw + trueh_negw)
        
        #one sigma
        hist_sigma = 1.*trueh*hist_sigma_0/hist_mu_0
        """    
        #investigate difference to ppf
        #Does not work since at some point values are zero
        CL = .68
        
        hist_s_pos = skellam.ppf(CL,trueh_posw,trueh_negw) - hist_mu_0
        hist_s_pos = 1.*trueh*hist_s_pos/hist_mu_0
        
        hist_s_neg = hist_mu_0 - skellam.ppf(1.-CL,trueh_posw,trueh_negw)
        hist_s_neg = 1.*trueh*hist_s_neg/hist_mu_0
        """
        
        chi2 = np.sum((trueh-predh)**2 / (hist_sigma)**2)
        
        top.errorbar( xc, data,ls='None', xerr=np.ones_like(data)*binw*0.5, yerr=hist_sigma[first:], color='black', 
                     label=r'true $1 \sigma$')
        
         
    if (column == 'class') :
        bottom.set_xticklabels(['high','medium','low'])
    if (column == 'recoPt') :
        lab = ['0-15','15-30','30-45','45-85','85-125','125-200','200-350','350+']
        var[1] = 'GeV'
        bottom.set_xticklabels(np.hstack((lab,lab,lab)),rotation=90,fontsize=12)
    
    if (column == 'recoNjets2p5') :
        lab = ['0','1','2','3','4+']
        bottom.set_xticklabels(np.hstack((lab,lab,lab)))
    
    
    
    #set lines to separate mres cat's
    top.axvline((nstats-1)/3-0.5,linewidth=1.5)
    top.axvline(2*(nstats-1)/3-0.5,linewidth=1.5)
    
    bottom.axvline((nstats-1)/3-0.5,linewidth=1.5)
    bottom.axvline(2*(nstats-1)/3-0.5,linewidth=1.5)
    
    
    #add titles
    DrawMassResCat(plot_instance=top)
    print title
    #print title[-1]
    
    
    if axes == None: axes = fig.axes
    
    if ratio:
        bottom.xaxis.set_ticks(bins+0.5)
    
        
        corr_color = 'black'
            
        if absolute :    
            rdata = corr / data 
            rdata_err = rdata * np.sqrt(data) / data 
            bottom.errorbar( xc, rdata,ls='None', xerr=np.ones_like(rdata)*binw*0.5, yerr=rdata_err, color=corr_color)
        else :
            
            rdata = corr / data 
            rdata_err = rdata * hist_sigma[first:] / data 
            bottom.errorbar( xc, rdata,ls='None', xerr=np.ones_like(rdata)*binw*0.5, yerr=rdata_err, color=corr_color)
            
            
        bottom.set_ylim(0.8,1.2)
        bottom.yaxis.set_ticks(np.arange(0.8,1.3,0.1))
        
        
        bottom.plot( (bins[0],bins[-1]), (1,1), 'k--',linewidth=.6 )
        bottom.set_ylabel('pred / true')
        
    if logy:
        axes[0].set_yscale('log')
    axes[0].set_xlim(bins[0],bins[-1])
    
    unit = None    
    if var != None:
        if type(var) != str:
            var, unit = var
        if unit: var += " [%s]" % unit
        axes[-1].set_xlabel(var)
    ylabel = 'Events'#%1.3g' % binw
    if unit:
        ylabel += ' /' + ' %s' % unit
    axes[0].set_ylabel(ylabel)

    #top.legend(loc='best') 
    top.legend(bbox_to_anchor=(1.35, 1.2))
    
    
    process = r'      process(es): '+'\n'+"\n".join(model_processes)
    details = '\n'+'      details:'+'\n'+ "\n".join([GetBSMDetails(model=m) for m in model_processes if 'BSM' in m])
    
    if DiffGenVariable :
        if 'recoPt' in target :
            gen_level = r'$l=$ '+'gen-$p_{\mathrm{T}}^{\gamma \gamma}$'
            gen_level += ':   ' +GetPtBinRange (bin_index=title[-1])     
    
        if 'recoNjets2p5' in target :
            gen_level = r'$l=$ '+'gen-$N_{\mathrm{jets}}$'
            gen_level += ':   ' +GetNjetsBinRange (bin_index=title[-1])     

        process = r'      process(es): '+'\n'+"\n".join(model_processes)
        details = '\n'+'      details:'+'\n'+ "\n".join([GetBSMDetails(model=m) for m in model_processes if 'BSM' in m])

        if ('BSM' in details) :
            text = gen_level +'\n'+'\n'+process+'\n'+details
        else :
            text = gen_level +'\n'+'\n'+process
        """
        generate response matrix and give it back
        """
        N_reco_pred_ij_l = predh
        N_reco_true_ij_l = trueh
        N_err_reco_true_ij_l = hist_sigma
        
    else :
        gof = r"$\chi^2_\nu = {:.{p}f}$   $\nu = {:}$ dof".format(chi2/(nstats-1.),nstats-1,p=2)
        process = r'      process(es): '+'\n'+"\n".join(model_processes)
        details = '\n'+'      details:'+'\n'+ "\n".join([GetBSMDetails(model=m) for m in model_processes if 'BSM' in m])
        print model_processes
        if ('BSM' in details) :
            print details
            text = gof+'\n'+process+'\n'+details
        else :
            text = gof+'\n'+process
    DrawAdditionalTextBox(text=text,ax=top)

        
    if (savepath != None) :
        try :
            plt.savefig(savepath+'/'+title,bbox_inches='tight')
        except IOError :
            os.mkdir(savepath)
            plt.savefig(savepath+'/'+title,bbox_inches='tight')
     
    return N_reco_pred_ij_l, N_reco_true_ij_l, N_err_reco_true_ij_l
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------    
def draw_data_mc_2 (df,column,first=0,figsize=(8,6),var=None,logy=False,ratio=False,
                savepath = None,title=None,plot_title=[],absolute=True,
                model_processes=[],DiffGenVariable=False,re_weight=False):
    
    """
    Same function as draw_data_mc above but here the mean accuracy and the truncated
    accuracy are reported in the additional legend box.
    params:
                          df : pandas dataframe
                      column : string - specifying the classifer, e.g. recoPt or recoNjets2p5
                     first=0 : int - being the first argument to plot in the histograms. Use first=1
                               if you want to omit the category not reconstructed
               figsize=(8,6) : tuple
                    var=None :
                  logy=False :
                 ratio=False : boolean - specifying whether to plot also the ratio
             savepath = None : string - specifying the save directory
                  title=None : string - title for the save-path
               plot_title=[] :
               absolute=True : boolean - should the histograms be plotted absolute or 
                               with weights
          model_processes=[] : list - ints specifying which processes to be included in
                               the histogram
       DiffGenVariable=False : boolean - put True if the histogram is differntially in 
                               a gen-level bin. 
    returns:
            N_reco_pred_ij_l : 2d-array - giving the number of predicted events in each (i,j) reco-level
                               bin coming from the l gen-level bin.
            N_reco_true_ij_l : 2d-array - giving the number of true events in each (i,j) reco-level
                               bin coming from the l gen-level bin.
    """
    
    target = target_name(column)
    #extract the number of features that belong to column
    nstats = np.unique(df[target]).size
    
    #In case DiffGenVariable==True this will be later filled with the BDT 
    #preiction of number reco events in (ij) reco coming from l gen bin
    N_reco_pred_ij_l = []
    N_reco_true_ij_l = []
    N_err_reco_true_ij_l = []
    
    if 'recoPt' in target :
        nstats =25
    
    if 'recoNjets2p5' in target :
        nstats =16
    
    
    print("There are " + str(nstats) + " features of type " + str(target))
    
    pred_cols = map(lambda x: ("%s_prob_%d" % (target, x)), range(nstats) ) 
    pred_cols_class = map(lambda x: ("%s_prob_%d" % ('class', x)), range(4) ) 
    
     
    if absolute :    
        #the outcome of trueh is an int
        trueh = np.histogram(df[target],np.arange(-1.5,nstats-0.5))[0].ravel() 
        #the predicted hist sums over all events weighted by their probability
        #hence the result is a float
        predh = np.array((df[pred_cols]).sum(axis=0)).ravel()  

        #reorder the histograms per mass category:
        trueh = OrderPerCategory(trueh,3)
        predh = OrderPerCategory(predh,3)
    
    else :
        predh = []
        predh_class = []
        for c in pred_cols :
            predh.append(weighted_average(df,c,'weight'))
        for c in pred_cols_class :
            predh_class.append(weighted_average(df,c,'weight'))
        
        
        
        trueh = np.histogram(df[target],np.arange(-1.5,nstats-0.5),
                             weights=df['weight'])[0].ravel() 
        
        #reorder the histograms per mass category:
        trueh = OrderPerCategory(trueh,3)
        predh = OrderPerCategory(predh,3)
        predh_class = OrderPerCategory(predh_class,3)
        
        if ('Njets' in target) :
            if re_weight :
                print predh
                print predh_class
                y = np.add.reduceat(predh[1:],np.arange(0,len(predh[1:]),5))
                alpha = np.divide(predh_class[1:],y)

                predh[1:][:5] = predh[1:][:5] * alpha[0]
                predh[1:][5:10] = predh[1:][5:10] * alpha[1]
                predh[1:][10:15] = predh[1:][10:15] * alpha[2]
            
    bins = np.arange(-1.5+first,nstats-0.5)
    corr = predh[first:]
    data = trueh[first:]    
    binw=bins[1]-bins[0]
    
    if ratio:
        fig, axes = plt.subplots(2,figsize=figsize,sharex=True,gridspec_kw = {'height_ratios':[3, 1]})
        top = axes[0]
        bottom = axes[1]
    else:
        fig = plt.figure(figsize=figsize)
        axes = None
        top = plt
        
    fig.tight_layout()
    #fig.suptitle(r'response matrix '+ r'$K^{ij}_l$'+'\n',fontsize=20,y=1.03)
    fig.suptitle(plot_title+'\n'+'\n',fontsize=25,y=1.2)
    
    
    xc = bins[1:]-binw*0.5     
    corr_label = 'BDT pred'
    
    top.bar(xc-.5,corr,width=binw,label=corr_label,
            alpha=0.5,color='green',linewidth=0.5, edgecolor='black')
    
    
    
    if absolute :
        top.errorbar( xc, data,ls='None', xerr=np.ones_like(data)*binw*0.5, yerr=np.sqrt(data), color='black', label=r'true $1 \sigma$' )
    else :
        trueh_posw = np.histogram(df[df['weight']>=0][target],np.arange(-1.5,nstats-0.5))[0].ravel() 
        trueh_negw = np.histogram(df[df['weight']<0][target],np.arange(-1.5,nstats-0.5))[0].ravel() 
        
        #reorder the histograms per mass category:
        trueh_posw = OrderPerCategory(trueh_posw,3)
        trueh_negw = OrderPerCategory(trueh_negw,3)
        
        
        #print 'no of pos weight',trueh_posw
        #print 'no of neg weight',trueh_negw
        
        hist_mu_0 = trueh_posw - trueh_negw
        
        weight2 = np.histogram(df['weight']**2,np.arange(-1.5,nstats-0.5))[0].ravel() 
        #print 'weight square', weight2
        
        hist_sigma_0 = np.sqrt(trueh_posw + trueh_negw)
        #print hist_sigma_0
        
        #one sigma
        hist_sigma = 1.*trueh*hist_sigma_0/hist_mu_0
        
        #investigate difference to ppf
        #Does not work since at some point values are zero
        CL = .68
        
        hist_s_pos = skellam.ppf(CL,trueh_posw,trueh_negw) - hist_mu_0
        #print(hist_s_pos)
        hist_s_pos = 1.*trueh*hist_s_pos/hist_mu_0
        
        hist_s_neg = hist_mu_0 - skellam.ppf(1.-CL,trueh_posw,trueh_negw)
        hist_s_neg = 1.*trueh*hist_s_neg/hist_mu_0
        
        #print(hist_s_pos)
        #print(hist_s_neg)
        
        
        #print trueh
        #print predh
        #print hist_sigma
        
        chi2 = np.sum((trueh-predh)**2 / (hist_sigma)**2)
        KS = ks_2samp(trueh,predh)
        KS_reduced = ks_2samp(trueh[1:],predh[1:])
        
        #print chi2
        #print 'reduced chi2: ', chi2/(nstats-1.)
        
        """
        top.errorbar( xc, data,ls='None', xerr=np.ones_like(data)*binw*0.5, yerr=[abs(hist_s_neg[first:]),hist_s_pos[first:]], color='black', 
                     label='true '+str(int(100*CL))+'% CL' )
        """
        top.errorbar( xc, data,ls='None', xerr=np.ones_like(data)*binw*0.5, yerr=hist_sigma[first:], color='black', 
                     label=r'true $1 \sigma$')
        
        
        
        """
        norm = np.histogram(df[target],np.arange(-1.5,nstats-0.5))[0].ravel() 

        abspuw = np.histogram(df[target],np.arange(-1.5,nstats-0.5),weights=df['absweight'])[0].ravel() 
        average = np.divide(abspuw,norm)        
        print(average)
        print(np.divide(trueh,hist_mu_0))
        """
        
    
    
    if (column == 'class') :
        bottom.set_xticklabels(['high','medium','low'])
    if (column == 'recoPt') :
        lab = ['0-15','15-30','30-45','45-85','85-125','125-200','200-350','350+']
        var[1] = 'GeV'
        bottom.set_xticklabels(np.hstack((lab,lab,lab)),rotation=90,fontsize=12)
    
    if (column == 'recoNjets2p5') :
        lab = ['0','1','2','3','4+']
        bottom.set_xticklabels(np.hstack((lab,lab,lab)))
    
    
    
    #set lines to separate mres cat's
    top.axvline((nstats-1)/3-0.5,linewidth=1.5)
    top.axvline(2*(nstats-1)/3-0.5,linewidth=1.5)
    
    bottom.axvline((nstats-1)/3-0.5,linewidth=1.5)
    bottom.axvline(2*(nstats-1)/3-0.5,linewidth=1.5)
    
    
    #add titles
    DrawMassResCat(plot_instance=top)
    print title
    #print title[-1]
    
    
    if axes == None: axes = fig.axes
    
    if ratio:
        bottom.xaxis.set_ticks(bins+0.5)
    
        
        corr_color = 'black'
            
        if absolute :    
            rdata = corr / data 
            rdata_err = rdata * np.sqrt(data) / data 
            bottom.errorbar( xc, rdata,ls='None', xerr=np.ones_like(rdata)*binw*0.5, yerr=rdata_err, color=corr_color)
        else :
            
            rdata = corr / data 
            
            bla = list(rdata)
            #del bla[7::8]
            mean_acc = np.mean(abs(np.array(bla)-1.))
            
            if 'class' in target :
                n_80 = 1
            else :
                n_80 = int(0.8*len(rdata))
                print 'sorted'
                trunc_acc = np.mean(np.sort(abs(np.array(bla)-1.))[:n_80])
                print trunc_acc
                print n_80
                
            
            rdata_err = rdata * hist_sigma[first:] / data 
            bottom.errorbar( xc, rdata,ls='None', xerr=np.ones_like(rdata)*binw*0.5, yerr=rdata_err, color=corr_color)
            
            
        bottom.set_ylim(0.8,1.2)
        bottom.yaxis.set_ticks(np.arange(0.8,1.3,0.1))
        
        
        bottom.plot( (bins[0],bins[-1]), (1,1), 'k--',linewidth=.6 )
        bottom.set_ylabel('pred / true')
        
    if logy:
        axes[0].set_yscale('log')
    axes[0].set_xlim(bins[0],bins[-1])
    
    unit = None    
    if var != None:
        if type(var) != str:
            var, unit = var
        if unit: var += " [%s]" % unit
        axes[-1].set_xlabel(var)
    ylabel = 'Events'#%1.3g' % binw
    if unit:
        ylabel += ' /' + ' %s' % unit
    axes[0].set_ylabel(ylabel)

    #top.legend(loc='best') 
    top.legend(bbox_to_anchor=(1.35, 1.2))
    
    #plt.text(25., 0.2, 'boxed italics text in data coords', style='italic',
    #    bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

    
    
    
    process = r'      process(es): '+'\n'+"\n".join(model_processes)
    details = '\n'+'      details:'+'\n'+ "\n".join([GetBSMDetails(model=m) for m in model_processes if 'BSM' in m])
    
    if DiffGenVariable :
        if 'recoPt' in target :
            gen_level = r'$l=$ '+'gen-$p_{\mathrm{T}}^{\gamma \gamma}$'
            gen_level += ':   ' +GetPtBinRange (bin_index=title[-1])     
    
        if 'recoNjets2p5' in target :
            gen_level = r'$l=$ '+'gen-$N_{\mathrm{jets}}$'
            gen_level += ':   ' +GetNjetsBinRange (bin_index=title[-1])     

        process = r'      process(es): '+'\n'+"\n".join(model_processes)
        details = '\n'+'      details:'+'\n'+ "\n".join([GetBSMDetails(model=m) for m in model_processes if 'BSM' in m])

        if ('BSM' in details) :
            text = gen_level +'\n'+'\n'+process+'\n'+details
        else :
            text = gen_level +'\n'+'\n'+process
        """
        generate response matrix and give it back
        """
        N_reco_pred_ij_l = predh
        N_reco_true_ij_l = trueh
        N_err_reco_true_ij_l = hist_sigma
        
    else :
        gof = r"mean accuracy: {:.{p}f} %".format(100.*mean_acc,p=1)
        if ('recoPt' in target or 'recoNjets2p5' in target) :
            gof += '\n'+r"trunc accuracy: {:.{p}f} %".format(100.*trunc_acc,p=1)
        #gof = r"$\chi^2_\nu = {:.{p}f}$   $\nu = {:}$ dof".format(chi2/(nstats-1.),nstats-1,p=2)
        #gof += '\n'+'KS-test (p-value): '+r"$p = {:.{p}f}$".format(KS[1],p=3)
        #gof += '\n'+'KS-test: '+r"$p = {:.{p}f}$".format(KS_reduced[1],p=2)
        process = r'      process(es): '+'\n'+"\n".join(model_processes)
        details = '\n'+'      details:'+'\n'+ "\n".join([GetBSMDetails(model=m) for m in model_processes if 'BSM' in m])
        print model_processes
        if ('BSM' in details) :
            print details
            text = gof+'\n'+process+'\n'+details
        else :
            text = gof+'\n'+process
    DrawAdditionalTextBox(text=text,ax=top)

        
    if (savepath != None) :
        try :
            plt.savefig(savepath+'/'+title,bbox_inches='tight')
        except IOError :
            os.mkdir(savepath)
            plt.savefig(savepath+'/'+title,bbox_inches='tight')
     
    return N_reco_pred_ij_l, N_reco_true_ij_l, N_err_reco_true_ij_l
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
def DrawAdditionalTextBox(ax,text) :
    """
    This function displays some text on the right of the plot
    params:
          ax : axis object
        text : string - specifies the text to be added in the plot
    """
    plt.text(1.04, 0.5, text, size=18, rotation=0.,
             ha="left", va="center",
             transform = ax.transAxes,
             bbox=dict(boxstyle="round",
                       fc=(1, 1, 1),
                      ),
            )
# ---------------------------------------------------------------------------

    
# ---------------------------------------------------------------------------
def GetBSMDetails (model) :
    """
    This function retruns some details about the BSM model
    params:
        model : string - being one of the following options:
                1) BSM1
                2) BMS2
                3) BSM3
    """
    if ('BSM1' in model) :
        return 'BSM1: '+'\n'+ r'$m_\tilde{b}=350 \, \mathrm{ GeV}$ '+'\n'+'$m_\mathrm{LSP}=150 \, \mathrm{ GeV}$ '
    if ('BSM2' in model) :
        return 'BSM2: '+'\n'+ r'$m_\tilde{b}=450 \, \mathrm{ GeV}$ '+'\n'+'$m_\mathrm{LSP}=200 \, \mathrm{ GeV}$ '
    if ('BSM3' in model) :
        return 'BSM3: '+'\n'+ r'$m_\tilde{b}=500 \, \mathrm{ GeV}$ '+'\n'+'$m_\mathrm{LSP}=1 \, \mathrm{ GeV}$ '
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def GetNjetsBinRange (bin_index) :
    """
    This function returns the Njets range as a string given the Njets-bin index
    params:
        bin_index : string - specifies the bin index 0,1,2,3,...,N
    returns:
        bin_range : string - being the Njets range corresponding to the
                    bin with bin index bin_index
    """
    if (bin_index == '0') :
        return '0'
    if (bin_index == '1') :
        return '1'
    if (bin_index == '2') :
        return '2'
    if (bin_index == '3') :
        return '3'
    if (bin_index == '4') :
        return '4+'
# ---------------------------------------------------------------------------

    
    
# ---------------------------------------------------------------------------
def GetPtBinRange (bin_index) :
    """
    This function returns the pT range as a string given the pT-bin index
    params:
        bin_index : string - specifies the bin index 0,1,2,3,...,N
    returns:
        bin_range : string - being the pT range corresponding to the
                    bin with bin index bin_index
    """
    if (bin_index == '0') :
        return '0-15 GeV'
    if (bin_index == '1') :
        return '15-30 GeV'
    if (bin_index == '2') :
        return '30-45 GeV'
    if (bin_index == '3') :
        return '45-85 GeV'
    if (bin_index == '4') :
        return '85-125 GeV'
    if (bin_index == '5') :
        return '125-200 GeV'
    if (bin_index == '6') :
        return '200-350 GeV'
    if (bin_index == '7') :
        return '350+ GeV'
# ---------------------------------------------------------------------------
    
    
    
# ---------------------------------------------------------------------------
def DrawMassResCat (plot_instance) :
    """
    This function divides the plot with 2 blue lines in 3 mass resolution 
    categories.
    params:
        plot_instance : axis object 
    returns:
    """
    top = plot_instance
    
    leftT, width = .1, .5
    bottomT, height = 1.1, .5
    top.text(leftT, bottomT, r'high $\frac{\sigma_M}{M}$',
        horizontalalignment='left',
        verticalalignment='top',
        color='blue',
        fontsize=16,
        transform=top.transAxes)

    leftT, width = .41, .5
    bottomT, height = 1.1, .5
    top.text(leftT, bottomT, r'medium $\frac{\sigma_M}{M}$',
        horizontalalignment='left',
        verticalalignment='top',
        color='blue',
        fontsize=16,
        transform=top.transAxes)
    
    leftT, width = .77, .5
    bottomT, height = 1.1, .5
    top.text(leftT, bottomT, r'low $\frac{\sigma_M}{M}$',
        horizontalalignment='left',
        verticalalignment='top',
        color='blue',
        fontsize=16,
        transform=top.transAxes)
    
    leftT, width = .75, .5
    bottomT, height = 1.25, .5
    top.text(leftT, bottomT, '(Work in progress)',
        horizontalalignment='left',
        verticalalignment='top',
        fontweight='bold',
        color='black',
        fontsize=12,
        transform=top.transAxes)
# ---------------------------------------------------------------------------



#----------------------------------------------------------------------------
def GetXsecinBins (profile_object, binWidth) :
    """
    params  :
        profile_object : instance of the class LikelihoodProfile inside 
                         LikelihoodProfile.py
              binWidth : 1-d array corresponding to the binning of the
                         kinematic observable of interest.
    retruns :
    """
    binw = binWidth
    
    data = profile_object.GetDeltaSigmaTrue()
    corr = profile_object.DeltaSigma_BF
    corr_yerr_pos = profile_object.plus_uncert
    corr_yerr_neg = profile_object.minus_uncert
    
    # divide by binwidth
    data = data / binw
    corr = corr / binw
    corr_yerr_pos = corr_yerr_pos / binw
    corr_yerr_neg = corr_yerr_neg / binw
    
    # multiply last bin by 10 to make it visible
    data[-1] = data[-1]*10
    corr[-1] = corr[-1]*10
    corr_yerr_pos[-1] = corr_yerr_pos[-1]*10
    corr_yerr_neg[-1] = corr_yerr_neg[-1]*10
    return data, corr, corr_yerr_pos, corr_yerr_neg
#============================================================================

#----------------------------------------------------------------------------
def PlotdifferentialPtXsecSpectrum(profile_object,profileSMresponse=None) :
    """
    This function is plotting the fiducial differential cross section (x-sec) 
    as a function of recoPt.
    It can plots the true x-sec together with the computed from the Likelihood
    profiles. One can also compare 2 computed profiles simultaneously.
    params  :
        profile_object : instance of the class LikelihoodProfile inside LikelihoodProfile.py
     profileSMresponse : instance of the class LikelihoodProfile inside LikelihoodProfile.py
                         with the profiles computed with the SM MC test sample or any other
                         response matrix that should be compared to the one used for the 
                         profiles in profile_object. (default : None)
    returns :
    """
    bins = np.array([0.,15.,30.,45.,85.,125.,200.,350.,450.])
            
    binw = np.array([15.,15.,15.,40.,40.,75.,150.,100.])
    data, corr, corr_yerr_pos, corr_yerr_neg = GetXsecinBins(profile_object=profile_object,binWidth=binw)
    
    fig, axes = plt.subplots(2,figsize=(8,8),sharex=True,gridspec_kw = {'height_ratios':[3, 1]})
    TitelAddtext = 'Differential cross section'+'\n'+'Model: '+profile_object.mode
    fig.suptitle(TitelAddtext,y=1.05,fontsize=12)
    
    
    
    top = axes[0]
    bottom = axes[1]
    fig.tight_layout()
    top.set_ylabel(r'${\mathrm{d}\sigma} / {p_{\mathrm{T}}^{\gamma \gamma}} \, [\mathrm{pb} \cdot \mathrm{GeV}^{-1}]$ ',
                   fontsize=15)
    
    
    top.bar(bins[:-1],data,width=binw,color='blue', label=r'true fid diff x-sec',alpha=.6,linewidth=0)
    top.errorbar(bins[:-1]+binw/2.,corr,xerr=binw/2.,yerr=[corr_yerr_pos,corr_yerr_neg],color='black',ls='None',
                label=r'fid diff x-sec pred BDT',elinewidth=1.5,capsize=4,capthick=1.4,marker='o')
    corr_color = 'black'
    rdata = corr / data 
    bottom.errorbar(bins[:-1]+binw/2. , rdata,ls='None', xerr=binw/2., 
                    yerr=[corr_yerr_pos/data,corr_yerr_neg/data], color=corr_color,elinewidth=1.5,capsize=4,
                    capthick=1.,marker='o')

    if (profileSMresponse != None) :
        data, corr, corr_yerr_pos, corr_yerr_neg = GetXsecinBins(profile_object=profileSMresponse,binWidth=binw)
        
        top.errorbar(bins[:-1]+binw/2.,corr,xerr=binw/2.,yerr=[corr_yerr_pos,corr_yerr_neg],color='red',ls='None',
                label=r'fid diff x-sec SM response',linewidth=1.5,elinewidth=1.5,capthick=1.,marker='^')
    
        
        #stupid work around to get nice label in the legend corresponding to rectabgular patch
        #top.bar(500,1,1,color='r',alpha=.5,label='fid diff x-sec SM response',linewidth=0)
        
        #makeErrorBoxes(ax=top,xdata=bins[:-1]+binw/2.,
        #               ydata=corr,
        #               xerror=np.array([binw/2.,binw/2.]),
        #               yerror=np.array([corr_yerr_pos,corr_yerr_neg]))
        
        
        
        corr_color = 'red'
        rdata = corr / data 
        bottom.errorbar(bins[:-1]+binw/2. , rdata,ls='None', xerr=binw/2., 
                    yerr=[corr_yerr_pos/data,corr_yerr_neg/data], color=corr_color,elinewidth=1.5,capthick=1.,
                       marker='^')
        
        #bottom.plot(bins[:-1]+binw/2. , rdata,color=corr_color,marker='^',linestyle='') 
        #makeErrorBoxes(ax=bottom,
        #               xdata=bins[:-1]+binw/2.,
        #               ydata=rdata,
        #               xerror=np.array([binw/2.,binw/2.]),
        #               yerror=np.array([corr_yerr_neg/data,corr_yerr_pos/data]),
        #             fc=corr_color)
  

    #top.set_ylim(0.001)
    top.set_xlim(0,450.)
    top.set_yscale('log')
    top.legend(loc='best')
    
    top.text(400,corr[-1]+corr_yerr_pos[-1]*2.5,r'$\times 10$',rotation='0')
    
    bottom.xaxis.set_ticks(bins)


    

    bottom.set_ylim(0.5,1.5)
    bottom.yaxis.set_ticks(np.arange(0.5,1.6,0.5))
    bottom.set_xlabel(r'$p_{\mathrm{T}}^{\gamma \gamma} \, [\mathrm{GeV}]$',fontsize=15)


    bottom.plot( (bins[0],bins[-1]), (1,1), 'k-',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (1.1,1.1), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (1.2,1.2), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (1.3,1.3), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (1.4,1.4), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (0.9,0.9), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (0.8,0.8), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (0.7,0.7), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (0.6,0.6), color='gray',linewidth=.6 )
    bottom.set_ylabel('pred / true')

    path = '/mnt/t3nfs01/data01/shome/jandrejk/higgs_model_dep/MoriondAnalysis/plots/XsecHist/'
    path = path+profile_object.obs+'/'
    lp.ensure_dir(file_path=path)
    print 'save', path+'xsec_'+profile_object.mode+'.png'
    plt.savefig(path+'xsec_'+profile_object.mode+'.png',bbox_inches='tight')
    
    
    
    plt.show()
#============================================================================


def PlotdifferentialNjetsXsecSpectrum(profile_object,profileSMresponse=None) :
    """
    This function is plotting the fiducial differential cross section (x-sec) 
    as a function of recoNjets2p5.
    It can plots the true x-sec together with the computed from the Likelihood
    profiles. One can also compare 2 computed profiles simultaneously.
    params  :
        profile_object : instance of the class LikelihoodProfile inside LikelihoodProfile.py
     profileSMresponse : instance of the class LikelihoodProfile inside LikelihoodProfile.py
                         with the profiles computed with the SM MC test sample or any other
                         response matrix that should be compared to the one used for the 
                         profiles in profile_object. (default : None)
    returns :
    """
    bins = np.array([0.,1.,2.,3.,4.,5.])-.5
            
    binw = np.array([1.,1.,1.,1.,1.])
    data, corr, corr_yerr_pos, corr_yerr_neg = GetXsecinBins(profile_object=profile_object,binWidth=binw)
    
    fig, axes = plt.subplots(2,figsize=(6,6),sharex=True,gridspec_kw = {'height_ratios':[3, 1]})
    
    TitelAddtext = 'Differential cross section'+'\n'+'Model: '+profile_object.mode
    fig.suptitle(TitelAddtext,y=1.05,fontsize=12)
    
    
    
    top = axes[0]
    bottom = axes[1]
    fig.tight_layout()
    top.set_ylabel(r'${\mathrm{d}\sigma} / {N_{\mathrm{jets}}} \, [\mathrm{pb}]$ ',
                   fontsize=15)
    
    
    
    top.bar(bins[:-1],data,width=binw,color='blue', label=r'true fid diff x-sec',alpha=.6,linewidth=0)
    top.errorbar(bins[:-1]+binw/2.,corr,xerr=binw/2.,yerr=[corr_yerr_pos,corr_yerr_neg],color='black',ls='None',
                label=r'fid diff x-sec pred BDT',elinewidth=1.5,capsize=4,capthick=1.4,marker='o')
    corr_color = 'black'
    rdata = corr / data 
    bottom.errorbar(bins[:-1]+binw/2. , rdata,ls='None', xerr=binw/2., 
                    yerr=[corr_yerr_pos/data,corr_yerr_neg/data], color=corr_color,elinewidth=1.5,capsize=4,
                    capthick=1.,marker='o')

    if (profileSMresponse != None) :
        data, corr, corr_yerr_pos, corr_yerr_neg = GetXsecinBins(profile_object=profileSMresponse,binWidth=binw)
        
        top.errorbar(bins[:-1]+binw/2.,corr,xerr=binw/2.,yerr=[corr_yerr_pos,corr_yerr_neg],color='red',ls='None',
                label=r'fid diff x-sec SM response',linewidth=1.5,elinewidth=1.5,capthick=1.,marker='^')
    
        
        corr_color = 'red'
        rdata = corr / data 
        bottom.errorbar(bins[:-1]+binw/2. , rdata,ls='None', xerr=binw/2., 
                    yerr=[corr_yerr_pos/data,corr_yerr_neg/data], color=corr_color,elinewidth=1.5,capthick=1.,
                       marker='^')
        
   
    #top.set_ylim(0.001)
    top.set_xlim(-0.5,4.5)
    bottom.set_xlim(-0.5,4.5)
    
    top.set_yscale('log')
    top.legend(loc='best')
    
    
    bottom.xaxis.set_ticks(bins[:-1]+0.5)


    

    bottom.set_ylim(0.5,1.5)
    bottom.yaxis.set_ticks(np.arange(0.5,1.6,0.5))
    bottom.set_xlabel(r'$N_{\mathrm{jets}} $',fontsize=15)


    bottom.plot( (bins[0],bins[-1]), (1,1), 'k-',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (1.1,1.1), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (1.2,1.2), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (1.3,1.3), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (1.4,1.4), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (0.9,0.9), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (0.8,0.8), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (0.7,0.7), color='gray',linewidth=.6 )
    bottom.plot( (bins[0],bins[-1]), (0.6,0.6), color='gray',linewidth=.6 )
    bottom.set_ylabel('pred / true')

    path = '/mnt/t3nfs01/data01/shome/jandrejk/higgs_model_dep/MoriondAnalysis/plots/XsecHist/'
    path = path+profile_object.obs+'/'
    lp.ensure_dir(file_path=path)
    print 'save', path+'xsec_'+profile_object.mode+'.png'
    plt.savefig(path+'xsec_'+profile_object.mode+'.png',bbox_inches='tight')
    
    
    
    plt.show()

    
#-------------------------------------------------------------------------------
def makeErrorBoxes(ax,xdata,ydata,xerror,yerror,fc='r',ec='None',alpha=0.5):

    # Create list for all the error patches
    errorboxes = []

    # Loop over data points; create box from errors at each point
    for xc,yc,xe,ye in zip(xdata,ydata,xerror.T,yerror.T):
        rect = Rectangle((xc-xe[0],yc-ye[0]),xe.sum(),ye.sum())
        errorboxes.append(rect)

    # Create patch collection with specified colour/alpha
    pc = PatchCollection(errorboxes,facecolor=fc,alpha=alpha,edgecolor=ec)

    # Add collection to axes
    
    ax.add_collection(pc)
#===============================================================================
    