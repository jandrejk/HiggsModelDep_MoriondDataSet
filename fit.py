# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 09:45:40 2017

@author: christina reissel
"""
import ROOT as r
import matplotlib.pyplot as plt
import rootpy.plotting.root2matplotlib as rplt
from rootpy.plotting import Hist, HistStack, Legend, Canvas
from rootpy.io import root_open
import rootpy
import numpy as np
import scipy.optimize as optimize
import scipy.stats    as stats
from array import *
import matplotlib.pylab as pylab

# Generation of toy data from generator information
def generate_data(input_file, output_file, alpha, beta):
	file = root_open(input_file,'READ')

	##### Signal
	signal = file.Get('ttH_hbb__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit')

	##### Background
	diff_background = ['ttbarPlusB', 'ttbarPlus2B', 'ttbarOther', 'ttbarPlusBBbar', 'ttbarPlusCCbar']
	background = { i : file.Get('{0}__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit'.format(i)) for i in diff_background}	

	##### Variation
	variation_up = { i : file.Get('{0}__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit__CMS_scaleFragmentation_jUp'.format(i)) for i in diff_background}
	variation_down = { i : file.Get('{0}__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit__CMS_scaleFragmentation_jDown'.format(i)) for i in diff_background}

	# Rebinning histograms to avoid empty bins
	xbins = array('d',[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
	signal = signal.Rebin(4, 'signal_rebinned', xbins)
	for i in diff_background:
		background[i] = background[i].Rebin(4, 'background_{0}_rebinned'.format(i), xbins)
		variation_up[i] = variation_up[i].Rebin(4, 'variationUp_{0}_rebinned'.format(i), xbins)
		variation_down[i] = variation_down[i].Rebin(4, 'variationDown_{0}_rebinned'.format(i), xbins)
	

	# Saving data file
	output_file = root_open(output_file, 'RECREATE')

	signal.Write('signal__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit')
	print 'signal, nbins=', signal.GetSize()
	print 'signal, rate=', signal.Integral()


	# Generate data set:
	data = signal.Clone('data_obs__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit')

	signal_coeff = alpha
	background_coeff = {i : beta for i in diff_background}

	data.Scale(signal_coeff)

	for i in diff_background:
		data.Add(background[i], background_coeff[i])

	data.Write()
	print 'data, nbins=', data.GetSize()

	for i in diff_background:
		if i != 'ttbarPlusB':
			background['ttbarPlusB'].Add(background[i])
			variation_up['ttbarPlusB'].Add(variation_up[i])
			variation_down['ttbarPlusB'].Add(variation_down[i])
	variation_down['ttbarPlusB'].Write('background__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit__CMS_scaleFragmentation_jDown')
	variation_up['ttbarPlusB'].Write('background__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit__CMS_scaleFragmentation_jUp')
	background['ttbarPlusB'].Write('background__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit')
	print 'var_up, nbins=', variation_up['ttbarPlusB'].GetSize()
	print 'var_down, nbins=', variation_down['ttbarPlusB'].GetSize()
	print 'background, nbins=', background['ttbarPlusB'].GetSize()
	print 'background, rate=', background['ttbarPlusB'].Integral()

	print 'Signal:', signal_coeff
	for i in diff_background:
		print 'Background '+str(i)+':', background_coeff[i]

	output_file.close()

	print 'Generation of data finished'


def fit_python(file_mc, file_data):

	file_mc = root_open(file_mc,'READ')
	file_data = root_open(file_data, 'UPDATE')

	# Load histograms from file

	##### Data
	data = file_data.Get('data_obs__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit')

	##### Signal
	signal = file_mc.Get('ttH_hbb__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit')

	##### Background
	diff_background = ['ttbarPlusB', 'ttbarPlus2B', 'ttbarOther', 'ttbarPlusBBbar', 'ttbarPlusCCbar']
	background = { i : file_mc.Get('{0}__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit'.format(i)) for i in diff_background}

	variation_up = { i : file_mc.Get('{0}__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit__CMS_scaleFragmentation_jUp'.format(i)) for i in diff_background}
	variation_down = { i : file_mc.Get('{0}__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit__CMS_scaleFragmentation_jDown'.format(i)) for i in diff_background}

	# Rebinning histograms to avoid empty bins
	xbins = array('d',[4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
	#data = data.Rebin(4, 'data_rebinned', xbins)
	signal = signal.Rebin(4, 'signal_rebinned', xbins)
	for i in diff_background:
		background[i] = background[i].Rebin(4, 'background_{0}_rebinned'.format(i), xbins)
		variation_up[i] = variation_up[i].Rebin(4, 'variationUp_{0}_rebinned'.format(i), xbins)
		variation_down[i] = variation_down[i].Rebin(4, 'variationDown_{0}_rebinned'.format(i), xbins)
		

	# Initializing arrays from histograms
	Nbin = data.GetSize()

	arr_data = np.zeros(Nbin)
	arr_signal = np.zeros(Nbin)
	arr_background = { i : np.zeros(Nbin) for i in diff_background}
	arr_variationUp = { i : np.zeros(Nbin) for i in diff_background}
	arr_variationDown = { i : np.zeros(Nbin) for i in diff_background}


	for i in range(Nbin):
		arr_data[i] = data.GetBinContent(i)
		arr_signal[i] = signal.GetBinContent(i)
		for j in diff_background:
			(arr_background[j])[i] = background[j].GetBinContent(i)
			(arr_variationUp[j])[i] = variation_up[j].GetBinContent(i)
			(arr_variationDown[j])[i] = variation_down[j].GetBinContent(i)

	print 'Array Data:', arr_data
	print 'Array Signal:', arr_signal
	#for j in diff_background:
	#	print j, arr_background[j]

	arr_combined_background = np.zeros(Nbin)
	arr_combined_variationUp = np.zeros(Nbin)
	arr_combined_variationDown = np.zeros(Nbin)
	for j in diff_background:
		arr_combined_background += arr_background[j]
		arr_combined_variationUp += arr_variationUp[j]
		arr_combined_variationDown += arr_variationDown[j]

	print 'Array Background:', arr_combined_background
	print 	'Array Variation up:', arr_combined_variationUp
	print 'Array Variation down:', arr_combined_variationDown

	# Calculation error
	sigma_up = (arr_combined_variationUp - arr_combined_background)/(arr_combined_background)
	sigma_down = (arr_combined_background - arr_combined_variationDown)/(arr_combined_background)
	sigma = 1.0*(np.abs(sigma_up) + np.abs(sigma_down))/2.0
	print 'Sigma:', sigma

	"""# Fill variation histograms with symmetric sigma
	hist_variationUp = rootpy.plotting.Hist(4,4.0,8.0,name='background__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit__CMS_scaleFragmentation_jUp')
	hist_variationDown = rootpy.plotting.Hist(4,4.0,8.0,name='background__sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit__CMS_scaleFragmentation_jDown')
	variationUp = arr_combined_background + sigma
	variationDown = arr_combined_background - sigma
	for i in range(4):
		hist_variationUp.SetBinContent(i+1, variationUp[i+1])
		hist_variationDown.SetBinContent(i+1, variationDown[i+1])

	hist_variationUp.Rebin(4,'hist_variationUp_rebinned', xbins)
	hist_variationDown.Rebin(4, 'hist_variationDwon_rebinned', xbins)

	hist_variationUp.Write()
	hist_variationDown.Write()"""
	 

	
	#print 'Sigma_up:', sigma_up
	#print 'Sigma_down:', sigma_down

	
	# Building function for calculating likelihood for each bin
	def nll(p, arr_data, arr_signal, arr_combined_background, sigma, i):
		
		alpha = p[0]
		theta = p[1]
		#sigma = 0.1
		mu = alpha*arr_signal[i] + theta*arr_combined_background[i]
		llog = stats.poisson.logpmf(int(arr_data[i]), mu) + np.log(1/np.sqrt(2*3.14*sigma[i])*np.exp(-(theta-1)**2/(2*sigma[i])))
		#print 'exp:', stats.poisson.logpmf(int(arr_data[i]), mu)		
		#print 'norm:', np.log(1/np.sqrt(2*3.14*sigma)*np.exp(-(theta-1)**2/(2*sigma)))
		#print -(llog)
		return -(llog)


	#p0 = np.array([10.0, 2.0])
	#for j in range(Nbin):
	#	print nll(p0, arr_data, arr_signal, arr_combined_background, j)

	
	# Minimizing nll for each bin
	dim = len(diff_background) + 1

	"""p0 = np.array([1.0, 2.0])
	for j in range(Nbin):
		result = optimize.minimize(nll, p0, (arr_data, arr_signal, arr_combined_background, j), bounds = [(None,None), (None, None)])
		print 'alpha:', result.x[0], 'theta:', result.x[1]
		#for i in range(dim):
		#	if i == 0:
		#		print 'Signal'
		#	else:
		#		print diff_background[i-1]

		#	print result.x[i]

	
	# Minimizing nll for all bins simultaneously
	# Definition of summed nll
	def sum_nll(p, arr_data, arr_signal, arr_combined_background, Nbin):
		list_nll = []
		for j in range(Nbin):
			list_nll.append(nll(p, arr_data, arr_signal, arr_combined_background, j))
		sum_nll = 0.0
		for l in range(len(list_nll)):
			sum_nll = sum_nll + list_nll[l]

		return sum_nll


	p0 = np.array([1.0, 2.0])
	result = optimize.minimize(sum_nll, p0, (arr_data, arr_signal, arr_combined_background, Nbin), bounds = [(None,None), (None,None)])
	print result
	#print 'Sum_nll minimizing result:'
	#for i in range(dim):
	#	print result.x[i]"""


	# Profile likelihood for each bin
	def profile_likelihood(alpha, arr_data, arr_signal, arr_combined_background, sigma, i):
		# nominator
		def nll_nom(theta, alpha, arr_data, arr_signal, arr_combined_background, sigma, i):
			p = np.array([alpha, theta])
			return nll(p, arr_data, arr_signal, arr_combined_background, sigma, i)

		theta = 2.0
		result_nom = optimize.minimize(nll_nom, theta, (alpha, arr_data, arr_signal, arr_combined_background, sigma, i))
		#print result_nom


		# denominator
		p0 = np.array([1.0, 2.0])
		result_denom = optimize.minimize(nll, p0, (arr_data, arr_signal, arr_combined_background, sigma, i))
		#print result_denom

		# Combine
		return nll_nom(result_nom.x[0], alpha, arr_data, arr_signal, arr_combined_background, sigma, i) - nll(result_denom.x, arr_data, arr_signal, arr_combined_background, sigma, i)

	#print profile_likelihood(10.0, arr_data, arr_signal, arr_combined_background, 4)
	#alpha = 9.0
	#result = optimize.minimize(profile_likelihood, alpha, (arr_data, arr_signal, arr_combined_background, 4), method = 'Powell')
	#print result
	#for i in range(12):
	#	plt.plot(i, profile_likelihood(i, arr_data, arr_signal, arr_combined_background, 4),'b*')
	#plt.show()

	

	# Definition of summed profile likelihood
	def sum_profile(alpha, arr_data, arr_signal, arr_combined_background, sigma, Nbin):
		list_profile = []
		for j in range(Nbin):
			list_profile.append(profile_likelihood(alpha, arr_data, arr_signal, arr_combined_background, sigma, j))
		sum_profile = 0.0
		for l in range(len(list_profile)):
			sum_profile = sum_profile + list_profile[l]

		return sum_profile

	# Function to plot likelihood


	x = np.linspace(-2.0,2.0,100)
	x.tolist()
	y = []
	for i in x:
		y.append(sum_profile(i, arr_data, arr_signal, arr_combined_background, sigma, Nbin))
	plt.rc('text', usetex=True)
	plt.rc('font', family='serif')

	params = {'legend.fontsize': 'x-large', 'axes.labelsize': 'x-large', 'axes.titlesize':'x-large', 'xtick.labelsize':'x-large', 'ytick.labelsize':'x-large'}
	pylab.rcParams.update(params)

	fig = plt.figure()
	plt.plot(x,y,'b-')
	xh = np.linspace(-2.0,2.0,200)
	yh = np.array([0.5 for i in xrange(len(xh))])
	plt.plot(xh, yh,'r-') 
	plt.xlabel(r'$\mu$')
	plt.ylabel(r'$-\ln(\lambda(\mu))$')
	fig.savefig('profile_likelihood.pdf')
	#plt.show()

	alpha = 0.0
	result = optimize.minimize(sum_profile, alpha, (arr_data, arr_signal, arr_combined_background, sigma, Nbin), method = 'Powell')
	print result.x

	# Error of estimator
	err = lambda x: sum_profile(x, arr_data, arr_signal, arr_combined_background, sigma, Nbin)-(sum_profile(result.x, arr_data, arr_signal, arr_combined_background, sigma, Nbin)+0.5)

	down = result.x - optimize.fsolve(err,(result.x - 3.0))[0]
	up = optimize.fsolve(err,(result.x + 0.01))[0] - result.x
	print down
	print up
	

	print 'Function fit_python() works'	

	

##### Main
if __name__ == "__main__":  

	##### Settings  
	
	input_file = '/mnt/t3nfs01/data01/shome/creissel/data/simulation/sl_jge6_tge4__btag_LR_4b_2b_btagCSV_logit.root'
	#output_file = 'data.root'

	#beta = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
	#output = {0.0 : 'file00.root', 0.1 : 'file01.root', 0.2 : 'file02.root', 0.3 : 'file03.root', 0.4 : 'file04.root', 0.5 : 'file05.root', 0.6 : 'file07.root', 0.7 : 'file07.root', 0.8 : 'file08.root', 0.9 : 'file09.root', 1.0 : 'file1.root', 2.0 : 'file2.root', 3.0 : 'file3.root', 4.0 : 'file4.root', 5.0 : 'file5.root', 6.0 : 'file6.root', 7.0 : 'file7.root', 8.0 : 'file8.root', 9.0 : 'file9.root', 10.0 : 'file10.root'}

	beta = [1.1, 1.2, 1.3, 1.4, 1.5]
	output = {1.1 : 'file11.root', 1.2 : 'file12.root', 1.3 : 'file13.root', 1.4 : 'file14.root', 1.5 : 'file15.root'}

	print 'Settings complete'
	
	for i in beta:
		generate_data(input_file, output[i], 1.0, i)
		#fit_python(input_file, output[i])"""

	#generate_data(input_file, 'control2.root', 1.010, 2.000)
	#generate_data(input_file, output_file, 1.0, beta)
	#fit_python(input_file, 'file05.root')
	
	
	
	


	
	
