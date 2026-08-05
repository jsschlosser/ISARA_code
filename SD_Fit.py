import fit
import numpy as np
import SD_Fit_Figs
from scipy.stats.stats import pearsonr 
def Run(n_modes,p0,dp,dndlogdp,abs_coef,sca_coef,file_location):
	"""
	Returns N (cm-3), GM (nm), and GSD from a measured number concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) fit.py must be present in a directory that is in your PATH
    
    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)    

    Requirements
    ------------ 
    * ``numpy``
    * ``scipy``

	:param p0: 3n guesses where n is the number of modes you are fitting. They should be in the order [Ni, GMi, GSDi] for i=1 to i=number of modes
	:type p0: list                      
	:return: Dictionary (Results) with the retrieved complex refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in measured and validation wavelengths
	:rtype: numpy dictionary
	"""
	results = {}
	Ldp = len(dndlogdp[:,0])
	Ldata = len(dndlogdp[0,:])
	Lparams = 3
	results['predicted_size_distribution'] = np.full((Ldp,Ldata),np.nan)
	for i_mode in range(n_modes):
		results[f'fit_params_{i_mode}'] = np.full((Lparams,Ldata),np.nan)
		results[f'stdev_fit_params_{i_mode}'] = np.full((Lparams,Ldata),np.nan)
	results['log10_P'] = np.full((Ldata),np.nan)
	results['R'] = np.full((Ldata),np.nan)
	model = fit.LogNormal()
	for data_i in range(Ldata):
		fit_results = None
		nanfilter = np.where(np.logical_not(np.isnan(dndlogdp[:,data_i])))[0]
		if len(nanfilter)>4:
			try:
				fit_results = model.fit(dp[nanfilter], dndlogdp[nanfilter,data_i], modes=n_modes, p0=p0)
			except Exception as e:
				print("Failed due to maximum number of iterations")
		if fit_results is not None:
			R,P = pearsonr(fit_results.fittedvalues,dndlogdp[nanfilter,data_i])
			log10_pvalues=np.log10(P)
			fitres = fit_results.predict(dp)
			fitres[fitres < 0] = 0
			results['predicted_size_distribution'][nanfilter.min():nanfilter.max()+1,data_i] = fitres[nanfilter.min():nanfilter.max()+1]
			for i_mode in range(n_modes):
				results[f'fit_params_{i_mode}'][:,data_i] = fit_results.params[i_mode] # N (cm-3), GM (nm), and GSD
				results[f'stdev_fit_params_{i_mode}'][:,data_i] = fit_results.errors[i_mode] 
			results['log10_P'][data_i] = log10_pvalues
			results['R'][data_i] = R			
			if ((R > 0.99) & (log10_pvalues < -4)):
				SD_Fit_Figs.plot_SD(dp,dndlogdp[:,data_i],results['predicted_size_distribution'][:,data_i],data_i,R,log10_pvalues,file_location)
	return results