import numpy as np
import scipy.stats as st 
from scipy.stats import t
from scipy.stats import ttest_ind  
from scipy.stats.stats import pearsonr 
from sklearn.feature_selection import f_regression
from libpysal.weights import lat2W
from esda.moran import Moran
from libpysal.weights import W

def Comparison(x,y,prctils):
	"""
	Calculate closure statistics between two measurements of the same property. This procedure filters out missing values internally and computes key statistical indicators to evaluate the agreement between two 1-D arrays.

    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)    
    
    .. note::
       Missing or invalid values in `x` and `y` are automatically filtered
       out before calculations take place.

    Requirements
    ------------ 
    * ``numpy``
    * ``scipy``
	* ``esda``
	* ``libpysal``

    :param x: First measurement array.
    :type x: 1-D array
    :param y: Second measurement array. Must have the same length as `x`.
    :type y: 1-D array
    :param prctils: The physical units of the measurements, defaults to 'units'.
    :type prctils: str, optional
    :return: A 1-D array containing the calculated closure statistics in the following order:

		* **R**: Correlation coefficient
		* **p-value**: Probability that the two parameters are not correlated
		* **NMAD**: Normalized Mean Absolute Deviation
		* **MAD_[units]**: Mean Absolute Deviation in user-provided units
		* **NRMSD**: Normalized Root-Mean Squared Deviation
		* **RMSD_[units]**: Root-Mean Squared Deviation in user-provided units
		* **x_min_[units]**: Minimum valid value of x
		* **x_max_[units]**: Maximum valid value of x
		* **y_min_[units]**: Minimum valid value of y
		* **y_max_[units]**: Maximum valid value of y
		* **count**: Number of points where both x and y had valid values

    :rtype: numpy.ndarray
    :raises ValueError: If `x` and `y` have mismatched lengths.
    """
	y = y.reshape(-1, 1)
	y[np.isinf(y)]=np.nan
	y[y<0]=np.nan
	x = x.reshape(-1, 1)
	x[np.isinf(x)]=np.nan
	x[x<0]=np.nan
	II3 = np.where(np.logical_not(np.isnan((x)))&np.logical_not(np.isnan(y)))
	if len(y[II3])>0:
		x = x[II3]
		y = y[II3]
	#	print(x,y)
		xy = np.matrix(np.vstack((x,y)))
		MinMax_xy = [np.min(xy),np.max(xy)]
		rng =	np.add(MinMax_xy[1],-MinMax_xy[0])
		npt = len(y)
		xstdev = np.std(x,ddof=1)
		ystdev = np.std(y,ddof=1)
		x_mean = np.mean(x)
		y_mean = np.mean(y)
		
		if len(x)>1:
			R,pval = pearsonr(x,y)
			log10_pvalues=np.log10(pval)
		else:
			R = np.nan
			pval = np.nan
			log10_pvalues = np.nan
		
		mean_ary = np.mean(xy,0)
		dif_ary = y - x
		abs_dif_ary = np.absolute(y - x)
		b_mean = np.mean(dif_ary)
		bstdev = np.std(dif_ary,ddof=1)
		ab_mean = np.mean(abs_dif_ary)
		abstdev = np.std(abs_dif_ary,ddof=1)		
		rb = np.divide(dif_ary,mean_ary)
	#	print(rb)
		arb = np.divide(abs_dif_ary,mean_ary)
		rb_mean = np.mean(rb)
		rbstdev = np.std(rb,ddof=1)
		arb_mean = np.mean(arb)
		arbstdev = np.std(arb,ddof=1)
		bias_prctiles = np.zeros((len(prctils)))
		abs_bias_prctiles = np.zeros((len(prctils)))
		relative_bias_prctiles = np.zeros((len(prctils)))
		abs_relative_bias_prctiles = np.zeros((len(prctils)))		
		x_prctiles = np.zeros((len(prctils)))
		y_prctiles = np.zeros((len(prctils)))
		for i1 in range(len(prctils)):
			relative_bias_prctiles[i1] = np.percentile(rb,prctils[i1],axis=1)
			abs_relative_bias_prctiles[i1] = np.percentile(arb,prctils[i1],axis=1)
			bias_prctiles[i1] = np.percentile(dif_ary,prctils[i1],axis=0)
			abs_bias_prctiles[i1] = np.percentile(abs_dif_ary,prctils[i1],axis=0)			
			x_prctiles[i1] = np.percentile(x,prctils[i1],axis=0)
			y_prctiles[i1] = np.percentile(y,prctils[i1],axis=0)	

		nrmsd = np.sqrt(np.sum((y-x)**2)/npt)/rng
		rmsd  = np.sqrt(np.sum((y-x)**2)/npt)
		mad = np.sum(np.absolute(y-x))/npt
		nmad =  np.sum(np.absolute(y-x))/npt/rng
		
		# Create the matrix of weights 
		w1 = lat2W(xy.shape[0], xy.shape[1])
		mi = Moran(xy, w1, two_tailed=True)		
		mi_I = mi.I
		mi_EI = mi.EI
		mi_p_norm = mi.p_norm
		mi_z_norm = mi.z_norm
		mi_p_rand = mi.p_rand
		mi_z_rand = mi.z_rand
	else:
		R = np.nan
		pval = np.nan
		log10_pvalues = np.nan
		relative_bias_prctiles = np.full((len(prctils)),np.nan)
		abs_bias_prctiles = np.full((len(prctils)),np.nan)
		bias_prctiles = np.full((len(prctils)),np.nan)
		abs_relative_bias_prctiles = np.full((len(prctils)),np.nan)
		x_prctiles = np.full((len(prctils)),np.nan)
		y_prctiles = np.full((len(prctils)),np.nan)
		nrmsd = np.nan
		rmsd  = np.nan
		mad = np.nan
		nmad = np.nan
		xstdev = np.nan
		ystdev = np.nan
		x_mean = np.nan
		y_mean = np.nan
		b_mean = np.nan
		bstdev = np.nan
		ab_mean = np.nan
		abstdev = np.nan		
		rb_mean = np.nan
		rbstdev = np.nan
		arb_mean = np.nan
		arbstdev = np.nan
		mi_I  = np.nan
		mi_EI = np.nan
		mi_p_norm = np.nan
		mi_z_norm = np.nan
		mi_p_rand = np.nan
		mi_z_rand = np.nan
		npt = np.nan
	Results = np.hstack((R,log10_pvalues,bias_prctiles,b_mean,bstdev,abs_bias_prctiles,ab_mean,abstdev,
						relative_bias_prctiles,rb_mean,rbstdev,abs_relative_bias_prctiles,
						arb_mean,arbstdev,nmad,mad,nrmsd,rmsd,x_prctiles,x_mean,xstdev,y_prctiles,y_mean,
						ystdev,mi_I,mi_EI,mi_p_norm,mi_z_norm,mi_p_rand,mi_z_rand,npt))
	return Results


def Survey(x,prctils):
	prctiles = np.full((len(prctils),len(x[:,0])),np.nan)
	confint = np.full((2,len(x[:,0])),np.nan)
	npt = np.zeros((1,len(x[:,0])))
	#x[np.isinf(x)]=np.nan
	#x[x==0]=np.nan	
	mn = np.squeeze(np.nanmean(x,1,where=np.logical_not(np.isnan((x)))))
	xstdev = np.squeeze(np.std(x,1,where=np.logical_not(np.isnan((x))),ddof=1))
	for i2 in range(len(x[:,0])):
		#print(i2)
		x1 = x[i2,:]
		fltr = np.where(np.logical_not(np.isnan((x1))))[0]
		npt[0,i2] = len(x1[fltr])

		if npt[0,i2]>1:	
			for i1 in range(len(prctils)):
				prctiles[i1,i2] = np.squeeze(np.percentile(x1[fltr],prctils[i1]))
			#print(mn[i2],xstdev[i2],npt[0,i2])
			confint[:,i2] = st.norm.interval(0.80, loc=mn[i2], scale=st.sem(x1[fltr])) 
	op = np.vstack((prctiles,mn,xstdev,confint,npt))		
	return op