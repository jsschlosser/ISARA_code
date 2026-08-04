import LUT
run_LUT = LUT.run_simple
from scipy.interpolate import pchip_interpolate
import numpy as np

def Run(dNdlnRpg,dAdlnRpg,dVdlnRpg,rri,iri,rho,LUT,shape_dist,wvl,kys,kys_sizes):
	"""
	Returns aerosol particle real and imaginary refractive index from three scattering coefficeint measurements, three absorption coefficient measurements, a measured volume concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) mopsmap_wrapper.py must be present in a directory that is in your PATH

    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)    

    Requirements
    ------------ 
    * ``scipy``
    * ``numpy``

	:param wvl_dict: Dictionary of wavelengths associated with each of the scattering and absorption measurements
	:type wvl_dict: numpy dictionary
	:param val_wvl: Dictionary of wavelengths associated with validation measurements
	:type val_wvl: numpy dictionary
	:param measurement_data: Dictionary containing measured dry scattering and absorption coefficients in Mm^-1, size resolved volume concentrations in um^3 cm^-3, and geometric mean particle diameters of each size bin in mum ***NOTE: The optical coefficients should be one key per channel (e.g., measurement_data['dry_meas_sca_coef_450_Mm-1'], measurement_data['dry_meas_abs_coef_470_Mm-1'], etc.).***
	:type measurement_data: numpy dictionary	 
	:param CRI_p: 2-D array containing the prescribed RRI and IRI range to be searched
	:type CRI_p: numpy array[float]								 
	:return: Dictionary (Results) with the retrieved complex refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in measured and validation wavelengths
	:rtype: numpy dictionary
	""" 
	output_data = {}
	n_data = len(rho["Nucl"])
	n_wvl = len(wvl)
	n_modes = len(rho)
	RRItot = 0
	IRItot = 0
	output_data = {}
	output_data["amb_N_cm-3"] = np.full((n_modes,n_data),np.nan)# prepare variables for summing 
	output_data["amb_A_um2.cm-3"] = np.full((n_modes,n_data),np.nan)
	output_data["amb_V_um3.cm-3"] = np.full((n_modes,n_data),np.nan)
	output_data["amb_M_g.cm-3"] = np.full((n_modes,n_data),np.nan)
	output_data["amb_r_eff_um"] = np.full((n_modes,n_data),np.nan)
	output_data["amb_v_eff_unitless"] = np.full((n_modes,n_data),np.nan)
	output_data['amb_ext_coef_Mm-1'] = np.full((n_modes,n_wvl,n_data),np.nan)
	output_data['amb_sca_coef_Mm-1'] = np.full((n_modes,n_wvl,n_data),np.nan)
	output_data['amb_ssa_unitless'] = np.full((n_modes,n_wvl,n_data),np.nan)
	output_data['amb_asym_unitless'] = np.full((n_modes,n_wvl,n_data),np.nan)
	output_data['amb_RRI_unitless'] = np.full((n_modes,n_wvl,n_data),np.nan)
	output_data['amb_IRI_unitless'] = np.full((n_modes,n_wvl,n_data),np.nan)
	output_data['amb_back_coef_Mm-1.sr-1'] = np.full((n_modes,n_wvl,n_data),np.nan)
	output_data['amb_lidar_ratio_sr'] = np.full((n_modes,n_wvl,n_data),np.nan)
	output_data['amb_LDR_unitless'] = np.full((n_modes,n_wvl,n_data),np.nan)
	i_mode = 0
	for mode in LUT:
		dndlnr = dNdlnRpg.copy()
		dadlnr = dAdlnRpg.copy()
		dvdlnr = dVdlnRpg.copy()
		rpg = np.array(LUT[mode][0]['radii_grid_bins'])
		out_of_bounds = np.where((rpg < kys_sizes[mode][0]/2) | (rpg > kys_sizes[mode][1]/2))[0] # search for values outside of measured size ranges.
		dndlnr[out_of_bounds,:] = 0 # set concentrations corresponding of non-measured sizes to zero
		dadlnr[out_of_bounds,:] = 0 # ^^
		dvdlnr[out_of_bounds,:] = 0 # ^^
		A = np.trapezoid(dadlnr, x=np.log(rpg),axis=0) # calculate A for this mode (um2.cm-3)
		V = np.trapezoid(dvdlnr,x=np.log(rpg),axis=0)
		N = np.trapezoid(dndlnr, x=np.log(rpg),axis=0) 
		M = V*rho[mode]*10**(-12)
		output_data["amb_N_cm-3"][i_mode,:] = N
		output_data["amb_A_um2.cm-3"][i_mode,:] = A
		output_data["amb_V_um3.cm-3"][i_mode,:] = V
		output_data["amb_M_g.cm-3"][i_mode,:] = M
		condition = A>0
		np.divide(3 * V, A, out = output_data["amb_r_eff_um"][i_mode,:], where = condition)
		val1 = ((rpg[:,None]-output_data["amb_r_eff_um"][i_mode,:])**2)
		val2 = (output_data["amb_r_eff_um"][i_mode,:]**2)
		integrand = val1 * np.pi * val2 * dndlnr
		numerator = np.trapezoid(integrand, x = rpg,axis = 0)
		denominator = (output_data["amb_r_eff_um"][i_mode,:]**2) * np.trapezoid(np.pi * (output_data["amb_r_eff_um"][i_mode,:]**2) * dndlnr, x=rpg,axis=0)
		condition = denominator > 0
		np.squeeze(np.divide(numerator, denominator, out = output_data["amb_v_eff_unitless"][i_mode,:], where = condition)) 
		output_data['amb_RRI_unitless'][i_mode,:,:] = rri[mode]
		output_data['amb_IRI_unitless'][i_mode,:,:] = iri[mode]
		iw = 0
		for w in wvl:
			ishape = 0
			for shp in LUT[mode]:
				Results = run_LUT(w,"extended",rri[mode][iw,:],iri[mode][iw,:],dvdlnr.T,LUT[mode][shp])#,
				arrstack = np.stack((output_data['amb_ext_coef_Mm-1'][i_mode,iw,:],Results['extinction_coefficient'] * shape_dist[mode][ishape]))
				output_data['amb_ext_coef_Mm-1'][i_mode,iw,:] = np.nansum(arrstack,axis=0)
				arrstack = np.stack((output_data['amb_sca_coef_Mm-1'][i_mode,iw,:],shape_dist[mode][ishape] * Results['scattering_coefficient']))
				output_data['amb_sca_coef_Mm-1'][i_mode,iw,:] = np.nansum(arrstack,axis=0)						
				arrstack = np.stack((output_data['amb_asym_unitless'][i_mode,iw,:],Results['asymmetry'] * shape_dist[mode][ishape] * Results['scattering_coefficient']))
				output_data['amb_asym_unitless'][i_mode,iw,:] = np.nansum(arrstack,axis=0)
				arrstack = np.stack((output_data['amb_back_coef_Mm-1.sr-1'][i_mode,iw,:],Results['backscattering_coefficient'] * shape_dist[mode][ishape]))
				output_data['amb_back_coef_Mm-1.sr-1'][i_mode,iw,:] = np.nansum(arrstack,axis=0)
				arrstack = np.stack((output_data['amb_LDR_unitless'][i_mode,iw,:], Results['ldr'] * shape_dist[mode][ishape] * Results['backscattering_coefficient']))
				output_data['amb_LDR_unitless'][i_mode,iw,:] = np.nansum(arrstack,axis=0)
				ishape += 1
			output_data['amb_LDR_unitless'][i_mode,iw,:] /= output_data['amb_back_coef_Mm-1.sr-1'][i_mode,iw,:]
			output_data['amb_asym_unitless'][i_mode,iw,:] /= output_data['amb_sca_coef_Mm-1'][i_mode,iw,:]
			output_data['amb_ssa_unitless'][i_mode,iw,:] = output_data['amb_sca_coef_Mm-1'][i_mode,iw,:] / output_data['amb_ext_coef_Mm-1'][i_mode,iw,:]
			output_data['amb_lidar_ratio_sr'][i_mode,iw,:] = output_data['amb_ext_coef_Mm-1'][i_mode,iw,:] / output_data['amb_back_coef_Mm-1.sr-1'][i_mode,iw,:]
			iw += 1
		i_mode += 1	

	for key2 in output_data:
		if not key2.__contains__("LDR"):
			output_data[key2][output_data[key2] == 0] = np.nan			
	#output_data['angle'] = Results['angle']
	return output_data
		