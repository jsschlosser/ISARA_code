"""
Functions for retrieving aerosol particle real and imaginary refractive index 
and hygroscopicity from absorption and scattering coefficient measurements.

:Authors: Joseph Schlosser
:Revised: 4 Aug 2026
:Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)

Requirements
------------ 
* ``numpy``
* ``sizedistmerge``
"""
import LUT
run_LUT = LUT.run
import sizedistmerge as sdm#from scipy.interpolate import pchip_interpolate
import numpy as np

CRI_SCA_RELATIVE_TOLERANCE = 0.2
CRI_ABS_ABSOLUTE_TOLERANCE_MM1 = 2.0
KAPPA_SCA_RELATIVE_TOLERANCE = 0.01
FIT_SCORE_THRESHOLD = 1.0


def _normalized_fit_scores(residuals, scales):
    """Return per-channel normalized residuals and RMS score per candidate."""
    residuals = np.asarray(residuals, dtype=float)
    scales = np.asarray(scales, dtype=float).reshape(-1)
    if residuals.ndim != 2 or residuals.shape[0] != scales.size:
        raise ValueError("Residuals must have shape (channels, candidates).")
    if residuals.shape[0] == 0:
        raise ValueError("At least one measurement channel is required.")

    normalized_residuals = residuals / np.maximum(
        np.abs(scales)[:, np.newaxis], np.finfo(float).tiny
    )
    scores = np.sqrt(np.mean(np.square(normalized_residuals), axis=0))
    scores[~np.all(np.isfinite(normalized_residuals), axis=0)] = np.inf
    return normalized_residuals, scores


def Retr_CRI(wvl_dict, val_wvl, measurement_data,CRI_p,LUT_output_variables):  
    """
    Returns aerosol particle real and imaginary refractive index from three scattering coefficeint measurements, three absorption coefficient measurements, a measured volume concentration for an aerosol size distribution.
    
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

    CRI_p = np.asarray(CRI_p, dtype=float)
    if CRI_p.ndim != 2 or CRI_p.shape[1] != 2:
        raise ValueError("CRI_p must have shape (candidates, 2).")
    L1 = len(CRI_p[:,0]) # length of array with all possible cri values
    L2 = len(wvl_dict["sca"]) # number of measured scattering (sca) coefficient channels
    L3 = len(wvl_dict["abs"]) # number of measured absorption (abs) coefficient channels    
    # Prepare output arrays and dictionary
    Results = dict()  
    Results["dry_RRI_unitless"] = None
    Results["dry_IRI_unitless"] = None
    Results["dry_fit_score_unitless"] = np.nan
    Results["dry_fit_best_RRI_unitless"] = np.nan
    Results["dry_fit_best_IRI_unitless"] = np.nan
    Results["dry_fit_best_at_grid_edge_unitless"] = np.nan
    for wavelength in wvl_dict["sca"]:
        Results[f'dry_cal_sca_coef_{wavelength}_Mm-1'] = None
        Results[f'dry_cal_ext_coef_{wavelength}_Mm-1'] = None
        Results[f'dry_cal_SSA_{wavelength}_unitless'] = None
        Results[f'dry_fit_sca_residual_{wavelength}_Mm-1'] = np.nan
        Results[f'dry_fit_sca_normalized_residual_{wavelength}_unitless'] = np.nan
    for wavelength in wvl_dict["abs"]:
        Results[f'dry_cal_abs_coef_{wavelength}_Mm-1'] = None
        Results[f'dry_cal_ext_coef_{wavelength}_Mm-1'] = None
        Results[f'dry_cal_SSA_{wavelength}_unitless'] = None
        Results[f'dry_fit_abs_residual_{wavelength}_Mm-1'] = np.nan
        Results[f'dry_fit_abs_normalized_residual_{wavelength}_unitless'] = np.nan
    if L1 == 0 or L2 + L3 == 0:
        return Results
    ref_scat_coef = np.full((L2,L1),np.nan)# prepare arrays of measured scattering and absorption coefficients
    ref_abs_coef = np.full((L3,L1),np.nan) 
    scat_coef = np.full((L2,L1),np.nan)# prepare arrays of calculated scattering and absorption coefficients
    abs_coef = np.full((L3,L1),np.nan)      
    sd_n = measurement_data['dndlogdp_cm-3']
    dpg = measurement_data['dpg_um']
    dpg_edges = sdm.edges_from_mids_geometric(dpg)
    dp_grid = np.array(LUT_output_variables['radii_grid_bins'])*2
    dp_grid_edges = sdm.edges_from_mids_geometric(dp_grid)
    target_sd_n = sdm.rebin_dndlog_by_edges_overlap(dpg_edges*1000, dp_grid_edges*1000, sd_n)
    target_sd_n[np.isnan(target_sd_n)] = 0
    sd_v = (4*np.pi/3)*((dp_grid/2)**3)*target_sd_n/(np.log(10)) # calculate dvdlnrp from dndlogdp and rpg
    sd_v_tile = np.tile(sd_v,(L1,1))# create sd array equal in dimension to possible cri values
    for i2 in range(L2):
        ref_scat_coef[i2,:] = np.tile(measurement_data[f'dry_meas_sca_coef_{wvl_dict["sca"][i2]}_Mm-1'], (L1))# Assign values to prepared measured coefficients
        results = run_LUT(wvl_dict["sca"][i2],"basic",CRI_p[:,0],CRI_p[:,1],sd_v_tile,LUT_output_variables) # calculate microphysical properties for a given wavelength
        scat_coef[i2,:] = results['scattering_coefficient']
    for i2 in range(L3):  
        ref_abs_coef[i2,:] = np.tile(measurement_data[f'dry_meas_abs_coef_{wvl_dict["abs"][i2]}_Mm-1'], (L1))
        results = run_LUT(wvl_dict["abs"][i2],"basic",CRI_p[:,0],CRI_p[:,1],sd_v_tile,LUT_output_variables)# calculate microphysical properties for a given wavelength
        abs_coef[i2,:] = results['absorption_coefficient']  
    residuals = np.vstack((
        scat_coef - ref_scat_coef,
        abs_coef - ref_abs_coef,
    ))
    # Preserve the former per-channel acceptance tolerances as score scales.
    scales = np.concatenate((
        np.maximum(np.abs(ref_scat_coef[:, 0]), np.finfo(float).tiny)
        * CRI_SCA_RELATIVE_TOLERANCE,
        np.full(L3, CRI_ABS_ABSOLUTE_TOLERANCE_MM1),
    ))
    normalized_residuals, scores = _normalized_fit_scores(residuals, scales)
    finite_candidates = np.all(np.isfinite(CRI_p), axis=1)
    scores[~finite_candidates] = np.inf
    if not np.any(np.isfinite(scores)):
        return Results

    best_index = int(np.argmin(scores))
    best_score = float(scores[best_index])
    rri, iri = CRI_p[best_index]
    Results["dry_fit_score_unitless"] = best_score
    Results["dry_fit_best_RRI_unitless"] = float(rri)
    Results["dry_fit_best_IRI_unitless"] = float(iri)
    candidate_bounds = CRI_p[finite_candidates]
    Results["dry_fit_best_at_grid_edge_unitless"] = int(
        np.any(CRI_p[best_index] == np.min(candidate_bounds, axis=0))
        or np.any(CRI_p[best_index] == np.max(candidate_bounds, axis=0))
    )
    for i2, wavelength in enumerate(wvl_dict["sca"]):
        Results[f'dry_fit_sca_residual_{wavelength}_Mm-1'] = float(
            residuals[i2, best_index]
        )
        Results[f'dry_fit_sca_normalized_residual_{wavelength}_unitless'] = float(
            normalized_residuals[i2, best_index]
        )
    for i2, wavelength in enumerate(wvl_dict["abs"]):
        channel_index = L2 + i2
        Results[f'dry_fit_abs_residual_{wavelength}_Mm-1'] = float(
            residuals[channel_index, best_index]
        )
        Results[f'dry_fit_abs_normalized_residual_{wavelength}_unitless'] = float(
            normalized_residuals[channel_index, best_index]
        )

    if best_score > FIT_SCORE_THRESHOLD:
        return Results

    accepted_candidates = scores <= FIT_SCORE_THRESHOLD
    Results["dry_RRI_unitless"] = float(rri)
    Results["dry_IRI_unitless"] = float(iri)
    Results["dry_uncert_RRI_unitless"] = float(
        np.std(CRI_p[accepted_candidates, 0])
    )
    Results["dry_uncert_IRI_unitless"] = float(
        np.std(CRI_p[accepted_candidates, 1])
    )
    for i2 in range(L2):
        results = run_LUT(
            wvl_dict["sca"][i2], "basic", np.array([rri]), np.array([iri]),
            sd_v[np.newaxis, :], LUT_output_variables
        )
        Results[f'dry_cal_sca_coef_{wvl_dict["sca"][i2]}_Mm-1'] = float(
            results['scattering_coefficient'][0]
        )
        Results[f'dry_cal_ext_coef_{wvl_dict["sca"][i2]}_Mm-1'] = float(
            results['extinction_coefficient'][0]
        )
        Results[f'dry_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = float(
            results['ssa'][0]
        )
    for i2 in range(L3):
        results = run_LUT(
            wvl_dict["abs"][i2], "basic", np.array([rri]), np.array([iri]),
            sd_v[np.newaxis, :], LUT_output_variables
        )
        Results[f'dry_cal_abs_coef_{wvl_dict["abs"][i2]}_Mm-1'] = float(
            results['absorption_coefficient'][0]
        )
        Results[f'dry_cal_ext_coef_{wvl_dict["abs"][i2]}_Mm-1'] = float(
            results['extinction_coefficient'][0]
        )
        Results[f'dry_cal_SSA_{wvl_dict["abs"][i2]}_unitless'] = float(
            results['ssa'][0]
        )
    if val_wvl is not None:
        for wavelength in val_wvl:
            results = run_LUT(
                wavelength, "basic", np.array([rri]), np.array([iri]),
                sd_v[np.newaxis, :], LUT_output_variables
            )
            Results[f'dry_cal_sca_coef_{wavelength}_Mm-1'] = float(
                results['scattering_coefficient'][0]
            )
            Results[f'dry_cal_abs_coef_{wavelength}_Mm-1'] = float(
                results['absorption_coefficient'][0]
            )
            Results[f'dry_cal_ext_coef_{wavelength}_Mm-1'] = float(
                results['extinction_coefficient'][0]
            )
            Results[f'dry_cal_SSA_{wavelength}_unitless'] = float(
                results['ssa'][0]
            )
    return Results # return dictionary (Results) of dry cri and dry calculated extinction, scattering, and absorption coefficients and SSA in all measured and validation wavelengths

def Retr_kappa(wvl_dict,val_wvl,measurement_data,RH,kappa_p,CRI_d,LUT_output_variables):
    """
    Returns aerosol particle hygroscopic growth factor from a humdified scattering coefficeint measurement, dry complex refractive index, and a measured number concentration for an aerosol size distribution. 

    :param wvl_dict: Dictionary of wavelengths associated with each of the scattering and absorption measurements
    :type wvl_dict: numpy dictionary
    :param val_wvl: Dictionary of wavelengths associated with validation measurements
    :type val_wvl: numpy dictionary
    :param measurement_data: Dictionary containing measured humidified scattering coefficients in Mm^-1, size resolved number concentrations in cm^-3, geometric mean particle diameters of each size bin in mum ***NOTE: The optical coefficients should be one key per channel (e.g., measurement_data['dry_meas_sca_coef_450_Mm-1'], measurement_data['dry_meas_abs_coef_470_Mm-1'], etc.).***
    :type measurement_data: numpy dictionary
    :param RH: Array containing the percent relative humidity associated with the measured humidified scattering coefficients
    :type RH: int   
    :param kappa_p: Array containing the desired kappa range to be searched.
    :type kappa_p: numpy array    
    :param CRI_d: Array containing the desired dry RRI and IRI.
    :type CRI_d: numpy array                                      
    :return: Real refractive index, imaginary refractive index, calculated scattering and absorption coefficients in native measurements, and calculated single scattering albedo and extinction coefficients in all wavelengths
    :rtype: numpy dictionary
    """ 

    kappa_p = np.asarray(kappa_p, dtype=float)
    if kappa_p.ndim != 1:
        raise ValueError("kappa_p must be one-dimensional.")
    L1 = len(kappa_p) # length of array with all possible kappa values
    L2 = len(wvl_dict["sca"]) # number of measured scattering (Sc) coefficient channels 
    wvl = np.asarray(wvl_dict["sca"])
    # Prepare output dictionary
    Results = dict()
    for wavelength in wvl:
        Results[f'kappa-{wavelength}_unitless'] = None
        Results[f'wet_cal_sca_coef_{wavelength}_Mm-1'] = None
        Results[f'wet_cal_SSA_{wavelength}_unitless'] = None
        Results[f'wet_cal_ext_coef_{wavelength}_Mm-1'] = None
        Results[f'wet_fit_score_{wavelength}_unitless'] = np.nan
        Results[f'wet_fit_best_kappa_{wavelength}_unitless'] = np.nan
        Results[f'wet_fit_sca_residual_{wavelength}_Mm-1'] = np.nan
        Results[f'wet_fit_sca_normalized_residual_{wavelength}_unitless'] = np.nan
    if L1 == 0 or L2 == 0:
        return Results
    #  
    stop_indx = 0 # initate stop index for first valid solution
    RRIw = 1.33 # set rri of water 
    IRIw = 0 # set iri of water 
    #for i1 in range(L1): # loop through each possible kappa value 
    sd_n = measurement_data['dndlogdp_cm-3'] 
    dpg = measurement_data['dpg_um']
    gf = np.power((1+kappa_p*RH/(100-RH)),1/3) # calculate growth factor given the incrimental kappa value and the measurement relative humidity for each size mode
    rri_w = np.empty(L1)
    iri_w = np.empty(L1)
    sd_v_w = np.empty((L1,LUT_output_variables['num_radii_grid_bins']))
    dp_grid = np.array(LUT_output_variables['radii_grid_bins'])*2
    dp_grid_edges = sdm.edges_from_mids_geometric(dp_grid)
    for i_gf in range(L1):
        dpgw = gf[i_gf]*dpg
        rri_w[i_gf] = (CRI_d[0]+((gf[i_gf]**3)-1)*RRIw)/(gf[i_gf]**3) # volume weighted humidified rri for each size mode
        iri_w[i_gf] = (CRI_d[1]+((gf[i_gf]**3)-1)*IRIw)/(gf[i_gf]**3) # volume weighted humidified iri for each size mode
        dpg_edges = sdm.edges_from_mids_geometric(dpgw)
        target_sd_n = sdm.rebin_dndlog_by_edges_overlap(dpg_edges * 1000, dp_grid_edges * 1000, sd_n)
        target_sd_n[np.isnan(target_sd_n)] = 0
        sd_v_w[i_gf,:] = (4*np.pi/3)*((dp_grid/2)**3)*target_sd_n/np.log(10)

    for i2, wavelength in enumerate(wvl):
        results = run_LUT(
            wavelength, "basic", rri_w, iri_w, sd_v_w, LUT_output_variables
        )
        measured = float(
            measurement_data[f'wet_meas_sca_coef_{wavelength}_Mm-1']
        )
        residuals = np.asarray(results['scattering_coefficient']) - measured
        scale = max(
            abs(measured) * KAPPA_SCA_RELATIVE_TOLERANCE,
            np.finfo(float).tiny,
        )
        normalized_residuals, scores = _normalized_fit_scores(
            residuals[np.newaxis, :], np.array([scale])
        )
        scores[~np.isfinite(kappa_p)] = np.inf
        if not np.any(np.isfinite(scores)):
            continue

        best_index = int(np.argmin(scores))
        best_score = float(scores[best_index])
        best_kappa = float(kappa_p[best_index])
        Results[f'wet_fit_score_{wavelength}_unitless'] = best_score
        Results[f'wet_fit_best_kappa_{wavelength}_unitless'] = best_kappa
        Results[f'wet_fit_sca_residual_{wavelength}_Mm-1'] = float(
            residuals[best_index]
        )
        Results[f'wet_fit_sca_normalized_residual_{wavelength}_unitless'] = float(
            normalized_residuals[0, best_index]
        )
        if best_score > FIT_SCORE_THRESHOLD:
            continue

        Results[f'kappa-{wavelength}_unitless'] = best_kappa
        Results[f'wet_cal_sca_coef_{wavelength}_Mm-1'] = float(
            results['scattering_coefficient'][best_index]
        )
        Results[f'wet_cal_abs_coef_{wavelength}_Mm-1'] = float(
            results['absorption_coefficient'][best_index]
        )
        Results[f'wet_cal_ext_coef_{wavelength}_Mm-1'] = float(
            results['extinction_coefficient'][best_index]
        )
        Results[f'wet_cal_SSA_{wavelength}_unitless'] = float(
            results['ssa'][best_index]
        )
        if val_wvl is not None:
            for validation_wavelength in val_wvl:
                results = run_LUT(
                    validation_wavelength,
                    "basic",
                    np.array([rri_w[best_index]]),
                    np.array([iri_w[best_index]]),
                    sd_v_w[best_index][np.newaxis, :],
                    LUT_output_variables,
                )
                Results[
                    f'wet_cal_sca_coef_{validation_wavelength}_Mm-1'
                ] = float(results['scattering_coefficient'][0])
                Results[
                    f'wet_cal_abs_coef_{validation_wavelength}_Mm-1'
                ] = float(results['absorption_coefficient'][0])
                Results[
                    f'wet_cal_ext_coef_{validation_wavelength}_Mm-1'
                ] = float(results['extinction_coefficient'][0])
                Results[
                    f'wet_cal_SSA_{validation_wavelength}_unitless'
                ] = float(results['ssa'][0])
    return Results # return dictionary (Results) of kappa and wet calculated extinction, scattering, and absorption coefficients and SSA in all measured and validation wavelengths
    