import lut
run_LUT = lut.run_simple
import sizedistmerge as sdm#from scipy.interpolate import pchip_interpolate
import numpy as np

def Retr_CRI(wvl_dict, val_wvl, measurement_data,CRI_p,LUT_output_variables):  
    """
    Returns aerosol particle real and imaginary refractive index from three scattering coefficeint measurements, three absorption coefficient measurements, a measured volume concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) mopsmap_wrapper.py must be present in a directory that is in your PATH
        
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

    L1 = len(CRI_p[:,0]) # length of array with all possible cri values
    L2 = len(wvl_dict["sca"]) # number of measured scattering (sca) coefficient channels
    L3 = len(wvl_dict["abs"]) # number of measured absorption (abs) coefficient channels    
    L4 = len(measurement_data['dndlogdp_cm-3'])
    # Prepare output arrays and dictionary
    iri = np.full((L1), np.nan)
    rri = np.full((L1), np.nan)
    Results = dict()  
    Results["dry_RRI_unitless"] = None
    Results["dry_IRI_unitless"] = None
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
        Results[f'dry_cal_sca_coef_{wvl_dict["sca"][i2]}_Mm-1'] = None# prepare final results 
        Results[f'dry_cal_abs_coef_{wvl_dict["abs"][i2]}_Mm-1'] = None# prepare final results 
        ref_scat_coef[i2,:] = np.tile(measurement_data[f'dry_meas_sca_coef_{wvl_dict["sca"][i2]}_Mm-1'], (L1))# Assign values to prepared measured coefficients
        results = run_LUT(wvl_dict["sca"][i2],"basic",CRI_p[:,0],CRI_p[:,1],sd_v_tile,LUT_output_variables) # calculate microphysical properties for a given wavelength
        scat_coef[i2,:] = results['scattering_coefficient']
    for i2 in range(L3):  
        Results[f'dry_cal_sca_coef_{wvl_dict["sca"][i2]}_Mm-1'] = None# prepare final results   
        Results[f'dry_cal_abs_coef_{wvl_dict["abs"][i2]}_Mm-1'] = None# prepare final results 
        ref_abs_coef[i2,:] = np.tile(measurement_data[f'dry_meas_abs_coef_{wvl_dict["abs"][i2]}_Mm-1'], (L1))
        results = run_LUT(wvl_dict["abs"][i2],"basic",CRI_p[:,0],CRI_p[:,1],sd_v_tile,LUT_output_variables)# calculate microphysical properties for a given wavelength
        abs_coef[i2,:] = results['absorption_coefficient']  
    Cdif1 = abs(ref_scat_coef-scat_coef)/ref_scat_coef # calculate absolute relative difference of scattering coefficients in each channel
    Cdif2 = abs(ref_abs_coef-abs_coef)# calculate absolute difference of absoprtion coefficients in each channel 
    Cdif3 = abs(ref_scat_coef/(ref_scat_coef+ref_abs_coef)-scat_coef/(scat_coef+abs_coef))     
    a1 = ((Cdif1)<0.2).astype('int')# check if relative difference in scattering coefficient is within 20% for all and channels that the difference in absorption coefficient is within 1 Mm-1 for all channels
    a2 = ((Cdif2)<1).astype('int')#
    a3 = ((Cdif3)<0.05).astype('int')
    valid_solns = np.where(((np.sum(a1,0)==L2)&(np.sum(a2,0)==L3)&(np.sum(a3,0)==L3)))[0]  
    if valid_solns.size>0:
        results = {}
        # take mean rri and iri of all valid solutions and recalculate aerosol properties with mean cri values.    
        rri = np.mean(CRI_p[valid_solns,0])
        iri = np.mean(CRI_p[valid_solns,1])
        unc_rri = np.std(CRI_p[valid_solns,0])
        unc_iri = np.std(CRI_p[valid_solns,1])
        # same as before, check for to ensure recalculated scattering coefficients are within 20% and absorption coefficients are with 1 Mm-1 when using mean cri
        scat_coef = np.full((L2),np.nan)
        abs_coef = np.full((L3),np.nan)     
        for i2 in range(L2):
          results[f'{wvl_dict["sca"][i2]}'] = run_LUT(wvl_dict["sca"][i2],"basic",CRI_p[valid_solns,0],CRI_p[valid_solns,1],sd_v_tile[valid_solns,:],LUT_output_variables) # calculate microphysical properties for a given cri 
          scat_coef[i2] = np.mean(results[f'{wvl_dict["sca"][i2]}'][f'scattering_coefficient'])    
        for i2 in range(L3):
          results[f'{wvl_dict["abs"][i2]}'] = run_LUT(wvl_dict["abs"][i2],"basic",CRI_p[valid_solns,0],CRI_p[valid_solns,1],sd_v_tile[valid_solns,:],LUT_output_variables) # calculate microphysical properties for a given cri  
          abs_coef[i2] = np.mean(results[f'{wvl_dict["abs"][i2]}'][f'absorption_coefficient'])     
        Cd1 = abs(ref_scat_coef[:,0]-scat_coef)/ref_scat_coef[:,0]
        Cd2 = abs(ref_abs_coef[:,0]-abs_coef)
        Cd3 = abs(ref_scat_coef[:,0]/(ref_scat_coef[:,0]+ref_abs_coef[:,0])-scat_coef/(scat_coef+abs_coef))     
        a1 = ((Cd1)<0.2).astype('int')
        a2 = ((Cd2)<1).astype('int')
        a3 = ((Cd3)<0.05).astype('int')
        valid_solns = (np.sum(a1,0)==L2)&(np.sum(a2,0)==L3)&(np.sum(a3,0)==L3)  # if solution is valid, store dry cri and dry calculated extinction, scattering, and absorption coefficients and SSA in all measured wavelengths
        if valid_solns.size>0:
          Results["dry_RRI_unitless"] = rri
          Results["dry_IRI_unitless"] = iri
          Results["dry_uncert_RRI_unitless"] = unc_rri
          Results["dry_uncert_IRI_unitless"] = unc_iri              
          for i2 in range(L2):
            Results[f'dry_cal_sca_coef_{wvl_dict["sca"][i2]}_Mm-1'] = np.mean(results[f'{wvl_dict["sca"][i2]}'][f'scattering_coefficient'])
            Results[f'dry_cal_ext_coef_{wvl_dict["sca"][i2]}_Mm-1'] = np.mean(results[f'{wvl_dict["sca"][i2]}'][f'extinction_coefficient'])
            Results[f'dry_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = np.mean(results[f'{wvl_dict["sca"][i2]}'][f'ssa'])    
            Results[f'dry_cal_abs_coef_{wvl_dict["abs"][i2]}_Mm-1'] = np.mean(results[f'{wvl_dict["abs"][i2]}'][f'absorption_coefficient']) 
            Results[f'dry_cal_ext_coef_{wvl_dict["abs"][i2]}_Mm-1'] = np.mean(results[f'{wvl_dict["abs"][i2]}'][f'extinction_coefficient'])
            Results[f'dry_cal_SSA_{wvl_dict["abs"][i2]}_unitless'] = np.mean(results[f'{wvl_dict["abs"][i2]}'][f'ssa'])
          if val_wvl is not None: # if validation wavelengths are requested, provide outputs for those wavelengths as well
            wvl2 = None
            for iwvl in range(len(val_wvl)):
              if iwvl == 0:
                wvl2 = val_wvl
              else:
                wvl2 = np.hstack((wvl2,val_wvl[iwvl]))
            for iwvl in range(len(val_wvl)):
                results = run_LUT(wvl2[iwvl],"basic",np.array([rri,rri]),np.array([iri,iri]),np.array([target_sd_n,target_sd_n]),LUT_output_variables)
                Results[f'dry_cal_sca_coef_{val_wvl[iwvl]}_Mm-1'] = np.mean(results[f'scattering_coefficient'])
                Results[f'dry_cal_abs_coef_{val_wvl[iwvl]}_Mm-1'] = np.mean(results[f'absorption_coefficient'])
                Results[f'dry_cal_ext_coef_{val_wvl[iwvl]}_Mm-1'] = np.mean(results[f'extinction_coefficient'])
                Results[f'dry_cal_SSA_{val_wvl[iwvl]}_unitless'] = np.mean(results[f'ssa'])
    return Results # return dictionary (Results) of dry cri and dry calculated extinction, scattering, and absorption coefficients and SSA in all measured and validation wavelengths

def Retr_kappa(wvl_dict,val_wvl,measurement_data,RH,kappa_p,CRI_d,LUT_output_variables):
    """
    Returns aerosol particle hygroscopic growth factor from a humdified scattering coefficeint measurement, dry complex refractive index, and a measured number concentration for an aerosol size distribution. WARNINGS: 1) numpy must be installed to the python environment 2) mopsmap_wrapper.py must be present in a directory that is in your PATH.
        
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

    L1 = len(kappa_p) # length of array with all possible kappa values
    L2 = len(wvl_dict["sca"]) # number of measured scattering (Sc) coefficient channels 
    # collect scattering coefficient channel wavelengths (wvl) into array and sort in ascending order
    wvl = None
    for i1 in range(L2):
        if i1 == 0:
            wvl = np.array([wvl_dict["sca"][i1]])
        else:
            wvl = np.hstack((wvl,np.array([wvl_dict["sca"][i1]])))
        wvl = np.sort(wvl, axis=None)
    #
    # Prepare output dictionary
    Results = dict()
    Results[f'kappa-{wvl_dict["sca"][0]}_unitless'] = None
    for i2 in range(L2):
        Results[f'wet_cal_sca_coef_{wvl_dict["sca"][i2]}_Mm-1'] = None
        Results[f'wet_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = None
        Results[f'wet_cal_ext_coef_{wvl_dict["sca"][i2]}_Mm-1'] = None
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

    scat_coef = np.full((L2,L1),np.nan)# prepare array of calculated scattering coefficients
    ref_scat_coef = np.full((L2,L1),np.nan) # prepare array of measured scattering coefficients
    for i2 in range(L2):
        results  = run_LUT(wvl[i2],"basic",rri_w,iri_w,sd_v_w,LUT_output_variables)# calculate microphysical properties for a given kappa
        scat_coef[i2,:] = results['scattering_coefficient']# Assign values to prepared measured and calculated coefficients
        ref_scat_coef[i2,:] = np.tile(measurement_data[f'wet_meas_sca_coef_{wvl_dict["sca"][i2]}_Mm-1'],(L1))
        Cdif = abs(ref_scat_coef[i2,:]-scat_coef[i2,:])/ref_scat_coef[i2,:]
        # calculate absolute relative difference of scattering coefficients in each channel
        valid_solns = np.where((Cdif<0.01))[0]  # solution is valid if scattering coefficients are within 1%
        if valid_solns.size>0:
            Results[f'kappa-{wvl_dict["sca"][i2]}_unitless'] = np.array(np.mean(kappa_p[valid_solns])) # store retrieved kappa
            gf = np.power((1 + Results[f'kappa-{wvl_dict["sca"][i2]}_unitless']*RH/(100-RH)),1/3) # calculate growth factor given the incrimental kappa value and the measurement relative humidity for each size mode
            dpg_w = np.multiply(gf,dpg) # adjust the size distribution by multplying the growth factor by each dry particle diameter in each size mode
            dpg_edges = sdm.edges_from_mids_geometric(dpg_w)
            target_sd_n = sdm.rebin_dndlog_by_edges_overlap(dpg_edges * 1000, dp_grid_edges * 1000, sd_n)
            target_sd_n[np.isnan(target_sd_n)] = 0
            rri_w = (CRI_d[0]+((gf**3)-1)*RRIw)/(gf**3) # volume weighted humidified rri for each size mode
            iri_w = (CRI_d[1]+((gf**3)-1)*IRIw)/(gf**3) # volume weighted humidified iri for each size mode
            target_sd_v_w = 4*(np.pi/3)*((dp_grid/2)**3)*target_sd_n/np.log(10)       
            #results  = run_LUT(wvl[i2],np.array([rri_w,rri_w]),np.array([iri_w,iri_w]),np.array([target_sd_v_w,target_sd_v_w]))# calculate microphysical properties for a given kappa
            # store calculated scattering and extinction coefficients and SSA for measured and validation wavelengths
            for i2 in range(L2):
              Results[f'wet_cal_sca_coef_{wvl_dict["sca"][i2]}_Mm-1'] = np.mean(results[f'scattering_coefficient'][valid_solns]) 
              Results[f'wet_cal_abs_coef_{wvl_dict["sca"][i2]}_Mm-1'] = np.mean(results[f'absorption_coefficient'][valid_solns])
              Results[f'wet_cal_ext_coef_{wvl_dict["sca"][i2]}_Mm-1'] = np.mean(results[f'extinction_coefficient'][valid_solns])
              Results[f'wet_cal_SSA_{wvl_dict["sca"][i2]}_unitless'] = np.mean(results[f'ssa'][valid_solns])
            if val_wvl is not None:
              wvl2 = None
              for iwvl in range(len(val_wvl)):
                if iwvl == 0:
                  wvl2 = val_wvl
                else:
                  wvl2 = np.hstack((wvl2,val_wvl))
              for iwvl in range(len(val_wvl)):
                results = run_LUT(val_wvl[iwvl],"basic",np.array([rri_w,rri_w]),np.array([iri_w,iri_w]),np.array([target_sd_v_w,target_sd_v_w]),LUT_output_variables)
                Results[f'wet_cal_sca_coef_{val_wvl[iwvl]}_Mm-1'] = np.mean(results[f'scattering_coefficient'])
                Results[f'wet_cal_abs_coef_{val_wvl[iwvl]}_Mm-1'] = np.mean(results[f'absorption_coefficient'])      
                Results[f'wet_cal_ext_coef_{val_wvl[iwvl]}_Mm-1'] = np.mean(results[f'extinction_coefficient'])
                Results[f'wet_cal_SSA_{val_wvl[iwvl]}_unitless'] = np.mean(results[f'ssa'])
        #        
        #stop_indx = 1 # change stop index when first valid solution is reached
    return Results # return dictionary (Results) of kappa and wet calculated extinction, scattering, and absorption coefficients and SSA in all measured and validation wavelengths
    