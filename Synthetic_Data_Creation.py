import APS_rho
import importICARTT
import load_sizebins
import lut
initialize = lut.initialize_spheres
run_LUT = lut.run_simple
import sizedistmerge as sdm
import numpy as np
import os
import sys
import SDFitFigs

def Run():
    """
    Saves a dictionary file of each of the merged data files in source directory that includes ISARA retrievals of CRI and kappa. Dictionary includes metadata for netCDF compliancy.   

    >>> import ISARA_Data_Retrieval
    >>> ISARA_Data_Retrieval.RunISARA()
    activate-mrg-activate-large-smps_hu25_20200214_R0_20230831T150854.ict
    182
    182
    182
    """ 
    sys.path.insert(0, os.path.abspath("../"))  
    def pause():
        programPause = input("Press the <ENTER> key to continue...")
    def grab_data(data,key_name):
        for key in data.keys():
            if key.__contains__(key_name):
                return data[key]
    def dict_reconfig(
    dictionaryname,
    ):
        OP = dict()
        io = 0
        for key in dictionaryname.item():
            value = dictionaryname.item().get(key)
            OP[key] = value
        return OP 
    def add_noise(noiseless_value,expected_er):
        accuracy = expected_er[0]/2
        precision = expected_er[1]/2
        noisy_measurement = noiseless_value*np.random.uniform(1-accuracy, 1+accuracy)+np.random.uniform(-precision, precision)#np.random.normal(i5,accuracy*i5)+np.random.normal(0, precision)#sythetic_measurement[icount]*a#
        return (noisy_measurement)
    dataset_length = int(input('Enter the number of synthetic data points to generate: '))
    DN = input("Enter the reference campaign name (e.g., ACTIVATE): ") 
    numwvl = int(input("Enter number of dry spectral channels measured (e.g., 3): "))
    dry_wvl = {}
    dry_wvl["sca"] = np.full(numwvl,0).astype(int)
    dry_wvl["abs"] = np.full(numwvl,0).astype(int)
    dry_channel_color = np.full(numwvl,np.nan).astype(str) 
    for iwvl in range(numwvl):
        dry_wvl["sca"][iwvl] = input(f"Enter scattering wavelength associated with channel {iwvl+1} in nm (e.g., 450): ")
        dry_wvl["abs"][iwvl] = input(f"Enter absorption wavelength associated with channel {iwvl+1} in nm (e.g., 465): ")
        dry_channel_color[iwvl] = input(f"Enter the wavelength color to represent channel {iwvl+1} (e.g., Blue, Green, or Red): ")
    numwvl = int(input("Enter number of humidified spectral channels measured (e.g., 1): "))
    wet_wvl = {}
    wet_wvl["sca"] = np.full(numwvl,0).astype(int)
    wet_channel_color = np.full(numwvl,np.nan).astype(str) 
    for iwvl in range(numwvl):
        wet_wvl["sca"][iwvl]= input(f"Enter scattering wavelength associated with channel {iwvl+1} in nm (e.g., 450): ")
        wet_channel_color[iwvl] = input(f"Enter the wavelength color to represent channel {iwvl+1} (e.g., Blue, Green, or Red): ")
    camp_name_lower = DN.lower()
    resolution = input("Enter the temporal resolution of interest in seconds (e.g., 30): ") 
    reference_platform = input("Enter the platform of interest (e.g., cirpas-to or MARINA-TOWER): ")
    LUT_output_variables = initialize(f'./LUT_data/AerosolLUT_1000_100_0.355_650bins_2325CRI_ln2rKr_Twomey.dat')
    output_filename_suffix = f'{camp_name_lower}-mrg{resolution}_{reference_platform}'
    SD_Fit_Dictionary = dict_reconfig(np.load(f"../ISARA_data_files/{DN}/FitSDResults/{output_filename_suffix}_SD_Fit_Data.npy",allow_pickle='TRUE'))
    R_values = SD_Fit_Dictionary['R']
    log10_P = SD_Fit_Dictionary['log10_P']
    fit_params_mode0 = SD_Fit_Dictionary['fit_params_0']
    fit_params_mode1 = SD_Fit_Dictionary['fit_params_1']
    bad_fits_filter = np.where((R_values>0.99)&(10**(log10_P)<0.0001))[0]
    print(len(bad_fits_filter))
    SDfitted = SD_Fit_Dictionary['predicted_size_distribution'][:,bad_fits_filter]
    dpg_d = SD_Fit_Dictionary['measured_dpg']
    dpg_d_edges = sdm.edges_from_mids_geometric(dpg_d)
    sd_selector = np.random.choice(SDfitted.shape[1], dataset_length, replace=True)
    R_values = R_values[bad_fits_filter]
    log10_P = log10_P[bad_fits_filter]
    fit_params_mode0 = fit_params_mode0[:,bad_fits_filter]
    fit_params_mode1 = fit_params_mode1[:,bad_fits_filter]
    finalout = {}
    finalout['synthetic_dpg'] = dpg_d
    finalout['dry_wavelengths'] = dry_wvl
    dry_wvls = np.sort(np.hstack((dry_wvl['sca'],dry_wvl['abs'])), axis=None)
    finalout['wet_wavelengths'] = wet_wvl
    finalout['synthetic_kappa'] = np.random.choice(np.arange(1,1400), dataset_length, replace=True)/1000#np.random.randint(1,1400)
    finalout['synthetic_RRI'] = np.full(dataset_length,1.53)
    finalout['synthetic_IRI'] = np.random.choice(np.arange(1,81), dataset_length, replace=True)/1000#np.random.randint(1,800)/10000
    RRI_water = 1.33
    IRI_water = 0 ##D/Ddry = (1+kappa*RH/(100-RH))**(1/3))
    gf = np.power((1+finalout['synthetic_kappa']*80/(100-80)),1/3) # calculate growth factor given the incrimental kappa value and the measurement relative humidity for each size mode
    RRI_w = (finalout['synthetic_RRI']+((gf**3)-1)*RRI_water)/(gf**3) # volume weighted humidified rri for each size mode
    IRI_w = (finalout['synthetic_IRI']+((gf**3)-1)*IRI_water)/(gf**3) # volume weighted humidified iri for each size mode
    dp_grid = np.array(LUT_output_variables['radii_grid_bins'])*2
    grid_length = LUT_output_variables['num_radii_grid_bins']
    dp_grid_edges = sdm.edges_from_mids_geometric(dp_grid) 
    dNdlnr_d_synthetic = np.full((dataset_length,len(dpg_d)),np.nan).astype(float)
    finalout['SD_fit_params_mode0'] = np.full((len(fit_params_mode0[:,0]),dataset_length),np.nan).astype(float)
    finalout['SD_fit_params_mode1'] = np.full((len(fit_params_mode0[:,0]),dataset_length),np.nan).astype(float)
    dVdlnr_d_interp = np.full((dataset_length,grid_length),np.nan).astype(float)
    dVdlnr_w_interp = np.full((dataset_length,grid_length),np.nan).astype(float)
    for i1 in range(dataset_length):
        sd_n = SDfitted[:,sd_selector[i1]]
        finalout['SD_fit_params_mode0'][:,i1] = fit_params_mode0[:,sd_selector[i1]]
        finalout['SD_fit_params_mode1'][:,i1] = fit_params_mode1[:,sd_selector[i1]]
        dNdlnr_d_synthetic[i1,:] = sd_n
        dpg_w = gf[i1] * dpg_d
        dpg_w_edges = sdm.edges_from_mids_geometric(dpg_w)
        target_sd_n_d = sdm.rebin_dndlog_by_edges_overlap(dpg_d_edges * 1000, dp_grid_edges * 1000, sd_n)
        target_sd_n_w = sdm.rebin_dndlog_by_edges_overlap(dpg_w_edges * 1000, dp_grid_edges * 1000, sd_n)
        target_sd_n_d[np.isnan(target_sd_n_d)] = 0
        target_sd_n_w[np.isnan(target_sd_n_w)] = 0
        dVdlnr_d_interp[i1,:] = (4*np.pi / 3)*((dp_grid / 2)**3)*target_sd_n_d / np.log(10)
        dVdlnr_w_interp[i1,:] = (4*np.pi / 3) * ((dp_grid / 2)**3)*target_sd_n_w / np.log(10)
    i_abs = 0
    i_sca = 0
    for iwvl in range(len(dry_wvls)):
        dry_properties = run_LUT(dry_wvls[iwvl],"basic",finalout['synthetic_RRI'],finalout['synthetic_IRI'],dVdlnr_d_interp,LUT_output_variables)
        if dry_wvls[iwvl] in dry_wvl["sca"]:
            finalout[f'dry_sca_coef_{dry_wvl["sca"][i_sca]}'] = dry_properties[f'scattering_coefficient']
            finalout[f'noisy_dry_sca_coef_{dry_wvl["sca"][i_sca]}'] = np.full((dataset_length), np.nan)
            i_sca += 1
        if dry_wvls[iwvl] in dry_wvl["abs"]:
            finalout[f'dry_abs_coef_{dry_wvl["abs"][i_abs]}'] = dry_properties[f'absorption_coefficient']
            finalout[f'noisy_dry_abs_coef_{dry_wvl["abs"][i_abs]}'] = np.full((dataset_length), np.nan)
            i_abs += 1
    for i2 in range(len(wet_wvl["sca"])):
        wet_properties = run_LUT(wet_wvl["sca"][i2],"basic",RRI_w,IRI_w,dVdlnr_w_interp,LUT_output_variables)
        finalout[f'wet_sca_coef_{wet_wvl["sca"][i2]}'] = wet_properties[f'scattering_coefficient']
        finalout[f'noisy_wet_sca_coef_{wet_wvl["sca"][i2]}'] = np.full((dataset_length), np.nan)
    for idp in range(len(dpg_d)):
        finalout[f'noisy_SD_Bin{idp}'] = np.full((dataset_length), np.nan)
    expected_er = dict()
    #expected_er["dndlogdp"] = np.array([0.2/3, np.multiply(1, pow(10, -12))])#
    expected_er["dndlogdp"] = np.array([0.2, 10**(-3)])#
    expected_er["sca"] = np.array([0.2, 2/(np.sqrt(45))])#
    expected_er["abs"] = np.array([0.15, 1/(np.sqrt(45))])#
    #expected_er["RH"] = np.array([0.15, np.multiply(1, pow(10, -6))/(np.sqrt(45))])#
    for i1 in range(dataset_length):
        for iwvl in range(len(dry_wvl["sca"])): 
            finalout[f'noisy_dry_sca_coef_{dry_wvl["sca"][iwvl]}'][i1] = add_noise(finalout[f'dry_sca_coef_{dry_wvl["sca"][iwvl]}'][i1], expected_er['sca'])
        for iwvl in range(len(dry_wvl["abs"])):
            finalout[f'noisy_dry_abs_coef_{dry_wvl["abs"][iwvl]}'][i1] = add_noise(finalout[f'dry_abs_coef_{dry_wvl["abs"][iwvl]}'][i1], expected_er['abs'])
        for iwvl in range(len(wet_wvl["sca"])):
            finalout[f'noisy_wet_sca_coef_{wet_wvl["sca"][iwvl]}'][i1] = add_noise(finalout[f'wet_sca_coef_{wet_wvl["sca"][iwvl]}'][i1] , expected_er['sca'])
        for idp in range(len(dpg_d)):
            noisy_SD = add_noise(dNdlnr_d_synthetic[i1,idp],expected_er['dndlogdp'])
            finalout[f'noisy_SD_Bin{idp}'][i1] = noisy_SD
    output_dictionary = finalout
    output_dictionary['SD_Fit_Dictionary_ACTIVATE'] = SD_Fit_Dictionary
    output_FN = f'{camp_name_lower}-mrg{resolution}_{reference_platform}_Synthetic_Data.npy'
    np.save(f'../ISARA_data_files/{DN}/SyntheticData/{output_FN}', output_dictionary) 
