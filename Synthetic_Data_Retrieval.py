import ISARA
import LUT
initialize = LUT.initialize_spheres
run_LUT = LUT.run
import numpy as np
import os
import sys
#from pathos.multiprocessing import ProcessPool
def Run():
    """
    Runs ISARA retrievals on a set of synthetically generated data for testing the sensitivity of ISARA CRI and kappa retrievals.   

    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)    

    Requirements
    ------------ 
    * ``numpy``
    * ``os``
    * ``sys`` 
    """ 
    sys.path.insert(0, os.path.abspath("../"))  

    # Number of cores you want to use
    #number_of_cores = 32    

    # This should be at the start of the code to minimize the fork size
    #pool = ProcessPool(ncpus=number_of_cores)   
    LUT_output_variables = initialize(f'../ISARA_data_files/LUT_data/AerosolLUT_1000_100_0.355_650bins_2325CRI_ln2rKr_Twomey.dat')
  
    def dict_reconfig(
        dictionaryname,
      ):
        OP = dict()
        io = 0
        for key in dictionaryname.item():
            #print(key,io)
            value = dictionaryname.item().get(key)
            OP[key] = value
            #print(key)
        return OP       
    def grabvaluessd(
      dictionaryname,
    ):
      OP = dict()
      io = 0
      dp = None
      for key in dictionaryname:
        if key.startswith("noisy_SD_Bin"):
          #print(key,io)
          value = dictionaryname[key]
          OP[io] = np.squeeze(value.T)
          io += 1
      return OP
    def grab_synthetic_Data(filename):
        data = dict_reconfig(np.load(filename,allow_pickle='TRUE'))
        dry_wvl = data['dry_wavelengths']   
        wet_wvl = data['wet_wavelengths']                       
        noisy_sca = dict()
        noisy_abs = dict()
        noisy_wet_sca = dict()
        #SSA = {}
        Lwvl = len(dry_wvl["sca"])
        for iwvl in range(Lwvl):
            #print(iwvl)
            noisy_sca[f'{dry_wvl["sca"][iwvl]}'] = data[f'noisy_dry_sca_coef_{dry_wvl["sca"][iwvl]}']
            noisy_abs[f'{dry_wvl["abs"][iwvl]}'] = data[f'noisy_dry_abs_coef_{dry_wvl["abs"][iwvl]}']
        Lwvl2 = len(wet_wvl["sca"])
        for iwvl in range(Lwvl2):
            #print(iwvl)
            noisy_wet_sca[f'{wet_wvl["sca"][iwvl]}'] = data[f'noisy_wet_sca_coef_{wet_wvl["sca"][iwvl]}']
        noisy_SD = grabvaluessd(data)
        dpg = data['synthetic_dpg']
        noisy_sd = np.zeros((len(dpg),len(noisy_SD[0])))
        print(len(noisy_SD[0]))
        for i1 in range(len(noisy_SD)):
            noisy_sd[i1,:] = noisy_SD[i1]

         
        #pause()
        return (data, noisy_sd, dpg, noisy_sca, noisy_abs, noisy_wet_sca, dry_wvl, wet_wvl)   
    
    def handle_line(i1,dry_wvl, wet_wvl, sd, dpg, measured_Sc_dry, measured_Abs_dry, measured_Sc_wet, CRI_p, kappa_p, LUT_output_variables):
        # So this code may look a bit funky, but we are doing what is called currying. This is simply the idea of returning a function inside of a function. 
        # It may look weird doing this, but this is actually required so that each worker has the necessary data. What ends up happening is each worker is 
        # passed a full copy of all the data contained within this function, so it has to know what data needs to be copied. Anyhow, the inner `curry` 
        # function is what is actually being called for each iteration of the for loop.
        # You will notice that in the code we are assining the value for this row and they will be merged later
        #def curry(i1):  
        val_wvl = None
        finalout = {}
        meas_data = {}
        #finalout['dry_wvl'] = dry_wvl
        measflg = 0 
        Lwvl = len(dry_wvl["sca"])
        iwvl = 0        
        for kwvl in measured_Abs_dry: 
            finalout[f'dry_meas_abs_coef_{dry_wvl["abs"][iwvl]}_Mm-1'] = measured_Abs_dry[kwvl][i1]
            measflg += 1 
            meas_data[f'dry_meas_abs_coef_{dry_wvl["abs"][iwvl]}_Mm-1'] = finalout[f'dry_meas_abs_coef_{dry_wvl["abs"][iwvl]}_Mm-1']
            iwvl += 1    
        iwvl = 0
        keycheck = None                          
        for kwvl in measured_Sc_dry: 
            finalout[f'dry_meas_sca_coef_{dry_wvl["sca"][iwvl]}_Mm-1'] = measured_Sc_dry[kwvl][i1]  
            measflg += 1
            meas_data[f'dry_meas_sca_coef_{dry_wvl["sca"][iwvl]}_Mm-1'] = finalout[f'dry_meas_sca_coef_{dry_wvl["sca"][iwvl]}_Mm-1']
            if kwvl.__contains__(str(dry_wvl["sca"][1])):
                finalout[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1'] = measured_Sc_wet[kwvl][i1] 
                finalout[f'wet_meas_ext_coef_{dry_wvl["sca"][1]}_Mm-1'] = finalout[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1'] + finalout[f'dry_meas_abs_coef_{dry_wvl["abs"][1]}_Mm-1']
                finalout[f'meas_fRH_{dry_wvl["sca"][1]}_unitless'] = finalout[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1'] / finalout[f'dry_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1']
                meas_data[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1'] = finalout[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1']
            iwvl += 1   
        finalout['attempt_flag_CRI_unitless'] = 0
        finalout['attempt_flag_kappa_unitless'] = 0
        meas_data['dndlogdp_cm-3'] = sd[:,i1]
        #meas_data['dndlogdp_cm-3'][np.isnan(meas_data['dndlogdp_cm-3'])] = 0
        meas_data['dpg_um'] = dpg  
        if measflg == 6:        
            finalout['attempt_flag_CRI_unitless'] = 1
            Results = ISARA.Retr_CRI(dry_wvl, val_wvl, meas_data, CRI_p,LUT_output_variables) 
            if Results["dry_RRI_unitless"] is not None:
                finalout['attempt_flag_CRI_unitless'] = 2
                #print(Results["RRIdry"])
                CRI_dry = np.array([Results["dry_RRI_unitless"],Results["dry_IRI_unitless"]])
                for key in Results:
                    finalout[key] = Results[key]
                finalout['attempt_flag_kappa_unitless'] = 1
                Results = ISARA.Retr_kappa(wet_wvl, val_wvl, meas_data, 80, kappa_p, CRI_dry,LUT_output_variables)
                if Results[f"kappa-{dry_wvl["sca"][1]}_unitless"] is not None:
                    finalout['attempt_flag_kappa_unitless'] = 2
                    for key in Results:
                        finalout[key] = Results[key]   
                    finalout[f'cal_fRH_{dry_wvl["sca"][1]}_unitless'] = finalout[f'wet_cal_sca_coef_{dry_wvl["sca"][1]}_Mm-1'] / finalout[f'dry_cal_sca_coef_{dry_wvl["sca"][1]}_Mm-1']
                else:
                    finalout[f'cal_fRH_{dry_wvl["sca"][1]}_unitless'] = np.nan
                    finalout[f'kappa-{dry_wvl["sca"][1]}_unitless'] = np.nan
                    for i2 in range(len(wet_wvl["sca"])):
                        finalout[f'wet_cal_sca_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan
                        finalout[f'wet_cal_SSA_{wet_wvl["sca"][i2]}_unitless'] = np.nan
                        finalout[f'wet_cal_ext_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan               
            else:
                finalout["dry_RRI_unitless"] = np.nan
                finalout["dry_IRI_unitless"] = np.nan
                for i2 in range(Lwvl):
                    finalout[f'dry_cal_sca_coef_{dry_wvl["sca"][i2]}_Mm-1'] = np.nan
                    finalout[f'dry_cal_abs_coef_{dry_wvl["abs"][i2]}_Mm-1'] = np.nan
                    finalout[f'dry_cal_SSA_{dry_wvl["sca"][i2]}_unitless'] = np.nan
                    finalout[f'dry_cal_SSA_{dry_wvl["abs"][i2]}_unitless'] = np.nan
                    finalout[f'dry_cal_ext_coef_{dry_wvl["sca"][i2]}_Mm-1'] = np.nan
                    finalout[f'dry_cal_ext_coef_{dry_wvl["abs"][i2]}_Mm-1'] = np.nan    
                finalout[f'cal_fRH_550_unitless'] = np.nan
                finalout[f"kappa-{dry_wvl["sca"][1]}_unitless"] = np.nan
                for i2 in range(len(wet_wvl["sca"])):
                    finalout[f'wet_cal_sca_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan
                    finalout[f'wet_cal_SSA_{wet_wvl["sca"][i2]}_unitless'] = np.nan
                    finalout[f'wet_cal_ext_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan   
        else:
            for i2 in range(Lwvl):
                finalout[f'dry_cal_sca_coef_{dry_wvl["sca"][i2]}_Mm-1'] = np.nan
                finalout[f'dry_cal_abs_coef_{dry_wvl["abs"][i2]}_Mm-1'] = np.nan
                finalout[f'dry_cal_SSA_{dry_wvl["sca"][i2]}_unitless'] = np.nan
                finalout[f'dry_cal_SSA_{dry_wvl["abs"][i2]}_unitless'] = np.nan
                finalout[f'dry_cal_ext_coef_{dry_wvl["sca"][i2]}_Mm-1'] = np.nan
                finalout[f'dry_cal_ext_coef_{dry_wvl["abs"][i2]}_Mm-1'] = np.nan
            finalout[f'cal_fRH_550_unitless'] = np.nan
            finalout[f"kappa-{dry_wvl["sca"][1]}_unitless"] = np.nan
            for i2 in range(len(wet_wvl["sca"])):
                finalout[f'wet_cal_sca_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan
                finalout[f'wet_cal_SSA_{wet_wvl["sca"][i2]}_unitless'] = np.nan
                finalout[f'wet_cal_ext_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan                                                                       
        return (finalout)   

        #return curry    

    OP_Dictionary = {}  

    # set desired output wavelengths in micrometer
    #wvl = [0.450, 0.470, 0.532, 0.550, 0.660, 0.700]    
    #wvl = [0.450, 0.465, 0.520, 0.550, 0.640, 0.700] 
    size_equ = 'cs' 

    RRIp = np.arange(1.51,1.55,0.01).reshape(-1)#np.arange(1.5,1.6,0.02).reshape(-1)#np.array([1.53])#np.arange(1.45,2.01,0.01).reshape(-1)
    IRIp = np.arange(0,0.081,0.001).reshape(-1)
    #np.hstack((0,10**(-7),10**(-6),10**(-5),10**(-4),np.arange(0.001,0.101,0.001).reshape(-1),np.arange(0.1,0.96,0.01).reshape(-1)))
    #np.arange(0.0,0.08,0.001).reshape(-1)
    CRI_p = np.empty((len(IRIp)*len(RRIp), 2))
    io = 0
    for i1 in range(len(IRIp)):
        for i2 in range(len(RRIp)):
            CRI_p[io, :] = [RRIp[i2], IRIp[i1]]
            io += 1 

    kappa_p = np.arange(0.0, 1.40, 0.001).reshape(-1)  


    # import the .ict data into a dictonary
    (output_dict, noisy_sd, dpg, noisy_sca, noisy_abs, noisy_wet_sca, dry_wvl, wet_wvl) = grab_synthetic_Data(f'../ISARA_data_files/ACTIVATE/SyntheticData/activate-mrg-activate-large-smps_hu25_Synthetic_Data.npy')
    L1 = len(noisy_sd[0,:])
    # Loop through each of the rows here using multiprocessing. This will split the rows across multiple different cores. Each row will be its own index in `line_data` 
    # with a tuple full of information. So, for instance, line_data[0] will contain (CRI_dry, CalCoef_dry, meas_coef_dry, Kappa, CalCoef_wet, meas_coef_wet, results) 
    # for the first line of data
#    line_data = pool.map(
#        # This is a pain, I know, but all the data has to be cloned and accessible within each worker
#        handle_line(dry_wvl, wet_wvl, noisy_sd, dpg, noisy_sca, noisy_abs, noisy_wet_sca, CRI_p, LUT_output_variables),
#        range(L1),
#    )
    # Now that the data has been fetched, we have to join together all the results into aggregated arrays. The `enumerate` function simply loops through the elements in 
    # the array and attaches the associated array index to it.
    # The general trend for merging the values is pretty simple. If the value is not None, that means that it has a value set because it was reached conditionally. 
    #Therefore, if it does have a value, we will just update that part of the array. Now, I know you're probably thinking "why are we doing all this work again." Well, 
    # true, it is repeated work, but this will allow for much faster times overall (well, that's the hope anyhow).
    # def merge_in(line_val, merged_vals):
    #for i1, line_data in enumerate(line_data):
    for i1 in range(L1):
        results_line = handle_line(i1, dry_wvl, wet_wvl, noisy_sd, dpg, noisy_sca, noisy_abs, noisy_wet_sca, CRI_p, kappa_p, LUT_output_variables)   
        for key2 in results_line:
            if key2 in output_dict:
                output_dict[key2][i1] = results_line[key2]
            else:
                output_dict[key2] = np.full((L1),np.nan)
                output_dict[key2][i1] = results_line[key2]
    print(output_dict[f"kappa-{dry_wvl["sca"][1]}_unitless"].size)       
    output_filename = 'activate_Synthetic_retrievals'
    np.save(f'../ISARA_data_files/ACTIVATE/SyntheticRetrievals/{output_filename}.npy', output_dict)  

    # Close the pool to any new jobs and remove it
    #pool.close()
    #pool.clear()
