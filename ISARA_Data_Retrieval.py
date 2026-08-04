import APS_Rho
import Import_ICARTT
import Load_Size_Dists
import ISARA
Retr_CRI = ISARA.Retr_CRI
Retr_kappa = ISARA.Retr_kappa
import LUT
initialize = LUT.initialize_spheres
import numpy as np
import os
import sys
import struct

def RunISARA():

    """
    After a series of user requested inputs,  retrieves complex refractive index and hygroscopicty from the measured aerosol size distributions and optical coefficients in each file (in .ict format) of the source directory.
        
    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)

    Requirements
    ------------ 
    * ``numpy``
    * ``os``
    * ``sys``
    * ``struct``

    """ 
    sys.path.insert(0, os.path.abspath("../"))  
    def grab_data(data,key_name):
        for key in data.keys():
            if key.__contains__(key_name):
                return data[key]    
    def grab_ICT_Data(filename,modelist,dry_wvl):
        data = importICARTT.imp(filename,2) 
        def grab_keydata(key_starts_with,does_not_contain=None):
            for key in data.keys():
                if does_not_contain is None:
                    if key.startswith(key_starts_with):
                        return data[key], key    
                else:
                    if key.startswith(key_starts_with)&np.logical_not(key.__contains__(does_not_contain)):
                        return data[key], key                             
        Sc = dict()
        Abs = dict()
        #SSA = {}
        Lwvl = len(dry_wvl["sca"])
        for iwvl in range(Lwvl):
            #print(iwvl)
            Scat, ScatKey = grab_keydata(f'Sc{int(dry_wvl["sca"][iwvl])}','amb')
            Sc[ScatKey]=Scat
            Absor, AbsorKey = grab_keydata(f'Abs{int(dry_wvl["abs"][iwvl])}','amb')
            Abs[AbsorKey]=Absor
        RHsc,kynmRH = grab_keydata('RH_Sc')
        RHsc = np.array(RHsc)
        gamma,kynmgamma = grab_keydata('gamma550')
        gamma= np.array(gamma)
        print(RHsc.size,gamma.size)
        time,keynmtime = grab_keydata('Time_Start')
        time= np.array(time)        
        frmttime,kynm = grab_keydata('datetime_Start')
        frmttime= np.array(frmttime) 
        print(len(frmttime))
        date,knmdate = grab_keydata('date')
        sd = {}
        for imode in modelist:
            if imode == "FIMS":
                sd[imode] = np.array([v for k, v in data.items() if k.startswith(f'n_Dp_')])
            else:
                sd[imode] = np.array([v for k, v in data.items() if k.startswith(f'{imode}_')])
        return (data, time, date, sd, Sc, Abs, RHsc, gamma)   

    def handle_line(i1,modelist, sd, dpg, dpu, dpl, full_dp, UBcutoff, LBcutoff, measured_Sc_dry, measured_Abs_dry, RHsc, gamma,dry_wvl, wet_wvl, val_wvl, CRI_p, kappa_p,LUT_output_variables): 
        
        finalout = {}
        meas_data = {}
        #finalout['dry_wvl'] = dry_wvl
        measflg = 0 
        Lwvl = len(dry_wvl["sca"])
        iwvl = 0        
        for kwvl in measured_Abs_dry: 
            finalout[f'dry_meas_abs_coef_{dry_wvl["abs"][iwvl]}_Mm-1'] = measured_Abs_dry[kwvl][i1]
            if (np.logical_not(np.isnan(finalout[f'dry_meas_abs_coef_{dry_wvl["abs"][iwvl]}_Mm-1']))&(finalout[f'dry_meas_abs_coef_{dry_wvl["abs"][iwvl]}_Mm-1']>=0)):
                measflg += 1 
                meas_data[f'dry_meas_abs_coef_{dry_wvl["abs"][iwvl]}_Mm-1'] = finalout[f'dry_meas_abs_coef_{dry_wvl["abs"][iwvl]}_Mm-1']
            iwvl += 1    
        iwvl = 0
        keycheck = None                          
        for kwvl in measured_Sc_dry: 
            if np.isnan(gamma[i1]):
                finalout[f'dry_meas_sca_coef_{dry_wvl["sca"][iwvl]}_Mm-1'] = measured_Sc_dry[kwvl][i1]
            else:
                finalout[f'dry_meas_sca_coef_{dry_wvl["sca"][iwvl]}_Mm-1'] = measured_Sc_dry[kwvl][i1]/(np.exp(gamma[i1]*np.log((100)/(100-RHsc[i1]))))#scat_calc=scat_rh=measured(e^(GAMMA*ln((100-calcRH)/(100-measRH))))#f
            keycheck = kwvl     
            if (np.logical_not(np.isnan(finalout[f'dry_meas_sca_coef_{dry_wvl["sca"][iwvl]}_Mm-1']))&(finalout[f'dry_meas_sca_coef_{dry_wvl["sca"][iwvl]}_Mm-1']>1)):
                measflg += 1
                meas_data[f'dry_meas_sca_coef_{dry_wvl["sca"][iwvl]}_Mm-1'] = finalout[f'dry_meas_sca_coef_{dry_wvl["sca"][iwvl]}_Mm-1']
            if kwvl.__contains__(str(dry_wvl["sca"][1])):
                finalout[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1'] = measured_Sc_dry[kwvl][i1]/(np.exp(gamma[i1]*np.log((100-80)/(100-RHsc[i1]))))
                finalout[f'wet_meas_ext_coef_{dry_wvl["sca"][1]}_Mm-1'] = finalout[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1']+finalout[f'dry_meas_abs_coef_{dry_wvl["abs"][1]}_Mm-1']
                finalout[f'meas_fRH_{dry_wvl["sca"][1]}_unitless'] = finalout[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1']/finalout[f'dry_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1']
                meas_data[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1'] = finalout[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1']
            iwvl += 1     
        dndlogdp = {}
        for imode in sd:
            dndlogdp[imode] = sd[imode][:, i1]
        if "APS" in modelist[:]:
            output_dictionary_1 = APS_rho.Align(dpg["UHSAS"],dndlogdp["UHSAS"],dpg["APS"],dndlogdp["APS"])
            rho_dry = output_dictionary_1["rho"]
            peak = output_dictionary_1["peak"]
        else:
            rho_dry = 1
            peak = np.nan
        finalout['dry_rho_g.cm-3'] = rho_dry
        finalout['peak_diameter_um'] = peak
        finalout['attempt_flag_CRI_unitless'] = 0
        finalout['attempt_flag_kappa_unitless'] = 0
        dpflg = 0
        icount = 0
        Dpg = {}
        Dpu = {}
        Dpl = {}  
        Dndlogdp = {}
        fullsd = None
        fulldpg = None
        fulldpu = None
        fulldpl = None
        for imode in sd:
            icount += 1
            if len(dpg[imode]) > 0:
                if imode == "APS":
                    a = np.divide(dpl[imode],np.sqrt(rho_dry))
                    b = np.divide(dpu[imode],np.sqrt(rho_dry))
                    modeflg = np.where(np.logical_not(np.isnan(dndlogdp[imode]))&(a>=LBcutoff[imode])&(b<=UBcutoff[imode]))[0] 
                else:
                    modeflg = np.where(np.logical_not(np.isnan(dndlogdp[imode]))&(dpl[imode]>=LBcutoff[imode])&(dpu[imode]<=UBcutoff[imode]))[0]    
                Dndlogdp[imode] = dndlogdp[imode][modeflg]
                dpflg += 1
                if imode == "APS":
                    Dpg[imode] = np.divide(dpg[imode],np.sqrt(rho_dry))[modeflg]
                    Dpu[imode] = np.divide(dpu[imode],np.sqrt(rho_dry))[modeflg]
                    Dpl[imode] = np.divide(dpl[imode],np.sqrt(rho_dry))[modeflg]
                else:
                    Dpg[imode] = dpg[imode][modeflg]
                    Dpu[imode] = dpu[imode][modeflg]
                    Dpl[imode] = dpl[imode][modeflg]
                if dpflg == 1:
                    fullsd = Dndlogdp[imode]
                    fulldpg = Dpg[imode]
                    fulldpu = Dpu[imode]
                    fulldpl = Dpl[imode]
                else:
                    fullsd = np.hstack((fullsd,Dndlogdp[imode]))
                    fulldpg = np.hstack((fulldpg,Dpg[imode]))
                    fulldpu = np.hstack((fulldpu,Dpu[imode]))
                    fulldpl = np.hstack((fulldpl,Dpl[imode])) 
        if (keycheck.__contains__('submicron')):
            submicronfilter = np.where(fulldpg<=1)[0]
            fullsd= fullsd[submicronfilter]
            fulldpg = fulldpg[submicronfilter]                    
        if (dpflg==icount) & (measflg == 6) & len(fullsd[fullsd>6]):        
            full_sd = np.full(len(full_dp["dpg"]),np.nan)
            full_dpl= np.full(len(full_dp["dpg"]),np.nan)
            full_dpg = np.full(len(full_dp["dpg"]),np.nan)
            full_dpu = np.full(len(full_dp["dpg"]),np.nan)          
            for idpg in range(len(full_dp["dpg"])):
                fulldpflg = np.where((fulldpg>=full_dp["dpl"][idpg])&(fulldpg<=full_dp["dpu"][idpg]))[0]
                if len(fulldpflg)>0:
                    full_sd[idpg] = fullsd[fulldpflg]
                    full_dpl[idpg] = fulldpl[fulldpflg]
                    full_dpg[idpg] = fulldpg[fulldpflg]
                    full_dpu[idpg] = fulldpu[fulldpflg]
            for idpg in range(len(full_dp["dpg"])):
                finalout[f'dndlogdp_bin{idpg}_cm-3'] = full_sd[idpg]       
                finalout[f'dpl_bin{idpg}_um'] = full_dpl[idpg]
                finalout[f'dpg_bin{idpg}_um'] = full_dpg[idpg]
                finalout[f'dpu_bin{idpg}_um'] = full_dpu[idpg]        

            finalout['attempt_flag_CRI_unitless'] = 1
            meas_data['dndlogdp_cm-3'] = fullsd
            meas_data['dpg_um'] = fulldpg  
            finalout['attempt_flag_CRI_unitless'] = 1    
            Results = Retr_CRI(dry_wvl, val_wvl, meas_data, CRI_p,LUT_output_variables)    
            if Results["dry_RRI_unitless"] is not None:
                finalout['attempt_flag_CRI_unitless'] = 2
                #print(Results["RRIdry"])
                CRI_dry = np.array([Results["dry_RRI_unitless"],Results["dry_IRI_unitless"]])
                for key in Results:
                    finalout[key] = Results[key]
                if np.logical_not(np.isnan(finalout[f'wet_meas_sca_coef_{dry_wvl["sca"][1]}_Mm-1'])):
                    finalout['attempt_flag_kappa_unitless'] = 1
                    Results = Retr_kappa(wet_wvl, val_wvl, meas_data, 80, kappa_p, CRI_dry,LUT_output_variables)
                    if Results[f'kappa-{dry_wvl["sca"][1]}_unitless'] is not None:
                        finalout['attempt_flag_kappa_unitless'] = 2
                        for key in Results:
                            finalout[key] = Results[key]   
                        finalout[f'cal_fRH_{dry_wvl["sca"][1]}_unitless'] = finalout[f'wet_cal_sca_coef_{dry_wvl["sca"][1]}_Mm-1']/finalout[f'dry_cal_sca_coef_{dry_wvl["sca"][1]}_Mm-1']
                    else:
                        finalout[f'cal_fRH_{dry_wvl["sca"][1]}_unitless'] = np.nan
                        finalout[f'kappa-{dry_wvl["sca"][1]}_unitless'] = np.nan
                        for i2 in range(len(wet_wvl["sca"])):
                            finalout[f'wet_cal_sca_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan
                            finalout[f'wet_cal_SSA_{wet_wvl["sca"][i2]}_unitless'] = np.nan
                            finalout[f'wet_cal_ext_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan
                        if val_wvl is not None:
                            for i2 in range(len(val_wvl)):
                                finalout[f'wet_cal_sca_coef_{val_wvl[i2]}_Mm-1'] = np.nan
                                finalout[f'wet_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                                finalout[f'wet_cal_ext_coef_{val_wvl[i2]}_Mm-1'] = np.nan                                     
                else:
                    for i2 in range(Lwvl):
                        finalout[f'dry_cal_sca_coef_{dry_wvl["sca"][i2]}_Mm-1'] = np.nan
                        finalout[f'dry_cal_abs_coef_{dry_wvl["abs"][i2]}_Mm-1'] = np.nan
                        finalout[f'dry_cal_SSA_{dry_wvl["sca"][i2]}_unitless'] = np.nan
                        finalout[f'dry_cal_SSA_{dry_wvl["abs"][i2]}_unitless'] = np.nan
                        finalout[f'dry_cal_ext_coef_{dry_wvl["sca"][i2]}_Mm-1'] = np.nan
                        finalout[f'dry_cal_ext_coef_{dry_wvl["abs"][i2]}_Mm-1'] = np.nan
                    if val_wvl is not None:
                        for i2 in range(len(val_wvl)):
                            finalout[f'dry_cal_sca_coef_{val_wvl[i2]}_Mm-1'] = np.nan
                            finalout[f'dry_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                            finalout[f'dry_cal_ext_coef_{val_wvl[i2]}_Mm-1'] = np.nan                       
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
                finalout[f'kappa-{dry_wvl["sca"][1]}_unitless'] = np.nan
                for i2 in range(len(wet_wvl["sca"])):
                    finalout[f'wet_cal_sca_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan
                    finalout[f'wet_cal_SSA_{wet_wvl["sca"][i2]}_unitless'] = np.nan
                    finalout[f'wet_cal_ext_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan   
                if val_wvl is not None:
                    for i2 in range(len(val_wvl)):
                        finalout[f'dry_cal_sca_coef_{val_wvl[i2]}_Mm-1'] = np.nan
                        finalout[f'dry_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                        finalout[f'dry_cal_ext_coef_{val_wvl[i2]}_Mm-1'] = np.nan    
                        finalout[f'wet_cal_sca_coef_{val_wvl[i2]}_Mm-1'] = np.nan
                        finalout[f'wet_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                        finalout[f'wet_cal_ext_coef_{val_wvl[i2]}_Mm-1'] = np.nan                                                           
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
                    if val_wvl is not None:
                        for i2 in range(len(val_wvl)):
                            finalout[f'dry_cal_sca_coef_{val_wvl[i2]}_Mm-1'] = np.nan
                            finalout[f'dry_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                            finalout[f'dry_cal_ext_coef_{val_wvl[i2]}_Mm-1'] = np.nan                       
        else:
            for idpg in range(len(full_dp["dpg"])):
                finalout[f'dndlogdp_bin{idpg}_cm-3'] = np.nan       
                finalout[f'dpl_bin{idpg}_um'] = np.nan
                finalout[f'dpg_bin{idpg}_um'] = np.nan
                finalout[f'dpu_bin{idpg}_um'] = np.nan        
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
            finalout[f'kappa-{dry_wvl["sca"][1]}_unitless'] = np.nan
            for i2 in range(len(wet_wvl["sca"])):
                finalout[f'wet_cal_sca_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan
                finalout[f'wet_cal_SSA_{wet_wvl["sca"][i2]}_unitless'] = np.nan
                finalout[f'wet_cal_ext_coef_{wet_wvl["sca"][i2]}_Mm-1'] = np.nan   
            if val_wvl is not None:
                for i2 in range(len(val_wvl)):
                    finalout[f'dry_cal_sca_coef_{val_wvl[i2]}_Mm-1'] = np.nan
                    finalout[f'dry_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                    finalout[f'dry_cal_ext_coef_{val_wvl[i2]}_Mm-1'] = np.nan    
                    finalout[f'wet_cal_sca_coef_{val_wvl[i2]}_Mm-1'] = np.nan
                    finalout[f'wet_cal_SSA_{val_wvl[i2]}_unitless'] = np.nan
                    finalout[f'wet_cal_ext_coef_{val_wvl[i2]}_Mm-1'] = np.nan                                                           
        return (finalout)   

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

    #rho_dry = 2.63
    rho_wet = 1.00  

    DN = input("Enter the campaign name (e.g., ACTIVATE): ")   
    #dryorSP = input("Is the dry RH specified? Enter yes or no: ")
    nummodes = int(input("Enter number of size distributions measured: "))
    modelist = np.empty(nummodes).astype(str)  
    UBcutoff = {}    
    LBcutoff = {}   
    dpg = {}
    dpu = {}
    dpl = {}
    full_dp = {}
    full_dp["dpg"] = None
    full_dp["dpu"] = None
    full_dp["dpl"] = None
    maxdpglength = 0
    for i1 in range(nummodes):
        keyname = input(f"Enter the instrument name for mode {i1+1} data (e.g., LAS): ")
        modelist[i1] = keyname
        ifn = [f for f in os.listdir(f'../ISARA_data_files/{DN}/SDBinInfo/') if f.__contains__(keyname)]
        dpData = load_sizebins.Load(f'../ISARA_data_files/{DN}/SDBinInfo/{ifn[0]}')
        #print(dpData)
        dpg[keyname] = grab_data(dpData,"Mid Points")*pow(10,-3) 
        dpu[keyname] = grab_data(dpData,"Upper Bounds")*pow(10,-3) 
        dpl[keyname] = grab_data(dpData,"Lower Bounds")*pow(10,-3) 
        UBcutoff[keyname] = float(input(f"Enter the upper bound of particle sizes\nfor {keyname} data in nm (e.g., 125): "))*pow(10,-3)
        LBcutoff[keyname] = float(input(f"Enter the lower bound of particle sizes\nfor {keyname} data in nm (e.g., 10): "))*pow(10,-3)
        dpcutoffflg = np.where((dpl[keyname]>=LBcutoff[keyname])&(dpu[keyname]<=UBcutoff[keyname]))[0]
        maxdpglength += len(dpcutoffflg)
        if i1 == 0:
            full_dp["dpg"] = dpg[keyname][dpcutoffflg]
            full_dp["dpu"] = dpu[keyname][dpcutoffflg]
            full_dp["dpl"] = dpl[keyname][dpcutoffflg]
        else:
            full_dp["dpg"] = np.hstack((full_dp["dpg"],dpg[keyname][dpcutoffflg]))
            full_dp["dpu"] = np.hstack((full_dp["dpu"],dpu[keyname][dpcutoffflg]))
            full_dp["dpl"] = np.hstack((full_dp["dpl"],dpl[keyname][dpcutoffflg]))
    numwvl = int(input("Enter number of dry spectral channels measured (e.g., 3): "))
    dry_wvl = {}
    dry_wvl["sca"] = np.full(numwvl,-1).astype(int)
    dry_wvl["abs"] = np.full(numwvl,-1).astype(int)
    dry_channel_color = np.full(numwvl,np.nan).astype(str) 
    for iwvl in range(numwvl):
        dry_wvl["sca"][iwvl] = int(input(f"Enter scattering wavelength associated with channel {iwvl+1} in nm (e.g., 450): "))
        dry_wvl["abs"][iwvl] = int(input(f"Enter absorption wavelength associated with channel {iwvl+1} in nm (e.g., 465): "))
        dry_channel_color[iwvl] = input(f"Enter the wavelength color to represent channel {iwvl+1} (e.g., Blue, Green, or Red): ")
    numwvl = int(input("Enter number of humidified spectral channels measured (e.g., 1): "))
    wet_wvl = {}
    wet_wvl["sca"] = np.full(numwvl,-1).astype(int)
    wet_channel_color = np.full(numwvl,np.nan).astype(str) 
    for iwvl in range(numwvl):
        wet_wvl["sca"][iwvl]  = int(input(f"Enter scattering wavelength associated with channel {iwvl+1} in nm (e.g., 450): "))
        wet_channel_color[iwvl] = input(f"Enter the wavelength color to represent channel {iwvl+1} (e.g., Blue, Green, or Red): ")
    addwvl = input(f"Are there any additional wavelengths needed? (yes or no): ")
    if addwvl == "yes":
        valwvl = input(f"Enter the additional wavelength channels speparated\nby a comma and a space (e.g., 370, 530, 1060): ")
        val_wvl =  np.array(valwvl.split(", ")).astype(int)
        valcolor = input(f"Enter the additional wavelength channels colors speparated\nby a comma and a space (e.g., Blue, Green, Red): ")
        val_channel_color =  np.array(valcolor.split(", ")).astype(str)
    else:
        val_wvl = None
    data_directory = input("Enter the name of the directory that contains\nin-situ measurements (e.g., InsituData): ")
    IFN = [f for f in os.listdir(f'../ISARA_data_files/{DN}/{data_directory}/') if f.endswith('.ict')]
    desired_LUT = input("Enter the filename of the desired look-up table\n(e.g., AerosolLUT_1000_100_0.355_650bins_2325CRI_ln2rKr_Twomey.dat): ")
    LUT_output_variables = initialize(f'./LUT_data/{desired_LUT}')

    for input_filename in IFN:
        print(input_filename)
        # import the .ict data into a dictonary
        (output_dict, time, date, sd, Sc, Abs, RHsc, gamma)  = grab_ICT_Data(f'../ISARA_data_files/{DN}/{data_directory}/{input_filename}', modelist, dry_wvl)
        output_dict['SourceFlag'] = {}
        output_dict['Dims'] = {}
        for key in output_dict['VariableAttributes'].keys():
            #if np.logical_not(isinstance(output_dict[key],dict)):
            output_dict['SourceFlag'][key] = 'source'
            output_dict['Dims'][key] = 'time'

        if ((gamma.size > 1)&(len(Sc.keys()) > 1)):
            L1 = gamma.size
            output_dict['full_dp'] = full_dp
            output_dict["dpg"] = dpg
            output_dict["dpu"] = dpu
            output_dict["dpl"] = dpl
            output_dict["UBcutoff"] = UBcutoff
            output_dict["LBcutoff"] = LBcutoff 
            output_dict['VariableAttributes']["dry_RRI_unitless"] = {}
            output_dict['VariableAttributes']["dry_RRI_unitless"]['short_name'] = 'dry_RRI'
            output_dict['VariableAttributes']["dry_RRI_unitless"]['units'] = '1'
            output_dict['VariableAttributes']["dry_RRI_unitless"]['long_name'] = 'Real refractive index of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
            output_dict['VariableAttributes']["dry_RRI_unitless"]['ACVSNC_standard_name'] = 'AerOpt_m_InSitu_BluetoRed_RHd_Bulk_STP'
            output_dict['VariableAttributes']["dry_IRI_unitless"] = {}
            output_dict['VariableAttributes']["dry_IRI_unitless"]['short_name'] = 'dry_IRI'
            output_dict['VariableAttributes']["dry_IRI_unitless"]['units'] = '1'
            output_dict['VariableAttributes']["dry_IRI_unitless"]['long_name'] = 'Imaginary refractive index of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
            output_dict['VariableAttributes']["dry_IRI_unitless"]['ACVSNC_standard_name'] = 'AerOpt_k_InSitu_BluetoRed_RHd_Bulk_STP'
            output_dict['VariableAttributes']['dry_rho_g.cm-3'] = {}
            output_dict['VariableAttributes']["dry_rho_g.cm-3"]['short_name'] = 'dry_rho'
            output_dict['VariableAttributes']["dry_rho_g.cm-3"]['units'] = 'g.cm-3'
            output_dict['VariableAttributes']["dry_rho_g.cm-3"]['long_name'] = 'Effective particle density of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
            output_dict['VariableAttributes']["dry_rho_g.cm-3"]['ACVSNC_standard_name'] = 'none'    
            output_dict['VariableAttributes']['peak_diameter_um'] = {}
            output_dict['VariableAttributes']["peak_diameter_um"]['short_name'] = 'peak_diameter'
            output_dict['VariableAttributes']["peak_diameter_um"]['units'] = 'um'
            output_dict['VariableAttributes']["peak_diameter_um"]['long_name'] = 'Peak dry diameter of APS size distribution.'
            output_dict['VariableAttributes']["peak_diameter_um"]['ACVSNC_standard_name'] = 'none'   
            output_dict['VariableAttributes']['attempt_flag_CRI_unitless'] = {}
            output_dict['VariableAttributes']["attempt_flag_CRI_unitless"]['short_name'] = 'attempt_flag_CRI'
            output_dict['VariableAttributes']["attempt_flag_CRI_unitless"]['long_name'] = 'Flags points where all measurements required for ISARA CRI retrieval and whether or not CRI was successfully retrieved.'
            output_dict['VariableAttributes']["attempt_flag_CRI_unitless"]['flag_values'] = '0 1 2'   
            output_dict['VariableAttributes']["attempt_flag_CRI_unitless"]['flag_meanings'] = 'no_attempt attempt success' 
            output_dict['VariableAttributes']['attempt_flag_kappa_unitless'] = {}
            output_dict['VariableAttributes']["attempt_flag_kappa_unitless"]['short_name'] = 'attempt_flag_kappa'
            output_dict['VariableAttributes']["attempt_flag_kappa_unitless"]['long_name'] = 'Flags points where all measurements required for ISARA CRI and kappa retrieval and whether or not kappa was successfully retrieved.'
            output_dict['VariableAttributes']["attempt_flag_kappa_unitless"]['flag_values'] = '0 1 2'   
            output_dict['VariableAttributes']["attempt_flag_kappa_unitless"]['flag_meanings'] = 'no_attempt attempt success' 
            output_dict['VariableAttributes']["cal_fRH_550_unitless"] = {}    
            output_dict['VariableAttributes']["cal_fRH_550_unitless"]['short_name'] = 'cal_fRH'
            output_dict['VariableAttributes']["cal_fRH_550_unitless"]['units'] = '1'
            output_dict['VariableAttributes']["cal_fRH_550_unitless"]['long_name'] = 'Optical hygrsocopic growth factor at 550 nm of BULK particles derived from ISARA.'
            output_dict['VariableAttributes']["cal_fRH_550_unitless"]['ACVSNC_standard_name'] = 'AerOpt_fRHScat_InSitu_Green_RHd_Bulk_None'
            output_dict['VariableAttributes']["meas_fRH_550_unitless"] = {}    
            output_dict['VariableAttributes']["meas_fRH_550_unitless"]['short_name'] = 'meas_fRH'
            output_dict['VariableAttributes']["meas_fRH_550_unitless"]['units'] = '1'
            output_dict['VariableAttributes']["meas_fRH_550_unitless"]['long_name'] = 'Optical hygrsocopic growth factor at 550 nm of BULK particles derived from gamma measurement.'
            output_dict['VariableAttributes']["meas_fRH_550_unitless"]['ACVSNC_standard_name'] = 'AerOpt_fRHScat_InSitu_Green_RHd_Bulk_None'
            output_dict['VariableAttributes']["kappa_unitless"] = {}
            output_dict['VariableAttributes']["kappa_unitless"]['short_name'] = 'kappa'
            output_dict['VariableAttributes']["kappa_unitless"]['units'] = '1'
            output_dict['VariableAttributes']["kappa_unitless"]['long_name'] = 'Hygroscopicity of BULK particles derived from ISARA.'
            output_dict['VariableAttributes']["kappa_unitless"]['ACVSNC_standard_name'] = 'AerMP_gRH_InSitu_None_Optical_Bulk_None' 
            for i2 in range(len(dry_wvl["sca"])):
                sc_wvl = dry_wvl["sca"][i2]
                abs_wvl = dry_wvl["abs"][i2]
                color_dry_wvl = dry_channel_color[i2]
                output_dict['VariableAttributes'][f'dry_meas_sca_coef_{sc_wvl}_Mm-1'] = {}
                output_dict['VariableAttributes'][f'dry_meas_sca_coef_{sc_wvl}_Mm-1']['short_name'] = f'dry_meas_sca_coef_{sc_wvl}'
                output_dict['VariableAttributes'][f'dry_meas_sca_coef_{sc_wvl}_Mm-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_meas_sca_coef_{sc_wvl}_Mm-1']['long_name'] = f'Scattering coefficient at {sc_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from gamma and scattering measurement at specified relative humidity.'
                output_dict['VariableAttributes'][f'dry_meas_sca_coef_{sc_wvl}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_dry_wvl}_RHd_Bulk_STP'
                output_dict['VariableAttributes'][f'dry_meas_abs_coef_{abs_wvl}_Mm-1'] = {}
                output_dict['VariableAttributes'][f'dry_meas_abs_coef_{abs_wvl}_Mm-1']['short_name'] = f'dry_meas_abs_coef_{abs_wvl}'
                output_dict['VariableAttributes'][f'dry_meas_abs_coef_{abs_wvl}_Mm-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_meas_abs_coef_{abs_wvl}_Mm-1']['long_name'] = f'Absorption coefficient at {abs_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from absorption measurement.'
                output_dict['VariableAttributes'][f'dry_meas_abs_coef_{abs_wvl}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Absorption_InSitu_{color_dry_wvl}_RHd_Bulk_STP'
                output_dict['VariableAttributes'][f'dry_cal_sca_coef_{sc_wvl}_Mm-1'] = {}
                output_dict['VariableAttributes'][f'dry_cal_sca_coef_{sc_wvl}_Mm-1']['short_name'] = f'dry_cal_sca_coef_{sc_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_sca_coef_{sc_wvl}_Mm-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_cal_sca_coef_{sc_wvl}_Mm-1']['long_name'] = f'Scattering coefficient at {sc_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_sca_coef_{sc_wvl}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_dry_wvl}_RHd_Bulk_STP'
                output_dict['VariableAttributes'][f'dry_cal_abs_coef_{abs_wvl}_Mm-1'] = {}
                output_dict['VariableAttributes'][f'dry_cal_abs_coef_{abs_wvl}_Mm-1']['short_name'] = f'dry_cal_abs_coef_{abs_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_abs_coef_{abs_wvl}_Mm-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_cal_abs_coef_{abs_wvl}_Mm-1']['long_name'] = f'Absorption coefficient at {abs_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_abs_coef_{abs_wvl}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Absorption_InSitu_{color_dry_wvl}_RHd_Bulk_STP'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{sc_wvl}_unitless'] = {}
                output_dict['VariableAttributes'][f'dry_cal_SSA_{sc_wvl}_unitless']['short_name'] = f'dry_cal_SSA_{sc_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{sc_wvl}_unitless']['units'] = '1'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{sc_wvl}_unitless']['long_name'] = f'Single scattering albedo at {sc_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{sc_wvl}_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_{color_dry_wvl}_RHd_Bulk_None'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{abs_wvl}_unitless'] = {}
                output_dict['VariableAttributes'][f'dry_cal_SSA_{abs_wvl}_unitless']['short_name'] = f'dry_cal_SSA_{abs_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{abs_wvl}_unitless']['units'] = '1'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{abs_wvl}_unitless']['long_name'] = f'Single scattering albedo at {abs_wvl} nm of BULK particles at DRY relative humidity of 20% derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_SSA_{abs_wvl}_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_{color_dry_wvl}_RHd_Bulk_None'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{sc_wvl}_Mm-1'] = {}
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{sc_wvl}_Mm-1']['short_name'] = f'dry_cal_ext_coef_{sc_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{sc_wvl}_Mm-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{sc_wvl}_Mm-1']['long_name'] = f'Extinction coefficient at {sc_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{sc_wvl}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_dry_wvl}_RHd_Bulk_STP'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{abs_wvl}_Mm-1'] = {}
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{abs_wvl}_Mm-1']['short_name'] = f'dry_cal_ext_coef_{abs_wvl}'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{abs_wvl}_Mm-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{abs_wvl}_Mm-1']['long_name'] = f'Extinction coefficient at {abs_wvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'dry_cal_ext_coef_{abs_wvl}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_dry_wvl}_RHd_Bulk_STP'          

            for i2 in range(len(wet_wvl["sca"])):
                wet_wvl_val = wet_wvl["sca"][i2]
                color_wet_wvl = wet_channel_color[i2]
                output_dict['VariableAttributes'][f'wet_meas_sca_coef_{wet_wvl_val}_Mm-1'] = {}
                output_dict['VariableAttributes'][f'wet_meas_sca_coef_{wet_wvl_val}_Mm-1']['short_name'] = f'wet_meas_sca_coef_{wet_wvl_val}'
                output_dict['VariableAttributes'][f'wet_meas_sca_coef_{wet_wvl_val}_Mm-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'wet_meas_sca_coef_{wet_wvl_val}_Mm-1']['long_name'] = f'Scattering coefficient at {wet_wvl_val} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived from gamma and scattering measurement at specified relative humidity.'
                output_dict['VariableAttributes'][f'wet_meas_sca_coef_{wet_wvl_val}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_wet_wvl}_RHsp_Bulk_STP'
                output_dict['VariableAttributes'][f'wet_meas_ext_coef_{wet_wvl_val}_Mm-1'] = {}
                output_dict['VariableAttributes'][f'wet_meas_ext_coef_{wet_wvl_val}_Mm-1']['short_name'] = f'wet_meas_ext_coef_{wet_wvl_val}'
                output_dict['VariableAttributes'][f'wet_meas_ext_coef_{wet_wvl_val}_Mm-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'wet_meas_ext_coef_{wet_wvl_val}_Mm-1']['long_name'] = f'Extinction coefficient at {wet_wvl_val} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived from humidified scattering and dry absorption.'
                output_dict['VariableAttributes'][f'wet_meas_ext_coef_{wet_wvl_val}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_wet_wvl}_RHsp_Bulk_STP'
                output_dict['VariableAttributes'][f'wet_cal_sca_coef_{wet_wvl_val}_Mm-1'] = {}
                output_dict['VariableAttributes'][f'wet_cal_sca_coef_{wet_wvl_val}_Mm-1']['short_name'] = f'wet_cal_sca_coef_{wet_wvl_val}'
                output_dict['VariableAttributes'][f'wet_cal_sca_coef_{wet_wvl_val}_Mm-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'wet_cal_sca_coef_{wet_wvl_val}_Mm-1']['long_name'] = f'Scattering coefficient at {wet_wvl_val} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived derived from ISARA.'
                output_dict['VariableAttributes'][f'wet_cal_sca_coef_{wet_wvl_val}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_wet_wvl}_RHsp_Bulk_STP'
                output_dict['VariableAttributes'][f'wet_cal_SSA_{wet_wvl_val}_unitless'] = {}
                output_dict['VariableAttributes'][f'wet_cal_SSA_{wet_wvl_val}_unitless']['short_name'] = f'wet_cal_SSA_{wet_wvl_val}'
                output_dict['VariableAttributes'][f'wet_cal_SSA_{wet_wvl_val}_unitless']['units'] = '1'
                output_dict['VariableAttributes'][f'wet_cal_SSA_{wet_wvl_val}_unitless']['long_name'] = f'Single scattering albedo at {wet_wvl_val} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived derived from ISARA.'
                output_dict['VariableAttributes'][f'wet_cal_SSA_{wet_wvl_val}_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_{color_wet_wvl}_RHsp_Bulk_None'
                output_dict['VariableAttributes'][f'wet_cal_ext_coef_{wet_wvl_val}_Mm-1'] = {}
                output_dict['VariableAttributes'][f'wet_cal_ext_coef_{wet_wvl_val}_Mm-1']['short_name'] = f'wet_cal_ext_coef_{wet_wvl_val}'
                output_dict['VariableAttributes'][f'wet_cal_ext_coef_{wet_wvl_val}_Mm-1']['units'] = 'm-1'
                output_dict['VariableAttributes'][f'wet_cal_ext_coef_{wet_wvl_val}_Mm-1']['long_name'] = f'Extinction coefficient at {wet_wvl_val} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived from ISARA.'
                output_dict['VariableAttributes'][f'wet_cal_ext_coef_{wet_wvl_val}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_wet_wvl}_RHsp_Bulk_STP'

            if val_wvl is not None:
                for i2 in range(len(val_wvl)):
                    valwvl = val_wvl[i2]
                    color_val_wvl = val_channel_color[i2]
                    output_dict['VariableAttributes'][f'dry_cal_sca_coef_{valwvl}_Mm-1'] = {}
                    output_dict['VariableAttributes'][f'dry_cal_sca_coef_{valwvl}_Mm-1']['short_name'] = f'dry_cal_sca_coef_{valwvl}'
                    output_dict['VariableAttributes'][f'dry_cal_sca_coef_{valwvl}_Mm-1']['units'] = 'm-1'
                    output_dict['VariableAttributes'][f'dry_cal_sca_coef_{valwvl}_Mm-1']['long_name'] =  f'Scattering coefficient at {valwvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                    output_dict['VariableAttributes'][f'dry_cal_sca_coef_{valwvl}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_val_wvl}_RHd_Bulk_STP'
                    output_dict['VariableAttributes'][f'dry_cal_SSA_{valwvl}_unitless'] = {}
                    output_dict['VariableAttributes'][f'dry_cal_SSA_{valwvl}_unitless']['short_name'] = f'dry_cal_SSA_{valwvl}'
                    output_dict['VariableAttributes'][f'dry_cal_SSA_{valwvl}_unitless']['units'] = '1'
                    output_dict['VariableAttributes'][f'dry_cal_SSA_{valwvl}_unitless']['long_name'] = f'Single scattering albedo at {valwvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                    output_dict['VariableAttributes'][f'dry_cal_SSA_{valwvl}_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_{color_val_wvl}_RHd_Bulk_None'
                    output_dict['VariableAttributes'][f'dry_cal_ext_coef_{valwvl}_Mm-1'] = {}
                    output_dict['VariableAttributes'][f'dry_cal_ext_coef_{valwvl}_Mm-1']['short_name'] = f'dry_cal_ext_coef_{valwvl}'
                    output_dict['VariableAttributes'][f'dry_cal_ext_coef_{valwvl}_Mm-1']['units'] = 'm-1'
                    output_dict['VariableAttributes'][f'dry_cal_ext_coef_{valwvl}_Mm-1']['long_name'] = f'Extinction coefficient at {valwvl} nm of BULK particles at DRY relative humidity of 20% and STANDARD temperature and pressure derived from ISARA.'
                    output_dict['VariableAttributes'][f'dry_cal_ext_coef_{valwvl}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_val_wvl}_RHd_Bulk_STP'
                    output_dict['VariableAttributes'][f'wet_cal_sca_coef_{valwvl}_Mm-1'] = {}
                    output_dict['VariableAttributes'][f'wet_cal_sca_coef_{valwvl}_Mm-1']['short_name'] = f'wet_cal_sca_coef_{valwvl}'
                    output_dict['VariableAttributes'][f'wet_cal_sca_coef_{valwvl}_Mm-1']['units'] = 'm-1'
                    output_dict['VariableAttributes'][f'wet_cal_sca_coef_{valwvl}_Mm-1']['long_name'] = f'Scattering coefficient at {valwvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived derived from ISARA.'
                    output_dict['VariableAttributes'][f'wet_cal_sca_coef_{valwvl}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Scattering_InSitu_{color_val_wvl}_RHsp_Bulk_STP'
                    output_dict['VariableAttributes'][f'wet_cal_SSA_{valwvl}_unitless'] = {}
                    output_dict['VariableAttributes'][f'wet_cal_SSA_{valwvl}_unitless']['short_name'] = f'wet_cal_SSA_{valwvl}'
                    output_dict['VariableAttributes'][f'wet_cal_SSA_{valwvl}_unitless']['units'] = '1'
                    output_dict['VariableAttributes'][f'wet_cal_SSA_{valwvl}_unitless']['long_name'] = f'Single scattering albedo at {valwvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived derived from ISARA.'
                    output_dict['VariableAttributes'][f'wet_cal_SSA_{valwvl}_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_{color_val_wvl}_RHsp_Bulk_None'
                    output_dict['VariableAttributes'][f'wet_cal_ext_coef_{valwvl}_Mm-1'] = {}
                    output_dict['VariableAttributes'][f'wet_cal_ext_coef_{valwvl}_Mm-1']['short_name'] = f'wet_cal_ext_coef_{valwvl}'
                    output_dict['VariableAttributes'][f'wet_cal_ext_coef_{valwvl}_Mm-1']['units'] = 'm-1'
                    output_dict['VariableAttributes'][f'wet_cal_ext_coef_{valwvl}_Mm-1']['long_name'] = f'Extinction coefficient at {valwvl} nm of BULK particles at WET relative humidity of 80% and STANDARD temperature and pressure derived from ISARA.'
                    output_dict['VariableAttributes'][f'wet_cal_ext_coef_{valwvl}_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_{color_val_wvl}_RHsp_Bulk_STP'  


            for i1 in range(L1):
                results_line = handle_line(i1,modelist, sd, dpg, dpu, dpl, full_dp, UBcutoff, LBcutoff, Sc, Abs, RHsc, gamma, dry_wvl, wet_wvl, val_wvl, CRI_p,kappa_p,LUT_output_variables)   
                for key2 in results_line:
                    if key2 in output_dict:
                        output_dict[key2][i1] = results_line[key2]
                    else:
                        output_dict[key2] = np.full((L1),np.nan)
                        output_dict['SourceFlag'][key2] = 'derived' 
                        output_dict['Dims'][key2] = 'time'
                        if key2 in output_dict['VariableAttributes']:
                            output_dict['VariableAttributes'][key2]['_FillValue'] = np.nan
                        else:
                            output_dict['VariableAttributes'][key2] = {}
                            output_dict['VariableAttributes'][key2]['_FillValue'] = np.nan
                        
                        output_dict[key2][i1] = results_line[key2]

            print(output_dict[f'kappa-550_unitless'].size)           
            output_filename = np.array(input_filename.split('.ict'))
            output_filename = output_filename[0]
            np.save(f'../ISARA_data_files/{DN}/Retrievals/{output_filename}.npy', output_dict)  

if __name__ == "__main__":
    RunISARA()