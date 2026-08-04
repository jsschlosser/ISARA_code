import importICARTT
import External_Consistency_Procedure
CaseStudy = External_Consistency_Procedure.CaseStudy
import StatsCode
import numpy as np
import pandas as pd
import itertools
import os
import sys
from datetime import datetime
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DayLocator, HourLocator, DateFormatter
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
#from mpl_toolkits.basemap import Basemap
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams

def Run():
    camp_name = input("Enter the campaign name in upper case (e.g., ARCSIX): ") 
    nonsphere_type = input("Enter the (LDR>0.08 & LR <35) coarse-mode nonsphere assumption: ") #LDR>0.08 & LR <35
    nonsphere_type2 = input("Enter the (LDR>0.08 & LR >35) coarse-mode nonsphere assumption: ") #LDR>0.08 & LR >35
    out_directory_name = input("Enter the output directory: ") 
    PrimaryAircraft = input("Enter the primary platform ID (e.g., KingAir): ") #'KingAir'
    SecondaryAircraft = input("Enter the secondary platform ID (e.g., Falcon): ") #'Falcon'  
    POI_files = input("Enter name of the directory conatining the\n periods of interest (e.g., KingAir_flightdata): ")
    

    #additional_spatial_separation_constraint = np.arange(3,18,3) #km
    #additional_time_separation_constraint = np.arange(3,33,3) #min
    additional_time_separation_constraint = np.array([6]) #min
    additional_spatial_separation_constraint = np.array([5]) #km
    CNT_LMT = np.array([0,2])   

    bns = {}
    #bns[0] = np.array([1,10,100,1000])
    #bns[1] = np.array([1,10,100,1000])
    bns[0] = np.arange(0,4000,1000)#np.array([1,10,100,1000,10000])
    bns[1] = np.arange(0,400,100)#np.array([1,10,100,1000])
    bns[2] = np.array([0, 0.1,0.2,0.3,0.4,0.5])
    bns[3] = np.array([0.7,0.8,0.9,1])
    bns[4] = np.array([0.7,0.8,0.9,1])
    bns[5] = np.array([0.7,0.8,0.9,1])
    bns[6] = np.array([1.3,1.35,1.4,1.45,1.50,1.55])
    bns[7] = np.array([0.0,0.01,0.02,0.03,0.04])
    bns[8] = np.arange(0,4000,1000)#np.array([1,10,100,1000,10000])
    bns[9] = np.arange(0,4000,1000)#np.array([1,10,100,1000])
    bns[10] = np.arange(0,400,100)  

    instlbls = {}
    instlbls[0] = "HSRL-2 & RSP"
    instlbls[1] = "HSRL-2"
    instlbls[2] = "RSP"
    instlbls[3] = "RSP"
    instlbls[4] = "RSP"
    instlbls[5] = "RSP"
    instlbls[6] = "RSP"
    instlbls[7] = "RSP"
    instlbls[8] = "RSP"
    instlbls[9] = "RSP+HSRL-2"
    instlbls[10] = "HSRL-2" 

    HSRLAerosolType = {}
    HSRLAerosolType["Ice"] = 1 
    HSRLAerosolType["Dusty_Mix"] = 2
    HSRLAerosolType["Marine"] = 3
    HSRLAerosolType["Urban_Pollution"] = 4
    HSRLAerosolType["Smoke"] = 5
    HSRLAerosolType["Fresh_Smoke"] = 6
    HSRLAerosolType["Pol_Marine"] = 7
    HSRLAerosolType["Dust"] = 8
    HSRLAerosolType["untyped_ambiguous_1"] = 9
    HSRLAerosolType["untyped_ambiguous_2"] = 10 

    def getPercentileList(
        prctile,
        suffix
      ):
        prctile_lst = np.array([f"{x}_percentile_{suffix}" for x in prctile])
        return prctile_lst          
    

    prctile = [0,50,68,95,100]
    prctile_lst_b = getPercentileList(prctile,"B")
    prctile_lst_ab = getPercentileList(prctile,"AB")
    prctile_lst_rb = getPercentileList(prctile,"RB")
    prctile_lst_arb = getPercentileList(prctile,"ARB")
    prctile_lst_x = getPercentileList(prctile,"x")
    prctile_lst_y = getPercentileList(prctile,"y")                  

    fs = 14
    ms = 75 

    plt.rcParams.update({'font.size': fs})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']#  

    if camp_name == "PACEPAX":
        source_key_vars = {
                            "latitude":"Latitude_BUCHOLTZ",
                            "longitude":"Longitude_BUCHOLTZ",
                            "altitude":"GPS_Altitude_BUCHOLTZ",
                            "LWC":"LWC_PVM_BUCHOLTZ",
                            "Air_Temp":"Temp_Ambient_BUCHOLTZ",
                            "InletFlag":"None",
                            "RH":"Relative_Humidity_Ambient_BUCHOLTZ",
                            "ssa_m":"SSA_amb_550nm_ZIEMBA"
                            }
        G = open(f'./{camp_name}/{POI_files}/SpiralProfiles_{camp_name}.csv', 'r') # open .csv    SpiralProfiles_{camp_name}.csv
        g = G.read().splitlines() # read .csv 
        hdrs = g[0].split(",") # define headers
        format_string = "%m/%d/%Y %H:%M:%S"
        LegID_dictionary = {}
        for sep_dist in range(1,np.add(len(g), - 1)):
            a = g[sep_dist].split(",") # split string into array and define as number array
            start_datetime_object = datetime.strptime(a[0], format_string)
            start_datstr = start_datetime_object
            start_time_seconds = start_datstr.hour*3600+start_datstr.minute*60+start_datstr.second
            stop_datetime_object = datetime.strptime(a[1], format_string)
            stop_datstr = stop_datetime_object
            stop_time_seconds = stop_datstr.hour*3600+stop_datstr.minute*60+stop_datstr.second
            kystr = start_datstr.strftime("%Y%m%d")
            if kystr in LegID_dictionary:
                LegID_dictionary[kystr]["datetime_Start_UTC"] = np.hstack((LegID_dictionary[kystr]["datetime_Start_UTC"],start_datstr))
                LegID_dictionary[kystr]["Time_Start_Seconds"] = np.hstack((LegID_dictionary[kystr]["Time_Start_Seconds"],start_time_seconds))
                LegID_dictionary[kystr]["datetime_Stop_UTC"] = np.hstack((LegID_dictionary[kystr]["datetime_Stop_UTC"],stop_datstr))
                LegID_dictionary[kystr]["Time_Stop_Seconds"] = np.hstack((LegID_dictionary[kystr]["Time_Stop_Seconds"],stop_time_seconds))
                LegID_dictionary[kystr]["LegIndex_flag"] = np.hstack((LegID_dictionary[kystr]["LegIndex_flag"],int(f"{sep_dist}09")))
            else:
                LegID_dictionary[kystr] = {}
                LegID_dictionary[kystr]["datetime_Start_UTC"] = start_datstr 
                LegID_dictionary[kystr]["datetime_Stop_UTC"] = stop_datstr 
                LegID_dictionary[kystr]["Time_Start_Seconds"] = start_time_seconds
                LegID_dictionary[kystr]["Time_Stop_Seconds"] = stop_time_seconds
                LegID_dictionary[kystr]["LegIndex_flag"] = int(f"{sep_dist}09")
    if camp_name == "ACTIVATE":
        source_key_vars = {
                            "latitude":"Latitude_deg_THORNHILL",
                            "longitude":"Longitude_deg_THORNHILL",
                            "altitude":"gpsALT_m_THORNHILL",
                            "LWC":"LWC_CDP_MOORE",
                            "Air_Temp":"Tstat_degC_THORNHILL",
                            "InletFlag":"InletFlag_LARGE_ZIEMBA",
                            "RH":"RHw_DLH_DISKIN",
                            "ssa_m":"SSA_amb_550nm_ZIEMBA"
                            }
        LegID_dictionary = {}
        for POI_file in os.listdir(f'./{camp_name}/{POI_files}/'):
            G = importICARTT.imp(f'./{camp_name}/{POI_files}/{POI_file}',2)
            kystr = str(int(G["Date_YYYYMMDD"][0]))
            LegID_dictionary[kystr] = G
    prctile = [0,50,68,95,100]  

    sys.path.insert(0, os.path.abspath("../"))  

    output_dictionary = {}
    stdv_dictionary = {}
    for sep_dist in additional_spatial_separation_constraint:
        print(sep_dist)
        for sep_time in additional_time_separation_constraint:
            print(sep_time)
            output_filename = f"{camp_name}-External_Closure_{sep_dist}km_{sep_time}min.npy" 
            output_dictionary[f'{sep_dist}km-{sep_time}min'] = {} 
            stdv_dictionary[f'{sep_dist}km-{sep_time}min'] = {} 
            for datstr in LegID_dictionary:
                #a = g[i1].split(",") # split string into array and define as number array
                #datstr = str(a[0])
                RSP_Filename =  [f for f in os.listdir(f'./{camp_name}/RSP') if f.__contains__(datstr)]
                InSitu_Filename_spheres =  [f for f in os.listdir(f'./{camp_name}/AmbientDataFiles/Sphere_kappa0-cri1-33') if (f.__contains__(datstr))]
                InSitu_Filename_nonspheres = [f for f in os.listdir(f'./{camp_name}/AmbientDataFiles/{nonsphere_type}') if (f.__contains__(datstr))]#InSitu_Filename_nonspheres = InSitu_Filename_spheres#
                InSitu_Filename_nonspheres2 = [f for f in os.listdir(f'./{camp_name}/AmbientDataFiles/{nonsphere_type2}') if (f.__contains__(datstr))]
                Lidar_Filename = [f for f in os.listdir(f'./{camp_name}/HSRL') if (f.__contains__(datstr))]
                if (len(InSitu_Filename_spheres) == 1) & (len(RSP_Filename) == 1) & (len(Lidar_Filename) == 1):
                    print(InSitu_Filename_spheres[0])
                    IS1_DirFN = f'./{camp_name}/AmbientDataFiles/Sphere_kappa0-cri1-33/{InSitu_Filename_spheres[0]}'
                    IS2_DirFN = f'./{camp_name}/AmbientDataFiles/{nonsphere_type}/{InSitu_Filename_nonspheres[0]}'#IS2_DirFN = IS1_DirFN #
                    IS3_DirFN = f'./{camp_name}/AmbientDataFiles/{nonsphere_type2}/{InSitu_Filename_nonspheres2[0]}'#IS2_DirFN = IS1_DirFN #
                    RSP_DirFN = f'./{camp_name}/RSP/{RSP_Filename[0]}'
                    Lid_DirFN = f'./{camp_name}/HSRL/{Lidar_Filename[0]}'
                    FN_suffix = f'{sep_dist}km_{sep_time}min_{datstr}'
                    OPdict = CaseStudy(camp_name,IS1_DirFN,IS2_DirFN,IS3_DirFN,RSP_DirFN,Lid_DirFN,FN_suffix,sep_time,sep_dist,
                                        out_directory_name,source_key_vars,LegID_dictionary[datstr])  
                    if OPdict is not None:
                        output_dictionary[f'{sep_dist}km-{sep_time}min'][f"{datstr}"] = OPdict['data']
                        stdv_dictionary[f'{sep_dist}km-{sep_time}min'][f"{datstr}"] = OPdict['sigma']  
                elif (len(InSitu_Filename_spheres) >1) & (len(RSP_Filename) == 1) & (len(Lidar_Filename) == 1):
                    RSP_DirFN = f'./{camp_name}/RSP/{RSP_Filename[0]}'
                    Lid_DirFN = f'./{camp_name}/HSRL/{Lidar_Filename[0]}'
                    for ifn in range(len(InSitu_Filename_spheres)):
                        print(InSitu_Filename_spheres[ifn])
                        IS1_DirFN = f'./{camp_name}/AmbientDataFiles/Sphere_kappa0-cri1-33/{InSitu_Filename_spheres[ifn]}'
                        IS2_DirFN = f'./{camp_name}/AmbientDataFiles/{nonsphere_type}/{InSitu_Filename_nonspheres[ifn]}'#IS2_DirFN = IS1_DirFN #
                        IS3_DirFN = f'./{camp_name}/AmbientDataFiles/{nonsphere_type2}/{InSitu_Filename_nonspheres2[ifn]}'#IS2_DirFN = IS1_DirFN #
                        FN_suffix = f'{sep_dist}km_{sep_time}min_{datstr}_{ifn}'
                        OPdict = CaseStudy(camp_name,IS1_DirFN,IS2_DirFN,IS3_DirFN,RSP_DirFN,Lid_DirFN,FN_suffix,sep_time,sep_dist,
                                            out_directory_name,source_key_vars,LegID_dictionary[datstr]) 
                        if OPdict is not None: 
                            output_dictionary[f'{sep_dist}km-{sep_time}min'][f"{datstr}L{ifn}"] = OPdict['data']
                            stdv_dictionary[f'{sep_dist}km-{sep_time}min'][f"{datstr}L{ifn}"] = OPdict['sigma']            
                elif (len(InSitu_Filename_spheres) >1) & (len(RSP_Filename) == 1) & (len(Lidar_Filename) >1):
                    RSP_DirFN = f'./{camp_name}/RSP/{RSP_Filename[0]}'
                    for ifn in range(len(InSitu_Filename_spheres)):
                        print(InSitu_Filename_spheres[ifn])  
                        Lid_DirFN = f'./{camp_name}/HSRL/{Lidar_Filename[ifn]}'
                        IS1_DirFN = f'./{camp_name}/AmbientDataFiles/Sphere_kappa0-cri1-33/{InSitu_Filename_spheres[ifn]}'
                        IS2_DirFN = f'./{camp_name}/AmbientDataFiles/{nonsphere_type}/{InSitu_Filename_nonspheres[ifn]}'#IS2_DirFN = IS1_DirFN #
                        IS3_DirFN = f'./{camp_name}/AmbientDataFiles/{nonsphere_type2}/{InSitu_Filename_nonspheres2[ifn]}'#IS2_DirFN = IS1_DirFN #
                        FN_suffix = f'{sep_dist}km_{sep_time}min_{datstr}_L{ifn}'
                        OPdict = CaseStudy(camp_name,IS1_DirFN,IS2_DirFN,IS3_DirFN,RSP_DirFN,Lid_DirFN,FN_suffix,sep_time,sep_dist,
                                            out_directory_name,source_key_vars,LegID_dictionary[datstr])  
                        if OPdict is not None:
                            output_dictionary[f'{sep_dist}km-{sep_time}min'][f"{datstr}L{ifn}"] = OPdict['data']
                            stdv_dictionary[f'{sep_dist}km-{sep_time}min'][f"{datstr}L{ifn}"] = OPdict['sigma']       
            VertStats = {}
            VertStats['extaltstats'] = {}
            VertStats['bscaltstats'] = {}
            VertStats['ldraltstats'] = {}
            VertStats['lraltstats'] = {}
            VertStats['naltstats'] = None
            for key in output_dictionary[f'{sep_dist}km-{sep_time}min']:
                if len(output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_total_ext_coef_legstats'][532])>0:
                    if VertStats['naltstats'] is None:
                        VertStats['naltstats'] =  output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_optical_N_legstats']      
                        for key2 in output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_total_ext_coef_legstats']:
                            VertStats['extaltstats'][key2] = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_total_ext_coef_legstats'][key2] 
                            VertStats['bscaltstats'][key2] = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_total_bsc_coef_legstats'][key2]
                            VertStats['ldraltstats'][key2] = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_total_ldr_legstats'][key2]
                            if key2 != 1064:
                                VertStats['lraltstats'][key2] = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_total_lr_legstats'][key2]  
                    else:
                        nstats = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_optical_N_legstats']
                        VertStats['naltstats'] = np.vstack((VertStats['naltstats'],nstats))
                        for key2 in VertStats['extaltstats']:
                            exstats = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_total_ext_coef_legstats'][key2]   
                            VertStats['extaltstats'][key2] = np.vstack((VertStats['extaltstats'][key2],exstats))
                            bsstats = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_total_bsc_coef_legstats'][key2]   
                            VertStats['bscaltstats'][key2] = np.vstack((VertStats['bscaltstats'][key2],bsstats))
                            ldrstats = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_total_ldr_legstats'][key2]   
                            VertStats['ldraltstats'][key2] = np.vstack((VertStats['ldraltstats'][key2],ldrstats))                            
                            if key2 != 1064:
                                lrstats = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_total_lr_legstats'][key2]   
                                VertStats['lraltstats'][key2] = np.vstack((VertStats['lraltstats'][key2],lrstats))                      

            colnames = np.hstack(('LegID','in-situ_AOD','R','log10_p-value',prctile_lst_b,'mean_b','stdev_b',prctile_lst_ab,'mean_ab','stdev_ab',
                            prctile_lst_rb,'mean_rb','stdev_rb',prctile_lst_arb,'mean_arb','stdev_arb','NMAD','MAD_Mm-1','NRMSD','RMSD_Mm-1',prctile_lst_x,'mean_x_Mm-1','stdev_x_Mm-1',
                            prctile_lst_y,'mean_y_Mm-1','stdev_y_Mm-1','MoranI','MoranEI','MoranI_znorm','MoranI_pnorm','MoranI_zrand','MoranI_prand','count'))
            colnames = ','.join(colnames)
            nalt_profilestats_output_filename = {}
            exalt_profilestats_output_filename = {}  
            bsalt_profilestats_output_filename = {}  
            ldralt_profilestats_output_filename = {}
            lralt_profilestats_output_filename = {}     

            VertData = {}
            VertStdev = {}
            VertData = {}
            VertStdev = {}
            VertData['Collocated_optical_N_IS'] = None
            VertData['Collocated_optical_N_HSRL+RSP'] = None
            VertStdev['Collocated_optical_N_IS'] = None
            VertStdev['Collocated_optical_N_HSRL+RSP'] = None        
            for key in VertStats['extaltstats']:
                VertData[f'Collocated_total_ext_coef_{key}_IS'] = None
                VertData[f'Collocated_total_ext_coef_{key}_HSRL'] = None 
                VertStdev[f'Collocated_total_ext_coef_{key}_IS'] = None
                VertStdev[f'Collocated_total_ext_coef_{key}_HSRL'] = None                
                VertData[f'Collocated_total_bsc_coef_{key}_IS'] = None
                VertData[f'Collocated_total_bsc_coef_{key}_HSRL'] = None 
                VertStdev[f'Collocated_total_bsc_coef_{key}_IS'] = None
                VertStdev[f'Collocated_total_bsc_coef_{key}_HSRL'] = None   
                VertData[f'Collocated_total_ldr_{key}_IS'] = None
                VertData[f'Collocated_total_ldr_{key}_HSRL'] = None 
                VertStdev[f'Collocated_total_ldr_{key}_IS'] = None
                VertStdev[f'Collocated_total_ldr_{key}_HSRL'] = None                   
                if key != 1064:
                    VertData[f'Collocated_total_lr_{key}_IS'] = None
                    VertData[f'Collocated_total_lr_{key}_HSRL'] = None 
                    VertStdev[f'Collocated_total_lr_{key}_IS'] = None
                    VertStdev[f'Collocated_total_lr_{key}_HSRL'] = None                                     
            VertData['smoke_counts'] = None
            VertData['aircraft_altitude_m'] = None
            #VertData['aircraft_horizontal_separation_m'] = None  
            countflg = {}
            filename_prefix =  f"../ISARA_data_files/{camp_name}/{out_directory_name}/{camp_name}"
            for cntlmt in CNT_LMT:
                countflg[cntlmt] = np.where((VertStats['naltstats'][:,-1]>cntlmt))[0]    
                nstats_trim = VertStats['naltstats']
                nstats_trim = nstats_trim[countflg[cntlmt],:]
                str_data = np.char.mod("%10.6f", nstats_trim )
                nalt_profilestats_output_filename[cntlmt] = f"{filename_prefix}-External_Closure_nalt_Stats_{sep_dist}km_{sep_time}min_{cntlmt}.csv"
                with open(nalt_profilestats_output_filename[cntlmt], 'w') as f:
                  np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)    #
                  
                for key in VertStats['extaltstats']:
                    if key in exalt_profilestats_output_filename:
                        extstats_trim = VertStats['extaltstats'][key]
                        extstats_trim = extstats_trim[countflg[cntlmt],:]
                        str_data = np.char.mod("%10.6f", extstats_trim)
                        exalt_profilestats_output_filename[key][cntlmt] = f"{filename_prefix}-External_Closure_ext{key}alt_Stats_{sep_dist}km_{sep_time}min_{cntlmt}.csv"
                        with open(exalt_profilestats_output_filename[key][cntlmt], 'w') as f:
                          np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)    #  
                    else:
                        exalt_profilestats_output_filename[key] = {}
                        extstats_trim = VertStats['extaltstats'][key]
                        extstats_trim = extstats_trim[countflg[cntlmt],:]
                        str_data = np.char.mod("%10.6f", extstats_trim)
                        exalt_profilestats_output_filename[key][cntlmt] = f"{filename_prefix}-External_Closure_ext{key}alt_Stats_{sep_dist}km_{sep_time}min_{cntlmt}.csv"
                        with open(exalt_profilestats_output_filename[key][cntlmt], 'w') as f:
                          np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)    #   

                for key in VertStats['bscaltstats']:
                    if key in bsalt_profilestats_output_filename:
                        bsstats_trim = VertStats['bscaltstats'][key]
                        bsstats_trim = bsstats_trim[countflg[cntlmt],:]
                        str_data = np.char.mod("%10.6f", bsstats_trim)
                        bsalt_profilestats_output_filename[key][cntlmt] = f"{filename_prefix}-External_Closure_bsc{key}alt_Stats_{sep_dist}km_{sep_time}min_{cntlmt}.csv"
                        with open(bsalt_profilestats_output_filename[key][cntlmt], 'w') as f:
                          np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)    #  
                    else:
                        bsalt_profilestats_output_filename[key] = {}
                        bsstats_trim = VertStats['bscaltstats'][key]
                        bsstats_trim = bsstats_trim[countflg[cntlmt],:]
                        str_data = np.char.mod("%10.6f", bsstats_trim)
                        bsalt_profilestats_output_filename[key][cntlmt] = f"{filename_prefix}-External_Closure_bsc{key}alt_Stats_{sep_dist}km_{sep_time}min_{cntlmt}.csv"
                        with open(bsalt_profilestats_output_filename[key][cntlmt], 'w') as f:
                          np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)    #   

                for key in VertStats['ldraltstats']:
                    if key in ldralt_profilestats_output_filename:
                        bsstats_trim = VertStats['ldraltstats'][key]
                        bsstats_trim = bsstats_trim[countflg[cntlmt],:]
                        str_data = np.char.mod("%10.6f", bsstats_trim)
                        ldralt_profilestats_output_filename[key][cntlmt] = f"{filename_prefix}-External_Closure_LDR{key}alt_Stats_{sep_dist}km_{sep_time}min_{cntlmt}.csv"
                        with open(ldralt_profilestats_output_filename[key][cntlmt], 'w') as f:
                          np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)    #  
                    else:
                        ldralt_profilestats_output_filename[key] = {}
                        bsstats_trim = VertStats['ldraltstats'][key]
                        bsstats_trim = bsstats_trim[countflg[cntlmt],:]
                        str_data = np.char.mod("%10.6f", bsstats_trim)
                        ldralt_profilestats_output_filename[key][cntlmt] = f"{filename_prefix}-External_Closure_LDR{key}alt_Stats_{sep_dist}km_{sep_time}min_{cntlmt}.csv"
                        with open(ldralt_profilestats_output_filename[key][cntlmt], 'w') as f:
                          np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)    #

                for key in VertStats['lraltstats']:
                    if key in lralt_profilestats_output_filename:
                        lrstats_trim = VertStats['lraltstats'][key]
                        lrstats_trim = lrstats_trim[countflg[cntlmt],:]
                        str_data = np.char.mod("%10.6f", lrstats_trim)
                        lralt_profilestats_output_filename[key][cntlmt] = f"{filename_prefix}-External_Closure_LR{key}alt_Stats_{sep_dist}km_{sep_time}min_{cntlmt}.csv"
                        with open(lralt_profilestats_output_filename[key][cntlmt], 'w') as f:
                          np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)    #  
                    else:
                        lralt_profilestats_output_filename[key] = {}
                        lrstats_trim = VertStats['lraltstats'][key]
                        lrstats_trim = lrstats_trim[countflg[cntlmt],:]
                        str_data = np.char.mod("%10.6f", lrstats_trim)
                        lralt_profilestats_output_filename[key][cntlmt] = f"{filename_prefix}-External_Closure_LR{key}alt_Stats_{sep_dist}km_{sep_time}min_{cntlmt}.csv"
                        with open(lralt_profilestats_output_filename[key][cntlmt], 'w') as f:
                          np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)    #  
                    

            wvls = None     
            for key in output_dictionary[f'{sep_dist}km-{sep_time}min']:
                #if len(output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_optical_N_legstats'][np.where(np.logical_not(np.isnan(output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical']['Collocated_optical_N_legstats'])))[0]])>0:
                for key2 in VertData:
                    if VertData[key2] is None:
                        if key2.__contains__('ext_coef')|key2.__contains__('bsc_coef'):
                            ksplit = key2.split('_')
                            keywvl = int(ksplit[4])
                            keyname = '_'.join(np.hstack([ksplit[0:4],ksplit[-1]]).astype(str))
                            VertData[key2] = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][keyname][keywvl]
                            wvls = keywvl 
                        elif key2.__contains__('_lr_')|key2.__contains__('_ldr_'):
                            ksplit = key2.split('_')
                            keywvl = int(ksplit[3])
                            keyname = '_'.join(np.hstack([ksplit[0:3],ksplit[-1]]).astype(str))
                            VertData[key2] = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][keyname][keywvl]
                            wvls = keywvl                           
                        else:
                            VertData[key2] = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][key2] 
                    else:
                        if key2.__contains__('ext_coef')|key2.__contains__('bsc_coef'):
                            ksplit = key2.split('_')
                            keywvl = int(ksplit[4])
                            keyname = '_'.join(np.hstack([ksplit[0:4],ksplit[-1]]).astype(str))
                            VertData[key2] = np.vstack((VertData[key2],output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][keyname][keywvl]))
                            wvls = np.hstack((wvls,keywvl))
                        elif key2.__contains__('_lr_')|key2.__contains__('_ldr_'):
                            ksplit = key2.split('_')
                            keywvl = int(ksplit[3])
                            keyname = '_'.join(np.hstack([ksplit[0:3],ksplit[-1]]).astype(str))
                            VertData[key2] = np.vstack((VertData[key2],output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][keyname][keywvl]))
                            wvls = np.hstack((wvls,keywvl))                            
                        else:
                            VertData[key2] = np.vstack((VertData[key2],output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][key2])) 
                for key2 in VertStdev:
                    if VertStdev[key2] is None:
                        if key2.__contains__('ext_coef')|key2.__contains__('bsc_coef'):
                            ksplit = key2.split('_')
                            keywvl = int(ksplit[4])
                            keyname = '_'.join(np.hstack([ksplit[0:4],ksplit[-1]]).astype(str))
                            VertStdev[key2] = stdv_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][keyname][keywvl]
                            wvls = keywvl 
                        elif key2.__contains__('_lr_')|key2.__contains__('_ldr_'):
                            ksplit = key2.split('_')
                            keywvl = int(ksplit[3])
                            keyname = '_'.join(np.hstack([ksplit[0:3],ksplit[-1]]).astype(str))
                            VertStdev[key2] = stdv_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][keyname][keywvl]
                            wvls = keywvl 
                        else:
                            VertStdev[key2] = stdv_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][key2] 
                    else:
                        if key2.__contains__('ext_coef')|key2.__contains__('bsc_coef'):
                            ksplit = key2.split('_')
                            keywvl = int(ksplit[4])
                            keyname = '_'.join(np.hstack([ksplit[0:4],ksplit[-1]]).astype(str))
                            VertStdev[key2] = np.vstack((VertStdev[key2],stdv_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][keyname][keywvl]))
                            wvls = np.hstack((wvls,keywvl))
                        elif key2.__contains__('_lr_')|key2.__contains__('_ldr_'):
                            ksplit = key2.split('_')
                            keywvl = int(ksplit[3])
                            keyname = '_'.join(np.hstack([ksplit[0:3],ksplit[-1]]).astype(str))
                            VertStdev[key2] = np.vstack((VertStdev[key2],stdv_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][keyname][keywvl]))
                            wvls = np.hstack((wvls,keywvl))
                        else:
                            VertStdev[key2] = np.vstack((VertStdev[key2],stdv_dictionary[f'{sep_dist}km-{sep_time}min'][key]['vertical'][key2]))
            VertData_trimmed = {}
            VertStdev_trimmed = {}       
            vertallstats_output_filename = {}
            for cntlmt in CNT_LMT:    
                VertData_trimmed[cntlmt] = {}
                VertStdev_trimmed[cntlmt] = {}                              
                for key2 in VertData:
                    VertData_trimmed[cntlmt][key2] = np.squeeze(VertData[key2][countflg[cntlmt],:]).reshape(1,-1)
                for key2 in VertStdev:    
                    VertStdev_trimmed[cntlmt][key2] = np.squeeze(VertStdev[key2][countflg[cntlmt],:]).reshape(1,-1)           
                aflg1 = np.where((np.logical_not(np.isnan(VertData_trimmed[cntlmt]['Collocated_optical_N_IS'])))&np.logical_not((np.isnan(VertData_trimmed[cntlmt]['Collocated_optical_N_HSRL+RSP']))))                    
                N_stats_dict = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_optical_N_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_optical_N_HSRL+RSP'][aflg1],prctile)    

                ext_stats_dict355 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_ext_coef_355_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_ext_coef_355_HSRL'][aflg1],prctile)
                ext_stats_dict532 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_ext_coef_532_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_ext_coef_532_HSRL'][aflg1],prctile)
                ext_stats_dict1064 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_ext_coef_1064_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_ext_coef_1064_HSRL'][aflg1],prctile)      

                bsc_stats_dict355 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_bsc_coef_355_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_bsc_coef_355_HSRL'][aflg1],prctile)
                bsc_stats_dict532 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_bsc_coef_532_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_bsc_coef_532_HSRL'][aflg1],prctile)
                bsc_stats_dict1064 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_bsc_coef_1064_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_bsc_coef_1064_HSRL'][aflg1],prctile)      

                ldr_stats_dict355 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_ldr_355_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_ldr_355_HSRL'][aflg1],prctile)
                ldr_stats_dict532 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_ldr_532_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_ldr_532_HSRL'][aflg1],prctile)
                ldr_stats_dict1064 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_ldr_1064_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_ldr_1064_HSRL'][aflg1],prctile)   

                lr_stats_dict355 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_lr_355_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_lr_355_HSRL'][aflg1],prctile)
                lr_stats_dict532 = StatsCode.Comparison(VertData_trimmed[cntlmt]['Collocated_total_lr_532_IS'][aflg1],VertData_trimmed[cntlmt]['Collocated_total_lr_532_HSRL'][aflg1],prctile)      

                colnames = ['Collocated_optical_N_cm-3','Collocated_total_355ext_coef_Mm-1','Collocated_total_532ext_coef_Mm-1','Collocated_total_1064ext_coef_Mm-1','Collocated_total_355bsc_coef_Mm-1sr-1',
                            'Collocated_total_532bsc_coef_Mm-1sr-1','Collocated_total_1064bsc_coef_Mm-1sr-1','Collocated_total_355LDR_unitless',
                            'Collocated_total_532LDR_unitless','Collocated_total_1064LDR_unitless','Collocated_total_355LR_sr-1','Collocated_total_532LR_sr-1']
                rows = np.hstack(('R','log10_p-value',prctile_lst_b,'mean_b','stdev_b',prctile_lst_ab,'mean_ab','stdev_ab',
                                prctile_lst_rb,'mean_rb','stdev_rb',prctile_lst_arb,'mean_arb','stdev_arb','NMAD','MAD','NRMSD','RMSD',prctile_lst_x,'mean_x','stdev_x',
                                prctile_lst_y,'mean_y','stdev_y','MoranI','MoranEI','MoranI_znorm','MoranI_pnorm','MoranI_zrand','MoranI_prand','count'))
                colnames = ",".join(colnames)
                stats_dict = np.array([N_stats_dict,ext_stats_dict355,ext_stats_dict532,ext_stats_dict1064,bsc_stats_dict355,bsc_stats_dict532,
                                        bsc_stats_dict1064,ldr_stats_dict355,ldr_stats_dict532,ldr_stats_dict1064,lr_stats_dict355,lr_stats_dict532])
                str_data = np.char.mod("%10.6f", stats_dict.T)
                str_data= np.column_stack((rows,str_data))
                vertallstats_output_filename[cntlmt] = f"{filename_prefix}-External_Closure_Vert_Stats_{sep_dist}km_{sep_time}min_{cntlmt}.csv"
                with open(vertallstats_output_filename[cntlmt], 'w') as f:
                  np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)  #                         

            ColmData = {}
            ColmData['legstart_date_time'] = None
            ColmData['legend_date_time'] = None      
            ColmData['RSP_date_time'] = None
            ColmData['Min_insitu_altitude_m'] = None
            ColmData['Max_insitu_altitude_m'] = None
            ColmData['smoke_counts_above_2.5km'] = None
            iaid = 0 
            for aid_key in HSRLAerosolType:
                ColmData[f'{aid_key}_count'] = None
            ColmData['total_AOT_355_RSP'] = None
            ColmData['fine_AOT_355_RSP'] = None
            ColmData['coarse_AOT_355_RSP'] = None
            ColmData['total_AOT_355_HSRL'] = None
            ColmData['total_AOT_532_RSP'] = None
            ColmData['fine_AOT_532_RSP'] = None
            ColmData['coarse_AOT_532_RSP'] = None
            ColmData['total_AOT_532_HSRL'] = None        
            ColmData['LegID'] = None
            ColmData['aircraft_horizontal_separation_m'] = None
            ColmData['count'] = None
            ColmData['min_CtoT_ext'] = None 
            ColmData['mean_CtoT_ext'] = None 
            ColmData['max_CtoT_ext'] = None 
            ColmData['min_LDR_532_HSRL'] = None 
            ColmData['mean_LDR_532_HSRL'] = None 
            ColmData['max_LDR_532_HSRL'] = None 
            ColmData['max_N5um_IS_cm-3'] = None
            ColmData['min_lat_IS'] = None
            ColmData['max_lat_IS'] = None
            ColmData['lat_RSP'] = None
            ColmData['min_lon_IS'] = None
            ColmData['max_lon_IS'] = None
            ColmData['lon_RSP'] = None
            ColmData['Collocated_fine_reff_IS_um'] =None
            ColmData['Collocated_fine_reff_RSP_um'] =None
            ColmData['Collocated_fine_veff_IS'] =None
            ColmData['Collocated_fine_veff_RSP'] =None  
            ColmData['Collocated_coarse_reff_IS_um'] =None
            ColmData['Collocated_coarse_reff_RSP_um'] =None
            ColmData['Collocated_coarse_veff_IS'] =None
            ColmData['Collocated_coarse_veff_RSP'] =None               
            ColmData['Collocated_fine_ssa_LARGE'] =None   
            ColmData['Collocated_optical_kext_IS_um2'] = None
            ColmData['Collocated_optical_kext_RSP_um2'] = None
            for keywvl in wvls:    
                ColmData[f'Collocated_fine_ssa_{keywvl}_IS'] =None
                ColmData[f'Collocated_fine_ssa_{keywvl}_RSP'] =None
                ColmData[f'Collocated_total_ssa_{keywvl}_IS'] =None
                ColmData[f'Collocated_total_ssa_{keywvl}_RSP'] =None
            ColmData['Collocated_fine_rri_IS'] =None
            ColmData['Collocated_fine_rri_RSP'] =None
            ColmData['Collocated_fine_iri_IS'] =None
            ColmData['Collocated_fine_iri_RSP'] =None
            ColmData['Collocated_optical_N_IS_cm-3']=None
            ColmData['Collocated_optical_N_RSP_cm-3']=None  

            ColmStd = {}
            ColmStd['Collocated_fine_reff_IS_um'] =None
            ColmStd['Collocated_fine_reff_RSP_um'] =None
            ColmStd['Collocated_fine_veff_IS'] =None
            ColmStd['Collocated_fine_veff_RSP'] =None  
            ColmStd['Collocated_coarse_reff_IS_um'] =None
            ColmStd['Collocated_coarse_reff_RSP_um'] =None
            ColmStd['Collocated_coarse_veff_IS'] =None
            ColmStd['Collocated_coarse_veff_RSP'] =None               
            ColmStd['Collocated_fine_ssa_LARGE'] =None  
            ColmStd['Collocated_optical_kext_IS_um2'] = None
            ColmStd['Collocated_optical_kext_RSP_um2'] = None
            for keywvl in wvls:    
                ColmStd[f'Collocated_fine_ssa_{keywvl}_IS'] =None
                ColmStd[f'Collocated_fine_ssa_{keywvl}_RSP'] =None
                ColmStd[f'Collocated_total_ssa_{keywvl}_IS'] =None
                ColmStd[f'Collocated_total_ssa_{keywvl}_RSP'] =None
            ColmStd['Collocated_fine_rri_IS'] =None
            ColmStd['Collocated_fine_rri_RSP'] =None
            ColmStd['Collocated_fine_iri_IS'] =None
            ColmStd['Collocated_fine_iri_RSP'] =None
            ColmStd['Collocated_optical_N_IS_cm-3']=None
            ColmStd['Collocated_optical_N_RSP_cm-3']=None   

            for key in output_dictionary[f'{sep_dist}km-{sep_time}min']:
                if len(output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['column'])>0:
                    for key2 in ColmData:
                        if ColmData[key2] is None:
                            ColmData[key2] = output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['column'][key2]
                        else:#
                            ColmData[key2] = np.hstack((ColmData[key2],output_dictionary[f'{sep_dist}km-{sep_time}min'][key]['column'][key2])) 
                    for key2 in ColmStd:
                        if ColmStd[key2] is None:
                            ColmStd[key2] = stdv_dictionary[f'{sep_dist}km-{sep_time}min'][key]['column'][key2]
                        else:#
                            ColmStd[key2] = np.hstack((ColmStd[key2],stdv_dictionary[f'{sep_dist}km-{sep_time}min'][key]['column'][key2]))                             
            columndata_filename = {}                    
            for cntlmt in CNT_LMT:
                theflg = ColmData['count']>cntlmt
                ColmData_trimed = {}
                for key2 in ColmData:
                    ColmData_trimed[key2] = ColmData[key2][theflg]  
                ColmStd_trimed = {}
                for key2 in ColmStd:
                    ColmStd_trimed[key2] = ColmStd[key2][theflg]                  
                x_col = {}
                x_col['fine_reff'] = ColmData_trimed['Collocated_fine_reff_IS_um']
                x_col['fine_veff'] = ColmData_trimed['Collocated_fine_veff_IS']
                x_col['coarse_reff'] = ColmData_trimed['Collocated_coarse_reff_IS_um']
                x_col['coarse_veff'] = ColmData_trimed['Collocated_coarse_veff_IS']            
                x_col['fine_ssa_532_large'] =ColmData_trimed['Collocated_fine_ssa_LARGE']
                for keywvl in wvls: 
                    x_col[f'fine_ssa_{keywvl}'] =ColmData_trimed[f'Collocated_fine_ssa_{keywvl}_IS'] 
                    x_col[f'total_ssa_{keywvl}'] = ColmData_trimed[f'Collocated_total_ssa_{keywvl}_IS'] 
                x_col['fine_rri'] = ColmData_trimed['Collocated_fine_rri_IS'] 
                x_col['fine_iri'] = ColmData_trimed['Collocated_fine_iri_IS'] 
                x_col['optical_N'] = ColmData_trimed['Collocated_optical_N_IS_cm-3']
                x_col['optical_kext'] = ColmData_trimed['Collocated_optical_kext_IS_um2']
                sdx_col = {}
                sdx_col['fine_reff'] = ColmStd_trimed['Collocated_fine_reff_IS_um']
                sdx_col['fine_veff'] = ColmStd_trimed['Collocated_fine_veff_IS']
                sdx_col['coarse_reff'] = ColmStd_trimed['Collocated_coarse_reff_IS_um']
                sdx_col['coarse_veff'] = ColmStd_trimed['Collocated_coarse_veff_IS']            
                sdx_col['fine_ssa_532_large'] =ColmStd_trimed['Collocated_fine_ssa_LARGE']
                for keywvl in wvls: 
                    sdx_col[f'fine_ssa_{keywvl}'] =ColmStd_trimed[f'Collocated_fine_ssa_{keywvl}_IS'] 
                    sdx_col[f'total_ssa_{keywvl}'] = ColmStd_trimed[f'Collocated_total_ssa_{keywvl}_IS'] 
                sdx_col['fine_rri'] = ColmStd_trimed['Collocated_fine_rri_IS'] 
                sdx_col['fine_iri'] = ColmStd_trimed['Collocated_fine_iri_IS'] 
                sdx_col['optical_N'] = ColmStd_trimed['Collocated_optical_N_IS_cm-3']  
                sdx_col['optical_kext'] = ColmStd_trimed['Collocated_optical_kext_IS_um2']
                y_col = {}
                y_col['fine_reff'] = ColmData_trimed['Collocated_fine_reff_RSP_um']
                y_col['fine_veff'] = ColmData_trimed['Collocated_fine_veff_RSP']
                y_col['coarse_reff'] = ColmData_trimed['Collocated_coarse_reff_RSP_um']
                y_col['coarse_veff'] = ColmData_trimed['Collocated_coarse_veff_RSP']                
                y_col['fine_ssa_532_large'] =ColmData_trimed['Collocated_fine_ssa_532_RSP'] 
                for keywvl in wvls: 
                    y_col[f'fine_ssa_{keywvl}'] =ColmData_trimed[f'Collocated_fine_ssa_{keywvl}_RSP'] 
                    y_col[f'total_ssa_{keywvl}'] = ColmData_trimed[f'Collocated_total_ssa_{keywvl}_RSP'] 
                y_col['fine_rri'] = ColmData_trimed['Collocated_fine_rri_RSP'] 
                y_col['fine_iri'] = ColmData_trimed['Collocated_fine_iri_RSP'] 
                y_col['optical_N'] = ColmData_trimed['Collocated_optical_N_RSP_cm-3']  
                y_col['optical_kext'] = ColmData_trimed['Collocated_optical_kext_RSP_um2']
                sdy_col = {}
                sdy_col['fine_reff'] = ColmStd_trimed['Collocated_fine_reff_RSP_um']
                sdy_col['fine_veff'] = ColmStd_trimed['Collocated_fine_veff_RSP']
                sdy_col['coarse_reff'] = ColmStd_trimed['Collocated_coarse_reff_RSP_um']
                sdy_col['coarse_veff'] = ColmStd_trimed['Collocated_coarse_veff_RSP']                
                sdy_col['fine_ssa_532_large'] =ColmStd_trimed['Collocated_fine_ssa_532_RSP'] 
                for keywvl in wvls: 
                    sdy_col[f'fine_ssa_{keywvl}'] =ColmStd_trimed[f'Collocated_fine_ssa_{keywvl}_RSP'] 
                    sdy_col[f'total_ssa_{keywvl}'] = ColmStd_trimed[f'Collocated_total_ssa_{keywvl}_RSP'] 
                sdy_col['fine_rri'] = ColmStd_trimed['Collocated_fine_rri_RSP'] 
                sdy_col['fine_iri'] = ColmStd_trimed['Collocated_fine_iri_RSP'] 
                sdy_col['optical_N'] = ColmStd_trimed['Collocated_optical_N_RSP_cm-3']   
                sdy_col['optical_kext'] = ColmStd_trimed['Collocated_optical_kext_RSP_um2']
                colnames = ["LegID", "legstart_date_time", "legend_date_time", "RSP_date_time", "Min_insitu_altitude", "Max_insitu_altitude", 'min_lat_IS', 'max_lat_IS', 'lat_RSP', 'min_lon_IS', 'max_lon_IS', 'lon_RSP',"smoke_counts_above_2.5km", "min_LDR_532_HSRL", "mean_LDR_532_HSRL", "max_LDR_532_HSRL",
                             "total_AOT_355", "fine_AOT_355", "coarse_AOT_355", "Lidar_AOT_355", "total_AOT_532", "fine_AOT_532", "coarse_AOT_532", "Lidar_AOT_532", "max_Nc_cm-3_IS",'min_CtoT_ext','mean_CtoT_ext','max_CtoT_ext',"Aircraft_Horizontal_Separation_m", "count"]
                colnames = ",".join(colnames)
                str_data = np.column_stack((ColmData_trimed['LegID'] ,ColmData_trimed['legstart_date_time'].astype(str),ColmData_trimed['legend_date_time'].astype(str),ColmData_trimed['RSP_date_time'].astype(str),
                                            ColmData_trimed['Min_insitu_altitude_m'].astype(int),ColmData_trimed['Max_insitu_altitude_m'].astype(int),
                                            ColmData_trimed['min_lat_IS'],ColmData_trimed['max_lat_IS'],ColmData_trimed['lat_RSP'],ColmData_trimed['min_lon_IS'],
                                            ColmData_trimed['max_lon_IS'],ColmData_trimed['lon_RSP'],ColmData_trimed['smoke_counts_above_2.5km'].astype(int),
                                            ColmData_trimed['min_LDR_532_HSRL'],ColmData_trimed['mean_LDR_532_HSRL'],ColmData_trimed['max_LDR_532_HSRL'],
                                            ColmData_trimed['total_AOT_355_RSP'],ColmData_trimed['fine_AOT_355_RSP'],ColmData_trimed['coarse_AOT_355_RSP'],ColmData_trimed['total_AOT_355_HSRL'],
                                            ColmData_trimed['total_AOT_532_RSP'],ColmData_trimed['fine_AOT_532_RSP'],ColmData_trimed['coarse_AOT_532_RSP'],ColmData_trimed['total_AOT_532_HSRL'],
                                            ColmData_trimed['max_N5um_IS_cm-3'],ColmData_trimed['min_CtoT_ext'],ColmData_trimed['mean_CtoT_ext'],ColmData_trimed['max_CtoT_ext'],ColmData_trimed['aircraft_horizontal_separation_m'].astype(int),ColmData_trimed['count'].astype(int)))
                for aid_key in HSRLAerosolType:
                    colnames = ",".join([colnames,f"{aid_key}_count"])
                    str_data =  np.column_stack((str_data,ColmData_trimed[f'{aid_key}_count']))
                for key3 in x_col:
    #                xy = np.vstack((x_col[key3],y_col[key3]))
    #                mean_ary = np.mean(xy,0)
    #                dif_ary = y_col[key3] - x_col[key3]
    #                rb = np.divide(dif_ary,mean_ary)
                    colnames = ",".join([colnames,f"in-situ_{key3},RSP_{key3},in-situ_stdev_{key3},RSP_stdev_{key3}"])
                    str_data =  np.column_stack((str_data,x_col[key3],y_col[key3],sdx_col[key3],sdy_col[key3]))     

                columndata_filename[cntlmt] = f"{filename_prefix}-External_Closure_Column_Data_{sep_dist}km_{sep_time}min_{cntlmt}.csv"               
                with open(columndata_filename[cntlmt], 'w') as f:
                    np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)
            df1 = {}
            df2 = {}
            df3 = {}
            df4 = {}
            df4b = {}
            df5 = {} 
            df6 = {}        
            for cntlmt in CNT_LMT:
                df1[cntlmt] = pd.read_csv(columndata_filename[cntlmt]) 
                df2[cntlmt] = {}
                for key in exalt_profilestats_output_filename:
                    df2[cntlmt][key] = pd.read_csv(exalt_profilestats_output_filename[key][cntlmt]) 
                df3[cntlmt] = {}
                for key in bsalt_profilestats_output_filename:
                    df3[cntlmt][key] = pd.read_csv(bsalt_profilestats_output_filename[key][cntlmt])  
                df4[cntlmt] = {}
                for key in lralt_profilestats_output_filename:
                    df4[cntlmt][key] = pd.read_csv(lralt_profilestats_output_filename[key][cntlmt])             
                df4b[cntlmt] = {}
                for key in ldralt_profilestats_output_filename:
                    df4b[cntlmt][key] = pd.read_csv(ldralt_profilestats_output_filename[key][cntlmt])                                   
                df5[cntlmt] = pd.read_csv(nalt_profilestats_output_filename[cntlmt])
                df6[cntlmt] = pd.read_csv(vertallstats_output_filename[cntlmt])
            with pd.ExcelWriter(f"{filename_prefix}-External_Closure_{sep_dist}km_{sep_time}min.xlsx") as writer:
                for cntlmt in CNT_LMT:
                    df1[cntlmt].to_excel(writer, sheet_name=f"Col_Data_{cntlmt}")
                    for key in exalt_profilestats_output_filename:
                        df2[cntlmt][key].to_excel(writer, sheet_name=f"Vert_Ext{key}_Prof_Stats_{cntlmt}")
                    for key in bsalt_profilestats_output_filename:
                        df3[cntlmt][key].to_excel(writer, sheet_name=f"Vert_Bsc{key}_Prof_Stats_{cntlmt}")
                    for key in lralt_profilestats_output_filename:
                        df4[cntlmt][key].to_excel(writer, sheet_name=f"Vert_LR{key}_Prof_Stats_{cntlmt}")   
                    for key in ldralt_profilestats_output_filename:
                        df4b[cntlmt][key].to_excel(writer, sheet_name=f"Vert_LDR{key}_Prof_Stats_{cntlmt}")                                                              
                    df5[cntlmt].to_excel(writer, sheet_name=f"Vert_Na_Prof_Stats_{cntlmt}")
                    df6[cntlmt].to_excel(writer, sheet_name=f"Vert_Data_Stats_{cntlmt}")    
    
    

                  
            x_vert = {}
            y_vert = {}
            sdx_vert = {}
            sdy_vert = {}
            lbls = {}
            wvl = {}
            instlbls = {}
            bns = {}
            units = {}
            Lidar_wvl = np.array([355,532,1064]).astype(int)
            for iwvl in Lidar_wvl:
                lbls[f'total_ext_coef{iwvl}'] = r"$C_{\rm ext}$ (Mm$^{-1}$)"
                wvl[f'total_ext_coef{iwvl}'] = f"{iwvl} nm"
                instlbls[f'total_ext_coef{iwvl}'] = "HSRL-2"        
                units[f'total_ext_coef{iwvl}'] = '%i' 
                lbls[f'total_bsc_coef{iwvl}'] = r"$C_{\rm bsc}$ (Mm$^{-1}$sr$^{-1}$)"
                wvl[f'total_bsc_coef{iwvl}'] = f"{iwvl} nm"
                instlbls[f'total_bsc_coef{iwvl}'] = "HSRL-2"  
                units[f'total_bsc_coef{iwvl}'] = '%i' 
                lbls[f'total_ldr{iwvl}'] = r"LDR"
                wvl[f'total_ldr{iwvl}'] = f"{iwvl} nm"
                instlbls[f'total_ldr{iwvl}'] = "HSRL-2"  
                units[f'total_ldr{iwvl}'] = '%0.2f'                 
                if iwvl != 1064:
                    lbls[f'total_lr{iwvl}'] = r"LR (sr)"
                    wvl[f'total_lr{iwvl}'] = f"{iwvl} nm"
                    instlbls[f'total_lr{iwvl}'] = "HSRL-2"  
                    units[f'total_lr{iwvl}'] = '%i'            
            for cntlmt in CNT_LMT:
                x_vert[cntlmt] = {}
                y_vert[cntlmt] = {}
                sdx_vert[cntlmt] = {}
                sdy_vert[cntlmt] = {}
                for iwvl in Lidar_wvl:
                    x_vert[cntlmt][f'total_ext_coef{iwvl}'] = VertData_trimmed[cntlmt][f'Collocated_total_ext_coef_{iwvl}_IS'] 
                    y_vert[cntlmt][f'total_ext_coef{iwvl}'] = VertData_trimmed[cntlmt][f'Collocated_total_ext_coef_{iwvl}_HSRL']
                    sdx_vert[cntlmt][f'total_ext_coef{iwvl}'] = VertStdev_trimmed[cntlmt][f'Collocated_total_ext_coef_{iwvl}_IS']
                    sdy_vert[cntlmt][f'total_ext_coef{iwvl}'] = VertStdev_trimmed[cntlmt][f'Collocated_total_ext_coef_{iwvl}_HSRL']
                    x_vert[cntlmt][f'total_bsc_coef{iwvl}'] = VertData_trimmed[cntlmt][f'Collocated_total_bsc_coef_{iwvl}_IS'] 
                    y_vert[cntlmt][f'total_bsc_coef{iwvl}'] = VertData_trimmed[cntlmt][f'Collocated_total_bsc_coef_{iwvl}_HSRL']
                    sdx_vert[cntlmt][f'total_bsc_coef{iwvl}'] = VertStdev_trimmed[cntlmt][f'Collocated_total_bsc_coef_{iwvl}_IS']
                    sdy_vert[cntlmt][f'total_bsc_coef{iwvl}'] = VertStdev_trimmed[cntlmt][f'Collocated_total_bsc_coef_{iwvl}_HSRL']
                    x_vert[cntlmt][f'total_ldr{iwvl}'] = VertData_trimmed[cntlmt][f'Collocated_total_ldr_{iwvl}_IS'] 
                    y_vert[cntlmt][f'total_ldr{iwvl}'] = VertData_trimmed[cntlmt][f'Collocated_total_ldr_{iwvl}_HSRL']
                    sdx_vert[cntlmt][f'total_ldr{iwvl}'] = VertStdev_trimmed[cntlmt][f'Collocated_total_ldr_{iwvl}_IS']
                    sdy_vert[cntlmt][f'total_ldr{iwvl}'] = VertStdev_trimmed[cntlmt][f'Collocated_total_ldr_{iwvl}_HSRL']                    
                    if iwvl != 1064:
                        x_vert[cntlmt][f'total_lr{iwvl}'] = VertData_trimmed[cntlmt][f'Collocated_total_lr_{iwvl}_IS'] 
                        y_vert[cntlmt][f'total_lr{iwvl}'] = VertData_trimmed[cntlmt][f'Collocated_total_lr_{iwvl}_HSRL']
                        sdx_vert[cntlmt][f'total_lr{iwvl}'] = VertStdev_trimmed[cntlmt][f'Collocated_total_lr_{iwvl}_IS']
                        sdy_vert[cntlmt][f'total_lr{iwvl}'] = VertStdev_trimmed[cntlmt][f'Collocated_total_lr_{iwvl}_HSRL']
                 
            bns['total_ext_coef355'] = np.arange(0,400,100)
            bns['total_ext_coef532'] = np.arange(0,250,50)
            bns['total_ext_coef1064'] = np.arange(0,100,20)
            bns['total_bsc_coef355'] = np.arange(0,10,2)
            bns['total_bsc_coef532'] = np.arange(0,8,2)
            bns['total_bsc_coef1064'] = np.arange(0,4,1)
            bns['total_ldr355'] = np.arange(0,0.4,0.1)
            bns['total_ldr532'] = np.arange(0,0.4,0.1)
            bns['total_ldr1064'] = np.arange(0,0.4,0.1)

            bns['total_lr355'] = np.arange(0,150,25)
            bns['total_lr532'] = np.arange(0,150,25)
            for cntlmt in CNT_LMT:    
                y_vert[cntlmt]['optical_N'] = VertData_trimmed[cntlmt]['Collocated_optical_N_HSRL+RSP']  
                x_vert[cntlmt]['optical_N'] = VertData_trimmed[cntlmt]['Collocated_optical_N_IS'] 
                sdy_vert[cntlmt]['optical_N'] = VertStdev_trimmed[cntlmt]['Collocated_optical_N_HSRL+RSP']  
                sdx_vert[cntlmt]['optical_N'] = VertStdev_trimmed[cntlmt]['Collocated_optical_N_IS']         
            lbls['optical_N'] = r"$N$ (cm$^{-3}$)"
            wvl['optical_N'] = ""
            instlbls['optical_N'] = "HSRL-2+RSP"
            bns['optical_N'] = np.arange(0,4000,1000)
            units[f'optical_N'] = '%i'
            for cntlmt in CNT_LMT: 
                for key3 in x_vert[cntlmt]:
                    x = np.squeeze(x_vert[cntlmt][key3])
                    y = np.squeeze(y_vert[cntlmt][key3])
                    xerr = np.squeeze(sdx_vert[cntlmt][key3])
                    yerr = np.squeeze(sdy_vert[cntlmt][key3])   
                    a = np.squeeze(VertData_trimmed[cntlmt]['aircraft_altitude_m'])
                    xymax = bns[key3][-1]
                    xymin = bns[key3][0]  
                    ytks = bns[key3]
                    fig = plt.figure(figsize=(7, 6), dpi=300)
                    # Add a gridspec with two rows and two columns and a ratio of 1 to 4 between
                    # the size of the marginal axes and the main axes in both directions.
                    # Also adjust the subplot parameters for a square plot.
                    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                                          left=0.175, right=0.9, bottom=0.1, top=0.95,
                                          wspace=0.1, hspace=0.1)
                    # Create the Axes.
                    ax2 = fig.add_subplot(gs[1, 0])
                    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax2)
                    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax2)# create figure and subplot                     
                    bins = np.linspace(xymin,xymax,50)
                    ytklbls = ["%i"%ix for ix in ytks]
                    xtklbls = ["%i"%ix for ix in ytks]      
                    # Plot heatmap
                    #im = sns.jointplot(data=dat, x="x", y="y", kind="hist")
                    ax2.errorbar(x,y, xerr=xerr, yerr=yerr, linestyle='none', zorder=1, elinewidth=1.5, ecolor='k', capsize=5)      

                    im = ax2.scatter(x,y,ms,a/1000,cmap=plt.get_cmap('jet',10),zorder=2)
                    ax2.plot(ytks,ytks,'--',color='xkcd:fuchsia',zorder=3,lw=1.5)      
                    ax2.set_ylabel(f"{instlbls[key3]} {lbls[key3]}", fontsize=fs) # set xaxis label 
                    ax2.set_xlabel(f"ISARA {lbls[key3]}", fontsize=fs) # set yaxis label   
                    ax2.set_ylim(xymin,xymax) # cut y-axis off at zero   
                    ax2.set_xlim(xymin,xymax)
                    ax2.set_aspect('auto')
                    xtklbls[0] = ""
                    xtklbls[-1] = ""
                    ax2.set_xticks(ytks, xtklbls)
                    ax2.set_yticks(ytks, ytklbls)
                    ax2.tick_params(direction='in', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
                    ax2.tick_params(axis='both', labelsize=fs, rotation=0)
                    for label in ax2.get_xticklabels():
                        label.set_horizontalalignment('center')    
                    # set the line widths of the axes
                    for axis in ['top','bottom','left','right']:
                        ax2.spines[axis].set_linewidth(1.5)  
                        ax_histx.spines[axis].set_linewidth(1.5)
                        ax_histy.spines[axis].set_linewidth(1.5) 
                    ax_histx.hist(x, bins=bins)
                    ax_histy.hist(y, bins=bins, orientation='horizontal')
                    #ax_histy.set_xscale("log") 
                    #ax_histx.set_yscale("log") 
                    ax_histx.minorticks_off()
                    ax_histy.minorticks_off()
                    ax_histx.tick_params(axis="x", labelbottom=False)
                    ax_histy.tick_params(axis="y", labelleft=False) 
                    ax_histx.tick_params(direction='in', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
                    ax_histy.tick_params(direction='in', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width#
                    cax = plt.axes([0.75, 0.8, 0.2, 0.055])
                    cbar = plt.colorbar(im,cax=cax,cmap=plt.get_cmap('jet',10), format='%0.1f',orientation='horizontal')
                    cbar.outline.set_linewidth(1.5)
                    cbar.ax.tick_params(direction='in', length=8, width=1.5, which="major", labelsize=24)
                    cbar.set_label('Altitude (km)',labelpad=-120)#
                    plt.savefig(f"{filename_prefix}-External_Closure_{key3}_1to1_All_alt_{sep_dist}km_{sep_time}min_{cntlmt}", dpi=300)
                    plt.show() # function to display the plot        
                    plt.close() #       
            j1 = 0
            j2 = 0
            j0 = 0        
            PLT_colors = ["xkcd:red","xkcd:royal blue"]
            PLT_shapes = ["o","^"] 
            FIGLBLS = np.array([["(a)","(b)","(c)"],["(d)","(e)","(f)"],["(g)","(h)","(i)"],["(j)","(k)","(l)"]])
            rcParams['figure.figsize'] = 9, 11 # W, H
            fig,ax2=plt.subplots(4, 3) # create figure and subplot
            for key3 in x_vert[CNT_LMT[0]]:
     
                xymax = bns[key3][-1]
                xymin = bns[key3][0]  
                ytks = bns[key3]
                if key3.__contains__('_bsc_'):
                    ytklbls = ["%0.1f"%ix for ix in ytks]
                    xtklbls = ["%0.1f"%ix for ix in ytks]  
                elif key3.__contains__('_ldr'):
                    ytklbls = ["%0.02f"%ix for ix in ytks]
                    xtklbls = ["%0.02f"%ix for ix in ytks]  
                else:
                    ytklbls = ["%i"%ix for ix in ytks]
                    xtklbls = ["%i"%ix for ix in ytks]  
                bounds = np.arange(0,6,0.5)
                lenbnds = len(bounds)
                N = lenbnds
                Jet = plt.get_cmap('jet', N)
                newcolors = Jet(np.linspace(0, 1, N))
                gry = np.array([0.75, 0.75, 0.75, 1])
                blk = np.array([0, 0, 0, 1])
                newcolors[0, :] = gry
                newcolors[-1, :] = blk
                cmap = ListedColormap(newcolors)
                boundsLbs = np.arange(0,6,0.5).astype(str)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)      
                #im = ax2[j1,j2].scatter(x,y,ms,a/1000, cmap=cmap, norm=norm, edgecolors='black', zorder=2, linewidth=1.5)
                
                j0=0
                for cntlmt in CNT_LMT: 
                    x = np.squeeze(x_vert[cntlmt][key3])
                    y = np.squeeze(y_vert[cntlmt][key3])
                    xerr = np.squeeze(sdx_vert[cntlmt][key3])
                    yerr = np.squeeze(sdy_vert[cntlmt][key3])
                    if cntlmt == 0:
                        zordr = 1
                        ax2[j1,j2].errorbar(x,y, xerr=xerr, yerr=yerr, linestyle='none', elinewidth=1.5, ecolor='k', zorder=zordr, capsize=3.5,alpha=0.4)
                        zordr+=1
                        ax2[j1,j2].plot(x,y,marker=PLT_shapes[j0], color=gry, linestyle='none', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k', zorder=zordr,alpha=0.4)  
                        j0 += 1
                    else:
                        zordr = 1
                        ax2[j1,j2].errorbar(x,y, xerr=xerr, yerr=yerr, linestyle='none', elinewidth=1.5, ecolor='k', zorder=zordr, capsize=3.5)
                        zordr+=1
                        ax2[j1,j2].plot(x,y,marker=PLT_shapes[j0], color=PLT_colors[j0], linestyle='none', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k', zorder=zordr)  
                        j0 += 1
                ax2[j1,j2].plot(ytks,ytks,'--',color='xkcd:fuchsia',lw=1.5,zorder=zordr)     
                ax2[j1,j2].set_ylabel(f"{instlbls[key3]} {lbls[key3]}", fontsize=fs) # set xaxis label 
                ax2[j1,j2].set_xlabel(f"ISARA {lbls[key3]}", fontsize=fs) # set yaxis label   
                ax2[j1,j2].set_ylim(xymin,xymax) # cut y-axis off at zero   
                ax2[j1,j2].set_xlim(xymin,xymax)
                ax2[j1,j2].set_aspect('auto')
                xtklbls[0] = ""
                xtklbls[-1] = ""
                ax2[j1,j2].set_xticks(ytks, xtklbls)
                ax2[j1,j2].set_yticks(ytks, ytklbls)
                ax2[j1,j2].tick_params(direction='in', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
                ax2[j1,j2].tick_params(axis='both', labelsize=fs, rotation=0)
                at = AnchoredText(f"{FIGLBLS[j1,j2]} {wvl[key3]}", prop=dict(size=fs), frameon=False, loc='upper left')
                ax2[j1,j2].add_artist(at)
                for label in ax2[j1,j2].get_xticklabels():
                    label.set_horizontalalignment('center')    
                # set the line widths of the axes
                for axis in ['top','bottom','left','right']:
                    ax2[j1,j2].spines[axis].set_linewidth(1.5)   
                if j1 < 3:
                    j1 += 1
                else:
                    j1 = 0
                    j2 += 1
                    
            plt.tight_layout()
            plt.subplots_adjust(bottom=0.075, right=0.95, top=0.95)    
            #cax = plt.axes([0.87, 0.075, 0.055, 0.875])
            #cbar =  plt.colorbar(im,cax=cax,cmap=cmap, format='%0.1f', norm=norm,boundaries=bounds,ticks=bounds)
            #cbar.ax.tick_params(length=8, width=1.5, which="major")
            #cbar.outline.set_linewidth(1.5)
            #cbar.set_label('Altitude (km)',labelpad=5)  
            plt.savefig(f"{filename_prefix}-External_Closure_1to1_All_alt_{sep_dist}km_{sep_time}min", dpi=300)
            plt.show() # function to display the plot        
            plt.close() #   
       