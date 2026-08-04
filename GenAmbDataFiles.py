import Calculate_Ambient_Properties
calprps = Calculate_Ambient_Properties.Run
import lut
initialize_spheres = lut.initialize_spheres
initialize_spheroids  = lut.initialize_spheroids 
import sizedistmerge as sdm
import load_sizebins
import itertools
import os
import sys
import StatsCode
import datetime
from datetime import datetime
import h5py
import numpy as np
import nc_write
ncwrite = nc_write.cf19

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

    def APS_correction(sd_aps, dpg_aps, rho_aps,correction_params):
      # Ensure inputs are numpy arrays
      sd_aps = np.asarray(sd_aps)
      dpg_aps = np.asarray(dpg_aps)
      rho_aps = np.asarray(rho_aps)
      
      B = inlet_correction_params['base']#0
      M = inlet_correction_params['maximum']#0.9902
      xhalf = inlet_correction_params['xhalf']#3.1070
      rate = inlet_correction_params['rate']#-4.8032
      
      # Calculate denominator for all points
      denominator = B + (M - B) / (1 + (xhalf / (dpg_aps * np.sqrt(rho_aps))) ** rate)
      corrected = sd_aps / denominator
      
      # Apply threshold condition: use corrected if condition met, else keep original
      condition = (dpg_aps * rho_aps) < 5.8
      return np.where(condition, corrected, sd_aps)

    def dict_reconfig(
        dictionaryname,
      ):
        OP = dict()
        io = 0
        for key in dictionaryname.item():
            value = dictionaryname.item().get(key)
            OP[key] = value
            #print(key)
        return OP  

    def grabvalues(
        dictionaryname,
        startofkeyname
      ):
        OP = dict()
        io = 0
        for key in dictionaryname:
          if key.startswith(startofkeyname):
            value = dictionaryname[key]
            OP[io] = np.squeeze(value.T)
            io += 1
        return OP    

    def grab_data(data,key_name):
        for key in data.keys():
            if key.__contains__(key_name):
                return data[key]    

    def grabvaluessd(
        dictionaryname,
      ):
        OP = dict()
        OP['SD'] = dict()
        OP['dpg'] = dict()
        OP['dpu'] = dict()
        OP['dpl'] = dict()
        io = 0
        dp = None
        for key in dictionaryname:
          if key.startswith("dndlogdp_"):
            value = dictionaryname[key]
            OP['SD'][io] = np.squeeze(value.T)
            value = dictionaryname[f'dpu_bin{io}_um']
            OP['dpu'][io] = np.squeeze(value.T)
            value = dictionaryname[f'dpg_bin{io}_um']
            OP['dpg'][io] = np.squeeze(value.T)
            value = dictionaryname[f'dpl_bin{io}_um']
            OP['dpl'][io] = np.squeeze(value.T)            
            io += 1
        OP['bincount'] = io    
        return OP

    def genCRI(lambda_list):
      def SSCRI(x):
        n=(1+0.00055+0.19800/(1-(0.050/x)**2)+0.48398/(1-(0.100/x)**2)+0.38696/(1-(0.128/x)**2)+0.25998/(1-(0.158/x)**2)+0.08796/(1-(40.50/x)**2)+3.17064/(1-(60.98/x)**2)+0.30038/(1-(120.34/x)**2))**.5
        k=0
        return n,k
      cri_filename = f'SS.cri'
      CRI_file = open(cri_filename, 'w')
      lambda_list = np.array(lambda_list,ndmin = 1)
      RRI = np.zeros(len(lambda_list))
      IRI = np.zeros(len(lambda_list))
      for i_wvl in np.arange(len(lambda_list)):
        RRI[i_wvl],IRI[i_wvl]=SSCRI(lambda_list[i_wvl]/1000)

        #CRI_file.write('%10.4f %10.4f %10.4f\n'%(lambda_list[i_wvl]/1000,n,k))
      #CRI_file.close()
      return RRI,IRI

    #wvl = np.arange(355,1065,10).reshape(-1) 
    wvl = np.array([355, 400, 450, 532, 550, 555, 660, 700, 750, 1064])
    num_wvl = len(wvl)
    camp_name = input("Enter the campaign name in upper case (e.g., ARCSIX): ") 
    prodDOI = input("Enter the DOI for this data repository: ")  
    camp_name_lower = camp_name.lower()
    resolution = input("Enter the temporal resolution of interest in seconds (e.g., 30): ") 
    data_directory = input("Enter the name of the directory that contains\nISARA retrieval dictionaries (e.g., 30s): ")
    output_directory = input("Enter the name of the directory where the output files are to be stored: ")
    reference_platform = input("Enter the platform of interest (e.g., cirpas-to or MARINA-TOWER): ") 
    revision_number = input("Enter the revision number (e.g., A, B, 0, 1): ")
    update_info = input("Enter description of revision history or leave empty if first revision: ")
    coarse_shape_distribution  = input("Prescribe shape distribution (yes or no)? ")
    if coarse_shape_distribution == "no":
      LUT_c = input("Enter name of coarse-mode particle LUT file\n(e.g., AerosolLUT_1000_100_0.355_650bins_2325CRI_ln2rKr_Twomey.dat): ")
    else:
      shap_dist = input("Enter .csv filename with the shape distribution: ")
      LUT_c = input("Enter directory with the LUT files\n(e.g., KERNEL_n22k16_181_123): ")  

    kapc = float(input("Enter coarse-mode particle kappa (e.g., 0.1): "))
    coarse_wvl_dep_flag = input("Is the coarse-mode CRI wavelength dependant (yes or no)?: ")
    if coarse_wvl_dep_flag == 'yes':
      RRIc,IRIc = genCRI(wvl)
      rric = RRIc
      iric = IRIc
    else:
      rric = float(input("Set coarse-mode RRI (e.g., 1.52): "))
      iric = float(input("Set coarse-mode IRI (e.g., 10**(-8)): "))
      ##rric = 1.52
      ##iric = 10**(-8)               
      #if LUT_c=='sphere':
        #rric = 1.33
        #iric = 0 
      #else:
        #rric = 1.52
        #iric = 10**(-8)         
    num_theta = int(input("Enter number of scattering angles from 0 to 180: "))
    RH_name = input("Enter the short name of the variable representing ambient relative humidity in source data file: ") 
    coarsemode = input("Is there a probe used for coarse-mode aerosol (yes or no)? ")
    if coarsemode == "yes":
      coarsemode_keynames = input(f"Enter the instrument name(s) for coarse mode data separated by comma (e.g., CAS,FCDP,CDP): ").split(',')
      UBcutoff = float(input(f"Enter the upper bound of particle sizes\nfor coarse-mode data in um (e.g., 20): "))
      LBcutoff = float(input(f"Enter the lower bound of particle sizes\nfor coarse-mode data in um (e.g., 3): "))
      dpg_coarse = {}
      dpu_coarse = {}
      dpl_coarse = {}
      coarse_bins = {}
      for coarse_keyname in coarsemode_keynames:
        ifn = [f for f in os.listdir(f'../ISARA_data_files/{camp_name}/SDBinInfo/') if f.__contains__(coarse_keyname)]
        dpData = load_sizebins.Load(f'../ISARA_data_files/{camp_name}/SDBinInfo/{ifn[0]}')
        dpg_coarse0 = grab_data(dpData,"Mid Points")
        dpu_coarse0 = grab_data(dpData,"Upper Bounds")
        dpl_coarse0 = grab_data(dpData,"Lower Bounds")
        coarse_binbounds = np.where(((dpl_coarse0>=LBcutoff)&(dpu_coarse0<=UBcutoff)))[0]    
        dpg_coarse[coarse_keyname] = dpg_coarse0
        dpu_coarse[coarse_keyname] = dpu_coarse0
        dpl_coarse[coarse_keyname] = dpl_coarse0
        coarse_bins[coarse_keyname] = coarse_binbounds
    APS_inlet_correction = input(f"Is there an intet efficiency correction (yes or no)? ")  
    if APS_inlet_correction == "yes":
      inlet_correction_params = {}
      inlet_correction_params['base'] = float(input(f"Enter the base term: "))
      inlet_correction_params['maximum'] = float(input(f"Enter the maximum term: "))
      inlet_correction_params['xhalf'] = float(input(f"Enter the xhalf term: "))
      inlet_correction_params['rate'] = float(input(f"Enter the rate term: "))

    mode = ["Nucl","Accu","Coarse"]
    mode_sizes = {}
    mode_sizes[mode[0]] = np.array([0.001,1])
    mode_sizes[mode[1]] = np.array([0.1,1])
    mode_sizes[mode[2]] = np.array([1,20])
    bulk_names = ["NucltoCoarse","AccutoCoarse","NucltoAccu","Coarse"]
    bulk_shortnames = {
                        "NucltoCoarse":"total",
                        "AccutoCoarse":"optical",
                        "NucltoAccu":"fine",
                        "Coarse":"coarse"
                      }
    bulk_sizes = {}
    bulk_sizes[bulk_names[0]] = np.array([0.001,20])
    bulk_sizes[bulk_names[1]] = np.array([0.1,20])
    bulk_sizes[bulk_names[2]] = np.array([0.001,0.1])
    bulk_sizes[bulk_names[3]] = np.array([1,20])
    bulk_bounds = {}
    bulk_bounds[bulk_names[0]] = np.array([0,1,2])
    bulk_bounds[bulk_names[1]] = np.array([1,2])
    bulk_bounds[bulk_names[2]] = np.array([0,1])
    bulk_bounds[bulk_names[3]] = np.array([2])
    inputput_filename_suffix = f'{camp_name_lower}-mrg{resolution}_{reference_platform}'
    output_filename_suffix = f'{camp_name_lower}-ISARAProducts-{reference_platform}'
    insitu_Filename =  [f for f in os.listdir(f'../ISARA_data_files/{camp_name}/Retrievals/{data_directory}') if (f.startswith(inputput_filename_suffix)&f.endswith('.npy')&np.logical_not(f.endswith('_DataRetrievals.npy')))]
    LUT_output_variables_fine = {}
    fine_shap_dist_ary = np.array([1])
    if (coarse_shape_distribution == "no"):
      LUT_output_variables_coarse = {}
      LUT_file_num = 0
      coarse_shap_dist_ary = np.array([1])
      if LUT_c.__contains__("_Twomey"):
        LUT_output_variables_coarse[LUT_file_num] = initialize_spheres(f'./LUT_data/{LUT_c}')
        LUT_output_variables_fine[LUT_file_num] = LUT_output_variables_coarse[LUT_file_num]
        LUT_f = LUT_c
      else:
        LUT_output_variables_coarse[LUT_file_num] = initialize_spheroids(f'./LUT_data/{LUT_c}')
        pathsplit = LUT_c.split("/")
        LUT_f = np.squeeze([f for f in os.listdir(f'./LUT_data/{pathsplit[0]}') if f.__contains__("1.000")])
        LUT_output_variables_fine[LUT_file_num] = initialize_spheroids(f'./LUT_data/{pathsplit[0]}/{LUT_f}')   
    else:
      coarse_shap_dist_ary = np.genfromtxt(f'./ShapeDistributions/{shap_dist}.csv', delimiter=', ', skip_header=1)
      coarse_shap_dist_ary = coarse_shap_dist_ary[:,1].astype(float)
      LUT_output_variables_coarse = {}
      LUT_file_num = 0
      for f in os.listdir(f'./LUT_data/{LUT_c}'):
        LUT_output_variables_coarse[LUT_file_num] = initialize_spheroids(f'./LUT_data/{LUT_c}/{f}')
        if f.__contains__("1.000"):
          LUT_f = f
          LUT_output_variables_fine[0] = LUT_output_variables_coarse[LUT_file_num]
        LUT_file_num += 1   

    for f in insitu_Filename:
      print(f)
      flight_number = np.array(f.split("_"))[-2]
      OP_Dictionary= dict_reconfig(np.load(f"./{camp_name}/Retrievals/{data_directory}/{f}",allow_pickle='TRUE')) 

      for key in OP_Dictionary['VariableAttributes'].keys():
        if 'units' in OP_Dictionary['VariableAttributes'][key].keys():
          if OP_Dictionary['VariableAttributes'][key]['units'] == 'missing':
            del OP_Dictionary[key]
            #print(key)
      CRI_flag = grabvalues(OP_Dictionary,'attempt_flag_CRI')[0]
      Npt00 = np.nansum(CRI_flag==0)
      print(f'Number of points without enough data: {Npt00}')
      Npt01 = np.nansum(CRI_flag==1)
      Npt02 = np.nansum(CRI_flag==2)
      print(f'Attempts made: {Npt01+Npt02}')
      print(f'Number of successful CRI retrievals: {Npt02}')    
      print(r'(Successes)/(Attempts)x100%: ',((Npt02/(Npt01+Npt02))*100).astype(int))
      k_flag = grabvalues(OP_Dictionary,'attempt_flag_kappa')[0]
      Npt10 = np.nansum(k_flag==0)
      print(f'Number of points without enough data: {Npt10}')
      Npt11 = np.nansum(k_flag==1)
      Npt12 = np.nansum(k_flag==2)
      print(f'Attempts made: {Npt11+Npt12}')
      print(f'Number of successful kappa retrievals: {Npt12}')    
      print(r'(Successes)/(Attempts)x100%: ',((Npt12/(Npt11+Npt12))*100).astype(int))

      RH = grabvalues(OP_Dictionary,RH_name)[0]
      RH[np.where((RH>99))[0]] = 99
      stdPT_LAS = grabvalues(OP_Dictionary,'stdPT_ZIEMBA')[0]
      rhof = grabvalues(OP_Dictionary,'dry_rho')[0]
      dIRI = grabvalues(OP_Dictionary,'dry_IRI')[0]
      dIRI[np.isnan(dIRI)] = 0
      dRRI = grabvalues(OP_Dictionary,'dry_RRI')[0]
      dRRI[np.isnan(dRRI)] = 1.55
      length_data = len(dRRI)
      kappa = grabvalues(OP_Dictionary,'kappa')[0]          
      #kappa[np.where((np.logical_not(np.isnan(dIRI))&(RH<40)&np.isnan(kappa)))]=0
      fmtdatetime_Start = OP_Dictionary["datetime_Start_UTC"]
      utc_times = [ np.datetime_as_string(n,timezone='UTC') for n in fmtdatetime_Start ]        
      utc_str_arr = np.array(utc_times,dtype='<U35')
      OP_Dictionary["datetime_Start_UTC"] = utc_str_arr  
      fmtdatetime_Stop = OP_Dictionary["datetime_Stop_UTC"]
      utc_times = [ np.datetime_as_string(n,timezone='UTC') for n in fmtdatetime_Stop ]
      utc_str_arr = np.array(utc_times,dtype='<U35')
      OP_Dictionary["datetime_Stop_UTC"] = utc_str_arr 
      Date = fmtdatetime_Start[0].astype(datetime)
      Date = str(Date.date())
      Date = Date.split("-")
      Date = "".join(Date)
      Time_Start = OP_Dictionary["Time_Start_Seconds"]
      Time_Stop = OP_Dictionary["Time_Stop_Seconds"]
      Time_Mid = np.nanmean([Time_Start,Time_Stop],axis=0)
      gf = np.power((1+kappa*RH/(100-RH)),1/3)#D/Ddry = (1+kappa*RH/(100-RH))**(1/3))
      gf[((dRRI>0)&(RH<40)&np.isnan(kappa))]=1
      RRIw = np.squeeze(1.33*np.ones(length_data))
      IRIw = np.squeeze(np.zeros(length_data))#
      GF2 = np.squeeze(gf)
      IRIf = (np.squeeze(dIRI)+((GF2**3)-1)*IRIw)/(GF2**3)#dIRIf  #
      RRIf = (np.squeeze(dRRI)+((GF2**3)-1)*RRIw)/(GF2**3)  
      IRIf = np.tile(IRIf,(num_wvl,1))
      RRIf = np.tile(RRIf,(num_wvl,1))
      if (coarsemode == "yes"):
        gfc = np.power((1+kapc*RH/(100-RH)),1/3)#D/Ddry = (1+kappa*RH/(100-RH))**(1/3))
        GF2 = np.squeeze(gfc)
        RRIc = (np.squeeze(rric)+((GF2**3)-1)*RRIw)/(GF2**3)
        IRIc = (np.squeeze(iric)+((GF2**3)-1)*IRIw)/(GF2**3)#dIRIf  #
        if coarse_wvl_dep_flag != 'yes':
          RRIc = np.tile(RRIc,(num_wvl,1))
          IRIc = np.tile(IRIc,(num_wvl,1))
      else:
        RRIc = RRIf
        IRIc = IRIf    
      OP = grabvaluessd(OP_Dictionary)
      if 0 in OP['SD']:
        SD = OP['SD']
        dataset_length = len(SD[0])
        dpl = OP['dpl']
        dpu = OP['dpu']
        dpg = OP['dpg']
        fine_bin_length = OP['bincount']
        sd = np.full((fine_bin_length,dataset_length),np.nan).astype(float)
        dD_l = np.full((fine_bin_length,dataset_length),np.nan).astype(float)
        dD_u = np.full((fine_bin_length,dataset_length),np.nan).astype(float)
        dD_g = np.full((fine_bin_length,dataset_length),np.nan).astype(float)
        D_l = np.full((fine_bin_length,dataset_length),np.nan).astype(float)
        D_u = np.full((fine_bin_length,dataset_length),np.nan).astype(float)
        D_g = np.full((fine_bin_length,dataset_length),np.nan).astype(float)
        for i1 in range(len(SD)):
          if APS_inlet_correction=="yes":
            SD[i1] = APS_correction(SD[i1],dpg[i1],rhof,inlet_correction_params)
          sd[i1,:] = SD[i1]/stdPT_LAS
          D_l[i1,:] = dpl[i1]*gf 
          D_u[i1,:] = dpu[i1]*gf 
          D_g[i1,:] = dpg[i1]*gf
          dD_l[i1,:] = dpl[i1] 
          dD_u[i1,:] = dpu[i1] 
          dD_g[i1,:] = dpg[i1]

        coarseflag = None
        sd_coarse = None
        if coarsemode == "yes":
          for coarse_keyname in coarsemode_keynames:
            if coarseflag is None:
              idpg = 0
              for key in OP_Dictionary["VariableAttributes"]:
                if "long_name" in OP_Dictionary["VariableAttributes"][key]:
                  lngname =  OP_Dictionary["VariableAttributes"][key]["long_name"]
                  if lngname.__contains__(coarse_keyname+"}")&str(key.casefold()).__contains__("bin"):
                    if sd_coarse is None:
                      sd_coarse = np.full((len(dpg_coarse[coarse_keyname]),dataset_length),np.nan)
                    sd_coarse[idpg,:] = OP_Dictionary[key]
                    idpg += 1 
              if sd_coarse is not None:
                coarseflag = coarse_keyname
                coarse_binbounds = coarse_bins[coarse_keyname]
                dpg_coarse_in = dpg_coarse[coarse_keyname][coarse_binbounds]
                dpu_coarse_in = dpu_coarse[coarse_keyname][coarse_binbounds]
                dpl_coarse_in = dpl_coarse[coarse_keyname][coarse_binbounds]
                dlogdp = np.log10(dpu_coarse[coarse_keyname])-np.log10(dpl_coarse[coarse_keyname])
                ddp = dpu_coarse[coarse_keyname]-dpl_coarse[coarse_keyname]
                dpg_coarse_full = dpg_coarse[coarse_keyname]
                if coarse_keyname == "FCDP":
                  sd_n_coarse = sd_coarse*10**(-6)          
                else:
                  sd_n_coarse = sd_coarse
                if camp_name == "PACEPAX":
                  sd_n_coarse = sd_n_coarse * dpg_coarse_full[:,None] * np.log(10)
                  

        rp_interp = np.array(LUT_output_variables_fine[0]['radii_grid_bins']).astype(float) 
        dNdlnr_w_interp = np.full((LUT_output_variables_fine[0]['num_radii_grid_bins'],dataset_length),np.nan).astype(float)
        dp_grid = rp_interp*2
        dp_grid_edges = sdm.edges_from_mids_geometric(dp_grid).astype(float) 
        for i1 in range(dataset_length):
          sd_n_fine = sd[:,i1]
          dp_fine = gf[i1] * dD_g[:,i1]
          valid_data = np.where((np.logical_not(np.isnan(dp_fine))&np.logical_not(np.isnan(sd_n_fine))))[0]
          if len(valid_data)>2:
            sd_n = sd_n_fine[valid_data]
            dpg_edges = sdm.edges_from_mids_geometric(dp_fine[valid_data])
            target_sd_n = sdm.rebin_dndlog_by_edges_overlap(dpg_edges * 1000, dp_grid_edges * 1000, sd_n)
            target_sd_n[np.isnan(target_sd_n)] = 0
            dNdlnr_fine_amb = target_sd_n
            fine_bounds = np.where((rp_interp<LBcutoff/2))[0]
          valid_data2 = np.where((np.logical_not(np.isnan(sd_n_coarse[:,i1]))))[0]
          if len(valid_data2)>2:
            sd_n = sd_n_coarse[valid_data2,i1]
            dp_coarse = dpg_coarse_full[valid_data2]/2
            dpg_edges = sdm.edges_from_mids_geometric(dp_coarse)
            target_sd_n = sdm.rebin_dndlog_by_edges_overlap(dpg_edges * 1000, dp_grid_edges * 1000, sd_n)
            target_sd_n[np.isnan(target_sd_n)] = 0
            dNdlnr_coarse_amb = target_sd_n
            course_bounds = np.where((rp_interp>=LBcutoff/2))[0]
          if (len(valid_data)>=2) & (len(valid_data2)>=2):
            dNdlnr_w_interp[fine_bounds,i1] = dNdlnr_fine_amb[fine_bounds]
            dNdlnr_w_interp[course_bounds,i1] = dNdlnr_coarse_amb[course_bounds] 
          elif (len(valid_data)>=2) & (len(valid_data2<2)):  
            dNdlnr_w_interp[:,i1] = dNdlnr_fine_amb
          #elif (len(valid_data)==0) & (len(valid_data2)>2):   
          #  dNdlnr_w_interp[:,i1] = dNdlnr_coarse_amb
          #  RRIf[:,i1] = RRIc[:,i1]
          #  IRIf[:,i1] = IRIc[:,i1]
        dAdlnr_w_interp = (4*np.pi)*((rp_interp[:,None])**2)*dNdlnr_w_interp
        dVdlnr_w_interp = (4*np.pi/3)*((rp_interp[:,None])**3)*dNdlnr_w_interp     
        rri = dict()
        iri = dict()
        rho = dict()
        LUT = dict()
        shape_dist = dict() 
        for key in mode:
          rri[key] = dict()
          iri[key] = dict()
          rho[key] = dict()
          LUT[key] = dict()
          shape_dist[key] = dict()
        good_vals = np.where((RRIf[1,:]>0))[0]  
        if (len(good_vals) >0):
          rri["Coarse"] = RRIc#np.squeeze(np.full(length_data,rric))
          iri["Coarse"] = IRIc#np.squeeze(np.full(length_data,iric))#
          rho["Coarse"] = np.ones(length_data)
          LUT["Coarse"] = LUT_output_variables_coarse
          shape_dist["Coarse"] = coarse_shap_dist_ary
          rri["Nucl"] = RRIf
          iri["Nucl"] = IRIf
          rho["Nucl"] = rhof
          LUT["Nucl"] = LUT_output_variables_fine
          shape_dist["Nucl"] = fine_shap_dist_ary
          rri["Accu"] = RRIf
          iri["Accu"] = IRIf
          rho["Accu"] = rhof
          LUT["Accu"] = LUT_output_variables_fine
          shape_dist["Accu"] = fine_shap_dist_ary            
          AmbPrps_mode = calprps(dNdlnr_w_interp,dAdlnr_w_interp,dVdlnr_w_interp,rri,iri,rho,LUT,shape_dist,wvl,mode,mode_sizes)
          AmbPrps = {}
          for i_bulk in bulk_names:
            AmbPrps[i_bulk] = {}
            if len(bulk_bounds[i_bulk]) == 1:
              lim = bulk_bounds[i_bulk][0]
              AmbPrps[i_bulk]["amb_N_cm-3"] = AmbPrps_mode["amb_N_cm-3"][lim,...]
              AmbPrps[i_bulk]["amb_A_um2.cm-3"] = AmbPrps_mode["amb_A_um2.cm-3"][lim,...]
              AmbPrps[i_bulk]["amb_V_um3.cm-3"] = AmbPrps_mode["amb_V_um3.cm-3"][lim,...]
              AmbPrps[i_bulk]["amb_M_g.cm-3"] = AmbPrps_mode["amb_M_g.cm-3"][lim,...]
              AmbPrps[i_bulk]["amb_r_eff_um"] = AmbPrps_mode["amb_r_eff_um"][lim,...]
              AmbPrps[i_bulk]["amb_v_eff_unitless"] = AmbPrps_mode["amb_v_eff_unitless"][lim,...]
              AmbPrps[i_bulk]['amb_ext_coef_Mm-1'] = AmbPrps_mode['amb_ext_coef_Mm-1'][lim,...]
              AmbPrps[i_bulk]['amb_ssa_unitless'] = AmbPrps_mode['amb_ssa_unitless'][lim,...]
              AmbPrps[i_bulk]['amb_asym_unitless'] = AmbPrps_mode['amb_asym_unitless'][lim,...]
              AmbPrps[i_bulk]['amb_RRI_unitless'] = AmbPrps_mode['amb_RRI_unitless'][lim,...]
              AmbPrps[i_bulk]['amb_IRI_unitless'] = AmbPrps_mode['amb_IRI_unitless'][lim,...]
              AmbPrps[i_bulk]['amb_back_coef_Mm-1.sr-1'] = AmbPrps_mode['amb_back_coef_Mm-1.sr-1'][lim,...]
              AmbPrps[i_bulk]['amb_lidar_ratio_sr'] = AmbPrps_mode['amb_lidar_ratio_sr'][lim,...]
              AmbPrps[i_bulk]['amb_LDR_unitless'] = AmbPrps_mode['amb_LDR_unitless'][lim,...]
            else:
              lim = bulk_bounds[i_bulk]
              AmbPrps[i_bulk]["amb_N_cm-3"] = np.nansum(AmbPrps_mode["amb_N_cm-3"][lim,...],axis=0)
              AmbPrps[i_bulk]["amb_A_um2.cm-3"] = np.nansum(AmbPrps_mode["amb_A_um2.cm-3"][lim,...],axis=0)
              AmbPrps[i_bulk]["amb_V_um3.cm-3"] = np.nansum(AmbPrps_mode["amb_V_um3.cm-3"][lim,...],axis=0)
              AmbPrps[i_bulk]["amb_r_eff_um"] = 3*AmbPrps[i_bulk]["amb_V_um3.cm-3"] / AmbPrps[i_bulk]["amb_A_um2.cm-3"] 
              AmbPrps[i_bulk]["amb_M_g.cm-3"] = np.nansum(AmbPrps_mode["amb_M_g.cm-3"][lim,...],axis=0)     
              AmbPrps[i_bulk]['amb_ext_coef_Mm-1'] = np.nansum(AmbPrps_mode['amb_ext_coef_Mm-1'][lim,...],axis=0)
              AmbPrps[i_bulk]['amb_back_coef_Mm-1.sr-1'] = np.nansum(AmbPrps_mode['amb_back_coef_Mm-1.sr-1'][lim,...],axis=0)
              sca_cf =  np.nansum(AmbPrps_mode['amb_sca_coef_Mm-1'][lim,...],axis=0)
              AmbPrps[i_bulk]['amb_lidar_ratio_sr'] = AmbPrps[i_bulk]['amb_ext_coef_Mm-1'] / AmbPrps[i_bulk]['amb_back_coef_Mm-1.sr-1']
              AmbPrps[i_bulk]["amb_v_eff_unitless"] = (AmbPrps_mode["amb_v_eff_unitless"][lim[0],...]+1) * (AmbPrps_mode["amb_A_um2.cm-3"][lim[0],...]) / (AmbPrps_mode["amb_r_eff_um"][lim[0],...]**2)
              AmbPrps[i_bulk]['amb_asym_unitless'] = AmbPrps_mode['amb_asym_unitless'][lim[0],...] * AmbPrps_mode['amb_sca_coef_Mm-1'][lim[0],...]
              AmbPrps[i_bulk]['amb_RRI_unitless'] = AmbPrps_mode['amb_RRI_unitless'][lim[0],...] * AmbPrps_mode["amb_V_um3.cm-3"][lim[0],...]
              AmbPrps[i_bulk]['amb_IRI_unitless'] = AmbPrps_mode['amb_IRI_unitless'][lim[0],...] * AmbPrps_mode["amb_V_um3.cm-3"][lim[0],...]
              AmbPrps[i_bulk]['amb_LDR_unitless'] = AmbPrps_mode['amb_LDR_unitless'][lim[0],...] * AmbPrps_mode['amb_back_coef_Mm-1.sr-1'][lim[0],...]
              for i_mode in lim[1:]:
                AmbPrps[i_bulk]["amb_v_eff_unitless"] += (AmbPrps_mode["amb_v_eff_unitless"][i_mode,...]+1) * (AmbPrps_mode["amb_A_um2.cm-3"][i_mode,...]) / (AmbPrps_mode["amb_r_eff_um"][i_mode,...]**2)
                AmbPrps[i_bulk]['amb_asym_unitless'] += AmbPrps_mode['amb_asym_unitless'][i_mode,...] * AmbPrps_mode['amb_sca_coef_Mm-1'][i_mode,...]
                AmbPrps[i_bulk]['amb_RRI_unitless'] += AmbPrps_mode['amb_RRI_unitless'][i_mode,...] * AmbPrps_mode["amb_V_um3.cm-3"][i_mode,...]
                AmbPrps[i_bulk]['amb_IRI_unitless'] += AmbPrps_mode['amb_IRI_unitless'][i_mode,...] * AmbPrps_mode["amb_V_um3.cm-3"][i_mode,...]
                AmbPrps[i_bulk]['amb_LDR_unitless'] += AmbPrps_mode['amb_LDR_unitless'][i_mode,...] * AmbPrps_mode['amb_back_coef_Mm-1.sr-1'][i_mode,...]
              AmbPrps[i_bulk]["amb_v_eff_unitless"] *= (AmbPrps[i_bulk]["amb_r_eff_um"]**2)/AmbPrps[i_bulk]["amb_A_um2.cm-3"]
              AmbPrps[i_bulk]["amb_v_eff_unitless"] += -1  
              AmbPrps[i_bulk]['amb_asym_unitless'] /= sca_cf
              AmbPrps[i_bulk]['amb_RRI_unitless'] /= AmbPrps[i_bulk]["amb_V_um3.cm-3"]
              AmbPrps[i_bulk]['amb_IRI_unitless'] /= AmbPrps[i_bulk]["amb_V_um3.cm-3"]
              AmbPrps[i_bulk]['amb_LDR_unitless'] /= AmbPrps[i_bulk]['amb_back_coef_Mm-1.sr-1']
              AmbPrps[i_bulk]['amb_ssa_unitless'] = sca_cf / AmbPrps[i_bulk]['amb_ext_coef_Mm-1']  
            for varkey in AmbPrps[i_bulk]:
                AmbPrps[i_bulk][varkey][AmbPrps[i_bulk][varkey] == 0] = np.nan
          GlobParams = dict()
          GlobParams['conventions']='CF-1.9'
          GlobParams['data_product_group'] = 'derived'
          GlobParams['data_use_guideline'] = 'N/A'
          GlobParams['file_originator'] = 'Joseph Schlosser'
          GlobParams['file_originator_contact'] = 'joseph.s.schlosser@nasa.gov'
          if flight_number.startswith("L"):
            GlobParams['flight_number_day'] = flight_number
          GlobParams['flight_start_date'] = Date
          GlobParams['format'] = 'NETCDF4'
          GlobParams['history'] = f'Ambient aerosol properties derived using ISARA from various in-situ measurements. {update_info}'
          GlobParams['IdentifierProductDOI'] = prodDOI
          GlobParams['institution'] = 'ORAU-NASA Langley'
          GlobParams['last_modified_date'] = datetime.today().strftime('%Y-%m-%d')
          GlobParams['ACVSNC_standard_name_URL'] = 'https://www-air.larc.nasa.gov/missions/etc/AtmosphericCompositionVariableStandardNames.pdf'
          GlobParams['ACVSNC_standard_name_version'] = '1.0'
          GlobParams['measurement_platform'] = OP_Dictionary['GlobalAttributes']['measurement_platform']
          GlobParams['PI_contact'] = 'joseph.s.schlosser@nasa.gov'
          GlobParams['PI_name'] = 'Joseph Schlosser'
          GlobParams['platform_identifier'] = reference_platform
          GlobParams['ProcessingLevel'] = '4'
          GlobParams['project'] = 'In-Situ Aerosol Retrieval Algorithm (ISARA)'
          GlobParams['references'] = '10.5194/egusphere-2025-3088'
          GlobParams['source'] = f'Derived using the methods outlined in 10.5194/egusphere-2025-3088. The {LUT_f} and {LUT_c} files were used for the fine- and coarse-mode aerosol, respectively.'
          GlobParams['time_coverage_end'] = fmtdatetime_Stop[0].astype(str)
          GlobParams['time_coverage_resolution'] = resolution
          GlobParams['time_coverage_start'] = fmtdatetime_Start[-1].astype(str)
          GlobParams['title'] = 'ISARA-derived Ambient Aerosol Properties and Originating In-situ Measurements'
          GlobParams['VersionID'] = f'R{revision_number}'
          for bulkname in bulk_names:
            if len(AmbPrps[bulkname]) >0:
              for key in AmbPrps[bulkname]:
                kyname = f'{bulk_shortnames[bulkname]}_{key}'
                OP_Dictionary[kyname] = AmbPrps[bulkname][key].T  
                OP_Dictionary['SourceFlag'][kyname] = 'derived'
                if kyname in OP_Dictionary['VariableAttributes']:
                  OP_Dictionary['VariableAttributes'][kyname]['_FillValue'] = np.nan
                else:
                  OP_Dictionary['VariableAttributes'][kyname] = {}
                  OP_Dictionary['VariableAttributes'][kyname]['_FillValue'] = np.nan
          dims = {}
          dims['time'] = len(utc_str_arr)
          OP_Dictionary["time"] = Time_Mid
          OP_Dictionary['SourceFlag']['time'] = ''
          OP_Dictionary['VariableAttributes']["time"] = {}
          OP_Dictionary['VariableAttributes']["time"]['short_name'] = 'time'
          OP_Dictionary['VariableAttributes']["time"]['units'] = f'seconds after {Date} 00:00:00 UTC.'
          OP_Dictionary['VariableAttributes']["time"]['long_name'] = f'(Time_Start+Time_Stop)/2.'
          OP_Dictionary['VariableAttributes']["time"]['time_bnds'] ='Time_Start, Time_Stop'    
          dims['wavelength'] = len(wvl)
          OP_Dictionary["wavelength"] = wvl
          OP_Dictionary['SourceFlag']['wavelength'] = ''
          OP_Dictionary['VariableAttributes']["wavelength"] = {}
          OP_Dictionary['VariableAttributes']["wavelength"]['short_name'] = 'wavelength'
          OP_Dictionary['VariableAttributes']["wavelength"]['units'] = 'nm'
          OP_Dictionary['VariableAttributes']["wavelength"]['long_name'] = 'Wavelength bands in nm that are associated with the wavelength dimension.'   
          dims['bin'] = fine_bin_length
          OP_Dictionary["bin"] = np.arange(fine_bin_length)
          OP_Dictionary['SourceFlag']['bin'] = ''
          OP_Dictionary['VariableAttributes']["bin"] = {}
          OP_Dictionary['VariableAttributes']["bin"]['short_name'] = 'bin'
          OP_Dictionary['VariableAttributes']["bin"]['units'] = '1'
          OP_Dictionary['VariableAttributes']["bin"]['long_name'] = f'Bin number corresponding to the DRY and AMBIENT particle size bins.'
          OP_Dictionary['VariableAttributes']["bin"]['ancillary'] = f'dry_lower_cutoff_diameter, dry_geometric_mean_diameter, dry_upper_cutoff_diameter, amb_lower_cutoff_diameter, amb_geometric_mean_diameter, amb_upper_cutoff_diameter'
          OP_Dictionary["kappa-550_unitless"] = kappa
          OP_Dictionary['Dims']['kappa-550_unitless'] = 'time'
          OP_Dictionary['SourceFlag']['kappa-550_unitless'] = 'derived'
          OP_Dictionary['VariableAttributes']["kappa-550_unitless"] = {}
          OP_Dictionary['VariableAttributes']["kappa-550_unitless"]['short_name'] = 'fine_kappa'
          OP_Dictionary['VariableAttributes']["kappa-550_unitless"]['units'] = '1'
          OP_Dictionary['VariableAttributes']["kappa-550_unitless"]['long_name'] = f'Hygroscopicity (kappa) derieved from ISARA for the fine-mode aerosol'
          OP_Dictionary['VariableAttributes']["kappa-550_unitless"]['ACVSNC_standard_name'] = f'none' 
          if (coarsemode == "yes") & (coarseflag is not None):
            dims['coarse-diameter'] = len(dpg_coarse_full)
            OP_Dictionary["coarse-diameter"] = dpg_coarse_full
            OP_Dictionary['SourceFlag']['coarse-diameter'] = ''
            OP_Dictionary['VariableAttributes']["coarse-diameter"] = {}
            OP_Dictionary['VariableAttributes']["coarse-diameter"]['short_name'] = 'coarse-diameter'
            OP_Dictionary['VariableAttributes']["coarse-diameter"]['units'] = '1'
            OP_Dictionary['VariableAttributes']["coarse-diameter"]['long_name'] = f'Geometric mean diameter at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag}.'
            OP_Dictionary['VariableAttributes']["coarse-diameter"]['coarse_bin_diameter_bnds'] = f'coarse_lower_cutoff_diameter, coarse_upper_cutoff_diameter'   
            OP_Dictionary["coarse_lower_cutoff_diameter_um"] = dpl_coarse[coarseflag]
            OP_Dictionary['Dims']['coarse_lower_cutoff_diameter_um'] = 'coarse-diameter'
            OP_Dictionary['SourceFlag']['coarse_lower_cutoff_diameter_um'] = 'derived'
            OP_Dictionary['VariableAttributes']["coarse_lower_cutoff_diameter_um"] = {}
            OP_Dictionary['VariableAttributes']["coarse_lower_cutoff_diameter_um"]['short_name'] = 'coarse_lower_cutoff_diameter'
            OP_Dictionary['VariableAttributes']["coarse_lower_cutoff_diameter_um"]['units'] = 'um'
            OP_Dictionary['VariableAttributes']["coarse_lower_cutoff_diameter_um"]['long_name'] = f'Lower cutoff diameter at AMBIENT relative humidity of each particle bin derived from {coarseflag}.'
            OP_Dictionary['VariableAttributes']["coarse_lower_cutoff_diameter_um"]['ACVSNC_standard_name'] = 'none'   
            OP_Dictionary["coarse_upper_cutoff_diameter_um"] = dpu_coarse[coarseflag]
            OP_Dictionary['Dims']['coarse_upper_cutoff_diameter_um'] = 'coarse-diameter'
            OP_Dictionary['SourceFlag']['coarse_upper_cutoff_diameter_um'] = 'derived'
            OP_Dictionary['VariableAttributes']["coarse_upper_cutoff_diameter_um"] = {}
            OP_Dictionary['VariableAttributes']["coarse_upper_cutoff_diameter_um"]['short_name'] = 'coarse_upper_cutoff_diameter'
            OP_Dictionary['VariableAttributes']["coarse_upper_cutoff_diameter_um"]['units'] = 'um'
            OP_Dictionary['VariableAttributes']["coarse_upper_cutoff_diameter_um"]['long_name'] = f'upper cutoff diameter at AMBIENT relative humidity of each particle bin derived from {coarseflag}.'
            OP_Dictionary['VariableAttributes']["coarse_upper_cutoff_diameter_um"]['ACVSNC_standard_name'] = 'none'  
            OP_Dictionary["coarse_dndlogdp_cm-3"] = (sd_n_coarse.T)
            OP_Dictionary['Dims']['coarse_dndlogdp_cm-3'] = np.array(['time','coarse-diameter'])
            OP_Dictionary['SourceFlag']['coarse_dndlogdp_cm-3'] = 'derived'
            OP_Dictionary['VariableAttributes']["coarse_dndlogdp_cm-3"] = {}
            OP_Dictionary['VariableAttributes']["coarse_dndlogdp_cm-3"]['short_name'] = 'coarse_dndlogdp'
            OP_Dictionary['VariableAttributes']["coarse_dndlogdp_cm-3"]['units'] = 'cm-3'
            OP_Dictionary['VariableAttributes']["coarse_dndlogdp_cm-3"]['long_name'] = f'Aerosol number size distribution (dN/dlog[Dp]) at AMBIENT temperature and pressure derived from {coarseflag}.'
            OP_Dictionary['VariableAttributes']["coarse_dndlogdp_cm-3"]['ACVSNC_standard_name'] = 'AerMP_SizeDist_InSitu_Optical_Coarse_AMB' 
            OP_Dictionary['VariableAttributes']["coarse_dndlogdp_cm-3"]['ancillary'] = f'coarse_lower_cutoff_diameter, coarse_upper_cutoff_diameter'
            OP_Dictionary['VariableAttributes']["coarse_dndlogdp_cm-3"]['comments'] = f'Taken directly from the {coarseflag} source data.'
          #OP_Dictionary["angle"] = np.nanmean(OP_Dictionary["angle"],0)
          #dims['angle'] = len(OP_Dictionary["angle"])
          #OP_Dictionary['SourceFlag']["angle"] = ''
          #OP_Dictionary['VariableAttributes']["angle"] = {}
          #OP_Dictionary['VariableAttributes']["angle"]['short_name'] = 'angle'
          #OP_Dictionary['VariableAttributes']["angle"]['units'] = 'degree'
          #OP_Dictionary['VariableAttributes']["angle"]['long_name'] = 'Scattering angle.'
          #OP_Dictionary[f'{bulk_shrtname}_amb_RRI_unitless'] = RRIf
          #OP_Dictionary[f'{bulk_shrtname}_amb_IRI_unitless'] = IRIf    
          OP_Dictionary["dry_lower_cutoff_diameter_um"] = dD_l.T
          OP_Dictionary['Dims']['dry_lower_cutoff_diameter_um'] = np.array(['time','bin'])
          OP_Dictionary['SourceFlag']['dry_lower_cutoff_diameter_um'] = 'derived'
          OP_Dictionary['VariableAttributes']["dry_lower_cutoff_diameter_um"] = {}
          OP_Dictionary['VariableAttributes']["dry_lower_cutoff_diameter_um"]['short_name'] = 'dry_lower_cutoff_diameter'
          OP_Dictionary['VariableAttributes']["dry_lower_cutoff_diameter_um"]['units'] = 'um'
          OP_Dictionary['VariableAttributes']["dry_lower_cutoff_diameter_um"]['long_name'] = 'Lower cutoff diameter at DRY relative humidity of each particle bin derived from ISARA.'
          OP_Dictionary['VariableAttributes']["dry_lower_cutoff_diameter_um"]['ACVSNC_standard_name'] = 'none'              
          OP_Dictionary["dry_geometric_mean_diameter_um"] = dD_g.T
          OP_Dictionary['Dims']['dry_geometric_mean_diameter_um'] = np.array(['time','bin'])
          OP_Dictionary['SourceFlag']['dry_geometric_mean_diameter_um'] = 'derived'
          OP_Dictionary['VariableAttributes']["dry_geometric_mean_diameter_um"] = {}
          OP_Dictionary['VariableAttributes']["dry_geometric_mean_diameter_um"]['short_name'] = 'dry_geometric_mean_diameter'
          OP_Dictionary['VariableAttributes']["dry_geometric_mean_diameter_um"]['units'] = 'um'
          OP_Dictionary['VariableAttributes']["dry_geometric_mean_diameter_um"]['long_name'] = 'Geometric mean diameter at DRY relative humidity of each particle bin derived from ISARA.'
          OP_Dictionary['VariableAttributes']["dry_geometric_mean_diameter_um"]['ACVSNC_standard_name'] = 'AerMP_MedianSize_InSitu_RHd_Optical_NucltoCoarse_None'     
          OP_Dictionary["dry_upper_cutoff_diameter_um"] = dD_u.T
          OP_Dictionary['Dims']['dry_upper_cutoff_diameter_um'] = np.array(['time','bin'])
          OP_Dictionary['SourceFlag']['dry_upper_cutoff_diameter_um'] = 'derived'
          OP_Dictionary['VariableAttributes']["dry_upper_cutoff_diameter_um"] = {}
          OP_Dictionary['VariableAttributes']["dry_upper_cutoff_diameter_um"]['short_name'] = 'dry_upper_cutoff_diameter'
          OP_Dictionary['VariableAttributes']["dry_upper_cutoff_diameter_um"]['units'] = 'um'
          OP_Dictionary['VariableAttributes']["dry_upper_cutoff_diameter_um"]['long_name'] = 'Upper cutoff diameter at DRY relative humidity of each particle bin derived from ISARA.'
          OP_Dictionary['VariableAttributes']["dry_upper_cutoff_diameter_um"]['ACVSNC_standard_name'] = 'none'          
          OP_Dictionary["amb_lower_cutoff_diameter_um"] = D_l.T
          OP_Dictionary['Dims']['amb_lower_cutoff_diameter_um'] = np.array(['time','bin'])
          OP_Dictionary['SourceFlag']['amb_lower_cutoff_diameter_um'] = 'derived'
          OP_Dictionary['VariableAttributes']["amb_lower_cutoff_diameter_um"] = {}
          OP_Dictionary['VariableAttributes']["amb_lower_cutoff_diameter_um"]['short_name'] = 'amb_lower_cutoff_diameter'
          OP_Dictionary['VariableAttributes']["amb_lower_cutoff_diameter_um"]['units'] = 'um'
          OP_Dictionary['VariableAttributes']["amb_lower_cutoff_diameter_um"]['long_name'] = 'Lower cutoff diameter at AMBIENT relative humidity of each particle bin derived from ISARA.'
          OP_Dictionary['VariableAttributes']["amb_lower_cutoff_diameter_um"]['ACVSNC_standard_name'] = 'none'              
          OP_Dictionary["amb_geometric_mean_diameter_um"] = D_g.T
          OP_Dictionary['Dims']['amb_geometric_mean_diameter_um'] = np.array(['time','bin'])
          OP_Dictionary['SourceFlag']['amb_geometric_mean_diameter_um'] = 'derived'
          OP_Dictionary['VariableAttributes']["amb_geometric_mean_diameter_um"] = {}
          OP_Dictionary['VariableAttributes']["amb_geometric_mean_diameter_um"]['short_name'] = 'amb_geometric_mean_diameter'
          OP_Dictionary['VariableAttributes']["amb_geometric_mean_diameter_um"]['units'] = 'um'
          OP_Dictionary['VariableAttributes']["amb_geometric_mean_diameter_um"]['long_name'] = 'Geometric mean diameter at AMBIENT relative humidity of each particle bin derived from ISARA.'
          OP_Dictionary['VariableAttributes']["amb_geometric_mean_diameter_um"]['ACVSNC_standard_name'] = 'AerMP_MedianSize_InSitu_RHa_Optical_NucltoCoarse_None'     
          OP_Dictionary["amb_upper_cutoff_diameter_um"] = D_u.T
          OP_Dictionary['Dims']['amb_upper_cutoff_diameter_um'] = np.array(['time','bin'])
          OP_Dictionary['SourceFlag']['amb_upper_cutoff_diameter_um'] = 'derived'
          OP_Dictionary['VariableAttributes']["amb_upper_cutoff_diameter_um"] = {}
          OP_Dictionary['VariableAttributes']["amb_upper_cutoff_diameter_um"]['short_name'] = 'amb_upper_cutoff_diameter'
          OP_Dictionary['VariableAttributes']["amb_upper_cutoff_diameter_um"]['units'] = 'um'
          OP_Dictionary['VariableAttributes']["amb_upper_cutoff_diameter_um"]['long_name'] = 'Upper cutoff diameter at AMBIENT relative humidity of each particle bin derived from ISARA.'
          OP_Dictionary['VariableAttributes']["amb_upper_cutoff_diameter_um"]['ACVSNC_standard_name'] = 'none'          
          OP_Dictionary["dndlogdp_cm-3"] = sd.T
          OP_Dictionary['Dims']['dndlogdp_cm-3'] = np.array(['time','bin'])
          OP_Dictionary['SourceFlag']['dndlogdp_cm-3'] = 'derived'
          OP_Dictionary['VariableAttributes']["dndlogdp_cm-3"] = {}
          OP_Dictionary['VariableAttributes']["dndlogdp_cm-3"]['short_name'] = 'dndlogdp'
          OP_Dictionary['VariableAttributes']["dndlogdp_cm-3"]['units'] = 'cm-3'
          OP_Dictionary['VariableAttributes']["dndlogdp_cm-3"]['long_name'] = 'Aerosol number size distribution (dN/dlog[Dp]) at AMBIENT temperature and pressure derived from ISARA.'
          OP_Dictionary['VariableAttributes']["dndlogdp_cm-3"]['ACVSNC_standard_name'] = 'AerMP_NumSizeDist_InSitu_None_Optical_NucltoCoarse_AMB' 
          OP_Dictionary['VariableAttributes']["dndlogdp_cm-3"]['ancillary'] = f'dry_lower_cutoff_diameter, dry_geometric_mean_diameter, dry_upper_cutoff_diameter, amb_lower_cutoff_diameter, amb_geometric_mean_diameter, amb_upper_cutoff_diameter'
          OP_Dictionary['VariableAttributes']["dndlogdp_cm-3"]['comments'] = 'dndlogdp is linked to the ancillary variables via the bin dimension. The upper and lower cutoff diameters provide the bin bounds and the geometric median diameters are for each bin. The dry diameters are corrected for particle effective density. The ambient diameters are corrected for ambient relative humidity'
          ik = 0
          for k in AmbPrps: 
            if len(AmbPrps[k]) >0:
              bulk_shrtname = bulk_shortnames[k]          
              if (coarsemode != "yes"):
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_RRI_unitless'] = np.array(['time','wavelength'])
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless'] = {}
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['short_name'] = f'{bulk_shrtname}_amb_RRI'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['units'] = '1'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['long_name'] = f'Real refractive index of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['ACVSNC_standard_name'] = f'AerOpt_m_InSitu_Green_RHa_{bulk_shrtname}_None'
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_IRI_unitless'] = np.array(['time','wavelength'])
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless'] = {}
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['short_name'] = f'{bulk_shrtname}_amb_IRI'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['units'] = '1'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['long_name'] = f'Imaginary refractive index of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['ACVSNC_standard_name'] = f'AerOpt_k_InSitu_Green_RHa_{bulk_shrtname}_None' 
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_r_eff_um'] = 'time'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['short_name'] = f'{bulk_shrtname}_amb_r_eff'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['units'] = 'um'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['long_name'] = f'Effective radius of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['ACVSNC_standard_name'] = f'AerMP_EffSize_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_v_eff_unitless'] = 'time'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['short_name'] = f'{bulk_shrtname}_amb_v_eff'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['units'] = '1'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['long_name'] = f'Effective variance of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['ACVSNC_standard_name'] = f'AerMP_EffVar_InSitu_RHa_Optical_{bulk_shrtname}_AMB'               
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_N_cm-3'] = 'time'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['short_name'] = f'{bulk_shrtname}_amb_N'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['units'] = 'cm-3'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['long_name'] = f'Number concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['ACVSNC_standard_name'] = f'AerMP_NumConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_A_um2.cm-3'] = 'time'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['short_name'] = f'{bulk_shrtname}_amb_A'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['units'] = 'um2.cm-1'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['long_name'] = f'Surface area concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['ACVSNC_standard_name'] = f'AerMP_SurfAreaConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_V_um3.cm-3'] = 'time'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['short_name'] = f'{bulk_shrtname}_amb_V'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['units'] = 'um3.cm-3'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['long_name'] = f'Volume concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['ACVSNC_standard_name'] = f'AerMP_VolConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_M_g.cm-3'] = 'time' 
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['short_name'] = f'{bulk_shrtname}_amb_M'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['units'] = 'g.cm-3'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['long_name'] = f'Mass concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['ACVSNC_standard_name'] = f'AerMP_MassConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_ext_coef_Mm-1'] = np.array(['time','wavelength'])                                               
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['short_name'] = f'{bulk_shrtname}_amb_ext_coef'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['units'] = 'Mm-1'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['long_name'] = f'Extinction coefficient at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.' 
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_BluetoRed_RHa_{bulk_shrtname}_AMB'
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_ssa_unitless'] = np.array(['time','wavelength'])   
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['short_name'] = f'{bulk_shrtname}_amb_ssa'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['units'] = '1'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['long_name'] = f'Single scattering albedo at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from ISARA.' 
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_BluetoRed_RHa_{bulk_shrtname}_None' 
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_asym_unitless'] = np.array(['time','wavelength'])   
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['short_name'] = f'{bulk_shrtname}_amb_asym'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['units'] = '1'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['long_name'] = f'Asymmetry parameter at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from ISARA.' 
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['ACVSNC_standard_name'] = f'AerOpt_AsymmetryParameterScat_InSitu_BluetoRed_RHa_{bulk_shrtname}_None'
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1'] = np.array(['time','wavelength'])
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['short_name'] = f'{bulk_shrtname}_amb_back_coef'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['units'] = 'Mm-1.sr-1'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['long_name'] = f'Backscattering coefficient at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'   
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['ACVSNC_standard_name'] = f'AerOpt_BackScattering_InSitu_BluetoRed_RHa_{bulk_shrtname}_AMB'   
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_lidar_ratio_sr'] = np.array(['time','wavelength'])
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['short_name'] = f'{bulk_shrtname}_lidar_ratio'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['units'] = '1'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['long_name'] = f'Lidar ratio at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from ISARA.'  
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['ACVSNC_standard_name'] = f'AerOpt_LidarRatio_InSitu_BluetoRed_RHa_{bulk_shrtname}_None' 
                OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_LDR_unitless'] = np.array(['time','wavelength'])
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['short_name'] = f'{bulk_shrtname}_LDR'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['units'] = '1'
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['long_name'] = f'Linear depolarization ratio at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from ISARA.'     
                OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['ACVSNC_standard_name'] = f'AerOpt_DepolarizationRatio_InSitu_BluetoRed_RHa_{bulk_shrtname}_None'
                #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a1_unitless'] = np.array(['time','wavelength','angle'])                     
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['short_name'] = f'{bulk_shrtname}_amb_a1'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['units'] = '1'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['long_name'] = f'Scattering matrix component a1 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['ACVSNC_standard_name'] = 'none' 
                #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a2_unitless'] = np.array(['time','wavelength','angle'])    
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['short_name'] = f'{bulk_shrtname}_amb_a2'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['units'] = '1'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['long_name'] = f'Scattering matrix component a2 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['ACVSNC_standard_name'] = 'none' 
                #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a3_unitless'] = np.array(['time','wavelength','angle'])    
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['short_name'] = f'{bulk_shrtname}_amb_a3'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['units'] = '1'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['long_name'] = f'Scattering matrix component a3 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['ACVSNC_standard_name'] = 'none' 
                #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a4_unitless'] = np.array(['time','wavelength','angle'])    
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['short_name'] = f'{bulk_shrtname}_amb_a4'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['units'] = '1'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['long_name'] = f'Scattering matrix component a4 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['ACVSNC_standard_name'] = 'none' 
                #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_b1_unitless'] = np.array(['time','wavelength','angle'])    
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['short_name'] = f'{bulk_shrtname}_amb_b1'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['units'] = '1'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['long_name'] = f'Scattering matrix component b1 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['ACVSNC_standard_name'] = 'none' 
                #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_b2_unitless'] = np.array(['time','wavelength','angle'])    
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['short_name'] = f'{bulk_shrtname}_amb_b2'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['units'] = '1'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['long_name'] = f'Scattering matrix component b2 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['ACVSNC_standard_name'] = 'none' 
                #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a1_vol_unitless'] = np.array(['time','wavelength','angle'])    
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['short_name'] = f'{bulk_shrtname}_amb_a1_vol'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['units'] = '1'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['long_name'] = f'Volume scattering function at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['ACVSNC_standard_name'] = 'none'
              else: 
                if ("coarse" in bulk_shrtname):  
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_RRI_unitless'] =  np.array(['time','wavelength'])
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless'] = {}
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['short_name'] = f'{bulk_shrtname}_amb_RRI'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['long_name'] = f'Real refractive index of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA and volume weighted with coarse-mode CRI={rric}+{iric}i.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['ACVSNC_standard_name'] = f'AerOpt_m_InSitu_Green_RHa_{bulk_shrtname}_None'
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_IRI_unitless'] =  np.array(['time','wavelength'])
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless'] = {}
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['short_name'] = f'{bulk_shrtname}_amb_IRI'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['long_name'] = f'Imaginary refractive index of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA and volume weighted with coarse-mode CRI={rric}+{iric}i.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['ACVSNC_standard_name'] = f'AerOpt_k_InSitu_Green_RHa_{bulk_shrtname}_None'   
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_r_eff_um'] = 'time'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['short_name'] = f'{bulk_shrtname}_amb_r_eff'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['units'] = 'um'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['long_name'] = f'Effective radius of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag}.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['ACVSNC_standard_name'] = f'AerMP_EffSize_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_v_eff_unitless'] = 'time'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['short_name'] = f'{bulk_shrtname}_amb_v_eff'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['long_name'] = f'Effective variance of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag}.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['ACVSNC_standard_name'] = f'AerMP_EffVar_InSitu_RHa_Optical_{bulk_shrtname}_AMB'               
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_N_cm-3'] = 'time'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['short_name'] = f'{bulk_shrtname}_amb_N'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['units'] = 'cm-3'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['long_name'] = f'Number concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag}.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['ACVSNC_standard_name'] = f'AerMP_NumConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_A_um2.cm-3'] = 'time'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['short_name'] = f'{bulk_shrtname}_amb_A'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['units'] = 'um2.cm-3'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['long_name'] = f'Surface area concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag}.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['ACVSNC_standard_name'] = f'AerMP_SurfAreaConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_V_um3.cm-3'] = 'time'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['short_name'] = f'{bulk_shrtname}_amb_V'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['units'] = 'um3.cm-3'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['long_name'] = f'Volume concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag}.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['ACVSNC_standard_name'] = f'AerMP_VolConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_M_g.cm-3'] = 'time' 
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['short_name'] = f'{bulk_shrtname}_amb_M'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['long_name'] = f'Mass concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag}.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['ACVSNC_standard_name'] = f'AerMP_MassConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_ext_coef_Mm-1'] = np.array(['time','wavelength'])                                               
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['short_name'] = f'{bulk_shrtname}_amb_ext_coef'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['units'] = 'Mm-1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['long_name'] = f'Extinction coefficient at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.' 
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_BluetoRed_RHa_{bulk_shrtname}_AMB'
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_ssa_unitless'] = np.array(['time','wavelength'])   
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['short_name'] = f'{bulk_shrtname}_amb_ssa'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['long_name'] = f'Single scattering albedo at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.' 
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_BluetoRed_RHa_{bulk_shrtname}_None' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_asym_unitless'] = np.array(['time','wavelength'])   
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['short_name'] = f'{bulk_shrtname}_amb_asym'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['long_name'] = f'Asymmetry parameter at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.' 
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['ACVSNC_standard_name'] = f'AerOpt_AsymmetryParameterScat_InSitu_BluetoRed_RHa_{bulk_shrtname}_None'
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1'] = np.array(['time','wavelength'])
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['short_name'] = f'{bulk_shrtname}_amb_back_coef'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['units'] = 'Mm-1.sr-1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['long_name'] = f'Backscattering coefficient at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.'   
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['ACVSNC_standard_name'] = f'AerOpt_BackScattering_InSitu_BluetoRed_RHa_{bulk_shrtname}_AMB'   
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_lidar_ratio_sr'] = np.array(['time','wavelength'])
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['short_name'] = f'{bulk_shrtname}_lidar_ratio'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['long_name'] = f'Lidar ratio at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.'  
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['ACVSNC_standard_name'] = f'AerOpt_LidarRatio_InSitu_BluetoRed_RHa_{bulk_shrtname}_None' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_LDR_unitless'] = np.array(['time','wavelength'])
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['short_name'] = f'{bulk_shrtname}_LDR'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['long_name'] = f'Linear depolarization ratio at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.'     
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['ACVSNC_standard_name'] = f'AerOpt_DepolarizationRatio_InSitu_BluetoRed_RHa_{bulk_shrtname}_None'
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a1_unitless'] = np.array(['time','wavelength','angle'])                     
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['short_name'] = f'{bulk_shrtname}_amb_a1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['long_name'] = f'Scattering matrix component a1 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a2_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['short_name'] = f'{bulk_shrtname}_amb_a2'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['long_name'] = f'Scattering matrix component a2 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a3_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['short_name'] = f'{bulk_shrtname}_amb_a3'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['long_name'] = f'Scattering matrix component a3 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a4_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['short_name'] = f'{bulk_shrtname}_amb_a4'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['long_name'] = f'Scattering matrix component a4 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_b1_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['short_name'] = f'{bulk_shrtname}_amb_b1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['long_name'] = f'Scattering matrix component b1 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_b2_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['short_name'] = f'{bulk_shrtname}_amb_b2'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['long_name'] = f'Scattering matrix component b2 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a1_vol_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['short_name'] = f'{bulk_shrtname}_amb_a1_vol'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['long_name'] = f'Volume scattering function at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from {coarseflag} assuming coarse-mode CRI={rric}+{iric}i and kappa={kapc}.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['ACVSNC_standard_name'] = 'none'    
                else:
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_RRI_unitless'] =  np.array(['time','wavelength'])
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless'] = {}
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['short_name'] = f'{bulk_shrtname}_amb_RRI'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['long_name'] = f'Real refractive index of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_RRI_unitless']['ACVSNC_standard_name'] = f'AerOpt_m_InSitu_Green_RHa_{bulk_shrtname}_None'
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_IRI_unitless'] =  np.array(['time','wavelength'])
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless'] = {}
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['short_name'] = f'{bulk_shrtname}_amb_IRI'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['long_name'] = f'Imaginary refractive index of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_IRI_unitless']['ACVSNC_standard_name'] = f'AerOpt_k_InSitu_Green_RHa_{bulk_shrtname}_None' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_r_eff_um'] = 'time'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['short_name'] = f'{bulk_shrtname}_amb_r_eff'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['units'] = 'um'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['long_name'] = f'Effective radius of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_r_eff_um']['ACVSNC_standard_name'] = f'AerMP_EffSize_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_v_eff_unitless'] = 'time'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['short_name'] = f'{bulk_shrtname}_amb_v_eff'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['long_name'] = f'Effective variance of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_v_eff_unitless']['ACVSNC_standard_name'] = f'AerMP_EffVar_InSitu_RHa_Optical_{bulk_shrtname}_AMB'               
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_N_cm-3'] = 'time'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['short_name'] = f'{bulk_shrtname}_amb_N'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['units'] = 'cm-3'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['long_name'] = f'Number concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_N_cm-3']['ACVSNC_standard_name'] = f'AerMP_NumConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_A_um2.cm-3'] = 'time'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['short_name'] = f'{bulk_shrtname}_amb_A'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['units'] = 'um2.cm-1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['long_name'] = f'Surface area concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_A_um2.cm-3']['ACVSNC_standard_name'] = f'AerMP_SurfAreaConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_V_um3.cm-3'] = 'time'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['short_name'] = f'{bulk_shrtname}_amb_V'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['units'] = 'um3.cm-3'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['long_name'] = f'Volume concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_V_um3.cm-3']['ACVSNC_standard_name'] = f'AerMP_VolConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_M_g.cm-3'] = 'time' 
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['short_name'] = f'{bulk_shrtname}_amb_M'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['units'] = 'g.cm-3'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['long_name'] = f'Mass concentration of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_M_g.cm-3']['ACVSNC_standard_name'] = f'AerMP_MassConc_InSitu_RHa_Optical_{bulk_shrtname}_AMB' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_ext_coef_Mm-1'] = np.array(['time','wavelength'])                                               
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['short_name'] = f'{bulk_shrtname}_amb_ext_coef'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['units'] = 'Mm-1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['long_name'] = f'Extinction coefficient at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.' 
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ext_coef_Mm-1']['ACVSNC_standard_name'] = f'AerOpt_Extinction_InSitu_BluetoRed_RHa_{bulk_shrtname}_AMB'
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_ssa_unitless'] = np.array(['time','wavelength'])   
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['short_name'] = f'{bulk_shrtname}_amb_ssa'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['long_name'] = f'Single scattering albedo at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from ISARA.' 
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_ssa_unitless']['ACVSNC_standard_name'] = f'AerOpt_SSA_InSitu_BluetoRed_RHa_{bulk_shrtname}_None' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_asym_unitless'] = np.array(['time','wavelength'])   
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['short_name'] = f'{bulk_shrtname}_amb_asym'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['long_name'] = f'Asymmetry parameter at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from ISARA.' 
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_asym_unitless']['ACVSNC_standard_name'] = f'AerOpt_AsymmetryParameterScat_InSitu_BluetoRed_RHa_{bulk_shrtname}_None'
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1'] = np.array(['time','wavelength'])
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['short_name'] = f'{bulk_shrtname}_amb_back_coef'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['units'] = 'Mm-1.sr-1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['long_name'] = f'Backscattering coefficient at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'   
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_back_coef_Mm-1.sr-1']['ACVSNC_standard_name'] = f'AerOpt_BackScattering_InSitu_BluetoRed_RHa_{bulk_shrtname}_AMB'   
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_lidar_ratio_sr'] = np.array(['time','wavelength'])
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['short_name'] = f'{bulk_shrtname}_lidar_ratio'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['long_name'] = f'Lidar ratio at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from ISARA.'  
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_lidar_ratio_sr']['ACVSNC_standard_name'] = f'AerOpt_LidarRatio_InSitu_BluetoRed_RHa_{bulk_shrtname}_None' 
                  OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_LDR_unitless'] = np.array(['time','wavelength'])
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['short_name'] = f'{bulk_shrtname}_LDR'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['units'] = '1'
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['long_name'] = f'Linear depolarization ratio at specified wavelengths from {wvl[0]} to {wvl[-1]} nm of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity derived from ISARA.'     
                  OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_LDR_unitless']['ACVSNC_standard_name'] = f'AerOpt_DepolarizationRatio_InSitu_BluetoRed_RHa_{bulk_shrtname}_None'
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a1_unitless'] = np.array(['time','wavelength','angle'])                     
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['short_name'] = f'{bulk_shrtname}_amb_a1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['long_name'] = f'Scattering matrix component a1 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a2_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['short_name'] = f'{bulk_shrtname}_amb_a2'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['long_name'] = f'Scattering matrix component a2 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a2_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a3_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['short_name'] = f'{bulk_shrtname}_amb_a3'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['long_name'] = f'Scattering matrix component a3 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a3_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a4_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['short_name'] = f'{bulk_shrtname}_amb_a4'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['long_name'] = f'Scattering matrix component a4 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a4_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_b1_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['short_name'] = f'{bulk_shrtname}_amb_b1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['long_name'] = f'Scattering matrix component b1 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b1_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_b2_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['short_name'] = f'{bulk_shrtname}_amb_b2'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['long_name'] = f'Scattering matrix component b2 at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_b2_unitless']['ACVSNC_standard_name'] = 'none' 
                  #OP_Dictionary['Dims'][f'{bulk_shrtname}_amb_a1_vol_unitless'] = np.array(['time','wavelength','angle'])    
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['short_name'] = f'{bulk_shrtname}_amb_a1_vol'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['units'] = '1'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['long_name'] = f'Volume scattering function at specified wavelengths from {wvl[0]} to {wvl[-1]} nm and scattering angles from {OP_Dictionary["angle"][0]} to {OP_Dictionary["angle"][-1]} degree of {bulk_sizes[k]}-{bulk_sizes[k]} um particles at AMBIENT relative humidity, temperature, and pressure derived from ISARA.'
                  #OP_Dictionary['VariableAttributes'][f'{bulk_shrtname}_amb_a1_vol_unitless']['ACVSNC_standard_name'] = 'none'
            ik += 1      
          #OP_Dictionary['VariableAttributes']['amb_AE_unitless'] = {}
          #OP_Dictionary['VariableAttributes']['amb_AE_unitless']['short_name'] = 'amb_AE'
          #OP_Dictionary['VariableAttributes']['amb_AE_unitless']['units'] = '1'
          #OP_Dictionary['VariableAttributes']['amb_AE_unitless']['long_name'] = 'spectral bulk ambient extinction angstrom exponent derived from ISARA.'
          #OP_Dictionary['VariableAttributes']['amb_AE_unitless']['ACVSNC_standard_name'] = 'TBD'
          #OP_Dictionary['VariableAttributes']['amb_sca_angstrom_unitless'] = {}
          #OP_Dictionary['VariableAttributes']['amb_sca_angstrom_unitless']['short_name'] = 'amb_sca_angstrom'
          #OP_Dictionary['VariableAttributes']['amb_sca_angstrom_unitless']['units'] = '1'
          #OP_Dictionary['VariableAttributes']['amb_sca_angstrom_unitless']['long_name'] = 'spectral bulk ambient scattering angstrom exponent derived from ISARA.'
          #OP_Dictionary['VariableAttributes']['amb_sca_angstrom_unitless']['ACVSNC_standard_name'] = 'TBD'
          #OP_Dictionary['VariableAttributes']['amb_abs_angstrom_unitless'] = {}
          #OP_Dictionary['VariableAttributes']['amb_abs_angstrom_unitless']['short_name'] = 'amb_abs_angstrom'
          #OP_Dictionary['VariableAttributes']['amb_abs_angstrom_unitless']['units'] = '1'
          #OP_Dictionary['VariableAttributes']['amb_abs_angstrom_unitless']['long_name'] = 'spectral bulk ambient absorption angstrom exponent derived from ISARA.'
          #OP_Dictionary['VariableAttributes']['amb_abs_angstrom_unitless']['ACVSNC_standard_name'] = 'TBD'
          #OP_Dictionary['VariableAttributes']['amb_back_angstrom_unitless'] = {}
          #OP_Dictionary['VariableAttributes']['amb_back_angstrom_unitless']['short_name'] = 'amb_back_angstrom'
          #OP_Dictionary['VariableAttributes']['amb_back_angstrom_unitless']['units'] = '1'
          #OP_Dictionary['VariableAttributes']['amb_back_angstrom_unitless']['long_name'] = 'spectral bulk ambient backscattering angstrom exponent derived from ISARA.'
          #OP_Dictionary['VariableAttributes']['amb_back_angstrom_unitless']['ACVSNC_standard_name'] = 'TBD'

          
          if flight_number.startswith("L"):
            Output_Filename = f"./{camp_name}/{output_directory}/{output_filename_suffix}_Analysis_{Date}_R{revision_number}_{flight_number}.nc"      
          else:   
            Output_Filename = f"./{camp_name}/{output_directory}/{output_filename_suffix}_Analysis_{Date}_R{revision_number}.nc"       
          ncwrite(Output_Filename, OP_Dictionary, dims, GlobParams) 