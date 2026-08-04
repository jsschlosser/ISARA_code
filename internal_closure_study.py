import numpy as np
import SD_Fit
import csv
import Stats_Code
import Load_Size_Dists
import os
import datetime
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DayLocator, HourLocator, DateFormatter
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams
import numpy.matlib
import seaborn
import probscale
import warnings
warnings.simplefilter('ignore')
clear_bkgd = {'axes.facecolor':'none', 'figure.facecolor':'none'}
#seaborn.set(style='ticks', context='talk', color_codes=True, rc=clear_bkgd)
# load up some example data from the seaborn package
tips = seaborn.load_dataset("tips")
iris = seaborn.load_dataset("iris")
def Run():
    """
    Performs internal consistency analyses on retrieved dataset.   
    
    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)    

    Requirements
    ------------ 
    * ``numpy``
    * ``datetime``
    * ``itertools``
    * ``matplotlib``
    * ``mpl_toolkits``
    * ``pylab``
    * ``seaborn``
    * ``os``
    """ 


  def flatten(l):
      return [item for sublist in l for item in sublist]  

  def Line(m,x,b):
      y = m*x + b
      return y  

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
  def grabvalues(
      dictionaryname,
      startofkeyname
    ):
      OP = dict()
      io = 0
      for key in dictionaryname:
        if key.startswith(startofkeyname):
          #print(key,io)
          value = dictionaryname[key]
          OP[io] = np.squeeze(value.T)
          io += 1
      return OP    
  def grabvaluessd(
      dictionaryname,
    ):
      OP = dict()
      io = 0
      dp = None
      for key in dictionaryname:
        if key.startswith("dndlogdp_"):
          #print(key,io)
          value = dictionaryname[key]
          OP[io] = np.squeeze(value.T)
          io += 1
      return OP

  def getPercentileList(
      prctile,
      suffix
    ):
      prctile_lst = np.array([f"{x}_percentile_{suffix}" for x in prctile])
      return prctile_lst  
  

  def boundcreation(H):
      bounds = np.array([0, 1, 10, 100, 1000, 10000]).astype(int)
      #bounds = np.arange(0,12,1)
      lenbnds = len(bounds)
      N = lenbnds-1
      Jet = plt.get_cmap('jet', N)
      newcolors = Jet(np.linspace(0, 1, N))
      wht = np.array([1, 1, 1, 1])
      gry = np.array([0.75, 0.75, 0.75, 1])
      blk = np.array([0, 0, 0, 1])
      newcolors[0, :] = wht
      newcolors[-1, :] = blk
      newcolors = np.vstack((gry,newcolors))
      cmap = ListedColormap(newcolors)
      boundsLbs = np.array(["0", "1", "10", r"10$^2$", r"10$^3$", r"10$^4$"])
      endlbl = boundsLbs[-1] 

      norm = mpl.colors.BoundaryNorm(bounds, cmap.N)    

      return norm, cmap, bounds, boundsLbs, gry
      
  bds1 = np.vstack((np.hstack((0,1,2,range(10,120,10))),np.hstack((0,1,2,range(10,120,10)))))
  bds2 = np.vstack((np.hstack((0,1,2,range(10,120,10))),np.hstack((0,1,2,range(5,60,5)))))
  bds3 = np.vstack((np.hstack((0,1,2,range(10,120,10))),np.hstack((0,1,2,range(5,60,5)))))
  bds4 = np.hstack((0,1,2,range(400,4800,400)))#np.vstack((np.hstack((0,1,2,range(50,600,50))),np.hstack((0,1,2,range(50,600,50)))))  

  fs =14
  lw = 1.5
  full_wvl = {}        
  wvls_sca = input(f"Enter the scattering wavelength channels speparated\nby a comma and a space (e.g., 370, 530, 1060): ")
  full_wvl["Sc"] =  np.array(wvls_sca.split(", ")).astype(int)
  wvls_abs = input(f"Enter the scattering wavelength channels speparated\nby a comma and a space (e.g., 370, 530, 1060): ")
  full_wvl["Abs"] =  np.array(wvls_abs.split(", ")).astype(int)
  val_wvl = np.array([530]) 

  IRI = np.arange(0.0, 0.08, 0.001).reshape(-1) 
  kappa = np.arange(0.0, 0.8, 0.01).reshape(-1) 
  Bin = dict()
  Lst = ["IRI","Kappa","fRH"]
  Lst1 = ["reff_d","reff_wet"]
  Bin["IRI"] = IRI
  Bin["Kappa"] = kappa
  Bin["fRH"] = np.arange(1, 3.0, 0.025).reshape(-1) 
  N = len(bds1[0,:])-1
  Jet = plt.get_cmap('jet', N)
  newcolors = Jet(np.linspace(0, 1, N))
  wht = np.array([1, 1, 1, 1])
  gry = np.array([0.75, 0.75, 0.75, 1])
  blk = np.array([0, 0, 0, 1])
  newcolors[0, :] = wht
  newcolors[-1, :] = blk
  newcolors = np.vstack((gry,newcolors))
  cmap = ListedColormap(newcolors)
  camp_name = input("Enter the campaign name in capital case (e.g., ARCSIX): ") 
  camp_name_lower = camp_name.lower()
  resolution = input("Enter the temporal resolution of interest in seconds (e.g., 30): ") 
  reference_platform = input("Enter the platform of interest (e.g., cirpas-to or MARINA-TOWER): ")  
  

  prctile = [0,10,50,68,90,95,100]
  prctile_lst_b = getPercentileList(prctile,"B")
  prctile_lst_ab = getPercentileList(prctile,"AB")
  prctile_lst_rb = getPercentileList(prctile,"RB")
  prctile_lst_arb = getPercentileList(prctile,"ARB")
  prctile_lst_x = getPercentileList(prctile,"x")
  prctile_lst_y = getPercentileList(prctile,"y")

  rcParams['font.size'] = fs
  #rcParams['axes.formatter.useoffset'] = False    
  plt.rcParams.update({'font.size': fs})
  plt.rcParams['font.family'] = 'serif'
  plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']   
  plt.rcParams.update({'mathtext.fontset': 'stix',
   'mathtext.rm': 'Times New Roman',
   'mathtext.it': 'Times New Roman:italic',
   'mathtext.bf': 'Times New Roman:bold'})  

  rsindex = 0
  output_filename_suffix = f'{camp_name_lower}-mrg{resolution}_{reference_platform}'
  output_filename=f'../ISARA_data_files/{camp_name}/Retrievals/{output_filename_suffix}_DataRetrievals.npy'
  print(output_filename)
  OP_Dictionary = dict_reconfig(np.load(f'./{output_filename}',allow_pickle='TRUE'))
  CRI_flag = grabvalues(OP_Dictionary,'attempt_flag_CRI')[0]
  Npt00 = np.nansum(CRI_flag==0)
  print(f'Number of points without enough data: {Npt00}')
  Npt01 = np.nansum(CRI_flag==1)
  Npt02 = np.nansum(CRI_flag==2)
  print(f'Attempts made: {Npt01+Npt02}')
  print(f'Number of successful CRI retrievals: {Npt02}')    
  print('(Successes)/(Attempts)x100: %i'%((Npt02/(Npt01+Npt02))*100),'%')
  k_flag = grabvalues(OP_Dictionary,'attempt_flag_kappa')[0]
  Npt10 = np.nansum(k_flag==0)
  print(f'Number of points without enough data: {Npt10}')
  Npt11 = np.nansum(k_flag==1)
  Npt12 = np.nansum(k_flag==2)
  print(f'Attempts made: {Npt11+Npt12}')
  print(f'Number of successful kappa retrievals: {Npt12}')    
  print('(Successes)/(Attempts)x100: %i'%((Npt12/(Npt11+Npt12))*100),'%')

  RH_name = input("Enter the short name of the variable representing ambient relative humidity in source data file: ")  
  RHa = grabvalues(OP_Dictionary,RH_name)[0]
  InletFlag_name = input("Enter the short name of the variable associated with the inlet flag in source data file: ") 
  #InletFlag = grabvalues(OP_Dictionary,InletFlag_name)[0]
  #IDX3 = np.where((InletFlag>0))[0]
  Alt_name = input("Enter the short name of the variable representing aircraft altitude in source data file: ")  
  GPS_Alt = grabvalues(OP_Dictionary,Alt_name)[0] 
  #RH = grabvalues(OP_Dictionary,'RHw_DLH_DISKIN_ ')
  stdPT_LAS = grabvalues(OP_Dictionary,'stdPT_ZIEMBA')[0]
  IRI = grabvalues(OP_Dictionary,'dry_IRI')[0]
  RRI = grabvalues(OP_Dictionary,'dry_RRI')[0]
  kappa = grabvalues(OP_Dictionary,'kappa')[0]  #

  Sc = dict()
  Abs = dict()
  SSA = dict()
  Cal_Sc = dict()
  Cal_Abs = dict()
  Cal_SSA = dict()
  Lwvl = len(full_wvl["Sc"])
  for iwvl in range(Lwvl):
      #print(iwvl)
      Sc[f'{full_wvl["Sc"][iwvl]}'] = grabvalues(OP_Dictionary,f'dry_meas_sca_coef_{full_wvl["Sc"][iwvl]}')[0]#*10**6
      Abs[f'{full_wvl["Abs"][iwvl]}'] = grabvalues(OP_Dictionary,f'dry_meas_abs_coef_{full_wvl["Abs"][iwvl]}')[0]#*10**6
      SSA[f'{full_wvl["Sc"][iwvl]}'] = grabvalues(OP_Dictionary,f'SSA_{full_wvl["Sc"][iwvl]}')[0]
      Cal_Sc[f'{full_wvl["Sc"][iwvl]}'] = grabvalues(OP_Dictionary,f'dry_cal_sca_coef_{full_wvl["Sc"][iwvl]}')[0]#*10**6
      Cal_Abs[f'{full_wvl["Abs"][iwvl]}'] = grabvalues(OP_Dictionary,f'dry_cal_abs_coef_{full_wvl["Abs"][iwvl]}')[0]#*10**6
      Cal_SSA[f'{full_wvl["Sc"][iwvl]}'] = grabvalues(OP_Dictionary,f'dry_cal_SSA_{full_wvl["Sc"][iwvl]}')[0] 

  y0 = dict()
  y0[0] = Sc
  y0[1] = Abs   

  Sc_wet = grabvalues(OP_Dictionary,"wet_meas_sca_coef_550")[0]#*10**6
  SSA_wet = Sc_wet/grabvalues(OP_Dictionary,"wet_meas_ext_coef_550")[0]#*10**6
  fRH = grabvalues(OP_Dictionary,'meas_fRH')[0]
  Cal_Sc_wet = grabvalues(OP_Dictionary,"wet_cal_sca_coef_550")[0]#*10**6
  Cal_SSA_wet = grabvalues(OP_Dictionary,"wet_cal_ext_coef_550")[0]#*10**6
  Cal_fRH = grabvalues(OP_Dictionary,'cal_fRH')[0]  

  #Val_Sc_dry = grabvalues(OP_Dictionary,"Sc530_total_ZIEMBA")[0]
  #Cal_Sc_dry = grabvalues(OP_Dictionary,"dry_cal_sca_coef_530")[0]#*10**6
  #print(len(Cal_Sc_dry))
  #print(len(Val_Sc_dry))  

  #Val_Abs_dry = grabvalues(OP_Dictionary,"Ext530_total_ZIEMBA")[0]-grabvalues(OP_Dictionary,"Sc530_total_ZIEMBA")[0]
 # Cal_Abs_dry = grabvalues(OP_Dictionary,"dry_cal_abs_coef_530")[0]#*10**6
 # print(len(Cal_Abs_dry))
  #print(len(Val_Abs_dry)) 
  

  y1 = dict()
  y1[0] = Sc_wet
  #y1[1] = SSA_wet[0]
  y1[1] = fRH 

  x0 = dict()
  x0[0] = Cal_Sc
  x0[1] = Cal_Abs   

  x1 = dict()
  x1[0] = Cal_Sc_wet
  #x1[1] = Cal_SSA_wet[0][0]
  x1[1] = Cal_fRH 

    
  y2 = np.vstack((IRI,kappa,Cal_fRH))
  stats_y2 = StatsCode.Survey(y2,prctile)
  print(len(kappa))
  print(len(RHa))  

#  yv = dict()
#  yv[0] = Val_Sc_dry
#  yv[1] = Val_Abs_dry #

 # xv = dict()
 # xv[0] = Cal_Sc_dry
  # xv[1] = Cal_Abs_dry 

  SD = grabvaluessd(OP_Dictionary)
  dpl = OP_Dictionary["full_dp"][0].item().get("dpl")
  dpu = OP_Dictionary["full_dp"][0].item().get("dpu")
  dpg = OP_Dictionary["full_dp"][0].item().get("dpg")
  sd = np.zeros((len(dpg),len(SD[0])))
  print(len(SD[0]),len(dpg))
  for i1 in range(len(SD)):
    #print(i1)
    sd[i1,:] = SD[i1]
    sd[i1,k_flag<2]=np.nan
  gf = np.power((1+kappa*RHa/(100-RHa)),1/3)#D/Ddry = (1+kappa*RH/(100-RH))**(1/3)
  D_l = np.multiply(dpl.reshape(1,-1),gf.reshape(-1,1))  
  D_u = np.transpose(dpu.reshape(1,-1)*gf.reshape(-1,1))   
  D_g = np.transpose(dpg.reshape(1,-1)*gf.reshape(-1,1))  

  Latitude = grabvalues(OP_Dictionary,"Latitude")[0]
  Longitude = grabvalues(OP_Dictionary,"Longitude")[0]
  matdatetime = grabvalues(OP_Dictionary,"datetime_Start")[0]  

  year = [dt.year for dt in matdatetime.astype(object)]
  month = [dt.month for dt in matdatetime.astype(object)]
  day = [dt.day for dt in matdatetime.astype(object)]
  hour = [dt.hour for dt in matdatetime.astype(object)]
  minute = [dt.minute for dt in matdatetime.astype(object)]
  second = [dt.second for dt in matdatetime.astype(object)]
  lat = Latitude  

  lon = Longitude
  alt = GPS_Alt 

  prctile = [0,10,25,50,75,90,100]
  stats_sd = StatsCode.Survey(sd,prctile)#
  print(len(stats_sd[6,:]))

  p0 = [1e5, 1e-3, 2, 3e3, 20e-3, 2]# [5e4, 1e-3, 2, 5e4, 1e-3, 2, 1e1, 0.01, 2]#[1e5, 1e-3, 2, 3e3, 20e-3, 2]# initial guesses 
  n_modes = 2
  fitresults = SD_Fit.Run(n_modes,p0,dpg,sd,Cal_Abs[f'{full_wvl["Abs"][1]}'],Cal_Sc[f'{full_wvl["Sc"][1]}'],f'../ISARA_data_files/{camp_name}/FitSDResults/')#*10**(-6)*10**(-6)

  fitresults['measured_size_distribution'] = sd
  fitresults['measured_dpg'] = dpg
  fitresults['measured_dpu'] = dpu
  fitresults['measured_dpl'] = dpl

  np.save(f'../ISARA_data_files/{camp_name}/FitSDResults/{output_filename_suffix}_SD_Fit_Data.npy', fitresults) 
  consit_file_location = f'{camp_name}/InternalConsistency/{output_filename_suffix}'
  prctile = [0,10,50,68,90,95,100]  

  pltidx = np.zeros((Lwvl*2,2)).astype(int)
  j1 = 0
  j2 = 0
  for iwvl in range(Lwvl*2):
    if iwvl < 3:
      pltidx[iwvl,:] = [j1,j2]
      j1 = j1 + 1
    elif iwvl == 3:
      j1 = 0
      j2 = 1
      pltidx[iwvl,:] =[j1,j2]
    else:
      j1 = j1 + 1
      pltidx[iwvl,:] =[j1,j2] 

  ttl = [r"Dry $C_{\rm scat}$", r"Dry $C_{\rm abs}$"]
  FIGLBLS = np.array([["(a)","(d)"],["(b)","(e)"],["(c)","(f)"]])
  rcParams['figure.figsize'] = 7.5, 10
  fig,ax2=plt.subplots(3, 2) # create figure and subplot
  xy0mmax = 0
  xy1mmax = 0
  for iwvl in range(Lwvl):
    xy0max = np.nanmax([xy0mmax,np.nanmax(x0[0][f'{full_wvl["Sc"][iwvl]}']),np.nanmax(y0[0][f'{full_wvl["Sc"][iwvl]}'])])
    xy1max = 15#np.nanmax([xy1mmax,np.nanmax(np.multiply(x0[1][i1],pow(10,6))),np.nanmax(y0[1][i1])])
  #xymax = np.array([xy0max,xy1max])
  xymax = np.array([300,15])
  bounds = bds1[rsindex,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  #stats_dict = np.zeros((Lwvl*2+7,57))
  stats_dict = np.zeros((Lwvl*2+5,71))

  #stats_dict = []
  i0 = 0
  for iwvl in range(Lwvl*2):
    # Create heatmap
    if iwvl < 3:
      x = x0[0][f'{full_wvl["Sc"][iwvl]}']
      y = y0[0][f'{full_wvl["Sc"][iwvl]}']
      wvl = full_wvl["Sc"][iwvl]
    else:
      x = x0[1][f'{full_wvl["Abs"][iwvl-3]}']
      y = y0[1][f'{full_wvl["Abs"][iwvl-3]}'] 
      wvl = full_wvl["Abs"][iwvl-3]
    #Npt = len((y[np.where(np.logical_not(np.isnan(y)))]))   
    #
    idx = np.where(np.logical_not(np.isnan(x))&(np.logical_not(np.isnan(y))))[0]
    x = x[idx]
    y = y[idx]
    stats_dict[i0,:] = np.hstack((wvl,StatsCode.Comparison(x,y,prctile),Npt00,Npt01+Npt02,Npt02))
    i0 += 1   
  #  y = list(itertools.chain(*y))
  #  x = list(itertools.chain(*x))  
    H, xedges, yedges = np.histogram2d(x, y, bins=(64,64),range=([[0, xymax[pltidx[iwvl,1]]], [0, xymax[pltidx[iwvl,1]]]]))
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    # Plot heatmap
  #  im = ax2[pltidx[iwvl,0],pltidx[iwvl,1]].pcolormesh(X,Y,np.where(H == 0, np.nan, H), cmap=cmap, vmin=1, vmax=100)
    im = ax2[pltidx[iwvl,0],pltidx[iwvl,1]].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
    ax2[pltidx[iwvl,0],pltidx[iwvl,1]].set_facecolor(gry)
    if iwvl < 3: 
      ax2[pltidx[iwvl,0],pltidx[iwvl,1]].set_ylabel("measured", fontsize=fs) # set yaxis label   
    if pltidx[iwvl,0] == 0:
      ax2[pltidx[iwvl,0],pltidx[iwvl,1]].set_title("%s (Mm$^{-1}$)"%(ttl[pltidx[iwvl,1]]), fontsize=fs) #set title as flight date.
    elif pltidx[iwvl,0] == 2:
      ax2[pltidx[iwvl,0],pltidx[iwvl,1]].set_xlabel("ISARA-derived", fontsize=fs) # set xaxis label 
    ax2[pltidx[iwvl,0],pltidx[iwvl,1]].set_ylim(0,xymax[pltidx[iwvl,1]]) # cut y-axis off at zero   
    ax2[pltidx[iwvl,0],pltidx[iwvl,1]].set_xlim(0,xymax[pltidx[iwvl,1]])    
    # set the line widths of the axes
    for axis in ['top','bottom','left','right']:
        ax2[pltidx[iwvl,0],pltidx[iwvl,1]].spines[axis].set_linewidth(1.5)     
    ax2[pltidx[iwvl,0],pltidx[iwvl,1]].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
    ax2[pltidx[iwvl,0],pltidx[iwvl,1]].tick_params(axis='both', labelsize=fs, rotation=0)  
    for label in ax2[pltidx[iwvl,0],pltidx[iwvl,1]].get_xticklabels():
        label.set_horizontalalignment('center')
    at = AnchoredText("%s %i nm"%(FIGLBLS[pltidx[iwvl,0],pltidx[iwvl,1]],wvl), prop=dict(size=fs), frameon=False, loc='upper left')
    ax2[pltidx[iwvl,0],pltidx[iwvl,1]].add_artist(at)
    ytks = ax2[pltidx[iwvl,0],pltidx[iwvl,1]].get_yticks()
    ytklbls = ["%i"%ix for ix in ytks]
    xtklbls = ["%i"%ix for ix in ytks]
    xtklbls[0] = ""
    ax2[pltidx[iwvl,0],pltidx[iwvl,1]].set_xticks(ytks, xtklbls) 
    ax2[pltidx[iwvl,0],pltidx[iwvl,1]].set_yticks(ytks, ytklbls) 
    ax2[pltidx[iwvl,0],pltidx[iwvl,1]].plot(ytks,ytks,'--',color='xkcd:fuchsia',lw=lw)  
  # display and save figure using the *.ict data filename
  plt.subplots_adjust(bottom=0.1, right=0.77, top=0.9, wspace = 0.3, hspace = 0.3)
  cax = plt.axes([0.8, 0.1, 0.055, 0.8])
  #cbar = plt.colorbar(im,cax=cax,ticks=np.hstack((1, range(10, 100, 10))))
  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
  cbar.set_ticklabels(boundsLbs) 
  cbar.ax.tick_params(length=8, width=2, which="major")
  cbar.outline.set_linewidth(1.5)
  #cbar.ax.get_yaxis().set_ticks([])
  #for j, lab in enumerate(['0','$1$','$10$','$20$','$40$','$50$','$60$','$70$','$80$','$90$','$>100$']):
  cbar.set_label('count',labelpad=-10)
  plt.savefig(f"{consit_file_location}_DataRetrievals_1to1_dry", dpi=300)
  plt.show() # function to display the plot        
  plt.close() #   
  

  bounds = bds2[rsindex,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  rcParams['figure.figsize'] = 5, 10
  fig,ax2=plt.subplots(3, 1) # create figure and subplot
  #xymax = np.array([np.nanmax(np.vstack((Cal_SSA,SSA))),
  #                  np.nanmax(np.vstack((Cal_SSA,SSA)))])
  #fig,ax2=plt.subplots(1, 1) # create figure and subplot
  FIGLBLS = ["(a)","(b)","(c)"]
  ttl = "SSA" 

  xymin = 0.6  
  xymax = 1.00
    #xymax = 150
  for iwvl in range(Lwvl):
    # Create heatmap
    x = SSA[f'{full_wvl["Sc"][iwvl]}']
    y = Cal_SSA[f'{full_wvl["Sc"][iwvl]}'] 
    wvl = full_wvl["Sc"][iwvl]
    #Npt = len((y[np.where(np.logical_not(np.isnan(y)))]))  
    idx = np.where(np.logical_not(np.isnan(x))&(np.logical_not(np.isnan(y))))[0]
  #  print(idx)
    x = x[idx]
    y = y[idx]
    x = x[y>0]
    y = y[y>0]  
    stats_dict[i0,:] = np.hstack((wvl,StatsCode.Comparison(x,y,prctile),Npt00,Npt01+Npt02,Npt02))
    i0 += 1
    #y = list(itertools.chain(*y))
    #x = list(itertools.chain(*x))
    H, xedges, yedges = np.histogram2d(x, y, bins=(64,64),range=([[xymin, xymax], [xymin, xymax]]))
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    # Plot heatmap
    im = ax2[iwvl].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
    ax2[iwvl].set_facecolor(gry)
    if iwvl == 2:
      ax2[iwvl].set_xlabel("ISARA-derived", fontsize=fs) # set xaxis label 
    if iwvl == 1:
      ax2[iwvl].set_title(f'{ttl}', fontsize=fs) #set title as flight date.
    ax2[iwvl].set_ylabel("measured", fontsize=fs) # set yaxis label   
    ax2[iwvl].set_ylim(xymin,xymax) # cut y-axis off at zero   
    ax2[iwvl].set_xlim(xymin,xymax)    
    # set the line widths of the axes
    for axis in ['top','bottom','left','right']:
        ax2[iwvl].spines[axis].set_linewidth(1.5)     
    ax2[iwvl].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
    ax2[iwvl].tick_params(axis='both', labelsize=fs, rotation=0)  
    for label in ax2[iwvl].get_xticklabels():
        label.set_horizontalalignment('center')
    #display and save figure using the *.ict data filename 
    at = AnchoredText(FIGLBLS[iwvl], prop=dict(size=fs), frameon=False, loc='upper left')
    ax2[iwvl].add_artist(at)
    ytks = ax2[iwvl].get_yticks()
    ytklbls = ["%0.2f"%ix for ix in ytks]
    xtklbls = ["%0.2f"%ix for ix in ytks]
    xtklbls[0] = ""
    ax2[iwvl].set_xticks(ytks, xtklbls)
    ax2[iwvl].set_yticks(ytks, ytklbls)
    ax2[iwvl].plot(ytks,ytks,'--',color='xkcd:fuchsia',lw=lw)   
  plt.subplots_adjust(left=0.16, bottom=0.1, right=0.725, top=0.9, wspace = 0.3, hspace = 0.3)
  cax = plt.axes([0.775, 0.1, 0.055, 0.8])
  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
  cbar.set_ticklabels(boundsLbs) 
  cbar.outline.set_linewidth(lw)
  cbar.ax.tick_params(length=8, width=2, which="major")
  cbar.set_label('count',labelpad=-10)
  plt.savefig(f"{consit_file_location}_DataRetrievals_1to1_SSA", dpi=300)
  plt.show() # function to display the plot        
  plt.close() #   

  bounds = bds3[rsindex,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  rcParams['figure.figsize'] = 5, 7 # W, H
  fig,ax2=plt.subplots(2, 1) # create figure and subplot  
  FIGLBLS = ["(a)","(b)"]
  ttl = [r"Wet $C_{\rm scat}$ at 550 nm (Mm$^{-1}$)", r"$f$(RH)"]
  ws = [550,550]
 
  dattype = ["%i","%0.1f"]
  xymin = np.array([0,0.75,0])   
  xymax_ary = np.array([300,3])
  for i2 in [0,1]:
    # Create heatmap
    x = x1[i2]
    y = y1[i2]
    #print(x[np.where(np.logical_not(np.isnan(x)))],y[np.where(np.logical_not(np.isnan(y)))])
  #  print(x)
  #  print(y)  
    #Npt = len((y[np.logical_not(np.isnan(y))]))
    idx = np.where((np.logical_not(np.isnan(y)))&(np.logical_not(np.isnan(x))))[0]
  #  print(idx)
    x = x[idx]
    y = y[idx]  

    x = x[y>0]
    y = y[y>0]  
    stats_dict[i0,:] = np.hstack((ws[i2],StatsCode.Comparison(x,y,prctile),Npt10,Npt11+Npt12,Npt12))
    i0 += 1 
    #xymax = np.nanmax(np.vstack((x,y)))
    xymax = xymax_ary[i2]
  #  y = list(itertools.chain(*y))
  #  x = list(itertools.chain(*x))
    H, xedges, yedges = np.histogram2d(x, y, bins=(64,64),range=([[xymin[i2], xymax], [xymin[i2], xymax]]))
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    # Plot heatmap
    im = ax2[i2].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
    ax2[i2].set_facecolor(gry)  
    if i2 == 1:
      ax2[i2].set_xlabel("ISARA-derived", fontsize=fs) # set xaxis label 
    ax2[i2].set_title(f'{ttl[i2]}', fontsize=fs) #set title as flight date.
    ax2[i2].set_ylabel("measured", fontsize=fs) # set yaxis label   
    ax2[i2].set_ylim(xymin[i2],xymax) # cut y-axis off at zero   
    ax2[i2].set_xlim(xymin[i2],xymax)    
    # set the line widths of the axes
    for axis in ['top','bottom','left','right']:
        ax2[i2].spines[axis].set_linewidth(lw)     
    ax2[i2].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
    ax2[i2].tick_params(axis='both', labelsize=fs, rotation=0)  
    for label in ax2[i2].get_xticklabels():
        label.set_horizontalalignment('center')
    #display and save figure using the *.ict data filename 
    at = AnchoredText(FIGLBLS[i2], prop=dict(size=fs), frameon=False, loc='upper left')
    ax2[i2].add_artist(at)
    ytks = ax2[i2].get_yticks()
    ytklbls = [dattype[i2]%ix for ix in ytks]
    xtklbls = [dattype[i2]%ix for ix in ytks]
    xtklbls[0] = ""
    ax2[i2].set_xticks(ytks, xtklbls)
    ax2[i2].set_yticks(ytks, ytklbls)
    ax2[i2].plot(ytks,ytks,'--',color='xkcd:fuchsia',lw=lw)   
  plt.subplots_adjust(left=0.16, bottom=0.08, right=0.725, top=0.92, wspace = 0.3, hspace = 0.3)
  cax = plt.axes([0.775, 0.08, 0.055, 0.84])
  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
  cbar.set_ticklabels(boundsLbs) 
  cbar.outline.set_linewidth(1.5)
  cbar.ax.tick_params(length=8, width=2, which="major")
  cbar.set_label('count',labelpad=-10)
  plt.savefig(f"{consit_file_location}_DataRetrievals_1to1_wet", dpi=300)
  plt.show() # function to display the plot        
  plt.close() #   

#  bounds = bds1[rsindex,:]
#  lenbnds = len(bounds)
#  boundsLbs = bounds.astype(str)
#  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
#  boundsLbs[lenbnds-1] = ""
#  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
#  rcParams['figure.figsize'] = 10, 14
#  fig,ax2=plt.subplots(2, 1) # create figure and subplot  
#  FIGLBLS = ["(a)","(b)"]
#  ttl = ["Dry Scattering\nat 530 nm (Mm$^{-1}$)","Dry absorption\nat 530 nm (Mm$^{-1}$)"]
#  ws = [530,530]
#  #xymax = np.array([300,1,6])
#  dattype = ["%i","%i"]
#  xymin = np.array([0,0.75,0])  
#  for i2 in [0]:
#    # Create heatmap
#    x = xv[i2]
#    y = yv[i2]
#    #print(x[np.where(np.logical_not(np.isnan(x)))],y[np.where(np.logical_not(np.isnan(y)))])
#  #  print(x)
#  #  print(y)  
#    Npt = len((y[np.logical_not(np.isnan(y))]))
#    idx = np.where((np.logical_not(np.isnan(y)))&(np.logical_not(np.isnan(x))))[0]
#  #  print(idx)
#    x = x[idx]
#    y = y[idx]
#    x = x[y>0]
#    y = y[y>0]  
#    stats_dict[i0,:] = np.hstack((ws[i2],StatsCode.Comparison(x,y,prctile),Npt))
#    i0 += 1 
#    xymax = np.nanmax(np.vstack((x,y)))
#  #  y = list(itertools.chain(*y))
#  #  x = list(itertools.chain(*x))
#    H, xedges, yedges = np.histogram2d(x, y, bins=(64,64),range=([[xymin[i2], xymax], [xymin[i2], xymax]]))
#    H = H.T
#    X, Y = np.meshgrid(xedges, yedges)
#    # Plot heatmap
#    im = ax2[i2].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
#    ax2[i2].set_facecolor(gry)  
#    if i2 == 1:
#      ax2[i2].set_xlabel("ISARA-derived", fontsize=fs) # set xaxis label 
#    ax2[i2].set_title(f'{ttl[i2]}', fontsize=fs) #set title as flight date.
#    ax2[i2].set_ylabel("measured", fontsize=fs) # set yaxis label   
#    ax2[i2].set_ylim(xymin[i2],xymax) # cut y-axis off at zero   
#    ax2[i2].set_xlim(xymin[i2],xymax)    
#    # set the line widths of the axes
#    for axis in ['top','bottom','left','right']:
#        ax2[i2].spines[axis].set_linewidth(1.5)     
#    ax2[i2].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
#    ax2[i2].tick_params(axis='both', labelsize=fs, rotation=0)  
#    for label in ax2[i2].get_xticklabels():
#        label.set_horizontalalignment('center')
#    #display and save figure using the *.ict data filename 
#    at = AnchoredText(FIGLBLS[i2], prop=dict(size=fs), frameon=False, loc='upper left')
#    ax2[i2].add_artist(at)
#    ytks = ax2[i2].get_yticks()
#    ytklbls = [dattype[i2]%ix for ix in ytks]
#    xtklbls = [dattype[i2]%ix for ix in ytks]
#    xtklbls[0] = ""
#    ax2[i2].set_xticks(ytks, xtklbls)
#    ax2[i2].set_yticks(ytks, ytklbls)
#    ax2[i2].plot(ytks,ytks,'--',color='xkcd:fuchsia',lw=3)   
#  plt.subplots_adjust(left=0.16, bottom=0.08, right=0.725, top=0.92)
#  cax = plt.axes([0.775, 0.08, 0.055, 0.84])
#  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
#  cbar.set_ticklabels(boundsLbs) 
#  cbar.outline.set_linewidth(1.5)
#  cbar.ax.tick_params(length=8, width=2, which="major")
#  cbar.set_label('count',labelpad=-25)
#  plt.savefig(f"{consit_file_location}_DataRetrievals_1to1_val", dpi=300)
#  plt.show() # function to display the plot        
#  plt.close() #   #

#  cols = ['param','Dry_Scattering', 'Dry_Scattering', 'Dry_Scattering', 'Dry_Absorption', 
#          'Dry_Absorption', 'Dry_Absorption', 'Dry_SSA', 'Dry_SSA', 'Dry_SSA', 'Wet_Scattering', 'fRH','Dry_Scattering_Val','Dry_absorption_Val']
#  colnames = ','.join(str(e) for e in cols)
#  rows = np.hstack(('wavelength_nm','R','log10_p-value',prctile_lst_b,'mean_b','stdev_b',prctile_lst_ab,'mean_ab','stdev_ab',
#                  prctile_lst_rb,'mean_rb','stdev_rb',prctile_lst_arb,'mean_arb','stdev_arb','NMAD','MAD','NRMSD','RMSD',prctile_lst_x,'mean_x','stdev_x',
#                  prctile_lst_y,'mean_y','stdev_y','MoranI','MoranEI','MoranI_znorm','MoranI_pnorm','MoranI_zrand','MoranI_prand','count','total'))
#  str_data = np.char.mod("%10.6f", stats_dict.T)
#  str_data= np.column_stack((rows,str_data))
#  output_filename = f"{camp_name}/InternalConsistency/{output_filename_suffix}_DataRetrievals_1to1_stats.csv"
#  with open(output_filename, 'w') as f:
#      np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)    

  cols = ['param','Dry_Scattering', 'Dry_Scattering', 'Dry_Scattering', 'Dry_Absorption', 
          'Dry_Absorption', 'Dry_Absorption', 'Dry_SSA', 'Dry_SSA', 'Dry_SSA', 'Wet_Scattering', 'fRH']
  colnames = ','.join(str(e) for e in cols)
  rows = np.hstack(('wavelength_nm','R','log10_p-value',prctile_lst_b,'mean_b','stdev_b',prctile_lst_ab,'mean_ab','stdev_ab',
                  prctile_lst_rb,'mean_rb','stdev_rb',prctile_lst_arb,'mean_arb','stdev_arb','NMAD','MAD','NRMSD','RMSD',prctile_lst_x,'mean_x','stdev_x',
                  prctile_lst_y,'mean_y','stdev_y','MoranI','MoranEI','MoranI_znorm','MoranI_pnorm','MoranI_zrand','MoranI_prand','count','not_enough_measurements','total_attempts','successful_retrievals'))
  str_data = np.char.mod("%10.6f", stats_dict.T)
  str_data= np.column_stack((rows,str_data))
  output_filename = f"{camp_name}/InternalConsistency/{output_filename_suffix}_DataRetrievals_1to1_stats.csv"
  with open(output_filename, 'w') as f:
      np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)   

  rcParams['figure.figsize'] = 6, 7.5
  FIGLBLS = ["(a)","(b)","(c)"]
  bounds = bds4#[rsindex,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  fig,ax2=plt.subplots(3, 1) # create figure and subplot
  y = sd.T
  #y_min_max = np.array([0.1,10**12])  
  #ybins = np.logspace(-1,12, num=64, endpoint=True, base=10.0)
  D_grd = {}
  D_grd["dpg"] = np.logspace(0,4, num=64, endpoint=True, base=10.0)
  D_grd["lbls"] = np.logspace(0,4,5, base=10.0)
  xbins = dpg*1000
  x = np.ones((len(y[:,0]),1))*xbins
  x = np.squeeze(x.reshape(-1,1))
  x_min_max = np.array([D_grd["dpg"][0],D_grd["dpg"][-1]])
  y = np.squeeze(y.reshape(1,-1))
  Y100 = stats_sd[6,:]
  Y90 = stats_sd[5,:] 
  Y75 = stats_sd[4,:]
  Y50 = stats_sd[3,:]
  Y25 = stats_sd[2,:]
  Y10 = stats_sd[1,:]
  Y0 = stats_sd[0,:]
  Ymean = stats_sd[7,:]
  Ystdv = stats_sd[8,:]
  Ylow =stats_sd[9,:] #Y25#
  Yhigh = stats_sd[10,:]#Y75#
  y_min_max = np.array([np.nanmin(Y0),np.nanmax(Y100)])  
  ybins = np.logspace(np.log10(np.nanmin(Y0)),np.log10(np.nanmax(Y100)), num=64, endpoint=True, base=10.0)
  #y = list(itertools.chain(*y))
  #x = list(itertools.chain(*x))
  H, xedges, yedges = np.histogram2d(x,y,bins=([xbins, ybins]),range=([x_min_max, y_min_max]))#
  H = H.T
  X, Y = np.meshgrid(xedges, yedges)#
  im = ax2[0].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
  ax2[0].set_facecolor(gry)  
  #y_min_max = np.array([np.nanmin(Y10[Y10>0]),np.nanmax(Y10[Y10>0])])
  ax2[0].plot(xbins,Y90,'--',color='xkcd:fuchsia',lw=2)
  #ax2.plot(x,Y1,':m',lw=3)
  ax2[0].plot(xbins,Ymean,'-',color='xkcd:fuchsia',lw=2) 
  #ax2.plot(x,Y3,':m',lw=3)
  ax2[0].plot(xbins,Y10,'--',color='xkcd:fuchsia',lw=2)
  #ax2[0].fill_between(xbins, Y100, Ymean, color='green',alpha=0.5) 
  #ax2[0].fill_between(xbins, Ymean, Y0, color='green',alpha=0.5)     
  ax2[0].set_ylabel(r"$\dfrac{{\rm d}N}{{\rm d} \log D} \ (\rm cm^{-3})$") #,,fontsize=40font='serif',fontname="Times New Roman"
  #ax2[0].set_xlabel(r"Dry $D$ (nm)") 
  ax2[0].set_xscale("log")
  ax2[0].set_yscale("log")      
  ax2[0].set_ylim(y_min_max[0],y_min_max[1]) # cut y-axis off at zero   
  ax2[0].set_xlim(x_min_max[0],x_min_max[1])
  xtklbls = ["" for ix in range(0,len(D_grd["lbls"]))]
  ax2[0].set_xticks(D_grd["lbls"], xtklbls)  
  # set the line widths of the axes
  for axis in ['top','bottom','left','right']:
     ax2[0].spines[axis].set_linewidth(1.5)     
  ax2[0].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
  ax2[0].tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2[0].tick_params(axis='both',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2[0].get_xticklabels():
     label.set_horizontalalignment('center')
  at = AnchoredText(FIGLBLS[0], prop=dict(size=fs), frameon=False, loc='upper right')
  ax2[0].add_artist(at) 

  y = np.multiply(sd.T,(np.pi)*(dpg**2))
  y = np.squeeze(y.reshape(-1,1))
  y100 = np.multiply(Y100,(np.pi)*dpg**2).T
  y90 = np.multiply(Y90,(np.pi)*dpg**2).T 
  y75 = np.multiply(Y75,(np.pi)*dpg**2).T
  y0 = np.multiply(Y0,(np.pi)*dpg**2).T
  y10 = np.multiply(Y10,(np.pi)*dpg**2).T
  y25 = np.multiply(Y25,(np.pi)*dpg**2).T
  ymean = np.multiply(Ymean,(np.pi)*dpg**2).T
  y_min_max = np.array([np.nanmin(y0),np.nanmax(y100)])  
  #y_min_max = np.array([10**(-8),10**6])  
  ybins = np.logspace(np.log10(np.nanmin(y0)),np.log10(np.nanmax(y100)), num=64, endpoint=True, base=10.0)#np.logspace(-8,4, num=64, endpoint=True, base=10.0)
  H, xedges, yedges = np.histogram2d(x,y,bins=([xbins, ybins]),range=([x_min_max, y_min_max]))#
  H = H.T
  X, Y = np.meshgrid(xedges, yedges)#
  im = ax2[1].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
  ax2[1].set_facecolor(gry) 
  #y_min_max = np.array([np.nanmin(Y0[Y0>0]),np.nanmax(Y100[Y100>0])])
  ax2[1].plot(xbins,y90,'--',color='xkcd:fuchsia',lw=2)
  #ax2.plot(x,Y1,':m',lw=3)
  ax2[1].plot(xbins,ymean,'-',color='xkcd:fuchsia',lw=2) 
  #ax2.plot(x,Y3,':m',lw=3)
  ax2[1].plot(xbins,y10,'--',color='xkcd:fuchsia',lw=2)
  #ax2[1].fill_between(xbins, Y100, Ymean, color='green',alpha=0.5) 
  #ax2[1].fill_between(xbins, Ymean, Y0, color='green',alpha=0.5)     
  ax2[1].set_ylabel(r"$\dfrac{{\rm d}S}{{\rm d} \log D} \ (\rm nm^2 \ cm^{-3})$") #,,fontsize=40font='serif',fontname="Times New Roman"
  #ax2[1].set_xlabel(r"Dry $D$ (nm)") 
  ax2[1].set_xscale("log")
  ax2[1].set_yscale("log")      
  ax2[1].set_ylim(y_min_max[0],y_min_max[1]) # cut y-axis off at zero   
  ax2[1].set_xlim(x_min_max[0],x_min_max[1])
  xtklbls = ["" for ix in range(0,len(D_grd["lbls"]))]
  ax2[1].set_xticks(D_grd["lbls"], xtklbls)  
  # set the line widths of the axes
  for axis in ['top','bottom','left','right']:
     ax2[1].spines[axis].set_linewidth(1.5)     
  ax2[1].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
  ax2[1].tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2[1].tick_params(axis='both', which="minor", direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2[1].get_xticklabels():
     label.set_horizontalalignment('center')
  at = AnchoredText(FIGLBLS[1], prop=dict(size=fs), frameon=False, loc='upper left')
  ax2[1].add_artist(at)   
  y = np.multiply(sd.T,(np.pi/6)*(dpg**3))
  y = np.squeeze(y.reshape(-1,1))
  y100 = np.multiply(Y100,(np.pi/6)*(dpg**3)).T
  y0 = np.multiply(Y0,(np.pi/6)*(dpg**3)).T
  y75 = np.multiply(Y75,(np.pi/6)*(dpg**3)).T
  y25 = np.multiply(Y25,(np.pi/6)*(dpg**3)).T  
  y90 = np.multiply(Y90,(np.pi/6)*(dpg**3)).T
  y10 = np.multiply(Y10,(np.pi/6)*(dpg**3)).T  
  ymean = np.multiply(Ymean,(np.pi/6)*(dpg**3)).T
  y_min_max = np.array([np.nanmin(y0),np.nanmax(y100)])  
  #y_min_max = np.array([10**(-8),10**6])  
  ybins = np.logspace(np.log10(np.nanmin(y0)),np.log10(np.nanmax(y100)), num=64, endpoint=True, base=10.0)#np.logspace(-8,4, num=64, endpoint=True, base=10.0)
  H, xedges, yedges = np.histogram2d(x,y,bins=([xbins, ybins]),range=([x_min_max, y_min_max]))#
  H = H.T
  X, Y = np.meshgrid(xedges, yedges)#
  im = ax2[2].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
  ax2[2].set_facecolor(gry) 
  #y_min_max = np.array([np.nanmin(Y0[Y0>0]),np.nanmax(Y100[Y100>0])])
  ax2[2].plot(xbins,y90,'--',color='xkcd:fuchsia',lw=2)
  #ax2.plot(x,Y1,':m',lw=3)
  ax2[2].plot(xbins,ymean,'-',color='xkcd:fuchsia',lw=2) 
  #ax2.plot(x,Y3,':m',lw=3)
  ax2[2].plot(xbins,y10,'--',color='xkcd:fuchsia',lw=2)
  #ax2[2].fill_between(xbins, Y100, Ymean, color='green',alpha=0.5) 
  #ax2[2].fill_between(xbins, Ymean, Y0, color='green',alpha=0.5)     
  ax2[2].set_ylabel(r"$\dfrac{{\rm d}V}{{\rm d} \log D} \ (\rm nm^3 \ cm^{-3})$") #,,fontsize=40font='serif',fontname="Times New Roman"
  ax2[2].set_xlabel(r"Dry $D$ (nm)") 
  ax2[2].set_xscale("log")
  ax2[2].set_yscale("log")      
  ax2[2].set_ylim(y_min_max[0],y_min_max[1]) # cut y-axis off at zero   
  ax2[2].set_xlim(x_min_max[0],x_min_max[1])
  xtklbls = [r"10$^{%i}$"%np.log10(D_grd["lbls"][ix]) for ix in range(0,len(D_grd["lbls"]))]
  ax2[2].set_xticks(D_grd["lbls"],xtklbls)
  # set the line widths of the axes
  for axis in ['top','bottom','left','right']:
     ax2[2].spines[axis].set_linewidth(1.5)     
  ax2[2].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
  ax2[2].tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2[2].tick_params(axis='both', which="minor", direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2[2].get_xticklabels():
     label.set_horizontalalignment('center')
  at = AnchoredText(FIGLBLS[2], prop=dict(size=fs), frameon=False, loc='upper left')
  ax2[2].add_artist(at)   
  cax = plt.axes([0.775, 0.1, 0.055, 0.85])
  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
  cbar.set_ticklabels(boundsLbs) 
  cbar.outline.set_linewidth(1.5)
  cbar.ax.tick_params(length=8, width=2, which="major")
  cbar.set_label('count',labelpad=-10)
  #display and save figure using the *.ict data filename 
  plt.subplots_adjust(left=0.18, bottom=0.11, right=0.75, top=0.95)
  plt.savefig(f"{consit_file_location}_DataRetrievals_SD_heatmapplot", dpi=300)
  plt.show() # function to display the plot        
  plt.close() # 

  rcParams['figure.figsize'] = 10.5, 15
  FIGLBLS = ["(a)","(b)","(c)"]
  fig,ax2=plt.subplots(3, 1) # create figure and subplot
  x = dpg*1000
  x_min_max = np.array([x[0],x[-1]])
  Y100 = stats_sd[6,:]
  Y90 = stats_sd[5,:] 
  Y75 = stats_sd[4,:]
  Y50 = stats_sd[3,:]
  Y25 = stats_sd[2,:]
  Y10 = stats_sd[1,:]
  Y0 = stats_sd[0,:]
  Ymean = stats_sd[7,:]
  Ystdv = stats_sd[8,:]
  Ylow =stats_sd[9,:] #Y25#
  Yhigh = stats_sd[10,:]#Y75#
  y_min_max = np.array([np.nanmin(Y0),np.nanmax(Y100)])  
  #y_min_max = np.array([np.nanmin(Y10[Y10>0]),np.nanmax(Y10[Y10>0])])
  ax2[0].plot(x,Y100,'--',color='xkcd:fuchsia',lw=6)
  #ax2.plot(x,Y1,':m',lw=3)
  ax2[0].plot(x,Ymean,'-k',lw=6) 
  #ax2.plot(x,Y3,':m',lw=3)
  ax2[0].plot(x,Y0,'--',color='xkcd:fuchsia',lw=6)
  ax2[0].fill_between(x, Y100, Ymean, color='green',alpha=0.5) 
  ax2[0].fill_between(x, Ymean, Y0, color='green',alpha=0.5)     
  ax2[0].set_ylabel(r"$\dfrac{{\rm d}N}{{\rm d} \log D} \ (\rm cm^{-3})$") #,,fontsize=40font='serif',fontname="Times New Roman"
  #ax2[0].set_xlabel(r"Dry $D$ (nm)") 
  ax2[0].set_xscale("log")
  ax2[0].set_yscale("log")      
  ax2[0].set_ylim(y_min_max[0],y_min_max[1]) # cut y-axis off at zero   
  ax2[0].set_xlim(x_min_max[0],x_min_max[1])
  xtklbls = ["%i"%(x[ix]) for ix in range(0,len(x)+1,10)]
  ax2[0].set_xticks(x[range(0,len(x)+1,10)], xtklbls)   
  # set the line widths of the axes
  for axis in ['top','bottom','left','right']:
     ax2[0].spines[axis].set_linewidth(1.5)     
  ax2[0].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
  ax2[0].tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2[0].tick_params(axis='both',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2[0].get_xticklabels():
     label.set_horizontalalignment('center')
  at = AnchoredText(FIGLBLS[0], prop=dict(size=fs), frameon=False, loc='upper right')
  ax2[0].add_artist(at)
  y100 = np.multiply(Y100,(np.pi)*dpg**2)
  y0 = np.multiply(Y0,(np.pi)*dpg**2)
  ymean = np.multiply(Ymean,(np.pi)*dpg**2)
  y_min_max = np.array([np.nanmin(y0),np.nanmax(y100)])  
  #y_min_max = np.array([np.nanmin(Y0[Y0>0]),np.nanmax(Y100[Y100>0])])
  ax2[1].plot(x,y100,'--r',lw=6)
  #ax2.plot(x,Y1,':m',lw=3)
  ax2[1].plot(x,ymean,'-k',lw=6) 
  #ax2.plot(x,Y3,':m',lw=3)
  ax2[1].plot(x,y0,'--r',lw=6)
  ax2[1].fill_between(x, y100, ymean, color='green',alpha=0.5) 
  ax2[1].fill_between(x, ymean, y0, color='green',alpha=0.5)     
  ax2[1].set_ylabel(r"$\dfrac{{\rm d}A}{{\rm d} \log D} \ (\rm nm^2 \ cm^{-3})$") #,,fontsize=40font='serif',fontname="Times New Roman"
  #ax2[1].set_xlabel(r"Dry $D$ (nm)") 
  ax2[1].set_xscale("log")
  ax2[1].set_yscale("log")      
  ax2[1].set_ylim(y_min_max[0],y_min_max[1]) # cut y-axis off at zero   
  ax2[1].set_xlim(x_min_max[0],x_min_max[1])
  xtklbls = ["%i"%(x[ix]) for ix in range(0,len(x)+1,10)]
  ax2[1].set_xticks(x[range(0,len(x)+1,10)], xtklbls)   
  # set the line widths of the axes
  for axis in ['top','bottom','left','right']:
     ax2[1].spines[axis].set_linewidth(1.5)     
  ax2[1].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
  ax2[1].tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2[1].tick_params(axis='both',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2[1].get_xticklabels():
     label.set_horizontalalignment('center')
  at = AnchoredText(FIGLBLS[1], prop=dict(size=fs), frameon=False, loc='upper right')
  ax2[1].add_artist(at) 
  y100 = np.multiply(Y100,(np.pi/6)*dpg**3)
  y0 = np.multiply(Y0,(np.pi/6)*dpg**3)
  ymean = np.multiply(Ymean,(np.pi/6)*dpg**3)
  y_min_max = np.array([np.nanmin(y0),np.nanmax(y100)])  
  #y_min_max = np.array([np.nanmin(Y0[Y0>0]),np.nanmax(Y100[Y100>0])])
  ax2[2].plot(x,y100,'--r',lw=6)
  #ax2.plot(x,Y1,':m',lw=3)
  ax2[2].plot(x,ymean,'-k',lw=6) 
  #ax2.plot(x,Y3,':m',lw=3)
  ax2[2].plot(x,y0,'--r',lw=6)
  ax2[2].fill_between(x, y100, ymean, color='green',alpha=0.5) 
  ax2[2].fill_between(x, ymean, y0, color='green',alpha=0.5)     
  ax2[2].set_ylabel(r"$\dfrac{{\rm d}V}{{\rm d} \log D} \ (\rm nm^3 \ cm^{-3})$") #,,fontsize=40font='serif',fontname="Times New Roman"
  ax2[2].set_xlabel(r"Dry $D$ (nm)") 
  ax2[2].set_xscale("log")
  ax2[2].set_yscale("log")      
  ax2[2].set_ylim(y_min_max[0],y_min_max[1]) # cut y-axis off at zero   
  ax2[2].set_xlim(x_min_max[0],x_min_max[1])
  xtklbls = ["%i"%(x[ix]) for ix in range(0,len(x)+1,10)]
  ax2[2].set_xticks(x[range(0,len(x)+1,10)], xtklbls)   
  # set the line widths of the axes
  for axis in ['top','bottom','left','right']:
     ax2[2].spines[axis].set_linewidth(1.5)     
  ax2[2].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
  ax2[2].tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2[2].tick_params(axis='both',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2[2].get_xticklabels():
     label.set_horizontalalignment('center')
  at = AnchoredText(FIGLBLS[2], prop=dict(size=fs), frameon=False, loc='upper left')
  ax2[2].add_artist(at)   
  #display and save figure using the *.ict data filename 
  plt.subplots_adjust(left=0.21, bottom=0.10, right=0.95, top=0.95)
  plt.savefig(f"{consit_file_location}_DataRetrievals_SD_plot", dpi=300)
  plt.show() # function to display the plot        
  plt.close() #   
  cols = ["Dp","0","10","25","50","75","90","100","mean","stdev","-99confint","+99confint","count"]
  colnames = ','.join(e for e in cols)
  rows = ["%0.1f"%(x[ix]) for ix in range(0,len(x))]
  str_data = np.char.mod("%10.6f", stats_sd.T)
  str_data= np.column_stack((rows,str_data))
  output_filename = f"{camp_name}/InternalConsistency/{output_filename_suffix}_DataRetrievals_SD_stats.csv"
  with open(output_filename, 'w') as f:
      np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)  

  rcParams['figure.figsize'] = 12, 15
  fig,ax2=plt.subplots(3, 1) # create figure and subplot
  #fig,ax2=plt.subplots(1, 1) # create figure and subplot
  FIGLBLS = ["(a)","(b)","(c)"]
  ttl = ["IRI",r"$\kappa$",r"$f$(RH)"]  
  ymin = 0
  for i2 in [0,1,2]:
    y = y2[i2,:]  
    ymax = Bin[Lst[i2]][-1]
    #xymax = 150
    #  ax2[i2].set_xscale("log")
    ax2[i2].set_yscale("log")   
    ax2[i2].hist(y,bins=Bin[Lst[i2]]) 
    ax2[i2].set_xlabel(f'{ttl[i2]}', fontsize=fs) # set xaxis label 
    #ax2[i2].set_title(f'{ttl[i2]}', fontsize=fs) #set title as flight date.
    ax2[i2].set_ylabel("count", fontsize=fs) # set yaxis label   
    #ax2[i2].set_ylim(xymin[i2],xymax) # cut y-axis off at zero   
    ax2[i2].set_xlim(ymin,ymax)    
    # set the line widths of the axes
    for axis in ['top','bottom','left','right']:
        ax2[i2].spines[axis].set_linewidth(1.5)     
    ax2[i2].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
    ax2[i2].tick_params(axis='both', labelsize=fs, rotation=0)  
    ax2[i2].tick_params(axis='y',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width  
    for label in ax2[i2].get_xticklabels():
        label.set_horizontalalignment('center')
    #display and save figure using the *.ict data filename 
    at = AnchoredText(FIGLBLS[i2], prop=dict(size=fs), frameon=False, loc='upper right')
    ax2[i2].add_artist(at)
  plt.tight_layout()  
  #plt.subplots_adjust(bottom=0.1, right=0.75, top=0.9)  
  plt.savefig(f"{consit_file_location}_DataRetrievals_histograms", dpi=300)
  plt.show() # function to display the plot        
  plt.close() ##  

  #Mectar functions
  def forward(a):
      a = np.deg2rad(a)
      return np.rad2deg(np.log(np.abs(np.tan(a) + 1.0 / np.cos(a))))    

  def inverse(a):
      a = np.deg2rad(a)
      return np.rad2deg(np.arctan(np.sinh(a)))
  rcParams['figure.figsize'] = 8, 4
  fig,ax2=plt.subplots(1, 3) # create figure and subplot
  #fig,ax2=plt.subplots(1, 1) # create figure and subplot 
  FIGLBLS = ["(a)","(b)","(c)"]
  ttl = ["IRI",r"$\kappa$",r"$f$(RH)"] 
  digs = ["%0.2f","%0.1f","%0.1f"] 
  prctillst = np.array([25,50,75])
  ymin = np.nanmin(y)
  for i2 in np.arange(len(FIGLBLS)):
    y = y2[i2,:] 
    y = y[np.where(np.logical_not(np.isnan(y)))] 
    
    position, yy = probscale.plot_pos(y)
    position *= 100
    xmedian = np.median(y)
    print(xmedian)
    #ax2[i2].legend(loc='lower right')
    ax2[i2].plot(yy,position,'.k', linestyle='none')
    ax2[i2].axvline(x=xmedian, linestyle='--', color='xkcd:fuchsia', lw=2)
    ax2[i2].axhline(y=50, linestyle='--', color='xkcd:fuchsia', lw=2) 
    ymax = Bin[Lst[i2]][-1]
    ymin = Bin[Lst[i2]][0]#xymax = 150
    if i2 == 0:
      ax2[i2].set_ylabel('Probability (%)', fontsize=fs)
    #ax2[i2].set_ylabel('')
    #ax2[i2].set_xscale('log')
    #ax2[i2].set_yscale("log")    
    if i2 == 1:
      ax2[i2].set_xlabel(f'{ttl[i2]}', fontsize=fs+4) # set xaxis label 
    else:
      ax2[i2].set_xlabel(f'{ttl[i2]}', fontsize=fs) # set xaxis label   

    #ax2[i2].set_xlim(ymin,ymax) # cut y-axis off at zero
    xtks = np.linspace(Bin[Lst[i2]][0],Bin[Lst[i2]][-1],3)
    xtklbls = [digs[i2]%ix for ix in xtks]
    ax2[i2].set_xticks(xtks, xtklbls) 
    #print(prctillst)  
    # set the line widths of the axes
    for axis in ['top','bottom','left','right']:
        ax2[i2].spines[axis].set_linewidth(1.5)     
    ax2[i2].tick_params(direction='inout', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
    ax2[i2].tick_params(axis='both', labelsize=fs, rotation=0)  
    ax2[i2].tick_params(axis='y',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width  
    #for label in ax2[i2].get_xticklabels():
    #    label.set_horizontalalignment('center')
    #display and save figure using the *.ict data filename 
    at = AnchoredText(FIGLBLS[i2], prop=dict(size=fs), frameon=False, loc='lower right')
    ax2[i2].add_artist(at)
  plt.tight_layout()
  #plt.subplots_adjust(left=0.075,bottom=0.1, right=0.99, top=0.95)  
  plt.savefig(f"{consit_file_location}_DataRetrievals_logprob", dpi=300)
  plt.show() # function to display the plot        
  plt.close() ##
  rsindex += 1
  cols = ["param","0","50","68","95","100","mn","xstdev","confint-","confint+","npt"]
  colnames = ','.join(e for e in cols)  
  rows = Lst
  str_data = np.char.mod("%10.6f", stats_y2.T)
  str_data= np.column_stack((rows,str_data))
  output_filename = f"{camp_name}/InternalConsistency/{output_filename_suffix}_DataRetrievals_Kappa_IRI_stats.csv"
  with open(output_filename, 'w') as f:
        np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)  
 