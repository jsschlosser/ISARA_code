import Load_Size_Dists
import Stats_Code
import numpy as np
import datetime
import itertools
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DayLocator, HourLocator, DateFormatter
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib import rc
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.offsetbox import AnchoredText#
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams#   
def run():
  """
  Performs consistency analyses on synthetically generated dataset generated with ACTIVATE data.   
  
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
  """ 

  def grabvalues(
      dictionaryname,
      startofkeyname
    ):
      OP = dict()
      io = 0
      for key in dictionaryname.item():
        if key.startswith(startofkeyname):
          #print(key,io)
          value = dictionaryname.item().get(key)
          OP[io] = value.T
          io += 1
      return OP  
  def flatten(l):
      return [item for sublist in l for item in sublist]#
  def Line(m,x,b):
      y = m*x + b
      return y#
  def getPercentileList(
      prctile,
      suffix
    ):
      prctile_lst = np.array([f"{x}_percentile_{suffix}" for x in prctile])
      return prctile_lst  

      
  bds1 = np.vstack((np.hstack((0,1,2,range(5,60,5))),np.hstack((0,1,2,range(5,60,5)))))
  bds2 = np.vstack((np.hstack((0,1,2,range(5,60,5))),np.hstack((0,1,2,range(5,60,5)))))
  bds3 = np.vstack((np.hstack((0,1,2,range(5,60,5))),np.hstack((0,1,2,range(5,60,5)))))
  bds4 = np.hstack((0,1,2,range(400,4800,400)))#np.vstack((np.hstack((0,1,2,range(50,600,50))),np.hstack((0,1,2,range(50,600,50)))))  

  fs = 14
  lw = 1.5
  wvl=[0.450,0.550,0.700,0.470,0.532,0.660]#  

  Bin = dict()
  Lst = ["RRI","IRI","Kappa","fRH"]
  Bin["IRI"] = np.arange(0.0, 0.08, 0.002).reshape(-1) 
  Bin["RRI"] = np.arange(1.52, 1.54, 0.001).reshape(-1) 
  Bin["Kappa"] = np.arange(0.0, 1.40, 0.1).reshape(-1) 
  Bin["fRH"] = np.arange(1, 3, 0.5).reshape(-1) 
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
  resolution = np.array([60]) #
  prctile = [0,10,50,68,90,95,100]
  prctile_lst_b = getPercentileList(prctile,"B")
  prctile_lst_ab = getPercentileList(prctile,"AB")
  prctile_lst_rb = getPercentileList(prctile,"RB")
  prctile_lst_arb = getPercentileList(prctile,"ARB")
  prctile_lst_x = getPercentileList(prctile,"x")
  prctile_lst_y = getPercentileList(prctile,"y")
  #cmap = 'jet'
  #print(cmap[0])#  

  rcParams['font.size'] = fs
  #rcParams['axes.formatter.useoffset'] = False    
  plt.rcParams.update({'font.size': fs})
  plt.rcParams['font.family'] = 'serif'
  plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']   #
  plt.rcParams.update({'mathtext.fontset': 'stix',
   'mathtext.rm': 'Times New Roman',
   'mathtext.it': 'Times New Roman:italic',
   'mathtext.bf': 'Times New Roman:bold'})  

  output_filename_suffix = f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/activate_Synthetic_retrievals"
  output_filename = f'{output_filename_suffix}.npy'
  print(output_filename)
  OP_Dictionary = np.load("./%s"%output_filename,allow_pickle='TRUE')
  CRI_flag = grabvalues(OP_Dictionary,'attempt_flag_CRI')[0]
  Npt00 = np.nansum(CRI_flag == 0)
  print(f'Number of points without enough data: {Npt00}')
  Npt01 = np.nansum(CRI_flag == 1)
  Npt02 = np.nansum(CRI_flag == 2)
  print(f'Attempts made: {Npt01+Npt02}')
  print(f'Number of successful CRI retrievals: {Npt02}')    
  print('(Successes)/(Attempts)x100: %i'%((Npt02/(Npt01+Npt02))*100),'%')
  k_flag = grabvalues(OP_Dictionary,'attempt_flag_kappa')[0]
  Npt10 = np.nansum(k_flag == 0)
  print(f'Number of points without enough data: {Npt10}')
  Npt11 = np.nansum(k_flag == 1)
  Npt12 = np.nansum(k_flag == 2)
  print(f'Attempts made: {Npt11+Npt12}')
  print(f'Number of successful kappa retrievals: {Npt12}')    
  print('(Successes)/(Attempts)x100: %i'%((Npt12/(Npt11+Npt12))*100),'%') 

  dry_wvl = OP_Dictionary.item().get('dry_wavelengths')  
  wet_wvl = OP_Dictionary.item().get('wet_wavelengths') 
  IRI = grabvalues(OP_Dictionary,'dry_IRI')[0]
  RRI = grabvalues(OP_Dictionary,'dry_RRI')[0]
  kappa = grabvalues(OP_Dictionary,'kappa')[0]  
  Sc = dict()
  Abs = dict()
  SSA = dict()
  Cal_Sc = dict()
  Cal_Abs = dict()
  Cal_SSA = dict()
  Lwvl = len(dry_wvl["sca"])
  for iwvl in range(Lwvl):
      Sc[f'{dry_wvl["sca"][iwvl]}'] = grabvalues(OP_Dictionary,f'dry_meas_sca_coef_{dry_wvl["sca"][iwvl]}')[0]
      Abs[f'{dry_wvl["abs"][iwvl]}'] = grabvalues(OP_Dictionary,f'dry_meas_abs_coef_{dry_wvl["abs"][iwvl]}')[0]
      SSA[f'{dry_wvl["sca"][iwvl]}'] = Sc[f'{dry_wvl["sca"][iwvl]}']/(Sc[f'{dry_wvl["sca"][iwvl]}']+Abs[f'{dry_wvl["abs"][iwvl]}'])
      Cal_Sc[f'{dry_wvl["sca"][iwvl]}'] = grabvalues(OP_Dictionary,f'dry_cal_sca_coef_{dry_wvl["sca"][iwvl]}')[0]
      Cal_Abs[f'{dry_wvl["abs"][iwvl]}'] = grabvalues(OP_Dictionary,f'dry_cal_abs_coef_{dry_wvl["abs"][iwvl]}')[0]
      Cal_SSA[f'{dry_wvl["sca"][iwvl]}'] = grabvalues(OP_Dictionary,f'dry_cal_SSA_{dry_wvl["sca"][iwvl]}')[0] 
  x0 = dict()
  x0[0] = Sc
  x0[1] = Abs   
  Sc_wet = grabvalues(OP_Dictionary,"wet_meas_sca_coef_550")[0]
  SSA_wet = Sc_wet/grabvalues(OP_Dictionary,"wet_meas_ext_coef_550")[0]
  fRH = grabvalues(OP_Dictionary,'meas_fRH')[0]
  Cal_Sc_wet = grabvalues(OP_Dictionary,"wet_cal_sca_coef_550")[0]
  Cal_SSA_wet = grabvalues(OP_Dictionary,"wet_cal_ext_coef_550")[0]
  Cal_fRH = grabvalues(OP_Dictionary,'cal_fRH')[0]  
  x1 = dict()
  x1[0] = Sc_wet
  #y1[1] = SSA_wet[0]
  x1[1] = fRH 
  y0 = dict()
  y0[0] = Cal_Sc
  y0[1] = Cal_Abs   
  y1 = dict()
  y1[0] = Cal_Sc_wet
  y1[1] = Cal_fRH 
  fit_params_mode0 = np.squeeze(grabvalues(OP_Dictionary,'SD_fit_params_mode0')[0])
  fit_params_mode1 = np.squeeze(grabvalues(OP_Dictionary,'SD_fit_params_mode1')[0]) 

  fitparams = np.column_stack((fit_params_mode0,fit_params_mode1))
  stats_fitparams = StatsCode.Survey(fitparams.T,prctile) 

  cols = ["param","0","50","68","95","100","mn","xstdev","confint-","confint+","npt"]
  colnames = ','.join(e for e in cols)  
  rows = ["mode0_N","mode0_GM","mode0_GSD","mode1_N","mode1_GM","mode1_GSD"]
  str_data = np.char.mod("%10.6f", stats_fitparams)
  str_data= np.column_stack((rows,str_data.T))
  output_filename = f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_DataRetrievals_SD_Fit_stats.csv"
  with open(output_filename, 'w') as f:
       np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)   

  IRI_m = np.squeeze(grabvalues(OP_Dictionary,'synthetic_IRI')[0]).reshape(1,-1)
  RRI_m = np.squeeze(grabvalues(OP_Dictionary,'synthetic_RRI')[0]).reshape(1,-1)
  kappa_m = np.squeeze(grabvalues(OP_Dictionary,'synthetic_kappa')[0]).reshape(1,-1)
  x2 = np.vstack((RRI_m,IRI_m,kappa_m,fRH))   

  y2 = np.vstack((RRI,IRI,kappa,Cal_fRH))
  stats_y2 = StatsCode.Survey(y2,prctile)
  print(len(kappa)) 

  dpg = grabvalues(OP_Dictionary,'synthetic_dpg')[0]
  SD = grabvalues(OP_Dictionary,'noisy_SD_Bin')
  sd1 = np.zeros((len(dpg),len(SD[0])))
  print(len(SD[0]))
  for i1 in range(len(SD)):
    sd1[i1,:] = SD[i1]  
  

  i0 = 1  # index used to skip header row
  G = open('../ISARA_data_files/ACTIVATE/InternalConsistency/activate-mrg-activate-large-smps_hu25_DataRetrievals_SD_stats.csv', 'r')  # open .csv
  g = G.read().splitlines()  # read .csv
  hdrs = g[0].split(',')  # define headers
  sdfm = np.empty((len(g) - 1, len(hdrs))) # create empty array to be filled iteratively
  sdfm[:] = np.nan
  for i1 in range(len(g) - 1):
      sdfm[i1, :] = np.array(list(eval(g[i0]))) # split string into array and define as number array
      i0 += 1 

  D_grd = {}
  D_grd["dpg"] = np.logspace(0,4, num=64, endpoint=True, base=10.0)
  D_grd["lbls"] = np.logspace(0,4,5, base=10.0) 

  prctile = [10,50,90]
  stats_sd = StatsCode.Survey(sd1,prctile)
  pltidx = np.zeros((len(wvl),2)).astype(int)
  j1 = 0
  j2 = 0
  for i1 in range(len(wvl)):
   if i1 < 3:
     pltidx[i1,:] = [j1,j2]
     j1 = j1 + 1
   elif i1 == 3:
     j1 = 0
     j2 = 1
     pltidx[i1,:] = [j1,j2]
   else:
     j1 = j1 + 1
     pltidx[i1,:] = [j1,j2] 

  ttl = ["Dry Scattering", "Dry Absorption"]
  FIGLBLS = np.array([["(a)","(b)"],["(c)","(d)"],["(e)","(f)"]])
  rcParams['figure.figsize'] = 7.5, 10
  fig,ax2=plt.subplots(3, 2) # create figure and subplot  

  xymax = np.array([200,50])#
  bounds = bds1[0,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)#
  stats_dict = np.zeros((len(wvl)+9,71))  

  i0 = 0
  for i1 in range(len(wvl)):
   # Create heatmap
   if i1 < 3:
     x = x0[0][f'{dry_wvl["sca"][i1]}']
     y = y0[0][f'{dry_wvl["sca"][i1]}']
   else:
     x = x0[1][f'{dry_wvl["abs"][i1-3]}']
     y = y0[1][f'{dry_wvl["abs"][i1-3]}']  #
   idx = np.where((np.logical_not(np.isnan(y)))&(np.logical_not(np.isnan(x))))[0]
   Npt = len((y[idx]))  
   
   x = x[idx]
   y = y[idx]
   #x = x[y>0]
   #y = y[y>0]  
   prctile = [0,10,50,68,90,95,100]
   stats_dict[i0,:] = np.hstack((wvl[i1],StatsCode.Comparison(x,y,prctile),Npt00,Npt01+Npt02,Npt02))
   i0 += 1   #
  #  y = list(itertools.chain(*y))
  #  x = list(itertools.chain(*x))  #
   H, xedges, yedges = np.histogram2d(x, y, bins=(64,64),range=([[0, xymax[pltidx[i1,1]]], [0, xymax[pltidx[i1,1]]]]))
   H = H.T
   X, Y = np.meshgrid(xedges, yedges)
   # Plot heatmap
  #  im = ax2[pltidx[i1,0],pltidx[i1,1]].pcolormesh(X,Y,np.where(H == 0, np.nan, H), cmap=cmap, vmin=1, vmax=100)
   im = ax2[pltidx[i1,0],pltidx[i1,1]].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
   ax2[pltidx[i1,0],pltidx[i1,1]].set_facecolor(gry)
   if i1 < 3: 
     ax2[pltidx[i1,0],pltidx[i1,1]].set_ylabel("synthetic", fontsize=fs) # set yaxis label   
   if pltidx[i1,0] == 0:
     ax2[pltidx[i1,0],pltidx[i1,1]].set_title("%s\n(Mm$^{-1}$)"%(ttl[pltidx[i1,1]]), fontsize=fs) #set title as flight date.
   elif pltidx[i1,0] == 2:
     ax2[pltidx[i1,0],pltidx[i1,1]].set_xlabel("ISARA-derived", fontsize=fs) # set xaxis label 
   ax2[pltidx[i1,0],pltidx[i1,1]].set_ylim(0,xymax[pltidx[i1,1]]) # cut y-axis off at zero   
   ax2[pltidx[i1,0],pltidx[i1,1]].set_xlim(0,xymax[pltidx[i1,1]])    
   # set the line widths of the axes
   for axis in ['top','bottom','left','right']:
       ax2[pltidx[i1,0],pltidx[i1,1]].spines[axis].set_linewidth(1.5)     
   ax2[pltidx[i1,0],pltidx[i1,1]].tick_params(direction='inout', length=8, width=lw) # set inside facing ticks, ticklength, and tick line width
   ax2[pltidx[i1,0],pltidx[i1,1]].tick_params(axis='both', labelsize=fs, rotation=0)  
   for label in ax2[pltidx[i1,0],pltidx[i1,1]].get_xticklabels():
       label.set_horizontalalignment('center')
   at = AnchoredText(FIGLBLS[pltidx[i1,0],pltidx[i1,1]], prop=dict(size=fs), frameon=False, loc='upper left')
   ax2[pltidx[i1,0],pltidx[i1,1]].add_artist(at)
   ytks = ax2[pltidx[i1,0],pltidx[i1,1]].get_yticks()
   ytklbls = ["%i"%ix for ix in ytks]
   xtklbls = ["%i"%ix for ix in ytks]
   xtklbls[0] = ""
   ax2[pltidx[i1,0],pltidx[i1,1]].set_xticks(ytks, xtklbls) 
   ax2[pltidx[i1,0],pltidx[i1,1]].set_yticks(ytks, ytklbls) 
   ax2[pltidx[i1,0],pltidx[i1,1]].plot(ytks,ytks,'--',color='xkcd:fuchsia',lw=lw)  
  # display and save figure using the *.ict data filename
  plt.subplots_adjust(bottom=0.1, right=0.77, top=0.9)
  cax = plt.axes([0.8, 0.1, 0.055, 0.8])
  #cbar = plt.colorbar(im,cax=cax,ticks=np.hstack((1, range(10, 100, 10))))
  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
  cbar.set_ticklabels(boundsLbs) 
  cbar.ax.tick_params(length=8, width=lw, which="major")
  cbar.outline.set_linewidth(lw)
  #cbar.ax.get_yaxis().set_ticks([])
  #for j, lab in enumerate(['0','$1$','$10$','$20$','$40$','$50$','$60$','$70$','$80$','$90$','$>100$']):
  cbar.set_label('count',labelpad=-10)        
  plt.savefig(f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_DataRetrievals_1to1_dry", dpi=300)
  plt.close() # # 

  bounds = bds2[0,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)#
  rcParams['figure.figsize'] = 5, 10
  fig,ax2=plt.subplots(3, 1) # create figure and subplot
  #xymax = np.array([np.nanmax(np.vstack((Cal_SSA,SSA))),
  #                  np.nanmax(np.vstack((Cal_SSA,SSA)))])
  #fig,ax2=plt.subplots(1, 1) # create figure and subplot
  FIGLBLS = ["(a)","(b)","(c)"]
  ttl = "SSA"
  ws = [450,550,700]
  xymin = 0.6  
  xymax = 1.00
   #xymax = 150
  for i2 in [0,1,2]:
   # Create heatmap
   x = SSA[f'{dry_wvl["sca"][i2]}']
   y = Cal_SSA[f'{dry_wvl["sca"][i2]}'] #
   idx = np.where((np.logical_not(np.isnan(y)))&(np.logical_not(np.isnan(x))))[0]
   Npt = len((y[idx]))  
   x = x[idx]
   y = y[idx]
   stats_dict[i0,:] = np.hstack((ws[i2],StatsCode.Comparison(x,y,prctile),Npt00,Npt01+Npt02,Npt02))
   i0 += 1
   H, xedges, yedges = np.histogram2d(x, y, bins=(64,64),range=([[xymin, xymax], [xymin, xymax]]))
   H = H.T
   X, Y = np.meshgrid(xedges, yedges)
   # Plot heatmap
   im = ax2[i2].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
   ax2[i2].set_facecolor(gry)
   if i2 == 2:
     ax2[i2].set_xlabel("ISARA-derived", fontsize=fs) # set xaxis label 
   if i1 == 1:
     ax2[i2].set_title(f'{ttl}', fontsize=fs) #set title as flight date.
   ax2[i2].set_ylabel("synthetic", fontsize=fs) # set yaxis label   
   ax2[i2].set_ylim(xymin,xymax) # cut y-axis off at zero   
   ax2[i2].set_xlim(xymin,xymax)    
   # set the line widths of the axes
   for axis in ['top','bottom','left','right']:
       ax2[i2].spines[axis].set_linewidth(1.5)     
   ax2[i2].tick_params(direction='inout', length=8, width=lw) # set inside facing ticks, ticklength, and tick line width
   ax2[i2].tick_params(axis='both', labelsize=fs, rotation=0)  
   for label in ax2[i2].get_xticklabels():
       label.set_horizontalalignment('center')
   #display and save figure using the *.ict data filename 
   at = AnchoredText(FIGLBLS[i2], prop=dict(size=fs), frameon=False, loc='upper left')
   ax2[i2].add_artist(at)
   ytks = ax2[i2].get_yticks()
   ytklbls = ["%0.2f"%ix for ix in ytks]
   xtklbls = ["%0.2f"%ix for ix in ytks]
   xtklbls[0] = ""
   ax2[i2].set_xticks(ytks, xtklbls)
   ax2[i2].set_yticks(ytks, ytklbls)
   ax2[i2].plot(ytks,ytks,'--',color='xkcd:fuchsia',lw=1.5)   
  plt.subplots_adjust(left=0.16, bottom=0.1, right=0.725, top=0.9)
  cax = plt.axes([0.775, 0.1, 0.055, 0.8])
  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
  cbar.set_ticklabels(boundsLbs) 
  cbar.outline.set_linewidth(1.5)
  cbar.ax.tick_params(length=8, width=lw, which="major")
  cbar.set_label('count',labelpad=-10)
  plt.savefig(f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_DataRetrievals_1to1_SSA", dpi=300) 

  dattype = ["%i","%0.1f"]
  xymin = np.array([0,1])   
  xymax_ary = np.array([500,10])
  bounds = bds3[0,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)#
  ttl = ["Wet Scattering (Mm$^{-1}$)","f(RH)"]
  rcParams['figure.figsize'] = 5, 7.5
  fig,ax2=plt.subplots(2, 1) # create figure and subplot
  #xymax = np.array([np.nanmax(np.vstack((Cal_SSA,SSA))),
  #                  np.nanmax(np.vstack((Cal_SSA,SSA)))])
  #fig,ax2=plt.subplots(1, 1) # create figure and subplot
  FIGLBLS = ["(a)","(b)"]
  for i2 in x1:
    # Create heatmap
    x = x1[i2]
    y = y1[i2]
    idx = np.where((np.logical_not(np.isnan(y)))&(np.logical_not(np.isnan(x))))[0]
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
  plt.savefig(f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_DataRetrievals_1to1_amb", dpi=300)        
  plt.close() # 

  bounds = bds2[0,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)#
  rcParams['figure.figsize'] = 5, 7.5
  fig,ax2=plt.subplots(2, 1) # create figure and subplot
  #xymax = np.array([np.nanmax(np.vstack((Cal_SSA,SSA))),
  #                  np.nanmax(np.vstack((Cal_SSA,SSA)))])
  #fig,ax2=plt.subplots(1, 1) # create figure and subplot
  FIGLBLS = ["(a)","(b)"]
  ttl = ["IRI",r"$\kappa$"]  #
  xymin = [0,0]
  xymax = [0.08,1.4]
  i2 = 0
  for i3 in [1,2]:
    # Create heatmap
    x = x2[i3,:]
    y = y2[i3,:]
    Npt = len(y[np.where(np.logical_not(np.isnan(y)))[0]])  
    idx = np.where((np.logical_not(np.isnan(y)))&(np.logical_not(np.isnan(x))))[0]
    x = x[idx]
    y = y[idx]
    H, xedges, yedges = np.histogram2d(x, y, bins=(64,64),range=([[xymin[i2], xymax[i2]], [xymin[i2], xymax[i2]]]))
    H = H.T
    X, Y = np.meshgrid(xedges, yedges)
    # Plot heatmap
    im = ax2[i2].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
    ax2[i2].set_facecolor(gry)
    if i2 == 2:
      ax2[i2].set_xlabel("ISARA-derived", fontsize=fs) # set xaxis label 
    ax2[i2].set_title(f'{ttl[i2]}', fontsize=fs+4) #set title as flight date.
    ax2[i2].set_ylabel("synthetic", fontsize=fs) # set yaxis label   
    ax2[i2].set_ylim(xymin[i2],xymax[i2]) # cut y-axis off at zero   
    ax2[i2].set_xlim(xymin[i2],xymax[i2])    
    # set the line widths of the axes
    for axis in ['top','bottom','left','right']:
        ax2[i2].spines[axis].set_linewidth(1.5)     
    ax2[i2].tick_params(direction='inout', length=8, width=lw) # set inside facing ticks, ticklength, and tick line width
    ax2[i2].tick_params(axis='both', labelsize=fs, rotation=0)  
    for label in ax2[i2].get_xticklabels():
        label.set_horizontalalignment('center')
    #display and save figure using the *.ict data filename 
    at = AnchoredText(FIGLBLS[i2], prop=dict(size=fs), frameon=False, loc='upper left')
    ax2[i2].add_artist(at)
    ytks = ax2[i2].get_yticks()
    ytklbls = ["%0.2f"%ix for ix in ytks]
    xtklbls = ["%0.2f"%ix for ix in ytks]
    xtklbls[0] = ""
    ax2[i2].set_xticks(ytks, xtklbls)
    ax2[i2].set_yticks(ytks, ytklbls)
    ax2[i2].plot(ytks,ytks,'--',color='xkcd:fuchsia',lw=2)   
    i2 += 1
  plt.subplots_adjust(left=0.16, bottom=0.1, right=0.72, top=0.9)
  cax = plt.axes([0.775, 0.1, 0.055, 0.8])
  cbar =  plt.colorbar(im,cax=cax,cmap=cmap, norm=norm,boundaries=bounds,ticks=bounds,format='%1i')
  cbar.set_ticklabels(boundsLbs) 
  cbar.outline.set_linewidth(1.5)
  cbar.ax.tick_params(length=8, width=lw, which="major")
  cbar.set_label('count',labelpad=-10)
  plt.savefig(f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_DataRetrievals_1to1_KappaIRI", dpi=300)        
  plt.close() # #
  for i2 in [0,1,2,3]:
    x = x2[i2,:]
    y = y2[i2,:]
    Npt = len(y[np.where(np.logical_not(np.isnan(y)))[0]])  
    idx = np.where((np.logical_not(np.isnan(y)))&(np.logical_not(np.isnan(x))))[0]
    x = x[idx]
    y = y[idx]
    stats_dict[i0,:] = np.hstack((550,StatsCode.Comparison(x,y,prctile),Npt10,Npt11+Npt12,Npt12))
    i0 += 1 

  cols = ['param','Dry_Scattering', 'Dry_Scattering', 'Dry_Scattering', 'Dry_Absorption', 
         'Dry_Absorption', 'Dry_Absorption', 'Dry_SSA', 'Dry_SSA', 'Dry_SSA', 'Amb_Extinction', 'Amb_SSA','Dry_RRI','Dry_IRI','Kappa','f(RH)']
  colnames = ','.join(str(e) for e in cols)
  rows = np.hstack(('wavelength_nm','R','log10_p-value',prctile_lst_b,'mean_b','stdev_b',prctile_lst_ab,'mean_ab','stdev_ab',
                         prctile_lst_rb,'mean_rb','stdev_rb',prctile_lst_arb, 'mean_arb','stdev_arb','NMAD','MAD','NRMSD','RMSD',prctile_lst_x,'mean_x','stdev_x',
                          prctile_lst_y,'mean_y','stdev_y','MoranI','MoranEI','MoranI_znorm','MoranI_pnorm',
                          'MoranI_zrand','MoranI_prand','count','not_enough_measurements','total_attempts','successful_retrievals'))
  str_data = np.char.mod("%10.6f", stats_dict.T)
  str_data= np.column_stack((rows,str_data))
  output_filename = f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_DataRetrievals_1to1_stats.csv" 

  with open(output_filename, 'w') as f:
     np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)  #  

  rcParams['figure.figsize'] = 4.75, 5
  FIGLBLS = ["(a)","(b)"]
  fig,ax2=plt.subplots(2, 1) # create figure and subplot
  x = dpg*1000
  x_min_max = np.array([x[0],x[-1]])
  Y100 = stats_sd[2,:]
  Y0 = stats_sd[0,:]
  Ymean = stats_sd[3,:]
  y_min_max = np.array([0.1,10**7])  
  #y_min_max = np.array([np.nanmin(Y0[Y0>0]),np.nanmax(Y100[Y100>0])])
  ax2[0].plot(x,Y100,'--r',lw=2)
  ax2[0].plot(x,Ymean,'-k',lw=2) 
  ax2[0].plot(x,Y0,'--r',lw=2)
  ax2[0].fill_between(x, Y100, Ymean, color='green',alpha=0.5) 
  ax2[0].fill_between(x, Ymean, Y0, color='green',alpha=0.5)     
  ax2[0].set_ylabel(r"$\dfrac{{\rm d}N}{{\rm d} \log D} \ (\rm cm^{-3})$") #,,fontsize=40font='serif',fontname="Times New Roman"
  #ax2[0].set_xlabel(r"Dry $D$ (nm)") 
  ax2[0].set_xscale("log")
  ax2[0].set_yscale("log")      
  ax2[0].set_ylim(y_min_max[0],y_min_max[1]) # cut y-axis off at zero   
  ax2[0].set_xlim(x_min_max[0],x_min_max[1])
  xtklbls = ["" for ix in range(0,len(x)+1,10)]
  ax2[0].set_xticks(x[range(0,len(x)+1,10)], xtklbls)   
  # set the line widths of the axes
  for axis in ['top','bottom','left','right']:
     ax2[0].spines[axis].set_linewidth(1.5)     
  ax2[0].tick_params(direction='inout', length=8, width=lw) # set inside facing ticks, ticklength, and tick line width
  ax2[0].tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2[0].tick_params(axis='both',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2[0].get_xticklabels():
     label.set_horizontalalignment('center')
  at = AnchoredText(FIGLBLS[0], prop=dict(size=fs), frameon=False, loc='upper right')
  ax2[0].add_artist(at) 

  Y100 = np.multiply(stats_sd[2,:],(np.pi/6)*dpg**3)
  Y0 = np.multiply(stats_sd[0,:],(np.pi/6)*dpg**3)
  Ymean = np.multiply(stats_sd[3,:],(np.pi/6)*dpg**3)
  y_min_max = np.array([10**(-3),10**4])  
  #y_min_max = np.array([np.nanmin(Y0[Y0>0]),np.nanmax(Y100[Y100>0])])
  #ax2[1].plot(x,Y100,'--r',lw=2)
  #ax2[1].plot(x,Ymean,'-k',lw=2) 
  #ax2[1].plot(x,Y0,'--r',lw=2)
  ax2[1].fill_between(x, Y100, Ymean, color='green',alpha=0.5) 
  ax2[1].fill_between(x, Ymean, Y0, color='green',alpha=0.5)     
  ax2[1].set_ylabel(r"$\dfrac{{\rm d}V}{{\rm d} \log D} \ (\rm nm^3 \ cm^{-3})$") #,,fontsize=40font='serif',fontname="Times New Roman"
  ax2[1].set_xlabel(r"Dry $D$ (nm)") 
  ax2[1].set_xscale("log")
  ax2[1].set_yscale("log")      
  ax2[1].set_ylim(y_min_max[0],y_min_max[1]) # cut y-axis off at zero   
  ax2[1].set_xlim(x_min_max[0],x_min_max[1])
  xtklbls = ["%i"%(x[ix]) for ix in range(0,len(x)+1,10)]
  ax2[1].set_xticks(x[range(0,len(x)+1,10)], xtklbls)   
  # set the line widths of the axes
  for axis in ['top','bottom','left','right']:
     ax2[1].spines[axis].set_linewidth(1.5)     
  ax2[1].tick_params(direction='inout', length=8, width=lw) # set inside facing ticks, ticklength, and tick line width
  ax2[1].tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2[1].tick_params(axis='both',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2[1].get_xticklabels():
     label.set_horizontalalignment('center')
  at = AnchoredText(FIGLBLS[1], prop=dict(size=fs), frameon=False, loc='upper left')
  ax2[1].add_artist(at)     

  #display and save figure using the *.ict data filename 
  plt.subplots_adjust(left=0.21, bottom=0.10, right=0.95, top=0.95)
  plt.savefig(f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_FIMS&LAS_dry", dpi=300)        
  plt.close() # #
  cols = ["Dp","0","10","25","50","75","90","100","mean","stdev","count"]
  colnames = ','.join(e for e in cols)
  rows = ["%i"%(x[ix]) for ix in range(0,len(x))]
  str_data = np.char.mod("%10.6f", stats_sd.T)
  str_data= np.column_stack((rows,str_data))
  output_filename = f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_DataRetrievals_FIMS&LAS_SD_stats.csv"
  with open(output_filename, 'w') as f:
     np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)  #  

  rcParams['figure.figsize'] = 6, 7.5
  FIGLBLS = ["(a)","(b)","(c)"]
  bounds = bds4#[rsindex,:]
  lenbnds = len(bounds)
  boundsLbs = bounds.astype(str)
  boundsLbs[lenbnds-2] = f">{boundsLbs[lenbnds-2]}"
  boundsLbs[lenbnds-1] = ""
  norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
  fig,ax2=plt.subplots(3, 1) # create figure and subplot
  y = sd1.T
  y_min_max = np.array([0.01,10**7])  
  ybins = np.logspace(-1,7, num=64, endpoint=True, base=10.0)
  xbins = dpg*1000
  x = np.ones((len(y[:,0]),1))*xbins
  x = np.squeeze(x.reshape(-1,1))
  x_min_max = np.array([D_grd["dpg"][0],D_grd["dpg"][-1]])
  y = np.squeeze(y.reshape(1,-1))
  Y100 = stats_sd[2,:]
  Y0 = stats_sd[0,:]
  Ymean = stats_sd[3,:]
  Y10_measured = sdfm[:,2]
  Y90_measured = sdfm[:,6]
  #y = list(itertools.chain(*y))
  #x = list(itertools.chain(*x))
  H, xedges, yedges = np.histogram2d(x,y,bins=([xbins, ybins]),range=([x_min_max, y_min_max]))#
  H = H.T
  X, Y = np.meshgrid(xedges, yedges)#
  im = ax2[0].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
  ax2[0].set_facecolor(gry) 
  #y_min_max = np.array([np.nanmin(Y10[Y10>0]),np.nanmax(Y10[Y10>0])])
  #ax2[0].plot(xbins,Y100,'--',color='xkcd:fuchsia',lw=2)
  #ax2[0].plot(xbins,Ymean,'-',color='xkcd:fuchsia',lw=2) 
  #ax2[0].plot(xbins,Y0,'--',color='xkcd:fuchsia',lw=2)
  ax2[0].plot(xbins,Y10_measured,'--',color='xkcd:fuchsia',lw=2)#,'.',color='xkcd:red',lw=2)#
  ax2[0].plot(xbins,Y90_measured,'--',color='xkcd:fuchsia',lw=2)#,'.',color='xkcd:red',lw=2)#
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
  ax2[0].tick_params(direction='inout', length=8, width=lw) # set inside facing ticks, ticklength, and tick line width
  ax2[0].tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2[0].tick_params(axis='both',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2[0].get_xticklabels():
    label.set_horizontalalignment('center')
  at = AnchoredText(FIGLBLS[0], prop=dict(size=fs), frameon=False, loc='upper left')
  ax2[0].add_artist(at)
  y = np.multiply(sd1.T,(np.pi)*(dpg**2))
  y = np.squeeze(y.reshape(-1,1))
  Y100 = np.multiply(stats_sd[2,:],(np.pi)*(dpg)**2)
  Y0 = np.multiply(stats_sd[0,:],(np.pi)*(dpg)**2)
  Ymean = np.multiply(stats_sd[3,:],(np.pi)*(dpg)**2)
  Y10_measured = np.multiply(sdfm[:,2],(np.pi)*(dpg)**2)
  Y90_measured = np.multiply(sdfm[:,6],(np.pi)*(dpg)**2)
  y_min_max = np.array([np.nanmin(Y0),np.nanmax(Y100)])  
  y_min_max = np.array([10**(-5),10**4])  
  ybins = np.logspace(-8,4, num=64, endpoint=True, base=10.0)
  H, xedges, yedges = np.histogram2d(x,y,bins=([xbins, ybins]),range=([x_min_max, y_min_max]))#
  H = H.T
  X, Y = np.meshgrid(xedges, yedges)#
  im = ax2[1].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
  ax2[1].set_facecolor(gry) 
  #y_min_max = np.array([np.nanmin(Y0[Y0>0]),np.nanmax(Y100[Y100>0])])
  #ax2[1].plot(xbins,Y100,'--',color='xkcd:fuchsia',lw=2)
  #ax2[1].plot(xbins,Ymean,'-',color='xkcd:fuchsia',lw=2) 
  #ax2[1].plot(xbins,Y0,'--',color='xkcd:fuchsia',lw=2)
  ax2[1].plot(xbins,Y10_measured,'--',color='xkcd:fuchsia',lw=2)#,'.',color='xkcd:red',lw=2)#
  ax2[1].plot(xbins,Y90_measured,'--',color='xkcd:fuchsia',lw=2)#,'.',color='xkcd:red',lw=2)#
  #ax2[1].fill_between(xbins, Y100, Ymean, color='green',alpha=0.5) 
  #ax2[1].fill_between(xbins, Ymean, Y0, color='green',alpha=0.5)     
  ax2[1].set_ylabel(r"$\dfrac{{\rm d}S}{{\rm d} \log D} \ (\rm nm^3 \ cm^{-3})$") #,,fontsize=40font='serif',fontname="Times New Roman"
  ax2[1].set_xscale("log")
  ax2[1].set_yscale("log")      
  ax2[1].set_ylim(y_min_max[0],y_min_max[1]) # cut y-axis off at zero   
  ax2[1].set_xlim(x_min_max[0],x_min_max[1])
  xtklbls = ["" for ix in range(0,len(D_grd["lbls"]))]
  ax2[1].set_xticks(D_grd["lbls"], xtklbls)  
  # set the line widths of the axes
  for axis in ['top','bottom','left','right']:
    ax2[1].spines[axis].set_linewidth(1.5)     
  ax2[1].tick_params(direction='inout', length=8, width=lw) # set inside facing ticks, ticklength, and tick line width
  ax2[1].tick_params(axis='both', labelsize=fs, rotation=0)  
  ax2[1].tick_params(axis='both', which="minor", direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
  for label in ax2[1].get_xticklabels():
    label.set_horizontalalignment('center')
  at = AnchoredText(FIGLBLS[1], prop=dict(size=fs), frameon=False, loc='upper left')
  ax2[1].add_artist(at)   

  y = np.multiply(sd1.T,(np.pi/6)*(dpg**3))
  y = np.squeeze(y.reshape(-1,1))
  Y100 = np.multiply(stats_sd[2,:],(np.pi/6)*(dpg)**3)
  Y0 = np.multiply(stats_sd[0,:],(np.pi/6)*(dpg)**3)
  Ymean = np.multiply(stats_sd[3,:],(np.pi/6)*(dpg)**3)
  Y10_measured = np.multiply(sdfm[:,2],(np.pi/6)*(dpg)**3)
  Y90_measured = np.multiply(sdfm[:,6],(np.pi/6)*(dpg)**3)
  #y_min_max = np.array([np.nanmin(Y0),np.nanmax(Y100)])  
  y_min_max = np.array([10**(-8),10**3])  
  ybins = np.logspace(-8,4, num=64, endpoint=True, base=10.0)
  H, xedges, yedges = np.histogram2d(x,y,bins=([xbins, ybins]),range=([x_min_max, y_min_max]))#
  H = H.T
  X, Y = np.meshgrid(xedges, yedges)#
  im = ax2[2].pcolormesh(X,Y,H, cmap=cmap, norm=norm)
  ax2[2].set_facecolor(gry) 
  #y_min_max = np.array([np.nanmin(Y0[Y0>0]),np.nanmax(Y100[Y100>0])])
  #ax2[2].plot(xbins,Y100,'--',color='xkcd:fuchsia',lw=2)
  #ax2[2].plot(xbins,Ymean,'-',color='xkcd:fuchsia',lw=2) 
  #ax2[2].plot(xbins,Y0,'--',color='xkcd:fuchsia',lw=2)#
  ax2[2].plot(xbins,Y10_measured,'--',color='xkcd:fuchsia',lw=2)#,'.',color='xkcd:red',lw=2)#
  ax2[2].plot(xbins,Y90_measured,'--',color='xkcd:fuchsia',lw=2)#,'.',color='xkcd:red',lw=2)#
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
  ax2[2].tick_params(direction='inout', length=8, width=lw) # set inside facing ticks, ticklength, and tick line width
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
  cbar.ax.tick_params(length=8, width=lw, which="major")
  cbar.set_label('count',labelpad=-10)
  #display and save figure using the *.ict data filename 
  plt.subplots_adjust(left=0.18, bottom=0.11, right=0.75, top=0.95)
  plt.savefig(f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_DataRetrievals_heatmapplot_dry", dpi=300)        
  plt.close() #   
  

  rcParams['figure.figsize'] = 6, 9.5
  fig,ax2=plt.subplots(4, 1) # create figure and subplot
  #fig,ax2=plt.subplots(1, 1) # create figure and subplot
  FIGLBLS = ["(a)","(b)","(c)","(d)"]
  ttl = ["RRI","IRI",r"$\kappa$",r"$f$(RH)"]  #
  yMin = np.array([1.5,0,0,1])
  yMax = np.array([1.6,0.08, 1.4, 4])
  for i2 in range(len(y2[:,0])):
   y = y2[i2,:]  #
   print(np.std(y[np.where(np.logical_not(np.isnan(y)))[0]])," ",ttl[i2])
   ax2[i2].set_yscale("log")   
   ax2[i2].hist(y,bins=Bin[Lst[i2]]) 
   ax2[i2].set_xlabel(f'{ttl[i2]}', fontsize=fs) # set xaxis label 
   ax2[i2].set_ylabel("count", fontsize=fs) # set yaxis label      
   ax2[i2].set_xlim(yMin[i2],yMax[i2])    
   # set the line widths of the axes
   for axis in ['top','bottom','left','right']:
       ax2[i2].spines[axis].set_linewidth(1.5)     
   ax2[i2].tick_params(direction='inout', length=8, width=lw) # set inside facing ticks, ticklength, and tick line width
   ax2[i2].tick_params(axis='both', labelsize=fs, rotation=0)  
   ax2[i2].tick_params(axis='y',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width  #
   for label in ax2[i2].get_xticklabels():
       label.set_horizontalalignment('center')
   #display and save figure using the *.ict data filename 
   at = AnchoredText(FIGLBLS[i2], prop=dict(size=fs), frameon=False, loc='upper right')
   ax2[i2].add_artist(at)
  plt.subplots_adjust(bottom=0.1, right=0.75, top=0.9)  #
  plt.tight_layout()
  plt.savefig(f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_DataRetrievals_histograms", dpi=300)        
  plt.close() ##
  cols = ["param","0","50","68","95","100","mn","xstdev","confint-","confint+","npt"]
  colnames = ','.join(e for e in cols)  
  rows = Lst
  str_data = np.char.mod("%10.6f", stats_y2.T)
  str_data= np.column_stack((rows,str_data))
  output_filename = f"../ISARA_data_files/ACTIVATE/SyntheticRetrievals/Synthetic_DataRetrievals_Kappa_CRI_stats.csv"
  with open(output_filename, 'w') as f:
       np.savetxt(f, str_data, delimiter=', ', fmt='%s', header=colnames)  
