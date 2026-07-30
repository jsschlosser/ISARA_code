import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.offsetbox import AnchoredText
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DayLocator, HourLocator, DateFormatter
from matplotlib.colors import LogNorm
from matplotlib import cm
from matplotlib import rc
from matplotlib.collections import PolyCollection
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams#  

def plot_SD(dp,dndlogdp1,dndlogdp2,filename_suffix,r,p,file_location):

    rcParams['font.size'] = 12
    #rcParams['axes.formatter.useoffset'] = False    
    plt.rcParams.update({'font.size': 12})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']   #
    plt.rcParams.update({'mathtext.fontset': 'stix',
     'mathtext.rm': 'Times New Roman',
     'mathtext.it': 'Times New Roman:italic',
     'mathtext.bf': 'Times New Roman:bold'})

    #with sns.axes_style('ticks'):
    fig, ax = plt.subplots(1, figsize=(6, 3),dpi=300)				
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)  
    ax.tick_params(direction='in', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
    ax.tick_params(axis='both', labelsize=12, rotation=0)
    ax.tick_params(axis='both',which="minor",direction='in', length=4, width=1.5)
    ax.plot(dp*1000, dndlogdp1, marker='o', color='r', linestyle='none', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
    ax.plot(dp*1000, dndlogdp2, '-b',lw=2)				
    ax.set_xlabel(r"$D$ (nm)")
    ax.set_ylabel(r"$\dfrac{{\rm d}N}{{\rm d} \log D} \ (\rm cm^{-3})$")
    ax.semilogx()
    at = AnchoredText("$r$ = %0.3f \n$p$ = %0.3f"%(r,10**(p)), prop=dict(size=14), frameon=False, loc='upper right')
    ax.add_artist(at)
    ax.xaxis.set_major_formatter(mtick.FormatStrFormatter("%.4g"))
    plt.tight_layout()
    #ax.legend()				
    #sns.despine()
    plt.show()
    plt.savefig(f"{file_location}SD_Fit_Results_{filename_suffix}")
    plt.close()