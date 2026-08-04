import StatsCode
import load_sizebins
import collect_netcdf
import numpy as np
import numpy.matlib
import datetime
import itertools
from netCDF4 import Dataset
import matplotlib as mpl
import matplotlib.pyplot as plt 
from matplotlib.ticker import MaxNLocator
from matplotlib.dates import DayLocator, HourLocator, DateFormatter
from matplotlib import cm
from matplotlib.collections import PolyCollection
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.basemap import Basemap
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams
from statsmodels.stats.weightstats import DescrStatsW
from scipy.spatial import KDTree
from scipy.interpolate import pchip_interpolate
import warnings
warnings.simplefilter('ignore')

def CaseStudy(
    camp_name,
    Insitu_Spherical_Retrieval_filename,
    Insitu_Nonspherical_Retrieval_filename,
    Insitu_Nonspherical_Retrieval_filename_2,
    RSP_Filename,
    HSRL2_Filename,
    ouput_filename_suffix,
    time_separation_constraint,
    spatial_separation_constraint,
    out_directory_name,
    source_key_vars,
    LegID_dictionary
):
    """
    Performs a external closure analysis on between data from in-situ, RSP, HSRL files taken on the same day with coincident measurements.   

    :param camp_name: campaign name assocaiated with data file directory.
    :type camp_name: str
    :param Insitu_Spherical_Retrieval_filename: in-situ data file where coarse-mode assumed spherical.
    :type Insitu_Spherical_Retrieval_filename: str
    :param Insitu_Nonspherical_Retrieval_filename: in-situ data file where coarse-mode assumed non-spherical type 1.
    :type Insitu_Nonspherical_Retrieval_filename: str    
    :param Insitu_Nonspherical_Retrieval_filename_2: in-situ data file where coarse-mode assumed non-spherical type 2.
    :type Insitu_Nonspherical_Retrieval_filename_2: str
    :param RSP_Filename: RSP data file.
    :type RSP_Filename: str    
    :param HSRL2_Filename: HSRL data file.
    :type HSRL2_Filename: str  
    :param ouput_filename_suffix: output filename suffix.
    :type ouput_filename_suffix: str  
    :param time_separation_constraint: maximum allowed time separation in minutes.
    :type time_separation_constraint: int
    :param spatial_separation_constraint: maximum allowed spatial separation in km.
    :type spatial_separation_constraint: int  
    :param out_directory_name: ouput directory filename.
    :type out_directory_name: str  
    :param source_key_vars: source variable names that are dependent on each mission/insturment suite.    
    :type source_key_vars: str     
    :param LegID_dictionary: filename correspoding to the file with datetimes of interest.
    :type LegID_dictionary: str                       
    :returns: output_dictionary
    :rtype: dictionary
    """ 
    
    def flatten(l):
        """
        Flattens a list of lists into a single continuous list. 

        :param l: A list containing sublists to be flattened.
        :type l: list of lists
        :returns: A single flat list containing all items from the sublists.
        :rtype: list
        """
        return [item for sublist in l for item in sublist]  

    def Line(m,x,b):
        """
        Calculates the y-value of a linear equation (y = mx + b).   

        :param m: The slope of the line.
        :type m: float or int
        :param x: The independent variable (x-coordinate).
        :type x: float, int, or numpy.ndarray
        :param b: The y-intercept of the line.
        :type b: float or int
        :returns: The calculated y-value(s).
        :rtype: float, int, or numpy.ndarray
        """
        y = m*x + b
        return y

                           
    def getPercentileList(prctile,suffix):
        """
        Create a list of desired percentiles formatted with a specific suffix.  

        :param prctile: A sequence of percentile numbers to be formatted.
        :type prctile: list or numpy.ndarray
        :param suffix: A string to append to the end of each formatted percentile string.
        :type suffix: str
        :returns: An array of formatted percentile strings.
        :rtype: numpy.ndarray
        """
        prctile_lst = np.array([f"{x}_percentile_{suffix}" for x in prctile])
        return prctile_lst 

    def compute_weighted_stats(values, weights):
        """
        Computes weighted mean and weighted sample standard deviation along axis 0,
        ignoring np.nan values.

        :param values: Input array to compute the statistics over.
        :type values: np.ndarray of shape (M, N) or (M, N, P)
        :param weights: An array of weights associated with the values.
        :type weights: np.ndarray of shape (M,) or matching 'values' shape
        :returns: An array where the last dimension holds [weighted_mean, weighted_sample_std].
        :rtype: np.ndarray of shape (N, 2) or (N, P, 2)
        """
        nan_mask = np.isnan(values)# 1. Identify where NaNs live
        
        if weights.ndim == 1:# 2. Standardize weights to match values shape if they are 1D
            broadcast_shape = [values.shape[0]] + [1] * (values.ndim - 1)
            w_full = np.broadcast_to(weights.reshape(broadcast_shape), values.shape).copy()
        else:
            w_full = weights.copy()
        w_full[nan_mask] = 0.0# 3. Suppress NaN impact: set weight to 0.0 and replace NaN with 0.0
        v_clean = np.where(nan_mask, 0.0, values)
        sum_w = np.sum(w_full, axis=0)# 4. Total up valid weights and squared weights along axis 0
        sum_w2 = np.sum(w_full**2, axis=0)
        valid_counts = np.sum(~nan_mask, axis=0)# Count non-NaN data points per grid pixel to ensure we have at least 2 points
        sum_w_safe = np.where(sum_w == 0, np.nan, sum_w)# Catch any slices that have insufficient data to prevent a 0/0 or negative error
        weighted_mean = np.sum(v_clean * w_full, axis=0) / sum_w_safe# 5. Calculate Weighted Mean
        mean_expanded = np.expand_dims(weighted_mean, axis=0)# 6. Calculate Weighted Sample Standard Deviation (Unbiased). Expand mean back along axis 0 so it broadcasts cleanly against v_clean
        biased_variance = np.sum(w_full * (v_clean - mean_expanded)**2, axis=0) / sum_w_safe# Compute biased variance first
        denominator_correction = sum_w_safe**2 - sum_w2# Apply the unbiased sample correction factor for Reliability Weights:Correction = 1 / (1 - (sum(w^2) / sum(w)^2)), which simplifies to: sum(w)^2 / (sum(w)^2 - sum(w^2))
        safe_denom = np.where((denominator_correction <= 0) | (valid_counts < 2), np.nan, denominator_correction)# Guard against division by zero and force NaN if fewer than 2 valid elements exist
        unbiased_variance = biased_variance * (sum_w_safe**2 / safe_denom)
        weighted_sample_std = np.sqrt(unbiased_variance)
        output = np.stack([weighted_mean, weighted_sample_std], axis=-1)# 7. Stack metrics along a new trailing axis. Position 0: Mean | Position 1: Sample Std Dev
        
        return output

    def rebin_3d_by_altitude(altitudes, data, bin_edges):
        """
        Rebins an (M, N, P) array along its second axis (N) using altitude bins,
        collapsing both M and N dimensions to produce an (A, P, 2) output.      

        :param altitudes: 1D array matching the second dimension of 'data'.
        :type altitudes: np.ndarray of shape (N,)
        :param data: Input array. Can contain np.nan.
        :type data: np.ndarray of shape (M, N, P)
        :param bin_edges: 1D array defining the edges of the target altitude bins.
        :type bin_edges: np.ndarray of shape (A + 1,)
        :returns: Output array where the final dimension contains [mean, sample_std_dev].
        :rtype: np.ndarray of shape (A, P, 2)
        """
        M, N, P = data.shape
        A = len(bin_edges) 
        bin_indices = np.digitize(altitudes, bin_edges) # 1. Determine which bin each altitude index belongs to (0-indexed)
        output = np.full((A, P, 2), np.nan)# 2. Allocate the output array tracking the target shape (A, P, 2)
        for bin_idx in range(A):# 3. Loop through each target bin
            # Find which indices of axis N fall into the current bin
            matching_indices = np.where(bin_indices == bin_idx)
            if len(matching_indices) == 0:
                continue  # Leave bin as np.nan if no data points match
            bin_data = data[:, matching_indices, :]# Extract relevant subset along axis 1 -> Shape: (M, matching_count, P)
            # Flatten the M and matching_count dimensions into a single sample dimension
            # Shape becomes: (M * matching_count, P)
            bin_data_flat = bin_data.reshape(-1, P)
            nan_mask = np.isnan(bin_data_flat)# 4. Handle NaNs by masking
            valid_counts = np.sum(~nan_mask, axis=0)# Calculate valid entry counts per grid point (P) along axis 0
            clean_data = np.where(nan_mask, 0.0, bin_data_flat)# Suppress NaNs by turning them to 0 for sum operations
            safe_counts_mean = np.where(valid_counts == 0, np.nan, valid_counts)# 5. Compute the Mean. Prevent divide by zero where a grid cell has 0 valid values
            bin_mean = np.sum(clean_data, axis=0) / safe_counts_mean
            mean_expanded = np.expand_dims(bin_mean, axis=0) # 6. Compute the Sample Standard Deviation (ddof=1). Expand the mean vector back along axis 0 to match clean_data shape for broadcasting
            sq_deviations = np.sum(np.where(nan_mask, 0.0, (clean_data - mean_expanded) ** 2), axis=0)# Sum of squared deviations along axis 0
            safe_counts_std = np.where(valid_counts <= 1, np.nan, valid_counts - 1)# Guard for sample size (ddof=1 requires at least 2 points)
            bin_std = np.sqrt(sq_deviations / safe_counts_std)
            bin_std[np.isnan(bin_std)]==0
            output[bin_idx, :, 0] = bin_mean# 7. Pack results into the output array along the bin axis (A)
            output[bin_idx, :, 1] = bin_std
            
        return output


    def rebin_sum_2d_by_altitude(altitudes, data, bin_edges):
        """
        Rebins an (N, M) array along its first axis (N) using altitude bins,
        calculating the sum of all rows that fall into each target bin.     

        :param altitudes: 1D array containing the original fine altitude coordinates.
        :type altitudes: np.ndarray of shape (N,)
        :param data: 2D array to be aggregated. Can contain np.nan.
        :type data: np.ndarray of shape (N, M)
        :param bin_edges: 1D array defining the edges of the coarser target altitude bins.
        :type bin_edges: np.ndarray of shape (A + 1,)
        :returns: Output array containing the sum of valid elements per bin.
        :rtype: np.ndarray of shape (A, M)
        """
        N, M = data.shape
        A = len(bin_edges) 
        bin_indices = np.digitize(altitudes, bin_edges) 
        output_sums = np.full((A, M),np.nan)# 2. Allocate the output array tracking the target shape (A, M). 
        for bin_idx in range(A):# 3. Loop through each target bin to aggregate rows
            matching_indices = np.where(bin_indices == bin_idx)[0]
            bin_data = data[matching_indices, :]
            output_sums[bin_idx, :] = np.nansum(bin_data, axis=0)
        return output_sums


    def bin_data(x_data, y_data, grid):
        """
        Computes binned means and standard deviations across one or more dimensions of Y.       

        :param x_data: 1D array of length N.
        :type x_data: np.ndarray
        :param y_data: Array to be binned along with x_data.
        :type y_data: np.ndarray of shape (N,), (N, M), or (N, M, P)
        :param grid: 1D array defining bin edges.
        :type grid: np.ndarray
        :returns: 
            Binned means and standard deviations.
            - If y_data is (N,): Array of shape (n_bins, 2) -> [mean, std]
            - If y_data is (N, M): Array of shape (n_bins, M, 2) -> [..., 0]=mean, [..., 1]=std
            - If y_data is (N, M, P): Array of shape (n_bins, M, P, 2) -> [..., 0]=mean, [..., 1]=std
        :rtype: array
        """
        #
        orig_ndim = y_data.ndim #1. Track original dimensions to format the output later
        if orig_ndim == 1:# 2. Normalize y_data to always be a 3D matrix: (N, M, P)
            y_3d = y_data[:, np.newaxis, np.newaxis]  # (N,) -> (N, 1, 1)
        elif orig_ndim == 2:
            y_3d = y_data[:, :, np.newaxis]           # (N, M) -> (N, M, 1)
        elif orig_ndim == 3:
            y_3d = y_data
        else:
            raise ValueError("y_data must be a 1D, 2D, or 3D array.")
        n_bins = len(grid) 
        _, M, P = y_3d.shape
        bin_indices = np.digitize(x_data, grid)# 3. Map x positions to their corresponding bins
        valid_mask = (bin_indices >= 1) & (bin_indices <= n_bins)
        v_indices = bin_indices[valid_mask] - 1  # 0-indexed bin tracking
        v_y = y_3d[valid_mask]                    # Shape: (n_valid, M, P)
        counts = np.bincount(v_indices, minlength=n_bins)# 4. Track number of elements falling into each bin
        counts_expanded = counts[:, np.newaxis, np.newaxis]  # Shape (n_bins, 1, 1)
        sum_y = np.zeros((n_bins, M, P))# 5. Multi-dimensional accumulation via advanced indexing
        sum_y2 = np.zeros((n_bins, M, P))
        idx_tuple = (v_indices, slice(None), slice(None))# Explicitly index axis 0 with v_indices, while taking all elements of axes 1 and 2
        np.add.at(sum_y, idx_tuple, v_y)         
        np.add.at(sum_y2, idx_tuple, v_y**2)     
        bin_means = np.full((n_bins, M, P), np.nan)# 6. Calculate Vectorized Means
        has_data = counts > 0
        bin_means[has_data] = sum_y[has_data] / counts_expanded[has_data]
        bin_stdevs = np.full((n_bins, M, P), np.nan)# 7. Calculate Vectorized Sample Standard Deviations (ddof=1)
        has_enough_data = counts > 1
        variance = (sum_y2[has_enough_data] - (sum_y[has_enough_data]**2 / counts_expanded[has_enough_data])) / (counts_expanded[has_enough_data] - 1)
        bin_stdevs[has_enough_data] = np.sqrt(np.clip(variance, 0, None))
        bin_stdevs[np.isnan(bin_stdevs)]=0
        out = np.stack([bin_means, bin_stdevs], axis=-1)# 8. Stack outputs along the final axis -> Shape: (n_bins, M, P, 2)
        if orig_ndim == 1:# 9. Dynamically reshape back to match input dimensionality expectations
            # (n_bins, 1, 1, 2) -> Squeeze out the extra M and P dimensions -> (n_bins, 2)
            return np.squeeze(out, axis=(1, 2))
        elif orig_ndim == 2:
            return np.squeeze(out, axis=2)# (n_bins, M, 1, 2) -> Squeeze out the extra P dimension -> (n_bins, M, 2)
        else:
            return out
    
    def sample_stats(data):
        """
        Compute mean and sample standard deviation along dimension M (axis=0),
        ignoring np.nan values. 

        :param data: The input array to compute statistics over.
        :type data: numpy.ndarray of shape (M, N, P)
        :returns: An array stacked with mean and sample standard deviation along the last dimension.
        :rtype: numpy.ndarray of shape (N, P, 2)
        """
        mean_data = np.nanmean(data, axis=0)
        sample_stdev = np.nanstd(data, axis=0, ddof=1)
        return np.stack([mean_data, sample_stdev], axis=-1)# Combine outputs into shape (N, P, 2); Axis -1 stacks them along a new trailing dimension
 
    def haversine_distance(lat1, lon1, lat2, lon2, earth_radius=6371.0):
        """
        Calculates the Great-Circle distance between two sets of lat/lon points in km. From https://www.movable-type.co.uk/scripts/latlong.html.
        a = sin²(Δφ/2) + cos φ1 ⋅ cos φ2 ⋅ sin²(Δλ/2)
        c = 2 ⋅ atan2( √a, √(1−a) )
        d = R ⋅ c
        where   φ is latitude, λ is longitude, R is earth’s radius
        (mean radius = 6,371km); note that angles need to be in radians
        to pass to trig functions.
        
        :param lat1: Latitude of the first point in degrees.
        :type lat1: float or numpy.ndarray
        :param lon1: Longitude of the first point in degrees.
        :type lon1: float or numpy.ndarray
        :param lat2: Latitude of the second point in degrees.
        :type lat2: float or numpy.ndarray
        :param lon2: Longitude of the second point in degrees.
        :type lon2: float or numpy.ndarray
        :param earth_radius: Radius of the earth in kilometers. Defaults to 6371.0.
        :type earth_radius: float, optional
        :returns: The Great-Circle distance between the points in kilometers.
        :rtype: float or numpy.ndarray
        """
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return earth_radius * c

    def gen_KDtree(points, time_scale_factor):
        """
        Convert [lat, lon, time] array (lat/lon in degrees) to 3D Cartesian 
        coordinates on a unit sphere, append scaled time, and construct a KDTree.   

        :param points: 2D array where columns correspond to latitude, longitude, and time.
        :type points: numpy.ndarray of shape (N, 3)
        :param time_scale_factor: Scaling factor applied to the time dimension.
        :type time_scale_factor: float
        :returns: A KDTree built from the (x, y, z, scaled_time) coordinates.
        :rtype: scipy.spatial.KDTree
        """
        lat = np.radians(points[:, 0])
        lon = np.radians(points[:, 1])
        tim = points[:, 2]
        x = np.cos(lat) * np.cos(lon)
        y = np.cos(lat) * np.sin(lon)
        z = np.sin(lat)
        t_scale = tim*time_scale_factor
        xyzt = np.column_stack((x, y, z, t_scale))
        tree_out = KDTree(xyzt)
        return tree_out

    earth_radius = 6371.0 #km
    theta = spatial_separation_constraint / earth_radius# Calculate Spatial Search Radius (3D chord length)
    r_spatial  = 2 * np.sin(theta / 2) # Calculate the Time Scaling Factor. We want MAX_TIME_HOURS to mathematically equal r_spatial in the tree
    
    time_scale_factor = r_spatial / (time_separation_constraint / 60)
    # Define the Total 4D Search Radius for the tree. Because Euclidean distance combines both: radius = sqrt(r_spatial^2 + r_time^2)
    # Since we scaled them to be equal at their thresholds, r_time equals r_spatial
    r_4d = np.sqrt(r_spatial**2 + r_spatial**2)

    bds1 = np.vstack((np.hstack((0,1,2,range(50,600,50))),np.hstack((0,1,2,range(10,120,10)))))
    bds2 = np.vstack((np.hstack((0,1,2,range(10,120,10))),np.hstack((0,1,2,range(5,60,5)))))
    bds3 = np.vstack((np.hstack((0,1,2,range(10,120,10))),np.hstack((0,1,2,range(5,60,5)))))
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

    #cmap = 'jet'
    fs =14
    lw = 1.5
    rcParams['font.size'] = fs
    #rcParams['axes.formatter.useoffset'] = False    
    plt.rcParams.update({'font.size': fs})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']   
    plt.rcParams.update({'font.size': fs})
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']   #
    plt.rcParams.update({'mathtext.fontset': 'stix',
     'mathtext.rm': 'Times New Roman',
     'mathtext.it': 'Times New Roman:italic',
     'mathtext.bf': 'Times New Roman:bold'})


    N1 = 10 # number of possible segments to be consistent with the collocation procedure
    rsp_scn_dur = 60/72 # seconds per rsp scan 
    tres = 60 # native hsrl-2 extinction temporal resolution in seconds
    ares = 225 # native hsrl-2 extinction vertical resolution in meters
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
    D_grd = {}
    D_grd["dpg"] = np.logspace(-3,np.log10(20),50)

    IS_Dataset_spheres = Dataset(Insitu_Spherical_Retrieval_filename,'r')
    IS_Dataset_nonspheres = {}
    IS_Dataset_nonspheres[0] = Dataset(Insitu_Nonspherical_Retrieval_filename,'r')
    IS_Dataset_nonspheres[1] = Dataset(Insitu_Nonspherical_Retrieval_filename_2,'r')
    wvl = IS_Dataset_spheres.variables['wavelength'][:]
    wvl_ID = np.array([np.where(wvl==355)[0],np.where(wvl==532)[0],np.where(wvl==1064)[0]])

    source_dict = {}
    source_dict['timestart'] = IS_Dataset_spheres.groups['source'].variables['Time_Start'][:]
    source_dict['timestop'] = IS_Dataset_spheres.groups['source'].variables['Time_Stop'][:]
    source_dict['datetime_Start'] = IS_Dataset_spheres.groups['source'].variables['datetime_Start'][:] 
    source_dict['datetime_Stop'] = IS_Dataset_spheres.groups['source'].variables['datetime_Stop'][:]    
    for key in source_key_vars:
        if source_key_vars[key] in IS_Dataset_spheres.groups['source'].variables:
            source_dict[key] = IS_Dataset_spheres.groups['source'].variables[source_key_vars[key]][:]
        else:
            source_dict[key] = np.full(len(IS_Dataset_spheres.groups['source'].variables['Time_Start'][:]),np.nan)

    ssa_m = source_dict["ssa_m"]
    ssa_m = np.where(ssa_m == '--', np.nan, ssa_m)

    IDX = np.where((source_dict['LWC']>0.001))#|(source_dict['IceFlag']>0)))[0] # flag all data that is not cloud-free (Nd>5 & LWC>0.001)
    IDX3 = np.where((source_dict['InletFlag']>0))[0] # flag all data where the inlet is set to the CVI (inlet flag > 0) 

    derived_dict_spheres = {}         
    for key in IS_Dataset_spheres.groups['derived'].variables.keys():
        vals = IS_Dataset_spheres.groups['derived'].variables[key][:]
        if 'time' in IS_Dataset_spheres.groups['derived'].variables[key].dimensions:
            derived_dict_spheres[key] = vals.filled(np.nan)
            #derived_dict_spheres[key][IDX] = np.nan # remove all ambiguous and cloudy data from Nlas
            #derived_dict_spheres[key][IDX3] = np.nan# remove all CVI data from Nlas 

    derived_dict_nonspheres = {}
    for shptyp in IS_Dataset_nonspheres:
        derived_dict_nonspheres[shptyp] = {}        
        for key in IS_Dataset_nonspheres[shptyp].groups['derived'].variables.keys():
            vals = IS_Dataset_nonspheres[shptyp].groups['derived'].variables[key][:]
            if 'time' in IS_Dataset_nonspheres[shptyp].groups['derived'].variables[key].dimensions:
                derived_dict_nonspheres[shptyp][key] = vals.filled(np.nan)
                #derived_dict_nonspheres[shptyp][key][IDX] = np.nan # remove all ambiguous and cloudy data from Nlas
                #derived_dict_nonspheres[shptyp][key][IDX3] = np.nan# remove all CVI data from Nlas 

    coarseflg = None    
    if 'coarse-diameter' in IS_Dataset_spheres.variables.keys():
        coarseflg = 1
        L_csd = len(IS_Dataset_spheres.variables['coarse-diameter'][:])   
        coarse_diameter = IS_Dataset_spheres.variables['coarse-diameter'][:]   
        crsidx_5um = np.where((coarse_diameter>5))[0]
        coarse_dndlogdp = derived_dict_spheres['coarse_dndlogdp'][:,:]
        N_5um = np.trapezoid(coarse_dndlogdp[:,crsidx_5um],x=np.log10(coarse_diameter[crsidx_5um]),axis=1)
        N_5um = np.where(N_5um == '--', np.nan, N_5um)

    HSRL2_Dictionary = collect_netcdf.grabHSRL2(HSRL2_Filename)# convert hsrl data from their netcdf files to dictionaries using the collect_netcdf procedures 
    HSRL_data = {}
    HSRL_data["time"] = np.array(HSRL2_Dictionary["hsrl2_time_array"])*3600
    L_hsrl2_time = len(HSRL_data["time"])    
    HSRL_data["frmttime"] = np.array(HSRL2_Dictionary["hsrl2_frmttimedata"],dtype="datetime64[s]") 
    hsrl_altgrd = np.array(HSRL2_Dictionary["DataProducts"]["Altitude"])[0]  
    L_hsrl2_alt = len(hsrl_altgrd)
    altgrid = np.arange(0,9000,150)
    HSRL_data["lat"] = np.squeeze(np.array(HSRL2_Dictionary["Nav_Data"]["gps_lat"]))  
    HSRL_data["lon"] = np.squeeze(np.array(HSRL2_Dictionary["Nav_Data"]["gps_lon"])) 
    ares = hsrl_altgrd[1]-hsrl_altgrd[0] #
    
    HSRL_wvl = np.array([355,532,1064]).astype(int)
    n_wvl = len(HSRL_wvl)
    HSRL_data["ext"] = np.full((L_hsrl2_time,L_hsrl2_alt,n_wvl),np.nan)
    HSRL_data["bsc"] = np.full((L_hsrl2_time,L_hsrl2_alt,n_wvl),np.nan)
    HSRL_data["ldr"] = np.full((L_hsrl2_time,L_hsrl2_alt,n_wvl),np.nan)
    HSRL_data["lr"] = np.full((L_hsrl2_time,L_hsrl2_alt,n_wvl-1),np.nan)
    HSRL_data["aod"] = np.full((L_hsrl2_time,n_wvl-1),np.nan)
    for iwvl in range(n_wvl):
        HSRL_data["ext"][:,:,iwvl] = np.array(HSRL2_Dictionary["DataProducts"][f"{HSRL_wvl[iwvl]}_ext"])*pow(10,3) # km-1-> Mm-1
        HSRL_data["bsc"][:,:,iwvl] = np.array(HSRL2_Dictionary["DataProducts"][f"{HSRL_wvl[iwvl]}_bsc_cloud_screened"])*pow(10,3) # km-1.sr-1 -> Mm-1.sr-1
        HSRL_data["ldr"][:,:,iwvl] = np.array(HSRL2_Dictionary["DataProducts"][f"{HSRL_wvl[iwvl]}_aer_dep_cloud_screened"])     
        if HSRL_wvl[iwvl] != 1064:
            HSRL_data["lr"][:,:,iwvl] = np.array(HSRL2_Dictionary["DataProducts"][f"{HSRL_wvl[iwvl]}_Sa"]) # sr   
            HSRL_data["aod"][:,iwvl] = np.squeeze(HSRL2_Dictionary["DataProducts"][f"{HSRL_wvl[iwvl]}_AOT_hi"])
    aotflg = np.where((HSRL_data["aod"][:,0]<=0.08))
    HSRL_data["aod"][aotflg,:] = np.nan
    HSRL_data["ext"][aotflg,:,:] = np.nan
    HSRL_data["bsc"][aotflg,:,:] = np.nan
    HSRL_data["ldr"][aotflg,:,:] = np.nan
    if "Aerosol_ID" in HSRL2_Dictionary["DataProducts"]:
        HSRL_data["aid"] = np.array(HSRL2_Dictionary["DataProducts"]["Aerosol_ID"])
        HSRL_data["aid"][aotflg,:] = np.nan
    else:
        HSRL_data["aid"] = np.full((L_hsrl2_time,L_hsrl2_alt),np.nan)#

    RSP_Dictionary = collect_netcdf.grabRSP(RSP_Filename) # convert rsp data from their netcdf files to dictionaries using the collect_netcdf procedures 
    RSP_data = {}
    RSP_data["time"] = np.array(RSP_Dictionary["rsp_time_array"])*3600 # convert decimal HAM to SAM
    flightindx = np.where(((RSP_data["time"] >= HSRL_data["time"][0])&(RSP_data["time"] <= HSRL_data["time"][-1])))[0]
    RSP_data["time"] = RSP_data["time"][flightindx]
    L_rsptime = len(RSP_data["time"])#
    RSP_data["costfunction"] = np.array(RSP_Dictionary["retrieval_normalized_cost_function_total"][flightindx])
    RSP_data["frmttime"] = np.array(RSP_Dictionary["rsp_frmttimedata"])[flightindx]# assign variables for the formatted time (YYYY:MM:DD hh:mm:ss UTC) and time array (decimal HAM UTC) from RSP
    rsp_dur = rsp_scn_dur*int(RSP_Dictionary["rsp_navg_scn"])/2 # calculate and store RSP scan time using scan duration and number of scans 
    RSP_data["lat"] = np.array(RSP_Dictionary["lat"][flightindx])
    RSP_data["lon"] = np.array(RSP_Dictionary["lon"][flightindx])
    RSP_data["iri_f"] = np.array(RSP_Dictionary["aerosol_imag_fine_556"][flightindx])
    RSP_data["rri_f"] = np.array(RSP_Dictionary["aerosol_real_fine_556"][flightindx])
    RSP_data["iri_f_unc"] = np.array(RSP_Dictionary["aerosol_imag_fine_556_unc"][flightindx])
    RSP_data["rri_f_unc"] = np.array(RSP_Dictionary["aerosol_real_fine_556_unc"][flightindx])
    modes = np.array(["fine","coarse","total"])
    n_modes = len(modes)
    RSP_data["reff"] = np.full((L_rsptime,n_modes-1),np.nan)
    RSP_data["reff_unc"] = np.full((L_rsptime,n_modes-1),np.nan)
    RSP_data["veff"] = np.full((L_rsptime,n_modes-1),np.nan)
    RSP_data["veff_unc"] = np.full((L_rsptime,n_modes-1),np.nan)
    RSP_data["N"] = np.full((L_rsptime,n_modes),np.nan)
    RSP_data["N_unc"] = np.full((L_rsptime,n_modes),np.nan)
    RSP_data["crs"] = np.full((L_rsptime,n_modes-1,n_wvl),np.nan)
    RSP_data["crs_unc"] = np.full((L_rsptime,n_modes-1,n_wvl),np.nan)
    RSP_data["aod"] = np.full((L_rsptime,n_modes,n_wvl),np.nan)
    RSP_data["aod_unc"] = np.full((L_rsptime,n_modes,n_wvl),np.nan)
    RSP_data["ssa"] = np.full((L_rsptime,n_modes,n_wvl),np.nan)
    RSP_data["ssa_unc"] = np.full((L_rsptime,n_modes,n_wvl),np.nan)              
    RSP_data["ext"] = np.full((L_rsptime,n_modes,n_wvl),np.nan)
    RSP_wvls = np.array(RSP_Dictionary[f"aerosol_optical_depth_wavelengths"])
    rspwvl_id_map = {}
    for iwvl in range(n_wvl):
        rspwvl_id_map[iwvl] = np.where((RSP_wvls==HSRL_wvl[iwvl]))
    for im in range(n_modes-1):
        RSP_data["reff"][:,im] = np.array(RSP_Dictionary[f"aerosol_reff_{modes[im]}"][flightindx])
        RSP_data["veff"][:,im] = np.array(RSP_Dictionary[f"aerosol_veff_{modes[im]}"][flightindx])
        RSP_data["reff_unc"][:,im] = np.array(RSP_Dictionary[f"aerosol_reff_{modes[im]}_unc"][flightindx])
        RSP_data["veff_unc"][:,im] = np.array(RSP_Dictionary[f"aerosol_veff_{modes[im]}_unc"][flightindx])
        RSP_data["N"][:,im] = np.array(RSP_Dictionary[f"aerosol_number_concentration_{modes[im]}"][flightindx])
        RSP_data["N_unc"][:,im] = np.array(RSP_Dictionary[f"aerosol_number_concentration_{modes[im]}_unc"][flightindx])
        for iwvl in range(n_wvl):
            RSP_data["aod"][:,im,iwvl] = np.array(RSP_Dictionary[f"aerosol_optical_depth_{modes[im]}"])[rspwvl_id_map[iwvl],flightindx]
            RSP_data["aod_unc"][:,im,iwvl] = np.array(RSP_Dictionary[f"aerosol_optical_depth_{modes[im]}_unc"])[rspwvl_id_map[iwvl],flightindx]
            RSP_data["ssa"][:,im,iwvl] = np.array(RSP_Dictionary[f"aerosol_ssa_{modes[im]}"])[rspwvl_id_map[iwvl],flightindx]
            if im ==0:
                RSP_data["ssa_unc"][:,im,iwvl] = np.array(RSP_Dictionary[f"aerosol_ssa_{modes[im]}_unc"])[rspwvl_id_map[iwvl],flightindx]
            RSP_data["ext"][:,im,iwvl] = np.array(RSP_Dictionary[f"aerosol_extinction_coefficient_{modes[im]}"])[rspwvl_id_map[iwvl],flightindx]
            RSP_data["crs"][:,im,iwvl] = np.array(RSP_Dictionary[f"aerosol_cross_section_{modes[im]}"])[rspwvl_id_map[iwvl],flightindx]*pow(10,8) 
            RSP_data["crs_unc"][:,im,iwvl] = np.array(RSP_Dictionary[f"aerosol_cross_section_{modes[im]}_unc"])[rspwvl_id_map[iwvl],flightindx]*pow(10,8) 

    RSP_data["N"][:,-1] = np.nansum(RSP_data["N"][:,0:-2],axis=1)
    RSP_data["N_unc"][:,-1] = np.sqrt(np.nansum(RSP_data["N_unc"][:,0:-2]**2,axis=1))
    RSP_data["ext"][:,-1,:] = np.nansum(RSP_data["ext"][:,0:-2,:],axis=1)
    RSP_data["ssa"][:,-1,:] = (RSP_data["ext"][:,0,:]*RSP_data["ssa"][:,0,:]+RSP_data["ext"][:,1,:]*RSP_data["ssa"][:,1,:])/RSP_data["ext"][:,-1,:] 
    RSP_data["ssa_unc"][:,-1,:] = np.sqrt(np.nansum(RSP_data["ssa_unc"][:,0:-2]**2,axis=1))


    LegID = LegID_dictionary['LegIndex_flag'].astype(int)  # retrieve the legIDs for this flight
    Leg_start_time = LegID_dictionary['Time_Start_Seconds']
    Leg_stop_time = LegID_dictionary['Time_Stop_Seconds']
    lgix = np.where((Leg_start_time>=source_dict['timestart'][0])&(Leg_stop_time<=source_dict['timestop'][-1]))[0]
    for key in LegID_dictionary:
        if np.logical_not(isinstance(LegID_dictionary[key],str))&np.logical_not(isinstance(LegID_dictionary[key],dict)):
            LegID_dictionary[key]=LegID_dictionary[key][lgix]#
    LegID = LegID_dictionary['LegIndex_flag'].astype(int)
    Leg_start_time = LegID_dictionary['Time_Start_Seconds']
    Leg_stop_time = LegID_dictionary['Time_Stop_Seconds']   
    L_ID = len(LegID)
    legfmtime = np.column_stack((LegID_dictionary['datetime_Start_UTC'],LegID_dictionary['datetime_Stop_UTC'])).astype('datetime64[s]')
    Collocated_data_final = {}
    for i1 in range(L_ID):
        valid_hsrl_matches = None
        valid_rsp_matches = None
        legstart = Leg_start_time[i1]
        legstop = Leg_stop_time[i1]
        legid = LegID[i1].astype(str)

        if ((legid.endswith('07'))|(legid.endswith('08'))|(legid.endswith('09'))):
            indx1 = np.where(((source_dict['timestop'] <=legstop)&(source_dict['timestart']>=legstart)))[0]
            leg_dur = len(indx1)
            Collocated_data = {}
            Collocated_data["IS_Data"] = {}
            Collocated_data["IS_Data"]["rh"] = source_dict['RH'][indx1]
            Collocated_data["IS_Data"]["kappa"] = derived_dict_spheres['kappa-550'][indx1]
            Collocated_data["IS_Data"]["ssa_m"] = ssa_m[indx1]
            Collocated_data["IS_Data"]["alt"] = source_dict['altitude'][indx1]
            Collocated_data["IS_Data"]["lat"] = source_dict['latitude'][indx1]
            Collocated_data["IS_Data"]["lon"] = source_dict['longitude'][indx1]          
            Collocated_data["IS_Data"]["datetime_start"] = source_dict['datetime_Start'][indx1].astype("datetime64[s]")
            Collocated_data["IS_Data"]["datetime_stop"] = source_dict['datetime_Stop'][indx1].astype("datetime64[s]")
            Collocated_data["IS_Data"]["timemid"]=(source_dict["timestop"][indx1]+source_dict["timestart"][indx1])/2
            Collocated_data["IS_Data"]["dndlogdp_fine"] = derived_dict_spheres['dndlogdp'][indx1,:]
            Collocated_data["IS_Data"]["dpgf"] = derived_dict_spheres['amb_geometric_mean_diameter'][indx1,:]
            Collocated_data["IS_Data"]["dpuf"] = derived_dict_spheres['amb_upper_cutoff_diameter'][indx1,:]
            Collocated_data["IS_Data"]["dplf"] = derived_dict_spheres['amb_lower_cutoff_diameter'][indx1,:]
            Collocated_data["IS_Data"]["dpgf_dry"] = derived_dict_spheres['dry_geometric_mean_diameter'][indx1,:]
            Collocated_data["IS_Data"]["dpuf_dry"] = derived_dict_spheres['dry_upper_cutoff_diameter'][indx1,:]
            Collocated_data["IS_Data"]["dplf_dry"] = derived_dict_spheres['dry_lower_cutoff_diameter'][indx1,:]
            if coarseflg is not None: 
                Collocated_data["IS_Data"]["dndlogdp_coarse"] = coarse_dndlogdp[indx1,:]
                Collocated_data["IS_Data"]["N5"] = N_5um[indx1]
            Collocated_data["IS_Data"]["N"] = np.full((leg_dur,n_modes),np.nan)
            Collocated_data["IS_Data"]["reff"] = np.full((leg_dur,n_modes-1),np.nan)
            Collocated_data["IS_Data"]["veff"] = np.full((leg_dur,n_modes-1),np.nan)
            Collocated_data["IS_Data"]["rri"] = np.full((leg_dur,n_modes,n_wvl),np.nan)
            Collocated_data["IS_Data"]["iri"] = np.full((leg_dur,n_modes,n_wvl),np.nan)
            Collocated_data["IS_Data"]["ext"] = np.full((leg_dur,n_modes,n_wvl),np.nan)
            Collocated_data["IS_Data"]["bsc"] = np.full((leg_dur,n_modes,n_wvl),np.nan)
            Collocated_data["IS_Data"]["ldr"] = np.full((leg_dur,n_modes,n_wvl),np.nan)
            Collocated_data["IS_Data"]["ssa"] = np.full((leg_dur,n_modes,n_wvl),np.nan)
            Collocated_data["IS_Data"]["crs"] = np.full((leg_dur,n_modes,n_wvl),np.nan)
            Collocated_data["IS_Data"]["lr"] = np.full((leg_dur,n_modes,n_wvl-1),np.nan)
            for shptyp in derived_dict_nonspheres:
                Collocated_data["IS_Data"][f"ext_nonspheres_{shptyp}"] = np.full((leg_dur,n_modes,n_wvl),np.nan)
                Collocated_data["IS_Data"][f"bsc_nonspheres_{shptyp}"] = np.full((leg_dur,n_modes,n_wvl),np.nan)
                Collocated_data["IS_Data"][f"ssa_nonspheres_{shptyp}"] = np.full((leg_dur,n_modes,n_wvl),np.nan)
                Collocated_data["IS_Data"][f"ldr_nonspheres_{shptyp}"] = np.full((leg_dur,n_modes,n_wvl),np.nan) 
                Collocated_data["IS_Data"][f"crs_nonspheres_{shptyp}"] = np.full((leg_dur,n_modes,n_wvl),np.nan)    
                Collocated_data["IS_Data"][f"lr_nonspheres_{shptyp}"] = np.full((leg_dur,n_modes,n_wvl-1),np.nan)
            Collocated_data["IS_Data"]["N"][:,-1] = derived_dict_spheres[f"optical_amb_N"][indx1]
            for im in range(n_modes):
                if im < n_modes-1:
                    Collocated_data["IS_Data"]["N"][:,im] = derived_dict_spheres[f"{modes[im]}_amb_N"][indx1]
                    Collocated_data["IS_Data"]["reff"][:,im] = derived_dict_spheres[f"{modes[im]}_amb_r_eff"][indx1]
                    Collocated_data["IS_Data"]["veff"][:,im] = derived_dict_spheres[f"{modes[im]}_amb_v_eff"][indx1]
                for iwvl in range(n_wvl):
                    Collocated_data["IS_Data"]["rri"][:,im,iwvl] = derived_dict_spheres[f"{modes[im]}_amb_RRI"][indx1,wvl_ID[iwvl]]
                    Collocated_data["IS_Data"]["iri"][:,im,iwvl] = derived_dict_spheres[f"{modes[im]}_amb_IRI"][indx1,wvl_ID[iwvl]]                    
                    Collocated_data["IS_Data"]["ext"][:,im,iwvl] = derived_dict_spheres[f"{modes[im]}_amb_ext_coef"][indx1,wvl_ID[iwvl]]
                    Collocated_data["IS_Data"]["crs"][:,im,iwvl] = Collocated_data["IS_Data"]["ext"][:,im,iwvl]/Collocated_data["IS_Data"]["N"][:,im]
                    Collocated_data["IS_Data"]["bsc"][:,im,iwvl] = derived_dict_spheres[f"{modes[im]}_amb_back_coef"][indx1,wvl_ID[iwvl]]
                    Collocated_data["IS_Data"]["ldr"][:,im,iwvl] = derived_dict_spheres[f"{modes[im]}_amb_LDR"][indx1,wvl_ID[iwvl]] 
                    Collocated_data["IS_Data"]["ssa"][:,im,iwvl] = derived_dict_spheres[f"{modes[im]}_amb_ssa"][indx1,wvl_ID[iwvl]]
                    if HSRL_wvl[iwvl] != 1064:
                        Collocated_data["IS_Data"]["lr"][:,im,iwvl] = derived_dict_spheres[f"{modes[im]}_amb_lidar_ratio"][indx1,wvl_ID[iwvl]]
                    for shptyp in derived_dict_nonspheres:    
                        Collocated_data["IS_Data"][f"ext_nonspheres_{shptyp}"][:,im,iwvl] = derived_dict_nonspheres[shptyp][f"{modes[im]}_amb_ext_coef"][indx1,wvl_ID[iwvl]]
                        Collocated_data["IS_Data"][f"bsc_nonspheres_{shptyp}"][:,im,iwvl] = derived_dict_nonspheres[shptyp][f"{modes[im]}_amb_back_coef"][indx1,wvl_ID[iwvl]]
                        Collocated_data["IS_Data"][f"ldr_nonspheres_{shptyp}"][:,im,iwvl] = derived_dict_nonspheres[shptyp][f"{modes[im]}_amb_LDR"][indx1,wvl_ID[iwvl]] 
                        Collocated_data["IS_Data"][f"ssa_nonspheres_{shptyp}"][:,im,iwvl] = derived_dict_nonspheres[shptyp][f"{modes[im]}_amb_ssa"][indx1,wvl_ID[iwvl]]
                        if HSRL_wvl[iwvl] != 1064:
                            Collocated_data["IS_Data"][f"lr_nonspheres_{shptyp}"][:,im,iwvl] = derived_dict_nonspheres[shptyp][f"{modes[im]}_amb_lidar_ratio"][indx1,wvl_ID[iwvl]]

            pt_IS = np.array([Collocated_data["IS_Data"]["lat"],Collocated_data["IS_Data"]["lon"],Collocated_data["IS_Data"]["timemid"]/3600]).T   
            kdtree_IS = gen_KDtree(pt_IS,time_scale_factor)            
            pt_RSP = np.array([RSP_data["lat"],RSP_data["lon"],RSP_data["time"]/3600]).T
            kdtree_RSP = gen_KDtree(pt_RSP,time_scale_factor)  
            pt_HSRL = np.array([HSRL_data["lat"],HSRL_data["lon"],HSRL_data["time"]/3600]).T 
            kdtree_HSRL = gen_KDtree(pt_HSRL,time_scale_factor)   

            matches_RSP = kdtree_IS.query_ball_tree(kdtree_RSP,r=r_4d)
            flat_indices = [idx for sublist in matches_RSP for idx in sublist]
            RSP_points = np.unique(flat_indices)
            if len(RSP_points)>0:
                Collocated_data["RSP_Data"] = {}
                valid_rsp_matches = True
                for key in RSP_data:
                    Collocated_data["RSP_Data"][key] = RSP_data[key][RSP_points,...]    

            matches_HSRL = kdtree_IS.query_ball_tree(kdtree_HSRL,r=r_4d)
            flat_indices = [idx for sublist in matches_HSRL for idx in sublist]
            HSRL_points = np.unique(flat_indices)
            if len(HSRL_points)>0:
                Collocated_data["HSRL_Data"] = {}
                valid_hsrl_matches = True
                for key in HSRL_data:
                    Collocated_data["HSRL_Data"][key] = HSRL_data[key][HSRL_points,...] 

            if (valid_rsp_matches is True) | (valid_hsrl_matches is True):
                Collocated_data_final[legid] = Collocated_data
                Collocated_data_final[legid]["legstart"] = legstart
                Collocated_data_final[legid]["legstop"] = legstop
                Collocated_data_final[legid]["legstart_datetime"] = legfmtime[i1,0]
                Collocated_data_final[legid]["legstop_datetime"] = legfmtime[i1,1]

    L_ID2 = len(Collocated_data_final)   
    filename_prefix = f"../ISARA_data_files/{camp_name}/{out_directory_name}/{camp_name}"    
    prctile = [0,50,68,95,100]
    prctile_lst_b = getPercentileList(prctile,"B")
    prctile_lst_ab = getPercentileList(prctile,"AB")        
    prctile_lst_rb = getPercentileList(prctile,"RB")
    prctile_lst_arb = getPercentileList(prctile,"ARB")
    prctile_lst_x = getPercentileList(prctile,"x")
    prctile_lst_y = getPercentileList(prctile,"y")
    stats_dict = np.zeros((N1+1,55)) 

    # vertical profiles 
    #rcParams['font.size'] = 16
    fs2 = 12
    naltstats = np.full((L_ID2,57),np.nan)
    altstats = {}
    altstats['ext'] = {}
    altstats['bsc'] = {}
    altstats['ldr'] = {}
    altstats['lr'] = {}
    for iwvl in range(n_wvl):
        altstats['ext'][HSRL_wvl[iwvl]] = np.full((L_ID2,57),np.nan)
        altstats['bsc'][HSRL_wvl[iwvl]] = np.full((L_ID2,57),np.nan)
        altstats['ldr'][HSRL_wvl[iwvl]] = np.full((L_ID2,57),np.nan)
        if HSRL_wvl[iwvl] != 1064:
            altstats['lr'][HSRL_wvl[iwvl]] = np.full((L_ID2,57),np.nan)
    IS_alt_data = {}        
    IS_alt_data["alt"] = np.full((L_ID2,len(altgrid),2),np.nan) 
    IS_alt_data["lat"] = np.full((L_ID2,len(altgrid),2),np.nan)
    IS_alt_data["lon"] = np.full((L_ID2,len(altgrid),2),np.nan)
    IS_alt_data["N"] = np.full((L_ID2,len(altgrid),n_modes,2),np.nan) 
    IS_alt_data["rh"] = np.full((L_ID2,len(altgrid),2),np.nan) 
    IS_alt_data["kappa"] = np.full((L_ID2,len(altgrid),2),np.nan) 
    IS_alt_data["rri"] = np.full((L_ID2,len(altgrid),n_modes,n_wvl,2),np.nan) 
    IS_alt_data["iri"] = np.full((L_ID2,len(altgrid),n_modes,n_wvl,2),np.nan) 
    IS_alt_data["ssa"] = np.full((L_ID2,len(altgrid),n_modes,n_wvl,2),np.nan) 
    IS_alt_data["reff"] = np.full((L_ID2,len(altgrid),n_modes-1,2),np.nan) 
    IS_alt_data["veff"] = np.full((L_ID2,len(altgrid),n_modes-1,2),np.nan) 
    IS_alt_data["crs"] = np.full((L_ID2,len(altgrid),n_modes,n_wvl,2),np.nan) 
    IS_alt_data["N5"] = np.full((L_ID2,len(altgrid),2),np.nan) 
    IS_alt_data["ctotextratio"] = np.full((L_ID2,len(altgrid),2),np.nan) 
    IS_alt_data["ext"] = np.full((L_ID2,len(altgrid),n_modes,n_wvl,2),np.nan) 
    IS_alt_data["bsc"] = np.full((L_ID2,len(altgrid),n_modes,n_wvl,2),np.nan) 
    IS_alt_data["lr"] = np.full((L_ID2,len(altgrid),n_modes,n_wvl-1,2),np.nan) 
    IS_alt_data["ldr"] = np.full((L_ID2,len(altgrid),n_modes,n_wvl,2),np.nan) 

    HSRL_alt_data = {}        
    HSRL_alt_data["N"] = np.full((L_ID2,len(altgrid),2),np.nan) 
    HSRL_alt_data["ext"] = np.full((L_ID2,len(altgrid),n_wvl,2),np.nan) 
    HSRL_alt_data["bsc"] = np.full((L_ID2,len(altgrid),n_wvl,2),np.nan) 
    HSRL_alt_data["lr"] = np.full((L_ID2,len(altgrid),n_wvl-1,2),np.nan) 
    HSRL_alt_data["ldr"] = np.full((L_ID2,len(altgrid),n_wvl,2),np.nan) 
    HSRL_alt_data["aid"] = np.full((L_ID2,len(altgrid),len(HSRLAerosolType)),np.nan)
    
    IS_col_data = {}
    IS_col_data["frmttime"] = np.full((L_ID2,2),"NaT").astype("datetime64[s]") 
    IS_col_data["N_datacount"] = np.full((L_ID2),np.nan)
    IS_col_data["ext_datacount"] = np.full((L_ID2,3),np.nan)
    IS_col_data["bsc_datacount"] = np.full((L_ID2,3),np.nan)
    IS_col_data["ldr_datacount"] = np.full((L_ID2,3),np.nan)
    IS_col_data["lr_datacount"] = np.full((L_ID2,3),np.nan)    
    IS_col_data["sepatation"] = np.full((L_ID2),np.nan)
    IS_col_data["rri"] = np.full((L_ID2,n_modes,n_wvl,2),np.nan)
    IS_col_data["iri"] = np.full((L_ID2,n_modes,n_wvl,2),np.nan)
    IS_col_data["N"] = np.full((L_ID2,n_modes,2),np.nan)
    IS_col_data["reff"] = np.full((L_ID2,n_modes-1,2),np.nan)
    IS_col_data["veff"] = np.full((L_ID2,n_modes-1,2),np.nan)
    IS_col_data["crs"] = np.full((L_ID2,n_modes,n_wvl,2),np.nan)
    IS_col_data["ssa"] = np.full((L_ID2,n_modes,n_wvl,2),np.nan)
    IS_col_data["ssa_m"] = np.full((L_ID2,2),np.nan)

    HSRL_col_data = {}
    HSRL_col_data["frmttime"] = np.full((L_ID2,2),"NaT").astype("datetime64[s]") 
    HSRL_col_data["lat"] = np.full((L_ID2),np.nan)
    HSRL_col_data["lon"] = np.full((L_ID2),np.nan)
    HSRL_col_data["aod"] = np.full((L_ID2,n_wvl-1,2),np.nan) 
    HSRL_col_data["N"] = np.full((L_ID2,2),np.nan)

    RSP_col_data = {}
    RSP_col_data["frmttime"] = np.full((L_ID2,3),"NaT").astype("datetime64[s]") 
    RSP_col_data["lat"] = np.full((L_ID2,2),np.nan)
    RSP_col_data["lon"] = np.full((L_ID2,2),np.nan)
    RSP_col_data["aod"] = np.full((L_ID2,n_modes,n_wvl,2),np.nan)
    RSP_col_data["rri_f"] = np.full((L_ID2,2),np.nan)
    RSP_col_data["iri_f"] = np.full((L_ID2,2),np.nan)
    RSP_col_data["reff"] = np.full((L_ID2,n_modes-1,2),np.nan)
    RSP_col_data["veff"] = np.full((L_ID2,n_modes-1,2),np.nan)
    RSP_col_data["ssa"] = np.full((L_ID2,n_modes,n_wvl,2),np.nan)
    RSP_col_data["crs"] = np.full((L_ID2,n_modes-1,n_wvl,2),np.nan)
    RSP_col_data["aod_unc"] = np.full((L_ID2,n_modes,n_wvl,2),np.nan)
    RSP_col_data["iri_f_unc"] = np.full((L_ID2,2),np.nan)
    RSP_col_data["rri_f_unc"] = np.full((L_ID2,2),np.nan)
    RSP_col_data["reff_unc"] = np.full((L_ID2,n_modes-1,2),np.nan)
    RSP_col_data["veff_unc"] = np.full((L_ID2,n_modes-1,2),np.nan)
    RSP_col_data["crs_unc"] = np.full((L_ID2,n_modes-1,n_wvl,2),np.nan)
    RSP_col_data["ssa_unc"] = np.full((L_ID2,n_modes,n_wvl,2),np.nan)
    
    ytks = np.arange(0,6,1)
    ytklbls_int = ["%i"%ix for ix in ytks]
    ytklbls_int[0] = ""
    ytklbls_null = ["" for ix in ytks]  
    xtks = {}
    xtks['Na'] = range(0,2250,250)#range(0,4000,500)
    xtks['kext532'] = np.arange(0,0.1,0.01)
    xtks['ldr532'] = np.arange(0,0.24,0.02)
    xtks['RH'] = np.arange(10,90,10)
    xtks['kappa'] = np.arange(0,0.2,0.02)
    xtks['rri'] = np.arange(1.46,1.56,0.02)
    xtks['iri'] = np.arange(0,0.033,0.003)        
    xtks['ssa'] = np.arange(0.7,1.05,0.05)
    xtks['ext355'] = range(0,200,20)
    xtks['ext532'] = range(0,100,10)
    xtks['ext1064'] = range(0,55,5)
    xtks['bsc355'] = np.arange(0,5,0.5)
    xtks['bsc532'] = np.arange(0,3,0.3)
    xtks['bsc1064'] = np.arange(0,2,0.2)
    xtks['lr355'] = range(0,110,10)
    xtks['lr532'] = range(0,110,10)
    xtks['CtoTExtRatio'] = np.arange(0,0.6,0.05)
    wvlclrs_hsrl = np.array(['b','g','r'])
    wvlclrs_is = np.array(['c','y','xkcd:dark red'])
    xtks['reff_f'] = np.arange(0.1,0.18,0.01)
    xtks['veff_f'] = np.arange(0,0.4,0.05)
    xtks['reff_c'] = np.arange(1,4,0.5)
    xtks['veff_c'] = np.arange(0,0.6,0.05)#
    xtks['count'] = np.arange(0,110,10)

    i1 = 0
    for legid in Collocated_data_final:
        rcParams['figure.figsize'] = 9, 9 # W, H
        fig0,ax0=plt.subplots(3, 4)
        ax0[0,0].set_xlabel(r"$N$ (cm$^{-3}$)", fontsize=fs) # set xaxis label 
        ax0[0,0].set_ylabel("Altitude (km)", fontsize=fs) # set yaxis label   
        ax0[0,0].set_ylim(0,6) # cut y-axis off at zero   
        ax0[0,0].set_yticks(ytks, ytklbls_int)
        ax0[0,0].set_xlim(xtks['Na'][0],xtks['Na'][-1])    
        xtklbls = ["%0.0f"%ix for ix in xtks['Na']]
        ax0[0,0].set_xticks(xtks['Na'][0:-1:3], xtklbls[0:-1:3])    
        for axis in ['top','bottom','left','right']:
            ax0[0,0].spines[axis].set_linewidth(2)     
        ax0[0,0].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[0,0].tick_params(axis='x', labelsize=fs, rotation=0)  
        for label in ax0[0,0].get_xticklabels():
            label.set_horizontalalignment('center')
        fig0_ata = AnchoredText(f"(a)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[0,0].set_axisbelow(False)
        ax0[0,1].set_xlabel(r"$\sigma_{\rm ext}$ ($\rm \mu m^2$)", fontsize=fs) # set xaxis label  
        ax0[0,1].set_ylim(0,6) # cut y-axis off at zero   
        ax0[0,1].set_yticks(ytks, ytklbls_null)
        ax0[0,1].set_xlim(xtks['kext532'][0],xtks['kext532'][-1])    
        xtklbls = ["%0.2f"%ix for ix in xtks['kext532']]
        xtklbls[0] = "0"
        ax0[0,1].set_xticks(xtks['kext532'][0:-1:3], xtklbls[0:-1:3])    
        for axis in ['top','bottom','left','right']:
            ax0[0,1].spines[axis].set_linewidth(2)     
        ax0[0,1].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[0,1].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax0[0,1].get_xticklabels():
            label.set_horizontalalignment('center')
        fig0_atb = AnchoredText(f"(b)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[0,1].set_axisbelow(False)    
        ax0[0,2].set_xlabel(r'LDR', fontsize=fs) # set xaxis label   
        ax0[0,2].set_ylim(0,6) # cut y-axis off at zero   
        ax0[0,2].set_yticks(ytks, ytklbls_null)
        ax0[0,2].set_xlim(xtks['ldr532'][0],xtks['ldr532'][-1])     
        xtklbls = ["%0.2f"%ix for ix in xtks['ldr532']]
        xtklbls[0] = "0"
        ax0[0,2].set_xticks(xtks['ldr532'][0:-1:3], xtklbls[0:-1:3])
        for axis in ['top','bottom','left','right']:
            ax0[0,2].spines[axis].set_linewidth(2)     
        ax0[0,2].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[0,2].tick_params(axis='both', labelsize=fs2, rotation=0)  
        for label in ax0[0,2].get_xticklabels():
            label.set_horizontalalignment('center')
        fig0_atc = AnchoredText(f"(c)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[0,2].set_axisbelow(False) 
        ax0[0,3].set_xlabel(r'RH ($\%$)', fontsize=fs) # set xaxis label   
        ax0[0,3].set_ylim(0,6) # cut y-axis off at zero   
        ax0[0,3].set_yticks(ytks, ytklbls_null)
        ax0[0,3].set_xlim(xtks['RH'][0],xtks['RH'][-1])     
        xtklbls = ["%0.0f"%ix for ix in xtks['RH']]
        ax0[0,3].set_xticks(xtks['RH'][0:-1:3], xtklbls[0:-1:3])
        for axis in ['top','bottom','left','right']:
            ax0[0,3].spines[axis].set_linewidth(2)     
        ax0[0,3].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[0,3].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax0[0,3].get_xticklabels():
            label.set_horizontalalignment('center')
        fig0_atd = AnchoredText(f"(d)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[0,3].set_axisbelow(False)
        ax0[1,0].set_xlabel(r'$\kappa$', fontsize=fs) # set xaxis label  
        ax0[1,0].set_ylabel("Altitude (km)", fontsize=fs) # set yaxis label    
        ax0[1,0].set_ylim(0,6) # cut y-axis off at zero   
        ax0[1,0].set_xlim(xtks['kappa'][0],xtks['kappa'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax0[1,0].spines[axis].set_linewidth(2)     
        ax0[1,0].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[1,0].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax0[1,0].get_xticklabels():
            label.set_horizontalalignment('center')
        xtklbls = ["%0.2f"%ix for ix in xtks['kappa']]
        xtklbls[0] = "0"
        ax0[1,0].set_xticks(xtks['kappa'][0:-1:3], xtklbls[0:-1:3])
        ax0[1,0].set_yticks(ytks, ytklbls_int)
        fig0_ate = AnchoredText(f"(e)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[1,0].set_axisbelow(False)    
        ax0[1,1].set_xlabel(r'RRI', fontsize=fs) # set xaxis label   
        ax0[1,1].set_ylim(0,6) # cut y-axis off at zero   
        ax0[1,1].set_xlim(xtks['rri'][0],xtks['rri'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax0[1,1].spines[axis].set_linewidth(2)     
        ax0[1,1].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[1,1].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax0[1,1].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        ytklbls[0] = ""
        xtklbls = ["%0.2f"%ix for ix in xtks['rri']]
        ax0[1,1].set_xticks(xtks['rri'][0:-1:3], xtklbls[0:-1:3])
        ax0[1,1].set_yticks(ytks, ytklbls)
        fig0_atf = AnchoredText(f"(f)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[1,1].set_axisbelow(False)
        ax0[1,2].set_xlabel(r'IRI', fontsize=fs) # set xaxis label   
        ax0[1,2].set_ylim(0,6) # cut y-axis off at zero   
        ax0[1,2].set_xlim(xtks['iri'][0],xtks['iri'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax0[1,2].spines[axis].set_linewidth(2)     
        ax0[1,2].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[1,2].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax0[1,2].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%0.02f"%ix for ix in xtks['iri']]
        xtklbls[0] = "0"
        ax0[1,2].set_xticks(xtks['iri'][0:-1:3], xtklbls[0:-1:3])
        ax0[1,2].set_yticks(ytks, ytklbls)
        fig0_atg = AnchoredText(f"(g)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[1,2].set_axisbelow(False)
        ax0[1,3].set_xlabel(r'SSA', fontsize=fs) # set xaxis label   
        ax0[1,3].set_ylim(0,6) # cut y-axis off at zero   
        ax0[1,3].set_xlim(xtks['ssa'][0],xtks['ssa'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax0[1,3].spines[axis].set_linewidth(2)     
        ax0[1,3].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[1,3].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax0[1,3].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%.1f"%ix for ix in xtks['ssa']]                   
        ax0[1,3].set_xticks(xtks['ssa'][0:-1:3], xtklbls[0:-1:3])
        ax0[1,3].set_yticks(ytks, ytklbls)
        fig0_ath = AnchoredText(f"(h)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[1,3].set_axisbelow(False)
        ax0[2,0].set_ylabel("Altitude (km)", fontsize=fs) 
        ax0[2,0].set_xlabel(r'Fine $r_{\rm eff}$ ', fontsize=fs)
        ax0[2,0].set_ylim(0,6) # cut y-axis off at zero   
        ax0[2,0].set_xlim(xtks['reff_f'][0],xtks['reff_f'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax0[2,0].spines[axis].set_linewidth(2)     
        ax0[2,0].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[2,0].tick_params(axis='x', labelsize=fs, rotation=0)  
        for label in ax0[2,0].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_int
        ytklbls[0] = ""
        xtklbls = ["%0.02f"%ix for ix in xtks['reff_f']]
        ax0[2,0].set_xticks(xtks['reff_f'][0:-1:3], xtklbls[0:-1:3])
        ax0[2,0].set_yticks(ytks, ytklbls)
        fig2_ati = AnchoredText(f"(i)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[2,0].set_axisbelow(False)
        ax0[2,1].set_xlabel(r'Coarse $r_{\rm eff}$ (${\rm \mu m}$)', fontsize=fs) # set xaxis label   
        ax0[2,1].set_ylim(0,6) # cut y-axis off at zero   
        ax0[2,1].set_xlim(xtks['reff_c'][0],xtks['reff_c'][-1])     
        for axis in ['top','bottom','left','right']:
            ax0[2,1].spines[axis].set_linewidth(2)     
        ax0[2,1].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[2,1].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax0[2,1].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%i"%ix for ix in xtks['reff_c']]
        ax0[2,1].set_xticks(xtks['reff_c'][0:-1:2], xtklbls[0:-1:2])
        ax0[2,1].set_yticks(ytks, ytklbls)
        fig2_atj = AnchoredText(f"(j)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[2,1].set_axisbelow(False)  
        ax0[2,2].set_xlabel(r'Fine $v_{\rm eff}$', fontsize=fs) # set xaxis label   
        ax0[2,2].set_ylim(0,6) # cut y-axis off at zero   
        ax0[2,2].set_yticks(ytks, ytklbls)  
        ax0[2,2].set_xlim(xtks['veff_f'][0],xtks['veff_f'][-1])     
        for axis in ['top','bottom','left','right']:
            ax0[2,2].spines[axis].set_linewidth(2)     
        ax0[2,2].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[2,2].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax0[2,2].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%0.02f"%ix for ix in xtks['veff_f']]
        xtklbls[0] = "0"
        ax0[2,2].set_xticks(xtks['veff_f'][0:-1:3], xtklbls[0:-1:3])
        ax0[2,2].set_yticks(ytks, ytklbls)
        fig2_atk = AnchoredText(f"(k)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[2,2].set_axisbelow(False)
        ax0[2,3].set_xlabel(r'Coarse $v_{\rm eff}$', fontsize=fs) # set xaxis label   
        ax0[2,3].set_ylim(0,6) # cut y-axis off at zero   
        ax0[2,3].set_yticks(ytks, ytklbls)  
        ax0[2,3].set_xlim(xtks['veff_c'][0],xtks['veff_c'][-1])     
        for axis in ['top','bottom','left','right']:
            ax0[2,3].spines[axis].set_linewidth(2)     
        ax0[2,3].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax0[2,3].tick_params(axis='both', labelsize=fs2, rotation=0)  
        for label in ax0[2,3].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%0.02f"%ix for ix in xtks['veff_c']]
        xtklbls[0] = "0"
        ax0[2,3].set_xticks(xtks['veff_c'][0:-1:3], xtklbls[0:-1:3])
        ax0[2,3].set_yticks(ytks, ytklbls)
        fig2_atl = AnchoredText(f"(l)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax0[2,3].set_axisbelow(False)          
        plt.subplots_adjust(left=0.075, bottom=0.09, right=0.99, top=0.99)
        plt.tight_layout()# 

        rcParams['figure.figsize'] = 7, 9 # W, H
        fig1,ax1=plt.subplots(3, 3)   
        fig1_at1 = {}  
        fig1_at2 = {}  
        fig1_at3 = {}  
        #ax1[0,0].set_xlabel('355 nm Extinction (Mm$^{-1}$)', fontsize=fs) # set xaxis label 
        ax1[0,0].set_title("355 nm", fontsize=fs)
        ax1[0,0].set_ylabel("Altitude (km)", fontsize=fs) # set yaxis label   
        ax1[0,0].set_xlabel(r'$C_{\rm ext}$ (Mm$^{-1}$)', fontsize=fs) # set xaxis label   
        ax1[0,0].set_ylim(0,6) # cut y-axis off at zero   
        ax1[0,0].set_xlim(xtks['ext355'][0],xtks['ext355'][-1])    
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax1[0,0].spines[axis].set_linewidth(2)     
        ax1[0,0].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax1[0,0].tick_params(axis='both', labelsize=fs2, rotation=0)  
        for label in ax1[0,0].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_int
        ytklbls[0] = ""
        xtklbls = ["%0.0f"%ix for ix in xtks['ext355']]
        ax1[0,0].set_xticks(xtks['ext355'][0:-1:3], xtklbls[0:-1:3])
        ax1[0,0].set_yticks(ytks, ytklbls)
        fig1_at1[0] = AnchoredText(f"(a)", prop=dict(size=fs), frameon=False, loc='upper right')              
        ax1[0,0].set_axisbelow(False)
        ax1[0,1].set_title("532 nm", fontsize=fs)
        ax1[0,1].set_xlabel(r'$C_{\rm ext}$ (Mm$^{-1}$)', fontsize=fs) # set xaxis label   
        ax1[0,1].set_ylim(0,6) # cut y-axis off at zero   
        ax1[0,1].set_xlim(xtks['ext532'][0],xtks['ext532'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax1[0,1].spines[axis].set_linewidth(2)     
        ax1[0,1].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax1[0,1].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax1[0,1].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%0.0f"%ix for ix in xtks['ext532']]
        ax1[0,1].set_xticks(xtks['ext532'][0:-1:3], xtklbls[0:-1:3])
        ax1[0,1].set_yticks(ytks, ytklbls)
        fig1_at1[1] = AnchoredText(f"(b)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax1[0,1].set_axisbelow(False)
        #ax1[0,2].set_xlabel('1064 nm Extinction (Mm$^{-1}$)', fontsize=fs) # set xaxis label   
        ax1[0,2].set_title("1064 nm", fontsize=fs)
        ax1[0,2].set_xlabel(r'$C_{\rm ext}$ (Mm$^{-1}$)', fontsize=fs) # set xaxis label   
        ax1[0,2].set_ylim(0,6) # cut y-axis off at zero   
        ax1[0,2].set_xlim(xtks['ext1064'][0],xtks['ext1064'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax1[0,2].spines[axis].set_linewidth(2)     
        ax1[0,2].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax1[0,2].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax1[0,2].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%0.0f"%ix for ix in xtks['ext1064']]
        ax1[0,2].set_xticks(xtks['ext1064'][0:-1:3], xtklbls[0:-1:3])
        ax1[0,2].set_yticks(ytks, ytklbls)
        fig1_at1[2] = AnchoredText(f"(c)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax1[0,2].set_axisbelow(False)
        ax1[1,0].set_ylabel("Altitude (km)", fontsize=fs) # set yaxis label  
        ax1[1,0].set_xlabel(r'$C_{\rm bsc}$ (Mm$^{-1}$s$r^{-1}$)', fontsize=fs) # set xaxis label    
        ax1[1,0].set_ylim(0,6) # cut y-axis off at zero   
        ax1[1,0].set_xlim(xtks['bsc355'][0],xtks['bsc355'][-1])    
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax1[1,0].spines[axis].set_linewidth(2)     
        ax1[1,0].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax1[1,0].tick_params(axis='both', labelsize=fs2, rotation=0)  
        for label in ax1[1,0].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_int
        ytklbls[0] = ""
        xtklbls = ["%0.1f"%ix for ix in xtks['bsc355']]
        ax1[1,0].set_xticks(xtks['bsc355'][0:-1:3], xtklbls[0:-1:3])
        ax1[1,0].set_yticks(ytks, ytklbls)
        fig1_at2[0] = AnchoredText(f"(d)", prop=dict(size=fs), frameon=False, loc='upper right')              
        ax1[1,0].set_axisbelow(False)
        ax1[1,1].set_xlabel(r'$C_{\rm bsc}$ (Mm$^{-1}$s$r^{-1}$)', fontsize=fs) # set xaxis label   
        ax1[1,1].set_ylim(0,6) # cut y-axis off at zero   
        ax1[1,1].set_xlim(xtks['bsc532'][0],xtks['bsc532'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax1[1,1].spines[axis].set_linewidth(2)     
        ax1[1,1].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax1[1,1].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax1[1,1].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%0.1f"%ix for ix in xtks['bsc532']]
        ax1[1,1].set_xticks(xtks['bsc532'][0:-1:3], xtklbls[0:-1:3])
        ax1[1,1].set_yticks(ytks, ytklbls)
        fig1_at2[1] = AnchoredText(f"(e)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax1[1,1].set_axisbelow(False)
        #ax1[1,2].set_xlabel('1064 nm Extinction (Mm$^{-1}$)', fontsize=fs) # set xaxis label   
        ax1[1,2].set_xlabel(r'$C_{\rm bsc}$ (Mm$^{-1}$s$r^{-1}$)', fontsize=fs) # set xaxis label   
        ax1[1,2].set_ylim(0,6) # cut y-axis off at zero   
        ax1[1,2].set_xlim(xtks['bsc1064'][0],xtks['bsc1064'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax1[1,2].spines[axis].set_linewidth(2)     
        ax1[1,2].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax1[1,2].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax1[1,2].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%0.1f"%ix for ix in xtks['bsc1064']]
        ax1[1,2].set_xticks(xtks['bsc1064'][0:-1:3], xtklbls[0:-1:3])
        ax1[1,2].set_yticks(ytks, ytklbls)
        fig1_at2[2] = AnchoredText(f"(f)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax1[1,2].set_axisbelow(False)
        ax1[2,0].set_ylabel("Altitude (km)", fontsize=fs) # set yaxis label   
        ax1[2,0].set_xlabel(r'LR (sr)', fontsize=fs) # set xaxis label   
        ax1[2,0].set_ylim(0,6) # cut y-axis off at zero   
        ax1[2,0].set_xlim(xtks['lr355'][0],xtks['lr355'][-1])    
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax1[2,0].spines[axis].set_linewidth(2)     
        ax1[2,0].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax1[2,0].tick_params(axis='both', labelsize=fs2, rotation=0)  
        for label in ax1[2,0].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_int
        ytklbls[0] = ""
        xtklbls = ["%0.0f"%ix for ix in xtks['lr355']]
        ax1[2,0].set_xticks(xtks['lr355'][0:-1:3], xtklbls[0:-1:3])
        ax1[2,0].set_yticks(ytks, ytklbls)
        fig1_at3[0] = AnchoredText(f"(g)", prop=dict(size=fs), frameon=False, loc='upper right')              
        ax1[2,0].set_axisbelow(False)
        ax1[2,1].set_xlabel(r'LR (sr)', fontsize=fs) # set xaxis label   
        ax1[2,1].set_ylim(0,6) # cut y-axis off at zero   
        ax1[2,1].set_xlim(xtks['lr532'][0],xtks['lr532'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax1[2,1].spines[axis].set_linewidth(2)     
        ax1[2,1].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax1[2,1].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax1[2,1].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%0.0f"%ix for ix in xtks['lr532']]
        ax1[2,1].set_xticks(xtks['lr532'][0:-1:3], xtklbls[0:-1:3])
        ax1[2,1].set_yticks(ytks, ytklbls)
        fig1_at3[1] = AnchoredText(f"(h)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax1[2,1].set_axisbelow(False)
        ax1[2,2].set_xlabel(r'$\frac{{\rm coarse} \ C_{\rm ext}}{{\rm bulk} \ C_{\rm ext}}$', fontsize=fs) # set xaxis label   
        ax1[2,2].set_ylim(0,6) # cut y-axis off at zero   
        ax1[2,2].set_xlim(xtks['CtoTExtRatio'][0],xtks['CtoTExtRatio'][-1])     
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
            ax1[2,2].spines[axis].set_linewidth(2)     
        ax1[2,2].tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax1[2,2].tick_params(axis='both', labelsize=fs, rotation=0)  
        for label in ax1[2,2].get_xticklabels():
            label.set_horizontalalignment('center')
        ytklbls = ytklbls_null
        xtklbls = ["%0.2f"%ix for ix in xtks['CtoTExtRatio']]
        ax1[2,2].set_xticks(xtks['CtoTExtRatio'][0:-1:3], xtklbls[0:-1:3])
        ax1[2,2].set_yticks(ytks, ytklbls)
        fig1_at4 = AnchoredText(f"(i)", prop=dict(size=fs), frameon=False, loc='upper right')
        ax1[2,2].set_axisbelow(False)   

        plt.tight_layout()
        plt.subplots_adjust(left=0.1, bottom=0.09, right=0.99, top=0.95)
        bounds3 = np.hstack((-1,0,1,10,100,200,400,800,1000,2000,4000,8000,10000,12000))
        lenbnds = len(bounds3)
        boundsLbs3 = bounds3.astype(str)
        boundsLbs3[0] = f""
        boundsLbs3[lenbnds-2] = f">{boundsLbs3[lenbnds-2]}"
        boundsLbs3[lenbnds-1] = ""     
        N3 = len(bounds3)-1
        Jet = plt.get_cmap('jet', N3)
        newcolors = Jet(np.linspace(0, 1, N3))
        wht = np.array([1, 1, 1, 1])
        gry = np.array([0.75, 0.75, 0.75, 1])
        blk = np.array([0, 0, 0, 1])
        newcolors[0, :] = wht
        newcolors[-1, :] = blk
        newcolors = np.vstack((gry,newcolors))
        cmap3 = ListedColormap(newcolors)
        norm3 = mpl.colors.BoundaryNorm(bounds3, cmap3.N)
        rcParams['figure.figsize'] = 7, 4 
        fig3,ax3=plt.subplots(2, 1) # create figure and subplot    
        ax3[0].set_facecolor(gry)
        ax3[0].set_ylabel("Altitude (km)") #,,fontsize=40font='serif',fontname="Times New Roman"
        #ax3[0].set_xlabel(r"Dry D (nm)") 
        ax3[0].set_xlim(D_grd["dpg"][0]*1000,D_grd["dpg"][-1]*1000)
        ax3[0].set_xscale("log")     
        xtklbls = ["" for ix in np.array([0,1,2,3,4])]
        ax3[0].set_xticks(np.array([1,10,100,1000,10000]), xtklbls)
        ax3[0].set_ylim(0,6) # cut y-axis off at zero   
        ax3[0].set_yticks(ytks, ytklbls_int)  
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
           ax3[0].spines[axis].set_linewidth(1.5)     
        ax3[0].tick_params(direction='inout', length=16, width=1.5) # set inside facing ticks, ticklength, and tick line width
        ax3[0].tick_params(axis='both', labelsize=fs, rotation=0)  
        ax3[0].tick_params(axis='both',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
        for label in ax3[0].get_xticklabels():
           label.set_horizontalalignment('center')
        fig3_ata = AnchoredText('(a)', prop=dict(size=fs), frameon=False, loc='upper left')
        ax3[1].set_facecolor(gry)
        ax3[1].set_ylabel("Altitude (km)") #,,fontsize=40font='serif',fontname="Times New Roman"
        ax3[1].set_xlabel(r"$D$ (nm)") 
        ax3[1].set_xscale("log")      
        ax3[1].set_ylim(0,6) # cut y-axis off at zero   
        ax3[1].set_xlim(D_grd["dpg"][0]*1000,D_grd["dpg"][-1]*1000)
        xtklbls = [r"10$^{%i}$"%ix for ix in np.array([0,1,2,3,4])]
        ax3[1].set_xticks(np.array([1,10,100,1000,10000]), xtklbls)
        ax3[1].set_yticks(ytks, ytklbls_int) 
        for axis in ['top','bottom','left','right']:
           ax3[1].spines[axis].set_linewidth(1.5)     
        ax3[1].tick_params(direction='inout', length=16, width=1.5) # set inside facing ticks, ticklength, and tick line width
        ax3[1].tick_params(axis='both', labelsize=fs, rotation=0)  
        ax3[1].tick_params(axis='both', which="minor", direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
        for label in ax3[1].get_xticklabels():
           label.set_horizontalalignment('center')
        fig3_atb = AnchoredText('(b)', prop=dict(size=fs), frameon=False, loc='upper left')
        cax3 = plt.axes([0.765, 0.1, 0.055, 0.85])
        #plt.tight_layout()            
        plt.subplots_adjust(left=0.10, bottom=0.16, right=0.74, top=0.95)#
        bounds4 = np.array([-1,0,0.1,1,5,10,50,100,150,200,400,800,1000])
        lenbnds = len(bounds4)
        boundsLbs4 = ["-1","0","0.1","1","5","10","50","100","150","200","400","800","1000"]
        boundsLbs4[0] = f""
        boundsLbs4[lenbnds-2] = f">{boundsLbs4[lenbnds-2]}"
        boundsLbs4[lenbnds-1] = ""     
        N4 = len(bounds4)-1
        Jet = plt.get_cmap('jet', N4)
        newcolors = Jet(np.linspace(0, 1, N4))
        newcolors[0, :] = wht
        newcolors[-1, :] = blk
        newcolors = np.vstack((gry,newcolors))
        cmap4 = ListedColormap(newcolors)
        norm4 = mpl.colors.BoundaryNorm(bounds4, cmap4.N)
        rcParams['figure.figsize'] = 7, 4
        fig4,ax4=plt.subplots(2, 1) # create figure and subplot    
        ax4[0].set_facecolor(gry)
        ax4[0].set_ylabel("Altitude (km)") #,,fontsize=40font='serif',fontname="Times New Roman"
        #ax4[0].set_xlabel(r"Dry D (nm)") 
        ax4[0].set_xlim(D_grd["dpg"][0]*1000,D_grd["dpg"][-1]*1000)
        ax4[0].set_xscale("log")     
        xtklbls = ["" for ix in np.array([0,1,2,3,4])]
        ax4[0].set_xticks(np.array([1,10,100,1000,10000]), xtklbls)
        ax4[0].set_ylim(0,6) # cut y-axis off at zero   
        ax4[0].set_yticks(ytks, ytklbls_int)  
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
           ax4[0].spines[axis].set_linewidth(1.5)     
        ax4[0].tick_params(direction='inout', length=16, width=1.5) # set inside facing ticks, ticklength, and tick line width
        ax4[0].tick_params(axis='both', labelsize=fs, rotation=0)  
        ax4[0].tick_params(axis='both',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
        for label in ax4[0].get_xticklabels():
           label.set_horizontalalignment('center')
        fig4_ata = AnchoredText('(a)', prop=dict(size=fs), frameon=False, loc='upper left')
        ax4[1].set_facecolor(gry)
        ax4[1].set_ylabel("Altitude (km)") #,,fontsize=40font='serif',fontname="Times New Roman"
        ax4[1].set_xlabel(r"$D$ (nm)") 
        ax4[1].set_xscale("log")      
        ax4[1].set_ylim(0,6) # cut y-axis off at zero   
        ax4[1].set_xlim(D_grd["dpg"][0]*1000,D_grd["dpg"][-1]*1000)
        xtklbls = [r"10$^{%i}$"%ix for ix in np.array([0,1,2,3,4])]
        ax4[1].set_xticks(np.array([1,10,100,1000,10000]), xtklbls)
        ax4[1].set_yticks(ytks, ytklbls_int) 
        for axis in ['top','bottom','left','right']:
           ax4[1].spines[axis].set_linewidth(1.5)     
        ax4[1].tick_params(direction='inout', length=16, width=1.5) # set inside facing ticks, ticklength, and tick line width
        ax4[1].tick_params(axis='both', labelsize=fs, rotation=0)  
        ax4[1].tick_params(axis='both', which="minor", direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
        for label in ax4[1].get_xticklabels():
           label.set_horizontalalignment('center')
        fig4_atb = AnchoredText('(b)', prop=dict(size=fs), frameon=False, loc='upper left')
        cax4 = plt.axes([0.765, 0.1, 0.055, 0.85])
        #plt.tight_layout()            
        plt.subplots_adjust(left=0.10, bottom=0.16, right=0.74, top=0.95)#
        bounds5 = np.array([-1,0,0.0001,0.001,0.01,0.1,1,5,10,50,100])
        lenbnds = len(bounds5)
        boundsLbs5 = ["-1","0",r"$10^{-4}$",r"$10^{-3}$",r"$10^{-2}$",r"$10^{-1}$","1","5","10","50","100"]
        boundsLbs5[0] = f""
        boundsLbs5[lenbnds-2] = f">{boundsLbs5[lenbnds-2]}"
        boundsLbs5[lenbnds-1] = ""     
        N5 = len(bounds5)-1
        Jet = plt.get_cmap('jet', N5)
        newcolors = Jet(np.linspace(0, 1, N5))
        newcolors[0, :] = wht
        newcolors[-1, :] = blk
        newcolors = np.vstack((gry,newcolors))
        cmap5 = ListedColormap(newcolors)
        norm5 = mpl.colors.BoundaryNorm(bounds5, cmap5.N)
        rcParams['figure.figsize'] = 7, 4 
        fig5,ax5=plt.subplots(2, 1) # create figure and subplot    
        ax5[0].set_facecolor(gry)
        ax5[0].set_ylabel("Altitude (km)") #,,fontsize=40font='serif',fontname="Times New Roman"
        #ax5[0].set_xlabel(r"Dry D (nm)") 
        ax5[0].set_xlim(D_grd["dpg"][0]*1000,D_grd["dpg"][-1]*1000)
        ax5[0].set_xscale("log")     
        xtklbls = ["" for ix in np.array([0,1,2,3,4])]
        ax5[0].set_xticks(np.array([1,10,100,1000,10000]), xtklbls)
        ax5[0].set_ylim(0,6) # cut y-axis off at zero   
        ax5[0].set_yticks(ytks, ytklbls_int)  
        for axis in ['top','bottom','left','right']:    # set the line widths of the axes
           ax5[0].spines[axis].set_linewidth(1.5)     
        ax5[0].tick_params(direction='inout', length=16, width=1.5) # set inside facing ticks, ticklength, and tick line width
        ax5[0].tick_params(axis='both', labelsize=fs, rotation=0)  
        ax5[0].tick_params(axis='both',which="minor",direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
        for label in ax5[0].get_xticklabels():
           label.set_horizontalalignment('center')
        fig5_ata = AnchoredText('(a)', prop=dict(size=fs), frameon=False, loc='upper left')
        ax5[1].set_facecolor(gry)
        ax5[1].set_ylabel("Altitude (km)") #,,fontsize=40font='serif',fontname="Times New Roman"
        ax5[1].set_xlabel(r"$D$ (nm)") 
        ax5[1].set_xscale("log")      
        ax5[1].set_ylim(0,6) # cut y-axis off at zero   
        ax5[1].set_xlim(D_grd["dpg"][0]*1000,D_grd["dpg"][-1]*1000)
        xtklbls = [r"10$^{%i}$"%ix for ix in np.array([0,1,2,3,4])]
        ax5[1].set_xticks(np.array([1,10,100,1000,10000]), xtklbls)
        ax5[1].set_yticks(ytks, ytklbls_int) 
        for axis in ['top','bottom','left','right']:
           ax5[1].spines[axis].set_linewidth(1.5)     
        ax5[1].tick_params(direction='inout', length=16, width=1.5) # set inside facing ticks, ticklength, and tick line width
        ax5[1].tick_params(axis='both', labelsize=fs, rotation=0)  
        ax5[1].tick_params(axis='both', which="minor", direction='inout', length=8, width=1.5) # set inside facing ticks, ticklength, and tick line width
        for label in ax5[1].get_xticklabels():
           label.set_horizontalalignment('center')
        fig5_atb = AnchoredText('(b)', prop=dict(size=fs), frameon=False, loc='upper left')
        cax5 = plt.axes([0.765, 0.1, 0.055, 0.85])
        #plt.tight_layout()            
        plt.subplots_adjust(left=0.10, bottom=0.16, right=0.74, top=0.95)
        rcParams['figure.figsize'] = 4.5, 3.25 # W, H
        fig10,ax10=plt.subplots()
        ax10.set_xlabel(r"count", fontsize=fs) # set xaxis label 
        ax10.set_ylabel("Altitude (km)", fontsize=fs) # set yaxis label   
        ax10.set_ylim(0,6) # cut y-axis off at zero   
        ax10.set_yticks(ytks, ytklbls_int)
        ax10.set_xlim(xtks['count'][0],xtks['count'][-1])    
        xtklbls = ["%0.0f"%ix for ix in xtks['count']]
        ax10.set_xticks(xtks['count'][0:-1:3], xtklbls[0:-1:3])    
        for axis in ['top','bottom','left','right']:
            ax10.spines[axis].set_linewidth(2)     
        ax10.tick_params(direction='in', length=8, width=2) # set inside facing ticks, ticklength, and tick line width
        ax10.tick_params(axis='both', labelsize=fs2, rotation=0)  
        for label in ax10.get_xticklabels():
            label.set_horizontalalignment('center')
        ax10.set_axisbelow(False)
        #plt.tight_layout()    
        cax10 = plt.axes([0.5, 0.1, 0.055, 0.85])       
        plt.subplots_adjust(left=0.11, bottom=0.16, right=0.475, top=0.95)  

        aid_categories = ["Ice","Dusty Mix","Maritime","Urban","Smoke","Fresh Smoke","Polluted Maritime","Pure Dust","Untyped"]
        aid_colors = ["xkcd:royal blue","xkcd:fuchsia","xkcd:deep sky blue","xkcd:green","xkcd:red","xkcd:orange","xkcd:sienna","xkcd:dark violet","xkcd:black"]
        aid_shapes = ["d","o","^","s","d","o","^","s","h"] 
        aid_lines = ["-","-","-","-","--","--","--","--","-"]  
        category_to_color = dict(zip(aid_categories, aid_colors))# 1. Define Categories and Colors
        aid_cmap = mcolors.ListedColormap(aid_colors)# 2. Create a Colormap and Normalization
        aid_bounds = np.arange(len(aid_categories) + 1) - 0.5 # Boundaries for each category
        aid_norm = mcolors.BoundaryNorm(aid_bounds, aid_cmap.N)
        sm = plt.cm.ScalarMappable(cmap=aid_cmap, norm=aid_norm)# 3. Create a ScalarMappable
        sm.set_array([]) # Important for ScalarMappable without data

        legstart = Collocated_data_final[legid]["legstart"]
        legstop = Collocated_data_final[legid]["legstop"]
        IS_col_data["frmttime"][i1,:] = np.column_stack((Collocated_data_final[legid]["legstop_datetime"],Collocated_data_final[legid]["legstop_datetime"]))
        Alt_Binned_IS_data = {}
        is_alt = Collocated_data_final[legid]["IS_Data"]["alt"]
        leg_dur = len(is_alt)
        ext_weights = Collocated_data_final[legid]["IS_Data"]["ext"][:,-1,1]
        for key in Collocated_data_final[legid]["IS_Data"]:
            if (np.logical_not(key.__contains__("datetime"))):
                ydata = Collocated_data_final[legid]["IS_Data"][key]
                Alt_Binned_IS_data[key] = bin_data(is_alt,ydata,altgrid)
                if key in IS_col_data:
                    IS_col_data[key][i1,...] = compute_weighted_stats(ydata, ext_weights)
        Y_alt = {}
        Y_alt["ldr"] = rebin_3d_by_altitude(hsrl_altgrd,Collocated_data_final[legid]["HSRL_Data"]["ldr"],altgrid)
        Y_alt["lr"] = rebin_3d_by_altitude(hsrl_altgrd,Collocated_data_final[legid]["HSRL_Data"]["lr"],altgrid)
        Y_alt["ext"] = rebin_3d_by_altitude(hsrl_altgrd,Collocated_data_final[legid]["HSRL_Data"]["ext"],altgrid)
        Y_alt["bsc"] = rebin_3d_by_altitude(hsrl_altgrd,Collocated_data_final[legid]["HSRL_Data"]["bsc"],altgrid)
        HSRL_col_data["lat"][i1] = np.nanmean(Collocated_data_final[legid]["HSRL_Data"]["lat"])
        HSRL_col_data["lon"][i1] = np.nanmean(Collocated_data_final[legid]["HSRL_Data"]["lon"])
        is_lat = np.nanmean(Collocated_data_final[legid]["IS_Data"]["lat"])
        is_lon = np.nanmean(Collocated_data_final[legid]["IS_Data"]["lon"])
        IS_col_data["sepatation"][i1] = haversine_distance(HSRL_col_data["lat"][i1],HSRL_col_data["lon"][i1],is_lat,is_lon)*1000
        HSRL_col_data["aod"][i1,...] = sample_stats(Collocated_data_final[legid]["HSRL_Data"]["aod"])
        hrsl_aid = Collocated_data_final[legid]["HSRL_Data"]["aid"].astype(int)
        aid_counts = (hrsl_aid[:, :, None] == np.arange(len(HSRLAerosolType))).sum(axis=0)
        aid_counts_rebin = rebin_sum_2d_by_altitude(hsrl_altgrd,aid_counts,altgrid)
        HSRL_alt_data["aid"][i1,...] = aid_counts_rebin
        sphrs = np.where((Y_alt["ldr"][:,1,0]<=0.08))[0]
        nonsphrs = {}
        nonsphrs[0] = np.where((Y_alt["ldr"][:,1,0]>0.08)&(Y_alt["lr"][:,1,0]<35))[0]#(YLDR[532]>0.08)&(Y3[532]<35)
        nonsphrs[1] = np.where((Y_alt["ldr"][:,1,0]>0.08)&(Y_alt["lr"][:,1,0]>35))[0]#(YLDR[532]>0.08)&(Y3[532]<35)    
        X_alt = {}
        for key in Y_alt:
            X_alt[key] = Alt_Binned_IS_data[key]
            for shptyp in derived_dict_nonspheres:
                X_alt[key][nonsphrs[shptyp],...] = Alt_Binned_IS_data[f"{key}_nonspheres_{shptyp}"][nonsphrs[shptyp],...]
            IS_alt_data[key][i1,...] = X_alt[key]  
            HSRL_alt_data[key][i1,...] = Y_alt[key]                
        if Y_alt is not None:
            Alt_Binned_IS_data["ctotextratio"] = np.full((len(altgrid),2),np.nan)
            Alt_Binned_IS_data["ctotextratio"][:,0] = X_alt["ext"][:,0,1,0]/X_alt["ext"][:,0,-1,0]
            Alt_Binned_IS_data["ctotextratio"][:,1] = Alt_Binned_IS_data["ctotextratio"][:,0]*np.sqrt((X_alt["ext"][:,0,1,1]/X_alt["ext"][:,0,1,0])**2+(X_alt["ext"][:,0,-1,1]/X_alt["ext"][:,0,-1,0])**2)
            for key in Alt_Binned_IS_data:
                if np.logical_not(key.__contains__("_nonspheres_"))&(key not in Y_alt):
                    if key in IS_alt_data:
                        IS_alt_data[key][i1,...] = Alt_Binned_IS_data[key]
            IS_alt_data["alt"][i1,...] = Alt_Binned_IS_data["alt"]            
            if "RSP_Data" in Collocated_data_final[legid]:
                costfunc_flag = np.where((Collocated_data_final[legid]["RSP_Data"]["costfunction"]<0.15))[0]
                if len(costfunc_flag)>0:
                    rsp_dat = {}
                    for key in Collocated_data_final[legid]["RSP_Data"]:
                        if key in RSP_col_data:
                            dat = Collocated_data_final[legid]["RSP_Data"][key][costfunc_flag,...]
                            if (key.__contains__("frmttime")):
                                timestamps = [dt.timestamp() for dt in dat]
                                mean_timestamp = sum(timestamps) / len(timestamps)
                                mean_date = datetime.datetime.fromtimestamp(mean_timestamp)
                                RSP_col_data[key][i1,:] = np.stack((np.nanmin(dat),mean_date,np.nanmax(dat)))
                                rsp_dat[key] = RSP_col_data[key] 
                            else:
                                RSP_col_data[key][i1,:] = sample_stats(dat)
                                rsp_dat[key] = RSP_col_data[key][i1,...,0]
                else:
                    rsp_dat = None
            else:
                rsp_dat = None
            fine_sd_adj_time = np.full((leg_dur,len(D_grd["dpg"]),3),np.nan)
            coarse_sd_time = np.full((leg_dur,len(D_grd["dpg"]),3),np.nan)
            fine_sd_time = np.full((leg_dur,len(D_grd["dpg"]),3),np.nan)
            kappa_fine = Collocated_data_final[legid]["IS_Data"]["kappa"]
            rri_fine = Collocated_data_final[legid]["IS_Data"]["rri"][:,0,1]
            RH = Collocated_data_final[legid]["IS_Data"]["rh"]
            gf = np.power((1+kappa_fine*RH/(100-RH)),1/3)
            gf = np.where(gf == '--', np.nan, gf)
            gf[(rri_fine>0&(RH<40)&np.isnan(kappa_fine))]=1
            sd_f = Collocated_data_final[legid]["IS_Data"]["dndlogdp_fine"]
            dpg_f = Collocated_data_final[legid]["IS_Data"]["dpgf"]
            for i_time in range(leg_dur):
                nonzerovals = np.where(((dpg_f[i_time,:])>0)& (sd_f[i_time,:]>0))
                if np.nansum(nonzerovals)>0:
                    dpg_f_i = np.squeeze(dpg_f[i_time,nonzerovals])
                    sd_f_i = np.squeeze(sd_f[i_time,nonzerovals])
                    target_sd_n = pchip_interpolate(np.log10(dpg_f_i), sd_f_i, np.log10(D_grd["dpg"]))
                    target_sd_n = np.maximum(target_sd_n, 0.0)# Ensure no non-physical negative concentrations near the tails
                    out_of_bounds = np.where((D_grd["dpg"] < np.nanmin(dpg_f_i)) | (D_grd["dpg"] > np.nanmax(dpg_f_i)))[0] # search for values outside of measured size ranges.
                    target_sd_n[out_of_bounds] = 0 # set concentrations corresponding of non-measured sizes to zero
                    fine_sd_time[i_time,:,0] = target_sd_n
                    fine_sd_time[i_time,:,1] = np.pi*(D_grd["dpg"]**2)*target_sd_n
                    fine_sd_time[i_time,:,2] = np.pi*(D_grd["dpg"]**3)*target_sd_n/6                
                    dpg_amb = gf[i_time]*dpg_f_i
                    if gf[i_time]>0:
                        target_sd_n = pchip_interpolate(np.log10(dpg_amb), sd_f_i, np.log10(D_grd["dpg"]))# Interpolate dN/dlogDp curve in log10(Dp) space; PCHIP prevents non-physical negative concentrations or artificial oscillations
                        target_sd_n = np.maximum(target_sd_n, 0.0)# Ensure no non-physical negative concentrations near the tails
                        out_of_bounds = np.where((D_grd["dpg"] < np.nanmin(dpg_amb)) | (D_grd["dpg"] > np.nanmax(dpg_amb)))[0] # search for values outside of measured size ranges.
                        target_sd_n[out_of_bounds] = 0 # set concentrations corresponding of non-measured sizes to zero
                        fine_sd_adj_time[i_time,:,0] = target_sd_n
                        fine_sd_adj_time[i_time,:,1] = (np.pi)*((D_grd["dpg"])**2)*target_sd_n
                        fine_sd_adj_time[i_time,:,2] = (np.pi)*((D_grd["dpg"])**3)*target_sd_n/6  
                      
                if coarseflg is not None:
                    sdnc_i = Collocated_data_final[legid]["IS_Data"]["dndlogdp_coarse"][i_time,:]
                    nonzerovals = np.where(((sdnc_i)>0))[0]
                    if len(sdnc_i[nonzerovals])>2:
                        target_sd_n = pchip_interpolate(np.log10(coarse_diameter[nonzerovals]), sdnc_i[nonzerovals].T, np.log10(D_grd["dpg"]))# Interpolate dN/dlogDp curve in log10(Dp) space; PCHIP prevents non-physical negative concentrations or artificial oscillations
                        target_sd_n = np.maximum(target_sd_n, 0.0)# Ensure no non-physical negative concentrations near the tails
                        out_of_bounds = np.where((D_grd["dpg"] < np.nanmin(coarse_diameter[nonzerovals])) | (D_grd["dpg"] > np.nanmax(coarse_diameter[nonzerovals])))[0] # search for values outside of measured size ranges.
                        target_sd_n[out_of_bounds] = 0 # set concentrations corresponding of non-measured sizes to zero
                        coarse_sd_time[:,:,0] = target_sd_n
                        coarse_sd_time[:,:,1] = (coarse_sd_time[:,:,0])*np.pi*(D_grd["dpg"]**2)
                        coarse_sd_time[:,:,2] = (coarse_sd_time[:,:,0])*np.pi*(D_grd["dpg"]**3)/6
            if coarseflg is not None:            
                sd_c = bin_data(is_alt,coarse_sd_time,altgrid) 
            
            fine_sd_adj = bin_data(is_alt,fine_sd_adj_time,altgrid) 
            fine_sd = bin_data(is_alt,fine_sd_time,altgrid) 
            X_alt["N"] = bin_data(is_alt,Collocated_data_final[legid]["IS_Data"]["N"],altgrid)
            IS_alt_data["N"][i1,...] = X_alt["N"] 
            if rsp_dat is not None:
                Nopt = np.squeeze(Y_alt["ext"][:,1,0])/rsp_dat["crs"][0,1]
                Nopt_sigma = Y_alt["ext"][:,1,1]/rsp_dat["crs"][0,1] 
                Y_alt["N"] = np.column_stack((Nopt,Nopt_sigma))
                HSRL_col_data["N"][i1,:] = compute_weighted_stats(Nopt,Y_alt["ext"][:,1,1])
            else:
                Y_alt["N"] = np.full((len(altgrid),2),np.nan)
                HSRL_col_data["N"][i1,:] = np.full((2),np.nan)
            HSRL_alt_data["N"][i1,...] = Y_alt["N"]    

            fine_sd_2d = np.nanmean(fine_sd,axis=0)         
            fine_sd_adj_2d = np.nanmean(fine_sd_adj,axis=0)                 
            flx = np.where((X_alt["N"][:,-1,0]>0)&(Y_alt["N"][:,0]>0))[0]
            x0 = X_alt["N"][flx,-1,0]
            y0 = Y_alt["N"][flx,0]
            x1 = {}
            y1 = {}
            x2 = {}
            y2 = {} 
            x3 = {}
            y3 = {}        
            xLDR = {}
            yLDR = {}   
            datacount = len(np.where(np.logical_not(np.isnan(x0))&(np.logical_not(np.isnan(y0))))[0])
            IS_col_data["N_datacount"][i1] = datacount
            if np.nansum(datacount)>0:
                if len(x0)==1:  
                    naltstats[i1,:] = np.zeros(57)  
                    naltstats[i1,0] = legid 
                    naltstats[i1,1] = np.nan 
                    naltstats[i1,41] = x0   
                    naltstats[i1,48] = y0   
                    naltstats[i1,-1] = 1    
                else:   
                    naltstats[i1,:] = np.hstack((legid,np.nan,np.squeeze(StatsCode.Comparison(x0,y0,prctile))))   
            aotinsitu = {}        
            for iwvl in range(n_wvl):
                flx1 = np.where(np.logical_not(np.isnan(X_alt["ext"][:,-1,iwvl,0]))&(np.logical_not(np.isnan(Y_alt["ext"][:,iwvl,0]))))[0]
                x1[iwvl] = X_alt["ext"][flx1,-1,iwvl,0]
                y1[iwvl] = Y_alt["ext"][flx1,iwvl,0]
                flx2 = np.where(np.logical_not(np.isnan(X_alt["bsc"][:,-1,iwvl,0]))&(np.logical_not(np.isnan(Y_alt["bsc"][:,iwvl,0]))))[0]
                x2[iwvl] = X_alt["bsc"][flx2,-1,iwvl,0]
                y2[iwvl] = Y_alt["bsc"][flx2,iwvl,0] 
                flxLDR = np.where(np.logical_not(np.isnan(X_alt["ldr"][:,-1,iwvl,0]))&(np.logical_not(np.isnan(Y_alt["ldr"][:,iwvl,0]))))[0]
                xLDR[iwvl] = X_alt["ldr"][flxLDR,-1,iwvl,0]
                yLDR[iwvl] = Y_alt["ldr"][flxLDR,iwvl,0]
                if HSRL_wvl[iwvl] != 1064:
                    flx3 = np.where(np.logical_not(np.isnan(X_alt["lr"][:,-1,iwvl,0]))&(np.logical_not(np.isnan(Y_alt["lr"][:,iwvl,0]))))[0]
                    x3[iwvl] = X_alt["lr"][flx3,-1,iwvl,0]
                    y3[iwvl] = Y_alt["lr"][flx3,iwvl,0]

                IS_col_data["ext_datacount"][i1,iwvl] = len(flx1)
                IS_col_data["bsc_datacount"][i1,iwvl] = len(flx2)
                IS_col_data["ldr_datacount"][i1,iwvl] = len(flxLDR)
                IS_col_data["lr_datacount"][i1,iwvl] = len(flx3)
                if len(flx1)>0:
                    if len(flx1)>2:
                        aotinsitu[iwvl] = np.trapezoid(x1[iwvl]*10**(-6), x=altgrid[flx1], axis=-1)   
                    else:
                        aotinsitu[iwvl]= np.nan
                    if len(flx1)==1:
                        altstats['ext'][HSRL_wvl[iwvl]][i1,:] = np.zeros(57)
                        altstats['ext'][HSRL_wvl[iwvl]][i1,0] = legid
                        altstats['ext'][HSRL_wvl[iwvl]][i1,1] = aotinsitu[iwvl]
                        altstats['ext'][HSRL_wvl[iwvl]][i1,41] = x1[iwvl]
                        altstats['ext'][HSRL_wvl[iwvl]][i1,48] = y1[iwvl]
                        altstats['ext'][HSRL_wvl[iwvl]][i1,-1] = 1
                    else:
                        altstats['ext'][HSRL_wvl[iwvl]][i1,:] = np.hstack((legid,aotinsitu[iwvl],np.squeeze(StatsCode.Comparison(x1[iwvl],y1[iwvl],prctile)))) 
                if len(flx2)>0:       
                    if len(x2[iwvl])== 1:
                        altstats['bsc'][HSRL_wvl[iwvl]][i1,:] = np.zeros(57)
                        altstats['bsc'][HSRL_wvl[iwvl]][i1,0] = legid
                        altstats['bsc'][HSRL_wvl[iwvl]][i1,1] = np.nan
                        altstats['bsc'][HSRL_wvl[iwvl]][i1,41] = x2[iwvl]
                        altstats['bsc'][HSRL_wvl[iwvl]][i1,48] = y2[iwvl]
                        altstats['bsc'][HSRL_wvl[iwvl]][i1,-1] = 1
                    else:
                        altstats['bsc'][HSRL_wvl[iwvl]][i1,:] = np.hstack((legid,np.nan,np.squeeze(StatsCode.Comparison(x2[iwvl],y2[iwvl],prctile)))) 
                if len(flxLDR)>0:
                    if len(xLDR[iwvl])==1: 
                        altstats['ldr'][HSRL_wvl[iwvl]][i1,:] = np.zeros(57)    
                        altstats['ldr'][HSRL_wvl[iwvl]][i1,0] = legid
                        altstats['ldr'][HSRL_wvl[iwvl]][i1,1] = np.nan
                        altstats['ldr'][HSRL_wvl[iwvl]][i1,41] = xLDR[iwvl]
                        altstats['ldr'][HSRL_wvl[iwvl]][i1,48] = yLDR[iwvl]
                        altstats['ldr'][HSRL_wvl[iwvl]][i1,-1] = 1   
                    else:
                        altstats['ldr'][HSRL_wvl[iwvl]][i1,:] = np.hstack((legid,np.nan,np.squeeze(StatsCode.Comparison(xLDR[iwvl],yLDR[iwvl],prctile)))) 
                if len(flx3)>0:       
                    if HSRL_wvl[iwvl] != 1064:
                        if len(x3[iwvl])== 1:
                            altstats['lr'][HSRL_wvl[iwvl]][i1,:] = np.zeros(57)
                            altstats['lr'][HSRL_wvl[iwvl]][i1,0] = legid
                            altstats['lr'][HSRL_wvl[iwvl]][i1,1] = np.nan
                            altstats['lr'][HSRL_wvl[iwvl]][i1,41] = x3[iwvl]
                            altstats['lr'][HSRL_wvl[iwvl]][i1,48] = y3[iwvl]
                            altstats['lr'][HSRL_wvl[iwvl]][i1,-1] = 1       
                        else:
                            altstats['lr'][HSRL_wvl[iwvl]][i1,:] = np.hstack((legid,np.nan,np.squeeze(StatsCode.Comparison(x3[iwvl],y3[iwvl],prctile))))                                                                                    

            if rsp_dat is not None:       
                ax0[0,0].errorbar(Y_alt["N"][:,0], altgrid/1000, xerr=Y_alt["N"][:,1], linestyle='none', elinewidth=lw, ecolor='r', capsize=3)
                ax0[0,0].plot(Y_alt["N"][:,0], altgrid/1000, 'or', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[0,0].errorbar(X_alt["N"][:,-1,0], altgrid/1000, xerr=X_alt["N"][:,-1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)          #
                ax0[0,0].plot(X_alt["N"][:,-1,0], altgrid/1000, 'dk', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')       
                ax0[0,0].add_artist(fig0_ata)             
                ax0[0,1].axvspan(rsp_dat["crs"][0,1]-rsp_dat["crs_unc"][0,1], rsp_dat["crs"][0,1]+rsp_dat["crs_unc"][0,1], alpha=0.3, color='xkcd:fuchsia')
                ax0[0,1].vlines(rsp_dat["crs"][0,1],0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw) 
                ax0[0,1].errorbar(Alt_Binned_IS_data["crs"][:,0,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["crs"][:,0,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)        
                ax0[0,1].plot(Alt_Binned_IS_data["crs"][:,0,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     #
                ax0[0,1].add_artist(fig0_atb)
                iw = 0
                for iwvl in range(n_wvl):
                    ax0[0,2].errorbar(Y_alt["ldr"][:,iwvl,1], altgrid/1000, xerr=Y_alt["ldr"][:,iwvl,1], linestyle='none', elinewidth=lw, ecolor=wvlclrs_hsrl[iw], capsize=3)      
                    ax0[0,2].plot(Y_alt["ldr"][:,iwvl,1], altgrid/1000, 'o', color=wvlclrs_hsrl[iw], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     # 
                    ax0[0,2].errorbar(X_alt["ldr"][:,-1,iwvl,0], altgrid/1000, xerr=X_alt["ldr"][:,-1,iwvl,1], linestyle='none', elinewidth=lw, ecolor=wvlclrs_is[iw], capsize=3)      
                    ax0[0,2].plot(X_alt["ldr"][:,-1,iwvl,0], altgrid/1000, '^', color=wvlclrs_is[iw], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     # 
                    iw += 1 
                ax0[0,2].add_artist(fig0_atc)                                      
                ax0[0,3].errorbar(Alt_Binned_IS_data["rh"][:,0], altgrid/1000, xerr=Alt_Binned_IS_data["rh"][:,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)     #
                ax0[0,3].plot(Alt_Binned_IS_data["rh"][:,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[0,3].add_artist(fig0_atd)
                ax0[1,0].errorbar(Alt_Binned_IS_data["kappa"][:,0], altgrid/1000, xerr=Alt_Binned_IS_data["kappa"][:,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)     #
                ax0[1,0].errorbar(Alt_Binned_IS_data["kappa"][:,0], altgrid/1000, xerr=Alt_Binned_IS_data["kappa"][:,0]*0.2, linestyle='none', elinewidth=lw, ecolor='b', capsize=3)     #
                ax0[1,0].plot(Alt_Binned_IS_data["kappa"][:,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[1,0].add_artist(fig0_ate)
                ax0[1,1].axvspan(rsp_dat["rri_f"]-rsp_dat["rri_f_unc"], rsp_dat["rri_f"]+rsp_dat["rri_f_unc"], alpha=0.3, color='xkcd:fuchsia')
                ax0[1,1].vlines(rsp_dat["rri_f"],0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw) 
                ax0[1,1].errorbar(Alt_Binned_IS_data["rri"][:,0,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["rri"][:,0,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)     #
                ax0[1,1].plot(Alt_Binned_IS_data["rri"][:,0,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[1,1].add_artist(fig0_atf)
                ax0[1,2].axvspan(rsp_dat["iri_f"]-rsp_dat["iri_f_unc"], rsp_dat["iri_f"]+rsp_dat["iri_f_unc"], alpha=0.3, color='xkcd:fuchsia')
                ax0[1,2].vlines(rsp_dat["iri_f"],0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)     
                ax0[1,2].errorbar(Alt_Binned_IS_data["iri"][:,0,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["iri"][:,0,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)     #  
                ax0[1,2].errorbar(Alt_Binned_IS_data["iri"][:,0,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["iri"][:,0,1,0]*0.2, linestyle='none', elinewidth=lw, ecolor='b', capsize=3)     #          
                ax0[1,2].plot(Alt_Binned_IS_data["iri"][:,0,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[1,2].add_artist(fig0_atg)
                ax0[1,3].axvspan(rsp_dat["ssa"][0,1]-rsp_dat["ssa_unc"][0,1], rsp_dat["ssa"][0,1]+rsp_dat["ssa_unc"][0,1], alpha=0.3, color='g')
                ax0[1,3].vlines(rsp_dat["ssa"][0,1],0,10, colors='g', linestyles='dashed',linewidth=lw)  
                ax0[1,3].axvspan(rsp_dat["ssa"][-1,1]-rsp_dat["ssa_unc"][-1,1], rsp_dat["ssa"][-1,1]+rsp_dat["ssa_unc"][-1,1], alpha=0.3, color='xkcd:fuchsia')
                ax0[1,3].vlines(rsp_dat["ssa"][-1,1],0,10, colors='xkcd:fuchsia', linestyles='solid',linewidth=lw)                
                ax0[1,3].errorbar(Alt_Binned_IS_data["ssa"][:,-1,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["ssa"][:,-1,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)     #
                ax0[1,3].plot(Alt_Binned_IS_data["ssa"][:,-1,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[1,3].errorbar(Alt_Binned_IS_data["ssa"][:,0,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["ssa"][:,0,1,1], linestyle='none', elinewidth=lw, ecolor='g', capsize=3)     #
                ax0[1,3].plot(Alt_Binned_IS_data["ssa"][:,0,1,0], altgrid/1000, 'dg', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[1,3].add_artist(fig0_ath)
                ax0[2,0].axvspan(rsp_dat["reff"][0]-rsp_dat["reff_unc"][0], rsp_dat["reff"][0]+rsp_dat["reff_unc"][0], alpha=0.3, color='xkcd:fuchsia')
                ax0[2,0].vlines(rsp_dat["reff"][0],0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)           
                ax0[2,0].errorbar(Alt_Binned_IS_data["reff"][:,0,0], altgrid/1000, xerr=Alt_Binned_IS_data["reff"][:,0,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)          
                ax0[2,0].plot(Alt_Binned_IS_data["reff"][:,0,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     #
                ax0[2,0].add_artist(fig2_ati)
                ax0[2,1].axvspan(rsp_dat["reff"][1]-rsp_dat["reff_unc"][1], rsp_dat["reff"][1]+rsp_dat["reff_unc"][1], alpha=0.3, color='xkcd:fuchsia')
                ax0[2,1].vlines(rsp_dat["reff"][1],0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)     
                ax0[2,1].errorbar(Alt_Binned_IS_data["reff"][:,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["reff"][:,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3) 
                ax0[2,1].plot(Alt_Binned_IS_data["reff"][:,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     #
                ax0[2,1].add_artist(fig2_atj)
                ax0[2,2].axvspan(rsp_dat["veff"][0]-rsp_dat["veff_unc"][0], rsp_dat["veff"][0]+rsp_dat["veff_unc"][0], alpha=0.3, color='xkcd:fuchsia')
                ax0[2,2].vlines(rsp_dat["veff"][0],0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)           
                ax0[2,2].errorbar(Alt_Binned_IS_data["veff"][:,0,0], altgrid/1000, xerr=Alt_Binned_IS_data["veff"][:,0,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)           
                ax0[2,2].plot(Alt_Binned_IS_data["veff"][:,0,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     #
                ax0[2,2].add_artist(fig2_atk)
                ax0[2,3].axvspan(rsp_dat["veff"][1]-rsp_dat["veff_unc"][1], rsp_dat["veff"][1]+rsp_dat["veff_unc"][1], alpha=0.3, color='xkcd:fuchsia')
                ax0[2,3].vlines(rsp_dat["veff"][1],0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)         
                ax0[2,3].errorbar(Alt_Binned_IS_data["veff"][:,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["veff"][:,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)  #         
                ax0[2,3].plot(Alt_Binned_IS_data["veff"][:,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')    
                ax0[2,3].add_artist(fig2_atl)                
                fig0.savefig(f"{filename_prefix}-External_Closure_opt_mphys_alt_{legid}_{ouput_filename_suffix}.png", dpi=300)
                for panls in range(len(ax0[:,0])):
                    for panls2 in range(len(ax0[0,:])):
                        ax0[panls,panls2].cla()
                        for artist in ax0[panls,panls2].lines + ax0[panls,panls2].collections:
                            artist.remove()
            else:
                ax0[0,0].errorbar(X_alt["N"][:,-1,0], altgrid/1000, xerr=X_alt["N"][:,-1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)          #
                ax0[0,0].plot(X_alt["N"][:,-1,0], altgrid/1000, 'dk', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')       
                ax0[0,0].add_artist(fig0_ata)             
                ax0[0,1].errorbar(Alt_Binned_IS_data["crs"][:,0,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["crs"][:,0,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)        
                ax0[0,1].plot(Alt_Binned_IS_data["crs"][:,0,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     #
                ax0[0,1].add_artist(fig0_atb)
                iw = 0
                for iwvl in range(n_wvl):
                    ax0[0,2].errorbar(X_alt["ldr"][:,-1,iwvl,0], altgrid/1000, xerr=X_alt["ldr"][:,-1,iwvl,1], linestyle='none', elinewidth=lw, ecolor=wvlclrs_is[iw], capsize=3)      
                    ax0[0,2].plot(X_alt["ldr"][:,-1,iwvl,0], altgrid/1000, '^', color=wvlclrs_is[iw], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     # 
                    iw += 1 
                ax0[0,2].add_artist(fig0_atc)                                      
                ax0[0,3].errorbar(Alt_Binned_IS_data["rh"][:,0], altgrid/1000, xerr=Alt_Binned_IS_data["rh"][:,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)     #
                ax0[0,3].plot(Alt_Binned_IS_data["rh"][:,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[0,3].add_artist(fig0_atd)
                ax0[1,0].errorbar(Alt_Binned_IS_data["kappa"][:,0], altgrid/1000, xerr=Alt_Binned_IS_data["kappa"][:,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)     #
                ax0[1,0].errorbar(Alt_Binned_IS_data["kappa"][:,0], altgrid/1000, xerr=Alt_Binned_IS_data["kappa"][:,0]*0.2, linestyle='none', elinewidth=lw, ecolor='b', capsize=3)     #
                ax0[1,0].plot(Alt_Binned_IS_data["kappa"][:,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[1,0].add_artist(fig0_ate)
                ax0[1,1].errorbar(Alt_Binned_IS_data["rri"][:,0,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["rri"][:,0,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)     #
                ax0[1,1].plot(Alt_Binned_IS_data["rri"][:,0,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[1,1].add_artist(fig0_atf)  
                ax0[1,2].errorbar(Alt_Binned_IS_data["iri"][:,0,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["iri"][:,0,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)     #  
                ax0[1,2].errorbar(Alt_Binned_IS_data["iri"][:,0,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["iri"][:,0,1,0]*0.2, linestyle='none', elinewidth=lw, ecolor='b', capsize=3)     #          
                ax0[1,2].plot(Alt_Binned_IS_data["iri"][:,0,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[1,2].add_artist(fig0_atg)              
                ax0[1,3].errorbar(Alt_Binned_IS_data["ssa"][:,-1,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["ssa"][:,-1,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)     #
                ax0[1,3].plot(Alt_Binned_IS_data["ssa"][:,-1,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[1,3].errorbar(Alt_Binned_IS_data["ssa"][:,0,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["ssa"][:,0,1,1], linestyle='none', elinewidth=lw, ecolor='g', capsize=3)     #
                ax0[1,3].plot(Alt_Binned_IS_data["ssa"][:,0,1,0], altgrid/1000, 'dg', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                ax0[1,3].add_artist(fig0_ath)      
                ax0[2,0].errorbar(Alt_Binned_IS_data["reff"][:,0,0], altgrid/1000, xerr=Alt_Binned_IS_data["reff"][:,0,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)          
                ax0[2,0].plot(Alt_Binned_IS_data["reff"][:,0,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     #
                ax0[2,0].add_artist(fig2_ati)
                ax0[2,1].errorbar(Alt_Binned_IS_data["reff"][:,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["reff"][:,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3) 
                ax0[2,1].plot(Alt_Binned_IS_data["reff"][:,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     #
                ax0[2,1].add_artist(fig2_atj)        
                ax0[2,2].errorbar(Alt_Binned_IS_data["veff"][:,0,0], altgrid/1000, xerr=Alt_Binned_IS_data["veff"][:,0,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)           
                ax0[2,2].plot(Alt_Binned_IS_data["veff"][:,0,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')     #
                ax0[2,2].add_artist(fig2_atk)       
                ax0[2,3].errorbar(Alt_Binned_IS_data["veff"][:,1,0], altgrid/1000, xerr=Alt_Binned_IS_data["veff"][:,1,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)  #         
                ax0[2,3].plot(Alt_Binned_IS_data["veff"][:,1,0], altgrid/1000, 'ok', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')    
                ax0[2,3].add_artist(fig2_atl)                
                fig0.savefig(f"{filename_prefix}-External_Closure_opt_mphys_alt_{legid}_{ouput_filename_suffix}.png", dpi=300)
                for panls in range(len(ax0[:,0])):
                    for panls2 in range(len(ax0[0,:])):
                        ax0[panls,panls2].cla()
                        for artist in ax0[panls,panls2].lines + ax0[panls,panls2].collections:
                            artist.remove()
            plt.close(fig0)
              #  
            iw = 0
            for iwvl in range(n_wvl):
                ax1[0,iw].errorbar(Y_alt["ext"][:,iwvl,0], altgrid/1000, xerr=Y_alt["ext"][:,iwvl,1], linestyle='none', elinewidth=lw, ecolor=wvlclrs_hsrl[iw], capsize=3)
                ax1[0,iw].plot(Y_alt["ext"][:,iwvl,0], altgrid/1000, 'o', color=wvlclrs_hsrl[iw], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k') 
                ax1[0,iw].errorbar(X_alt["ext"][:,-1,iwvl,0], altgrid/1000, xerr=X_alt["ext"][:,-1,iwvl,1][iwvl], linestyle='none', elinewidth=lw, ecolor=wvlclrs_is[iw], capsize=3)        #
                ax1[0,iw].plot(X_alt["ext"][:,-1,iwvl,0], altgrid/1000, '^', color=wvlclrs_is[iw], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')   
                ax1[0,iw].add_artist(fig1_at1[iwvl])      
                ax1[1,iw].errorbar(Y_alt["bsc"][:,iwvl,0], altgrid/1000, xerr=Y_alt["bsc"][:,iwvl,1], linestyle='none', elinewidth=lw, ecolor=wvlclrs_hsrl[iw], capsize=3)
                ax1[1,iw].plot(Y_alt["bsc"][:,iwvl,0], altgrid/1000, 'o', color=wvlclrs_hsrl[iw], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')   
                ax1[1,iw].errorbar(X_alt["bsc"][:,-1,iwvl,0], altgrid/1000, xerr=X_alt["bsc"][:,-1,iwvl,1], linestyle='none', elinewidth=lw, ecolor=wvlclrs_is[iw], capsize=3)  #
                ax1[1,iw].plot(X_alt["bsc"][:,-1,iwvl,0], altgrid/1000, '^', color=wvlclrs_is[iw], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')   
                ax1[1,iw].add_artist(fig1_at2[iwvl])
                if HSRL_wvl[iwvl] != 1064:
                    ax1[2,iw].errorbar(Y_alt["lr"][:,iwvl,0], altgrid/1000, xerr=Y_alt["lr"][:,iwvl,1], linestyle='none', elinewidth=lw, ecolor=wvlclrs_hsrl[iw], capsize=3)
                    ax1[2,iw].plot(Y_alt["lr"][:,iwvl,0], altgrid/1000, 'o', color=wvlclrs_hsrl[iw], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                    ax1[2,iw].errorbar(X_alt["lr"][:,-1,iwvl,0], altgrid/1000, xerr=X_alt["lr"][:,-1,iwvl,1], linestyle='none', elinewidth=lw, ecolor=wvlclrs_is[iw], capsize=3)        #
                    ax1[2,iw].plot(X_alt["lr"][:,-1,iwvl,0], altgrid/1000, '^', color=wvlclrs_is[iw], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')
                    ax1[2,iw].add_artist(fig1_at3[iwvl])      
                iw += 1   
                ax1[2,2].errorbar(Alt_Binned_IS_data["ctotextratio"][:,0], altgrid/1000, xerr=Alt_Binned_IS_data["ctotextratio"][:,1], linestyle='none', elinewidth=lw, ecolor='k', capsize=3)  #
                ax1[2,2].plot(Alt_Binned_IS_data["ctotextratio"][:,0], altgrid/1000, '^', color='k', markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')   
                ax1[2,2].add_artist(fig1_at4)                                                 
            fig1.savefig(f"{filename_prefix}-External_Closure_extwvl_alt_{legid}_{ouput_filename_suffix}.png", dpi=300)
            for panls in range(len(ax1[:,0])):
                for panls2 in range(len(ax1[0,:])):
                    ax1[panls,panls2].cla()
                    for artist in ax1[panls,panls2].lines + ax1[panls,panls2].collections:
                        artist.remove()
            plt.close(fig1)
           #    
          #
            im3 = ax3[0].pcolor(D_grd["dpg"]*1000,altgrid/1000, fine_sd[:,:,0,0], cmap=cmap3, norm=norm3)    
            ax3[0].add_artist(fig3_ata)   
            im3 = ax3[1].pcolor(D_grd["dpg"]*1000,altgrid/1000, fine_sd_adj[:,:,0,0], cmap=cmap3, norm=norm3)
            ax3[1].add_artist(fig3_atb)
            cbar =  plt.colorbar(im3,cax=cax3,cmap=cmap3, norm=norm3,boundaries=bounds3,ticks=bounds3)
            cbar.set_ticklabels(boundsLbs3)    
            cbar.outline.set_linewidth(1.5)
            cbar.ax.tick_params(length=8, width=1.5, which="major")
            cbar.set_label(r"$\dfrac{{\rm d}N}{{\rm d} \log D} \ (\rm cm^{-3})$",labelpad=-40)
            fig3.savefig(f"{filename_prefix}-External_Closure_SD_alt_{legid}_{ouput_filename_suffix}", dpi=300)
            for panls in range(len(ax3)):
                for artist in ax3[panls].lines + ax3[panls].collections:
                    artist.remove()
            #plt.close(fig3)
            #
            im4 = ax4[0].pcolor(D_grd["dpg"]*1000,altgrid/1000, fine_sd[:,:,1,0], cmap=cmap4, norm=norm4)    
            ax4[0].add_artist(fig4_ata)   
            im4 = ax4[1].pcolor(D_grd["dpg"]*1000,altgrid/1000, fine_sd_adj[:,:,1,0], cmap=cmap4, norm=norm4)
            ax4[1].add_artist(fig4_atb)
            cbar =  plt.colorbar(im4,cax=cax4,cmap=cmap4, norm=norm4,boundaries=bounds4,ticks=bounds4)
            cbar.set_ticklabels(boundsLbs4)    
            cbar.outline.set_linewidth(1.5)
            cbar.ax.tick_params(length=8, width=1.5, which="major")
            cbar.set_label(r"$\dfrac{{\rm d}S}{{\rm d} \log D} \ (\rm \mu m^2 \ cm^{-3})$",labelpad=-35)
            fig4.savefig(f"{filename_prefix}-External_Closure_SDA_alt_{legid}_{ouput_filename_suffix}", dpi=300)
            for panls in range(len(ax4)):
                for artist in ax4[panls].lines + ax4[panls].collections:
                    artist.remove()
            #plt.close(fig4)
           #
            im5 = ax5[0].pcolor(D_grd["dpg"]*1000,altgrid/1000, fine_sd[:,:,2,0], cmap=cmap5, norm=norm5)    
            ax5[0].add_artist(fig5_ata)   
            im5 = ax5[1].pcolor(D_grd["dpg"]*1000,altgrid/1000, fine_sd_adj[:,:,2,0], cmap=cmap5, norm=norm5)
            ax5[1].add_artist(fig5_atb)
            cbar =  plt.colorbar(im5,cax=cax5,cmap=cmap5, norm=norm5,boundaries=bounds5,ticks=bounds5)
            cbar.set_ticklabels(boundsLbs5)    
            cbar.outline.set_linewidth(1.5)
            cbar.ax.tick_params(length=8, width=1.5, which="major")
            cbar.set_label(r"$\dfrac{{\rm d}V}{{\rm d} \log D} \ (\rm \mu m^3 \ cm^{-3})$",labelpad=-5)
            fig5.savefig(f"{filename_prefix}-External_Closure_SDV_alt_{legid}_{ouput_filename_suffix}", dpi=300)
            for panls in range(len(ax5)):
                for artist in ax5[panls].lines + ax5[panls].collections:
                    artist.remove()
            #plt.close(fig5)
            if coarseflg is not None:
               #
                crsidx = np.where((coarse_diameter>1))[0]
                finidx = np.where((D_grd["dpg"]<=1))[0]
                im3 = ax3[0].pcolor(D_grd["dpg"][finidx]*1000,altgrid/1000, fine_sd[:,finidx,0,0], cmap=cmap3, norm=norm3)
                im3 = ax3[0].pcolor(coarse_diameter[crsidx]*1000,altgrid/1000, sd_c[:,crsidx,0,0], cmap=cmap3, norm=norm3)
                ax3[0].vlines(1000,0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)
                ax3[0].add_artist(fig3_ata)   
                im3 = ax3[1].pcolor(D_grd["dpg"][finidx]*1000,altgrid/1000, fine_sd_adj[:,finidx,0,0], cmap=cmap3, norm=norm3)
                im3 = ax3[1].pcolor(coarse_diameter[crsidx]*1000,altgrid/1000, sd_c[:,crsidx,0,0], cmap=cmap3, norm=norm3)
                ax3[1].vlines(1000,0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)
                ax3[1].add_artist(fig3_atb)
                cbar =  plt.colorbar(im3,cax=cax3,cmap=cmap3, norm=norm3,boundaries=bounds3,ticks=bounds3)
                cbar.set_ticklabels(boundsLbs3)    
                cbar.outline.set_linewidth(1.5)
                cbar.ax.tick_params(length=8, width=1.5, which="major")
                cbar.set_label(r"$\dfrac{{\rm d}N}{{\rm d} \log D} \ (\rm cm^{-3})$",labelpad=-20)
                fig3.savefig(f"{filename_prefix}-External_Closure_SD_coarse_alt_{legid}_{ouput_filename_suffix}", dpi=300)
                for panls in range(len(ax3)):
                    for artist in ax3[panls].lines + ax3[panls].collections:
                        artist.remove()
                
               #
                im4 = ax4[0].pcolor(D_grd["dpg"][finidx]*1000,altgrid/1000, fine_sd[:,finidx,1,0], cmap=cmap4, norm=norm4)
                im4 = ax4[0].pcolor(coarse_diameter[crsidx]*1000,altgrid/1000, sd_c[:,crsidx,1,0], cmap=cmap4, norm=norm4)
                ax4[0].vlines(1000,0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)    
                ax4[0].add_artist(fig4_ata)  
                im4 = ax4[1].pcolor(D_grd["dpg"][finidx]*1000,altgrid/1000, fine_sd_adj[:,finidx,1,0], cmap=cmap4, norm=norm4) 
                im4 = ax4[1].pcolor(coarse_diameter[crsidx]*1000,altgrid/1000, sd_c[:,crsidx,1,0], cmap=cmap4, norm=norm4)
                ax4[1].vlines(1000,0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)
                ax4[1].add_artist(fig4_atb)
                cbar =  plt.colorbar(im4,cax=cax4,cmap=cmap4, norm=norm4,boundaries=bounds4,ticks=bounds4)
                cbar.set_ticklabels(boundsLbs4)    
                cbar.outline.set_linewidth(1.5)
                cbar.ax.tick_params(length=8, width=1.5, which="major")
                cbar.set_label(r"$\dfrac{{\rm d}S}{{\rm d} \log D} \ (\rm \mu m^2 \ cm^{-3})$",labelpad=-15)
                fig4.savefig(f"{filename_prefix}-External_Closure_SDA_coarse_alt_{legid}_{ouput_filename_suffix}", dpi=300)
                for panls in range(len(ax4)):
                    for artist in ax4[panls].lines + ax4[panls].collections:
                        artist.remove()
               #
                im5 = ax5[0].pcolor(D_grd["dpg"][finidx]*1000,altgrid/1000, fine_sd[:,finidx,2,0], cmap=cmap5, norm=norm5)
                im5 = ax5[0].pcolor(coarse_diameter[crsidx]*1000,altgrid/1000, sd_c[:,crsidx,2,0], cmap=cmap5, norm=norm5)
                ax5[0].vlines(1000,0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)    
                ax5[0].add_artist(fig5_ata)   
                im5 = ax5[1].pcolor(D_grd["dpg"][finidx]*1000,altgrid/1000, fine_sd_adj[:,finidx,2,0], cmap=cmap5, norm=norm5)
                im5 = ax5[1].pcolor(coarse_diameter[crsidx]*1000,altgrid/1000, sd_c[:,crsidx,2,0], cmap=cmap5, norm=norm5)
                ax5[1].vlines(1000,0,10, colors='xkcd:fuchsia', linestyles='dashed',linewidth=lw)
                ax5[1].add_artist(fig5_atb)
                cbar =  plt.colorbar(im5,cax=cax5,cmap=cmap5, norm=norm5,boundaries=bounds5,ticks=bounds5)
                cbar.set_ticklabels(boundsLbs5)    
                cbar.outline.set_linewidth(1.5)
                cbar.ax.tick_params(length=8, width=1.5, which="major")
                cbar.set_label(r"$\dfrac{{\rm d}V}{{\rm d} \log D} \ (\rm \mu m^3 \ cm^{-3})$",labelpad=0)
                fig5.savefig(f"{filename_prefix}-External_Closure_SDV_coarse_alt_{legid}_{ouput_filename_suffix}", dpi=300)
                for panls in range(len(ax5)):
                    for artist in ax5[panls].lines + ax5[panls].collections:
                        artist.remove()
            plt.close(fig3)
            plt.close(fig4)
            plt.close(fig5)            
            iaid = 0
            for aid_key in HSRLAerosolType:
                if HSRLAerosolType[aid_key] == 10:
                    cnum = np.full(len(aid_counts_rebin[:,iaid]),HSRLAerosolType[aid_key]-1)
                    sp = ax10.scatter(aid_counts_rebin[:,iaid], altgrid/1000, c=cnum, cmap=aid_cmap, norm=aid_norm)
                    ax10.plot(aid_counts_rebin[:,iaid], altgrid/1000, linestyle=aid_lines[iaid-1], marker=aid_shapes[iaid-1], color=aid_colors[iaid-1], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')  
                else:
                    cnum = np.full(len(aid_counts_rebin[:,iaid]),HSRLAerosolType[aid_key])
                    sp = ax10.scatter(aid_counts_rebin[:,iaid], altgrid/1000, c=cnum, cmap=aid_cmap, norm=aid_norm)
                    ax10.plot(aid_counts_rebin[:,iaid], altgrid/1000, linestyle=aid_lines[iaid], marker=aid_shapes[iaid], color=aid_colors[iaid], markeredgewidth=1.5, markersize=7.5, markeredgecolor='k')   
                iaid += 1 
            cbar = plt.colorbar(sp, cax=cax10, cmap=aid_cmap, ticks=np.arange(len(aid_categories)), norm=aid_norm, boundaries=aid_bounds, label='Aerosol Type')# 5. Add the Colorbar, cmap=aid_cmap, ax=ax10, ticks=np.arange(len(aid_categories)), norm=aid_norm,boundaries=aid_bounds, 
            cbar.ax.set_yticklabels(aid_categories) # Set custom tick labels for categories  
            cbar.outline.set_linewidth(1.5)
            cbar.ax.tick_params(length=8, width=1.5, which="major")               
            fig10.savefig(f"{filename_prefix}-External_Closure_AIDcount_alt_{legid}_{ouput_filename_suffix}.png", dpi=300)
            ax10.cla()
            for artist in ax10.lines + ax10.collections:
                artist.remove()
            plt.close(fig10)
        i1 += 1            
#leg meta data    
    if len(IS_col_data["frmttime"])>0:
        output_dictionary = {}
        output_dictionary['data'] = {}
        output_dictionary['data']['column'] = {}
        output_dictionary['data']['vertical'] = {}
        output_dictionary['sigma'] = {}
        output_dictionary['sigma']['column'] = {}
        output_dictionary['sigma']['vertical'] = {}  
        output_dictionary['data']['column']['legstart_date_time'] = np.squeeze(IS_col_data["frmttime"][:,0])
        output_dictionary['data']['column']['legend_date_time']= np.squeeze(IS_col_data["frmttime"][:,-1])
        output_dictionary['data']['column']['RSP_date_time'] = np.squeeze(RSP_col_data['frmttime'][:,1])
        output_dictionary['data']['column']['Min_insitu_altitude_m'] = np.squeeze(np.nanmin(IS_alt_data['alt'][...,0],axis=1))
        output_dictionary['data']['column']['Max_insitu_altitude_m'] = np.squeeze(np.nanmax(IS_alt_data['alt'][...,0],axis=1))
        output_dictionary['data']['column']['LegID'] = np.array(list(Collocated_data_final))
        output_dictionary['data']['column']['aircraft_horizontal_separation_m'] = np.squeeze(IS_col_data["sepatation"]) 
        output_dictionary['data']['column']['count'] = np.squeeze(IS_col_data["ext_datacount"][:,1]) 
        output_dictionary['data']['column']['min_lat_IS'] = np.squeeze(np.nanmin(IS_alt_data["lat"][...,0],axis=1))
        output_dictionary['data']['column']['max_lat_IS'] = np.squeeze(np.nanmax(IS_alt_data["lat"][...,0],axis=1))
        output_dictionary['data']['column']['lat_RSP'] = RSP_col_data["lat"][:,0]
        output_dictionary['data']['column']['min_lon_IS'] = np.squeeze(np.nanmin(IS_alt_data["lat"][...,0],axis=1))
        output_dictionary['data']['column']['max_lon_IS'] = np.squeeze(np.nanmax(IS_alt_data["lat"][...,0],axis=1))
        output_dictionary['data']['column']['lon_RSP'] = RSP_col_data["lon"][:,0]
        output_dictionary['data']['column']['min_LDR_532_HSRL'] = np.squeeze(np.nanmin(HSRL_alt_data["ldr"][...,1,0],axis=1)) 
        output_dictionary['data']['column']['mean_LDR_532_HSRL'] =  np.squeeze(np.nanmean(HSRL_alt_data["ldr"][...,1,0],axis=1)) 
        output_dictionary['data']['column']['max_LDR_532_HSRL'] =  np.squeeze(np.nanmax(HSRL_alt_data["ldr"][...,1,0],axis=1)) 
        output_dictionary['data']['column']['min_N5um_IS_cm-3'] = np.squeeze(np.nanmin(IS_alt_data['N5'][...,0],axis=1)) 
        output_dictionary['data']['column']['max_N5um_IS_cm-3'] = np.squeeze(np.nanmax(IS_alt_data['N5'][...,0],axis=1)) 
        smoke_counts = np.nansum(HSRL_alt_data["aid"][:,:,4:5],axis=2)
        elev_Smoke_counts = np.nansum(smoke_counts[:,altgrid>2500],axis=1)
        output_dictionary['data']['column']['smoke_counts_above_2.5km'] = np.squeeze(elev_Smoke_counts)
        output_dictionary['data']['column']['min_CtoT_ext'] = np.squeeze(np.nanmin(IS_alt_data["ctotextratio"][...,0],axis=1))
        output_dictionary['data']['column']['max_CtoT_ext'] = np.squeeze(np.nanmax(IS_alt_data["ctotextratio"][...,0],axis=1))
        output_dictionary['data']['column']['mean_CtoT_ext'] =  np.squeeze(np.nanmean(IS_alt_data["ctotextratio"][...,0],axis=1))
        iaid = 0
        for aid_key in HSRLAerosolType:
            aid_cnt = np.squeeze(np.nansum(HSRL_alt_data["aid"][:,:,iaid],axis=1))
            output_dictionary['data']['column'][f'{aid_key}_count']  = np.squeeze(aid_cnt) 
            iaid += 1
        for i_mode in range(n_modes):
            if i_mode<n_modes-1:
                output_dictionary['data']['column'][f'Collocated_{modes[i_mode]}_reff_IS_um'] = np.squeeze(IS_col_data["reff"][:,i_mode,0])
                output_dictionary['sigma']['column'][f'Collocated_{modes[i_mode]}_reff_IS_um'] = np.squeeze(IS_col_data["reff"][:,i_mode,1])
                output_dictionary['data']['column'][f'Collocated_{modes[i_mode]}_reff_RSP_um'] = np.squeeze(RSP_col_data["reff"][:,i_mode,0])
                output_dictionary['sigma']['column'][f'Collocated_{modes[i_mode]}_reff_RSP_um'] = np.squeeze(RSP_col_data["reff_unc"][:,i_mode,0])
                output_dictionary['data']['column'][f'Collocated_{modes[i_mode]}_veff_IS'] = np.squeeze(IS_col_data["veff"][:,i_mode,0])
                output_dictionary['sigma']['column'][f'Collocated_{modes[i_mode]}_veff_IS'] = np.squeeze(IS_col_data["veff"][:,i_mode,1])
                output_dictionary['data']['column'][f'Collocated_{modes[i_mode]}_veff_RSP'] = np.squeeze(RSP_col_data["veff"][:,i_mode,0])
                output_dictionary['sigma']['column'][f'Collocated_{modes[i_mode]}_veff_RSP'] = np.squeeze(RSP_col_data["veff_unc"][:,i_mode,0])
            for iwvl in range(n_wvl):
                if iwvl<n_wvl-1:
                    output_dictionary['data']['column'][f'{modes[i_mode]}_AOT_{HSRL_wvl[iwvl]}_RSP'] = np.squeeze(RSP_col_data["aod"][:,i_mode,iwvl,0])
                output_dictionary['data']['column'][f'Collocated_{modes[i_mode]}_ssa_{HSRL_wvl[iwvl]}_IS'] = np.squeeze(IS_col_data["ssa"][:,i_mode,iwvl,0])
                output_dictionary['sigma']['column'][f'Collocated_{modes[i_mode]}_ssa_{HSRL_wvl[iwvl]}_IS'] = np.squeeze(IS_col_data["ssa"][:,i_mode,iwvl,1])
                output_dictionary['data']['column'][f'Collocated_{modes[i_mode]}_ssa_{HSRL_wvl[iwvl]}_RSP'] = np.squeeze(RSP_col_data["ssa"][:,i_mode,iwvl,0])
                output_dictionary['sigma']['column'][f'Collocated_{modes[i_mode]}_ssa_{HSRL_wvl[iwvl]}_RSP'] = np.squeeze(RSP_col_data["ssa_unc"][:,i_mode,iwvl,0])
        for iwvl in range(n_wvl-1):       
            output_dictionary['data']['column'][f'{modes[-1]}_AOT_{HSRL_wvl[iwvl]}_HSRL'] = np.squeeze(HSRL_col_data["aod"][:,iwvl,0])    
            output_dictionary['sigma']['column'][f'{modes[-1]}_AOT_{HSRL_wvl[iwvl]}_HSRL'] = np.squeeze(HSRL_col_data["aod"][:,iwvl,1])
        output_dictionary['data']['column']['Collocated_fine_ssa_LARGE'] = np.squeeze(IS_col_data["ssa_m"][:,0])
        output_dictionary['sigma']['column']['Collocated_fine_ssa_LARGE'] = np.squeeze(IS_col_data["ssa_m"][:,1])
        output_dictionary['data']['column']['Collocated_fine_rri_IS'] = np.squeeze(IS_col_data["rri"][:,0,1,0])
        output_dictionary['sigma']['column']['Collocated_fine_rri_IS'] = np.squeeze(IS_col_data["rri"][:,0,1,1])
        output_dictionary['data']['column']['Collocated_fine_rri_RSP'] = np.squeeze(RSP_col_data["rri_f"][:,0])
        output_dictionary['sigma']['column']['Collocated_fine_rri_RSP'] = np.squeeze(RSP_col_data["rri_f_unc"][:,0])    
        output_dictionary['data']['column']['Collocated_fine_iri_IS'] = np.squeeze(IS_col_data["iri"][:,0,1,0])
        output_dictionary['sigma']['column']['Collocated_fine_iri_IS']= np.squeeze(IS_col_data["iri"][:,0,1,1])
        output_dictionary['data']['column']['Collocated_fine_iri_RSP']  = np.squeeze(RSP_col_data["iri_f"][:,0])
        output_dictionary['sigma']['column']['Collocated_fine_iri_RSP'] = np.squeeze(RSP_col_data["iri_f_unc"][:,0])   
        output_dictionary['data']['column']['Collocated_optical_N_IS_cm-3'] = np.squeeze(IS_col_data["N"][:,-1,0])
        output_dictionary['sigma']['column']['Collocated_optical_N_IS_cm-3'] = np.squeeze(IS_col_data["N"][:,-1,1])
        output_dictionary['data']['column']['Collocated_optical_N_RSP_cm-3'] = np.squeeze(HSRL_col_data["N"][:,0])
        output_dictionary['sigma']['column']['Collocated_optical_N_RSP_cm-3'] = np.squeeze(HSRL_col_data["N"][:,1])   
        output_dictionary['data']['column']['Collocated_optical_kext_IS_um2'] = np.squeeze(IS_col_data["crs"][:,0,1,0])
        output_dictionary['sigma']['column']['Collocated_optical_kext_IS_um2'] = np.squeeze(IS_col_data["crs"][:,0,1,1])
        output_dictionary['data']['column']['Collocated_optical_kext_RSP_um2'] = np.squeeze(RSP_col_data["crs"][:,0,1,0])
        output_dictionary['sigma']['column']['Collocated_optical_kext_RSP_um2'] = np.squeeze(RSP_col_data["crs_unc"][:,0,1,0])
        output_dictionary['data']['vertical']['Collocated_optical_N_legstats'] = naltstats
        output_dictionary['data']['vertical']['Collocated_optical_N_IS'] = IS_alt_data["N"][:,:,-1,0]
        output_dictionary['sigma']['vertical']['Collocated_optical_N_IS'] = IS_alt_data["N"][:,:,-1,1]
        output_dictionary['data']['vertical']['Collocated_optical_N_HSRL+RSP'] = HSRL_alt_data["N"][:,:,0]
        output_dictionary['sigma']['vertical']['Collocated_optical_N_HSRL+RSP'] = HSRL_alt_data["N"][:,:,1]
        finalkys = ["ext_coef","bsc_coef","ldr","lr"]
        finalkys2 = ["ext","bsc","ldr","lr"]
        for key in finalkys:
            output_dictionary['data']['vertical'][f'Collocated_total_{key}_legstats'] = {}
            output_dictionary['data']['vertical'][f'Collocated_total_{key}_IS'] = {}
            output_dictionary['sigma']['vertical'][f'Collocated_total_{key}_IS'] = {}
            output_dictionary['data']['vertical'][f'Collocated_total_{key}_HSRL'] = {}
            output_dictionary['sigma']['vertical'][f'Collocated_total_{key}_HSRL'] = {}
        for key in range(n_wvl):
            for idata in range(len(finalkys)-1): 
                output_dictionary['data']['vertical'][f'Collocated_total_{finalkys[idata]}_legstats'][HSRL_wvl[key]] = altstats[finalkys2[idata]][HSRL_wvl[key]]
                output_dictionary['data']['vertical'][f'Collocated_total_{finalkys[idata]}_IS'][HSRL_wvl[key]] = IS_alt_data[finalkys2[idata]][:,:,-1,key,0]
                output_dictionary['sigma']['vertical'][f'Collocated_total_{finalkys[idata]}_IS'][HSRL_wvl[key]] = IS_alt_data[finalkys2[idata]][:,:,-1,key,1]
                output_dictionary['data']['vertical'][f'Collocated_total_{finalkys[idata]}_HSRL'][HSRL_wvl[key]] = HSRL_alt_data[finalkys2[idata]][:,:,key,0]
                output_dictionary['sigma']['vertical'][f'Collocated_total_{finalkys[idata]}_HSRL'][HSRL_wvl[key]] = HSRL_alt_data[finalkys2[idata]][:,:,key,1]
            if HSRL_wvl[key] != 1064:
                output_dictionary['data']['vertical']['Collocated_total_lr_legstats'][HSRL_wvl[key]]=altstats['lr'][HSRL_wvl[key]]
                output_dictionary['data']['vertical']['Collocated_total_lr_IS'][HSRL_wvl[key]] = IS_alt_data["lr"][:,:,-1,key,0]
                output_dictionary['sigma']['vertical']['Collocated_total_lr_IS'][HSRL_wvl[key]] = IS_alt_data["lr"][:,:,-1,key,1]
                output_dictionary['data']['vertical']['Collocated_total_lr_HSRL'][HSRL_wvl[key]] = HSRL_alt_data["lr"][:,:,key,0]
                output_dictionary['sigma']['vertical']['Collocated_total_lr_HSRL'][HSRL_wvl[key]] = HSRL_alt_data["lr"][:,:,key,1]
        output_dictionary['data']['vertical']['smoke_counts'] = smoke_counts
        output_dictionary['data']['vertical']['aircraft_altitude_m'] = IS_alt_data["alt"][:,:,0]
        #output_dictionary['data']['vertical']['aircraft_horizontal_separation_m'] = Collocated_horizontal_separation_alt
        return output_dictionary
