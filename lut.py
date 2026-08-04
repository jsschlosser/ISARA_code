import numpy as np
import struct
from time import time
import copy
'''
Functions for loading SIR SCA look-up table (LUT) and calculating aerosol properties given the loaded LUT.

:Authors: Joseph Schlosser
:Revised: 4 Aug 2026
:Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)

Requirements
------------ 
* ``numpy``
* ``struct``
* ``time``
* ``copy``
'''    

# =========
# CONSTANTS
# =========
# Smallest block size in the LUT
BLOCK = 4
# Well, I was too lazy to write 5 characters everytime instead of 2
PI = np.pi

# Variable names
ABS_COEF = 'absorption_coefficient'
ASYMMETRY = 'asymmetry'
BSC_COEF = 'backscattering_coefficient'
CROSS_SECTION = 'cross_section'
EXT_COEF = 'extinction_coefficient'
NUMB_CONCEN = 'number_concentration'
PM25 = 'pm25'
SCT_COEF = 'scattering_coefficient'
SSA = 'ssa'
LDR = 'ldr'

def initialize_spheres(LUT_location):
    """
    Load in the look up table (LUT) variables for use within `run_LUT`. All variables are returned in the output_variables dictionary

    :param LUT_location: Path to the LUT
    :type LUT_location: str                               
    :return: Dictionary (output_variables) with the variables from a given LUT file
    :rtype: numpy dictionary
    """ 
    output_variables = {} 
    start_load_lut_time = time()
    print(f'Loading the Mie LUT found at `{LUT_location}`.')
    with open(LUT_location, mode='rb') as file:
        def _read_lut(buffer_format, multiplier=1, num_of_bytes=BLOCK):
            result_tuple = struct.unpack(buffer_format * multiplier,
                                         file.read(num_of_bytes * multiplier))
            # If the multiplier is 1 then it is a single value. Since all
            # values are unpacked as tuples, return it without the tuple.
            if multiplier == 1:
                result_tuple = result_tuple[0]
            return result_tuple

        output_variables['wavelength'] = _read_lut('f')
        output_variables['num_radii_grid_bins'] = _read_lut('i')
        output_variables['radii_grid_bins'] = _read_lut('f', output_variables['num_radii_grid_bins'])
        output_variables['num_scattering_angles'] = _read_lut('i')
        output_variables['scattering_angles'] = _read_lut('f', output_variables['num_scattering_angles'])
        output_variables['num_real_parts'] = _read_lut('i')
        output_variables['real_parts'] = _read_lut('f', output_variables['num_real_parts'])
        output_variables['num_imag_parts'] = _read_lut('i')
        output_variables['imag_parts'] = _read_lut('f', output_variables['num_imag_parts'])
        num_CRI = output_variables['num_real_parts'] * output_variables['num_imag_parts']
        output_variables['Cext'] = np.zeros((num_CRI, output_variables['num_radii_grid_bins']))
        output_variables['Csca'] = np.zeros((num_CRI, output_variables['num_radii_grid_bins']))
        output_variables['C11'] = np.zeros((num_CRI, output_variables['num_radii_grid_bins'], output_variables['num_scattering_angles']))
        output_variables['C12'] = np.zeros((num_CRI, output_variables['num_radii_grid_bins'], output_variables['num_scattering_angles']))
        output_variables['C33'] = np.zeros((num_CRI, output_variables['num_radii_grid_bins'], output_variables['num_scattering_angles']))
        output_variables['C34'] = np.zeros((num_CRI, output_variables['num_radii_grid_bins'], output_variables['num_scattering_angles']))
        for i in range(0, num_CRI):
            # real_part_record and imag_part_record at i
            file.read(BLOCK * 2)
            output_variables['Cext'][i] = _read_lut('f', output_variables['num_radii_grid_bins'])
            output_variables['Csca'][i] = _read_lut('f', output_variables['num_radii_grid_bins'])
            for iGB in range(0, output_variables['num_radii_grid_bins']):
                output_variables['C11'][i][iGB] = _read_lut('f', output_variables['num_scattering_angles'])
            for iGB in range(0, output_variables['num_radii_grid_bins']):    
                output_variables['C12'][i][iGB] = _read_lut('f', output_variables['num_scattering_angles'])
            for iGB in range(0, output_variables['num_radii_grid_bins']):   
                output_variables['C33'][i][iGB] = _read_lut('f', output_variables['num_scattering_angles'])
            for iGB in range(0, output_variables['num_radii_grid_bins']):    
                output_variables['C34'][i][iGB] = _read_lut('f', output_variables['num_scattering_angles'])
            # Handle reading of C12, C33, and C34
            #file.read(BLOCK * 3 * output_variables['num_scattering_angles'] * output_variables['num_radii_grid_bins'])
    total_load_lut_time = time() - start_load_lut_time
    print(f'Finished loading the Mie LUT ({total_load_lut_time:0.2f}s).')
    return output_variables

def initialize_spheroids(LUT_location):
    """
    Load in the look up table (LUT) variables for use within `run_LUT`. All variables are returned in the output_variables dictionary

    :param LUT_location: Path to the LUT
    :type LUT_location: str                               
    :return: Dictionary (output_variables) with the variables from a given LUT file
    :rtype: numpy dictionary
    """ 
    output_variables = {} 
    start_load_lut_time = time()
    print(f'Loading the Mie LUT found at `{LUT_location}`.')
    with open(LUT_location, mode='rb') as file:
        def _read_lut(buffer_format, multiplier=1, num_of_bytes=BLOCK):
            result_tuple = struct.unpack(buffer_format * multiplier,
                                         file.read(num_of_bytes * multiplier))
            # If the multiplier is 1 then it is a single value. Since all
            # values are unpacked as tuples, return it without the tuple.
            if multiplier == 1:
                result_tuple = result_tuple[0]
            return result_tuple
        output_variables['wavelength'] = _read_lut('f')
        output_variables['nonSphericalRatioLUT'] = _read_lut('f')
        output_variables['num_radii_grid_bins'] = _read_lut('i')
        output_variables['radii_grid_bins'] = _read_lut('f', output_variables['num_radii_grid_bins'])
        output_variables['num_scattering_angles'] = _read_lut('i')
        output_variables['scattering_angles'] = _read_lut('f', output_variables['num_scattering_angles'])
        output_variables['num_real_parts'] = _read_lut('i')
        output_variables['real_parts'] = _read_lut('f', output_variables['num_real_parts'])
        output_variables['num_imag_parts'] = _read_lut('i')
        output_variables['imag_parts'] = _read_lut('f', output_variables['num_imag_parts'])
        num_CRI = output_variables['num_real_parts'] * output_variables['num_imag_parts']
        output_variables['Cext'] = np.zeros((num_CRI, output_variables['num_radii_grid_bins']))
        output_variables['Csca'] = np.zeros((num_CRI, output_variables['num_radii_grid_bins']))
        output_variables['C11'] = np.zeros((num_CRI, output_variables['num_radii_grid_bins'], output_variables['num_scattering_angles']))
        output_variables['C12']= np.zeros((num_CRI, output_variables['num_radii_grid_bins'], output_variables['num_scattering_angles']))
        output_variables['C22']= np.zeros((num_CRI, output_variables['num_radii_grid_bins'], output_variables['num_scattering_angles']))
        output_variables['C33']= np.zeros((num_CRI, output_variables['num_radii_grid_bins'], output_variables['num_scattering_angles']))
        output_variables['C34']= np.zeros((num_CRI, output_variables['num_radii_grid_bins'], output_variables['num_scattering_angles']))
        output_variables['C44']= np.zeros((num_CRI, output_variables['num_radii_grid_bins'], output_variables['num_scattering_angles']))
        for i in range(0, num_CRI):
            # real_part_record and imag_part_record at i
            # real_part_record and imag_part_record at i
            file.read(BLOCK * 2)
            output_variables['Cext'][i] = _read_lut('f', output_variables['num_radii_grid_bins'])
            output_variables['Csca'][i] = _read_lut('f', output_variables['num_radii_grid_bins'])
            for iGB in range(0, output_variables['num_radii_grid_bins']):
                output_variables['C11'][i][iGB] = _read_lut('f', output_variables['num_scattering_angles'])
            for iGB in range(0, output_variables['num_radii_grid_bins']):
                output_variables['C12'][i][iGB] = _read_lut('f', output_variables['num_scattering_angles'])
            for iGB in range(0, output_variables['num_radii_grid_bins']):
                output_variables['C22'][i][iGB] = _read_lut('f', output_variables['num_scattering_angles'])                
            for iGB in range(0, output_variables['num_radii_grid_bins']):
                output_variables['C33'][i][iGB] = _read_lut('f', output_variables['num_scattering_angles'])
            for iGB in range(0, output_variables['num_radii_grid_bins']):
                output_variables['C34'][i][iGB] = _read_lut('f', output_variables['num_scattering_angles'])
            for iGB in range(0, output_variables['num_radii_grid_bins']):
                output_variables['C44'][i][iGB] = _read_lut('f', output_variables['num_scattering_angles']) 
    total_load_lut_time = time() - start_load_lut_time
    print(f'Finished loading the Mie LUT ({total_load_lut_time:0.2f}s).')
    return output_variables

def run(
    wavelength,
    output_type,
    real_part,
    imag_part,
    dV_dlnr,
    T_output_variables,
    num_int_angles=None,
):
    """
    Calulates the integrated aerosol micriphysical properties for an array of data points. `load_LUT` must be called first to load the necessary output variables dictionary. 

    :param wavelength: Wavelength to run the LUT at in nanometers.
    :type wavelength: float   
    :param output_type: Set output_type to "extended" for cross section, integrated mass, assymetry, and backscattering coefficient.
    :type output_type: float       
    :param real_part: Real refractive index of the data points.
    :type real_part: numpy array[float]
    :param imag_part: Real refractive index of the data points.
    :type imag_part: numpy array[float]  
    :param dV_dlnr: lognormal volume particle size distributions (in um^3.cm^-3) of the data points (must be interpolated to the LUT radii bin grid).
    :type dV_dlnr: numpy array[float]               
    :param T_output_variables: Dictionary containing the LUT output variables from 'loat_LUT'
    :type T_output_variables: numpy dictionary   
    :return: Dictionary (T_output_variables) with the integrated aerosol microphysical properties
    :rtype: numpy dictionary           
    """ 
    def _find_idxs_and_coef(grid_size, grid, points, asymmetry_inc_mask=False):
        grid = np.array(grid) # Needs to be a NumPy array, a python tuple does not allow indexing
        grid_trimmed = np.array(grid[:-1]) # Two arrays of the same length where the second array is shifted to be one index off. This would take [1,2,3] and proudce [1,2] and [2,3]. Returns (trimmed, shifted), or, (i, i+1)
        grid_shifted = np.roll(grid, -1)[:-1]
        points_bc = points[:, None] # Create a copy of of the points for each of the grid, this is necessary for array broadcasting. The syntax below is equivalent to: np.repeat(points, grid_size - 1).reShape(-1, grid_size - 1)
        mask = (points_bc >= grid_trimmed) & (points_bc <= grid_shifted)# Create a mask of where the points are between the grid
        idxs = np.argmax(mask, axis=1)# The first index where the mask is true for each row
        idxs[points < grid[0]] = 0 # Set any points that fall outside the grid bounds to the boundary
        idxs[points > grid[grid_size - 1]] = grid_size - 2
        inc_mask = idxs < 0.5 * grid_size if asymmetry_inc_mask else idxs == 0# The mask by which to incrememnt certain indexes
        idxs[inc_mask] += 1
        idxs_m_1 = idxs - 1
        idxs_p_1 = idxs + 1
        idxs_arr = np.array([idxs_m_1, idxs, idxs_p_1])
        x1 = grid[idxs_m_1]
        x2 = grid[idxs]
        x3 = grid[idxs_p_1]
        coef = np.array([
            (points - x2) * (points - x3) / (x1 - x2) / (x1 - x3),
            (points - x1) * (points - x3) / (x2 - x1) / (x2 - x3),
            (points - x1) * (points - x2) / (x3 - x1) / (x3 - x2),
        ])
        return idxs_arr, coef, mask

    # Convert from nanometers to micrometers
    wavelength = wavelength * 0.001
    # Necessary to use the LUT at other wavelengths
    wavelength_ratio = T_output_variables['wavelength'] / wavelength

    # Find the radius coefs and mask
    _, radius_coefs, radius_mask = _find_idxs_and_coef(
        T_output_variables['num_radii_grid_bins'],
        T_output_variables['radii_grid_bins'],
        wavelength_ratio * np.array(T_output_variables['radii_grid_bins']),
    )
    all_false_rows = (~radius_mask).all(axis=1)
    radius_index_min = np.argmax(~all_false_rows)
    radius_coefs[:, all_false_rows] = 0

    # Find the real/imag coefs and indexes
    real_part_idxs, real_coefs, _ = _find_idxs_and_coef(
        T_output_variables['num_real_parts'],
        T_output_variables['real_parts'],
        real_part,
    )

    imag_part_idxs, imag_coefs, _ = _find_idxs_and_coef(
        T_output_variables['num_imag_parts'],
        T_output_variables['imag_parts'],
        imag_part,
    )
    radius_coefs_wlr = radius_coefs * wavelength_ratio

    record_idxs = T_output_variables['num_imag_parts'] * real_part_idxs[:, None] + imag_part_idxs # Calculate the radius coefs PSD
    coef_real_imag = real_coefs[:, None] * imag_coefs
    coef1 = radius_coefs_wlr[0] * dV_dlnr
    coef2 = radius_coefs_wlr[1] * dV_dlnr
    coef3 = radius_coefs_wlr[2] * dV_dlnr
    def _calc_LUT(T_arr):
        # These einsum strings work for both the EXT/SCA coefficients and the
        # P11 matrix. This means, these strings will return the corresponding
        # 1D and 2D arrays. These strings equate to:[
                #                  | EXT/SCA        | P11
                #     -------------+----------------+-----------------]
        #     einsum_str_1 | 'ijk,k->ijk'   | 'ijkl,k->ijkl'
        #     einsum_str_2 | 'ijkl,kl->ijk' | 'ijklm,kl->ijkm'
        #     einsum_str_3 | 'ijk,ijk->k'   | 'ijkl,ijk->kl'
        einsum_str_1 = 'ijk...,k->ijk...'
        einsum_str_2 = 'ijkl...,kl->ijk...'
        einsum_str_3 = 'ijk...,ijk->k...'

        def _einsum(func):
            return func(0, coef1) + func(1, coef2) + func(2, coef3)

        T_arr_chunk = T_arr[record_idxs]
        idx_min = radius_index_min
        base = _einsum(lambda n, coef: np.einsum(
            einsum_str_1,
            T_arr_chunk[:, :, :, n],
            coef[:, idx_min],
        ))
        idx_min += 1
        if idx_min == 1:
            base *= 2
            idx_min += 1
        num_radius = T_output_variables['num_radii_grid_bins'] - idx_min
        computed_LUT_arr = base + _einsum(lambda n, coef: np.einsum(
            einsum_str_2,
            T_arr_chunk[:, :, :, n:n + num_radius],
            coef[:, idx_min:],
        ))
        # Multiply the first three dimensions of `computed_LUT_arr` by
        # `coef_real_imag` and sum the resulting first two dimensions
        return np.einsum(einsum_str_3, computed_LUT_arr, coef_real_imag)

    # Call the LUT to find the extinction and scattering coefficients
    ext_coef = _calc_LUT(T_output_variables['Cext'])
    sca_coef = _calc_LUT(T_output_variables['Csca'])
    abs_coef = np.abs(ext_coef - sca_coef)
    ssa = sca_coef / ext_coef
    final_returns = {
        ABS_COEF: abs_coef,
        EXT_COEF: ext_coef,
        SCT_COEF: sca_coef,
        SSA: ssa,
    }
    if output_type =="extended":
        # The angles being used to calculate asymmetry
        if num_int_angles is None:
            num_int_angles = 2000
        ASYMMETRY_ANGLES = np.arange(num_int_angles) * 180 / (num_int_angles - 1)

        # Call the LUT to find the P11 array
        p11  = _calc_LUT(T_output_variables['C11'])
        if 'C22' in T_output_variables:
            p22  = _calc_LUT(T_output_variables['C22'])
            p11_180 = p11[:, -1]
            p22_180 = p22[:, -1]
            p11_180[p11_180==0]=np.nan
            p22_180[p22_180==0]=np.nan
            ldr_out = (p11_180-p22_180)/(p11_180+p22_180)
        else:
            ldr_out = 0 
        final_returns[BSC_COEF] = p11[:, -1] * 0.25 / PI
        final_returns[LDR] = ldr_out
        # sca_coef needs to be broadcasted in the second dimension for p11
        p11 /= sca_coef[:, None]
        angle_idxs, angle_coefs, _ = _find_idxs_and_coef(
            T_output_variables['num_scattering_angles'],
            T_output_variables['scattering_angles'],
            ASYMMETRY_ANGLES,
            True,
        )
        def _asy_calc(idx):
            return np.log(p11[:, angle_idxs[idx]]) * angle_coefs[idx]
        # Numerical integration using Simpson's rule.
        # For each point, ends up computing something like:
        #   np.exp(
        #       np.log(p11[angle_idx[0]]) * angle_coef[0] +
        #       np.log(p11[angle_idx[1]]) * angle_coef[1] +
        #       np.log(p11[angle_idx[2]]) * angle_coef[2] +
        #   ) + np.cos(angles * pi / 180) * np.sin(angles * pi / 180)
        #     * pi * (numb_angles - 1) / 3
        c = np.sin(ASYMMETRY_ANGLES * PI / 90) / 2
        asymmetry = np.exp(_asy_calc(0) + _asy_calc(1) + _asy_calc(2)) * c
        asymmetry[:, [0, -1]] *= 0.5
        asymmetry[:, ::2] *= 2
        asymmetry = np.sum(asymmetry, axis=1)
        asymmetry *= PI / (num_int_angles - 1) / 3
        final_returns[ASYMMETRY] = asymmetry
        final_returns['angle'] = ASYMMETRY_ANGLES
    return final_returns


