from netCDF4 import Dataset    # Note: python is case-sensitive!
import numpy as np

def cf19(data_path, data, Dims, GlobParams):
    """Writes a netCDF data file that is CF1.9 compliant.
    
    This function creates a new netCDF file, sets up its dimensions, 
    applies global configuration metadata, and populates the file with 
    compressed variables and their associated variable-level attributes.

    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)

    Requirements
    ------------ 
    * ``numpy``
    * ``netCDF4``

    :param data_path: The filesystem path where the netCDF file should be created.
    :type data_path: str
    :param data: A nested dictionary containing data arrays, variable grouping maps, 
                 dimension associations, and variable-level attributes.
                 
                 Must contain the following keys:
                 
                 * **SourceFlag**: (dict) Maps variable keys to their group directory string.
                 * **Dims**: (dict) Maps variable keys to a tuple/list of dimension names.
                 * **VariableAttributes**: (dict) Nested dictionary mapping variable keys to another dictionary of attribute name-value pairs.
                 * **[variable_keys]**: (numpy.ndarray) The actual data arrays to be written.
    :type data: dict
    :param Dims: A dictionary mapping dimension names to their integer lengths.
                 Example: ``{"time": 24, "lat": 180, "lon": 360}``
    :type Dims: dict
    :param GlobParams: Global metadata attributes to apply directly to the netCDF dataset root.
                       Must include a ``'format'`` key specifying the netCDF variant 
                       (e.g., ``'NETCDF4'``, ``'NETCDF4_CLASSIC'``).
    :type GlobParams: dict
    :return: None
    :rtype: NoneType
    :raises KeyError: If ``GlobParams['format']`` is missing, or if required keys 
                      are missing from the `data` sub-dictionaries.
    :raises IOError: If the file data_path cannot be opened for writing.

    .. note::
       * Floating-point data (``float64``) is downcasted to single-precision (``f4``).
       * 64-bit integer data (``int64``) is downcasted to standard integer (``i4``).
       * Deflate compression (``zlib=True``) is enabled by default at level 4.
       * Variable attributes defined as ``'unitless'`` are automatically replaced with ``'1'`` to adhere to CF conventions.
    """
    # FIX: Safely check if 'ncfile' exists in the namespace before calling close()
    if 'ncfile' in locals() or 'ncfile' in globals():
        try: 
            ncfile.close()  
        except: 
            pass

    ncfile = Dataset(data_path, mode='w', format=GlobParams['format']) 

    # Creating dimensions 
    for dim_key in Dims:    
        ncfile.createDimension(dim_key, Dims[dim_key])

    for dimensions in ncfile.dimensions.items():
        print(f"Dimension created: {dimensions}")  

    # Creating global attributes 
    ncfile.setncatts(GlobParams)

    for key, values in data.items():
        if isinstance(values, dict):
            for k_dicts in values:
                if k_dicts == 'GlobalAttributes':
                    ncfile.setncatts(values[k_dicts])
        elif isinstance(values, str):
            continue  # FIX: Skip structural meta-strings cleanly
        else:   
            if np.logical_not(key.startswith("dndlogdp_bin")):
                if np.logical_not(key.startswith("dp")):
                    
                    key2 = key.split('_')
                    if len(key2) > 1:
                        var_name = '_'.join(key2[0:-1])
                    else:
                        var_name = key
                        
                    var_dir = data['SourceFlag'].get(key, '') 
                    if var_dir == '':
                        var_dirname = var_name
                    else:
                        var_dirname = f'{var_dir}/{var_name}'
                    
                    # FIX: Always pull from data['Dims'] instead of using the variable name string. 
                    # Passing a string to createVariable splits each character into an explicit dimension!
                    shap_keys = data['Dims'].get(key, ())      
                    
                    # Allocate data type variants
                    if values.dtype == 'float64':
                        a = ncfile.createVariable(var_dirname, 'f4', shap_keys, zlib=True, complevel=4)
                    elif values.dtype == 'int64':
                        a = ncfile.createVariable(var_dirname, 'i4', shap_keys, zlib=True, complevel=4)
                    else:
                        a = ncfile.createVariable(var_dirname, values.dtype, shap_keys, zlib=True, complevel=4)       
                    
                    # Apply attributes
                    for key3 in data['VariableAttributes'].get(key, {}):
                        attr_val = data['VariableAttributes'][key][key3]
                        if attr_val == 'unitless':
                            attr_val = '1'                        
                        a.UnusedNameAttribute = attr_val       
                        a.renameAttribute("UnusedNameAttribute", key3)
                    
                    # FIX: Using ellipsis [...] handles any multi-dimensional shape 
                    # (1D, 2D, 3D, 4D+) cleanly without checking conditional lens.
                    a[...] = values

    print("\n--- Generated NetCDF File Structure ---")
    print(ncfile)
    ncfile.close()
    print('Dataset is closed cleanly!')
    
if __name__ == "__main__":
    # 1. Setup sample system dimensions
    mock_dims = {
        "time": 5,
        "lat": 4,
        "lon": 3
    }

    # 2. Setup structural file layout parameters
    mock_glob_params = {
        "format": "NETCDF4",
        "title": "Aerosol Measurement Campaign Data",
        "conventions": "CF-1.9",
        "history": "Created for functional integration testing."
    }

    # 3. Setup core variables and required metadata mapping hierarchies
    mock_data = {
        # Internal configuration elements tracking group targets
        "SourceFlag": {
            "temperature_kelvin": "met_vars",
            "relative_humidity": ""  # Written to root directory directly
        },
        # Internal mapping identifying dimension bounds
        "Dims": {
            "temperature_kelvin": ("time", "lat", "lon"),
            "relative_humidity": ("time",)
        },
        # Attribute configuration profiles
        "VariableAttributes": {
            "temperature_kelvin": {
                "long_name": "Ambient Air Temperature",
                "units": "K"
            },
            "relative_humidity": {
                "long_name": "Relative Humidity",
                "units": "unitless"  # Will automatically translate to "1"
            }
        },
        # Dummy matrix values matching specified shapes
        "temperature_kelvin": np.random.uniform(270.0, 310.0, size=(5, 4, 3)).astype('float64'),
        "relative_humidity": np.random.uniform(10.0, 95.0, size=(5,)).astype('float64')
    }

    # Run the execution wrapper
    print("Initializing netCDF Generation Routine...\n")
    cf19(path="test_output.nc", data=mock_data, Dims=mock_dims, GlobParams=mock_glob_params)
