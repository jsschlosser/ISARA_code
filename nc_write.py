from netCDF4 import Dataset    # Note: python is case-sensitive!
import numpy as np
def cf19(path, data, Dims, GlobParams):


    #Opening a file, creating a new Dataset 

    try: ncfile.close()  # just to be safe, make sure dataset is not already open.
    except: pass
    ncfile = Dataset(path,mode='w',format=GlobParams['format']) 
    #print(ncfile)   

    #Creating dimensions 
    for dim_key in Dims:    
        a_dim = ncfile.createDimension(dim_key, Dims[dim_key])

    #time_dim = ncfile.createDimension('time', Dims['time'])     # latitude axis
    for dimensions in ncfile.dimensions.items():
        print(dimensions)  

    #Creating attributes 
    ncfile.setncatts(GlobParams)

    for key, values in data.items():
        if isinstance(values,dict):
            for k_dicts in values:
                #print(k_dicts)
                if k_dicts == 'GlobalAttributes':
                    ncfile.setncatts(values[k_dicts])
        elif isinstance(values,str):
            values
        else:   
            if np.logical_not(key.startswith("dndlogdp_bin")):
                if (np.logical_not(key.startswith("dp"))):
                    values_shape = np.array(values.shape)
                    key2 = key.split('_')
                    if len(key2)>1:
                        var_name = '_'.join(key2[0:-1])
                    else:
                        var_name = key
                    var_dir = data['SourceFlag'][key] 
                    if var_dir == '':
                        var_dirname = var_name
                        shap_keys = var_name
                    else:
                        var_dirname = f'{var_dir}/{var_name}'
                        shap_keys = data['Dims'][key]       
                    if values.dtype=='float64':
                        a = ncfile.createVariable(var_dirname, 'f4', shap_keys,zlib=True,complevel=4)
                    elif values.dtype=='int64':
                        a = ncfile.createVariable(var_dirname, 'i4', shap_keys,zlib=True,complevel=4)
                    else:
                        a = ncfile.createVariable(var_dirname, values.dtype, shap_keys,zlib=True,complevel=4)       
                    for key3 in data['VariableAttributes'][key]:
                        if data['VariableAttributes'][key][key3] == 'unitless':
                            data['VariableAttributes'][key][key3] = '1'                        
                        a.UnusedNameAttribute = data['VariableAttributes'][key][key3]       
                        a.renameAttribute("UnusedNameAttribute", key3)
                    if len(data[key].shape)==1:    
                        a[:] = values
                    elif len(data[key].shape)==2:
                        a[:,:] = values
                    elif len(data[key].shape)==3:    
                        a[:,:,:] = values
    # first print the Dataset object to see what we've got
    print(ncfile)
    # close the Dataset.
    ncfile.close(); print('Dataset is closed!')

if __name__ == "__main__":
    cf19()