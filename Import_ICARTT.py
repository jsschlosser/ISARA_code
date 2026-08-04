import numpy as np
import datetime

def imp(filename, num_time_columns): 
    """
    procedure to open ICARTT file and format the data and headers into the output (output_dictionary) python dictionary.WARNINGS: 1) numpy must be installed to the python environment, 2) importICART.py and file with the corresponding filename must be present in a directory that is in your PATH  
    
    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)    

    Requirements
    ------------ 
    * ``numpy``
    * ``datetime``

    :param filename: String  containing the name of the desired ICARTT v2.0 (.ict) file.
    :type filename: str  
    :return: the output of this code is output_dictionary, which is a python3 dictionary containing column-arrays for each of the parameters in the .ict file
    :rtype: numpy dictionary    
    
    Example
    -------

    .. code-block:: console

        >>> output_dictionary = importICT.imp("activate-mrg1_hu25_20200214_R0.ict")
        >>> print(output_dictionary)
        output_dictionary =
            {'Time_Start_seconds': array([61301., 61302., 61303., ..., 72258., 72259., 72260.]), 
            'Time_Stop_seconds': array([61302., 61303., 61304., ..., 72259., 72260., 72261.]), 
            'Latitude_THORNHILL_ deg': array([37.085528, 37.085798, 37.086065, ..., 37.126424, 37.126694, ...            
    """
    # creat empty dictionary
    output_dictionary = {}
    output_dictionary['VariableAttributes']={}
    output_dictionary['GlobalAttributes']={}
    
    G = open(filename, 'r') # open .ict file
    g = G.readlines() # read .ict file
    DATEinfo = np.array(g[6].split(",")) 
    DATE = DATEinfo[0:3] # save date to add to global attribute
    output_dictionary['GlobalAttributes']['Date'] = DATE
    measurement_platform = g[3] # save measurement platform to add to global attribute
    output_dictionary['GlobalAttributes']['measurement_platform'] = measurement_platform
    

    Fv = g[11]  # locate line with fill values, wich are located on line 11 of .ict file, to replace with 'nan'

    if Fv.__contains__(";"):
        fva = np.array(Fv.split(";"))
        Fv = fva[0]
    fv = np.array(Fv.split(",")).astype(float) # create array of fill values
    varend = int(g[9]) # locate line with number of variables, which is located on line 9 of .ict file

    full_var_titles = ["" for x in range(np.add(varend,1))] # create empty string array for full variable titles

    # fill full variable title string array with full variable titles starting on line 12 of .ict file
    i2 = 12
    for i1 in range(varend):
        full_var_titles[np.add(i1,1)] = g[i2]
        i2 = np.add(i2,1)

    # fill first variable title string array position with full variable title of first variable on line 8 of .ict file    
    starttime = g[8]
    full_var_titles[0] = starttime[0:len(starttime)-1]    
    
    # locate data header and data start rows of .ict file using line 0 of .ict file
    st = g[0]
    Vr_id = np.array(st.split(","))
    vr_id = int(Vr_id[0])

    rawdata = g[vr_id::] # create raw data array starting at data start row

    # create empty string arrays for final variable names, units, and extra info
    var_shortnames = ["" for x in range(len(full_var_titles))]
    var_units = ["" for x in range(len(full_var_titles))]
    var_macienames = ["" for x in range(len(full_var_titles))]
    var_longnames = ["" for x in range(len(full_var_titles))]

    # iteratively fill string arrays with final variable names, units, and extra info
    for i1 in np.arange(0,len(full_var_titles)).reshape(-1):
        fvt = full_var_titles[i1]
        FVT = np.array(fvt.split(","))
        var_shortnames[i1] = FVT[0].strip()
        var_units[i1] = FVT[1].strip()
        if len(FVT)>2:
            var_macienames[i1] = FVT[2].strip()
        else:
            var_macienames[i1] = 'none'   
        if len(FVT)>3:
            var_longnames[i1] = ",".join(FVT[3:])
        else:
            var_longnames[i1] = 'none'       

    # create data arraw with length of dataset and width of number of variables       
    data = np.zeros((len(rawdata),len(full_var_titles)))

    # iteratively fill data array with data and replace fv values with NAN
    for i1 in np.arange(0,len(rawdata)).reshape(-1):
        processdata1 = rawdata[i1]
        processdata2 = np.array(processdata1.split(",")).astype(float)
        for i2 in np.arange(1,len(processdata2)):
            if not processdata2[i2]:
                processdata2[i2] = np.nan
            elif processdata2[i2] == fv[i2-1]:
                processdata2[i2] = np.nan
        data[i1,:] = processdata2

    # fill dictionary with data and keys   
    for i1 in range(len(full_var_titles)-num_time_columns):
        dict_name = "%s_%s"%(var_shortnames[i1+num_time_columns],var_units[i1+num_time_columns])
        output_dictionary[dict_name] = data[:,i1+num_time_columns]
        output_dictionary['VariableAttributes'][dict_name] = {}
        output_dictionary['VariableAttributes'][dict_name]['short_name'] = var_shortnames[i1+num_time_columns]
        output_dictionary['VariableAttributes'][dict_name]['units'] = var_units[i1+num_time_columns]
        output_dictionary['VariableAttributes'][dict_name]['long_name'] = var_longnames[i1+num_time_columns]
        output_dictionary['VariableAttributes'][dict_name]['ACVSNC_standard_name'] = var_macienames[i1+num_time_columns]
        output_dictionary['VariableAttributes'][dict_name]['_FillValue'] = np.nan

    #output_dictionary["deployement"] = str(g[4]) # Add date to dictionary      
    #output_dictionary["date"] = np.array(DATE).astype(str) # Add date to dictionary  
    dta = data[:,range(num_time_columns)] # fill dictionary with date 
    mattimedata = dict()# create array of zeros for datetime data
    SAMtime = np.full((len(dta[:,0]),len(dta[0,:])),np.nan)
    # fill empty arrays formated datetime and matix date time
    frmttimedata = np.full((len(dta[:,0]),len(dta[0,:])),"NaT").astype('datetime64[s]')
    
    for i1 in range(len(dta[:,0])):
        mattimedata[i1] = dict()
        for i2 in range(num_time_columns):
            if np.isnan(dta[i1,i2]):
                mattimedata[i1][i2] = np.nan
                SAMtime[i1,i2] = np.nan
            else:   
                Hrs = int(np.floor(dta[i1,i2]/(60*60)))
                Mnts = int(np.floor((dta[i1,i2]/(60*60)-np.floor(dta[i1,i2]/(60*60)))*60))
                Secd = int(((dta[i1,i2]/(60*60)-np.floor(dta[i1,i2]/(60*60)))*60-
                                np.floor((dta[i1,i2]/(60*60)-np.floor(dta[i1,i2]/(60*60)))*60))*60)
                Yr = int(DATE[0])
                Mon = int(DATE[1])
                Day = int(DATE[2])
               
                if (i1 > 0) & (dta[i1,i2] < dta[i1-1,i2]):
                    dte = datetime.datetime(Yr,Mon,Day,0,0,0) + datetime.timedelta(days=1, hours=Hrs, seconds=Secd, minutes=Mnts, 
                                                                                microseconds=0, milliseconds=0, weeks=0)
                    SAMtime[i1,i2] = dta[i1,i2] + dta[i1-1,i2] 
                elif Hrs>=24:
                    dte = datetime.datetime(Yr,Mon,Day,0,0,0) + datetime.timedelta(days=1, hours=0, seconds=Secd, minutes=Mnts, 
                                                                                microseconds=0, milliseconds=0, weeks=0)   
                    SAMtime[i1,i2] = dta[i1,i2]                                                                             
                else:
                    dte = datetime.datetime(Yr,Mon,Day,Hrs,Mnts,Secd)
                    SAMtime[i1,i2] = dta[i1,i2]

                mattimedata[i1][i2] = dte.timetuple()
                frmttimedata[i1,i2] = dte

    # Add frmttimedata and mattimedata to dictionary   
    if num_time_columns == 1:
        output_dictionary['Time_Start_Seconds'] = SAMtime        
        output_dictionary['VariableAttributes']['Time_Start_Seconds'] = {}
        output_dictionary['VariableAttributes']['Time_Start_Seconds']['short_name'] = var_shortnames[0]
        output_dictionary['VariableAttributes']['Time_Start_Seconds']['units'] = var_units[0]
        output_dictionary['VariableAttributes']['Time_Start_Seconds']['long_name'] = var_longnames[0] 
        output_dictionary['VariableAttributes']['Time_Start_Seconds']['_FillValue'] = np.nan
        output_dictionary["datetime_Start_UTC"] = frmttimedata 
        output_dictionary['VariableAttributes']['datetime_Start_UTC'] = {}
        output_dictionary['VariableAttributes']['datetime_Start_UTC']['short_name'] = 'datetime_Start'
        output_dictionary['VariableAttributes']['datetime_Start_UTC']['units'] = 'UTC'
        output_dictionary['VariableAttributes']['datetime_Start_UTC']['long_name'] = 'Datetime stamp of corresponding to sample time.'   
        output_dictionary['VariableAttributes']['datetime_Start_UTC']['_FillValue'] = "NaT"
    else:
        output_dictionary['Time_Start_Seconds'] = SAMtime[:,0]
        output_dictionary['VariableAttributes']['Time_Start_Seconds'] = {}
        output_dictionary['VariableAttributes']['Time_Start_Seconds']['short_name'] = var_shortnames[0]
        output_dictionary['VariableAttributes']['Time_Start_Seconds']['units'] = var_units[0]
        output_dictionary['VariableAttributes']['Time_Start_Seconds']['long_name'] = var_longnames[0] 
        output_dictionary['VariableAttributes']['Time_Start_Seconds']['_FillValue'] = np.nan      
        output_dictionary['Time_Stop_Seconds'] = SAMtime[:,1]
        output_dictionary['VariableAttributes']['Time_Stop_Seconds'] = {}
        output_dictionary['VariableAttributes']['Time_Stop_Seconds']['short_name'] = var_shortnames[1]
        output_dictionary['VariableAttributes']['Time_Stop_Seconds']['units'] = var_units[1]
        output_dictionary['VariableAttributes']['Time_Stop_Seconds']['long_name'] = var_longnames[1]
        output_dictionary['VariableAttributes']['Time_Stop_Seconds']['_FillValue'] = np.nan        
        output_dictionary["datetime_Start_UTC"] = frmttimedata[:,0]  
        output_dictionary['VariableAttributes']['datetime_Start_UTC'] = {}
        output_dictionary['VariableAttributes']['datetime_Start_UTC']['short_name'] = 'datetime_Start'
        output_dictionary['VariableAttributes']['datetime_Start_UTC']['units'] = 'UTC'
        output_dictionary['VariableAttributes']['datetime_Start_UTC']['long_name'] = 'Datetime stamp of corresponding to sample start time.'
        output_dictionary['VariableAttributes']['datetime_Start_UTC']['_FillValue'] = "NaT"       
        output_dictionary["datetime_Stop_UTC"] = frmttimedata[:,1]  
        output_dictionary['VariableAttributes']['datetime_Stop_UTC'] = {}
        output_dictionary['VariableAttributes']['datetime_Stop_UTC']['short_name'] = 'datetime_Stop'
        output_dictionary['VariableAttributes']['datetime_Stop_UTC']['units'] = 'UTC'
        output_dictionary['VariableAttributes']['datetime_Stop_UTC']['long_name'] = 'Datetime stamp of corresponding to sample stop time.'
        output_dictionary['VariableAttributes']['datetime_Stop_UTC']['_FillValue'] = "NaT"
         
    G.close() # close data file      
    return output_dictionary
    ##
    return output_dictionary