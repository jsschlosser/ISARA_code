"""
Procedures for reading the HSRL-2 and RSP data files from their .h5 format 
into a Python dictionary.

:Authors: Joseph Schlosser
:Revised: 4 Aug 2026
:Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)

.. note::
   The resulting dictionary structure strictly follows the hierarchy of the 
   corresponding source .h5 file.

Requirements
------------
* ``numpy``
* ``datetime``
* ``h5py``

.. warning::
   ``collect_netcdf.py`` and the target data files must be present in a 
   directory that is included in your system environment ``PATH``.
"""
import numpy as np
import datetime
import h5py


def grabRSP(RSP_filename):
    """
    Read RSP data from an HDF5 file into a Python dictionary.

    :param str filename: The file name or path of the RSP's .h5 file.
    :return: A dictionary containing data and metadata parameters.
    :rtype: dict
    """
    # read the RSP data and display all avalable data products
    rsp_dictionary = h5py.File(RSP_filename, "r")
    #print(f"RSP data products: {rsp_dictionary.keys()}")   # remove comment out to see structure

    # take the flight date from the RSP readme line 9
    splt_filename = np.array(RSP_filename.split("/"))
    DATEinfo = np.array(splt_filename[-1].split("_"))[3]
    rsp_Date = [str(DATEinfo)[0:4],str(DATEinfo)[4:6],str(DATEinfo)[6:8]]
#    rsp_Date = [str(rsp_dictionary["000-README"][9])[24:28],str(rsp_dictionary["000-README"][9])[28:30],
#                str(rsp_dictionary["000-README"][9])[30:32]]  
    #rsp_navg_scn = str(rsp_dictionary["000-README"][10])[42:44]
    rsp_navg_scn = 4.27
    rsp_time_array = np.array(rsp_dictionary["rsp_time"]) # select rsp time
    #print(len(rsp_time_array))
    SAMtime = np.zeros((len(rsp_time_array)))
    rsp_frmttimedata = ["" for x in range(len(rsp_time_array))] # create array of zeros for rsp datetime data
    rsp_mattimedata = dict() # create array of zeros for start datetime data
    # fill empty arrays formated datetime and matix date time
    for i1 in range(len(rsp_time_array)):   
        # convert DATE to separate integers
        Yr = int(rsp_Date[0])
        Mon = int(rsp_Date[1])
        Day = int(rsp_Date[2])
        # fractional hours after midnight to hour, minute, and second integers for rsp times
        Hr = int(np.floor(rsp_time_array[i1])) 
        Mnt = int(np.floor((rsp_time_array[i1]-np.floor(rsp_time_array[i1]))*60))
        Secd = int(((rsp_time_array[i1]-np.floor(rsp_time_array[i1]))*60-
        np.floor((rsp_time_array[i1]-np.floor(rsp_time_array[i1]))*60))*60) 
        dte = datetime.datetime(Yr,Mon,Day,Hr,Mnt,Secd)
        if (i1 > 0) & (rsp_time_array[i1] < rsp_time_array[i1-1]):
            dte = datetime.datetime(Yr,Mon,Day,0,0,0) + datetime.timedelta(days=1, hours=Hr, seconds=Secd, minutes=Mnt, 
                                                                        microseconds=0, milliseconds=0, weeks=0)
            SAMtime[i1] = rsp_time_array[i1] + rsp_time_array[i1-1] 
        else:
            dte = datetime.datetime(Yr,Mon,Day,Hr,Mnt,Secd)
            SAMtime[i1] = rsp_time_array[i1]

        rsp_mattimedata[i1] = dte.timetuple() # store the matrix of year, month, day, hour, minute, second 
        rsp_frmttimedata[i1] = dte # store the formatted datetime
    op = dict()
    for key in rsp_dictionary:
        op[key] = rsp_dictionary[key]
    op["rsp_Date"] = rsp_Date
    op["rsp_frmttimedata"] = rsp_frmttimedata
    op["rsp_time_array"] = SAMtime
    op["rsp_mattimedata"] = rsp_mattimedata
    op["rsp_navg_scn"] = rsp_navg_scn
    
    return op


def grabHSRL2(HSRL2_filename):
    """
    Read HSRL-2 data from an HDF5 file into a Python dictionary.

    :param str filename: The file name or path of the HSRL-2's .h5 file.
    :return: A dictionary containing data and metadata parameters.
    :rtype: dict

    :Example:

    .. code-block:: python

       import collect_netcdf

       # Load the data
       hsrl2_dict = collect_netcdf.grabHSRL2("ACTIVATE_HSRL2_UC12_20200215_R1")
       
       # Print metadata keys
       print(f"HSRL-2 metadata: {hsrl2_dict.keys()}")
       # Output: KeysViewHDF5 ['000_Readme', 'DataProducts', 'Nav_Data', 'State', 'UserInput', 'header']
    """

    # read the HSRL-2 data and display all groups and avalable data products
    hsrl2_dictionary = h5py.File(HSRL2_filename, "r")
    # remove below comment outs to see structure
    #    print(f"HSRL-2 metadata: {hsrl2_dictionary.keys()}") 
    #print(hsrl2_dictionary["DataProducts"].keys())
    #    print(f"HSRL-2 navigation and time data: {hsrl2_dictionary["Nav_Data"].keys()}")

    hsrldate = hsrl2_dictionary["header"]["date"][0].astype(int)
    hsrldate = hsrldate[0]
    hsrl2_Date = [str(hsrldate)[0:4], str(hsrldate)[4:6], str(hsrldate)[6:8]]
    
    hsrl2_time_array  = np.array(hsrl2_dictionary["Nav_Data"]["gps_time"])# select hsrl-2 time
    hsrl2_frmttimedata = np.full((len(hsrl2_time_array)),"NaT").astype("datetime64[s]") # create array of zeros for hsrl-2 datetime data
    hsrl2_mattimedata = dict() # create array of zeros for start datetime data 
    SAMtime = np.zeros((len(hsrl2_time_array)))
    # fill empty arrays formated datetime and matix date time
    for i1 in range(len(hsrl2_time_array)):   
        # convert DATE to separate integers
        Yr = int(hsrl2_Date[0])
        Mon = int(hsrl2_Date[1])
        Day = int(hsrl2_Date[2])
        #dte = datetime.date(Yr,Mon,Day)
                    
        # convert fractional hours after midnight to hour, minute, and second integers for hsrl-2 times
        Hr = int(np.floor(hsrl2_time_array[i1])) 
        Mnt = int(np.floor((hsrl2_time_array[i1]-np.floor(hsrl2_time_array[i1]))*60))
        Secd = int(((hsrl2_time_array[i1]-np.floor(hsrl2_time_array[i1]))*60-
        np.floor((hsrl2_time_array[i1]-np.floor(hsrl2_time_array[i1]))*60))*60) 

        

        if (i1 > 0) & (hsrl2_time_array[i1] < hsrl2_time_array[i1-1]):
            dte = datetime.datetime(Yr,Mon,Day,0,0,0) + datetime.timedelta(days=1, hours=Hr, seconds=Secd, minutes=Mnt, 
                                                                        microseconds=0, milliseconds=0, weeks=0)
            SAMtime[i1] = hsrl2_time_array[i1] + hsrl2_time_array[i1-1] 
        else:
            if Hr>23:
                dte = datetime.datetime(Yr,Mon,Day,0,0,0) + datetime.timedelta(days=1, hours=0, seconds=Secd, minutes=Mnt, 
                                                                        microseconds=0, milliseconds=0, weeks=0)
                SAMtime[i1] = hsrl2_time_array[i1]
            else:
                dte = datetime.datetime(Yr,Mon,Day,Hr,Mnt,Secd)
                SAMtime[i1] = hsrl2_time_array[i1]
                
        hsrl2_mattimedata[i1] = dte.timetuple() # store the matrix of year, month, day, hour, minute, second 
        hsrl2_frmttimedata[i1] = dte # store the formatted datetime
    op = dict()
    for key in hsrl2_dictionary:
        op[key] = hsrl2_dictionary[key]
    op["hsrl2_Date"] = hsrl2_Date
    op["hsrl2_frmttimedata"] = hsrl2_frmttimedata
    op["hsrl2_time_array"] = SAMtime
    op["hsrl2_mattimedata"] = hsrl2_mattimedata     
    return op