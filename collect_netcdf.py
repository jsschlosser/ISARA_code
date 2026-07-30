########################################################################################################################
# collect_netcdf.py                                   by:  Joseph Schlosser
#                                                revised:  02 Jan 2023   
#                                    language (revision):  python3 (3.8.2-0ubuntu2)
# 
# DESCRIPTION: Procedures for reading the HSRL-2 and RSP data files from their *.h5 format into a python dictionary.
# 
# If grabRSP is called, the input is the file name of the RSP's .h5 file and the output is a python3 dictionary with the 
# data and metadata parameters from the RSP's .h5 file.
#   -> the dictionary structure follows that of the corresponding .h5 file
#
# If grabHSRL2 is called, the input is the file name of the HSRL-2's .h5 file and the output is a python3 dictionary 
# with the data and metadata parameters from the HSRL-2's .h5 file.
#   -> the dictionary structure follows that of the corresponding .h5 file
# 
# EXAMPLE:
#           HSRL2_Dictionary = collect_netcdf.grabHSRL2("ACTIVATE-HSRL2_UC12_20200215_R1") 
#           print(f"HSRL-2 metadata: {hsrl2_dictionary.keys()}")
#           HSRL-2 metadata: <KeysViewHDF5 ['000_Readme', 'DataProducts', 'Nav_Data', 'State', 'UserInput', 'header']>
#
# WARNINGS:
# 1) numpy, datetime, matplotlib, and h5py must be installed to the python environment
# 2) collect_netcdf.py and file with the corresponding filename must be present in a directory that is in your PATH
########################################################################################################################
import numpy as np
import datetime
import h5py

# function to retrieve RSP data from the appropriate .h5 file
def grabRSP(RSP_filename):
    # read the RSP data and display all avalable data products
    rsp_dictionary = h5py.File(RSP_filename, "r")
    #print(f"RSP data products: {rsp_dictionary.keys()}")   # remove comment out to see structure
    #print(np.array(rsp_dictionary["000-README"])) # remove comment out to see structure

    # take the flight date from the RSP readme line 9
    DATEinfo = np.array(RSP_filename.split("_"))[3] 
    #DATEinfo = np.array(RSP_filename.split("_"))[2] 
    #DATE = str(DATEinfo)[0:8]
    #print(DATE)
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

# function to retrieve HSRL-2 data from the appropriate .h5 file
def grabHSRL2(HSRL2_filename):
    # read the HSRL-2 data and display all groups and avalable data products
    hsrl2_dictionary = h5py.File(HSRL2_filename, "r")
    # remove below comment outs to see structure
    #    print(f"HSRL-2 metadata: {hsrl2_dictionary.keys()}") 
    #print(hsrl2_dictionary["DataProducts"].keys())
    #    print(f"HSRL-2 navigation and time data: {hsrl2_dictionary["Nav_Data"].keys()}")

    # take the flight date from the HSRL readme line 4
#    hsrl2_Date = [str(hsrl2_dictionary['000_Readme'][4])[2:6],str(hsrl2_dictionary['000_Readme'][4])[7:9],
#                    str(hsrl2_dictionary['000_Readme'][4])[10:12]]    
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