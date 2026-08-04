import numpy as np

def Load(filename):
    """
    Returns an dictionary of bin diameters from a specified .csv file
    
    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)

    Requirements
    ------------ 
    * ``numpy``

    :param filename: String containing the name of the desired .csv file containing the size distribution bin diameters.
    :type filename: str  
    :return dp: dictionary of bin diameters in micrometer.
    :rtype: numpy dictionary
    """   

    i0 = 1  # index used to skip header row
    G = open(filename, 'r')  # open .csv
    g = G.read().splitlines()  # read .csv
    hdrs = g[0].split(',')  # define headers
    # create zeros array to be filled iteratively
    Dp = np.zeros((len(g) - 1, len(hdrs)))
    for i1 in range(len(g) - 1):
        # split string into array and define as number array
        Dp[i1, :] = np.array(list(eval(g[i0])))
        i0 += 1

    dp = {} 
    i0 = 0   
    for hdr in hdrs:
        dp[hdr] = np.array(Dp[:, i0]) 
        i0 += 1
    G.close()  # close .csv
    return dp