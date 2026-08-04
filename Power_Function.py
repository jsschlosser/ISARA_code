import numpy as np
def f_model(x,a,c): 
	"""
	function y = a*x^c 
    
    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)    

    Requirements
    ------------ 
    * ``numpy``
    	
	:param x: base of power function
	:type x: double, float, int  
	:param a: scale of power function
	:type a: double, float, int
	:param c: exponent of power function
	:type c: double, float, int
	:return y: 
	:rtype: double, float, int
	"""   
	y = np.multiply(a,pow(x, c))
	return y