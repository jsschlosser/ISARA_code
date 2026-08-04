import numpy as np
import os

def Run():
	"""
	Takes a set of saved retrieval dictionaries and combines them given a set of user inputs.

    :Authors: Joseph Schlosser
    :Revised: 4 Aug 2026
    :Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)    

    Requirements
    ------------ 
    * ``numpy``
    * ``os``

	"""
	camp_name = input("Enter the campaign name in upper case (e.g., ARCSIX): ") 
	camp_name_lower = camp_name.lower()
	resolution = input("Enter the temporal resolution of interest in seconds (e.g., 30): ") 
	reference_platform = input("Enter the platform of interest (e.g., cirpas-to or MARINA-TOWER): ") 
	IFN = [f for f in os.listdir(f'./{camp_name}/Retrievals/') if (f.startswith(f'{camp_name_lower}-mrg{resolution}_{reference_platform}'))&(f.endswith('.npy'))]	
	#IFN = [f for f in os.listdir(r'./') if (f.startswith('pacepax-mrg30_MARINA-TOWER'))&(f.endswith('.npy'))]	
	#b = range(0,40,1)#np.array([39,126,169,170,171]).astype(int)#39,126,138,169,170,171#
	#IFN2 = [IFN[i] for i in b]
	OP_Dictionary = dict()
	for input_filename in IFN:
		print(input_filename)
		output_dict = np.load(f'./{camp_name}/Retrievals/{input_filename}',allow_pickle='TRUE')
		for key in output_dict.item():
			value = np.squeeze(output_dict.item().get(key)).reshape(-1,1)   
			if key in OP_Dictionary:
				OP_Dictionary[key] = np.vstack((OP_Dictionary[key], value))
			else:
				OP_Dictionary[key] = value
			
	np.save(f'./{camp_name}/Retrievals/{camp_name_lower}-mrg{resolution}_{reference_platform}_DataRetrievals.npy', OP_Dictionary) 
	#np.save('pacepax-mrg30_MARINA-TOWER_DataRetrievals.npy', OP_Dictionary) 

if __name__ == "__Run__":
    Run()	