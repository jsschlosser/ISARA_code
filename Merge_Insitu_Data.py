import Import_ICARTT
import numpy as np
import os
import sys
import datetime

def Merge():
	"""Merges multiple ICARTT (.ict) airborne/field instrument files into a uniform time grid.
	
	This function uses interactive terminal inputs to locate instrument subdirectories, 
	parse ICARTT telemetry files chronologically using the ``importICARTT`` module, 
	and downsample/average disparate parameters to a uniform user-defined temporal step.
	The final product is exported as an aggregated NumPy dictionary object.

	:Authors: Joseph Schlosser
	:Revised: 4 Aug 2026
	:Language Revision: Python 3.12.13 (Ubuntu 26.04 LTS)

	Requirements
	------------ 
	* ``numpy``
	* ``os``
	* ``sys``
	* ``datetime``

	.. note::
	   * Input source files must reside in structured paths relative to execution: 
		 ``./[Data_ID]/filename.ict``
	   * Datetime strings containing the token substring ``'fmtdatetime'`` are handled 
		 separately by downcasting to 64-bit integer views (``i8``) before averaging.
	   * Outputs are cleanly dumped to current working directory as 
		 ``pacepax-mrg[Resolution]_MARINA-TOWER_[Date]_RA.npy``.

	:Interactive Inputs:
		* **Data IDs**: A comma-separated prompt string mapping subdirectories to parse 
		  (e.g., ``"APS, UHSAS, MICROPHYSICAL, OPTICAL"``).
		* **Temporal Resolution**: An integer window duration mapped in seconds 
		  used for bin-averaging data fields.

	:raises FileNotFoundError: If an instrument data directory specified in the terminal 
							  prompt does not exist on disk.

	"""

	# Prompt user for instrument identifiers to establish merge targets
	Data_List_Str = input("Enter the list of .ict data IDs to be merged\nseparated by a comma and a space (e.g., APS, UHSAS, MICROPHYSICAL, OPTICAL): ")   
	Data_List = np.array(Data_List_Str.split(", ")).astype(str)	
	
	# Prompt user for time-bin delta tracking structures
	temporal_resolution = int(input("Enter the desired temporal resolution in seconds (e.g., 30): "))
	separated_data = {}
	fileID_list = {}
	
	# Phase 1: Ingest directory hierarchies and match flight tracking dates
	for DN in Data_List:
		separated_data[DN] = {}
		IFN = [f for f in os.listdir(f'./{DN}/') if f.endswith('.ict')] 

		# CRITICAL BUG FIX/FEATURE: Check if directory actually has data files to merge
		if len(IFN) == 0:
			raise FileNotFoundError(f"No valid '.ict' data files found within target path '{DN}'.")

		fileID_list[DN] = np.full(len(IFN), np.nan).astype(str)
		icount = 0
		print(DN)
		for f in IFN:
			filename = f'./{DN}/{f}'
			data = importICARTT.imp(filename, 1) 
			# Synthesize indexing strings out of flight headers (YearMonthDay format)
			DATE = f"{data['date'][0]}{data['date'][1]}{data['date'][2]}"
			fileID_list[DN][icount] = DATE
			icount += 1
			separated_data[DN][DATE] = data

	# Phase 2: Iterate across aligned observational periods to perform calculations
	for ID in fileID_list[Data_List[0]]:
		merged_data = {}
		for DN in Data_List:
			# Locate indices mapping directly to target calculation date
			FOI = np.squeeze(np.where(fileID_list[DN] == ID)[0])
			avgmergdat = {}
			sepdata = separated_data[DN][fileID_list[DN][FOI]]
			Tstart = sepdata['Time_Start_Seconds'] 
			Ldata = len(Tstart)
			print(Ldata, np.nanmin(Tstart), np.nanmax(Tstart), temporal_resolution)
			
			# Establish linear uniform timeline baseline matrix
			tgrd = np.arange(np.nanmin(Tstart), np.nanmax(Tstart), temporal_resolution)
			
			# Segment values dynamically inside window steps
			for t in np.arange(len(tgrd)):
				time_idx = np.where(((Tstart >= tgrd[t] - temporal_resolution) & (Tstart < tgrd[t] + temporal_resolution)))[0]
				for key in sepdata:
					if len(sepdata[key]) == Ldata:
						# Special Case: Cast time matrices down to integers to compute means
						if key.__contains__('fmtdatetime'):
							avgdata = np.squeeze(sepdata[key][time_idx]).view('i8').mean(axis=0).astype('datetime64[s]')
						else:
							avgdata = np.nanmean(sepdata[key][time_idx])
						
						# Dynamically append scalar aggregates into variable dictionaries
						if key in avgmergdat:
							avgmergdat[key][t] = avgdata
						else:
							if isinstance(avgdata, datetime.datetime):
								avgmergdat[key] = np.full(len(tgrd), "NaT").astype('datetime64[s]')
							else:
								avgmergdat[key] = np.full(len(tgrd), np.nan)
							avgmergdat[key][t] = avgdata
			
			# Extract computed channels cleanly into comprehensive collection frame
			for key in avgmergdat:
				if key not in merged_data:
					merged_data[key] = avgmergdat[key]

		# Phase 3: Export output configurations to local workspace files
		output_filename = f'pacepax-mrg{temporal_resolution}_MARINA-TOWER_{ID}_RA.npy'
		print(output_filename)
		np.save(output_filename, merged_data)
