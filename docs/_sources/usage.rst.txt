Usage
=====
Acknowledgements
----------------

ISARA is developed in collaboration with NASA Langley Research Center, Hampton University, and the University of Arizona.


Copyright
---------

.. include:: ../LICENSE


Environmental Setup
-------------------

1) Install dependencies listed in requirements.txt. This includes SizeDistMerge from https://github.com/bochens/SizeDistMerge.

2) Create directory for code and data files.
	.. code-block:: console

		$ mkdir ISARA_Repo
		$ cd ISARA_Repo

3) Clone ISARA.

4) Create directory for data files that includes directories for the LUT data, desired shape distributions, and each mission of interest.
	.. code-block:: console

		$ mkdir ISARA_data_files
		$ cd ISARA_data_files
		$ mkdir LUT_data
		$ mkdir ShapeDistributions
		$ mkdir ACTIVATE

5) Create directories for HSRL, RSP, and in situ data. Create directories for size distributiution info and retrievals 
	.. code-block:: console

		$ mkdir AmbientDataFiles FalconLegID HSRL InternalConsistency MatchedData Retrievals SyntheticData ExternalClosure FitSDResults InsituData RSP SDBinInfo SyntheticRetrievals

6) Place data in correct files and proceed with internal closure.

Internal Closure
----------------

1) Perform ISARA with in-situ data:
	.. code-block:: console

		$ cd ./ISARA_Repo/ISARA_code
		$ python3 
		>>> import ISARA_Data_Retrieval
		>>> ISARA_Data_Retrieval.RunISARA()
		>>> ACTIVATE
		>>> 2
		>>> SMPS
		>>> 100
		>>> 0
		>>> LAS
		>>> 10000
		>>> 100
		>>> 3
		>>> 450
		>>> 470
		>>> Blue
		>>> 550
		>>> 532
		>>> Green
		>>> 700
		>>> 660
		>>> Red
		>>> 1
		>>> 550
		>>> Green
		>>> no
		>>> InsituData
		>>> AerosolLUT_1000_100_0.355_650bins_2325CRI_ln2rKr_Twomey.dat
		>>> exit()

2) Collate retrievals for internal consistency analysis:
	.. code-block:: console

		$ cd ./ISARA_Repo/ISARA_code
		$ rm ./ACTIVATE/Retrievals/activate-mrg-activate-large-smps_hu25_DataRetrievals.npy
		$ python3 
		>>> import CollateDataFiles
		>>> CollateDataFiles.Run()
		>>> ACTIVATE
		>>> -activate-large-smps
		>>> hu25
		>>> exit()

3) Perform internal consistency analysis:
	.. code-block:: console

		$ cd ./ISARA_Repo/ISARA_code
		$ python3 
		>>> import internal_closure_study
		>>> internal_closure_study.Run()
		>>> 450, 550, 700
		>>> 470, 532, 660
		>>> ACTIVATE
		>>> -activate-large-smps
		>>> hu25
		>>> RHw_DLH_DISKIN_
		>>> InletFlag_LARGE_ZIEMBA
		>>> gpsALT_m_THORNHILL
		>>> exit()
		$ cd ./ISARA_Repo/ISARA_code/ACTIVATE/Retrievals

Synthetic Closure
-----------------

1) Generate synthetic data:
	.. code-block:: console

		$ cd ./ISARA_Repo/ISARA_code
		$ python3 
		>>> import Synthetic_Data_Creation
		>>> Synthetic_Data_Creation.Run()
		>>> 10000
		>>> ACTIVATE
		>>> 3
		>>> 450
		>>> 470
		>>> Blue
		>>> 550
		>>> 532
		>>> Green
		>>> 700
		>>> 660
		>>> Red
		>>> 1
		>>> 550
		>>> Green
		>>> -activate-large-smps
		>>> hu25
		>>> exit()

2) Run ISARA on synthetic data:
	.. code-block:: console

		$ cd ./ISARA_Repo/ISARA_code
		$ python3 
		>>> import Synthetic_Data_Retrieval
		>>> Synthetic_Data_Retrieval.Run()
		>>> exit()

3) Synthetic data closure study:
	.. code-block:: console

		$ cd ./ISARA_Repo/ISARA_code
		$ python3 Synthetic_Data_Closure_Study.py


External Closure
----------------

1) Generate ambient data files assuming coarse mode is spherical sea salt:
	.. code-block:: console

		$ cd ./ISARA_Repo/ISARA_code
		$ mkdir ./ACTIVATE/AmbientDataFiles/
		$ mkdir ./ACTIVATE/AmbientDataFiles/Sphere_kappa0-cri1-33
		$ python3 
		>>> import GenAmbDataFiles
		>>> GenAmbDataFiles.Run()
		>>> ACTIVATE
		>>> 10.5067/ASDC/SUBORBITAL/ACTIVATE/Analysis/ISARA_1
		>>> -activate-large-smps
		>>> SMPS
		>>> AmbientDataFiles/Sphere_kappa0-cri1-33
		>>> hu25
		>>> 6
		>>> Added 'coarse_dndlogdp' and associated dimensions and lower/upper cutoff diameters to the 'derived' group; Added veff variable for all modes to 'derived' group; Corrected units of reff to be um. Extended SD truncation to 2 um for the retrieval step; Extended dry RRI search range to 1.51--1.55 from 1.52--1.54; Set coarse-mode non-absorbing portion to 0. Corrected calculation of f(RH) from ISARA measurements. Lowered gamma adjustment RH to 0 from 40. Spheres assumed for coarse-mode; Data produced with SIR SCA instead of MOPSMAP; Fixed long names of source variables to include id.
		>>> no
		>>> AerosolLUT_1000_100_0.355_650bins_2325CRI_ln2rKr_Twomey.dat
		>>> 0
		>>> no
		>>> 1.33
		>>> 0
		>>> 180
		>>> RHw_DLH_DISKIN_
		>>> yes
		>>> CAS,CDP,FCDP
		>>> 20
		>>> 1
		>>> no
		>>> exit()

2) Perform external consistency analysis assuming all particles are spheres
	.. code-block:: console

		$ cd ./ISARA_Repo/ISARA_code
		$ mkdir ./ACTIVATE/ExternalClosure/
		$ mkdir ./ACTIVATE/ExternalClosure/Sphere_kappa0-cri1-33
		$ python3 
		>>> import Collocate_and_Plot_Amb_Data_3shapes
		>>> Collocate_and_Plot_Amb_Data_3shapes.Run()
		>>> ACTIVATE
		>>> Sphere_kappa0-cri1-33
		>>> Sphere_kappa0-cri1-33
		>>> ExternalClosure/Sphere_kappa0-cri1-33
		>>> KingAir
		>>> Falcon
		>>> FalconLegID
		>>> exit()
