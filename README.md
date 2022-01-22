# simstack3
Welcome to SIMSTACK3, a simultaneous stacking code, now compatible with python 3.

For literature describing how SIMSTACK works see [Viero et al. 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...779...32V/abstract).
<!---
If you've arrived via the newly published Viero & Moncelsi 2022, welcome, all code to reproduce the results is contained in this repository.  
Find data products (e.g., maps, catalog, cosmos2020.ini) [here](https://sites.astro.caltech.edu/viero/simstack/cosmos/).
--->
Improvements include the addition of a background layer, and masking, following [Duivenvoorden et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.1355D/abstract).

This stacking algorithm is separated into two distinct parts:
1. Performing the stack and saving the results (can take a long time, so better to number of times you do this!)  
2. Importing the saved results and plotting them.  

This has been tested on Windows and Mac. I've gone to great pains to make this simple to use.  Reach out if you encounter problems.  

## Setup
### Environment Variables
First step is to set the environment variables for maps, catalogs, and pickles, which tells the code where to find the data.  
See instructions for [Mac/Linux](https://phoenixnap.com/kb/set-environment-variable-mac), 
or [Windows](https://phoenixnap.com/kb/windows-set-environment-variable). You will only ever need to do this once. 

>export MAPSPATH=$MAPSPATH/Users/marcoviero/data/Astronomy/maps/

>export CATSPATH=$CATSPATH/Users/marcoviero/data/Astronomy/catalogs/

>export PICKLESPATH=$PICKLESPATH/Users/marcoviero/data/Astronomy/pickles/

### Virtual Python Environment

Setup a virtual environment with conda.  This ensures that you are using a clean, up-to-date version of python3.
This is easy to do in the [Anaconda Navigator GUI](https://medium.com/cluj-school-of-ai/python-environments-management-in-anaconda-navigator-ad2f0741eba7).  Alternatively from a terminal type:
> conda create -n simstack python=3.9

And then activate the environment.  This terminal is where you will run the python code (or open a Jupyter window.)
> conda activate simstack

Do this second step every time you open a new terminal. 

### Required Packages
Within the simstack environment (i.e., after **conda activate simstack**), install:
- matplotlib (> conda install matplotlib)
- seaborn (> conda install seaborn)
- numpy (> conda install numpy)
- pandas (> conda install pandas)
- astropy (> conda install astropy)
- lmfit (> conda install -c conda-forge lmfit)
- jupyterlab, if you want to use notebooks

## Usage
### The Configuration File
The code centers around the configuration file, config.ini (or whatever name you like.)  If simstack fails this is probably the first place to look.  

We use uvista.ini as an example of the format. The configuration file has the following sections:
#### general
> binning = {"stack_all_z_at_once":"True"}
- stack_all_z_at_once: True to stack all redshifts together.  Optimal, but also requires a lot of memory.  Alternative is stacking in redshift slices.

> error_estimator = {"bootstrap": "False", "emcee"="False"}
- bootstrap: Errors derived via. bootstrap (not working yet)
- emcee: Errors derived via. MCMC (not working yet)

> cosmology = Planck18
- Options are: Planck15, Planck18

#### io
> output_folder = PICKLESPATH simstack stacked_flux_densities
- output_folder: PICKLESPATH is a pre-defined environment variable; simstack/stacked_flux_densities will be created if does not exist already.
> shortname = uVista_DR2
- shortname: Name given to the folder where results are written.  If already exists, will create a new folder folllowed with understore, e.g., simstack/stacked_flux_densities/uVista_DR2_

#### catalog
> path = CATSPATH uVista
- path: Similar to above, where CATSPATH is a pre-defined environment variable, and uVista is a directory in CATSPATH.

> file = UVISTA_DR2_master_v2.1_USE.csv
- file: Catalog filename.

> astrometry = {"ra":"ra", "dec":"dec"}
- astrometry: Dictionary containing labels for "ra" and "dec".  Other catalog examples include {"ra":"ALPHA_J2000", "dec":"DELTA_J2000"}, and  

> classification = {"split_type":"uvj", "redshift":{"id":"z_peak", "bins":"[0.01, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]"}, "stellar_mass":{"id":"lmass", "bins":"[8.5, 9.5, 10.0, 10.5, 11.0, 12.0]"}, "split_params":{"id":"sfg", "bins":{"U-V":"rf_U_V", "V-J":"rf_V_J"}} }
- classification specifies how catalog is split into bins.  
  - split_type: options are "labels", "uvj", "nuvrj"
    - labels: Generic.  Will split into bins as specified in dicts, where catalog column is "id", e.g., "stellar_mass":{"id":"lmass", "bins":"[8.5, 9.5, 10.0, 10.5, 11.0, 12.0]"}
    - uvj: Specific for splitting sample into quiescent and star-forming using UVJ criteria.  Requires dict naming the U-V and V-J columns of catalog, e.g.  "split_params":{"id":"sfg", "bins":{"U-V":"rf_U_V", "V-J":"rf_V_J"}}
    - nuvrj: Specific for splitting sample into quiescent and star-forming using NUVRJ criteria.  Requires dict naming the NUV-R and R-J columns of catalog, e.g.  "split_params":{"id":"sfg", "bins":{"U-V":"rf_U_V", "V-J":"rf_V_J"}}
    
#### maps
> mips_24 = {"wavelength": 24.0, "beam":{"fwhm":6.32,"area":1.550e-09}, "color_correction":1.25, "path_map": "MAPSPATH /cosmos/cutouts/mips_24_GO3_sci_10.cutout.fits", "path_noise":"MAPSPATH /cosmos/cutouts/mips_24_GO3_unc_10.cutout.fits"}
- Each line defines a new map object.  To prevent from stacking (testing, e.g.,) comment out using ;
  - wavelength: float in microns.
  - beam: 
    - fwhm: in arcsec
    - area: if map is already in Jy/beam, this is 1.  If it is in MJy/steradian of MJy/pixel, then this is the solid angle of beam/pixel.   
  - color_correction: 1.0 unless known (see [Viero et al. 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...779...32V/abstract) for SPIRE values).  
  - path_map: Similar to above, where MAPSPATH is a pre-defined environment variable, and following is a directory in MAPSPATH.
  - path_noise: Similar to path_map.  If the signal and noise are in seperate layers in the the same, then this is identical to path_map.
    
### Stacking with SIMSTACK
You can run the code in two ways, 
- from the command line (i.e., in a terminal window), or 
- in a Jupyter Notebook. 

#### From a Terminal Window
To run from command line (equivalently directly in pyCharm):
- First make the *run_simstack_cmd_line.py* file executable (only needed once), e.g.:
> chmod +x run_simstack_cmd_line.py
- Run by calling the script followed by the config.ini file:
> python run_simstack_cmd_line.py config/uvista.ini

If you fail to add the config.ini file it will default to the one hardcoded in run_simstack_cmd_line (line 72).  
Change that to your config file, if you prefer it that way.  

Returned object contains:
- simstack_object.config_dict; dict_keys(['general', 'io', 'catalog', 'maps', 'cosmology_dict', 'distance_bins', 'parameter_names', 'pickles_path'])
- simstack_object.catalog_dict; dict_keys(['tables'])
- simstack_object.maps_dict; dict_keys(['spire_psw', 'spire_pmw', ...])
- simstack_object.results_dict; dict_keys(['maps_dict', 'SED_df']) 

#### From a Jupyter Notebook

See example_perform_simstack.ipynb.  Steps are 
1. Import Classes
    > from simstackwrapper import SimstackWrapper
   > 
    > from simstackresults import SimstackResults
2. Call instance of SimstackWrapper with read_maps=True, read_catalog=True, stack_automatically=True
    > path_ini_file = os.path.join("examples","cosmos2020_highz.ini")
   > 
   >cosmos2020_highz = SimstackWrapper(path_ini_file, read_maps=True, read_catalog=True, stack_automatically=True)

And that's it.  Results will be saved in folder defined in config.ini file.

### Admiring your Results
Stacking was successful! Great, now what?  

Results are stored in a *pickle*, in the PICKLEsPATH/simstack/stacked_flux_densities/shortname (e.g., on my computer that would be D:\pickles\simstack\stacked_flux_densities\uVista_DR2_example\uVista_DR2_example.pkl)

Inside the pickle is an object containing raw stacked flux densities (in Jy) and errors, and all data used to estimate them, and are accessed via python dictionaries:
- simstack_object.config_dict
- simstack_object.maps_dict
- simstack_object.catalog_dict
- simstack_object.results_dict

However, initially *results_dict* is empty; to populate the results_dict requires 
1. from simstackresults import SimstackResults
2. simstack_object = import_saved_pickles(path_file)
3. simstack_object.results_dict

The reason for this extra step is because the **SimstackResults** Class, which plots fluxes and estimates values like SEDs and LIRs, will be constantly improving, while the original stacked fluxes never change so should only need to be estimated once.  

#### Inside the results_dict
Like a Russian doll, there are many layers to the results dict.  The first layer contains
- maps_dict: which contains dictionaries for each wavelength stacked, each of which contains
  - results_df: this is where the fluxes and errors are stored for "easy" access. 
    -flux_df: depending on number of ways catalog is split (i.e., just stellar mass and redshift, or star-forming/quiescent too?)
      -split_params: keys are stellar_mass labels
        -stellar_mass_label: pandas DataFrame where index is redshift and column is flux
    -error_df - same structure as flux_df
Other layers inside maps_dict are the raw format of the stacked fluxes, not as useful as the results_df, but summarized here:
  - wavelength: the wavelength in microns
  - redshift: key is redshift labels, values are:
    - flux_density: contains redshift keys
      - redshift_key: contains fluxes
      - population_key: contains fluxes
    - std_error
      - redshift_key: contains errors
      - population_key: contains errors
  - stellar_mass: key is stellar mass labels
    - flux_density
    - std_error
    - redshift
- SED_df (will only be calculated is stacking was more than one wavelength, and ideally many more than two)
  - SED: contains keys of redshift labels (e.g., 'redshift_0.01_0.5')
    - contains keys of split_param labels (e.g., 'split_params_0')
      - contains keys of stellar_mass labels (e.g., 'stellar_mass_9.5_10.0')
        - contains Parameter objects which are the result of fitting fast_sed (located in SimstackToolbox)
  - LIR
    - contains keys of split_param labels (e.g., 'split_params_0')
      - contains keys of stellar_mass labels (e.g., 'stellar_mass_9.5_10.0')
        - contains infrared luminosity estimates (L_sun estimated from 8-1000um) 
  - wavelengths: contains list of wavelengths corresponding to SED
  - flux_density: contains keys of redshift labels (e.g., 'redshift_0.01_0.5')
    - contains keys of split_param labels (e.g., 'split_params_0')
      - contains pandas DataFrames where index is redshift and columns are flux densities organized by stellar_mass labels (e.g., 'stellar_mass_9.5_10.0')
  - std_error: contains keys of redshift labels (e.g., 'redshift_0.01_0.5')
    - contains keys of split_param labels (e.g., 'split_params_0')
      - contains pandas DataFrames where index is redshift and columns are flux densities organized by stellar_mass labels (e.g., 'stellar_mass_9.5_10.0')
    
So for example, if you want flux densities of your stack at redshift 0.5-1.0 and star-forming galaxies are split_params_1, look at the pandas table stored in

simstack_object.results_dict['SED_df']['flux_density']['redshift_0.5_1.0']['split_params_1']

#### Visualizing results

Existing plots so far are:
simstack_object.plot_flux_densities()
simstack_object.plot_seds()
simstack_object.plot_lir_vs_z()

See example_plot_results.ipynb for importing stacking pickles, parsing results, and plotting them. 
