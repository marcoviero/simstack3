# simstack3

If you are here for **The Early Universe was Dust-Rich and Extremely Hot**, (Viero et al. 2022) the Python code you will need to reproduce the result is contained in this repository.  
Instructions in the form of Jupyter Notebooks, and links to the data products needed (e.g., maps, catalog, config file) can be found [here](https://github.com/marcoviero/simstack3/tree/main/viero2022).

Welcome to SIMSTACK3, a simultaneous stacking code, now compatible with python 3.  For literature describing how SIMSTACK works see [Viero et al. 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...779...32V/abstract). <br>
Improvements on the original simstack code include the addition of a background layer, and masking, following [Duivenvoorden et al. 2020](https://ui.adsabs.harvard.edu/abs/2020MNRAS.491.1355D/abstract).

This stacking algorithm is separated into two distinct parts:
1. Performing the stack and saving the results (this can take a long time, so better to limit the number of times you do this!)  
2. Importing the stacking results saving in step 1, analyzing, and plotting them.  

This has been tested on Windows and Mac. I've gone to great pains to make this simple to use.  Reach out if you encounter problems.  

## Setup
### Environment Variables
Setting up environment variables is a good idea, especially if you have more than one computer.  It hard-codes the data paths for specific machines so you don't have to change the code every time you switch machines.  
Create environment variables for maps, catalogs, and pickles, which tells the code where to find the data.  
See instructions for [Mac/Linux](https://phoenixnap.com/kb/set-environment-variable-mac), 
or [Windows](https://phoenixnap.com/kb/windows-set-environment-variable). You will only ever need to do this once. 

>export MAPSPATH=$MAPSPATH/Users/marcoviero/data/Astronomy/maps/

>export CATSPATH=$CATSPATH/Users/marcoviero/data/Astronomy/catalogs/

>export PICKLESPATH=$PICKLESPATH/Users/marcoviero/data/Astronomy/pickles/

### Virtual Python Environment

Setup a virtual environment with conda.  This ensures that you are using a clean, up-to-date version of python3.
This is easy to do in the [Anaconda Navigator GUI](https://medium.com/cluj-school-of-ai/python-environments-management-in-anaconda-navigator-ad2f0741eba7).  Alternatively from a terminal type:
> conda create -n simstack python=3.9

And then activate the environment
> conda activate simstack

This terminal is where you will run the python code (or open a Jupyter window.)  Do this second step every time you open a new terminal. 

### Required Packages
Within the simstack environment (i.e., after **conda activate simstack**), install:
- matplotlib (> conda install matplotlib)
- seaborn (> conda install seaborn)
- numpy (> conda install numpy)
- pandas (> conda install pandas)
- astropy (> conda install astropy)
- lmfit (> conda install -c conda-forge lmfit)
- sklearn (> conda install scikit-learn)
- emcee (> conda install emcee)  
- jupyterlab, if you want to use notebooks

## Usage
### The Configuration File
The code centers around the configuration file, config.ini (or whatever name you like.)  If simstack fails this is probably the first place to look.  **Note, in the configuration file 1 is True and 0 is False**.

We use uvista.ini as an example of the format. The configuration file has the following sections:
#### general
> binning = {"stack_all_z_at_once": 1, "add_background": 1, "crop_circles": 1}
- stack_all_z_at_once: (default True) True to stack all redshifts together.  Optimal, but also requires a lot of memory.  Alternative is stacking in redshift slices.
- add_background: (default True) Adds an additional layer, a background of all 1's, to the simultaneous stack.
- crop_circles: (default True) draw circles around each source in each layer and flatten, keeping only pixels in fit.
> error_estimator = {"bootstrap": {"initial_bootstrap": 1, "iterations": 150}, "write_simmaps": 0, "randomize": 0}
- bootstrap: Errors derived via. bootstrap method.  Layer parameters are named incrimentaly, beginning with "initial_bootstrap", which also happens to define the seed for the random shuffling, so that the bootstrap is identical from band to band. 
- write_simmaps: Write simulated image, e.g. each layer summed together, at each wavelength.
- randomize: for null-testing purposes, will implement np.shuffle(x,y) when building the layers.
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
    
### Error bars
The preferred method for generating error bars with SIMSTACK is via the "bootstrap".  
Unlike a traditional bootstrap, which draws from the original sample (with replacement, i.e., the same source can be drawn multiple times) 
SIMSTACK splits each bin in two, with a size ratio of 80:20, and stacks them simultaneously.  This is keeping with the 
motivation behind SIMSTACK in the first place, that correlated sources will bias the measurement, so all sources that generate signal must 
be stacked together.  
Note, this slows down the calculation considerably.  If you originally had 50 bins (e.g., 5 redshift, 5 stellar mass, 2 types), resulting in 
stacking 50 layers (51 if you include background), the bootstrap calculation is 100 bins (101 with background), which requires a lot of RAM!
If this is a problem you have two options:
1. Stack in Redshift Layers;
2. Split the sample into redshift chunks, e.g., bins z=[0,1,2] and z=[2,3,4] in each chunk, and combine the results afterward.  
See ["Merging Results"](https://github.com/marcoviero/simstack3/blob/main/notebooks/2_merge_and_save_pickles.ipynb) for instructions on how to do this.  

### Running SIMSTACK
You can run the code in two ways, 
- from the command line (i.e., in a terminal window), or 
- in a Jupyter Notebook. 

#### From a Terminal Window
To run from command line (equivalently directly in pyCharm):
- First make the *run_simstack_cmd_line.py* file executable (only needed once), e.g.:
> chmod +x run_simstack_cmd_line.py
- Run by calling the script followed by the config.ini file:
> python run_simstack_cmd_line.py config/uvista.ini

** If you are using example.ini and get the error "Catalog not found: ../test_data/catalogs/UVISTA_DR2_master_v2.1_USE.csv", change the path of the catalog and the 4 maps from ../ to ./ **

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

Results are stored in a *pickle*, in the PICKLESPATH/simstack/stacked_flux_densities/shortname (e.g., on my computer that would be D:\pickles\simstack\stacked_flux_densities\uVista_DR2_example\uVista_DR2_example.pkl)

These are raw results (i.e., initially *results_dict* is empty) and need to be passed into *SimstackResults*, in order to apply any of the cosmology estimators on them. E.g.;
- simstack_object = SimstackResults(simstack_object)

Similary, pass object into *SimstackPlots* in order to plot the estimated results, e.g.;
- simstack_object = SimstackPlots(simstack_object)

The reason for this extra step is because the **SimstackResults** Class, which plots fluxes and estimates values like SEDs and LIRs, will be constantly improving, while the original stacked fluxes never change, so should only need to be estimated once.  

Inside the object containing raw stacked flux densities (in Jy) and errors, and all data used to estimate them, and are accessed via python dictionaries:
- simstack_object.config_dict
- simstack_object.maps_dict
- simstack_object.catalog_dict
- simstack_object.results_dict

#### Inside the results_dict
There are many layers to the results_dict, but once you understand the labeling format, it is pretty intuitive.
All bins have a unique id_label = '__'.join([zlabel,mlabel,plabel]).  Typically, but not necessarily, they look like:
- zlabel = redshift_0.5_1.0
- mlabel = stellar_mass_10.5_11.0
- plabel = split_label_1

The results_dict contains:
- 'band_results_dict',  contains keys
  - ['mips_24', 'pacs_green', 'pacs_red', 'spire_psw', 'spire_pmw', 'spire_plw', 'scuba_850'], each containing the single band result and bootstraps
    - ['stacked_flux_densities', 'bootstrap_flux_densities_1', 'bootstrap_flux_densities_2', ..., 'raw_fluxes_dict'], each containing the result for each bin (or id_label)
      - ['redshift_0p01_0p5__stellar_mass_9p5_10p0__split_params_0', 'redshift_0p01_0p5__stellar_mass_9p5_10p0__split_params_0__bootstrap2', ...]
- 'sed_bootstrap_results_dict', contains processed results from bootstrap, i.e.;
  - ['wavelengths', 'z_median', 'm_median', 'ngals', 'sed_fluxes_dict', 'std_fluxes_dict', 'sed_bootstrap_fluxes_dict'], where
    - 'sed_fluxes_dict', and 'std_fluxes_dict' contain sed fluxes and errors, by bin, i.e.;
      - ['redshift_0.01_0.5__stellar_mass_9.5_10.0__split_params_0', 'redshift_0.01_0.5__stellar_mass_9.5_10.0__split_params_1', ...]
      - 
    - 'sed_bootstrap_fluxes_dict' contains all 150 (or whatever number of bootstraps) by bin, i.e.,;
      - ['redshift_0.01_0.5__stellar_mass_9.5_10.0__split_params_0', 'redshift_0.01_0.5__stellar_mass_9.5_10.0__split_params_1', ...]
    - 'z_median', 'm_median', and 'ngals' are also accessed by id_label, e.g.,
      - simstack_object.results_dict['sed_bootstrap_results_dict']['z_median'][id_label] is 2.245

By nested looping through the bins, results you would access data with the id_label, e.g.;

#### Visualizing results

Existing plots so far are:
- simstack_object.plot_flux_densities()
- simstack_object.plot_seds()
- simstack_object.plot_lir_vs_z()
- simstack_object.plot_mcmc_seds(flat_samples_dict, show_qt=show_qt, bootstrap_dict=None, errors=('25', '75'),save_path=fig_path,save_filename=fig_filename)
- simstack_object.plot_total_sfrd(total_sfrd_dict, save_path=fig_path, save_filename=fig_filename)
- simstack_object.plot_rest_frame_temperature(lir_dict, ylog=False, save_path=fig_path,save_filename=fig_filename)
- simstack_object.plot_star_forming_main_sequence(lir_dict, ylim=[5e-1, 5e3])
- simstack_object.plot_cib(cib_dict)
- simstack_object.plot_cib_layers(cib_dict, show_total=True)

See (estimate_and_save_values)[notebooks/4_estimate_and_save_values.ipynb] and (load_and_plot)[notebooks/5_load_and_plot.ipynb] for importing stacking pickles, parsing results, and plotting them. 
