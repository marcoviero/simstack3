#!/usr/bin/env python

'''

Set Environment Variables

export MAPSPATH=$MAPSPATH/Users/marcoviero/data/Astronomy/maps/
export CATSPATH=$CATSPATH/Users/marcoviero/data/Astronomy/catalogs/
export PICKLESPATH=$PICKLESPATH/Users/marcoviero/data/Astronomy/pickles/

Setup New Virtual Environment
> conda create -n simstack python=3.9
> conda activate simstack

Install Packages
- matplotlib (> conda install matplotlib)
- seaborn (> conda install seaborn)
- numpy (> conda install numpy)
- pandas (> conda install pandas)
- astropy (> conda install astropy)
- lmfit (> conda install -c conda-forge lmfit)
- [jupyterlab, if you want to use notebooks]

To run from command line:
- First make this file executable (only needed once), e.g.:
> chmod +x run_simstack_cmd_line.py
- Run script:
> python run_simstack_cmd_line.py

Returned object contains:
- simstack_object.config_dict; dict_keys(['general', 'io', 'catalog', 'maps', 'cosmology_dict', 'distance_bins', 'parameter_names', 'pickles_path'])
- simstack_object.catalog_dict; dict_keys(['tables'])
- simstack_object.maps_dict; dict_keys(['spire_psw', 'spire_pmw', ...])
- simstack_object.results_dict; dict_keys(['maps_dict', 'SED_df'])

'''

# Standard modules
import os
import pdb
import sys
import time
import logging

# Modules within this package
from simstackwrapper import SimstackWrapper

def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s  %(message)s',
        datefmt='%Y-%d-%m %I:%M:%S %p')

    # Get parameters from the provided parameter file
    if len(sys.argv) > 1:
        param_file_path = sys.argv[1]
    else:
        param_file_path = os.path.join('config', 'cosmos2020_null_test.ini')

    # Instantiate SIMSTACK object
    simstack_object = SimstackWrapper(param_file_path, save_automatically=False,
                                      read_maps=True, read_catalog=True)

    simstack_object.copy_config_file(param_file_path, overwrite_results=False)
    print('Now Stacking', param_file_path)
    t0 = time.time()

    # Stack according to parameters in parameter file
    # Bootstrap
    num_boots = 0
    if 'bootstrap' in simstack_object.config_dict['general']['error_estimator']:
        if simstack_object.config_dict['general']['error_estimator']['bootstrap']['iterations'] > 0:
            num_boots = simstack_object.config_dict['general']['error_estimator']['bootstrap']['iterations']
            init_boot = simstack_object.config_dict['general']['error_estimator']['bootstrap']['initial_bootstrap']
            print('Bootstrapping {} iterations starting at {}'.format(num_boots, init_boot))

    # Shuffle x/y positions as Null test
    randomize = False
    if 'randomize' in simstack_object.config_dict['general']['error_estimator']:
        if simstack_object.config_dict['general']['error_estimator']['randomize']:
            randomize = True

    # Convolve maps to have same psf
    force_fwhm = False
    if 'force_fwhm' in simstack_object.config_dict['general']['binning']:
        if simstack_object.config_dict['general']['binning']['force_fwhm']:
            force_fwhm = simstack_object.config_dict['general']['binning']['force_fwhm']

    for boot in range(num_boots + 1):
        if boot:
            boot_in = boot - 1 + init_boot
            simstack_object.perform_simstack(bootstrap=boot_in, randomize=randomize, force_fwhm=force_fwhm)
        else:
            simstack_object.perform_simstack(bootstrap=boot, randomize=randomize, force_fwhm=force_fwhm)

    # Save Results
    saved_pickle_path = simstack_object.save_stacked_fluxes(param_file_path)

    # Summarize timing
    t1 = time.time()
    tpass = t1 - t0

    logging.info("Stacking Successful!")
    logging.info("Find Results in {}".format(saved_pickle_path))
    logging.info("")
    logging.info("Total time  : {:.4f} minutes\n".format(tpass / 60.))

if __name__ == "__main__":
    main()
else:
    logging.info("Note: `mapit` module not being run as main executable.")
