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
- simstack_object.config_dict; dict_keys(['general', 'cosmology', 'io', 'catalog', 'maps'])
- simstack_object.catalog_dict; dict_keys(['table'])
- simstack_object.maps_dict; dict_keys(['spire_psw', 'spire_pmw', ...])
- simstack_object.results_dict; dict_keys(['spire_plw', 'wavelengths']) # change to results, metadata

Internal methods (i.e., functions) include:
- import_catalog
- import_maps
- import_pickles

Keyword arguments include:
- Estimate Extragalactic Background Light (EBL)
- Estimate Temperatures
- Estimate Bayesian uncertainties via. MCMC
'''

# Standard modules
import os
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

    # Debug Flag
    debug = True

    # Get parameters from the provided parameter file
    try:
        param_file_path = sys.argv[1]
    except:
        param_file_path = os.path.join('examples', 'hers_hetdex.ini')

    # Instantiate SIMSTACK object
    simstack_object = SimstackWrapper(param_file_path, save_automatically=False,
                                      read_maps=True, read_catalog=True, debug=debug)

    print('Now Stacking', param_file_path)
    t0 = time.time()

    # Stack according to parameters in parameter file
    simstack_object.perform_simstack()

    # Save Results
    saved_pickle_path = simstack_object.save_stacked_fluxes(param_file_path)

    # Summarize timing
    t1 = time.time()
    tpass = t1 - t0

    logging.info("Stacking Successful!")
    logging.info("Find Results in {}".format(simstack_object.config_dict['pickles_path']))
    logging.info("")
    logging.info("Total time                        : {:.4f} minutes\n".format(tpass / 60.))

    #pdb.set_trace()

if __name__ == "__main__":
    main()
else:
    logging.info("Note: `mapit` module not being run as main executable.")
