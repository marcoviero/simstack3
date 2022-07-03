# The Early Universe was Dust-Rich and Extremely Hot
## Marco P. Viero, Guochao Sun, Dongwoo T. Chung, Lorenzo Moncelsi, and Sam S. Condon

Welcome to the public site for [Viero et al. 2022](https://arxiv.org/abs/2203.14312). The stacking algorithm used here is described in detail in [Viero et al. 2013](https://ui.adsabs.harvard.edu/abs/2013ApJ...779...32V/abstract).
The Python3 version (Release X 06/2022) algorithm is at [simstack3](https://github.com/marcoviero/simstack3), which contains a detailed [README](https://github.com/marcoviero/simstack3/blob/main/README.md) with instructions on installation and usage.    

Contained in this directory is the appliction of SIMSTACK on public data to arrive at the results in *The Early Universe was Dust-Rich and Extremely Hot*.  
It includes Jupyter notebooks with instructions detailing each step to reproduce the results.  

### Follow these instructions to reproduce the results from the accepted version of [Viero et al. 2022](https://arxiv.org/abs/2203.14312):

#### Setup

0. Setup Environment variables for MAPSPATH, CATSPATH, and PICKLESPATH: </br>
   - these tell the code which directories to find the data on your specific machine
    - see instructions for [Mac/Linux](https://phoenixnap.com/kb/set-environment-variable-mac), 
or [Windows](https://phoenixnap.com/kb/windows-set-environment-variable).

1. Download data and put where specified in step 0: </br>
   - see [V0_data_aquisition](./notebooks/Repositories/simstack3/viero2022/notebooks/V0_data_acquisition.ipynb) notebook for details of data origin.
   - alternatively, find direct download [here](https://sites.astro.caltech.edu/viero/viero2022/data/)
2. Clone SIMSTACK3 </br>
   - get SIMSTACK3 from [Gitub](https://github.com/marcoviero/simstack3)
3. Create a python environment for simstack in one of two ways: </br>
   - manually
        - conda create -n simstack python=3.9 and installing the
        - conda activate simstack
        - install packages described in SIMSTACK [README](../simstack3/README.md)
   - or directly installing via the [provided yaml file](../simstack3/config/simstack.yml)
        - conda env create -f simstack.yml
        - conda activate simstack
    
With everything installed and set up you're ready to stack.

#### Stack

Stacking the full bootstrap will take a long time on a powerful computer.  
If you want to skip this step and just download the stacked pickle, it can be found [here](https://sites.astro.caltech.edu/viero/viero2022/data/pickles/cosmos2020_farmer_nuvrj_0p01_0p5_1_1p5_2_2p5_3_3p5_4_5_6_8_10_X_4_foregnd_atonce_bootstrap_1-150/mcmc_samples_15000-3000-3sigma_mixed_prior_qt_dict.pkl).

Otherwise, stacking can be performed in three different ways:
1. [Jupyter Notebook](./notebooks/V1_perform_simstack.ipynb)  
2. Command line.  
    - From the command line (in the simstack environement, i.e.; conda activate simstack) type:
       - python run_simstack_cmd_line.py config/cosmos2020_farmer.ini 
3. IDE.
     - Confirm that line 59 of file run_simstack_cmd_line.py has the correct config file (default is example.ini, change that to cosmos_farmer.ini).
     - From PyCharm (or whatever you prefer) configure to use the simstack environment and Run.

Exactly reproducing the results (with bootstrap-derived error bars) requires a powerful computer, and can be time consuming. </br>
We recommend splitting the bootstrap estimation into multiple chunks, and combining them upon completion.  </br>
For example, we divided into 5 chunks, so that the 5 configuration files contained one of the following lines:
> error_estimator = {"bootstrap": {"initial_bootstrap": 1, "iterations": 10}, "write_simmaps": 0, "randomize": 0} </br>
> error_estimator = {"bootstrap": {"initial_bootstrap": 11, "iterations": 30}, "write_simmaps": 0, "randomize": 0} </br>
> error_estimator = {"bootstrap": {"initial_bootstrap": 41, "iterations": 30}, "write_simmaps": 0, "randomize": 0} </br>
> error_estimator = {"bootstrap": {"initial_bootstrap": 71, "iterations": 30}, "write_simmaps": 0, "randomize": 0} </br>
> error_estimator = {"bootstrap": {"initial_bootstrap": 101, "iterations": 50}, "write_simmaps": 0, "randomize": 0} </br>

Once stacking is completed, the resulting pickles will have to be merged.  See [this Notebook](./notebooks/V2_merge_and_save_pickles.ipynb) for details. 

#### Fit and Interpret
Fitting is done with emcee.  See [this Notebook](./notebooks/V3_emcee_seds.ipynb) for details. 

Once fitting is completed estimators like luminosities, dust temperatures, and dust masses must be calculated.  
See See [this Notebook](./notebooks/V4_estimate_and_save_values.ipynb) for details.  

#### Plots
See [this Notebook](./notebooks/V3_emcee_seds.ipynb) for details. 

#### More
We hope this will provide a springboard for your own explorations.  SIMSTACK is incredibly powerful, and far-infrared/submillimeter 
maps of high quality are publicly available.   