import pdb
import os

from simstackalgorithm import SimstackAlgorithm
from simstackresults import SimstackResults

class SimstackWrapper(SimstackAlgorithm):
    ''' SimstackWrapper consolidates each step required to stack:
        - Read in parameters from config.ini file
        - Read in catalogs
        - Read in maps
        - Split catalog into bins specified in config file
        - Run stacking algorithm, which includes:
            -- create convolved layer cube at each wavelength [and optionally redshift]
        - Parse results into user-friendly pandas DataFrames

        :param param_path_file: (str)  path to the config.ini file
        :param read_maps: (bool) [optional; default True] If prefer to do this manually then set False
        :param read_catalogs: (bool) [optional; default True] If prefer to do this manually then set False
        :param stack_automatically: (bool) [optional; default False] If prefer to do this automatically then set True

        TODO:
        - counts in bins
        - agn selection to pops
        - CIB estimates
        - Simulated map of best-fits

    '''
    def __init__(self,
                 param_file_path,
                 read_maps=False,
                 read_catalog=False,
                 keep_catalog=False,
                 stack_automatically=False,
                 save_automatically=True,
                 parse_automatically=False,
                 overwrite_results=False):
        super().__init__(param_file_path)

        # Copy configuration file immediately, before it can be changed.
        if save_automatically:
            self.copy_config_file(param_file_path, overwrite_results=overwrite_results)

        if read_catalog:
            self.import_catalog(keep_catalog=keep_catalog)  # This happens in skycatalogs.py

        if read_maps:
            self.import_maps()  # This happens in skymaps.py

        if stack_automatically:
            # Bootstrap
            num_boots = 0
            if 'bootstrap' in self.config_dict['general']['error_estimator']:
                if self.config_dict['general']['error_estimator']['bootstrap']['iterations'] > 0:
                    num_boots = self.config_dict['general']['error_estimator']['bootstrap']['iterations']
                    init_boot = self.config_dict['general']['error_estimator']['bootstrap']['initial_bootstrap']
                    print('Bootstrapping {} iterations starting at {}'.format(num_boots, init_boot))

            for boot in range(num_boots + 1):
                if boot:
                    boot_in = boot - 1 + init_boot
                    self.perform_simstack(bootstrap=boot_in)
                else:
                    self.perform_simstack(bootstrap=boot)  # This happens in simstackalgorithm.py

        if self.stack_successful and parse_automatically:
            results_object = SimstackResults(self)
            results_object.parse_results()  # This happens in simstackresults.py
            setattr(self, 'results_dict', getattr(results_object, 'results_dict'))

        if save_automatically:
            saved_pickle_path = self.save_stacked_fluxes()
            print('saved to ', saved_pickle_path)
