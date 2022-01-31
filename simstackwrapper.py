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
    def __init__(self, param_file_path, read_maps=False, read_catalog=False,
                 stack_automatically=False, save_automatically=True, parse_automatically=False,
                 overwrite_results=False, debug=False):
        super().__init__(param_file_path)

        if read_catalog:
            self.import_catalog()  # This happens in skycatalogs.py

        if read_maps:
            self.import_maps()  # This happens in skymaps.py

        if stack_automatically:
            # Bootstrap
            boots = 0
            if 'bootstrap' in self.config_dict['general']['error_estimator']:
                if self.config_dict['general']['error_estimator']['bootstrap']['iterations'] > 0:
                    boots = self.config_dict['general']['error_estimator']['bootstrap']['iterations']
                    seed = self.config_dict['general']['error_estimator']['bootstrap']['seed']
                    print('Bootstrapping {} iterations'.format(boots))

            for boot in range(boots+1):
                self.perform_simstack(bootstrap=boot)  # This happens in simstackalgorithm.py

        if self.stack_successful and parse_automatically:
            results_object = SimstackResults(self)
            results_object.parse_results()  # This happens in simstackresults.py
            setattr(self, 'results_dict', getattr(results_object, 'results_dict'))

        if save_automatically:
            saved_pickle_path = self.save_stacked_fluxes(param_file_path, overwrite_results=overwrite_results)
            print('saved to ', saved_pickle_path)
