
from fluctfitalgorithm import FluctFitAlgorithm
#from simstackresults import SimstackResults

class FluctFitWrapper(FluctFitAlgorithm):
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
                 fit_automatically=False,
                 save_automatically=True,
                 populations=['sf', 'qt', 'agn'],
                 overwrite_results=False):
        super().__init__(param_file_path)

        # Copy configuration file immediately, before it can be changed.
        if save_automatically:
            self.copy_config_file(param_file_path, overwrite_results=overwrite_results)

        if read_catalog:
            self.import_catalog(keep_catalog=keep_catalog)  # This happens in skycatalogs.py

        if read_maps:
            self.import_maps()  # This happens in skymaps.py

        if fit_automatically:
            simmap_dict = self.get_datacube(populations=populations)
            cov_direct_fit = self.perform_fluctfit()

        '''
        if fit_automatically:
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
                    self.perform_fluctfit(bootstrap=boot_in)
                else:
                    self.perform_fluctfit(bootstrap=boot)  # This happens in fluctfitalgorithm.py

        if self.stack_successful and parse_automatically:
            results_object = FluctFitResults(self)
            results_object.parse_results()  # This happens in fluctfitresults.py
            setattr(self, 'results_dict', getattr(results_object, 'results_dict'))

        if save_automatically:
            saved_pickle_path = self.save_stacked_fluxes()
            print('saved to ', saved_pickle_path)
        '''
