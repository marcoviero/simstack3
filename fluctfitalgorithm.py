import pdb
import numpy as np
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as planck18
from astropy.cosmology import Planck15 as planck15
from lmfit import Parameters, minimize, fit_report
from skymaps import Skymaps
from skycatalogs import Skycatalogs
from simstacktoolbox import SimstackToolbox
from fluctfitmodels import FluctFitModels
import time

class FluctFitAlgorithm(FluctFitModels, SimstackToolbox, Skymaps, Skycatalogs):

    results_dict = {}
    config_dict = {}
    datacube_dict = {}
    init_dict = {
        'A':
            {'sf': {'offset': -42,
                    'slope_redshift': 0.16,
                    'slope_stellar_mass': 0.7,
                    'slope_agn_fraction': 0.1,
                    },
             'qt': {'offset': -35,
                    'slope_redshift': 0.2,
                    'slope_stellar_mass': 0.1,
                    'slope_agn_fraction': 0.2,
                    },
             'agn': {'offset': -41,
                     'slope_redshift': 0.4,
                     'slope_stellar_mass': 0.6,
                     'slope_agn_fraction': -0.6,
                     },
             },
        'T':
            {'sf': {'offset': 49,
                    'slope_redshift': 5.6,
                    'slope_stellar_mass': -2.7,
                    'slope_agn_fraction': 0.5,
                    },
             'qt': {'offset': -8,
                    'slope_redshift': 7,
                    'slope_stellar_mass': 2,
                    'slope_agn_fraction': 2.3,
                    },
             'agn': {'offset': 18,
                     'slope_redshift': 0.8,
                     'slope_stellar_mass': -0.001,
                     'slope_agn_fraction': 24,
                     },
             },
    }

    def __init__(self, param_path_file):
        super().__init__()

        # Import parameters from config.ini file
        self.config_dict = self.get_params_dict(param_path_file)
        self.results_dict = {'band_results_dict': {}}

        # Define Cosmologies and identify chosen cosmology from config.ini
        cosmology_key = self.config_dict['general']['cosmology']
        self.config_dict['cosmology_dict'] = {'Planck18': planck18, 'Planck15': planck15}
        self.config_dict['cosmology_dict']['cosmology'] = self.config_dict['cosmology_dict'][cosmology_key]

    def get_init_parameters(self):
        fit_params_init = Parameters()
        for pop in self.datacube_dict['populations']:
            fit_params_init.add('A_offset_'+pop, value=self.init_dict['A'][pop]['offset'])
            fit_params_init.add('T_offset_'+pop, value=self.init_dict['T'][pop]['offset'])
            for key, val in self.config_dict['catalog']['classification'].items():
                if 'split_params' not in key:
                    fit_params_init.add('A_slope_' + key + '_' + pop,
                                       value=self.init_dict['A'][pop]['slope_' + key])
                    fit_params_init.add('T_slope_' + key + '_' + pop,
                                       value=self.init_dict['T'][pop]['slope_' + key])
        return fit_params_init

    def perform_fluctfit(self):
        t0 = time.time()
        fit_params_init = self.get_init_parameters()
        '''fit_params_init = Parameters()
        if 'sf' in self.datacube_dict['populations']:
            # Define Fit Parameters
            fit_params_init.add('A_offset_sf', value=self.init_dict['A']['sf']['offset'])
            fit_params_init.add('T_offset_sf', value=self.init_dict['T']['sf']['offset'])
            for key, val in self.config_dict['catalog']['classification'].items():
                if 'split_params' not in key:
                    fit_params_init.add('A_slope_' + key + '_sf',
                                       value=self.init_dict['A']['sf']['slope_' + key])
                    fit_params_init.add('T_slope_' + key + '_sf',
                                       value=self.init_dict['T']['sf']['slope_' + key])
                    
        if 'qt' in self.datacube_dict['populations']:
            fit_params_init.add('A_offset_qt', value=self.init_dict['A']['qt']['offset'])
            fit_params_init.add('T_offset_qt', value=self.init_dict['T']['qt']['offset'])
            for key, val in self.config_dict['catalog']['classification'].items():
                if 'split_params' not in key:
                    fit_params_init.add('A_slope_' + key + '_qt',
                                       value=self.init_dict['A']['qt']['slope_' + key])
                    fit_params_init.add('T_slope_' + key + '_qt',
                                       value=self.init_dict['T']['qt']['slope_' + key])
                    
        if 'agn' in self.datacube_dict['populations']:
            fit_params_init.add('A_offset_agn', value=self.init_dict['A']['agn']['offset'])
            fit_params_init.add('T_offset_agn', value=self.init_dict['T']['agn']['offset'])
            for key, val in self.config_dict['catalog']['classification'].items():
                if 'split_params' not in key:
                    fit_params_init.add('A_slope_' + key + '_agn',
                                       value=self.init_dict['A']['agn']['slope_' + key])
                    fit_params_init.add('T_slope_' + key + '_agn',
                                       value=self.init_dict['T']['agn']['slope_' + key])'''

        # Perform FluctFit
        if len(self.datacube_dict['populations']) == 1:
            self.results_dict['cov_direct_fit'] = minimize(self.direct_convolved_fit_A_Tdust_one_pop, fit_params_init,
                                                           args=(self.datacube_dict['datacube'][self.datacube_dict['idx']['sf']].to_numpy().T,),
                                                           kws={'y': self.datacube_dict['simmap_dict']},
                                                           nan_policy='propagate')
        # Perform FluctFit
        if len(self.datacube_dict['populations']) == 2:
            self.results_dict['cov_direct_fit'] = minimize(self.direct_convolved_fit_A_Tdust_two_pop, fit_params_init,
                                                           args=(self.datacube_dict['datacube'].to_numpy().T,),
                                                           kws={'y': self.datacube_dict['simmap_dict']},
                                                           nan_policy='propagate')
        # Perform FluctFit
        if len(self.datacube_dict['populations']) == 3:
            self.results_dict['cov_direct_fit'] = minimize(self.direct_convolved_fit_A_Tdust_three_pop, fit_params_init,
                                                           args=(self.datacube_dict['datacube'].to_numpy().T,),
                                                           kws={'y': self.datacube_dict['simmap_dict']},
                                                           nan_policy='propagate')
        # Summarize timing
        t1 = time.time()
        tpass = t1 - t0
        print("Fluctfit Successful!")
        print("Total time  : {0:0.4f} minutes\n".format(tpass / 60.))

        return self.results_dict['cov_direct_fit']

    def get_datacube(self, populations=['sf', 'qt', 'agn']):

        self.datacube_dict['populations'] = populations
        self.datacube_dict['catalog_keys'] = \
            [self.config_dict['catalog']['classification'][i]['id']
             for i in self.config_dict['catalog']['classification']]
        self.datacube_dict['catalog_dict'] = \
            {i: self.config_dict['catalog']['classification'][i]['id']
             for i in self.config_dict['catalog']['classification']}

        self.datacube_dict['datacube'] = \
            self.catalog_dict['tables']['full_table'][self.datacube_dict['catalog_keys']]
        if 'agn' in populations:
            if 'F_ratio' in self.datacube_dict['datacube']:
                fratio_min = 1e-2
                Fratio_cut = 1e-2
                self.datacube_dict['datacube']['F_ratio'] = np.log10(self.datacube_dict['datacube']['F_ratio'] + fratio_min)
                self.datacube_dict['datacube']['sfg'][self.datacube_dict['datacube']['F_ratio'] >= Fratio_cut] = 2
            elif 'a_hat_AGN' in self.datacube_dict['datacube']:
                ahat_cut = 0
                self.datacube_dict['datacube']['sfg'][self.datacube_dict['datacube']['a_hat_AGN'] > ahat_cut] = 2

        self.datacube_dict['idx'] = {
            'sf': self.datacube_dict['datacube']['sfg'] == 1,
            'qt': self.datacube_dict['datacube']['sfg'] == 0,
        }

        if np.sum(self.datacube_dict['datacube']['sfg'] == 2):
            self.datacube_dict['idx']['agn'] = self.datacube_dict['datacube']['sfg'] == 2

        self.datacube_dict['simmap_dict'] = {}
        for map_key in self.maps_dict:
            map_object = self.maps_dict[map_key].copy()
            map_object['map_coords'] = {}
            for pop_key, pop_idx in self.datacube_dict['idx'].items():
                ra = self.catalog_dict['tables']['full_table'].loc[pop_idx]['ra']
                dec = self.catalog_dict['tables']['full_table'].loc[pop_idx]['dec']
                cms = map_object['map'].shape
                hd = map_object['header']
                wmap = WCS(hd)
                x, y = self.get_x_y_from_ra_dec(wmap, cms, pop_idx, ra, dec)
                map_object['map_coords'][pop_key] = [x, y]
            self.datacube_dict['simmap_dict'][map_key] = map_object

        return self.datacube_dict['simmap_dict']



