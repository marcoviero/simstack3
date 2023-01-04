import pdb
import os
import json
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.cosmology import Planck18 as planck18
from astropy.cosmology import Planck15 as planck15
from sklearn.model_selection import train_test_split
from lmfit import Parameters, minimize, fit_report
from skymaps import Skymaps
from skycatalogs import Skycatalogs
from simstacktoolbox import SimstackToolbox

class SimstackAlgorithm(SimstackToolbox, Skymaps, Skycatalogs):

    stack_successful = False
    config_dict = {}

    def __init__(self, param_path_file):
        super().__init__()

        # Import parameters from config.ini file
        self.config_dict = self.get_params_dict(param_path_file)
        self.results_dict = {'band_results_dict': {}}

        # Define Cosmologies and identify chosen cosmology from config.ini
        cosmology_key = self.config_dict['general']['cosmology']
        self.config_dict['cosmology_dict'] = {'Planck18': planck18, 'Planck15': planck15}
        self.config_dict['cosmology_dict']['cosmology'] = self.config_dict['cosmology_dict'][cosmology_key]

        # Store redshifts and lookback times.
        zbins = json.loads(self.config_dict['catalog']['classification']['redshift']['bins'])
        self.config_dict['distance_bins'] = {'redshift': zbins,
                                             'lookback_time': self.config_dict['cosmology_dict']['cosmology'].lookback_time(zbins)}

    def perform_simstack(self,
                         bootstrap=0,
                         add_foreground=False,
                         crop_circles=True,
                         stack_all_z_at_once=False,
                         write_simmaps=False,
                         force_fwhm=None,
                         randomize=False):
        '''
        perform_simstack takes the following steps:
        0. Get catalog and drop nans
        1. Assign parameter labels
        2. Call stack_in_wavelengths

        Following parameters are overwritten if included in config file.
        :param add_foreground: (bool) add additional foreground layer.
        :param crop_circles: (bool) exclude masked areas.
        :params stack_all_z_at_once: (bool) choose between stacking in redshift slices or all at once.
        '''
        if 'stack_all_z_at_once' not in self.config_dict['general']['binning']:
            self.config_dict['general']['binning']['stack_all_z_at_once'] = stack_all_z_at_once
        if 'crop_circles' not in self.config_dict['general']['binning']:
            self.config_dict['general']['binning']['crop_circles'] = crop_circles
        if 'add_foreground' not in self.config_dict['general']['binning']:
            self.config_dict['general']['binning']['add_foreground'] = add_foreground
        if 'write_simmaps' not in self.config_dict['general']['error_estimator']:
            self.config_dict['general']['error_estimator']['write_simmaps'] = write_simmaps
        stack_all_z_at_once = self.config_dict['general']['binning']['stack_all_z_at_once']
        crop_circles = self.config_dict['general']['binning']['crop_circles']
        add_foreground = self.config_dict['general']['binning']['add_foreground']
        write_simmaps = self.config_dict['general']['error_estimator']['write_simmaps']

        # Get catalog.  Clean NaNs
        catalog = self.catalog_dict['tables']['split_table'].dropna()

        # Get binning details
        split_dict = self.config_dict['catalog']['classification']
        if 'split_type' in split_dict:
            print('split_dict looks to be broken')
            pdb.set_trace()
        nlists = []
        for k in split_dict:
            kval = split_dict[k]['bins']
            if type(kval) is str:
                nlists.append(len(json.loads(kval))-1)  # bins so subtract 1
            elif type(kval) is dict:
                nlists.append(len(kval))
            else:
                nlists.append(kval)
        nlayers = np.prod(nlists[1:])

        # Stack in redshift slices if stack_all_z_at_once is False
        bins = json.loads(split_dict["redshift"]['bins'])
        distance_labels = []
        if not bootstrap:
            flux_density_key = 'stacked_flux_densities'
        else:
            flux_density_key = 'bootstrap_flux_densities_'+str(bootstrap)
        print(flux_density_key)

        if stack_all_z_at_once == False:
            redshifts = catalog.pop("redshift")
            for i in np.unique(redshifts):
                catalog_in = catalog[redshifts == i]
                distance_label = "_".join(["redshift", str(bins[int(i)]), str(bins[int(i) + 1])]).replace('.', 'p')
                distance_labels.append(distance_label)
                if bootstrap:
                    labels = self.split_bootstrap_labels(self.catalog_dict['tables']['parameter_labels'][int(i*nlayers):int((i+1)*nlayers)])
                else:
                    labels = self.catalog_dict['tables']['parameter_labels'][int(i*nlayers):int((i+1)*nlayers)]
                if add_foreground:
                    labels.append("ones_foreground")
                cov_ss_out = self.stack_in_wavelengths(catalog_in, labels=labels, distance_interval=distance_label,
                                                       crop_circles=crop_circles, add_foreground=add_foreground,
                                                       bootstrap=bootstrap, force_fwhm=force_fwhm, randomize=randomize,
                                                       write_fits_layers=write_simmaps)
                for wv in cov_ss_out:
                    if wv not in self.results_dict['band_results_dict']:
                        self.results_dict['band_results_dict'][wv] = {}
                    if flux_density_key not in self.results_dict['band_results_dict'][wv]:
                        self.results_dict['band_results_dict'][wv][flux_density_key] = cov_ss_out[wv].params
                    else:
                        self.results_dict['band_results_dict'][wv][flux_density_key].update(cov_ss_out[wv].params)
        else:
            labels = []
            for i in np.unique(catalog['redshift']):
                if bootstrap:
                    labels = self.split_bootstrap_labels(self.catalog_dict['tables']['parameter_labels'])
                else:
                    labels.extend(self.catalog_dict['tables']['parameter_labels'][int(i*nlayers):int((i+1)*nlayers)])
                distance_labels.append("_".join(["redshift", str(bins[int(i)]), str(bins[int(i) + 1])]).replace('.', 'p'))
            if add_foreground:
                labels.append("ones_foreground")
            cov_ss_out = self.stack_in_wavelengths(catalog, labels=labels, distance_interval='all_redshifts',
                                                   crop_circles=crop_circles, add_foreground=add_foreground,
                                                   bootstrap=bootstrap, force_fwhm=force_fwhm, randomize=randomize,
                                                   write_fits_layers=write_simmaps)

            for wv in cov_ss_out:
                if wv not in self.results_dict['band_results_dict']:
                    self.results_dict['band_results_dict'][wv] = {}
                self.results_dict['band_results_dict'][wv][flux_density_key] = cov_ss_out[wv].params

        self.config_dict['catalog']['distance_labels'] = distance_labels
        self.stack_successful = True

    def build_cube(self,
                   map_dict,
                   catalog,
                   labels=None,
                   add_foreground=False,
                   crop_circles=False,
                   bootstrap=False,
                   force_fwhm=None,
                   randomize=False,
                   write_fits_layers=False):
        ''' Construct layer cube containing smoothed 2D arrays with positions defined by binning algorithm.
        Optionally, foreground layer can be added; positions can be randomized for null testing; layers can be
        smoothed to forced fwhm.

        :param map_dict: Dict containing map (and optionally noise).
        :param catalog: Catalog containing columns for ra, dec, and defining bins.
        :param labels: Cube layer labels.
        :param add_foreground: If True adds foreground layer.
        :param crop_circles: If True crops unnecessary pixels.
        :param bootstrap: Integer number as rng seed.
        :param force_fwhm: Float target fwhm to smooth degrade maps to.
        :param randomize: If True randomize source positions.
        :param write_fits_layers: If True write layers to .fits.
        :return: Dictionary containing 'cube' and 'labels'
        '''

        cmap = map_dict['map'].copy()
        if 'noise' in map_dict:
            cnoise = map_dict['noise'].copy()
        else:
            cnoise = cmap * 0
        pix = map_dict['pixel_size']
        hd = map_dict['header']
        fwhm = map_dict['fwhm']
        wmap = WCS(hd)

        # Extract RA and DEC from catalog
        ra_series = catalog.pop('ra')
        dec_series = catalog.pop('dec')
        keys = list(catalog.keys())

        # FIND SIZES OF MAP AND LISTS
        cms = np.shape(cmap)

        label_dict = self.config_dict['parameter_names']
        ds = [len(label_dict[k]) for k in label_dict]

        if (len(labels) - add_foreground) == np.prod(ds[1:]):
            nlists = ds[1:]
            llists = np.prod(nlists)
        elif (len(labels) - add_foreground)/2 == np.prod(ds[1:]):
            nlists = ds[1:]
            llists = 2 * np.prod(nlists)
        elif (len(labels) - add_foreground)/2 == np.prod(ds):
            nlists = ds
            llists = 2 * np.prod(nlists)
        elif (len(labels) - add_foreground) == np.prod(ds):
            nlists = ds
            llists = np.prod(nlists)
            #pdb.set_trace()

        # STEP 1  - Make Layers Cube
        layers = np.zeros([llists, cms[0], cms[1]])

        trimmed_labels = []
        ilayer = 0
        ilabel = 0
        for ipop in range(nlists[0]):
            if len(nlists) > 1:
                for jpop in range(nlists[1]):
                    if len(nlists) > 2:
                        for kpop in range(nlists[2]):
                            if len(nlists) > 3:
                                for lpop in range(nlists[3]):
                                    if len(nlists) > 4:
                                        for mpop in range(nlists[4]):
                                            if len(nlists) > 5:
                                                for npop in range(nlists[5]):
                                                    ind_src = (catalog[keys[0]] == ipop) & (catalog[keys[1]] == jpop) & \
                                                              (catalog[keys[2]] == kpop) & (catalog[keys[3]] == lpop) & \
                                                              (catalog[keys[4]] == mpop) & (catalog[keys[5]] == npop)
                                                    if bootstrap:
                                                        if sum(ind_src) > 4:
                                                            real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms,
                                                                                                      ind_src,
                                                                                                      ra_series,
                                                                                                      dec_series)
                                                            bt_split = 0.80
                                                            # jk_split = np.random.uniform(0.3, 0.7)
                                                            # print('jackknife split = ', jk_split)
                                                            left_x, right_x, left_y, right_y = \
                                                                train_test_split(real_x, real_y, test_size=bt_split,
                                                                                 andom_state=int(bootstrap),
                                                                                 shuffle=True)
                                                            layers[ilayer, left_x, left_y] += 1.0
                                                            layers[ilayer + 1, right_x, right_y] += 1.0
                                                            # layers[ilayer, right_x, right_y] += 1.0
                                                            # layers[ilayer + 1, left_x, left_y] += 1.0
                                                            trimmed_labels.append(labels[ilabel])
                                                            trimmed_labels.append(labels[ilabel + 1])
                                                            ilayer += 2
                                                        else:
                                                            layers = np.delete(layers, ilayer + 1, 0)
                                                            layers = np.delete(layers, ilayer, 0)
                                                        ilabel += 2
                                                    else:
                                                        if sum(ind_src) > 0:
                                                            real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms,
                                                                                                      ind_src,
                                                                                                      ra_series,
                                                                                                      dec_series)
                                                            if randomize:
                                                                # print('Shuffling!',len(real_x))
                                                                # print(real_x[0], real_y[0])
                                                                # np.random.shuffle(real_x)
                                                                # np.random.shuffle(real_y)
                                                                real_x = np.random.random_integers(min(real_x),
                                                                                                   max(real_x),
                                                                                                   len(real_x))
                                                                real_y = np.random.random_integers(min(real_y),
                                                                                                   max(real_y),
                                                                                                   len(real_y))
                                                                # pdb.set_trace()
                                                                # print(real_x[0], real_y[0])
                                                            layers[ilayer, real_x, real_y] += 1.0
                                                            trimmed_labels.append(labels[ilabel])
                                                            ilayer += 1
                                                        else:
                                                            layers = np.delete(layers, ilayer, 0)
                                                        ilabel += 1
                                            else:

                                                ind_src = (catalog[keys[0]] == ipop) & (catalog[keys[1]] == jpop) & \
                                                          (catalog[keys[2]] == kpop) & (catalog[keys[3]] == lpop) & \
                                                          (catalog[keys[4]] == mpop)
                                                if bootstrap:
                                                    if sum(ind_src) > 4:
                                                        real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src,
                                                                                                  ra_series,
                                                                                                  dec_series)
                                                        bt_split = 0.80
                                                        # jk_split = np.random.uniform(0.3, 0.7)
                                                        # print('jackknife split = ', jk_split)
                                                        left_x, right_x, left_y, right_y = \
                                                            train_test_split(real_x, real_y, test_size=bt_split,
                                                                             andom_state=int(bootstrap),shuffle=True)
                                                        layers[ilayer, left_x, left_y] += 1.0
                                                        layers[ilayer + 1, right_x, right_y] += 1.0
                                                        # layers[ilayer, right_x, right_y] += 1.0
                                                        # layers[ilayer + 1, left_x, left_y] += 1.0
                                                        trimmed_labels.append(labels[ilabel])
                                                        trimmed_labels.append(labels[ilabel + 1])
                                                        ilayer += 2
                                                    else:
                                                        layers = np.delete(layers, ilayer + 1, 0)
                                                        layers = np.delete(layers, ilayer, 0)
                                                    ilabel += 2
                                                else:
                                                    if sum(ind_src) > 0:
                                                        real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src,
                                                                                                  ra_series,
                                                                                                  dec_series)
                                                        if randomize:
                                                            # print('Shuffling!',len(real_x))
                                                            # print(real_x[0], real_y[0])
                                                            # np.random.shuffle(real_x)
                                                            # np.random.shuffle(real_y)
                                                            real_x = np.random.random_integers(min(real_x), max(real_x),
                                                                                               len(real_x))
                                                            real_y = np.random.random_integers(min(real_y), max(real_y),
                                                                                               len(real_y))
                                                            # pdb.set_trace()
                                                            # print(real_x[0], real_y[0])
                                                        layers[ilayer, real_x, real_y] += 1.0
                                                        trimmed_labels.append(labels[ilabel])
                                                        ilayer += 1
                                                    else:
                                                        layers = np.delete(layers, ilayer, 0)
                                                    ilabel += 1
                                    else:
                                        ind_src = (catalog[keys[0]] == ipop) & (catalog[keys[1]] == jpop) & (
                                                    catalog[keys[2]] == kpop) & (catalog[keys[3]] == lpop)
                                        if bootstrap:
                                            if sum(ind_src) > 4:
                                                real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series,
                                                                                          dec_series)
                                                bt_split = 0.80
                                                # jk_split = np.random.uniform(0.3, 0.7)
                                                # print('jackknife split = ', jk_split)
                                                left_x, right_x, left_y, right_y = train_test_split(real_x, real_y,
                                                                                                    test_size=bt_split,
                                                                                                    random_state=int(
                                                                                                        bootstrap),
                                                                                                    shuffle=True)
                                                layers[ilayer, left_x, left_y] += 1.0
                                                layers[ilayer + 1, right_x, right_y] += 1.0
                                                # layers[ilayer, right_x, right_y] += 1.0
                                                # layers[ilayer + 1, left_x, left_y] += 1.0
                                                trimmed_labels.append(labels[ilabel])
                                                trimmed_labels.append(labels[ilabel + 1])
                                                ilayer += 2
                                            else:
                                                layers = np.delete(layers, ilayer + 1, 0)
                                                layers = np.delete(layers, ilayer, 0)
                                            ilabel += 2
                                        else:
                                            if sum(ind_src) > 0:
                                                real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series,
                                                                                          dec_series)
                                                if randomize:
                                                    # print('Shuffling!',len(real_x))
                                                    # print(real_x[0], real_y[0])
                                                    # np.random.shuffle(real_x)
                                                    # np.random.shuffle(real_y)
                                                    real_x = np.random.random_integers(min(real_x), max(real_x),
                                                                                       len(real_x))
                                                    real_y = np.random.random_integers(min(real_y), max(real_y),
                                                                                       len(real_y))
                                                    # pdb.set_trace()
                                                    # print(real_x[0], real_y[0])
                                                layers[ilayer, real_x, real_y] += 1.0
                                                trimmed_labels.append(labels[ilabel])
                                                ilayer += 1
                                            else:
                                                layers = np.delete(layers, ilayer, 0)
                                            ilabel += 1
                            else:
                                ind_src = (catalog[keys[0]] == ipop) & (catalog[keys[1]] == jpop) & (catalog[keys[2]] == kpop)
                                if bootstrap:
                                    if sum(ind_src) > 4:
                                        real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series, dec_series)
                                        bt_split = 0.80
                                        #jk_split = np.random.uniform(0.3, 0.7)
                                        #print('jackknife split = ', jk_split)
                                        left_x, right_x, left_y, right_y = train_test_split(real_x, real_y,
                                                                                            test_size=bt_split,
                                                                                            random_state=int(bootstrap),
                                                                                            shuffle=True)
                                        layers[ilayer, left_x, left_y] += 1.0
                                        layers[ilayer + 1, right_x, right_y] += 1.0
                                        #layers[ilayer, right_x, right_y] += 1.0
                                        #layers[ilayer + 1, left_x, left_y] += 1.0
                                        trimmed_labels.append(labels[ilabel])
                                        trimmed_labels.append(labels[ilabel + 1])
                                        ilayer += 2
                                    else:
                                        layers = np.delete(layers, ilayer+1, 0)
                                        layers = np.delete(layers, ilayer, 0)
                                    ilabel += 2
                                else:
                                    if sum(ind_src) > 0:
                                        real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series, dec_series)
                                        if randomize:
                                            #print('Shuffling!',len(real_x))
                                            #print(real_x[0], real_y[0])
                                            #np.random.shuffle(real_x)
                                            #np.random.shuffle(real_y)
                                            real_x = np.random.random_integers(min(real_x), max(real_x), len(real_x))
                                            real_y = np.random.random_integers(min(real_y), max(real_y), len(real_y))
                                            #pdb.set_trace()
                                            #print(real_x[0], real_y[0])
                                        layers[ilayer, real_x, real_y] += 1.0
                                        trimmed_labels.append(labels[ilabel])
                                        ilayer += 1
                                    else:
                                        layers = np.delete(layers, ilayer, 0)
                                    ilabel += 1
                    else:
                        ind_src = (catalog[keys[0]] == ipop) & (catalog[keys[1]] == jpop)
                        if bootstrap:
                            if sum(ind_src) > 4:
                                real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series, dec_series)
                                if randomize:
                                    #np.random.shuffle(real_x)
                                    #np.random.shuffle(real_y)
                                    real_x = np.random.random_integers(min(real_x), max(real_x), len(real_x))
                                    real_y = np.random.random_integers(min(real_y), max(real_y), len(real_y))
                                bt_split = 0.80
                                left_x, right_x, left_y, right_y = train_test_split(real_x, real_y,
                                                                                    test_size=bt_split,
                                                                                    random_state=int(bootstrap),
                                                                                    shuffle=True)
                                layers[ilayer, left_x, left_y] += 1.0
                                layers[ilayer + 1, right_x, right_y] += 1.0
                                #layers[ilayer, right_x, right_y] += 1.0  #Change to these for final stack
                                #layers[ilayer + 1, left_x, left_y] += 1.0
                                trimmed_labels.append(labels[ilabel])
                                trimmed_labels.append(labels[ilabel+1])
                                ilayer += 2
                            else:
                                layers = np.delete(layers, ilayer+1, 0)
                                layers = np.delete(layers, ilayer, 0)
                            ilabel += 2
                        else:
                            if sum(ind_src) > 0:
                                real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series, dec_series)
                                if randomize:
                                    #np.random.shuffle(real_x)
                                    #np.random.shuffle(real_y)
                                    real_x = np.random.random_integers(min(real_x), max(real_x), len(real_x))
                                    real_y = np.random.random_integers(min(real_y), max(real_y), len(real_y))
                                layers[ilayer, real_x, real_y] += 1.0
                                trimmed_labels.append(labels[ilabel])
                                ilayer += 1
                            else:
                                layers = np.delete(layers, ilayer, 0)
                            ilabel += 1
            else:
                ind_src = (catalog[keys[0]] == ipop)
                if bootstrap:
                    if sum(ind_src) > 4:
                        real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series, dec_series)
                        if randomize:
                            #np.random.shuffle(real_x)
                            #np.random.shuffle(real_y)
                            real_x = np.random.random_integers(min(real_x), max(real_x), len(real_x))
                            real_y = np.random.random_integers(min(real_y), max(real_y), len(real_y))
                        bt_split = 0.80
                        left_x, right_x, left_y, right_y = train_test_split(real_x, real_y, test_size=bt_split,
                                                                            random_state=int(bootstrap),
                                                                            shuffle=True)
                        layers[ilayer, left_x, left_y] += 1.0
                        layers[ilayer + 1, right_x, right_y] += 1.0
                        #layers[ilayer, right_x, right_y] += 1.0
                        #layers[ilayer + 1, left_x, left_y] += 1.0
                        trimmed_labels.append(labels[ilabel])
                        trimmed_labels.append(labels[ilabel + 1])
                        ilayer += 2
                    else:
                        layers = np.delete(layers, ilayer + 1, 0)
                        layers = np.delete(layers, ilayer, 0)
                    ilabel += 2
                else:
                    if sum(ind_src) > 0:
                        real_x, real_y = self.get_x_y_from_ra_dec(wmap, cms, ind_src, ra_series, dec_series)
                        if randomize:
                            #np.random.shuffle(real_x)
                            #np.random.shuffle(real_y)
                            real_x = np.random.random_integers(min(real_x), max(real_x), len(real_x))
                            real_y = np.random.random_integers(min(real_y), max(real_y), len(real_y))
                        layers[ilayer, real_x, real_y] += 1.0
                        trimmed_labels.append(labels[ilabel])
                        ilayer += 1
                    else:
                        layers = np.delete(layers, ilayer, 0)
                    ilabel += 1

        nlayers = np.shape(layers)[0]

        # STEP 2  - Convolve Layers and put in pixels
        if "write_simmaps" in self.config_dict["general"]["error_estimator"]:
            if self.config_dict["general"]["error_estimator"]["write_simmaps"] == 1:
                map_dict["convolved_layer_cube"] = np.zeros(np.shape(layers))

        if crop_circles:
            radius = 1.1
            flattened_pixmap = np.sum(layers, axis=0)
            total_circles_mask = self.circle_mask(flattened_pixmap, radius * fwhm, pix)
            ind_fit = np.where(total_circles_mask >= 1)
        else:
            ind_fit = np.where(0 * np.sum(layers, axis=0) == 0)

        nhits = np.shape(ind_fit)[1]
        if add_foreground:
            cfits_maps = np.zeros([nlayers + 3, nhits])  # +3 to append foreground, cmap and cnoise
            trimmed_labels.append('foreground_layer')
        else:
            cfits_maps = np.zeros([nlayers + 2, nhits])  # +2 to append cmap and cnoise

        # If smoothing maps to all have same FWHM
        if force_fwhm:
            if force_fwhm > fwhm:
                fwhm_eff = np.sqrt(force_fwhm**2 - fwhm**2)
                print("convolving {0:0.1f} map with {1:0.1f} arcsec psf".format(fwhm, fwhm_eff))
                kern_eff = self.gauss_kern(fwhm_eff, np.floor(fwhm_eff * 10) / pix, pix)
                kern_eff = kern_eff / np.sum(kern_eff)  # * (force_fwhm / fwhm_eff) ** 2.  # Adopted from IDL code.
                cmap = self.smooth_psf(cmap, kern_eff)
                cnoise = self.smooth_psf(cnoise, kern_eff)  # what to do with noise?
                kern = self.gauss_kern(force_fwhm, np.floor(force_fwhm * 10) / pix, pix)
            else:
                print("not convolving {0:0.1f} map ".format(fwhm))
                kern = self.gauss_kern(fwhm, np.floor(fwhm * 10) / pix, pix)
        else:
            kern = self.gauss_kern(fwhm, np.floor(fwhm * 10) / pix, pix)

        for umap in range(nlayers):
            layer = layers[umap, :, :]
            tmap = self.smooth_psf(layer, kern)

            # write layers to fits files here
            if write_fits_layers:
                path_layer = r'D:\maps\cutouts\layers'
                name_layer = 'layer_'+str(umap)+'.fits'
                name_layer = '{0}__fwhm_{1:0.1f}'.format(trimmed_labels[umap], fwhm).replace('.','p')+'.fits'
                if 'foreground_layer' not in trimmed_labels[umap]:
                    hdu = fits.PrimaryHDU(tmap, header=hd)
                    hdul = fits.HDUList([hdu])
                    hdul.writeto(os.path.join(path_layer, name_layer), overwrite=True)

            # Remove mean from map
            cfits_maps[umap, :] = tmap[ind_fit] - np.mean(tmap[ind_fit])

            # Add layer to write_simmaps cube
            if "convolved_layer_cube" in map_dict:
                map_dict["convolved_layer_cube"][umap, :, :] = tmap - np.mean(tmap[ind_fit])

            # If add_foreground=True, add foreground layer of ones.
        if add_foreground:
            cfits_maps[-3, :] = np.ones(np.shape(cmap[ind_fit]))

        # put map and noisemap in last two layers
        cfits_maps[-2, :] = cmap[ind_fit]
        cfits_maps[-1, :] = cnoise[ind_fit]

        return {'cube': cfits_maps, 'labels': trimmed_labels}

    def stack_in_wavelengths(self,
                             catalog,
                             labels=None,
                             distance_interval=None,
                             force_fwhm=None,
                             crop_circles=False,
                             add_foreground=False,
                             bootstrap=False,
                             randomize=False,
                             write_fits_layers=False):
        ''' Loop through wavelengths and perform simstack.

        :param catalog: Table containing ra, dec, and columns defining bins.
        :param labels: Labels for each layer.
        :param distance_interval: If stacking in redshift slices.
        :param force_fwhm: If smoothing images to a forced fwhm (work in progress)
        :param crop_circles: If True crops unused pixels.
        :param add_foreground: If True add foreground layer.
        :param bootstrap: Integer seed for random number generator.
        :param randomize: If True shuffles ra/dec positions.
        :param write_fits_layers: If True writes layers into .fits.
        :return cov_ss_dict: Dict containing stacked fluxes per wavelengths.
        '''

        map_keys = list(self.maps_dict.keys())
        cov_ss_dict = {}

        # Loop through wavelengths
        for wv in map_keys:

            map_dict = self.maps_dict[wv].copy()

            # Construct cube and labels for regression via lmfit.
            cube_dict = self.build_cube(map_dict, catalog.copy(), labels=labels, crop_circles=crop_circles,
                                        add_foreground=add_foreground, bootstrap=bootstrap, randomize=randomize,
                                        force_fwhm=force_fwhm, write_fits_layers=write_fits_layers)
            cube_labels = cube_dict['labels']
            print("Simultaneously Stacking {} Layers in {}".format(len(cube_labels), wv))

            # Regress cube (i.e., this is simstack!)
            cov_ss_1d = self.regress_cube_layers(cube_dict['cube'], labels=cube_dict['labels'])

            # Store in redshift slices.
            if 'stacked_flux_densities' not in map_dict:
                map_dict['stacked_flux_densities'] = {distance_interval: cov_ss_1d}
            else:
                map_dict['stacked_flux_densities'][distance_interval] = cov_ss_1d

            # Add stacked fluxes to output dict
            cov_ss_dict[wv] = cov_ss_1d

            # Write simulated maps from best-fits
            if self.config_dict["general"]["error_estimator"]["write_simmaps"]:
                for i, iparam_label in enumerate(cube_dict['labels']):
                    param_label = iparam_label.replace('.', 'p')
                    if 'foreground' not in iparam_label:
                        map_dict["convolved_layer_cube"][i, :, :] *= cov_ss_1d.params[param_label].value

                self.maps_dict[wv]["flattened_simmap"] = np.sum(map_dict["convolved_layer_cube"], axis=0)
                if 'foreground_layer' in cube_dict['labels']:
                    self.maps_dict[wv]["flattened_simmap"] += cov_ss_1d.params["foreground_layer"].value

        return cov_ss_dict

    def regress_cube_layers(self,
                            cube,
                            labels=None):
        ''' Performs simstack algorithm on layers contained in the cube.  The map and noisemap are the last two
        layers in the cube and are extracted before stacking.  LMFIT is used to perform the regresssion.

        :param cube: ndarray containing N-2 layers representing bins, a map layer, and a noisemap layer.
        :param labels: Labels for each layer in the stack.
        :return cov_ss_1d: lmfit object of the simstacked cube layers.
        '''

        # Extract Noise and Signal Maps from Cube (and then delete layers)
        ierr = cube[-1, :]
        cube = cube[:-1, :]
        imap = cube[-1, :]
        cube = cube[:-1, :]

        # Step backward through cube so removal of rows does not affect order
        fit_params = Parameters()
        for iarg in range(len(cube)):
            # Assign Parameter Labels
            if not labels:
                parameter_label = self.catalog_dict['tables']['parameter_labels'][iarg].replace('.', 'p')
            else:
                try:
                    parameter_label = labels[iarg].replace('.', 'p')
                except:
                    pdb.set_trace()
            # Add parameter
            fit_params.add(parameter_label, value=1e-3 * np.random.randn())

        cov_ss_1d = minimize(self.simultaneous_stack_array_oned, fit_params,
                             args=(np.ndarray.flatten(cube),),
                             kws={'data1d': np.ndarray.flatten(imap), 'err1d': np.ndarray.flatten(ierr)},
                             nan_policy='propagate')
        return cov_ss_1d

    def simultaneous_stack_array_oned(self,
                                      p,
                                      layers_1d,
                                      data1d,
                                      err1d=None,
                                      arg_order=None):
        ''' Function to Minimize written specifically for lmfit

        :param p: Parameters dictionary
        :param layers_1d: Cube layers flattened to 1d
        :param data1d: Map flattened to 1d
        :param err1d: Noise flattened to 1d
        :param arg_order: If forcing layers to correspond to labels
        :return: data-model/error, or data-model if err1d is None.
        '''

        v = p.valuesdict()

        len_model = len(data1d)
        nlayers = len(layers_1d) // len_model

        model = np.zeros(len_model)

        for i in range(nlayers):
            if arg_order != None:
                model[:] += layers_1d[i * len_model:(i + 1) * len_model] * v[arg_order[i]]
            else:
                model[:] += layers_1d[i * len_model:(i + 1) * len_model] * v[list(v.keys())[i]]

        # Subtract the mean of the layers after they've been summed together
        model -= np.mean(model)

        if (err1d is None) or 0 in err1d:
            return (data1d - model)
        else:
            return (data1d - model) / err1d

    def get_x_y_from_ra_dec(self,
                            wmap,
                            cms,
                            ind_src,
                            ra_series,
                            dec_series):
        ''' Get x and y positions from ra, dec, and header.

        :param wmap: astropy object
        :param cms: map dimensions
        :param ind_src: sources indicies
        :param ra_series: ra
        :param dec_series: dec
        :return: x, y
        '''

        ra = ra_series[ind_src].values
        dec = dec_series[ind_src].values
        # CONVERT FROM RA/DEC to X/Y
        ty, tx = wmap.wcs_world2pix(ra, dec, 0)
        # CHECK FOR SOURCES THAT FALL OUTSIDE MAP
        ind_keep = np.where((tx >= 0) & (np.round(tx) < cms[0]) & (ty >= 0) & (np.round(ty) < cms[1]))
        real_x = np.round(tx[ind_keep]).astype(int)
        real_y = np.round(ty[ind_keep]).astype(int)

        return real_x, real_y


