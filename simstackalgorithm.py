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
        #zbins = json.loads(json.loads(self.config_dict['catalog']['classification'])['redshift']['bins'])
        zbins = json.loads(self.config_dict['catalog']['classification']['redshift']['bins'])
        self.config_dict['distance_bins'] = {'redshift': zbins,
                                             'lookback_time': self.config_dict['cosmology_dict']['cosmology'].lookback_time(zbins)}

    def perform_simstack(self, add_background=False, crop_circles=True, stack_all_z_at_once=False, write_simmaps=False, bootstrap=0):
        '''
        perform_simstack takes the following steps:
        0. Get catalog and drop nans
        1. Assign parameter labels
        2. Call stack_in_wavelengths

        Following parameters are overwritten if included in config file.
        :param add_background: (bool) add additional background layer.
        :param crop_circles: (bool) exclude masked areas.
        :params stack_all_z_at_once: (bool) choose between stacking in redshift slices or all at once.
        '''
        if 'stack_all_z_at_once' not in self.config_dict['general']['binning']:
            self.config_dict['general']['binning']['stack_all_z_at_once'] = stack_all_z_at_once
        if 'crop_circles' not in self.config_dict['general']['binning']:
            self.config_dict['general']['binning']['crop_circles'] = crop_circles
        if 'add_background' not in self.config_dict['general']['binning']:
            self.config_dict['general']['binning']['add_background'] = add_background
        if 'write_simmaps' not in self.config_dict['general']['error_estimator']:
            self.config_dict['general']['error_estimator']['write_simmaps'] = write_simmaps
        stack_all_z_at_once = self.config_dict['general']['binning']['stack_all_z_at_once']
        crop_circles = self.config_dict['general']['binning']['crop_circles']
        add_background = self.config_dict['general']['binning']['add_background']
        write_simmaps = self.config_dict['general']['error_estimator']['write_simmaps']

        # Get catalog.  Clean NaNs
        catalog = self.catalog_dict['tables']['split_table'].dropna()

        # Get binning details
        #binning = self.config_dict['general']['binning']

        split_dict = self.config_dict['catalog']['classification']
        #split_type = split_dict.pop('split_type')
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
            #uncertainty_key = 'stacked_uncertainties'
        else:
            flux_density_key = 'bootstrap_flux_densities_'+str(bootstrap)
            #uncertainty_key = 'bootstrap_uncertainties_'+str(bootstrap)
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
                if add_background:
                    labels.append("ones_background")
                cov_ss_out = self.stack_in_wavelengths(catalog_in, labels=labels, distance_interval=distance_label,
                                          crop_circles=crop_circles, add_background=add_background, bootstrap=bootstrap)
                for wv in cov_ss_out:
                    if wv not in self.results_dict['band_results_dict']:
                        self.results_dict['band_results_dict'][wv] = {}
                    if flux_density_key not in self.results_dict['band_results_dict'][wv]:
                        self.results_dict['band_results_dict'][wv][flux_density_key] = cov_ss_out[wv].params
                        #self.results_dict['band_results_dict'][wv][uncertainty_key] = cov_ss_out[wv].params
                    else:
                        self.results_dict['band_results_dict'][wv][flux_density_key].update(cov_ss_out[wv].params)
                        #self.results_dict['band_results_dict'][wv][uncertainty_key].update(cov_ss_out[wv].params)
        else:
            labels = []
            for i in np.unique(catalog['redshift']):
                if bootstrap:
                    labels = self.split_bootstrap_labels(self.catalog_dict['tables']['parameter_labels'])
                else:
                    labels.extend(self.catalog_dict['tables']['parameter_labels'][int(i*nlayers):int((i+1)*nlayers)])
                distance_labels.append("_".join(["redshift", str(bins[int(i)]), str(bins[int(i) + 1])]).replace('.', 'p'))
            if add_background:
                labels.append("ones_background")
            cov_ss_out = self.stack_in_wavelengths(catalog, labels=labels, distance_interval='all_redshifts',
                                      crop_circles=crop_circles, add_background=add_background, bootstrap=bootstrap)

            for wv in cov_ss_out:
                if wv not in self.results_dict['band_results_dict']:
                    self.results_dict['band_results_dict'][wv] = {}
                self.results_dict['band_results_dict'][wv][flux_density_key] = cov_ss_out[wv].params
                #self.results_dict['band_results_dict'][wv][uncertainty_key] = cov_ss_out[wv].params

        self.config_dict['catalog']['distance_labels'] = distance_labels
        self.stack_successful = True

    def build_cube(self, map_dict, catalog, labels=None, add_background=False, crop_circles=False, bootstrap=False,
                   write_fits_layers=False):

        cmap = map_dict['map']
        cnoise = map_dict['noise']
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
        if (len(labels) - add_background) == np.prod(ds[1:]):
            nlists = ds[1:]
            llists = np.prod(nlists)
        elif (len(labels) - add_background)/2 == np.prod(ds[1:]):
            nlists = ds[1:]
            llists = 2 * np.prod(nlists)
        elif (len(labels) - add_background)/2 == np.prod(ds):
            nlists = ds
            llists = 2 * np.prod(nlists)
        else:
            nlists = ds
            llists = np.prod(nlists)

        if np.sum(cnoise) == 0: cnoise = cmap * 0.0 + 1.0

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
                                bt_split = 0.80
                                #jk_split = np.random.uniform(0.3, 0.7)
                                #print('jackknife split = ', jk_split)
                                left_x, right_x, left_y, right_y = train_test_split(real_x, real_y,
                                                                                    test_size=bt_split,
                                                                                    random_state=int(bootstrap),
                                                                                    shuffle=True)
                                layers[ilayer, left_x, left_y] += 1.0
                                layers[ilayer + 1, right_x, right_y] += 1.0
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
                        bt_split = 0.80
                        #jk_split = np.random.uniform(0.3, 0.7)
                        #print('jackknife split = ', jk_split)
                        left_x, right_x, left_y, right_y = train_test_split(real_x, real_y, test_size=bt_split,
                                                                            random_state=int(bootstrap),
                                                                            shuffle=True)
                        layers[ilayer, left_x, left_y] += 1.0
                        layers[ilayer + 1, right_x, right_y] += 1.0
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
        if add_background:
            cfits_maps = np.zeros([nlayers + 3, nhits])  # +3 to append background, cmap and cnoise
            trimmed_labels.append('background_layer')
        else:
            cfits_maps = np.zeros([nlayers + 2, nhits])  # +2 to append cmap and cnoise

        kern = self.gauss_kern(fwhm, np.floor(fwhm * 10) / pix, pix)
        for umap in range(nlayers):
            layer = layers[umap, :, :]
            tmap = self.smooth_psf(layer, kern)

            # write layers to fits files here
            if write_fits_layers:
                path_layer = r'D:\maps\cutouts\layers'
                name_layer = 'layer_'+str(umap)+'.fits'
                pdb.set_trace()
                hdu = fits.PrimaryHDU(tmap, header=hd)
                hdul = fits.HDUList([hdu])
                hdul.writeto(os.path.join(path_layer, name_layer))

            # Remove mean from map
            cfits_maps[umap, :] = tmap[ind_fit] - np.mean(tmap[ind_fit])

            # Add layer to write_simmaps cube
            if "convolved_layer_cube" in map_dict:
                map_dict["convolved_layer_cube"][umap, :, :] = tmap - np.mean(tmap[ind_fit])

            # If add_background=True, add background layer of ones.
        if add_background:
            cfits_maps[-3, :] = np.ones(np.shape(cmap[ind_fit]))

        # put map and noisemap in last two layers
        cfits_maps[-2, :] = cmap[ind_fit]
        cfits_maps[-1, :] = cnoise[ind_fit]

        return {'cube': cfits_maps, 'labels': trimmed_labels}

    def stack_in_wavelengths(self, catalog, labels=None, distance_interval=None, crop_circles=False,
                             add_background=False, bootstrap=False):

        map_keys = list(self.maps_dict.keys())
        cov_ss_dict = {}
        for wv in map_keys:
            map_dict = self.maps_dict[wv].copy()
            cube_dict = self.build_cube(map_dict, catalog.copy(), labels=labels, crop_circles=crop_circles,
                                        add_background=add_background, bootstrap=bootstrap)
            cube_labels = cube_dict['labels']
            print("Simultaneously Stacking {} Layers in {}".format(len(cube_labels), wv))
            cov_ss_1d = self.regress_cube_layers(cube_dict['cube'], labels=cube_dict['labels'])
            if 'stacked_flux_densities' not in map_dict:
                map_dict['stacked_flux_densities'] = {distance_interval: cov_ss_1d}
            else:
                map_dict['stacked_flux_densities'][distance_interval] = cov_ss_1d
            cov_ss_dict[wv] = cov_ss_1d

            # Write simulated maps from best-fits
            #pdb.set_trace()
            if self.config_dict["general"]["error_estimator"]["write_simmaps"]:
                for i, iparam_label in enumerate(cube_dict['labels']):
                    param_label = iparam_label.replace('.', 'p')
                    if 'background' not in iparam_label:
                        map_dict["convolved_layer_cube"][i, :, :] *= cov_ss_1d.params[param_label].value

                self.maps_dict[wv]["flattened_simmap"] = np.sum(map_dict["convolved_layer_cube"], axis=0)
                if 'background_layer' in cube_dict['labels']:
                    self.maps_dict[wv]["flattened_simmap"] += cov_ss_1d.params["background_layer"].value
        #pdb.set_trace()
        return cov_ss_dict

    def regress_cube_layers(self, cube, labels=None):

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
                parameter_label = labels[iarg].replace('.', 'p')
            # Add parameter
            fit_params.add(parameter_label, value=1e-3 * np.random.randn())

        cov_ss_1d = minimize(self.simultaneous_stack_array_oned, fit_params,
                             args=(np.ndarray.flatten(cube),),
                             kws={'data1d': np.ndarray.flatten(imap), 'err1d': np.ndarray.flatten(ierr)},
                             nan_policy='propagate')
        return cov_ss_1d

    def simultaneous_stack_array_oned(self, p, layers_1d, data1d, err1d=None, arg_order=None):
        ''' Function to Minimize written specifically for lmfit '''

        v = p.valuesdict()

        len_model = len(data1d)
        nlayers = len(layers_1d) // len_model

        model = np.zeros(len_model)

        for i in range(nlayers):
            if arg_order != None:
                model[:] += layers_1d[i * len_model:(i + 1) * len_model] * v[arg_order[i]]
            else:
                model[:] += layers_1d[i * len_model:(i + 1) * len_model] * v[list(v.keys())[i]]

        # Take the mean of the layers after they've been summed together
        model -= np.mean(model)

        if (err1d is None) or 0 in err1d:
            return (data1d - model)
        else:
            return (data1d - model) / err1d

    def get_x_y_from_ra_dec(self, wmap, cms, ind_src, ra_series, dec_series):

        ra = ra_series[ind_src].values
        dec = dec_series[ind_src].values
        # CONVERT FROM RA/DEC to X/Y
        ty, tx = wmap.wcs_world2pix(ra, dec, 0)
        # CHECK FOR SOURCES THAT FALL OUTSIDE MAP
        ind_keep = np.where((tx >= 0) & (np.round(tx) < cms[0]) & (ty >= 0) & (np.round(ty) < cms[1]))
        real_x = np.round(tx[ind_keep]).astype(int)
        real_y = np.round(ty[ind_keep]).astype(int)

        return real_x, real_y


