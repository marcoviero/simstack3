import pdb
import os
import shutil
import logging
import pickle
import json
import numpy as np
from astropy.io import fits
from configparser import ConfigParser
from lmfit import Parameters, minimize, fit_report
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit
from simstackcosmologyestimators import SimstackCosmologyEstimators

pi = 3.141592653589793
L_sun = 3.839e26  # W
c = 299792458.0  # m/s
#conv_lir_to_sfr = 1.72e-10
conv_lir_to_sfr = 1.728e-10 / 10 ** 0.23
conv_luv_to_sfr = 2.17e-10
a_nu_flux_to_mass = 6.7e19
flux_to_specific_luminosity = 1.78  # 1e-23 #1.78e-13
h = 6.62607004e-34  # m2 kg / s  #4.13e-15 #eV/s
k = 1.38064852e-23  # m2 kg s-2 K-1 8.617e-5 #eV/K

class SimstackToolbox(SimstackCosmologyEstimators):

    def __init__(self):
        super().__init__()

    #def repopulate_self(self, imported_object):

    #    dict_list = dir(imported_object)
    #    for i in dict_list:
    #        if '__' not in i:
    #            setattr(self, i, getattr(imported_object, i))

    def combine_objects(self, second_object):

        wavelength_keys = list(self.results_dict['band_results_dict'].keys())
        wavelength_check = list(second_object.results_dict['band_results_dict'].keys())
        if wavelength_keys != wavelength_check:
            "Can't combine these objects. Missing bands"
            pdb.set_trace()

        label_dict = self.config_dict['parameter_names']
        label_dict_hi = second_object.config_dict['parameter_names']
        label_dict['redshift'].extend(label_dict_hi['redshift'])
        self.config_dict['catalog']['distance_labels'].extend(second_object.config_dict['catalog']['distance_labels'])
        self.config_dict['distance_bins']['redshift'].extend(second_object.config_dict['distance_bins']['redshift'])
        self.config_dict['distance_bins']['redshift'] = np.unique(
            self.config_dict['distance_bins']['redshift']).tolist()

        for k, key in enumerate(wavelength_keys):
            len_results_dict_keys = np.sum(
                ['flux_densities' in i for i in self.results_dict['band_results_dict'][key].keys()])
            for iboot in np.arange(len_results_dict_keys):
                if not iboot:
                    boot_label = 'stacked_flux_densities'
                else:
                    boot_label = 'bootstrap_flux_densities_' + str(int(iboot))

                self.results_dict['band_results_dict'][key][boot_label].update(
                    second_object.results_dict['band_results_dict'][key][boot_label])

    def construct_longname(self, basename):
        try:
            type_suffix = self.config_dict['catalog']['classification']['split_type']
        except:
            type_suffix = "nuvrj"

        dist_bins = json.loads(self.config_dict['catalog']['classification']['redshift']['bins'])
        dist_suffix = "_".join([str(i).replace('.', 'p') for i in dist_bins]).replace('p0_', '_')
        #dist_suffix = "_".join([str(len(dist_bins)-1), 'redshift_bins'])
        background_suffix = ''
        at_once_suffix = 'layers'
        catalog_suffix = ''
        bootstrap_suffix = ''
        stellar_mass_suffix = ''
        if 'stellar_mass' in self.config_dict['catalog']['classification']:
            mass_bins = json.loads(self.config_dict['catalog']['classification']['stellar_mass']['bins'])
            stellar_mass_suffix = "_".join(['X', str(len(mass_bins)-1)])
        if 'add_background' in self.config_dict['general']['binning']:
            if self.config_dict['general']['binning']['add_background']:
                background_suffix = 'background'
        if 'stack_all_z_at_once' in self.config_dict['general']['binning']:
            if self.config_dict['general']['binning']['stack_all_z_at_once']:
                at_once_suffix = 'atonce'
        if 'farmer' in self.config_dict['catalog']['file'].lower():
            catalog_suffix = 'farmer'
        elif 'classic' in self.config_dict['catalog']['file'].lower():
            catalog_suffix = 'classic'
        if 'bootstrap' in self.config_dict['general']['error_estimator']:
            if self.config_dict['general']['error_estimator']['bootstrap']['iterations']:
                first_boot = self.config_dict['general']['error_estimator']['bootstrap']['initial_bootstrap']
                last_boot = first_boot + self.config_dict['general']['error_estimator']['bootstrap']['iterations'] - 1
                bootstrap_suffix = "_".join(['bootstrap', "-".join([str(first_boot), str(last_boot)])])

        longname = "_".join([basename, type_suffix, dist_suffix, stellar_mass_suffix, at_once_suffix,
                             catalog_suffix, bootstrap_suffix])

        #pdb.set_trace()
        self.config_dict['io']['longname'] = longname
        return longname

    def copy_config_file(self, fp_in, overwrite_results=False):
        '''Copy Parameter File Right Away'''

        if 'shortname' in self.config_dict['io']:
            shortname = self.config_dict['io']['shortname']
        else:
            shortname = os.path.basename(fp_in).split('.')[0]

        longname = self.construct_longname(shortname)

        out_file_path = os.path.join(self.parse_path(self.config_dict['io']['output_folder']), longname)

        if not os.path.exists(out_file_path):
            os.makedirs(out_file_path)
        else:
            if not overwrite_results:
                while os.path.exists(out_file_path):
                    out_file_path = out_file_path + "_"
                os.makedirs(out_file_path)
        self.config_dict['io']['saved_data_path'] = out_file_path

        # Copy Config File
        fp_name = os.path.basename(fp_in)
        fp_out = os.path.join(out_file_path, fp_name)
        logging.info("Copying parameter file...")
        logging.info("  FROM : {}".format(fp_in))
        logging.info("    TO : {}".format(fp_out))
        logging.info("")
        shutil.copyfile(fp_in, fp_out)
        self.config_dict['io']['config_ini'] = fp_out

    def save_stacked_fluxes(self, drop_maps=True, drop_catalogs=False):
        if 'drop_maps' in self.config_dict['io']:
            drop_maps = self.config_dict['io']['drop_maps']
        if 'drop_catalogs' in self.config_dict['io']:
            drop_catalogs = self.config_dict['io']['drop_catalogs']

        longname = self.config_dict['io']['longname']
        out_file_path = self.config_dict['io']['saved_data_path']

        fpath = os.path.join(out_file_path, longname + '.pkl')

        print('pickling to ' + fpath)
        self.config_dict['pickles_path'] = fpath

        # Write simmaps
        if self.config_dict["general"]["error_estimator"]["write_simmaps"] == 1:
            for wv in self.maps_dict:
                name_simmap = wv + '_simmap.fits'
                hdu = fits.PrimaryHDU(self.maps_dict[wv]["flattened_simmap"], header=self.maps_dict[wv]["header"])
                hdul = fits.HDUList([hdu])
                hdul.writeto(os.path.join(out_file_path, name_simmap))
                # self.maps_dict[wv].pop("convolved_layer_cube")
                self.maps_dict[wv].pop("flattened_simmap")

        # Get rid of large files
        if drop_maps:
            print('Removing maps_dict')
            self.maps_dict = {}
        if drop_catalogs:
            print('Removing full_table from catalog_dict')
            self.catalog_dict = {}
            # self.catalog_dict['tables']['full_table'] = {}

        try:
            with open(fpath, "wb") as pickle_file_path:
                pickle.dump(self, pickle_file_path)
        except:
            print("Something wrong with save directory??")
            pdb.set_trace()

        return fpath

    @classmethod
    def import_saved_pickles(cls, pickle_fn):
        with open(pickle_fn, "rb") as file_path:
            encoding = pickle.load(file_path)
        return encoding

    @classmethod
    def save_to_pickles(cls, save_path, save_file):
        with open(save_path, "wb") as pickle_file_path:
            pickle.dump(save_file, pickle_file_path)

    def parse_path(self, path_in):
        # print(path_in)
        path_in = path_in.split(" ")
        if len(path_in) == 1:
            return path_in[0]
        else:
            path_env = os.environ[path_in[0]]
            if len(path_in) == 2:
                if 'nt' in os.name:
                    return path_env + os.path.join('\\', path_in[1].replace('/', '\\'))
                else:
                    return path_env + os.path.join('/', path_in[1])
            else:
                if 'nt' in os.name:
                    path_rename = [i.replace('/', '\\') for i in path_in[1:]]
                    return path_env + os.path.join('\\', *path_rename)
                else:
                    return path_env + os.path.join('/', *path_in[1:])

    def split_bootstrap_labels(self, labels):
        labels_out = []
        for ilabel in labels:
            if 'background' in ilabel:
                labels_out.append(ilabel)
            else:
                # labels_out.append(ilabel+'__bootstrap1')
                labels_out.append(ilabel)
                labels_out.append(ilabel + '__bootstrap2')
        # pdb.set_trace()
        return labels_out

    def get_params_dict(self, param_file_path):
        config = ConfigParser()
        config.read(param_file_path)

        dict_out = {}
        for section in config.sections():
            dict_sect = {}
            for (each_key, each_val) in config.items(section):
                # Remove quotations from dicts
                try:
                    dict_sect[each_key] = json.loads(each_val)
                except:
                    dict_sect[each_key] = each_val.replace("'", '"')

            dict_out[section] = dict_sect

        # Further remove quotations from embedded dicts
        for dkey in dict_out:
            for vkey in dict_out[dkey]:
                try:
                    dict_out[dkey][vkey] = json.loads(dict_out[dkey][vkey])
                except:
                    pass
        return dict_out

    def write_config_file(self, params_out, config_filename_out):
        config_out = ConfigParser()

        for ikey, idict in params_out.items():
            if not config_out.has_section(ikey):

                config_out.add_section(ikey)
                for isect, ivals in idict.items():
                    # pdb.set_trace()
                    # print(ikey, isect, ivals)
                    config_out.set(ikey, isect, str(ivals))

        # Write config_filename_out (check if overwriting externally)
        with open(config_filename_out, 'w') as conf:
            config_out.write(conf)

    def make_array_from_dict(self, input_dict, x=None):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
        if x is not None:
            array_out = np.zeros([len(x), *ds])
        else:
            array_out = np.zeros(ds)

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    label = "__".join([zlab, mlab, plab]).replace('.', 'p')
                    if id_label in input_dict:
                        if x is not None:
                            array_out[:, iz, im, ip] = input_dict[id_label]
                        else:
                            array_out[iz, im, ip] = input_dict[id_label]
                    elif label in input_dict:
                        if x is not None:
                            array_out[:, iz, im, ip] = input_dict[label]
                        else:
                            array_out[iz, im, ip] = input_dict[label]
        return array_out

    def get_fast_sed_dict(self, sed_bootstrap_dict):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        x = sed_bootstrap_dict['wavelengths']

        wv_array = self.loggen(8, 1000, 100)
        sed_params_dict = {}
        graybody_dict = {}
        lir_dict = {}
        return_dict = {'wv_array': wv_array, 'sed_params': sed_params_dict, 'graybody': graybody_dict, 'lir': lir_dict}
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    #label = "__".join([zlab, mlab, plab]).replace('.', 'p')

                    y = sed_bootstrap_dict['sed_fluxes_dict'][id_label]
                    yerr = np.cov(sed_bootstrap_dict['sed_bootstrap_fluxes_dict'][id_label], rowvar=False)
                    zin = sed_bootstrap_dict['z_median'][id_label]

                    sed_params_dict[id_label] = self.fast_sed_fitter(x, y, yerr)
                    graybody_dict[id_label] = self.fast_sed(sed_params_dict[id_label], wv_array)[0]
                    theta = sed_params_dict[id_label]['A'].value, sed_params_dict[id_label]['T_observed'].value
                    lir_dict[id_label] = self.fast_LIR(theta, zin)

        return return_dict

    def lambda_to_ghz(self, lam):
        c_light = 299792458.0  # m/s
        return np.array([1e-9 * c_light / (i * 1e-6) for i in lam])

    def graybody_fn(self, theta, x, alphain=2.0, betain=1.8):
        A, T = theta

        c_light = 299792458.0  # m/s

        nu_in = np.array([c_light * 1.e6 / wv for wv in x])
        ng = np.size(A)

        base = 2.0 * (6.626) ** (-2.0 - betain - alphain) * (1.38) ** (3. + betain + alphain) / (2.99792458) ** 2.0
        expo = 34.0 * (2.0 + betain + alphain) - 23.0 * (3.0 + betain + alphain) - 16.0 + 26.0
        K = base * 10.0 ** expo
        w_num = 10 ** A * K * (T * (3.0 + betain + alphain)) ** (3.0 + betain + alphain)
        w_den = (np.exp(3.0 + betain + alphain) - 1.0)
        w_div = w_num / w_den
        nu_cut = (3.0 + betain + alphain) * 0.208367e11 * T
        graybody = np.reshape(10 ** A, (ng, 1)) * nu_in ** np.reshape(betain, (ng, 1)) * self.black(nu_in, T) / 1000.0
        powerlaw = np.reshape(w_div, (ng, 1)) * nu_in ** np.reshape(-1.0 * alphain, (ng, 1))
        graybody[np.where(nu_in >= np.reshape(nu_cut, (ng, 1)))] = \
            powerlaw[np.where(nu_in >= np.reshape(nu_cut, (ng, 1)))]

        return graybody

    def fast_sed(self, m, wavelengths):

        v = m.valuesdict()
        A = np.asarray(v['A'])
        T = np.asarray(v['T_observed'])
        betain = np.asarray(v['beta'])
        alphain = np.asarray(v['alpha'])
        theta_in = A, T

        return self.graybody_fn(theta_in, wavelengths, alphain=alphain, betain=betain)

    def comoving_volume_given_area(self, area_deg2, zz1, zz2):
        vol0 = self.config_dict['cosmology_dict']['cosmology'].comoving_volume(zz2) - \
               self.config_dict['cosmology_dict']['cosmology'].comoving_volume(zz1)
        vol = (area_deg2 / (180. / np.pi) ** 2.) / (4. * np.pi) * vol0
        return vol

    def estimate_lird(self, lir, ngals, area_deg2, zlo, zhi, completeness=1.0):
        vol = self.comoving_volume_given_area(area_deg2, zlo, zhi)
        return lir * 1e0 * ngals / vol.value / completeness

    # From Weaver 2022
    def estimate_mlim_70(self, zin):
        return -1.51 * 1e6 * (1 + zin) + 6.81 * 1e7 * (1 + zin) ** 2

    #def weaver_completeness(self, z):
    #    return -3.55 * 1e8 * (1 + z) + 2.70 * 1e8 * (1 + z) ** 2.0

    def moster2011_cosmic_variance(self, z, dz=0.2, field='cosmos'):
        cv_params = {'cosmos': [0.069, -.234, 0.834], 'udf': [0.251, 0.364, 0.358]
            , 'goods': [0.261, 0.854, 0.684], 'gems': [0.161, 0.520, 0.729]
            , 'egs': [0.128, 0.383, 0.673]}

        field_params = cv_params[field]
        sigma_cv_ref = field_params[0] / (z ** field_params[2] + field_params[1])

        if dz == 0.2:
            sigma_cv = sigma_cv_ref
        else:
            sigma_cv = sigma_cv_ref * (dz / 0.2) ** (-0.5)

        return sigma_cv

    def estimate_quadri_correction(self, z, m):
        # uVista Completeness
        #p = np.array([20.00, 5.186, 6.389, 24.539])
        # this matches:  -3.55 * 1e8 * (1 + z) + 2.70 * 1e8 * (1 + z) ** 2.0
        #p = np.array([65.400, 5.186, 20, 25.539])
        # this matches Weaver2022: -1.51 * 1e6 * (1 + zin) + 6.81 * 1e7 * (1 + zin) ** 2
        p = np.array([160, 5.186, 39, 30])
        corr = 1 - 1 / (1 + np.exp(-p[1] * (z - p[0]) + p[2] * (-p[3] + m)))
        return corr

    def estimate_nuInu(self, wavelength_um, flux_Jy, area_deg2, ngals, completeness=1):
        area_sr = (area_deg2 / (180. / np.pi) ** 2.) / (4. * np.pi)
        return 1e-1 * flux_Jy * (self.lambda_to_ghz(wavelength_um) * 1e9) * 1e-26 * 1e9 / area_sr * ngals / completeness

    def fast_sed_fitter(self, wavelengths, fluxes, covar, betain=1.8, redshiftin=0):
        t_in = (10 ** (1.2 + 0.1 * redshiftin)) / (1+redshiftin)
        a_in = -34.0
        fit_params = Parameters()
        fit_params.add('A', value=a_in, vary=True)
        fit_params.add('T_observed', value=t_in, max=t_in*1.3, vary=True)
        fit_params.add('beta', value=betain, vary=False)
        fit_params.add('alpha', value=2.0, vary=False)

        # nu_in = c * 1.e6 / wavelengths
        fluxin = fluxes #[np.max([i, 1e-7]) for i in fluxes]
        #pdb.set_trace()
        try:
            sed_params = minimize(self.find_sed_min, fit_params,
                                  args=(wavelengths,),
                                  kws={'fluxes': fluxin, 'covar': covar})
            m = sed_params.params
        except:
            #pdb.set_trace()
            #print('fucked!')
            m = fit_params

        return m

    def find_sed_min(self, params, wavelengths, fluxes, covar=None):

        graybody = self.fast_sed(params, wavelengths)[0]
        delta_y = (fluxes - graybody)

        if (covar is None) or (np.sum(covar) == 0):
            return delta_y
        else:
            if np.shape(covar) == np.shape(fluxes):
                return delta_y ** 2 / covar
            else:
                #pdb.set_trace()
                #return np.matmul(delta_y, np.matmul(np.linalg.inv(covar), delta_y))
                return np.matmul(delta_y**2, np.linalg.inv(covar))
                #return delta_y ** 2 / np.diag(covar)

    def find_sed_min_old(self, params, wavelengths, fluxes, covar=None):

        graybody = self.fast_sed(params, wavelengths)[0]
        #pdb.set_trace()
        return (fluxes - graybody)

    def find_sed_min_add_noise(self, p, wavelengths, fluxes, covar=None):

        graybody = self.fast_sed(p, wavelengths)[0]

        if (covar is None) or (np.sum(covar) == 0):
            return (fluxes - graybody)
        else:
            #covar_use = np.min(np.array([np.sqrt(covar), np.sqrt(np.sqrt(fluxes**2))]), axis=0)
            return (fluxes + 1 * np.sqrt(covar) * np.random.randn(len(covar))) - graybody

    def fast_LIR(self, theta, zin):  # Tin,betain,alphain,z):
        '''This calls graybody_fn instead of fast_sed'''
        wavelength_range = self.loggen(8, 1000, 1000)
        model_sed = self.graybody_fn(theta, wavelength_range)

        nu_in = c * 1.e6 / wavelength_range
        dnu = nu_in[:-1] - nu_in[1:]
        dnu = np.append(dnu[0], dnu)
        Lir = np.sum(model_sed * dnu, axis=1)
        conversion = 4.0 * np.pi * (
                    1.0E-13 * self.config_dict['cosmology_dict']['cosmology'].luminosity_distance(
                zin) * 3.08568025E22) ** 2.0 / L_sun  # 4 * pi * D_L^2    units are L_sun/(Jy x Hz)

        Lrf = Lir * conversion  # Jy x Hz
        return Lrf.value[0]

    def fast_Lir(self, m, zin):  # Tin,betain,alphain,z):
        '''I dont know how to do this yet'''
        wavelength_range = self.loggen(8, 1000, 1000)
        model_sed = self.fast_sed(m, wavelength_range)

        nu_in = c * 1.e6 / wavelength_range
        # ns = len(nu_in)
        # dnu = nu_in[0:ns - 1] - nu_in[1:ns]
        dnu = nu_in[:-1] - nu_in[1:]
        dnu = np.append(dnu[0], dnu)
        Lir = np.sum(model_sed * dnu, axis=1)
        conversion = 4.0 * np.pi * (1.0E-13 * self.config_dict['cosmology_dict']['cosmology'].luminosity_distance(
            zin) * 3.08568025E22) ** 2.0 / L_sun  # 4 * pi * D_L^2    units are L_sun/(Jy x Hz)

        Lrf = Lir * conversion  # Jy x Hz
        return Lrf

    def black(self, nu_in, T):
        # h = 6.623e-34     ; Joule*s
        # k = 1.38e-23      ; Joule/K
        # c = 3e8           ; m/s
        # (2*h*nu_in^3/c^2)*(1/( exp(h*nu_in/k*T) - 1 )) * 10^29

        a0 = 1.4718e-21  # 2*h*10^29/c^2
        a1 = 4.7993e-11  # h/k

        num = a0 * nu_in ** 3.0
        den = np.exp(a1 * np.outer(1.0 / T, nu_in)) - 1.0
        ret = num / den

        return ret

    def clean_nans(self, dirty_array, replacement_char=0.0):
        clean_array = dirty_array.copy()
        clean_array[np.isnan(dirty_array)] = replacement_char
        clean_array[np.isinf(dirty_array)] = replacement_char

        return clean_array

    def gauss(self, x, x0, y0, sigma):
        p = [x0, y0, sigma]
        return p[1] * np.exp(-((x - p[0]) / p[2]) ** 2)

    def gauss_kern(self, fwhm, side, pixsize):
        ''' Create a 2D Gaussian (size= side x side)'''

        sig = fwhm / 2.355 / pixsize
        delt = np.zeros([int(side), int(side)])
        delt[0, 0] = 1.0
        ms = np.shape(delt)
        delt = self.shift_twod(delt, ms[0] / 2, ms[1] / 2)
        kern = delt
        gaussian_filter(delt, sig, output=kern)
        kern /= np.max(kern)

        return kern

    def shift_twod(self, seq, x, y):
        out = np.roll(np.roll(seq, int(x), axis=1), int(y), axis=0)
        return out

    def smooth_psf(self, mapin, psfin):

        s = np.shape(mapin)
        mnx = s[0]
        mny = s[1]

        s = np.shape(psfin)
        pnx = s[0]
        pny = s[1]

        psf_x0 = pnx / 2
        psf_y0 = pny / 2
        psf = psfin
        px0 = psf_x0
        py0 = psf_y0

        # pad psf
        psfpad = np.zeros([mnx, mny])
        psfpad[0:pnx, 0:pny] = psf

        # shift psf so that centre is at (0,0)
        psfpad = self.shift_twod(psfpad, -px0, -py0)
        smmap = np.real(np.fft.ifft2(np.fft.fft2(mapin) *
                                     np.fft.fft2(psfpad))
                        )

        return smmap

    def dist_idl(self, n1, m1=None):
        ''' Copy of IDL's dist.pro
        Create a rectangular array in which each element is
        proportinal to its frequency'''

        if m1 == None:
            m1 = int(n1)

        x = np.arange(float(n1))
        for i in range(len(x)): x[i] = min(x[i], (n1 - x[i])) ** 2.

        a = np.zeros([int(n1), int(m1)])

        i2 = m1 // 2 + 1

        for i in range(i2):
            y = np.sqrt(x + i ** 2.)
            a[:, i] = y
            if i != 0:
                a[:, m1 - i] = y

        return a

    def circle_mask(self, pixmap, radius_in, pixres):
        ''' Makes a 2D circular image of zeros and ones'''

        radius = radius_in / pixres
        xy = np.shape(pixmap)
        xx = xy[0]
        yy = xy[1]
        beforex = np.log2(xx)
        beforey = np.log2(yy)
        if beforex != beforey:
            if beforex > beforey:
                before = beforex
            else:
                before = beforey
        else:
            before = beforey
        l2 = np.ceil(before)
        pad_side = int(2.0 ** l2)
        outmap = np.zeros([pad_side, pad_side])
        outmap[:xx, :yy] = pixmap

        dist_array = self.shift_twod(self.dist_idl(pad_side, pad_side), pad_side / 2, pad_side / 2)
        circ = np.zeros([pad_side, pad_side])
        ind_one = np.where(dist_array <= radius)
        circ[ind_one] = 1.
        mask = np.real(np.fft.ifft2(np.fft.fft2(circ) *
                                    np.fft.fft2(outmap))
                       ) * pad_side * pad_side
        mask = np.round(mask)
        ind_holes = np.where(mask >= 1.0)
        mask = mask * 0.
        mask[ind_holes] = 1.
        maskout = self.shift_twod(mask, pad_side / 2, pad_side / 2)

        return maskout[:xx, :yy]

    def map_rms(self, map, mask=None):
        if mask != None:
            ind = np.where((mask == 1) & (self.clean_nans(map) != 0))
            print('using mask')
        else:
            ind = self.clean_nans(map) != 0
        map /= np.max(map)

        x0 = abs(np.percentile(map, 99))
        hist, bin_edges = np.histogram(np.unique(map), range=(-x0, x0), bins=30, density=True)

        p0 = [0., 1., x0 / 3]
        x = .5 * (bin_edges[:-1] + bin_edges[1:])
        x_peak = 1 + np.where((hist - max(hist)) ** 2 < 0.01)[0][0]

        # Fit the data with the function
        fit, tmp = curve_fit(self.gauss, x[:x_peak], hist[:x_peak] / max(hist), p0=p0)
        rms_1sig = abs(fit[2])

        return rms_1sig

    def leja_mass_function(self, z, Mass=np.linspace(9, 13, 100), sfg=2):
        # sfg = 0  -  Quiescent
        # sfg = 1  -  Star Forming
        # sfg = 2  -  All

        nz = np.shape(z)

        a1 = [-0.10, -0.97, -0.39]
        a2 = [-1.69, -1.58, -1.53]
        p1a = [-2.51, -2.88, -2.46]
        p1b = [-0.33, 0.11, 0.07]
        p1c = [-0.07, -0.31, -0.28]
        p2a = [-3.54, -3.48, -3.11]
        p2b = [-2.31, 0.07, -0.18]
        p2c = [0.73, -0.11, -0.03]
        ma = [10.70, 10.67, 10.72]
        mb = [0.00, -0.02, -0.13]
        mc = [0.00, 0.10, 0.11]

        aone = a1[sfg] + np.zeros(nz)
        atwo = a2[sfg] + np.zeros(nz)
        phione = 10 ** (p1a[sfg] + p1b[sfg] * z + p1c[sfg] * z ** 2)
        phitwo = 10 ** (p2a[sfg] + p2b[sfg] * z + p2c[sfg] * z ** 2)
        mstar = ma[sfg] + mb[sfg] * z + mc[sfg] * z ** 2

        # P[0]=alpha, P[1]=M*, P[2]=phi*, P[3]=alpha_2, P[4]=M*_2, P[5]=phi*_2
        P = np.array([aone, mstar, phione, atwo, mstar, phitwo])
        return self.dschecter(Mass, P)

    def schecter(self, X, P, exp=None, plaw=None):
        ''' X is alog10(M)
            P[0]=alpha, P[1]=M*, P[2]=phi*
            the output is in units of [Mpc^-3 dex^-1] ???
        '''
        if exp != None:
            return np.log(10.) * P[2] * np.exp(-10 ** (X - P[1]))
        if plaw != None:
            return np.log(10.) * P[2] * (10 ** ((X - P[1]) * (1 + P[0])))
        return np.log(10.) * P[2] * (10. ** ((X - P[1]) * (1.0 + P[0]))) * np.exp(-10. ** (X - P[1]))

    def dschecter(self, X, P):
        '''Fits a double Schechter function but using the same M*
           X is alog10(M)
           P[0]=alpha, P[1]=M*, P[2]=phi*, P[3]=alpha_2, P[4]=M*_2, P[5]=phi*_2
        '''
        rsch1 = np.log(10.) * P[2] * (10. ** ((X - P[1]) * (1 + P[0]))) * np.exp(-10. ** (X - P[1]))
        rsch2 = np.log(10.) * P[5] * (10. ** ((X - P[4]) * (1 + P[3]))) * np.exp(-10. ** (X - P[4]))

        return rsch1 + rsch2

    def loggen(self, minval, maxval, npoints, linear=None):
        points = np.arange(npoints) / (npoints - 1)
        if (linear != None):
            return (maxval - minval) * points + minval
        else:
            return 10.0 ** ((np.log10(maxval / minval)) * points + np.log10(minval))

    def L_fun(self, p, zed):
        '''Luminosities in log(L)'''
        v = p.valuesdict()
        lum = v["s0"] - (1. + (zed / v["zed0"]) ** (-1.0 * v["gamma"]))
        return lum

    def L_fit(self, p, zed, L, Lerr):
        '''Luminosities in log(L)'''
        lum = self.L_fun(p, zed)
        return (L - lum) / Lerr

    def viero_2013_luminosities(self, z, mass, sfg=1):
        import numpy as np
        y = np.array([[-7.2477881, 3.1599509, -0.13741485],
                      [-1.6335178, 0.33489572, -0.0091072162],
                      [-7.7579780, 1.3741780, -0.061809584]])
        ms = np.shape(y)
        npp = ms[0]
        nz = len(z)
        nm = len(mass)

        ex = np.zeros([nm, nz, npp])
        logl = np.zeros([nm, nz])

        for iz in range(nz):
            for im in range(nm):
                for ij in range(npp):
                    for ik in range(npp):
                        ex[im, iz, ij] += y[ij, ik] * mass[im] ** (ik)
                for ij in range(npp):
                    logl[im, iz] += ex[im, iz, ij] * z[iz] ** (ij)

        T_0 = 27.0
        z_T = 1.0
        epsilon_T = 0.4
        Tdust = T_0 * ((1 + np.array(z)) / (1.0 + z_T)) ** (epsilon_T)

        return [logl, Tdust]

    def viero_2013_luminosities(z, mass, sfg=1):
        import numpy as np
        y = np.array([[-7.2477881, 3.1599509, -0.13741485],
                      [-1.6335178, 0.33489572, -0.0091072162],
                      [-7.7579780, 1.3741780, -0.061809584]])
        ms = np.shape(y)
        npp = ms[0]
        nz = len(z)
        nm = len(mass)

        ex = np.zeros([nm, nz, npp])
        logl = np.zeros([nm, nz])

        for iz in range(nz):
            for im in range(nm):
                for ij in range(npp):
                    for ik in range(npp):
                        ex[im, iz, ij] += y[ij, ik] * mass[im] ** (ik)
                for ij in range(npp):
                    logl[im, iz] += ex[im, iz, ij] * z[iz] ** (ij)

        T_0 = 27.0
        z_T = 1.0
        epsilon_T = 0.4
        Tdust = T_0 * ((1 + np.array(z)) / (1.0 + z_T)) ** (epsilon_T)

        return [logl, Tdust]
