import pdb
import os
import shutil
import logging
import emcee
import numpy as np
from astropy.io import fits
from lmfit import Parameters, minimize, fit_report
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

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

class SimstackCosmologyEstimators:

    def __init__(self):
        pass

    def log_likelihood(self, theta, x, y, cov):
        y_model = self.graybody_fn(theta, x)
        delta_y = y - y_model[0]
        return -0.5 * (np.matmul(delta_y, np.matmul(np.linalg.inv(cov), delta_y)) + len(y) * np.log(2 * np.pi) + np.log(
            np.linalg.det(cov)))

    def log_prior(self, theta, theta0):
        A, T = theta
        A0, T0 = theta0
        sigma2_A = 1 # 0.25
        sigma2_T = 1
        Amin = -40  # -38
        Amax = -30  # -30
        Tmin = 1  # 2
        Tmax = 30  # 24

        if Amin < A < Amax and Tmin < T < Tmax:
            return -0.5 * (np.sum((A - A0) ** 2 / sigma2_A) + np.sum((T - T0) ** 2 / sigma2_T))

        return -np.inf

    def log_probability(self, theta, x, y, yerr, theta0):
        lp = self.log_prior(theta, theta0)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, yerr)

    def estimate_lird(self, lir, ngals, area_deg2, zlo, zhi, completeness=1.0):
        vol = self.comoving_volume_given_area(area_deg2, zlo, zhi)
        return lir * 1e0 * ngals / vol.value / completeness

    def estimate_nuInu(self, wavelength_um, flux_Jy, area_deg2, ngals, completeness=1):
        area_sr = (area_deg2 / (180. / np.pi) ** 2.) / (4. * np.pi)
        return 1e-1 * flux_Jy * (self.lambda_to_ghz(wavelength_um) * 1e9) * 1e-26 * 1e9 / area_sr * ngals / completeness

    def estimate_mcmc_seds(self, bootstrap_dict, split_table, mcmc_iterations=2500, mcmc_discard=25):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]

        wvs = bootstrap_dict['wavelengths']
        wv_array = self.loggen(8, 1000, 100)
        z_bins = np.unique(self.config_dict['distance_bins']['redshift'])
        z_mid = [(z_bins[i] + z_bins[i + 1]) / 2 for i in range(len(z_bins) - 1)]

        ngals = np.zeros(ds)
        t_obs = np.zeros(ds)
        lir_16 = np.zeros(ds)
        lir_25 = np.zeros(ds)
        lir_32 = np.zeros(ds)
        lir_50 = np.zeros(ds)
        lir_68 = np.zeros(ds)
        lir_75 = np.zeros(ds)
        lir_84 = np.zeros(ds)
        mcmc_dict = {}
        y_dict = {}
        yerr_dict = {}
        lir_dict = {'16': lir_16, '25': lir_25, '32': lir_32, '50': lir_50, '68': lir_68, '75': lir_75, '84': lir_84,
                    'Tobs': t_obs, 'mcmc_dict': mcmc_dict, 'y': y_dict, 'yerr': yerr_dict, 'redshift_bins': z_bins, 'ngals': ngals, 'wavelengths': wvs}

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    ngals[iz, im, ip] = np.sum((split_table.redshift == iz) & (split_table.stellar_mass == im) & (
                            split_table.split_params == ip))
                    x = wvs
                    if ip:
                        pop = 'sf'
                    else:
                        pop = 'qt'
                    y = bootstrap_dict['sed_dict'][pop]['sed_measurement'][:, iz, im]
                    yerr = np.cov(bootstrap_dict['sed_dict'][pop]['sed_bootstrap'][:, :, iz, im], rowvar=False)
                    y_dict["_".join([zlab, mlab, plab])] = y
                    yerr_dict["_".join([zlab, mlab, plab])] = yerr

                    sed_params = self.fast_sed_fitter(x, y, yerr)

                    Ain = np.log10(sed_params['A'].value)
                    Tin = sed_params['T_observed'].value
                    theta0 = Ain, Tin
                    t_obs[iz, im, ip] = Tin
                    pos = np.array([Ain, Tin]) + 1e-1 * np.random.randn(32, 2)
                    nwalkers, ndim = pos.shape

                    if np.sum(y):
                        sampler = emcee.EnsembleSampler(
                            nwalkers, ndim, self.log_probability, args=(x, y, yerr, theta0)
                        )
                        sampler.run_mcmc(pos, mcmc_iterations, progress=True)
                        flat_samples = sampler.get_chain(discard=mcmc_discard, thin=15, flat=True)
                        mcmc_out = [np.percentile(flat_samples[:, i], [16, 25, 32, 50, 68, 75, 84]) for i in
                                    range(ndim)]
                        t_obs[iz, im, ip] = mcmc_out[1][3]
                        mcmc_dict["_".join([zlab,mlab,plab])] = mcmc_out

                        lir_16[iz, im, ip] = np.log10(self.fast_LIR([mcmc_out[0][0], mcmc_out[1][0]], z_mid[iz]))
                        lir_25[iz, im, ip] = np.log10(self.fast_LIR([mcmc_out[0][1], mcmc_out[1][1]], z_mid[iz]))
                        lir_32[iz, im, ip] = np.log10(self.fast_LIR([mcmc_out[0][2], mcmc_out[1][2]], z_mid[iz]))
                        lir_50[iz, im, ip] = np.log10(self.fast_LIR([mcmc_out[0][3], mcmc_out[1][3]], z_mid[iz]))
                        lir_68[iz, im, ip] = np.log10(self.fast_LIR([mcmc_out[0][4], mcmc_out[1][4]], z_mid[iz]))
                        lir_75[iz, im, ip] = np.log10(self.fast_LIR([mcmc_out[0][5], mcmc_out[1][5]], z_mid[iz]))
                        lir_84[iz, im, ip] = np.log10(self.fast_LIR([mcmc_out[0][6], mcmc_out[1][6]], z_mid[iz]))

        return lir_dict

    def estimate_cib(self, area_deg2, bootstrap_dict=None, split_table=None):
        if split_table is None:
            split_table = self.catalog_dict['tables']['split_table']
        if bootstrap_dict is None:
            bootstrap_dict = self.results_dict['bootstrap_results_dict']

        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]

        wvs = bootstrap_dict['wavelengths']
        z_bins = np.unique(self.config_dict['distance_bins']['redshift'])
        z_mid = [(z_bins[i] + z_bins[i + 1]) / 2 for i in range(len(z_bins) - 1)]

        nuInu = np.zeros([len(wvs), *ds])
        cib_dict_out = {'nuInu': nuInu, 'redshift_bins': z_bins, 'wavelengths': wvs}

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):

                    ngals = np.sum((split_table.redshift == iz) & (split_table.stellar_mass == im) & (
                                split_table.split_params == ip))
                    x = wvs
                    if ip:
                        pop = 'sf'
                    else:
                        pop = 'qt'

                    y = bootstrap_dict['sed_dict'][pop]['sed_measurement'][:, iz, im]
                    yerr = np.cov(bootstrap_dict['sed_dict'][pop]['sed_bootstrap'][:, :, iz, im], rowvar=False)
                    nuInu[:, iz, im, ip] = self.estimate_nuInu(wvs, y, area_deg2, ngals, completeness=1)

        return cib_dict_out

    #def estimate_luminosity_density(self, effective_map_area, tables, lir_dict):
    def estimate_luminosity_density(self, lir_dict, tables, effective_map_area):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        split_table = tables['split_table']
        full_table = tables['full_table']
        ngals_array = np.zeros(np.shape(lir_dict['50']))
        uv_sfrd = np.zeros(np.shape(lir_dict['50']))
        lird_16 = np.zeros(np.shape(lir_dict['16']))
        lird_25 = np.zeros(np.shape(lir_dict['25']))
        lird_32 = np.zeros(np.shape(lir_dict['32']))
        lird_50 = np.zeros(np.shape(lir_dict['50']))
        lird_68 = np.zeros(np.shape(lir_dict['68']))
        lird_75 = np.zeros(np.shape(lir_dict['75']))
        lird_84 = np.zeros(np.shape(lir_dict['84']))
        lird_dict = {'16': lird_16, '25': lird_25, '32': lird_32, '50': lird_50, '68': lird_68, '75': lird_75,
                     '84': lird_84, 'uv_sfrd': uv_sfrd, 'redshift_bins': lir_dict['redshift_bins'],
                     'ngals': ngals_array, 'parameter_names': self.config_dict['parameter_names']}

        for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                    ind_gals = (split_table.redshift == iz) & (split_table.stellar_mass == im) & (
                                split_table.split_params == ip)
                    ngals = np.sum(ind_gals)
                    ngals_array[iz, im, ip] = ngals
                    zlo = float(zlab.split('_')[-2])
                    zhi = float(zlab.split('_')[-1])

                    # Get median redshift and stellar-mass and completeness-correct
                    zmed = np.median(full_table['lp_zBEST'][ind_gals])
                    mmed = np.median(full_table['lp_mass_med'][ind_gals])
                    qcomp = self.estimate_quadri_correction(zmed, mmed)
                    comp = 1
                    if (qcomp > 0.3) and (qcomp < 0.99):
                        comp = qcomp
                        print("z={:0.2f}, m={:0.2f} , {:0.2f}".format(zmed, mmed, comp))

                    lird_16[iz, im, ip] = np.log10(
                        self.estimate_lird(10 ** lir_dict['16'][iz, im, ip], ngals, effective_map_area, zlo, zhi,
                                      completeness=comp))
                    lird_25[iz, im, ip] = np.log10(
                        self.estimate_lird(10 ** lir_dict['25'][iz, im, ip], ngals, effective_map_area, zlo, zhi,
                                      completeness=comp))
                    lird_32[iz, im, ip] = np.log10(
                        self.estimate_lird(10 ** lir_dict['32'][iz, im, ip], ngals, effective_map_area, zlo, zhi,
                                      completeness=comp))
                    lird_50[iz, im, ip] = np.log10(
                        self.estimate_lird(10 ** lir_dict['50'][iz, im, ip], ngals, effective_map_area, zlo, zhi,
                                      completeness=comp))
                    lird_68[iz, im, ip] = np.log10(
                        self.estimate_lird(10 ** lir_dict['68'][iz, im, ip], ngals, effective_map_area, zlo, zhi,
                                      completeness=comp))
                    lird_75[iz, im, ip] = np.log10(
                        self.estimate_lird(10 ** lir_dict['75'][iz, im, ip], ngals, effective_map_area, zlo, zhi,
                                      completeness=comp))
                    lird_84[iz, im, ip] = np.log10(
                        self.estimate_lird(10 ** lir_dict['84'][iz, im, ip], ngals, effective_map_area, zlo, zhi,
                                      completeness=comp))
                    uv_sfrd[iz, im, ip] = np.log10(
                        self.estimate_lird(np.median(10 ** full_table['lp_SFR_best'][ind_gals]), ngals,
                                      effective_map_area, zlo, zhi, completeness=1))

        return lird_dict

    def estimate_total_lird(self, lird_dict, errors=('25', '75')):

        lird_total = np.sum(10 ** lird_dict['50'][:, :, 1], axis=1) + np.sum(10 ** lird_dict['50'][:, :, 0], axis=1)
        lird_error = np.sqrt(np.sum(
            (((10**lird_dict[errors[1]][:, :, 1] - 10**lird_dict[errors[0]][:, :, 1])) ** 2) * 10**lird_dict['50'][:, :, 1],
            axis=1) / np.sum(10 ** lird_dict['50'][:, :, 1], axis=1))
        #lird_error = np.sqrt(np.sum(((10 ** lird_dict[errors[1][:, :, 1] - 10 ** lird_dict[errors[0][:, :, 1]) ** 2), axis=1))

        sfrd_total = conv_lir_to_sfr * (np.sum(10**lird_dict['50'][:, :, 1], axis=1) + np.sum(10**lird_dict['50'][:, :, 0], axis=1))
        sfrd_error = np.sqrt(np.sum(
            ((conv_lir_to_sfr * (10**lird_dict[errors[1]][:, :, 1] - 10**lird_dict[errors[0]][:, :, 1])) ** 2) * 10**lird_dict['50'][:, :, 1], axis=1) / np.sum(10**lird_dict['50'][:, :, 1], axis=1))
        #sfrd_error = np.sqrt(np.sum(((conv_lir_to_sfr * (10 ** lird_dict[errors[1][:, :, 1] - 10 ** lird_dict[errors[0][:, :, 1])) ** 2), axis=1))

        uvsfr_total = np.sum(10 ** lird_dict['uv_sfrd'][:, :, 1], axis=1) + np.sum(10 ** lird_dict['uv_sfrd'][:, :, 0],
                                                                                   axis=1)
        return {'lird_total': lird_total, 'lird_total_error': lird_error, 'sfrd_total': sfrd_total,
                'sfrd_total_error': sfrd_error, 'uvsfr_total': uvsfr_total}
