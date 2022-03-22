import pdb
import os
import shutil
import logging
import emcee
import scipy
from scipy.integrate import quad
from scipy import special
import decimal
import math
import numpy as np
from astropy.io import fits
from lmfit import Parameters, minimize, fit_report
from scipy.ndimage.filters import gaussian_filter
from scipy.optimize import curve_fit

pi = 3.141592653589793
L_sun = 3.839e26  # W
c = 299792458.0  # m/s
conv_lir_to_sfr = 1.728e-10 / 10 ** 0.23
conv_luv_to_sfr = 2.17e-10
a_nu_flux_to_mass = 6.7e19
#alpha_850 = a_nu_flux_to_mass
flux_to_specific_luminosity = 1.78  # 1e-23 #1.78e-13
h = 6.62607004e-34  # m2 kg / s  #4.13e-15 #eV/s
k = 1.38064852e-23  # m2 kg s-2 K-1 8.617e-5 #eV/K
sigma_upper_limit = 5

class SimstackCosmologyEstimators:

    def __init__(self):
        pass

    def log_likelihood(self, theta, x, y, cov):
        '''Log-likelihood for full covariance matrix '''
        y_model = self.graybody_fn(theta, x)
        delta_y = y - y_model[0]
        ll = -0.5 * (np.matmul(delta_y, np.matmul(np.linalg.inv(cov), delta_y)) +
                     len(y) * np.log(2 * np.pi) +
                     np.log(np.linalg.det(cov)))
        if not np.isfinite(ll):
            return -np.inf
        return ll

    def log_likelihood_slow(self, theta, x_d, y_d, cov_d, x_nd=None, y_nd=None, dy_nd=None, sigma_upper_limit=5):

        BETA_VALUE = 1.8
        ALPHA_VALUE = 2.0
        _sed_params = Parameters()
        _sed_params.add('A', value=theta[0], vary=True)
        _sed_params.add('T_observed', value=theta[1], vary=True)
        _sed_params.add('beta', value=BETA_VALUE, vary=False)
        _sed_params.add('alpha', value=ALPHA_VALUE, vary=False)

        # log likelihood for detections
        y_model_d = self.fast_sed(_sed_params, x_d)
        delta_y = y_d - y_model_d[0]
        ll_d = -0.5 * (np.matmul(delta_y, np.matmul(np.linalg.inv(cov_d), delta_y))
                       + len(y_d) * np.log(2 * np.pi)
                       + np.log(np.linalg.det(cov_d)))

        # log likelihood for non-detections
        ll_nd = 0.
        if x_nd is not None:
            y_model_nd = self.fast_sed(_sed_params, x_nd)
            for j, dy_nd_j in enumerate(dy_nd):
                _integrand_j = lambda yy: np.exp(decimal.Decimal(-0.5 * ((yy - y_model_nd[0][j]) ** 2 / dy_nd_j)))
                _ypts_j = np.array([_integrand_j(i) for i in np.linspace(0., np.sqrt(dy_nd_j) * sigma_upper_limit, 100)])
                _xpts_j = np.array([decimal.Decimal(i) for i in np.linspace(0., np.sqrt(dy_nd_j) * sigma_upper_limit, 100)])
                _integral_j = ((_ypts_j[1:] + _ypts_j[:-1]) * (_xpts_j[1:] - _xpts_j[:-1]) / decimal.Decimal(2)).sum()
                ll_nd += float(_integral_j.ln())
        else:
            pass

        if not np.isfinite(ll_d + ll_nd):
            return -np.inf

        return ll_d + ll_nd

    def log_likelihood_full(self, theta, x_d, y_d, cov_d, x_nd=None, y_nd=None, dy_nd=None,
                            beta_in=1.8, alpha_in=2.0, sigma_upper_limit=5):

        _sed_params = Parameters()
        _sed_params.add('A', value=theta[0], vary=True)
        _sed_params.add('T_observed', value=theta[1], vary=True)
        _sed_params.add('beta', value=beta_in, vary=False)
        _sed_params.add('alpha', value=alpha_in, vary=False)

        # log likelihood for detections
        y_model_d = self.fast_sed(_sed_params, x_d)
        delta_y = y_d - y_model_d[0]
        ll_d = -0.5 * (np.matmul(delta_y, np.matmul(np.linalg.inv(cov_d), delta_y))
                       + len(y_d) * np.log(2 * np.pi)
                       + np.log(np.linalg.det(cov_d)))

        # log likelihood for non-detections
        ll_nd = 0.
        if x_nd is not None:
            y_model_nd = self.fast_sed(_sed_params, x_nd)
            _integral_j = np.sqrt(np.pi / 2 * np.sqrt(dy_nd)) * (
                    special.erf((np.sqrt(dy_nd) * sigma_upper_limit - y_model_nd[0]) / np.sqrt(2 * dy_nd)) +
                    special.erf((y_model_nd[0]) / np.sqrt(2 * dy_nd)))
            ll_nd = np.sum(np.log(_integral_j))
        else:
            pass

        if not np.isfinite(ll_d + ll_nd):
            return -np.inf

        return ll_d + ll_nd

    def log_prior(self, theta):
        A, T = theta
        Amin = -42
        Amax = -33
        Tmin = 5
        Tmax = 32

        if Amin < A < Amax and Tmin < T < Tmax:
                return 0.0

        return -np.inf

    def log_prior_informative(self, theta, theta0):
        A, T = theta
        A0, T0, sigma_A, sigma_T = theta0
        Amin = -42
        Amax = -33
        Tmin = 5
        Tmax = 32
        error_infl = 1.0

        #if Amin < A < Amax and Tmin < T < Tmax and sigma_A is not None and sigma_T is not None:
        if Amin < A < Amax and Tmin < T < Tmax and sigma_T is not None:
            #lp = -0.5 * (np.sum((10**A - 10**A0) ** 2 / (10**sigma_A * error_infl) ** 2) +
            #             np.sum((T - T0) ** 2 / (sigma_T * error_infl) ** 2)) + \
            #     np.log(1.0/(np.sqrt(2*np.pi)*(10**sigma_A))) + np.log(1.0/(np.sqrt(2*np.pi)*sigma_T))
            lp = -0.5 * (np.sum((T - T0) ** 2 / (sigma_T * error_infl) ** 2)) + \
                 np.log(1.0/(np.sqrt(2*np.pi)*(sigma_T * error_infl) ** 2))
            return lp
        return -np.inf

    def log_probability(self, theta, x, y, yerr, theta0):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, x, y, yerr)

    def log_probability_slow(self, theta, x_d, y_d, cov_d, x_nd=None, y_nd=None, dy_nd=None, sigma_upper_limit=5):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_slow(theta, x_d, y_d, cov_d, x_nd, y_nd, dy_nd, sigma_upper_limit)

    def log_probability_full(self, theta, x_d, y_d, cov_d, x_nd, y_nd, dy_nd,
                             beta_in=1.8, alpha_in=2.0, sigma_upper_limit=5):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_full(theta, x_d, y_d, cov_d, x_nd, y_nd, dy_nd,
                                             beta_in, alpha_in, sigma_upper_limit)

    def log_probability_informative(self, theta, theta0, x_d, y_d, cov_d, x_nd, y_nd, dy_nd,
                                    beta_in=1.8, alpha_in=2.0, sigma_upper_limit=5):
        lp = self.log_prior_informative(theta, theta0)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood_full(theta, x_d, y_d, cov_d, x_nd, y_nd, dy_nd,
                                             beta_in, alpha_in, sigma_upper_limit)

    def mcmc_sed_estimator(self, x, y, yerr, theta, mcmc_iterations=2500, mcmc_discard=25):

        pos = np.array([theta[0], theta[1]]) + 1e-1 * np.random.randn(32, 2)
        nwalkers, ndim = pos.shape
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, self.log_probability, args=(x, y, yerr, theta)
        )
        sampler.run_mcmc(pos, mcmc_iterations, progress=True)
        flat_samples = sampler.get_chain(discard=mcmc_discard, thin=15, flat=True)

        return flat_samples

    def mcmc_sed_estimator_new(self, x, y, yerr, theta, mcmc_iterations=2500, mcmc_discard=25,
                               beta_in=1.8, alpha_in=2.0, sigma_upper_limit=5, slow=False, flat_prior=True):

        # Define non-detection as 1-sigma error below 0
        yerr_diag = np.diag(yerr)
        ind_nd = (y - np.sqrt(yerr_diag)) < 0

        # Split detections and non-detections (nd). Remove nd rows/columns from yerr matrix
        x = np.array(x)
        wvs = x[ind_nd == False]
        fluxes = y[ind_nd == False]
        cov_fluxes = yerr
        if np.sum(ind_nd):
            for i in np.where(ind_nd)[::-1]:
                cov_fluxes = np.delete(cov_fluxes, i, axis=0)
                cov_fluxes = np.delete(cov_fluxes, i, axis=1)

            wvs_nd = x[ind_nd == True]
            fluxes_nd = y[ind_nd == True]
            dfluxes_nd = yerr_diag[ind_nd == True]
        else:
            wvs_nd = None
            fluxes_nd = None
            dfluxes_nd = None

        pos = np.array([theta[0], theta[1]]) + 1e-1 * np.random.randn(32, 2)
        nwalkers, ndim = pos.shape

        if slow:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_probability_slow,
                                            args=(wvs, fluxes, cov_fluxes, wvs_nd, fluxes_nd, dfluxes_nd,
                                                  sigma_upper_limit))
        else:
            if flat_prior:
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, self.log_probability_full,
                    args=(wvs, fluxes, cov_fluxes, wvs_nd, fluxes_nd, dfluxes_nd,
                          beta_in, alpha_in, sigma_upper_limit))
            else:
                sampler = emcee.EnsembleSampler(
                    nwalkers, ndim, self.log_probability_informative,
                    args=(theta, wvs, fluxes, cov_fluxes, wvs_nd, fluxes_nd, dfluxes_nd,
                          beta_in, alpha_in, sigma_upper_limit))

        sampler.run_mcmc(pos, mcmc_iterations, progress=True)
        flat_samples = sampler.get_chain(discard=mcmc_discard, thin=15, flat=True)

        return flat_samples

    def loop_mcmc_sed_estimator(self, sed_bootstrap_dict, tables, mcmc_iterations=500, mcmc_discard=50,
                                slow=False, flat_prior=True, sigma_upper_limit=5):

        id_distance = self.config_dict['catalog']['classification']['redshift']['id']
        id_secondary = self.config_dict['catalog']['classification']['stellar_mass']['id']
        split_table = tables['split_table']
        full_table = tables['full_table']
        bin_keys = list(self.config_dict['parameter_names'].keys())

        wvs = sed_bootstrap_dict['wavelengths']
        mcmc_dict = {}
        y_dict = {}
        yerr_dict = {}
        z_dict = {}
        m_dict = {}
        ngals_dict = {}
        return_dict = {'wavelengths': wvs, 'z_median': z_dict, 'm_median': m_dict, 'ngals': ngals_dict,
                       'y': y_dict, 'yerr': yerr_dict,  'mcmc_dict': mcmc_dict}

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    ind_gals = (split_table.redshift == iz) & (split_table.stellar_mass == im) & (
                            split_table.split_params == ip)
                    y = sed_bootstrap_dict['sed_fluxes_dict'][id_label]
                    yerr = np.cov(sed_bootstrap_dict['sed_bootstrap_fluxes_dict'][id_label], rowvar=False)
                    y_dict[id_label] = y
                    yerr_dict[id_label] = yerr
                    z_median = np.median(full_table[id_distance][ind_gals])
                    m_median = np.median(full_table[id_secondary][ind_gals])
                    z_dict[id_label] = z_median
                    m_dict[id_label] = m_median
                    ngals_dict[id_label] = np.sum(ind_gals)

                    if type(flat_prior) is list:
                        if id_label in flat_prior:
                            flat_prior_in = False
                            print(id_label, ' informative prior')
                        else:
                            flat_prior_in = True
                    else:
                        flat_prior_in = flat_prior

                    mcmc_dict[id_label] = self.estimate_mcmc_sed(sed_bootstrap_dict, id_label,
                                                                 z_median=z_median,
                                                                 mcmc_iterations=mcmc_iterations,
                                                                 mcmc_discard=mcmc_discard,
                                                                 sigma_upper_limit=sigma_upper_limit,
                                                                 slow=slow, flat_prior=flat_prior_in)

        return return_dict

    def estimate_mcmc_sed(self, sed_bootstrap_dict, id_label, z_median=0,
                          mcmc_iterations=500, mcmc_discard=5, sigma_upper_limit=5, slow=False, flat_prior=True):

        if not z_median:
            z_label = id_label.split('_')[1:3]
            z_median = np.mean([float(i) for i in z_label])

        x = sed_bootstrap_dict['wavelengths']
        y = sed_bootstrap_dict['sed_fluxes_dict'][id_label]
        yerr = np.cov(sed_bootstrap_dict['sed_bootstrap_fluxes_dict'][id_label], rowvar=False)

        sed_params = self.fast_sed_fitter(x, y, yerr)
        #t_forced = 32.9 + 4.6 * (z_median - 2)
        #sed_forced_params = self.forced_sed_fitter(x, y, yerr, t_forced)
        Ain = sed_params['A'].value
        Aerr = sed_params['A'].stderr
        Tin = sed_params['T_observed'].value
        Terr = sed_params['T_observed'].stderr
        if Terr is None or Terr > Tin:
            Tin = (10 ** (1.4 + 0.075 * z_median)) / (1 + z_median)
            Terr = Tin
            Aerr = Ain
        else:
            Terr = np.min([np.max([Terr, 0.005 * Tin]), Tin])

        theta0 = Ain, Tin, Aerr, Terr

        if np.isfinite(np.log(np.linalg.det(yerr))):
            flat_samples = self.mcmc_sed_estimator_new(x, y, yerr, theta0, mcmc_iterations=mcmc_iterations,
                                                       mcmc_discard=mcmc_discard, sigma_upper_limit=sigma_upper_limit,
                                                       slow=slow, beta_in=1.8, alpha_in=2.0, flat_prior=flat_prior)
        else:
            return -np.inf

        return flat_samples

    def get_lir_from_mcmc_samples(self, mcmc_samples, percentiles=[16, 25, 32, 50, 68, 75, 84], min_detections=1):
        lir_dict = {}
        tobs_dict = {}
        mcmc_dict = {}
        bin_keys = list(self.config_dict['parameter_names'].keys())

        return_dict = {'lir_dict': lir_dict, 'Tobs_dict': tobs_dict, 'percentiles': percentiles,
                       'z_median': mcmc_samples['z_median'], 'm_median': mcmc_samples['m_median'],
                       'ngals': mcmc_samples['ngals'], 'mcmc_out': mcmc_dict}

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])

                    if type(mcmc_samples['mcmc_dict'][id_label]) is not float:
                        if np.sum((mcmc_samples['y'][id_label] - np.sqrt(np.diag(mcmc_samples['yerr'][id_label]))) > 0) >= min_detections:
                            mcmc_out = [np.percentile(mcmc_samples['mcmc_dict'][id_label][:, i], percentiles) for i in
                                        range(mcmc_samples['mcmc_dict'][id_label].shape[1])]
                            mcmc_dict[id_label] = mcmc_out
                            z_median = mcmc_samples['z_median'][id_label]

                            for i, vpercentile in enumerate(percentiles):
                                if id_label not in lir_dict:
                                    tobs_dict[id_label] = {str(vpercentile): mcmc_out[1][i]}
                                    lir_dict[id_label] = \
                                        {str(vpercentile):
                                            self.fast_LIR([mcmc_out[0][i], mcmc_out[1][i]], z_median)}

                                else:
                                    tobs_dict[id_label][str(vpercentile)] = mcmc_out[1][i]
                                    lir_dict[id_label][str(vpercentile)] = \
                                        self.fast_LIR([mcmc_out[0][i], mcmc_out[1][i]], z_median)

        return return_dict

    a_nu_flux_to_mass = 6.7e19

    def get_mmol_from_mcmc_samples(self, mcmc_samples, percentiles=[16, 25, 32, 50, 68, 75, 84], min_detections=1):
        lir_dict = {}
        sfr_dict = {}
        mmol_dict = {}
        tobs_dict = {}
        mcmc_dict = {}
        bin_keys = list(self.config_dict['parameter_names'].keys())

        return_dict = {'lir_dict': lir_dict, 'Tobs_dict': tobs_dict, 'sfr_dict': sfr_dict, 'Mmol_dict': mmol_dict,
                       'percentiles': percentiles,
                       'z_median': mcmc_samples['z_median'], 'm_median': mcmc_samples['m_median'],
                       'ngals': mcmc_samples['ngals'], 'mcmc_out': mcmc_dict}

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])

                    if type(mcmc_samples['mcmc_dict'][id_label]) is not float:
                        if np.sum((mcmc_samples['y'][id_label] - np.sqrt(
                                np.diag(mcmc_samples['yerr'][id_label]))) > 0) >= min_detections:
                            mcmc_out = [np.percentile(mcmc_samples['mcmc_dict'][id_label][:, i], percentiles) for i in
                                        range(mcmc_samples['mcmc_dict'][id_label].shape[1])]
                            mcmc_dict[id_label] = mcmc_out
                            z_median = mcmc_samples['z_median'][id_label]

                            for i, vpercentile in enumerate(percentiles):

                                rest_frame_850 = 850 * (1 + z_median)
                                flux_850 = self.graybody_fn([mcmc_out[0][i], mcmc_out[1][i]], [rest_frame_850])
                                L_850A = self.fast_L850(flux_850, z_median)
                                M_mol = L_850A / a_nu_flux_to_mass

                                # Scoville 2017, Eqn 2.
                                sfr = 35 * (M_mol / 1e10) ** (0.89) * ((1 + z_median) / 3) ** (0.95)

                                if id_label not in lir_dict:
                                    tobs_dict[id_label] = {str(vpercentile): mcmc_out[1][i]}
                                    sfr_dict[id_label] = {str(vpercentile): sfr}
                                    mmol_dict[id_label] = {str(vpercentile): M_mol}
                                    tobs_dict[id_label] = {str(vpercentile): mcmc_out[1][i]}

                                    lir_dict[id_label] = \
                                        {str(vpercentile):
                                             self.fast_LIR([mcmc_out[0][i], mcmc_out[1][i]], z_median)}

                                else:
                                    tobs_dict[id_label][str(vpercentile)] = mcmc_out[1][i]
                                    sfr_dict[id_label][str(vpercentile)] = sfr
                                    mmol_dict[id_label][str(vpercentile)] = M_mol
                                    lir_dict[id_label][str(vpercentile)] = \
                                        self.fast_LIR([mcmc_out[0][i], mcmc_out[1][i]], z_median)

        return return_dict

    def estimate_sfr_density(self, sfr_dict, effective_map_area):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        lird_dict = {}
        sfrd_dict = {}
        results_dict = {'sfrd_dict': sfrd_dict, 'lird_dict': lird_dict, 'effective_area': effective_map_area,
                        'z_median': sfr_dict['z_median'], 'm_median': sfr_dict['m_median'], 'ngals': sfr_dict['ngals']}

        for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                    id_label = "__".join([zlab, mlab, plab])

                    ngals = sfr_dict['ngals'][id_label]
                    zlo = float(zlab.split('_')[-2])
                    zhi = float(zlab.split('_')[-1])

                    # Get median redshift and stellar-mass and completeness-correct
                    #z_median = sfr_dict['z_median'][id_label]
                    #m_median = sfr_dict['m_median'][id_label]

                    if id_label in sfr_dict['sfr_dict']:
                        sfrd_dict[id_label] = \
                            self.estimate_lird(sfr_dict['sfr_dict'][id_label], ngals,
                                          effective_map_area, zlo, zhi, sfrd=True)  # , completeness=comp)
                        lird_dict[id_label] = \
                            self.estimate_lird(sfr_dict['lir_dict'][id_label], ngals,
                                               effective_map_area, zlo, zhi)  # , completeness=comp)
        return results_dict

    def estimate_luminosity_density(self, lir_dict, effective_map_area):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        lird_dict = {}
        results_dict = {'lird_dict': lird_dict, 'effective_area': effective_map_area,
                        'z_median': lir_dict['z_median'], 'm_median': lir_dict['m_median'], 'ngals': lir_dict['ngals']}

        for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                    id_label = "__".join([zlab, mlab, plab])

                    ngals = lir_dict['ngals'][id_label]
                    zlo = float(zlab.split('_')[-2])
                    zhi = float(zlab.split('_')[-1])

                    # Get median redshift and stellar-mass and completeness-correct
                    #z_median = lir_dict['z_median'][id_label]
                    #m_median = lir_dict['m_median'][id_label]

                    #qcomp = self.estimate_quadri_correction(z_median, m_median)
                    #comp = 1
                    #if (qcomp > 0.3) and (qcomp < 0.99):
                    #    comp = qcomp
                    #    print("z={:0.2f}, m={:0.2f} , {:0.2f}".format(z_median, m_median, comp))

                    if id_label in lir_dict['lir_dict']:
                        lird_dict[id_label] = \
                            self.estimate_lird(lir_dict['lir_dict'][id_label], ngals,
                                          effective_map_area, zlo, zhi)  # , completeness=comp)
        return results_dict

    def estimate_total_lird_array(self, lird_dict):
        conv_lir_to_sfr = 1.728e-10 / 10 ** 0.23

        ''' Estimate Weighted Errors'''
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
        lird_array_mid = np.zeros(ds)
        lird_array_err2 = np.zeros(ds)

        for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                    id_label = "__".join([zlab, mlab, plab])

                    if id_label in lird_dict['lird_dict']:
                        lird_array_mid[iz, im, ip] = lird_dict['lird_dict'][id_label]['lird']
                        lird_array_err2[iz, im, ip] = lird_dict['lird_dict'][id_label]['lird_err2']
                    #else:
                    #    print('excluding', id_label)

        lird_total = np.sum(lird_array_mid[:, :, 1], axis=1) + np.sum(lird_array_mid[:, :, 0], axis=1)
        lird_error = np.sqrt(np.sum(lird_array_err2[:, :, 1], axis=1) + np.sum(lird_array_err2[:, :, 0], axis=1))

        return {'lird_array': {'50': lird_array_mid, '32': lird_array_mid - np.sqrt(lird_array_err2),
                               '68': lird_array_mid + np.sqrt(lird_array_err2)},
                'lird_total': lird_total, 'lird_total_error': lird_error,
                'sfrd_total': conv_lir_to_sfr * lird_total, 'sfrd_total_error': conv_lir_to_sfr * lird_error}

    def estimate_total_sfrd_array(self, sfrd_dict):

        ''' Estimate Weighted Errors'''
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
        sfrd_array_mid = np.zeros(ds)
        sfrd_array_err2 = np.zeros(ds)
        lird_array_mid = np.zeros(ds)
        lird_array_err2 = np.zeros(ds)

        for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                    id_label = "__".join([zlab, mlab, plab])

                    if id_label in sfrd_dict['sfrd_dict']:
                        sfrd_array_mid[iz, im, ip] = sfrd_dict['sfrd_dict'][id_label]['sfrd']
                        sfrd_array_err2[iz, im, ip] = sfrd_dict['sfrd_dict'][id_label]['sfrd_err2']
                        lird_array_mid[iz, im, ip] = sfrd_dict['lird_dict'][id_label]['lird']
                        lird_array_err2[iz, im, ip] = sfrd_dict['lird_dict'][id_label]['lird_err2']

        sfrd_total = np.sum(sfrd_array_mid[:, :, 1], axis=1) + np.sum(sfrd_array_mid[:, :, 0], axis=1)
        sfrd_error = np.sqrt(np.sum(sfrd_array_err2[:, :, 1], axis=1) + np.sum(sfrd_array_err2[:, :, 0], axis=1))

        lird_total = np.sum(lird_array_mid[:, :, 1], axis=1) + np.sum(lird_array_mid[:, :, 0], axis=1)
        lird_error = np.sqrt(np.sum(lird_array_err2[:, :, 1], axis=1) + np.sum(lird_array_err2[:, :, 0], axis=1))

        return {'sfrd_array': {'50': sfrd_array_mid, '32': sfrd_array_mid - np.sqrt(sfrd_array_err2),
                               '68': sfrd_array_mid + np.sqrt(sfrd_array_err2)},
                'sfrd_total': sfrd_total, 'sfrd_total_error': sfrd_error,
                'lird_array': {'50': lird_array_mid, '32': lird_array_mid - np.sqrt(lird_array_err2),
                               '68': lird_array_mid + np.sqrt(lird_array_err2)},
                'lird_total': lird_total, 'lird_total_error': lird_error}

    def estimate_cib(self, sed_bootstrap_dict, area_deg2):

        bin_keys = list(self.config_dict['parameter_names'].keys())

        nuInu = {}
        wvs = sed_bootstrap_dict['wavelengths']
        cib_dict_out = {'wavelengths': wvs, 'nuInu': nuInu, 'ngals': sed_bootstrap_dict['ngals']}

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    if id_label in sed_bootstrap_dict['sed_fluxes_dict']:
                        y = sed_bootstrap_dict['sed_fluxes_dict'][id_label]
                        #yerr = np.cov(sed_bootstrap_dict['sed_bootstrap_fluxes_dict'][id_label], rowvar=False)
                        ngals = sed_bootstrap_dict['ngals'][id_label]
                        inuInu = self.estimate_nuInu(wvs, y, area_deg2, ngals, completeness=1)
                        nuInu[id_label] = np.max([inuInu, inuInu * 0], axis=0)

        return cib_dict_out

    #def estimate_lird(self, lir, ngals, area_deg2, zlo, zhi, completeness=1.0):
    #    vol = self.comoving_volume_given_area(area_deg2, zlo, zhi)
    #    return lir * 1e0 * ngals / vol.value / completeness

    def estimate_lird(self, lir, ngals, area_deg2, zlo, zhi, sfrd=False, completeness=1.0):
        vol = self.comoving_volume_given_area(area_deg2, zlo, zhi)
        lird = lir['50'] * ngals / vol.value / completeness

        cv = self.moster2011_cosmic_variance((zhi + zlo) / 2, zhi - zlo)

        dlir = (lir['68'] - lir['32']) / 2
        elird2 = ((ngals * dlir) ** 2 + (lir['50'] * ngals * cv) ** 2) / vol.value ** 2 / completeness ** 2
        if sfrd:
            lird_dict_out = {'sfrd': lird, 'sfrd_err2': elird2}
        else:
            lird_dict_out = {'lird': lird, 'lird_err2': elird2}

        return lird_dict_out

    def estimate_nuInu(self, wavelength_um, flux_Jy, area_deg2, ngals, completeness=1):
        area_sr = (area_deg2 / (180. / np.pi) ** 2.) / (4. * np.pi)
        return 1e-1 * flux_Jy * (self.lambda_to_ghz(wavelength_um) * 1e9) * 1e-26 * 1e9 / area_sr * ngals / completeness

