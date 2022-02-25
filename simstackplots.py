import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from simstacktoolbox import SimstackToolbox
conv_lir_to_sfr = 1.728e-10 / 10 ** 0.23

class SimstackPlots(SimstackToolbox):

    def __init__(self, SimstackPlotsObject):
        super().__init__()

        dict_list = dir(SimstackPlotsObject)
        for i in dict_list:
            if '__' not in i:
                setattr(self, i, getattr(SimstackPlotsObject, i))

    def plot_seds(self):
        colors = ['y', 'c', 'b', 'r', 'g']
        if self.results_dict['SED_df']['plot_sed']:
            zlen = len(self.results_dict['SED_df']['flux_density'])
            if len(self.config_dict['parameter_names']) == 3:
                zlen = len(self.results_dict['SED_df']['flux_density'])
                plen = 2
                fig, axs = plt.subplots(plen, zlen, figsize=(36, 10))
                for iz, zlab in enumerate(self.results_dict['SED_df']['flux_density']):
                    zmid = 0.5 * np.sum([float(i) for i in zlab.split('_')[-2:]])
                    for ip, plab in enumerate(self.results_dict['SED_df']['flux_density'][zlab]):
                        sed = self.results_dict['SED_df']['flux_density'][zlab][plab]
                        std = self.results_dict['SED_df']['std_error'][zlab][plab]
                        for im, mlab in enumerate(sed):
                            axs[ip, iz].scatter(sed.index, sed[mlab], color=colors[im])
                            axs[ip, iz].errorbar(sed.index, sed[mlab], std[mlab], 0, 'none', color=colors[im])
                            # axs[p, z].plot(sed.index, sed[mlab], label=mlab)

                            # pdb.set_trace()
                            LIR = self.results_dict['SED_df']['LIR'][zlab][plab][mlab][0]
                            sed_params = self.results_dict['SED_df']['SED'][zlab][plab][mlab]
                            # print(zlab, plab)
                            # print(sed_params)
                            T_obs = sed_params['T_observed'].value
                            T_rf = T_obs * (1 + zmid)
                            wv_array = self.loggen(8, 1000, 100)
                            sed_array = self.fast_sed(sed_params, wv_array)

                            line_label = ['-'.join(mlab.split('_')[-2:]), "Trf={:.1f}".format(T_rf),
                                          "LIR={:.1f}".format(np.log10(LIR))]
                            if LIR > 0:
                                axs[ip, iz].plot(wv_array, sed_array[0], label=line_label, color=colors[im])
                                axs[ip, iz].legend(loc='upper right')
                            else:
                                axs[ip, iz].plot(wv_array, sed_array[0], color=colors[im])

                            if not ip:
                                axs[ip, iz].set_title(zlab)
                            axs[ip, iz].set_xscale('log')
                            axs[ip, iz].set_yscale('log')
                            axs[ip, iz].set_xlim([10, 1000])
                            axs[ip, iz].set_ylim([1e-5, 5e-1])

            else:
                fig, axs = plt.subplots(1, zlen, figsize=(36, 10))
                for iz, zlab in enumerate(self.results_dict['SED_df']['flux_density']):
                    zmid = 0.5 * np.sum([float(i) for i in zlab.split('_')[-2:]])
                    for im, mlab in enumerate(self.results_dict['SED_df']['flux_density'][zlab]):
                        sed = self.results_dict['SED_df']['flux_density'][zlab][mlab]
                        std = self.results_dict['SED_df']['std_error'][zlab][mlab]
                        axs[iz].scatter(sed.index, sed.values)

                        LIR = self.results_dict['SED_df']['LIR'][zlab][mlab][0]
                        sed_params = self.results_dict['SED_df']['SED'][zlab][mlab]
                        # print(zlab, plab)
                        # print(sed_params)
                        T_obs = sed_params['T_observed'].value
                        T_rf = T_obs * (1 + zmid)
                        wv_array = self.loggen(8, 1000, 100)
                        sed_array = self.fast_sed(sed_params, wv_array)

                        line_label = ['-'.join(mlab.split('_')[-2:]), "Trf={:.1f}".format(T_rf),
                                      "LIR={:.1f}".format(np.log10(LIR))]
                        if LIR > 0:
                            axs[iz].plot(wv_array, sed_array[0], label=line_label)
                            axs[iz].legend(loc='upper right')
                        else:
                            axs[iz].plot(wv_array, sed_array[0])

                        if not im:
                            axs[iz].set_title(zlab)
                        axs[iz].set_xscale('log')
                        axs[iz].set_yscale('log')
                        axs[iz].set_xlim([10, 1000])
                        axs[iz].set_ylim([1e-5, 5e-1])
                # pdb.set_trace()
        else:
            print("Skipping SED plotting because only single wavelength measured.")

    def plot_flux_densities(self):
        wv_keys = list(self.results_dict['band_results_dict'].keys())
        wlen = len(wv_keys)
        if len(self.config_dict['parameter_names']) == 3:
            plen = 2
            fig, axs = plt.subplots(plen, wlen, figsize=(22, 10))
            for iwv, wlab in enumerate(wv_keys):
                for ip, plab in enumerate(self.results_dict['band_results_dict'][wlab]['raw_fluxes_dict']['flux_df']):
                    flux_df = self.results_dict['band_results_dict'][wlab]['raw_fluxes_dict']['flux_df'][plab]
                    error_df = self.results_dict['band_results_dict'][wlab]['raw_fluxes_dict']['error_df'][plab]
                    for mlab in flux_df:
                        if wlen > 1:
                            axs[ip, iwv].scatter(flux_df[mlab].index, flux_df[mlab].values * 1e3)
                            axs[ip, iwv].errorbar(flux_df[mlab].index, flux_df[mlab].values * 1e3,
                                                  error_df[mlab].values * 1e3,
                                                  label=mlab)
                            if not ip:
                                axs[ip, iwv].set_title(wlab)
                            else:
                                axs[ip, iwv].set_xlabel('redshift')
                            if not iwv:
                                axs[ip, iwv].set_ylabel('flux density (Jy)')
                            axs[ip, iwv].set_yscale('log')
                            # axs[ip, iwv].set_xlim([0., 8])
                            axs[ip, iwv].set_ylim([1e-3, 5e1])
                            if (ip == 1) & (iwv == 0):
                                axs[ip, iwv].legend(loc='upper right')
                        else:
                            axs[ip].scatter(flux_df[mlab].index, flux_df[mlab].values * 1e3)
                            axs[ip].errorbar(flux_df[mlab].index, flux_df[mlab].values * 1e3,
                                             error_df[mlab].values * 1e3,
                                             label=mlab)
                            if not ip:
                                axs[ip].set_title(wlab)
                            else:
                                axs[ip].set_xlabel('redshift')
                            if not iwv:
                                axs[ip].set_ylabel('flux density (Jy)')
                            axs[ip].set_yscale('log')
                            # axs[ip].set_xlim([0., 8])
                            axs[ip].set_ylim([1e-3, 5e1])
                            if (ip == 1):
                                axs[ip].legend(loc='upper right')
        else:
            plen = 1
            fig, axs = plt.subplots(plen, wlen, figsize=(22, 10))
            for iwv, wlab in enumerate(wv_keys):
                flux_df = self.results_dict['band_results_dict'][wlab]['raw_fluxes_dict']['flux_df']
                error_df = self.results_dict['band_results_dict'][wlab]['raw_fluxes_dict']['error_df']
                for mlab in flux_df:
                    # pdb.set_trace()
                    axs[iwv].scatter(flux_df[mlab].index, flux_df[mlab].values * 1e3)
                    axs[iwv].errorbar(flux_df[mlab].index, flux_df[mlab].values * 1e3, error_df[mlab].values * 1e3,
                                      label=mlab)
                    axs[iwv].set_title(wlab)
                    axs[iwv].set_xlabel('redshift')
                    if not iwv:
                        axs[iwv].set_ylabel('flux density (Jy)')
                    axs[iwv].set_yscale('log')
                    axs[iwv].set_ylim([1e-3, 5e1])
                    if (iwv == 0):
                        axs[iwv].legend(loc='upper right')

    def plot_lir_vs_z(self):
        if self.results_dict['SED_df']['plot_sed']:

            size_lir_vs_z_array = [len(self.config_dict['parameter_names'][i]) for i in
                                   self.config_dict['parameter_names']]
            sed_lir_vs_z_array = np.zeros(size_lir_vs_z_array)
            z_data_array = np.zeros(size_lir_vs_z_array[0])
            zlen = 1
            plen = len(size_lir_vs_z_array) - 1

            # Rearrange data into arrays
            for z, zlab in enumerate(self.results_dict['SED_df']['flux_density']):
                zmid = 0.5 * np.sum([float(i) for i in zlab.split('_')[-2:]])
                if plen == 2:
                    for p, plab in enumerate(self.results_dict['SED_df']['flux_density'][zlab]):
                        sed = self.results_dict['SED_df']['flux_density'][zlab][plab]
                        std = self.results_dict['SED_df']['std_error'][zlab][plab]
                        for im, mlab in enumerate(sed):
                            LIR = self.results_dict['SED_df']['LIR'][zlab][plab][mlab][0]
                            sed_params = self.results_dict['SED_df']['SED'][zlab][plab][mlab]
                            z_data_array[z] = zmid
                            sed_lir_vs_z_array[z, im, p] = np.log10(LIR)
                else:
                    sed = self.results_dict['SED_df']['flux_density'][zlab]
                    std = self.results_dict['SED_df']['std_error'][zlab]
                    for im, mlab in enumerate(sed):
                        LIR = self.results_dict['SED_df']['LIR'][zlab][mlab][0]
                        sed_params = self.results_dict['SED_df']['SED'][zlab][mlab]
                        z_data_array[z] = zmid
                        sed_lir_vs_z_array[z, im] = np.log10(LIR)

            # Plot LIR vs z
            keys = list(self.config_dict['parameter_names'])
            fig, axs = plt.subplots(1, plen, figsize=(36, 10))
            for im, mlab in enumerate(self.config_dict['parameter_names'][keys[1]]):
                if plen == 2:
                    for ip, plab in enumerate(self.config_dict['parameter_names'][keys[2]]):
                        axs[ip].scatter(z_data_array, sed_lir_vs_z_array[:, im, ip])
                        axs[ip].plot(z_data_array, sed_lir_vs_z_array[:, im, ip], label=mlab)
                        axs[ip].set_ylabel('LIR (M_sun)')
                        axs[ip].set_xlabel('redshift')
                        axs[ip].set_ylim([9, 13.5])
                        axs[ip].set_title(plab)
                else:
                    axs.scatter(z_data_array, sed_lir_vs_z_array[:, im])
                    axs.plot(z_data_array, sed_lir_vs_z_array[:, im], label=mlab)
                    axs.set_ylabel('LIR (M_sun)')
                    axs.set_xlabel('redshift')
                    axs.set_ylim([9, 13.5])
        else:
            print("Skipping SED plotting because only single wavelength measured.")

    def plot_rest_frame_temperature(self, tables, lir_in, ylim=[1e0, 2e2], fit_p=[1.3, 0.1]):
        full_table = tables['full_table']
        split_table = tables['split_table']

        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]

        sm = np.zeros(ds)
        zmed = np.zeros(ds)
        t_obs = np.zeros(ds)
        t_rf = np.zeros(ds)
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    if ip:
                        ind_gals = ((split_table.redshift == iz) & (split_table.stellar_mass == im) & (
                                split_table.split_params == ip))
                        t_obs[iz, im, ip] = lir_in['Tobs'][iz, im, ip]
                        t_rf[iz, im, ip] = lir_in['Tobs'][iz, im, ip] * (1 + np.median(
                            full_table['lp_zBEST'].loc[ind_gals]))
                        sm[iz, im, ip] = np.mean(
                            full_table['lp_mass_med'].loc[ind_gals])
                        zmed[iz, im, ip] = np.mean(
                            full_table['lp_zBEST'].loc[ind_gals])

        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
            label = "n=" + '-'.join(mlab.split('_')[1:])
            axs.plot(zmed[:, im, 1], (t_rf[:, im, 1]), ":o", label=label)
        z_in = np.linspace(0, zmed[-1, 0, 1])
        fit_label = "Tprior = 10^({0:.1f} + {1:.1f}z)".format(fit_p)
        axs.plot(z_in, (10 ** (fit_p[0] + fit_p[1] * z_in)), '--', label=fit_label)
        axs.plot(z_in, (1 + z_in) * 2.73, '--', label='CMB')
        # axs.set_xscale('log')
        axs.set_yscale('log')
        axs.set_ylim(ylim)
        axs.set_xlabel('redshift')
        axs.set_ylabel('T_restframe')
        axs.legend(loc='upper left')

    def plot_star_forming_main_sequence(self, tables, lir_in, ylim=[1e-1, 1e4]):
        colors = list(mcolors.CSS4_COLORS.keys())
        full_table = tables['full_table']
        split_table = tables['split_table']
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]

        sfr = np.zeros(ds)
        sm = np.zeros(ds)
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    if ip:
                        ind_gals = ((split_table.redshift == iz) & (split_table.stellar_mass == im) & (
                                split_table.split_params == ip))
                        sfr[iz, im, ip] = conv_lir_to_sfr * 10 ** lir_in['50'][iz, im, ip]
                        sm[iz, im, ip] = np.mean(
                            full_table['lp_mass_med'].loc[ind_gals])

        fig, axs = plt.subplots(1, 1, figsize=(8, 6))
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            label = "z=" + '-'.join(zlab.split('_')[1:])
            axs.plot(sm[iz, :, 1], (sfr[iz, :, 1]), ":o", color=colors[iz], label=label)
        # axs.set_xscale('log')
        axs.set_yscale('log')
        axs.set_ylim(ylim)
        axs.set_xlabel('Stellar Mass [Mstar]')
        axs.set_ylabel('SFR [Mstar/yr]')
        axs.legend(loc='lower right')

    def plot_cib(self, cib_dict=None, tables=None, area_deg2=None, zbins=[0, 1, 2, 3, 4, 6, 9]):

        if not cib_dict:
            if 'cib_dict' not in self.results_dict:
                self.results_dict['cib_dict'] = self.estimate_cib(area_deg2, tables)
            cib_dict = self.results_dict['cib_dict']

        wvs = cib_dict['wavelengths']
        nuInu_dict = cib_dict['nuInu']
        nuInu = self.make_array_from_dict(nuInu_dict, x=wvs)
        bin_keys = list(self.config_dict['parameter_names'].keys())

        fig, axs = plt.subplots(1, 2, figsize=(18, 7))
        ls = [':', '-']

        for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
            izb = 1
            nuInuz = 0 * np.sum(nuInu[:, 0, :, ip], axis=1)
            for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                zhi = float(zlab.split('_')[-1])
                # print(zhi, zbins[izb])
                if zhi > zbins[izb]:
                    # pdb.set_trace()
                    zlabel = "-".join([str(zbins[izb - 1]), str(zbins[izb])])
                    axs[0].plot(wvs, nuInuz, ls[ip], label=zlabel)
                    axs[0].set_xscale('log')
                    axs[0].set_yscale('log')
                    axs[0].set_ylim([1e-3, 1e2])
                    nuInuz = np.sum(nuInu[:, iz, :, ip], axis=1)
                    izb += 1
                else:
                    nuInuz += np.sum(nuInu[:, iz, :, ip], axis=1)

            zlabel = "z > {0:.1f}".format(zbins[izb - 1])
            axs[0].plot(wvs, nuInuz, ls[ip], label=zlabel)

        axs[0].set_title('CIB by Redshift Contribution')
        axs[0].set_xlabel('wavelength [um]')
        axs[0].set_ylabel('nuInu [nW/m^2/sr]')
        axs[0].plot(wvs, np.sum(np.sum(nuInu[:, :, :, ip], axis=1), axis=1), ls[ip], label='Total', lw=3)
        axs[0].legend(loc='lower left')

        for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
            for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                axs[1].plot(wvs, np.sum(nuInu[:, :, im, ip], axis=1), ls[ip], label=mlab)
                axs[1].set_xscale('log')
                axs[1].set_yscale('log')
                axs[1].set_ylim([1e-3, 1e2])

        axs[1].set_title('CIB by Stellar-Mass Contribution')
        axs[1].set_xlabel('wavelength [um]')
        axs[1].set_ylabel('nuInu [nW/m^2/sr]')
        axs[1].plot(wvs, np.sum(np.sum(nuInu[:, :, :, ip], axis=1), axis=1), ls[ip], label='Total', lw=3)
        axs[1].legend(loc='lower left')

    def plot_cib_layers(self, cib_dict=None, tables=None, area_deg2=None, show_total=False, zbins=[0, 1, 2, 3, 4, 6, 9]):

        if not cib_dict:
            if 'cib_dict' not in self.results_dict:
                self.results_dict['cib_dict'] = self.estimate_cib(area_deg2, tables)
            cib_dict = self.results_dict['cib_dict']

        wvs = cib_dict['wavelengths']
        nuInu_dict = cib_dict['nuInu']
        nuInu = self.make_array_from_dict(nuInu_dict, x=wvs)
        bin_keys = list(self.config_dict['parameter_names'].keys())

        fig, axs = plt.subplots(1, 2, figsize=(18, 7))
        ls = [':', '-']

        for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
            izb = 1
            nuInuz = 0 * np.sum(nuInu[:, 0, :, ip], axis=1)
            nuInuzL = 0 * np.sum(nuInu[:, 0, :, ip], axis=1)

            for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                zhi = float(zlab.split('_')[-1])
                # print(zhi, zbins[izb])
                if zhi > zbins[izb]:
                    # pdb.set_trace()
                    nuInuzL += nuInuz
                    zlabel = "-".join([str(zbins[izb - 1]), str(zbins[izb])])
                    zlabel = "z < {0:.1f}".format(zbins[izb])
                    axs[0].plot(wvs, nuInuzL, ls[ip], label=zlabel)
                    axs[0].set_xscale('log')
                    axs[0].set_yscale('log')
                    axs[0].set_ylim([5e-2, 1e1])

                    nuInuz = np.sum(nuInu[:, iz, :, ip], axis=1)
                    izb += 1
                else:
                    nuInuz += np.sum(nuInu[:, iz, :, ip], axis=1)

            zlabel = "z < {0:.1f}".format(zbins[izb])
            axs[0].plot(wvs, nuInuzL + nuInuz, ls[ip], label=zlabel)

        axs[0].set_title('CIB by Redshift Contribution')
        axs[0].set_xlabel('wavelength [um]')
        axs[0].set_ylabel('nuInu [nW/m^2/sr]')
        if show_total:
            axs[0].plot(wvs, np.sum(np.sum(nuInu[:, :, :, ip], axis=1), axis=1), ls[ip], label='Total', color='y', lw=4, alpha=0.4)
        axs[0].legend(loc='lower left')

        for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
            nuInuzL = 0 * np.sum(nuInu[:, 0, :, ip], axis=1)
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                nuInuzL += np.sum(nuInu[:, :, im, ip], axis=1)
                axs[1].plot(wvs, nuInuzL, ls[ip], label=mlab)
                axs[1].set_xscale('log')
                axs[1].set_yscale('log')
                axs[1].set_ylim([5e-2, 1e1])

        axs[1].set_title('CIB by Stellar-Mass Contribution')
        axs[1].set_xlabel('wavelength [um]')
        axs[1].set_ylabel('nuInu [nW/m^2/sr]')
        if show_total:
            axs[1].plot(wvs, np.sum(np.sum(nuInu[:, :, :, ip], axis=1), axis=1), ls[ip], label='Total', color='y', lw=4, alpha=0.4)
        axs[1].legend(loc='lower left')

    def plot_total_lird(self, total_lird_dict, plot_lird=False, plot_sfrd=True):

        z_bins = np.unique(self.config_dict['distance_bins']['redshift'])
        z_mid = [(z_bins[i] + z_bins[i + 1]) / 2 for i in range(len(z_bins) - 1)]

        if plot_lird:
            lird_total = total_lird_dict['lird_total']
            lird_error = total_lird_dict['lird_total_error']
            fig = plt.figure(figsize=(9, 6))
            bin_keys = list(self.config_dict['parameter_names'].keys())
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                label = "Star-Forming logM=" + '-'.join(mlab.split('_')[2:])
                plt.plot(z_mid, np.log10(total_lird_dict['lird_array']['50'][:, im, 1]), '-', label=label)
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                label = "Quiescent logM=" + '-'.join(mlab.split('_')[2:])
                plt.plot(z_mid, np.log10(total_lird_dict['lird_array']['50'][:, im, 0]), '--', label=label)

            plt.fill_between(z_mid, np.log10([np.max([i, 0.01]) for i in lird_total - lird_error]),
                             np.log10(lird_total + lird_error), facecolor='c', alpha=0.3, edgecolor='c')
            plt.plot(z_mid, np.log10(lird_total), '-', label='total', color='c')
            plt.xlabel('redshift')
            plt.ylabel('IR Luminosity Density [Lsun Mpc3]')
            plt.xlim([0, z_bins[-1]])
            plt.ylim([4.5, 9])
            plt.legend(loc='lower left', frameon=False)

        if plot_sfrd:
            sfrd_total = total_lird_dict['sfrd_total']
            sfrd_error = total_lird_dict['sfrd_total_error']
            fig = plt.figure(figsize=(9, 6))
            bin_keys = list(self.config_dict['parameter_names'].keys())
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                label = "Star-Forming logM=" + '-'.join(mlab.split('_')[2:])
                plt.plot(z_mid, np.log10(conv_lir_to_sfr * total_lird_dict['lird_array']['50'][:, im, 1]), '-',
                         label=label)
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                label = "Quiescent logM=" + '-'.join(mlab.split('_')[2:])
                plt.plot(z_mid, np.log10(conv_lir_to_sfr * total_lird_dict['lird_array']['50'][:, im, 0]), '--',
                         label=label)

            plt.fill_between(z_mid, np.log10([np.max([i, 0.00001]) for i in sfrd_total - sfrd_error]),
                             np.log10(sfrd_total + sfrd_error), facecolor='c', alpha=0.3, edgecolor='c')
            plt.plot(z_mid, np.log10(sfrd_total), '-', label='total', color='c')

            plt.xlabel('redshift')
            plt.ylabel('SFR Density [Msun/yr Mpc3]')
            plt.xlim([0, z_bins[-1]])
            plt.ylim([-5, -1])
            plt.legend(loc='lower left', frameon=False)

    def plot_mcmc_seds(self, mcmc_dict, bootstrap_dict=None, errors=('25', '75')):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        wvs = mcmc_dict['wavelengths']
        wv_array = self.loggen(8, 1000, 100)
        ngals = mcmc_dict['ngals']
        plen = len(self.config_dict['parameter_names'][bin_keys[1]]) * len(
            self.config_dict['parameter_names'][bin_keys[2]])
        zlen = len(self.config_dict['parameter_names'][bin_keys[0]])
        z_med = mcmc_dict['z_median']
        fig, axs = plt.subplots(plen, zlen, figsize=(33, 20))

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    label = "__".join([zlab, mlab, plab]).replace('.', 'p')

                    if type(mcmc_dict['mcmc_dict'][id_label]) is not float:

                        y = mcmc_dict['y'][id_label]
                        yerr = mcmc_dict['yerr'][id_label]
                        sed_params = self.fast_sed_fitter(wvs, y, yerr)
                        sed_array = self.fast_sed(sed_params, wv_array)
                        graybody = self.fast_sed(sed_params, wvs)[0]
                        delta_y = y - graybody
                        med_delta = np.median(y / delta_y)

                        flat_samples = mcmc_dict['mcmc_dict'][id_label]
                        mcmc_out = [np.percentile(flat_samples[:, i], [float(errors[0]), 50, float(errors[1])])
                                    for i in range(np.shape(flat_samples)[1])]

                        mcmc_lo = self.graybody_fn([mcmc_out[0][0], mcmc_out[1][0]], wv_array)
                        mcmc_50 = self.graybody_fn([mcmc_out[0][1], mcmc_out[1][1]], wv_array)
                        mcmc_hi = self.graybody_fn([mcmc_out[0][2], mcmc_out[1][2]], wv_array)

                        #colors = ['y', 'c', 'b', 'r', 'g']
                        if ip:
                            ix = im
                        else:
                            ix = im + len(self.config_dict['parameter_names'][bin_keys[1]])

                        axs[ix, iz].plot(wv_array, sed_array[0] * 1e3, color='k', lw=0.5)
                        if ix == 0:
                            axs[ix, iz].set_title(zlab)

                        # pdb.set_trace()
                        LIR = self.fast_LIR([mcmc_out[0][1], mcmc_out[1][1]], z_med[id_label])
                        mcmc_label = "A={0:.1f}, T={1:.1f}, LIR={2:.1f}".format(mcmc_out[0][1],
                                                                                mcmc_out[1][1] * (1 + z_med[id_label]),
                                                                                LIR)
                        #mcmc_label = "LIR={0:.1f}, T={1:.1f}".format(mcmc_out[0][1],
                        #                                             mcmc_out[1][1] * (1 + z_med[id_label]))

                        axs[ix, iz].plot(wv_array, mcmc_50[0] * 1e3, color='c', lw=0.8, label=mcmc_label)
                        axs[ix, iz].fill_between(wv_array, mcmc_lo[0] * 1e3, mcmc_hi[0] * 1e3, facecolor='c',
                                                 alpha=0.3, edgecolor='c')

                        axs[ix, iz].legend(loc='upper left', frameon=False)

                        # Ain = np.max([1e-39, sed_params['A'].value])
                        Ain = sed_params['A'].value
                        Aerr = sed_params['A'].stderr
                        Tin = sed_params['T_observed'].value
                        Terr = sed_params['T_observed'].stderr
                        if Tin is None:
                            Tin = (10 ** (1.2 + 0.1 * z_med[id_label])) / (1 + z_med[id_label])
                        if Terr is None:
                            Terr = Tin * med_delta
                        if Ain is None:
                            Ain = 39
                        if Aerr is None:
                            Aerr = Ain * med_delta

                        prior_label = "A={0:.1f}+-{1:.1f}, T={2:.1f}+-{3:.1f}".format(Ain, Aerr, Tin, Terr)
                        axs[ix, iz].text(9.0e0, 3e1, prior_label)

                        axs[ix, iz].text(9.0e0, 7e0, "Ngals={0:.0f}".format(ngals[id_label]))

                        for iwv, wv in enumerate(wvs):
                            if wv in [24, 70]:
                                color = 'b'
                            elif wv in [100, 160]:
                                color = 'g'
                            elif wv in [250, 350, 500]:
                                color = 'r'
                            elif wv in [850]:
                                color = 'y'

                            if bootstrap_dict is not None:
                                for iboot in range(len(bootstrap_dict['sed_bootstrap_fluxes_dict'][label])):
                                    yplot_boot = bootstrap_dict['sed_bootstrap_fluxes_dict'][label][iboot] * 1e3
                                    # pdb.set_trace()
                                    axs[ix, iz].scatter(wvs, yplot_boot, color=color, alpha=0.1)

                            #axs[ix, iz].scatter(wv, y[iwv] * 1e3, marker='o', s=90, facecolors='none', edgecolors=color)
                            #axs[ix, iz].errorbar(wv, y[iwv] * 1e3, yerr=np.sqrt(np.diag(yerr)[iwv]) * 1e3,
                            #                     fmt="." + color, capsize=0)

                            yerr_diag = np.sqrt(np.diag(yerr)[iwv])
                            sigma_upper_limit = 3
                            if y[iwv] - yerr_diag < 0:
                                #yplot = y[iwv] / yerr_diag * sigma_upper_limit
                                yplot = yerr_diag * sigma_upper_limit
                                axs[ix, iz].errorbar(wv, yplot * 1e3, yerr=np.sqrt(np.diag(yerr)[iwv]) * 1e3,
                                             fmt="." + color, uplims=True)
                            else:
                                uplims = False
                                yplot = y[iwv]
                                axs[ix, iz].scatter(wv, yplot * 1e3, marker='o', s=90, facecolors='none', edgecolors=color)
                                axs[ix, iz].errorbar(wv, yplot * 1e3, yerr=np.sqrt(np.diag(yerr)[iwv]) * 1e3,
                                             fmt="." + color, capsize=0)

                        axs[ix, iz].set_xscale('log')
                        axs[ix, iz].set_yscale('log')
                        axs[ix, iz].set_ylim([1e-2, 5e2])
                        # axs[ix, iz].set_title(id_label)

    def plot_seds(self, sed_results_dict):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        colors = ['y', 'c', 'b', 'r', 'g']
        seds = self.make_array_from_dict(sed_results_dict['sed_fluxes_dict'],
                                    sed_results_dict['wavelengths'])
        stds = self.make_array_from_dict(sed_results_dict['std_fluxes_dict'],
                                    sed_results_dict['wavelengths'])
        graybodies = self.get_fast_sed_dict(sed_results_dict)
        wvs = sed_results_dict['wavelengths']

        zlen = len(self.config_dict['parameter_names'][bin_keys[0]])
        if len(self.config_dict['parameter_names']) == 3:
            plen = len(self.config_dict['parameter_names'][bin_keys[2]])
            fig, axs = plt.subplots(plen, zlen, figsize=(36, 10))
            for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                zmid = 0.5 * np.sum([float(i) for i in zlab.split('_')[-2:]])
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                        id_label = "__".join([zlab, mlab, plab])

                        axs[ip, iz].scatter(wvs, seds[:, iz, im, ip], color=colors[im])
                        axs[ip, iz].errorbar(wvs, seds[:, iz, im, ip], stds[:, iz, im, ip], 0, 'none', color=colors[im])

                        LIR = graybodies['lir'][id_label]
                        sed_params = graybodies['sed_params'][id_label]
                        T_obs = sed_params['T_observed'].value
                        T_rf = T_obs * (1 + zmid)
                        wv_array = graybodies['wv_array']
                        sed_array = graybodies['graybody'][id_label]

                        line_label = ['-'.join(mlab.split('_')[-2:]), "Trf={:.1f}".format(T_rf),
                                      "LIR={:.1f}".format(np.log10(LIR))]
                        axs[ip, iz].plot(wv_array, sed_array, label=line_label, color=colors[im])
                        axs[ip, iz].legend(loc='upper right')

                        if not ip:
                            axs[ip, iz].set_title(zlab)
                        axs[ip, iz].set_xscale('log')
                        axs[ip, iz].set_yscale('log')
                        axs[ip, iz].set_xlim([10, 1000])
                        axs[ip, iz].set_ylim([1e-5, 5e-1])

        else:
            fig, axs = plt.subplots(1, zlen, figsize=(36, 10))
            for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                zmid = 0.5 * np.sum([float(i) for i in zlab.split('_')[-2:]])
                for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                    id_label = "__".join([zlab, mlab])

                    axs[iz].scatter(wvs, seds[:, iz, im],)

                    LIR = graybodies['lir'][id_label]
                    sed_params = graybodies['sed_params'][id_label]
                    T_obs = sed_params['T_observed'].value
                    T_rf = T_obs * (1 + zmid)
                    wv_array = graybodies['wv_array']
                    sed_array = graybodies['graybody'][id_label]

                    line_label = ['-'.join(mlab.split('_')[-2:]), "Trf={:.1f}".format(T_rf),
                                  "LIR={:.1f}".format(np.log10(LIR))]
                    if LIR > 0:
                        axs[iz].plot(wv_array, sed_array[0], label=line_label)
                        axs[iz].legend(loc='upper right')
                    else:
                        axs[iz].plot(wv_array, sed_array[0])

                    if not im:
                        axs[iz].set_title(zlab)
                    axs[iz].set_xscale('log')
                    axs[iz].set_yscale('log')
                    axs[iz].set_xlim([10, 1000])
                    axs[iz].set_ylim([1e-5, 5e-1])

    def plot_flux_densities(self):
        wv_keys = list(self.results_dict['band_results_dict'].keys())
        wlen = len(wv_keys)
        if len(self.config_dict['parameter_names']) == 3:
            plen = 2
            fig, axs = plt.subplots(plen, wlen, figsize=(22, 10))
            for iwv, wlab in enumerate(wv_keys):
                for ip, plab in enumerate(self.results_dict['band_results_dict'][wlab]['raw_fluxes_dict']['flux_df']):
                    flux_df = self.results_dict['band_results_dict'][wlab]['raw_fluxes_dict']['flux_df'][plab]
                    error_df = self.results_dict['band_results_dict'][wlab]['raw_fluxes_dict']['error_df'][plab]
                    for mlab in flux_df:
                        if wlen > 1:
                            axs[ip, iwv].scatter(flux_df[mlab].index, flux_df[mlab].values * 1e3)
                            axs[ip, iwv].errorbar(flux_df[mlab].index, flux_df[mlab].values * 1e3,
                                                  error_df[mlab].values * 1e3,
                                                  label=mlab)
                            if not ip:
                                axs[ip, iwv].set_title(wlab)
                            else:
                                axs[ip, iwv].set_xlabel('redshift')
                            if not iwv:
                                axs[ip, iwv].set_ylabel('flux density (Jy)')
                            axs[ip, iwv].set_yscale('log')
                            # axs[ip, iwv].set_xlim([0., 8])
                            axs[ip, iwv].set_ylim([1e-3, 5e1])
                            if (ip == 1) & (iwv == 0):
                                axs[ip, iwv].legend(loc='upper right')
                        else:
                            axs[ip].scatter(flux_df[mlab].index, flux_df[mlab].values * 1e3)
                            axs[ip].errorbar(flux_df[mlab].index, flux_df[mlab].values * 1e3,
                                             error_df[mlab].values * 1e3,
                                             label=mlab)
                            if not ip:
                                axs[ip].set_title(wlab)
                            else:
                                axs[ip].set_xlabel('redshift')
                            if not iwv:
                                axs[ip].set_ylabel('flux density (Jy)')
                            axs[ip].set_yscale('log')
                            # axs[ip].set_xlim([0., 8])
                            axs[ip].set_ylim([1e-3, 5e1])
                            if (ip == 1):
                                axs[ip].legend(loc='upper right')
        else:
            plen = 1
            fig, axs = plt.subplots(plen, wlen, figsize=(22, 10))
            for iwv, wlab in enumerate(wv_keys):
                flux_df = self.results_dict['band_results_dict'][wlab]['raw_fluxes_dict']['flux_df']
                error_df = self.results_dict['band_results_dict'][wlab]['raw_fluxes_dict']['error_df']
                for mlab in flux_df:
                    # pdb.set_trace()
                    axs[iwv].scatter(flux_df[mlab].index, flux_df[mlab].values * 1e3)
                    axs[iwv].errorbar(flux_df[mlab].index, flux_df[mlab].values * 1e3, error_df[mlab].values * 1e3,
                                      label=mlab)
                    axs[iwv].set_title(wlab)
                    axs[iwv].set_xlabel('redshift')
                    if not iwv:
                        axs[iwv].set_ylabel('flux density (Jy)')
                    axs[iwv].set_yscale('log')
                    axs[iwv].set_ylim([1e-3, 5e1])
                    if (iwv == 0):
                        axs[iwv].legend(loc='upper right')

    def plot_lir_vs_z(self):
        if self.results_dict['SED_df']['plot_sed']:

            size_lir_vs_z_array = [len(self.config_dict['parameter_names'][i]) for i in
                                   self.config_dict['parameter_names']]
            sed_lir_vs_z_array = np.zeros(size_lir_vs_z_array)
            z_data_array = np.zeros(size_lir_vs_z_array[0])
            zlen = 1
            plen = len(size_lir_vs_z_array) - 1

            # Rearrange data into arrays
            for z, zlab in enumerate(self.results_dict['SED_df']['flux_density']):
                zmid = 0.5 * np.sum([float(i) for i in zlab.split('_')[-2:]])
                if plen == 2:
                    for p, plab in enumerate(self.results_dict['SED_df']['flux_density'][zlab]):
                        sed = self.results_dict['SED_df']['flux_density'][zlab][plab]
                        #std = self.results_dict['SED_df']['std_error'][zlab][plab]
                        for im, mlab in enumerate(sed):
                            if mlab in self.results_dict['SED_df']['LIR'][zlab][plab]:
                                LIR = self.results_dict['SED_df']['LIR'][zlab][plab][mlab][0]
                                #sed_params = self.results_dict['SED_df']['SED'][zlab][plab][mlab]
                                z_data_array[z] = zmid
                                sed_lir_vs_z_array[z, im, p] = np.log10(LIR)
                else:
                    sed = self.results_dict['SED_df']['flux_density'][zlab]
                    #std = self.results_dict['SED_df']['std_error'][zlab]
                    for im, mlab in enumerate(sed):
                        LIR = self.results_dict['SED_df']['LIR'][zlab][mlab][0]
                        sed_params = self.results_dict['SED_df']['SED'][zlab][mlab]
                        z_data_array[z] = zmid
                        sed_lir_vs_z_array[z, im] = np.log10(LIR)

            # Plot LIR vs z
            keys = list(self.config_dict['parameter_names'])
            fig, axs = plt.subplots(1, plen, figsize=(36, 10))
            for im, mlab in enumerate(self.config_dict['parameter_names'][keys[1]]):
                if plen == 2:
                    for ip, plab in enumerate(self.config_dict['parameter_names'][keys[2]]):
                        axs[ip].scatter(z_data_array, sed_lir_vs_z_array[:, im, ip])
                        axs[ip].plot(z_data_array, sed_lir_vs_z_array[:, im, ip], label=mlab)
                        axs[ip].set_ylabel('LIR (M_sun)')
                        axs[ip].set_xlabel('redshift')
                        axs[ip].set_ylim([9, 13.5])
                        axs[ip].set_title(plab)
                else:
                    axs.scatter(z_data_array, sed_lir_vs_z_array[:, im])
                    axs.plot(z_data_array, sed_lir_vs_z_array[:, im], label=mlab)
                    axs.set_ylabel('LIR (M_sun)')
                    axs.set_xlabel('redshift')
                    axs.set_ylim([9, 13.5])
        else:
            print("Skipping SED plotting because only single wavelength measured.")

    def plot_rest_frame_temperature(self, lir_in, ylim=[1, 100], xlog=False, ylog=True, fit_p=[1.35, 0.09], show_prior=False):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]

        sm = np.zeros(ds)
        zmed = np.zeros(ds)
        t_obs = np.zeros(ds)
        t_rf = np.zeros(ds)
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    if ip:
                        t_obs[iz, im, ip] = lir_in['Tobs_dict'][id_label]['50']
                        t_rf[iz, im, ip] = t_obs[iz, im, ip] * (1 + lir_in['z_median'][id_label])
                        sm[iz, im, ip] = lir_in['m_median'][id_label]
                        zmed[iz, im, ip] = lir_in['z_median'][id_label]

        fig, axs = plt.subplots(1, 1, figsize=(10, 7))
        for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
            label = "n=" + '-'.join(mlab.split('_')[1:])
            axs.plot(zmed[:, im, 1], (t_rf[:, im, 1]), ":o", label=label)
        z_in = np.linspace(0, zmed[-1, 0, 1])
        if show_prior:
            fit_label = "Tprior = 10^({0:.1f} + {1:.1f}z)".format(fit_p[0], fit_p[1])
            axs.plot(z_in, (10 ** (fit_p[0] + fit_p[1] * z_in)), '--', label=fit_label)
        axs.plot(z_in, (1 + z_in) * 2.73, '--', label='CMB')

        # The Literature:
        axs.scatter(4.5, 47, c='k', s=80, marker='s', label='Bethermin 2020')
        axs.errorbar(5.5, 38, 8, c='r', markersize=8, marker='s', label='Faisst 2020')
        axs.errorbar(7, 52, 11, c='g', markersize=8, marker='s', label='Ferrara 2022')

        xmod = np.linspace(0, 8)
        axs.plot(xmod, xmod * (38 / 8) + 24, label='Schrieber 2018')
        axs.plot(xmod, 23 + xmod * ((39 - 23) / 4), label='Viero 2013')

        if xlog:
            axs.set_xscale('log')
        if ylog:
            axs.set_yscale('log')
        axs.set_xlim([z_in[0], z_in[-1] + .5])
        axs.set_ylim(ylim)
        axs.set_xlabel('redshift')
        axs.set_ylabel('T_restframe')
        axs.legend(loc='upper left')

    def plot_star_forming_main_sequence(self, lir_in, ylim=[1e-1, 1e4], xlog=False, ylog=True):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
        sfr = np.zeros(ds)
        sm = np.zeros(ds)
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    if ip:
                        sfr[iz, im, ip] = conv_lir_to_sfr * lir_in['lir_dict'][id_label]['50']
                        sm[iz, im, ip] = lir_in['m_median'][id_label]

        fig, axs = plt.subplots(1, 1, figsize=(10, 7))
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            label = "z=" + '-'.join(zlab.split('_')[1:])
            axs.plot(sm[iz, :, 1], (sfr[iz, :, 1]), ":o", label=label)
        if xlog:
            axs.set_xscale('log')
        if ylog:
            axs.set_yscale('log')
        axs.set_ylim(ylim)
        axs.set_xlabel('Stellar Mass [Mstar]')
        axs.set_ylabel('SFR [Mstar/yr]')
        axs.legend(loc='lower right')