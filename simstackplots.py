import pdb
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lmfit import Parameters, minimize, fit_report
import corner

from simstacktoolbox import SimstackToolbox
conv_lir_to_sfr = 1.728e-10 / 10 ** 0.23
a_nu_flux_to_mass = 6.7e19
sigma_upper_limit = 3

font = {'size': 14}
matplotlib.rc('font', **font)
#plt.rcParams.update({
#  "text.usetex": True,
#  "font.family": "Helvetica"
#})

#plt.rcParams['text.usetex'] = True
#matplotlib.rc('text', usetex=True)
#matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

class SimstackPlots(SimstackToolbox):

    def __init__(self, SimstackPlotsObject):
        super().__init__()

        dict_list = dir(SimstackPlotsObject)
        for i in dict_list:
            if '__' not in i:
                setattr(self, i, getattr(SimstackPlotsObject, i))

    def plot_cib(self, cib_dict=None, tables=None, area_deg2=None, zbins=[0, 1, 2, 3, 4, 6, 9],
                 save_path=None, save_filename="CIB.pdf"):

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
                    axs[0].set_ylim([1e-3, 1e1])
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
                axs[1].set_ylim([1e-3, 1e1])

        axs[1].set_title('CIB by Stellar-Mass Contribution')
        axs[1].set_xlabel('wavelength [um]')
        axs[1].set_ylabel('nuInu [nW/m^2/sr]')
        axs[1].plot(wvs, np.sum(np.sum(nuInu[:, :, :, ip], axis=1), axis=1), ls[ip], label='Total', lw=3)
        axs[1].legend(loc='lower left')

        if save_path is not None:
            plt.savefig(os.path.join(save_path, save_filename), format="pdf", bbox_inches="tight")

    def plot_cib_layers(self, cib_dict=None, tables=None, area_deg2=None, show_total=False,
                        zbins=[0, 1, 2, 3, 4, 6, 9], save_path=None, save_filename="CIB_layers.pdf"):

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
                if zhi >= zbins[izb]:
                    # pdb.set_trace()
                    nuInuzL += nuInuz
                    zlabel = "-".join([str(zbins[izb - 1]), str(zbins[izb])])
                    zlabel = "z < {0:.1f}".format(zbins[izb])
                    axs[0].plot(wvs, nuInuzL, ls[ip], label=zlabel)
                    axs[0].set_xscale('log')
                    axs[0].set_yscale('log')
                    axs[0].set_ylim([1e-3, 1e1])

                    nuInuz = np.sum(nuInu[:, iz, :, ip], axis=1)
                    izb += 1
                else:
                    nuInuz += np.sum(nuInu[:, iz, :, ip], axis=1)

            #zlabel = "z < {0:.1f}".format(zbins[izb-1])
            #axs[0].plot(wvs, nuInuzL + nuInuz, ls[ip], label=zlabel)

        axs[0].set_title('CIB by Redshift Contribution')
        axs[0].set_xlabel('Observed Wavelength [um]')
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
                axs[1].set_ylim([1e-3, 1e1])

        axs[1].set_title('CIB by Stellar-Mass Contribution')
        axs[1].set_xlabel('Observed Wavelength [um]')
        axs[1].set_ylabel('nuInu [nW/m^2/sr]')
        if show_total:
            axs[1].plot(wvs, np.sum(np.sum(nuInu[:, :, :, ip], axis=1), axis=1), ls[ip], label='Total', color='y', lw=4, alpha=0.4)
        axs[1].legend(loc='lower left')

        if save_path is not None:
            plt.savefig(os.path.join(save_path, save_filename), format="pdf", bbox_inches="tight")

    def plot_total_lird(self, total_lird_dict, plot_lird=False, plot_sfrd=True, ylim=[5, 9],
                        save_path=None, save_filename="Tdust.pdf"):

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
            plt.xlabel('Redshift')
            plt.ylabel('IR Luminosity Density [Lsun Mpc^-3]')
            plt.xlim([0, z_bins[-1]])
            plt.ylim(ylim)
            plt.legend(loc='lower left', frameon=False)

        if plot_sfrd:
            sfrd_total = total_lird_dict['sfrd_total']
            sfrd_error = total_lird_dict['sfrd_total_error']
            fig = plt.figure(figsize=(9, 6))

            xbow = [1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]
            ybow0 = [0.025, 0.057, 0.108, 0.142, 0.134, 0.077, 0.046, 0.0301, 0.023]
            ybow1 = [0.0044, 0.029, 0.0717, 0.089, 0.08, 0.0362, 0.012, 0.005, 0.0023]
            # plt.fill_between(xbow,np.log10(ybow0),np.log10(ybow1), facecolor='k', alpha=0.2, edgecolor='k', label='Bouwens+ 2020')

            xzav = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
            yzav0 = [0.009, 0.029, 0.065, 0.118, 0.155, 0.133, 0.084, 0.059, 0.043, 0.033, 0.026, 0.02, 0.016, 0.014,
                     0.012]
            yzav1 = [0.0048, 0.0158, 0.0347, 0.0555, 0.069, 0.0584, 0.0388, 0.0226, 0.0136, 0.009, 0.005, 0.004, 0.0039,
                     0.0032, 0.0026]
            plt.fill_between(xzav, np.log10(yzav0), np.log10(yzav1), facecolor='r', alpha=0.1, edgecolor='r',
                             label='Zavala+ 2022')

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
            plt.plot(z_mid, np.log10(sfrd_total), '-', label='All Galaxies', color='c')

            xsides = [0, 0.5, 1, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0]
            ysides = [0.009, 0.041, 0.067, 0.081, 0.0848, 0.0849, 0.0578, 0.0288, 0.0168, 0.0093, 0.0058]
            plt.plot(xsides, np.log10(ysides), '-.', c='y', lw=2, label='[SIDES] Bethermin+ 2017')

            xill = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
            yill = [0.0117, 0.0253, 0.0431, 0.0562, 0.0674, 0.0666, 0.0651, 0.0531, 0.0500, 0.0455, 0.0295, 0.0193,
                    0.0156, 0.0114, 0.008]
            plt.plot(xill, np.log10(yill), '--', c='g', lw=2, label='[IllustrisTNG] Pillepich+ 2018')

            # Gruppioni
            z0 = np.array([0.93, 1.92, 2.92, 3.92, 5.18])
            ze = np.zeros([2, len(z0)])
            zl = z0 - np.array([0.43, 1.42, 2.43, 3.43, 4.42])
            zh = np.array([1.42, 2.43, 3.43, 4.42, 5.94]) - z0
            y0 = np.array([0.1, 0.165, 0.264, 0.165, 0.212])
            ye = np.zeros([2, len(y0)])
            yh = np.array([0.185, 0.326, 0.40, 0.277, 0.574])
            yl = np.array([0.067, 0.095, 0.184, 0.105, 0.093])
            ze[0] = zl
            ze[1] = zh
            ye[0] = np.log10(y0) - np.log10(yl)
            ye[1] = np.log10(yh) - np.log10(y0)
            #plt.errorbar(z0, np.log10(y0), xerr=ze, yerr=ye, color='k', label='Gruppioni+ 2020')

            plt.xlabel('Redshift')
            plt.ylabel('SFR Density [Msun yr^-1 Mpc^-3]')
            plt.xlim([0, z_bins[-1] - 0.5])
            plt.ylim([-3.75, -0.75])
            # plt.legend(loc='lower left', frameon=False)
            plt.legend(bbox_to_anchor=(1, 1), loc="upper left")

        if save_path is not None:
            plt.savefig(os.path.join(save_path, save_filename), format="pdf", bbox_inches="tight")

    def plot_total_sfrd(self, total_sfrd_dict, plot_lird=True, ylim=[-3.75, -0.75],
                        show_qt=False, save_path=None, save_filename="SFRD.pdf"):

        bin_keys = list(self.config_dict['parameter_names'].keys())
        z_bins = np.unique(self.config_dict['distance_bins']['redshift'])
        z_mid = [(z_bins[i] + z_bins[i + 1]) / 2 for i in range(len(z_bins) - 1)]

        sfrd_total = total_sfrd_dict['sfrd_total']
        sfrd_error = total_sfrd_dict['sfrd_total_error']
        fig, ax = plt.subplots(figsize=(10, 7))

        xbow = [1, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 7.0]
        ybow0 = [0.025, 0.057, 0.108, 0.142, 0.134, 0.077, 0.046, 0.0301, 0.023]
        ybow1 = [0.0044, 0.029, 0.0717, 0.089, 0.08, 0.0362, 0.012, 0.005, 0.0023]
        # plt.fill_between(xbow,np.log10(ybow0),np.log10(ybow1), facecolor='k', alpha=0.2, edgecolor='k', label='Bouwens+ 2020')

        # OTHERS
        # Zavala
        xzav = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
        yzav0 = [0.009, 0.029, 0.065, 0.118, 0.155, 0.133, 0.084, 0.059, 0.043, 0.033, 0.026, 0.02, 0.016, 0.014,
                 0.012]
        yzav1 = [0.0048, 0.0158, 0.0347, 0.0555, 0.069, 0.0584, 0.0388, 0.0226, 0.0136, 0.009, 0.005, 0.004, 0.0039,
                 0.0032, 0.0026]
        ax.fill_between(xzav, np.log10(yzav0), np.log10(yzav1), facecolor='r', alpha=0.1, edgecolor='r',
                        label='Zavala+ 2022')

        ax.legend(loc='lower left', frameon=True, prop={'size': 12})

        # Bethermin
        xsides = [0, 0.5, 1, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 6.5, 7.0]
        ysides = [0.009, 0.041, 0.067, 0.081, 0.0848, 0.0849, 0.0578, 0.0288, 0.0168, 0.0093, 0.0058]
        ax.plot(xsides, np.log10(ysides), '-.', c='y', lw=2.5, label='[SIDES] Bethermin+ 2017')

        # Pillepich
        xill = [0, 0.5, 1, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0]
        yill = [0.0117, 0.0253, 0.0431, 0.0562, 0.0674, 0.0666, 0.0651, 0.0531, 0.0500, 0.0455, 0.0295, 0.0193,
                0.0156, 0.0114, 0.008]
        ax.plot(xill, np.log10(yill), '--', c='g', lw=2.5, label='[IllustrisTNG] Pillepich+ 2018')

        # Gruppioni
        z0 = np.array([0.93, 1.92, 2.92, 3.92, 5.18])
        ze = np.zeros([2, len(z0)])
        zl = z0 - np.array([0.43, 1.42, 2.43, 3.43, 4.42])
        zh = np.array([1.42, 2.43, 3.43, 4.42, 5.94]) - z0
        y0 = np.array([0.1, 0.165, 0.264, 0.165, 0.212])
        ye = np.zeros([2, len(y0)])
        yh = np.array([0.185, 0.326, 0.40, 0.277, 0.574])
        yl = np.array([0.067, 0.095, 0.184, 0.105, 0.093])
        ze[0] = zl
        ze[1] = zh
        ye[0] = np.log10(y0) - np.log10(yl)
        ye[1] = np.log10(yh) - np.log10(y0)
        # ax.errorbar(z0, np.log10(y0), xerr=ze, yerr=ye, color='k', label='Gruppioni+ 2020')

        # LIRD
        if plot_lird:

            lird_total = total_sfrd_dict['lird_total']
            lird_error = total_sfrd_dict['lird_total_error']
            ax.fill_between(z_mid, np.log10(
                conv_lir_to_sfr * np.array([np.max([i, 0.00001]) for i in (lird_total - lird_error)])),
                            np.log10(conv_lir_to_sfr * (lird_total + lird_error)), facecolor='k', alpha=0.2,
                            edgecolor='k',
                            label='This Work - L_SED')

            ax2 = ax.twinx()
            ax2.plot(z_mid, np.log10(lird_total), '-', color='k', alpha=0.75)
            # Adding Twin Axes to plot using dataset_2
            ax2.set_ylim(np.log10(1 / conv_lir_to_sfr * 10 ** np.array(ylim)))

            ax2.set_ylabel('log(LIR Density) [Lsun Mpc^-3]')
            ax2.tick_params(axis='y')
            # ax2.legend(loc='lower left', frameon=True, prop={'size': 12})

            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                label = "This Work - log(M/Msun)=" + '-'.join(mlab.split('_')[2:])
                ax.plot(z_mid, np.log10(conv_lir_to_sfr * total_sfrd_dict['lird_array']['50'][:, im, 1]), '-',
                        label=label)
            if show_qt:
                for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                    label = "Quiescent log(M/Msun)=" + '-'.join(mlab.split('_')[2:])
                    ax.plot(z_mid, np.log10(conv_lir_to_sfr * total_sfrd_dict['lird_array']['50'][:, im, 0]), '--',
                            label=label)
        else:
            # SFRD
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                label = "This Work - log(M/Msun)=" + '-'.join(mlab.split('_')[2:])
                ax.plot(z_mid, np.log10(total_sfrd_dict['sfrd_array']['50'][:, im, 1]), '-',
                        label=label)
            if show_qt:
                for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                    label = "Quiescent log(M/Msun)=" + '-'.join(mlab.split('_')[2:])
                    ax.plot(z_mid, np.log10(total_sfrd_dict['sfrd_array']['50'][:, im, 0]), '--',
                            label=label)

        ax.fill_between(z_mid, np.log10([np.max([i, 0.00001]) for i in sfrd_total - sfrd_error]),
                        np.log10(sfrd_total + sfrd_error), facecolor='c', alpha=0.3, edgecolor='c',
                        label='This Work - L_850')
        ax.plot(z_mid, np.log10(sfrd_total), '-', color='c')

        ax.set_xlabel('Redshift')
        ax.set_ylabel('log(SFR Density) [Msun yr^-1 Mpc^-3]')
        # ax.ylabel('\rho_{SFR} [M_{\odot} yr^-1 Mpc^-3]')
        ax.set_xlim([0, z_bins[-1] - 1])
        ax.set_ylim(ylim)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(loc='lower left', frameon=True, prop={'size': 11})

        if save_path is not None:
            plt.savefig(os.path.join(save_path, save_filename), format="pdf", bbox_inches="tight")

    def plot_mcmc_seds(self, mcmc_dict, bootstrap_dict=None, errors=('25', '75'), fontsize=11.5,
                       show_qt=False, save_path=None, save_filename="SEDs.pdf"):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        wvs = mcmc_dict['wavelengths']
        wv_array = self.loggen(8, 1000, 100)
        ngals = mcmc_dict['ngals']
        total_ngals = 0
        p2 = 1
        if show_qt: p2 = 2
        plen = len(self.config_dict['parameter_names'][bin_keys[1]] * p2)
        zlen = len(self.config_dict['parameter_names'][bin_keys[0]])
        z_med = mcmc_dict['z_median']
        m_med = mcmc_dict['m_median']

        width_ratios = [i for i in np.ones(zlen)]
        gs = gridspec.GridSpec(plen, zlen, width_ratios=width_ratios,
                               wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)
        fig = plt.figure(figsize=(34, 9 * p2))

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    if ip or show_qt:
                        id_label = "__".join([zlab, mlab, plab])
                        label = "__".join([zlab, mlab, plab]).replace('.', 'p')

                        if ip:
                            ix = im
                        else:
                            ix = im + len(self.config_dict['parameter_names'][bin_keys[1]])
                        ax = plt.subplot(gs[ix, iz])
                        ax.set_yticklabels([])
                        if type(mcmc_dict['mcmc_dict'][id_label]) is not float:

                            y = mcmc_dict['y'][id_label]
                            yerr = mcmc_dict['yerr'][id_label]
                            sed_params = self.fast_sed_fitter(wvs, y, yerr)
                            sed_array = self.fast_sed(sed_params, wv_array)
                            graybody = self.fast_sed(sed_params, wvs)[0]
                            delta_y = y - graybody
                            med_delta = np.median(y / delta_y)

                            # Compare Schrieber 2018 fit
                            t_forced = (32.9 + 4.6 * (z_med[id_label] - 2)) / (1 + z_med[id_label])
                            a_forced = self.get_A_given_z_M_T(z_med[id_label], m_med[id_label],
                                                              (32.9 + 4.6 * (z_med[id_label] - 2)))
                            sed_forced_array = self.graybody_fn((a_forced, t_forced), wv_array)

                            flat_samples = mcmc_dict['mcmc_dict'][id_label]
                            mcmc_out = [np.percentile(flat_samples[:, i], [float(errors[0]), 50, float(errors[1])])
                                        for i in range(np.shape(flat_samples)[1])]

                            mcmc_lo = self.graybody_fn([mcmc_out[0][0], mcmc_out[1][0]], wv_array)
                            mcmc_50 = self.graybody_fn([mcmc_out[0][1], mcmc_out[1][1]], wv_array)
                            mcmc_hi = self.graybody_fn([mcmc_out[0][2], mcmc_out[1][2]], wv_array)

                            # Plot model SEDs
                            # ax.plot(wv_array, sed_array[0] * 1e3, color='k', lw=0.5)
                            if ip:
                                ax.plot(wv_array, sed_forced_array[0] * 1e3, 'm:', lw=0.5)

                            LIR = self.fast_LIR([mcmc_out[0][1], mcmc_out[1][1]], z_med[id_label])

                            ax.plot(wv_array, mcmc_50[0] * 1e3, color='c', lw=0.8, label=None)
                            ax.plot(wv_array, mcmc_lo[0] * 1e3, ":", color='c', lw=0.6, label=None)
                            ax.plot(wv_array, mcmc_hi[0] * 1e3, ":", color='c', lw=0.6, label=None)
                            ax.fill_between(wv_array, mcmc_lo[0] * 1e3, mcmc_hi[0] * 1e3, facecolor='c',
                                            alpha=0.3, edgecolor='c')

                            # Get 850 SFR
                            rest_frame_850 = 850 * (1 + z_med[id_label])
                            flux_850 = self.graybody_fn([mcmc_out[0][1], mcmc_out[1][1]], [rest_frame_850])
                            L_850A = self.fast_L850(flux_850, z_med[id_label])
                            M_mol = L_850A / a_nu_flux_to_mass
                            sfr = 35 * (M_mol / 1e10) ** (0.89) * ((1 + z_med[id_label]) / 3) ** (0.95)

                            ax.text(9.0e0, 8e1, "log(LIR/Lsun)={0:0.1f}".format(np.log10(LIR)),fontsize=fontsize)
                            ax.text(9.0e0, 2.5e1, "SFR={0:.0f}Msun/yr".format(sfr[0]),fontsize=fontsize)
                            ax.text(9.0e0, 7.2e0, "Trf={0:0.1f}K".format(mcmc_out[1][1] * (1 + z_med[id_label])),fontsize=fontsize)
                            ax.text(9.0e0, 2e0, "Ngals={0:0.0f}".format(ngals[id_label]),fontsize=fontsize)

                            total_ngals += ngals[id_label]
                            #print('Total cumulative galaxies={0:0.0f}'.format(total_ngals))

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
                                        ax.scatter(wvs, yplot_boot, color=color, alpha=0.1)

                                sigma_upper_limit = 3
                                yerr_diag = np.sqrt(np.diag(yerr)[iwv])
                                if y[iwv] - yerr_diag < 0:
                                    yplot = yerr_diag * sigma_upper_limit
                                    ax.errorbar(wv, yplot * 1e3, yerr=np.sqrt(np.diag(yerr)[iwv]) * 1e3,
                                                fmt="." + color, uplims=True)
                                else:
                                    yplot = y[iwv]
                                    ax.scatter(wv, yplot * 1e3, marker='o', s=90, facecolors='none', edgecolors=color)
                                    ax.errorbar(wv, yplot * 1e3, yerr=np.sqrt(np.diag(yerr)[iwv]) * 1e3,
                                                fmt="." + color, capsize=0)

                            ax.set_xscale('log')
                            ax.set_yscale('log')

                            if ix == 0:
                                ax.set_title(zlab.replace('redshift_', 'z=').replace('_', '-'))
                            if iz:
                                ax.set_yticklabels([])

                            if ix != plen - 1:
                                ax.set_xticklabels([])

                            if show_qt:
                                if iz == 0:
                                    if ip:
                                        ax.set_ylabel("Star-Forming")
                                    else:
                                        ax.set_ylabel("Quiescent")

                            if iz == zlen - 1:
                                ax.yaxis.set_label_position("right")
                                #ax.set_ylabel(mlab.replace('stellar_mass_', 'log(M/Msun)=').replace('_', '-'))
                                ax.set_ylabel(mlab.replace('stellar_mass_', '').replace('_', '-'))

                            ax.set_ylim([1e-2, 5e2])

        fig.text(0.5, -0.0075, 'Observed Wavelength [micron]', ha='center')
        fig.text(0.146, 0.5, 'Flux Density [Jy beam^-1]', va='center', rotation='vertical')
        fig.text(0.8535, 0.5, 'log(Stellar Mass) [log(Msun)]', va='center', rotation='vertical')

        if save_path is not None:
            plt.savefig(os.path.join(save_path, save_filename), format="pdf", bbox_inches="tight")

    def plot_mcmc_seds_debug(self, mcmc_dict, bootstrap_dict=None, errors=('25', '75'),
                       save_path=None, save_filename="SEDs.pdf"):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        wvs = mcmc_dict['wavelengths']
        wv_array = self.loggen(8, 1000, 100)
        ngals = mcmc_dict['ngals']
        plen = len(self.config_dict['parameter_names'][bin_keys[1]]) * len(
            self.config_dict['parameter_names'][bin_keys[2]])
        zlen = len(self.config_dict['parameter_names'][bin_keys[0]])
        z_med = mcmc_dict['z_median']
        m_med = mcmc_dict['m_median']

        width_ratios = [i for i in np.ones(zlen)]
        gs = gridspec.GridSpec(plen, zlen, width_ratios=width_ratios,
                               wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845)
        fig = plt.figure(figsize=(34, 18))

        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    label = "__".join([zlab, mlab, plab]).replace('.', 'p')

                    if ip:
                        ix = im
                    else:
                        ix = im + len(self.config_dict['parameter_names'][bin_keys[1]])
                    ax = plt.subplot(gs[ix, iz])
                    ax.set_yticklabels([])
                    if type(mcmc_dict['mcmc_dict'][id_label]) is not float:

                        y = mcmc_dict['y'][id_label]
                        yerr = mcmc_dict['yerr'][id_label]
                        sed_params = self.fast_sed_fitter(wvs, y, yerr)
                        sed_array = self.fast_sed(sed_params, wv_array)
                        graybody = self.fast_sed(sed_params, wvs)[0]
                        delta_y = y - graybody
                        med_delta = np.median(y / delta_y)

                        # Compare Schrieber 2018 fit
                        t_forced = (32.9 + 4.6 * (z_med[id_label] - 2)) / (1 + z_med[id_label])
                        a_forced = self.get_A_given_z_M_T(z_med[id_label], m_med[id_label], (32.9 + 4.6 * (z_med[id_label] - 2)))
                        #sed_forced_params = self.forced_sed_fitter(wvs, y, yerr, t_forced)
                        sed_forced_array = self.graybody_fn((a_forced, t_forced), wv_array)

                        flat_samples = mcmc_dict['mcmc_dict'][id_label]
                        mcmc_out = [np.percentile(flat_samples[:, i], [float(errors[0]), 50, float(errors[1])])
                                    for i in range(np.shape(flat_samples)[1])]

                        mcmc_lo = self.graybody_fn([mcmc_out[0][0], mcmc_out[1][0]], wv_array)
                        mcmc_50 = self.graybody_fn([mcmc_out[0][1], mcmc_out[1][1]], wv_array)
                        mcmc_hi = self.graybody_fn([mcmc_out[0][2], mcmc_out[1][2]], wv_array)

                        #ax.plot(wv_array, sed_array[0] * 1e3, color='k', lw=0.5)
                        if ip:
                            ax.plot(wv_array, sed_forced_array[0] * 1e3, 'm:', lw=0.5)

                        # pdb.set_trace()
                        LIR = self.fast_LIR([mcmc_out[0][1], mcmc_out[1][1]], z_med[id_label])
                        # mcmc_label = "A={0:.1f}, T={1:.1f}, LIR={2:.1f}".format(mcmc_out[0][1],
                        #                                             mcmc_out[1][1] * (1 + z_med[id_label]), LIR)
                        mcmc_label = "A={0:.1f}, T={1:.1f}".format(mcmc_out[0][1],
                                                                   mcmc_out[1][1] * (1 + z_med[id_label]))

                        ax.plot(wv_array, mcmc_50[0] * 1e3, color='c', lw=0.8, label=mcmc_label)
                        ax.plot(wv_array, mcmc_lo[0] * 1e3, ":", color='c', lw=0.6, label=None)
                        ax.plot(wv_array, mcmc_hi[0] * 1e3, ":", color='c', lw=0.6, label=None)
                        ax.fill_between(wv_array, mcmc_lo[0] * 1e3, mcmc_hi[0] * 1e3, facecolor='c',
                                        alpha=0.3, edgecolor='c')

                        ax.legend(loc='upper left', frameon=False)

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

                        # prior_label = "Ap={0:.1f}+-{1:.1f}, Tp={2:.1f}+-{3:.1f}".format(Ain, Aerr, Tin, Terr)
                        #prior_label = "Ap={0:.1f}, Tp={1:.1f}".format(Ain, Tin * (1 + z_med[id_label]))
                        prior_label = "Ap={0:.1f}, Tp={1:.1f}+-{2:.1f}, Trf={3:.1f}".format(Ain, Tin, Terr, Tin * (1 + z_med[id_label]))
                        ax.text(9.0e0, 3e1, prior_label)
                        ax.text(9.0e0, 8e0, "Ngals={0:.0f}".format(ngals[id_label]))
                        ax.text(9.0e0, 2e0, "LIR={0:.1f}".format(np.log10(LIR)))

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
                                    ax.scatter(wvs, yplot_boot, color=color, alpha=0.1)

                            sigma_upper_limit = 3
                            yerr_diag = np.sqrt(np.diag(yerr)[iwv])
                            if y[iwv] - yerr_diag < 0:
                                yplot = yerr_diag * sigma_upper_limit
                                ax.errorbar(wv, yplot * 1e3, yerr=np.sqrt(np.diag(yerr)[iwv]) * 1e3,
                                            fmt="." + color, uplims=True)
                            else:
                                yplot = y[iwv]
                                ax.scatter(wv, yplot * 1e3, marker='o', s=90, facecolors='none', edgecolors=color)
                                ax.errorbar(wv, yplot * 1e3, yerr=np.sqrt(np.diag(yerr)[iwv]) * 1e3,
                                            fmt="." + color, capsize=0)

                        ax.set_xscale('log')
                        ax.set_yscale('log')

                        if ix == 0:
                            ax.set_title(zlab.replace('redshift_', 'z=').replace('_', '-'))
                        if iz:
                            ax.set_yticklabels([])
                        # else:
                        #    ax.set_ylabel('Flux Density [Jy/beam]')
                        if ix != plen - 1:
                            ax.set_xticklabels([])
                        if iz == 0:
                            if ip:
                                ax.set_ylabel("Star-Forming")
                            else:
                                ax.set_ylabel("Quiescent")
                        if iz == zlen - 1:
                            ax.yaxis.set_label_position("right")
                            ax.set_ylabel(mlab.replace('stellar_mass_', 'M=').replace('_', '-'))

                        ax.set_ylim([1e-2, 5e2])
                        # ax.set_xlabel('Wavelength [micron]')
        fig.text(0.5, 0.02, 'Observed Wavelength [micron]', ha='center')
        # fig.text(0.15, 0.5, 'Flux Density [Jy/beam]', va='center', rotation='vertical')
        fig.text(0.14, 0.5, 'Flux Density [Jy/beam]', va='center', rotation='vertical')

        if save_path is not None:
            plt.savefig(os.path.join(save_path, save_filename), format="pdf", bbox_inches="tight")

    def plot_test_mcmc_seds(self, mcmc_dict, flat_samples, theta0, id_label, bootstrap_dict=None, errors=('25', '75')):

        wvs = mcmc_dict['wavelengths']
        wv_array = self.loggen(8, 1000, 100)
        ngals = mcmc_dict['ngals']
        z_med = mcmc_dict['z_median']

        fig, axs = plt.subplots(1, 1, figsize=(10, 7))
        axs.text(9.0e0, 1e2, "Ngals={0:.0f}".format(ngals[id_label]))

        y = mcmc_dict['sed_fluxes_dict'][id_label]
        yerr = np.cov(mcmc_dict['sed_bootstrap_fluxes_dict'][id_label], rowvar=False)

        Ain, Tin, Aerr, Terr = theta0
        sed_params = Parameters()
        sed_params.add('A', value=Ain, vary=True)
        sed_params.add('T_observed', value=Tin, max=Tin * 1.3, vary=True)
        sed_params.add('beta', value=1.8, vary=False)
        sed_params.add('alpha', value=2.0, vary=False)
        sed_array = self.fast_sed(sed_params, wv_array)

        axs.plot(wv_array, sed_array[0] * 1e3, color='k', lw=0.5)

        if np.sum(y > 0):

            try:
                prior_label = "A0={0:.1f}+-{1:.1f}, T0={2:.1f}+-{3:.1f} (Trf0={4:.1f})".format(Ain, Aerr, Tin, Terr,
                                                                                               Tin * (1 + z_med[
                                                                                                   id_label]))
            except:
                prior_label = "A0={0:.1f}, T0={1:.1f}+-{2:.1f} (Trf0={3:.1f})".format(Ain, Tin, Terr,
                                                                                      Tin * (1 + z_med[id_label]))
            axs.text(9.0e0, 1.75e2, prior_label)

            mcmc_out = [np.percentile(flat_samples[:, i], [float(errors[0]), 50, float(errors[1])])
                        for i in range(np.shape(flat_samples)[1])]
            mcmc_lo = self.graybody_fn([mcmc_out[0][0], mcmc_out[1][0]], wv_array)
            mcmc_50 = self.graybody_fn([mcmc_out[0][1], mcmc_out[1][1]], wv_array)
            mcmc_hi = self.graybody_fn([mcmc_out[0][2], mcmc_out[1][2]], wv_array)
            mcmc_label = "A={0:.1f}, T={1:.1f} (Trf={2:.1f})".format(mcmc_out[0][1], mcmc_out[1][1],
                                                                     mcmc_out[1][1] * (1 + z_med[id_label]))

            axs.plot(wv_array, mcmc_50[0] * 1e3, color='c', lw=0.9, label=mcmc_label)
            axs.plot(wv_array, mcmc_lo[0] * 1e3, ":", color='c', lw=0.8, label=None)
            axs.plot(wv_array, mcmc_hi[0] * 1e3, ":", color='c', lw=0.8, label=None)
            axs.fill_between(wv_array, mcmc_lo[0] * 1e3, mcmc_hi[0] * 1e3, facecolor='c',
                             alpha=0.3, edgecolor='c')
        else:
            pdb.set_trace()

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
                for iboot in range(len(bootstrap_dict['sed_bootstrap_fluxes_dict'][id_label.replace('.', 'p')])):
                    yplot_boot = bootstrap_dict['sed_bootstrap_fluxes_dict'][id_label.replace('.', 'p')][iboot] * 1e3
                    axs.scatter(wvs, yplot_boot, color=color, alpha=0.1)

            yerr_diag = np.sqrt(np.diag(yerr)[iwv])
            if y[iwv] - yerr_diag < 0:
                yplot = y[iwv] / yerr_diag * sigma_upper_limit
                yplot = yerr_diag * sigma_upper_limit
                uplims = True
                capsize = 1
                axs.errorbar(wv, yplot * 1e3, yerr=np.sqrt(np.diag(yerr)[iwv]) * 1e3,
                             fmt="." + color, uplims=uplims)
            else:
                uplims = False
                yplot = y[iwv]
                capsize = 0
                axs.scatter(wv, yplot * 1e3, marker='o', s=90, facecolors='none', edgecolors=color)
                axs.errorbar(wv, yplot * 1e3, yerr=np.sqrt(np.diag(yerr)[iwv]) * 1e3,
                             fmt="." + color, capsize=capsize, uplims=uplims)

        axs.set_xscale('log')
        axs.set_yscale('log')
        axs.set_ylim([5e-3, 5e2])
        axs.legend(loc='upper left', frameon=False)
        axs.set_title(id_label)

        corner.corner(flat_samples, color='k', smooth=1.1, quantiles=(0.025, 0.975),
                      plot_datapoints=False, fill_contours=False, bins=25, levels=(1 - np.exp(-0.5), 1 - np.exp(-2.)),
                      truths=[mcmc_out[0][1], mcmc_out[1][1]], truth_color='r',
                      labels=[r'$A$', r'$T$'], label_kwargs={'fontsize': 20})
        plt.show()

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

    def plot_flux_densities(self, ylog=True, ylim=[1e-3, 5e1]):
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
                                axs[ip, iwv].set_xlabel('Redshift')
                            if not iwv:
                                axs[ip, iwv].set_ylabel('Flux Density (mJy beam^-1)')
                            if ylog:
                                axs[ip, iwv].set_yscale('log')
                            # axs[ip, iwv].set_xlim([0., 8])
                            axs[ip, iwv].set_ylim(ylim)
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
                                axs[ip].set_xlabel('Redshift')
                            if not iwv:
                                axs[ip].set_ylabel('Flux Density (mJy beam^-1)')
                            if ylog:
                                axs[ip].set_yscale('log')
                                axs[ip].set_ylim(ylim)
                            # axs[ip].set_xlim([0., 8])
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
                    axs[iwv].set_xlabel('Redshift')
                    if not iwv:
                        axs[iwv].set_ylabel('Flux Density (Jy)')
                    if ylog:
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
                        axs[ip].set_xlabel('Redshift')
                        axs[ip].set_ylim([9, 13.5])
                        if ip:
                            title_label = 'Star-Forming'
                        else:
                            title_label = 'Quiescent'
                        axs[ip].set_title(title_label)
                else:
                    axs.scatter(z_data_array, sed_lir_vs_z_array[:, im])
                    axs.plot(z_data_array, sed_lir_vs_z_array[:, im], label=mlab)
                    axs.set_ylabel('LIR (M_sun)')
                    axs.set_xlabel('Redshift')
                    axs.set_ylim([9, 13.5])
        else:
            print("Skipping SED plotting because only single wavelength measured.")

    def plot_rest_frame_temperature(self, lir_in, xlim=[0, 10], ylim=[2e1, 140], xlog=False, ylog=True,
                                    print_values=False,  show_fit=True, show_cmb=False, save_path=None,
                                    not_flat_prior=None, save_filename="Tdust.pdf"):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]

        sm = np.zeros(ds)
        zmed = np.zeros(ds)
        t_obs = np.zeros(ds)
        t_err = np.zeros(ds)
        t_rf = np.zeros(ds)
        z_pf = []
        t_pf = []
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            tave = []
            terr = []
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    if ip:
                        if id_label in lir_in['Tobs_dict']:
                            t_obs[iz, im, ip] = lir_in['mcmc_out'][id_label][1][3]
                            t_err[iz, im, ip] = np.sqrt(
                                (lir_in['mcmc_out'][id_label][1][5] - lir_in['mcmc_out'][id_label][1][1]) ** 2 +
                                ((t_obs[iz, im, ip] * (1 + lir_in['dz_median'][id_label][1])) -
                                 (t_obs[iz, im, ip] * (1 + lir_in['dz_median'][id_label][0]))) ** 2)
                            t_rf[iz, im, ip] = t_obs[iz, im, ip] * (1 + lir_in['z_median'][id_label])
                            if not_flat_prior is not None:
                                if id_label not in not_flat_prior:
                                    tave.append(t_rf[iz, im, ip])
                                    terr.append(t_err[iz, im, ip] ** 2)
                            else:
                                tave.append(t_rf[iz, im, ip])
                                terr.append(t_err[iz, im, ip] ** 2)
                            sm[iz, im, ip] = lir_in['m_median'][id_label]
                            zmed[iz, im, ip] = lir_in['z_median'][id_label]

                            z_pf.append(lir_in['z_median'][id_label])
                            t_pf.append(t_rf[iz, im, ip])
            if print_values:
                print(zlab + ' T = {0:0.1f}+= {1:0.1f}'.format(np.mean(tave), np.sqrt(np.mean(terr))))

        plt.figure(figsize=(9, 6))

        color = ['r', 'g', 'b', 'y', 'c']
        for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
            label = "log(M/Msun)=" + '-'.join(mlab.split('_')[2:])
            plt.scatter(zmed[:, im, 1], (t_rf[:, im, 1]), marker='o', s=90, facecolors='none', edgecolors=color[im],
                        label=label)
            plt.errorbar(zmed[:, im, 1], (t_rf[:, im, 1]), t_err[:, im, 1], fmt="." + color[im])

        if not_flat_prior is not None:
            for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
                for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                    id_label = "__".join([zlab, mlab, plab])
                    if id_label in not_flat_prior:
                        # pass
                        plt.scatter(zmed[iz, im, 1], (t_rf[iz, im, 1]), marker='o', s=90, color=color[im])

        z_in = np.linspace(0, zmed[-1, 0, 1] + 1.5)

        if show_cmb:
            plt.plot(z_in, (1 + z_in) * 2.73, '--', c='k', lw=2, label='CMB')

        plt.errorbar(4.5, 47, 5, c='k', markersize=8, marker='s', label='Bethermin+ 2020')
        plt.errorbar(5.5, 38, 8, c='r', markersize=8, marker='s', label='Faisst+ 2020')
        plt.errorbar(7, 52, 11, c='g', markersize=8, marker='s', label='Ferrara+ 2022')
        plt.errorbar(7.15, 54, 10, c='c', markersize=8, marker='s', label='Hashimoto+ 2019')
        plt.errorbar(7.075, 47, 6, c='b', markersize=8, marker='s', label='Sommovigo+ 2022')
        plt.errorbar(8.41, 91, 23, c='y', markersize=8, marker='s', label='Laporte+ 2017, Behrens+ 2018')
        plt.errorbar(8.29, 80, 10, fmt="." + 'm', markersize=12, lolims=True, label='Bakx+ 2020')
        xmod = np.linspace(0, 9)
        plt.plot(xmod, 23 + xmod * ((39 - 23) / 4), label='Viero+ 2013')
        plt.plot(xmod, 32.9 + 4.6 * (xmod - 2), label='Schreiber+ 2018')  # Eqn 15
        plt.plot(xmod, xmod * ((63 - 27) / 9.25) + 27, label='Bouwens+ 2020')

        pfit = np.polyfit(z_pf, t_pf, 2, cov=True)
        if print_values:
            print(pfit[0])
            print(np.sqrt(pfit[1]))

        if show_fit:
            fit_label = "Tfit = {0:.1f} + {1:.1f}z + {2:.1f}z^2".format(pfit[0][2], pfit[0][1], pfit[0][0])
            plt.plot(z_in, pfit[0][2] + pfit[0][1] * z_in + pfit[0][0] * z_in ** 2, '--', lw=2, label=fit_label)

        if xlog:
            plt.xscale('log')
        if ylog:
            plt.yscale('log')
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim([z_in[0], z_in[-1] + .5])
        plt.ylim(ylim)
        plt.xticks(fontsize=14)
        plt.xlabel('Redshift')
        plt.ylabel('Rest-frame Temperature [K]')
        plt.legend(loc='upper left', prop={'size': 10.5})

        if save_path is not None:
            plt.savefig(os.path.join(save_path, save_filename), format="pdf", bbox_inches="tight")

    def plot_dust_mass(self, sfr_dict, ylim=None, show_qt=False,
                       save_path=None, save_filename="Mdust.pdf"):

        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
        z_bins = np.unique(self.config_dict['distance_bins']['redshift'])
        z_mid = [(z_bins[i] + z_bins[i + 1]) / 2 for i in range(len(z_bins) - 1)]

        M_dust = sfr_dict['Mdust_dict']
        M_matrix = np.zeros([*ds])
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    if id_label in M_dust:
                        M_matrix[iz, im, ip] = M_dust[id_label]['50']

        fig = plt.figure(figsize=(9, 6))
        for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
            plt.plot(z_mid, np.log10(M_matrix[:, im, 1]), label=mlab)

        # Tamura 2018
        plt.scatter([8.312], [np.log10(4e6)])

        plt.xlabel('redshift')
        plt.ylabel('log(Mdust/Msun)')
        plt.xlim([0, z_bins[-1] - 0.5])
        # plt.ylim([-3.75, -0.75])
        plt.legend(loc='lower left', frameon=True)
        if save_path is not None:
            plt.savefig(os.path.join(save_path, save_filename), format="pdf", bbox_inches="tight")

    def plot_star_forming_main_sequence(self, lir_in, ylim=[1e-1, 1e4], xlog=False, ylog=True,
                                        save_path=None, save_filename="Tdust.pdf"):
        bin_keys = list(self.config_dict['parameter_names'].keys())
        ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
        sfr = np.zeros(ds)
        sm = np.zeros(ds)
        for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
            for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
                for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
                    id_label = "__".join([zlab, mlab, plab])
                    if ip and id_label in lir_in['lir_dict']:
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
        axs.set_xlabel('Stellar Mass [Msun]')
        axs.set_ylabel('SFR [Msun yr^-1]')
        axs.legend(loc='lower right')
        if save_path is not None:
            plt.savefig(os.path.join(save_path, save_filename), format="pdf", bbox_inches="tight")