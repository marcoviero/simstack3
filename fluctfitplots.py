import pdb
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from lmfit import Parameters, minimize, fit_report
import corner

from simstacktoolbox import SimstackToolbox
conv_lir_to_sfr = 1.728e-10 / 10 ** 0.23
a_nu_flux_to_mass = 6.7e19
sigma_upper_limit = 3

font = {'size': 14}
plt.rc('font', **font)

class FluctFitPlots(SimstackToolbox):

    def __init__(self, FluctFitPlotsObject):
        super().__init__()

        dict_list = dir(FluctFitPlotsObject)
        for i in dict_list:
            if '__' not in i:
                setattr(self, i, getattr(FluctFitPlotsObject, i))

    def scatterplot(self,
                    xfield=None,
                    yfield=None,
                    xlim=None, ylim=None,
                    legend=True):
        fig, ax = plt.subplots(1, 1, figsize=(9, 5))
        sns.set_color_codes(palette='pastel')
        fig = sns.scatterplot(Zs_sf[idx_keep_sf], Ts_sf[idx_keep_sf] - 1, hue=Ms_sf[idx_keep_sf])
        # fig.set(ylim=(-38,-32))
        fig.set(ylabel='Tdust')
        fig.set(xlabel='redshift')
        zs_model = np.linspace(0, 5, 20)
        ms_list = [9.6, 10, 10.4, 10.8, 11.2]
        as_list = [.1, .5, .9]
        as_ls = ['-', '--', ':']
        for i, im in enumerate(ms_list):
            ms_model = np.zeros_like(zs_model) + im
            try:
                for j, ja in enumerate(as_list):
                    as_model = np.zeros_like(zs_model) + ja
                    T_mod_plot = self.model_A_or_Tdust(vT_just_sf,
                                                  np.array([zs_model.tolist(), ms_model.tolist(), as_model.tolist()]))
                    plt.plot(zs_model, T_mod_plot, as_ls[j])
            except:
                T_mod_plot = self.model_A_or_Tdust(vT_just_sf, np.array([zs_model.tolist(), ms_model.tolist()]))
                plt.plot(zs_model, T_mod_plot)

        zs_model = np.linspace(0, 5, 20)
        plt.plot(zs_model, 27 * ((1 + zs_model) / (1 + 1)) ** (0.4), '-.k', lw=2.5, label='Viero 2013');
        Tv22 = 23.8 + 2.7 * zs_model + 0.9 * zs_model ** 2
        plt.plot(zs_model, Tv22, '--k', lw=2.5, label='Viero 2022');
        plt.legend();