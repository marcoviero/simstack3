import pdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from simstacktoolbox import SimstackToolbox
conv_lir_to_sfr = 1.728e-10 / 10 ** 0.23

class SimstackResults(SimstackToolbox):

	def __init__(self, SimstackResultsObject):
		super().__init__()

		dict_list = dir(SimstackResultsObject)
		for i in dict_list:
			if '__' not in i:
				setattr(self, i, getattr(SimstackResultsObject, i))

	def parse_results(self, beta_rj=1.8, catalog_object=None,
					  estimate_mcmcs=False, mcmc_iterations=2500, mcmc_discard=25, plot_seds=False):

		# Sort into fluxes for easy plotting
		fluxes_dict = self.parse_fluxes()

		# Sort into SEDs (if more than one band) for easy plotting
		# Optionally, estimate MCMC SED error region
		if len(fluxes_dict['wavelengths']) > 1:
			self.parse_seds(fluxes_dict, beta_rj=beta_rj)
			self.results_dict['bootstrap_results_dict'] = self.populate_results_dict()

			if estimate_mcmcs:
				self.results_dict['lir_dict'] = self.estimate_mcmc_seds(self.results_dict['bootstrap_results_dict'],
																		catalog_object.catalog_dict['tables']['split_table'],
																		mcmc_iterations=mcmc_iterations,
																		mcmc_discard=mcmc_discard,
																		plot_seds=plot_seds)
		else:
			print("Skipping SED estimates because only single wavelength measured.")
			self.results_dict['SED_df'] = {'plot_sed': False}

	def plot_cib(self, cib_dict=None, tables=None, area_deg2=None):

		if not cib_dict:
			if 'cib_dict' not in self.results_dict:
				self.results_dict['cib_dict'] = self.estimate_cib(area_deg2, split_table=tables['split_table'])
			cib_dict = self.results_dict['cib_dict']

		nuInu = cib_dict['nuInu']
		wvs = cib_dict['wavelengths']
		bin_keys = list(self.config_dict['parameter_names'].keys())

		fig, axs = plt.subplots(1, 2, figsize=(16, 6))
		ls = [':', '-']
		for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
			for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
				axs[0].plot(wvs, np.sum(nuInu[:, iz, :, ip], axis=1), ls[ip], label=zlab)
				axs[0].set_xscale('log')
				axs[0].set_yscale('log')
				axs[0].set_ylim([1e-3, 1e2])

		axs[0].plot(wvs, np.sum(np.sum(nuInu[:, :, :, ip], axis=1), axis=1), ls[ip], label=zlab, lw=3)
		# axs[0].legend(loc='upper right')

		for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
			for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
				axs[1].plot(wvs, np.sum(nuInu[:, :, im, ip], axis=1), ls[ip], label=zlab)
				axs[1].set_xscale('log')
				axs[1].set_yscale('log')
				axs[1].set_ylim([1e-3, 1e2])

		axs[1].plot(wvs, np.sum(np.sum(nuInu[:, :, :, ip], axis=1), axis=1), ls[ip], label=zlab, lw=3)


	def plot_total_lird(self, tables, area_deg2, total_lird_dict=None, lird_dict=None, lir_dict=None,
						plot_lird=False, plot_sfrd=True):

		if not total_lird_dict:
			if 'total_lird_dict' not in self.results_dict:
				if not lird_dict:
					if 'lird_dict' not in self.results_dict:
						if not lir_dict:
							if 'lir_dict' not in self.results_dict:
								self.results_dict['lir_dict'] = self.estimate_mcmc_seds(self.results_dict['bootstrap_results_dict'],
									tables['split_table'], plot_seds=False)
							lir_dict = self.results_dict['lir_dict']
						self.results_dict['lird_dict'] = self.estimate_luminosity_density(area_deg2, tables, lir_dict)
					lird_dict = self.results_dict['lird_dict']
				self.results_dict['total_lird_dict'] = self.estimate_total_lird(lird_dict)
			total_lird_dict = self.results_dict['total_lird_dict']

		z_bins = np.unique(self.config_dict['distance_bins']['redshift'])
		z_mid = [(z_bins[i] + z_bins[i + 1]) / 2 for i in range(len(z_bins) - 1)]
		if plot_lird:
			lird_total = total_lird_dict['lird_total']
			lird_error = total_lird_dict['lird_total_error']
			fig = plt.figure(figsize=(9, 6))
			bin_keys = list(self.config_dict['parameter_names'].keys())
			for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
				label = "Star-Forming logM=" + '-'.join(mlab.split('_')[2:])
				plt.plot(z_mid, lird_dict['50'][:, im, 1], '-', label=label)
			for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
				label = "Quiescent logM=" + '-'.join(mlab.split('_')[2:])
				plt.plot(z_mid, lird_dict['50'][:, im, 0], '--', label=label)

			plt.fill_between(z_mid, np.log10([np.max([i, 0.01]) for i in lird_total - lird_error]),
							 np.log10(lird_total + lird_error), facecolor='c', alpha=0.3, edgecolor='c')
			plt.plot(z_mid, np.log10(lird_total), '-', label='total', color='c')
			plt.xlabel('redshift')
			plt.ylabel('IR Luminosity Density [Lsun Mpc3]')
			plt.xlim([0, 9])
			plt.ylim([4.5, 9])
			plt.legend()

		if plot_sfrd:
			sfrd_total = total_lird_dict['sfrd_total']
			sfrd_error = total_lird_dict['sfrd_total_error']
			uvsfr_total = total_lird_dict['uvsfr_total']
			fig = plt.figure(figsize=(9, 6))
			bin_keys = list(self.config_dict['parameter_names'].keys())
			for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
				label = "Star-Forming logM=" + '-'.join(mlab.split('_')[2:])
				plt.plot(z_mid, np.log10(conv_lir_to_sfr * 10 ** lird_dict['50'][:, im, 1]), '-', label=label)
			for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
				label = "Quiescent logM=" + '-'.join(mlab.split('_')[2:])
				plt.plot(z_mid, np.log10(conv_lir_to_sfr * 10 ** lird_dict['50'][:, im, 0]), '--', label=label)

			plt.fill_between(z_mid, np.log10([np.max([i, 0.00001]) for i in sfrd_total - sfrd_error]),
							 np.log10(sfrd_total + sfrd_error), facecolor='c', alpha=0.3, edgecolor='c')
			plt.plot(z_mid, np.log10(sfrd_total), '-', label='total', color='c')

			plt.plot(z_mid, np.log10(uvsfr_total), '--', lw=2, label='LePhare SFRD', color='y')

			plt.xlabel('redshift')
			plt.ylabel('SFR Density [Msun/yr Mpc3]')
			plt.xlim([0, 9])
			plt.ylim([-5, -1])
			plt.legend()

	def parse_fluxes(self):

		wavelength_keys = list(self.results_dict['band_results_dict'].keys())
		wavelengths = []
		split_dict = self.config_dict['catalog']['classification']
		# split_type = split_dict.pop('split_type')
		label_keys = list(split_dict.keys())
		label_dict = self.config_dict['parameter_names']
		ds = [len(label_dict[k]) for k in label_dict]

		sed_flux_array = np.zeros([len(wavelength_keys), *ds])
		sed_error_array = np.zeros([len(wavelength_keys), *ds])

		for k, key in enumerate(wavelength_keys):
			self.results_dict['band_results_dict'][key]['raw_fluxes_dict'] = {}

			wavelengths.append(self.config_dict['maps'][key]['wavelength'])

			len_results_dict_keys = np.sum(['flux_densities' in i for i in self.results_dict['band_results_dict'][key].keys()])
			flux_array = np.zeros([len_results_dict_keys, *ds])
			error_array = np.zeros(ds)

			for iboot in np.arange(len_results_dict_keys):
				if not iboot:
					boot_label = 'stacked_flux_densities'
				else:
					boot_label = 'bootstrap_flux_densities_' + str(int(iboot))

				results_object = self.results_dict['band_results_dict'][key][boot_label]

				for z, zval in enumerate(self.config_dict['catalog']['distance_labels']):
					if 'all_redshifts' in results_object:
						zlab = 'all_redshifts'
					else:
						zlab = zval
					for i, ival in enumerate(label_dict[label_keys[1]]):
						if len(label_keys) > 2:
							for j, jval in enumerate(label_dict[label_keys[2]]):
								label = "__".join([zval, ival, jval]).replace('.', 'p')
								# print(label)
								# CHECK THAT LABEL EXISTS FIRST
								if label in results_object:
									# print(label, ' exists')
									flux_array[iboot, z, i, j] = results_object[label].value
									if len_results_dict_keys == 1:
										error_array[z, i, j] = results_object[label].stderr
						else:
							label = "__".join([zval, ival]).replace('.', 'p')
							if label in results_object:
								# print(label, ' exists')
								flux_array[iboot, z, i] = results_object[label].value
								if len_results_dict_keys == 1:
									error_array[z, i] = results_object[label].stderr

			z_bins = [i.replace('p', '.').split('_')[1:] for i in self.config_dict['catalog']['distance_labels']]
			z_mid = [(float(i[0]) + float(i[1])) / 2 for i in z_bins]

			if len(label_keys) > 2:
				self.results_dict['band_results_dict'][key]['raw_fluxes_dict']['flux_df'] = {}
				self.results_dict['band_results_dict'][key]['raw_fluxes_dict']['error_df'] = {}
				for j, jval in enumerate(label_dict[label_keys[2]]):
					self.results_dict['band_results_dict'][key]['raw_fluxes_dict']['flux_df'][jval] = \
						pd.DataFrame(flux_array[0, :, :, j],
									 columns=[label_dict[label_keys[1]]], index=z_mid)
					if len_results_dict_keys > 1:
						self.results_dict['band_results_dict'][key]['raw_fluxes_dict']['error_df'][jval] = \
							pd.DataFrame(np.std(flux_array[1:, :, :, j], axis=0),
										 columns=[label_dict[label_keys[1]]], index=z_mid)
					else:
						self.results_dict['band_results_dict'][key]['raw_fluxes_dict']['error_df'][jval] = \
							pd.DataFrame(error_array[:, :, j],
										 columns=[label_dict[label_keys[1]]], index=z_mid)
			else:
				self.results_dict['band_results_dict'][key]['raw_fluxes_dict']['flux_df'] = \
					pd.DataFrame(flux_array[0, :, :],
								 columns=[label_dict[label_keys[1]]], index=z_mid)
				if len_results_dict_keys > 1:
					self.results_dict['band_results_dict'][key]['raw_fluxes_dict']['error_df'] = \
						pd.DataFrame(np.std(flux_array[1:, :, :], axis=0),
									 columns=[label_dict[label_keys[1]]], index=z_mid)
				else:
					self.results_dict['band_results_dict'][key]['raw_fluxes_dict']['error_df'] = \
						pd.DataFrame(error_array[:, :],
									 columns=[label_dict[label_keys[1]]], index=z_mid)

			z_dict = {'flux_density': {}, 'std_error': {}, 'redshift': []}
			for z, zval in enumerate(label_dict[label_keys[0]]):
				z_dict['flux_density'][zval] = {}
				z_dict['std_error'][zval] = {}
				z_dict['redshift'].append(zval)
				if len(label_keys) > 2:
					for j, jval in enumerate(label_dict[label_keys[2]]):
						z_dict['flux_density'][zval][jval] = flux_array[0, z, :, j]
						z_dict['std_error'][zval][jval] = np.std(flux_array[1:, z, :, j], axis=0)
				else:
					z_dict['flux_density'][zval] = flux_array[z, :]
					z_dict['std_error'][zval] = error_array[z, :]

			m_dict = {'flux_density': {}, 'std_error': {}, 'stellar_mass': []}
			for i, ival in enumerate(label_dict[label_keys[1]]):
				m_dict['flux_density'][ival] = {}
				m_dict['std_error'][ival] = {}
				m_dict['stellar_mass'].append(ival)

				if len(label_keys) > 2:
					for j, jval in enumerate(label_dict[label_keys[2]]):
						m_dict['flux_density'][ival][jval] = flux_array[0, :, i, j]
						m_dict['std_error'][ival][jval] = error_array[:, i, j]
				else:
					m_dict['flux_density'][ival] = flux_array[0, :, i]
					m_dict['std_error'][ival] = error_array[:, i]
			if len(label_keys) > 2:
				self.results_dict['band_results_dict'][key]['raw_fluxes_dict'][label_keys[0]] = z_dict
				self.results_dict['band_results_dict'][key]['raw_fluxes_dict'][label_keys[1]] = m_dict
			else:
				self.results_dict['band_results_dict'][key]['raw_fluxes_dict'] = z_dict
				self.results_dict['band_results_dict'][key]['raw_fluxes_dict'] = m_dict

			sed_flux_array[k] = flux_array[0]
			sed_error_array[k] = np.std(flux_array, axis=0)

		return {'flux': sed_flux_array, 'error': sed_error_array, 'wavelengths': wavelengths}

	def parse_seds(self, results_dict_in, beta_rj=1.8):

		wavelengths = results_dict_in['wavelengths']
		sed_flux_array = results_dict_in['flux']
		sed_error_array = results_dict_in['error']

		split_dict = self.config_dict['catalog']['classification']
		# split_type = split_dict.pop('split_type')
		label_keys = list(split_dict.keys())
		label_dict = self.config_dict['parameter_names']

		# Organize into SEDs
		if len(wavelengths) == 1:
			print("Skipping SED estimates because only single wavelength measured.")
			self.results_dict['SED_df'] = {'plot_sed': False}
		else:
			self.results_dict['SED_df'] = {'flux_density': {}, 'std_error': {}, 'SED': {}, 'LIR': {},
										   'wavelengths': wavelengths, 'plot_sed': True}
			for z, zlab in enumerate(label_dict[label_keys[0]]):
				z_mid = np.mean([float(i) for i in zlab.split('_')[1:]])
				if len(label_keys) > 2:
					for j, jlab in enumerate(label_dict[label_keys[2]]):
						if zlab not in self.results_dict['SED_df']['flux_density']:
							self.results_dict['SED_df']['flux_density'][zlab] = {}
							self.results_dict['SED_df']['std_error'][zlab] = {}
							self.results_dict['SED_df']['LIR'][zlab] = {}
							self.results_dict['SED_df']['SED'][zlab] = {}

						self.results_dict['SED_df']['flux_density'][zlab][jlab] = \
							pd.DataFrame(sed_flux_array[:, z, :, j], index=wavelengths,
										 columns=label_dict[label_keys[1]])
						self.results_dict['SED_df']['std_error'][zlab][jlab] = \
							pd.DataFrame(sed_error_array[:, z, :, j], index=wavelengths,
										 columns=label_dict[label_keys[1]])

						self.results_dict['SED_df']['LIR'][zlab][jlab] = {}
						self.results_dict['SED_df']['SED'][zlab][jlab] = {}
						for i, ilab in enumerate(label_dict[label_keys[1]]):
							tst_m = self.fast_sed_fitter(wavelengths, sed_flux_array[:, z, i, j],
														 sed_error_array[:, z, i, j]**2,
														 betain=beta_rj, redshiftin=z_mid)
							tst_LIR = self.fast_Lir(tst_m, z_mid)
							self.results_dict['SED_df']['LIR'][zlab][jlab][ilab] = tst_LIR.value
							self.results_dict['SED_df']['SED'][zlab][jlab][ilab] = tst_m
				else:
					self.results_dict['SED_df']['flux_density'][zlab] = \
						pd.DataFrame(sed_flux_array[:, z, :], index=wavelengths, columns=label_dict[label_keys[1]])
					self.results_dict['SED_df']['std_error'][zlab] = \
						pd.DataFrame(sed_error_array[:, z, :], index=wavelengths, columns=label_dict[label_keys[1]])

					for i, ilab in enumerate(label_dict[label_keys[1]]):
						tst_m = self.fast_sed_fitter(wavelengths, sed_flux_array[:, z, i], sed_error_array[:, z, i],
													 betain=beta_rj, redshiftin=z_mid)
						tst_LIR = self.fast_Lir(tst_m, z_mid)

						if zlab not in self.results_dict['SED_df']['LIR']:
							self.results_dict['SED_df']['LIR'][zlab] = {}
							self.results_dict['SED_df']['SED'][zlab] = {}

						self.results_dict['SED_df']['LIR'][zlab][ilab] = tst_LIR.value
						self.results_dict['SED_df']['SED'][zlab][ilab] = tst_m

	def populate_results_dict(self, atonce_object=None):
		band_keys = list(self.config_dict['maps'].keys())
		bin_keys = list(self.config_dict['parameter_names'].keys())
		flux_dict = {}
		boot_dict = {}
		for band_label in band_keys:
			boots = len(self.results_dict['band_results_dict'][band_label].keys()) - 2
			ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
			flux_array = np.zeros([*ds])
			boot_array = np.zeros([boots, *ds])
			wv = self.config_dict['maps'][band_label]['wavelength']
			for iboot in range(boots):
				flux_label = 'stacked_flux_densities'
				boot_label = '_'.join(['bootstrap_flux_densities', str(int(iboot + 1))])
				# print(boot_label)
				for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
					for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
						if len(self.config_dict['parameter_names']) > 2:
							for ipop, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
								label = "__".join([zlab, mlab, plab]).replace(".", "p")
								if not iboot:
									if label in self.results_dict['band_results_dict'][band_label][flux_label]:
										if atonce_object is not None:
											#pdb.set_trace()
											flux_array[iz, im, ipop] = \
											atonce_object.results_dict['band_results_dict'][band_label][flux_label][
												label]
										else:
											flux_array[iz, im, ipop] = \
											self.results_dict['band_results_dict'][band_label][flux_label][label]
								if label in self.results_dict['band_results_dict'][band_label][boot_label]:
									boot_array[iboot, iz, im, ipop] = \
									self.results_dict['band_results_dict'][band_label][boot_label][label]

			flux_dict[wv] = flux_array
			boot_dict[wv] = boot_array

		# Group into SEDs
		sf_sed_array = np.zeros([boots, len(boot_dict), len(self.config_dict['parameter_names'][bin_keys[0]]),
								 len(self.config_dict['parameter_names'][bin_keys[1]])])
		qt_sed_array = np.zeros([boots, len(boot_dict), len(self.config_dict['parameter_names'][bin_keys[0]]),
								 len(self.config_dict['parameter_names'][bin_keys[1]])])
		sf_sed_measurement = np.zeros([len(boot_dict), len(self.config_dict['parameter_names'][bin_keys[0]]),
									   len(self.config_dict['parameter_names'][bin_keys[1]])])
		qt_sed_measurement = np.zeros([len(boot_dict), len(self.config_dict['parameter_names'][bin_keys[0]]),
									   len(self.config_dict['parameter_names'][bin_keys[1]])])

		for iwv, wv in enumerate(boot_dict):
			sf_sed_array[:, iwv, :] = boot_dict[wv][:, :, :, 1]
			qt_sed_array[:, iwv, :] = boot_dict[wv][:, :, :, 0]
			sf_sed_measurement[iwv, :] = flux_dict[wv][:, :, 1]
			qt_sed_measurement[iwv, :] = flux_dict[wv][:, :, 0]
		sed_dict = {'sf': {'sed_measurement': sf_sed_measurement, 'sed_bootstrap': sf_sed_array},
					'qt': {'sed_measurement': qt_sed_measurement, 'sed_bootstrap': qt_sed_array}}
		return {'flux_densities': flux_array, 'bootstrap_flux_densities': boot_array, 'sed_dict': sed_dict,
				'wavelengths': list(boot_dict.keys())}

	def plot_mcmc_seds(self, lir_dict, bootstrap_dict=None):
		bin_keys = list(self.config_dict['parameter_names'].keys())
		wvs = lir_dict['wavelengths']
		wv_array = self.loggen(8, 1000, 100)
		ngals = lir_dict['ngals']
		plen = len(self.config_dict['parameter_names'][bin_keys[1]]) * len(self.config_dict['parameter_names'][bin_keys[2]])
		zlen = len(self.config_dict['parameter_names'][bin_keys[0]])
		z_bins = lir_dict['redshift_bins']
		z_mid = [(z_bins[i] + z_bins[i + 1]) / 2 for i in range(len(z_bins) - 1)]
		fig, axs = plt.subplots(plen, zlen, figsize=(33, 20))

		for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
			for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
				for ip, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):

					y = lir_dict['y']["_".join([zlab, mlab, plab])]
					yerr = lir_dict['yerr']["_".join([zlab, mlab, plab])]
					sed_params = self.fast_sed_fitter(wvs, y, yerr)
					sed_array = self.fast_sed(sed_params, wv_array)

					plot_true = True
					if np.sum(y):

						mcmc_out = lir_dict['mcmc_dict']["_".join([zlab, mlab, plab])]
						lir_16 = lir_dict['16']
						lir_25 = lir_dict['25']
						lir_32 = lir_dict['32']
						lir_50 = lir_dict['50']
						lir_68 = lir_dict['68']
						lir_75 = lir_dict['75']
						lir_84 = lir_dict['84']
						mcmc_16 = self.graybody_fn([mcmc_out[0][0], mcmc_out[1][0]], wv_array)
						mcmc_25 = self.graybody_fn([mcmc_out[0][1], mcmc_out[1][1]], wv_array)
						mcmc_32 = self.graybody_fn([mcmc_out[0][2], mcmc_out[1][2]], wv_array)
						mcmc_50 = self.graybody_fn([mcmc_out[0][3], mcmc_out[1][3]], wv_array)
						mcmc_68 = self.graybody_fn([mcmc_out[0][4], mcmc_out[1][4]], wv_array)
						mcmc_75 = self.graybody_fn([mcmc_out[0][5], mcmc_out[1][5]], wv_array)
						mcmc_84 = self.graybody_fn([mcmc_out[0][6], mcmc_out[1][6]], wv_array)

					else:
						plot_true = False

					if plot_true:
						colors = ['y', 'c', 'b', 'r']
						if ip:
							ix = im
						else:
							ix = im + 4

						axs[ix, iz].plot(wv_array, sed_array[0] * 1e3, color='k', lw=0.5)
						if ix == 0:
							axs[ix, iz].set_title(zlab)

						mcmc_label = "LIR={0:.1f}, T={1:.1f}".format(lir_50[iz, im, ip],
																	 mcmc_out[1][3] * (1 + z_mid[iz]))
						axs[ix, iz].plot(wv_array, mcmc_50[0] * 1e3, color='c', lw=0.8, label=mcmc_label)
						axs[ix, iz].fill_between(wv_array, mcmc_25[0] * 1e3, mcmc_75[0] * 1e3, facecolor='c',
												 alpha=0.3, edgecolor='c')

						axs[ix, iz].legend(loc='upper left', frameon=False)

						axs[ix, iz].text(9.0e0, 2e1, "Ngals={0:.0f}".format(ngals[iz, im, ip]))

						for iwv, wv in enumerate(wvs):
							if wv in [24, 70]:
								color = 'b'
							elif wv in [100, 160]:
								color = 'g'
							elif wv in [250, 350, 500]:
								color = 'r'
							elif wv in [850]:
								color = 'y'

							if ip:
								pop = 'sf'
							else:
								pop = 'qt'

							if bootstrap_dict is not None:
								for iboot in range(len(bootstrap_dict['sed_dict'][pop]['sed_bootstrap'][iwv, :, iz, im])):
									axs[ix, iz].scatter(wv, bootstrap_dict['sed_dict'][pop]['sed_bootstrap'][
										iboot, iwv, iz, im] * 1e3, color=color, alpha=0.1)

							axs[ix, iz].scatter(wv, y[iwv] * 1e3, marker='o', s=90, facecolors='none', edgecolors=color)
							axs[ix, iz].errorbar(wv, y[iwv] * 1e3, yerr=np.sqrt(np.diag(yerr)[iwv]) * 1e3,
												 fmt="." + color, capsize=0)

						axs[ix, iz].set_xscale('log')
						axs[ix, iz].set_yscale('log')
						axs[ix, iz].set_ylim([1e-2, 5e2])

	def plot_seds(self):
		colors = ['y', 'c', 'b', 'r']
		if self.results_dict['SED_df']['plot_sed']:
			zlen = len(self.results_dict['SED_df']['flux_density'])
			if len(self.config_dict['parameter_names']) == 3:
				zlen = len(self.results_dict['SED_df']['flux_density'])
				plen = 2
				fig, axs = plt.subplots(plen, zlen, figsize=(36, 10))
				for z, zlab in enumerate(self.results_dict['SED_df']['flux_density']):
					zmid = 0.5 * np.sum([float(i) for i in zlab.split('_')[-2:]])
					for p, plab in enumerate(self.results_dict['SED_df']['flux_density'][zlab]):
						# pdb.set_trace()
						sed = self.results_dict['SED_df']['flux_density'][zlab][plab]
						std = self.results_dict['SED_df']['std_error'][zlab][plab]
						for im, mlab in enumerate(sed):
							axs[p, z].scatter(sed.index, sed[mlab], color=colors[im])
							axs[p, z].errorbar(sed.index, sed[mlab], std[mlab], 0, 'none', color=colors[im])
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
								axs[p, z].plot(wv_array, sed_array[0], label=line_label, color=colors[im])
								axs[p, z].legend(loc='upper right')
							else:
								axs[p, z].plot(wv_array, sed_array[0], color=colors[im])

							if not p:
								axs[p, z].set_title(zlab)
							axs[p, z].set_xscale('log')
							axs[p, z].set_yscale('log')
							axs[p, z].set_xlim([10, 1000])
							axs[p, z].set_ylim([1e-5, 5e-1])
			else:
				fig, axs = plt.subplots(1, zlen, figsize=(36, 10))
				for z, zlab in enumerate(self.results_dict['SED_df']['flux_density']):
					zmid = 0.5 * np.sum([float(i) for i in zlab.split('_')[-2:]])
					for m, mlab in enumerate(self.results_dict['SED_df']['flux_density'][zlab]):
						sed = self.results_dict['SED_df']['flux_density'][zlab][mlab]
						std = self.results_dict['SED_df']['std_error'][zlab][mlab]
						axs[z].scatter(sed.index, sed.values)

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
							axs[z].plot(wv_array, sed_array[0], label=line_label)
							axs[z].legend(loc='upper right')
						else:
							axs[z].plot(wv_array, sed_array[0])

						if not m:
							axs[z].set_title(zlab)
						axs[z].set_xscale('log')
						axs[z].set_yscale('log')
						axs[z].set_xlim([10, 1000])
						axs[z].set_ylim([1e-5, 5e-1])
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

			size_lir_vs_z_array = [len(self.config_dict['parameter_names'][i]) for i in self.config_dict['parameter_names']]
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
						axs[ip].plot(z_data_array, sed_lir_vs_z_array[:, im, ip],label=mlab)
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

	def plot_rest_frame_temperature(self, tables, lir_in):
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

		axs.plot(np.linspace(0,9),(1+np.linspace(0,9))*2.73,'--',label='CMB')
		# axs.set_xscale('log')
		axs.set_yscale('log')
		axs.set_xlabel('redshift')
		axs.set_ylabel('T_restframe')
		axs.legend(loc='upper left')

	def plot_star_forming_main_sequence(self, tables, lir_in):
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
		axs.set_xlabel('Stellar Mass [Mstar]')
		axs.set_ylabel('SFR [Mstar/yr]')
		axs.legend(loc='lower right')