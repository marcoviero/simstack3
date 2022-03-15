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
			self.results_dict['sed_bootstrap_results_dict'] = \
				self.populate_sed_dict(catalog_object.catalog_dict['tables'])

			if estimate_mcmcs:
				self.results_dict['lir_dict'] = self.loop_mcmc_sed_estimator(self.results_dict['bootstrap_results_dict'],
																			 catalog_object.catalog_dict['tables'],
																			 mcmc_iterations=mcmc_iterations,
																			 mcmc_discard=mcmc_discard)
		else:
			print("Skipping SED estimates because only single wavelength measured.")
			#self.results_dict['SED_df'] = {'plot_sed': False}

	def sed_density_wrapper(self):
		''' Estimate and save all steps of the luminosity and star-formation rate densities.'''
		pass

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
			outlier_array = np.zeros([len_results_dict_keys, *ds])
			error_array = np.zeros(ds)

			for iboot in np.arange(len_results_dict_keys):
				if not iboot:
					boot_label = 'stacked_flux_densities'
				else:
					boot_label = 'bootstrap_flux_densities_' + str(int(iboot))

				results_object = self.results_dict['band_results_dict'][key][boot_label]
				#pdb.set_trace()

				for z, zval in enumerate(self.config_dict['catalog']['distance_labels']):
					#if 'all_redshifts' in results_object:
					#	zlab = 'all_redshifts'
					#else:
					#	zlab = zval
					for i, ival in enumerate(label_dict[label_keys[1]]):
						if len(label_keys) > 2:
							for j, jval in enumerate(label_dict[label_keys[2]]):
								label = "__".join([zval, ival, jval]).replace('.', 'p')
								# print(label)
								# CHECK THAT LABEL EXISTS FIRST
								if label in results_object:
									# print(label, ' exists')
									flux_array[iboot, z, i, j] = results_object[label].value
									if label+'__bootstrap2' in results_object:
										outlier_array[iboot, z, i, j] = results_object[label+'__bootstrap2'].value
									#	flux_array[iboot, z, i, j] = results_object[label + '__bootstrap2'].value
									#else:
									#	flux_array[iboot, z, i, j] = results_object[label].value

									if len_results_dict_keys == 1:
										error_array[z, i, j] = results_object[label].stderr
						else:
							label = "__".join([zval, ival]).replace('.', 'p')
							if label in results_object:
								# print(label, ' exists')
								flux_array[iboot, z, i] = results_object[label].value
								if label + '__bootstrap2' in results_object:
									outlier_array[iboot, z, i] = results_object[label + '__bootstrap2'].value
								#	flux_array[iboot, z, i] = results_object[label+'__bootstrap2'].value
								#else:
								#	flux_array[iboot, z, i] = results_object[label].value

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
							if np.sum(sed_flux_array[:, z, i, j]):
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

	def populate_sed_dict(self, tables, atonce_object=None):
		''' Reorganize fluxes into SEDs '''
		id_distance = self.config_dict['catalog']['classification']['redshift']['id']
		id_secondary = self.config_dict['catalog']['classification']['stellar_mass']['id']
		split_table = tables['split_table']
		full_table = tables['full_table']
		band_keys = list(self.config_dict['maps'].keys())
		bin_keys = list(self.config_dict['parameter_names'].keys())
		flux_dict = {}
		error_dict = {}
		boot_dict = {}
		ngals_dict = {}
		z_dict = {}
		m_dict = {}
		#lir_dict = {}
		wvs = [self.config_dict['maps'][i]['wavelength'] for i in self.config_dict['maps']]
		results_dict = {'wavelengths': wvs, 'z_median': z_dict, 'm_median': m_dict,	'ngals': ngals_dict,
						'sed_fluxes_dict': flux_dict, 'std_fluxes_dict': error_dict,
						'sed_bootstrap_fluxes_dict': boot_dict} #, 'lir': lir_dict}

		for iz, zlab in enumerate(self.config_dict['parameter_names'][bin_keys[0]]):
			for im, mlab in enumerate(self.config_dict['parameter_names'][bin_keys[1]]):
				for ipop, plab in enumerate(self.config_dict['parameter_names'][bin_keys[2]]):
					id_label = "__".join([zlab, mlab, plab])
					label = "__".join([zlab, mlab, plab]).replace(".", "p")
					ind_gals = (split_table.redshift == iz) & (split_table.stellar_mass == im) & (
							split_table.split_params == ipop)
					ngals_dict[id_label] = np.sum(ind_gals)
					z_median = np.median(full_table[id_distance][ind_gals])
					m_median = np.median(full_table[id_secondary][ind_gals])
					z_dict[id_label] = z_median
					m_dict[id_label] = m_median

					flux_dict[id_label] = np.zeros(len(band_keys))
					boots = np.sum(['bootstrap_flux_densities' in i for i in
									self.results_dict['band_results_dict'][band_keys[0]].keys()])
					boot_dict[id_label] = np.zeros([boots, len(band_keys)])

					for iwv, band_label in enumerate(band_keys):
						for iboot in range(boots):
							flux_label = 'stacked_flux_densities'
							boot_label = '_'.join(['bootstrap_flux_densities', str(int(iboot + 1))])
							if len(self.config_dict['parameter_names']) > 2:
								if not iboot:
									if label in self.results_dict['band_results_dict'][band_label][flux_label]:
										if atonce_object is not None:
											flux_dict[id_label][iwv] = \
											atonce_object.results_dict['band_results_dict'][band_label][flux_label][
												label]
										else:
											flux_dict[id_label][iwv] = \
											self.results_dict['band_results_dict'][band_label][flux_label][label]
								if label in self.results_dict['band_results_dict'][band_label][boot_label]:
									boot_dict[id_label][iboot, iwv] = \
										self.results_dict['band_results_dict'][band_label][boot_label][label]

					error_dict[id_label] = np.std(boot_dict[id_label], axis=0)

		return results_dict

	def populate_results_dict(self, atonce_object=None):
		band_keys = list(self.config_dict['maps'].keys())
		bin_keys = list(self.config_dict['parameter_names'].keys())
		ds = [len(self.config_dict['parameter_names'][i]) for i in bin_keys]
		flux_dict = {}
		boot_dict = {}
		for band_label in band_keys:
			boots = np.sum(['bootstrap_flux_densities' in i for i in
							self.results_dict['band_results_dict'][band_label].keys()])
			flux_array = np.zeros([*ds])
			boot_array = np.zeros([boots, *ds])
			wv = self.config_dict['maps'][band_label]['wavelength']
			for iboot in range(boots):
				flux_label = 'stacked_flux_densities'
				boot_label = '_'.join(['bootstrap_flux_densities', str(int(iboot + 1))])
				#print(boot_label)
				#pdb.set_trace()
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
		return {'flux_densities': flux_array, 'bootstrap_flux_densities': boot_array, 'sed_array_dict': sed_dict,
				'wavelengths': list(boot_dict.keys())}
