import pdb
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simstacktoolbox import SimstackToolbox

class SimstackResults(SimstackToolbox):

	def __init__(self, SimstackResultsObject):
		super().__init__()

		dict_list = dir(SimstackResultsObject)
		for i in dict_list:
			if '__' not in i:
				setattr(self, i, getattr(SimstackResultsObject, i))

	def parse_results(self, beta_rj=1.8):

		self.results_dict = {'maps_dict': {}}
		wavelength_keys = list(self.maps_dict.keys())
		wavelengths = []
		#split_dict = json.loads(self.config_dict['catalog']['classification'])
		split_dict = self.config_dict['catalog']['classification']
		#split_type = split_dict.pop('split_type')
		#pdb.set_trace()
		label_keys = list(split_dict.keys())
		label_dict = self.config_dict['parameter_names']
		ds = [len(label_dict[k]) for k in label_dict]

		sed_flux_array = np.zeros([len(wavelength_keys), *ds])
		sed_error_array = np.zeros([len(wavelength_keys), *ds])
		for k, key in enumerate(wavelength_keys):
			self.results_dict['maps_dict'][key] = {}
			self.results_dict['maps_dict'][key]['wavelength'] = self.maps_dict[key]['wavelength']
			wavelengths.append(self.maps_dict[key]['wavelength'])

			flux_array = np.zeros(ds)
			error_array = np.zeros(ds)
			results_object = self.maps_dict[key]['stacked_flux_densities']

			for z, zval in enumerate(self.config_dict['catalog']['distance_labels']):
				if 'all_redshifts' in results_object:
					zlab = 'all_redshifts'
				else:
					zlab = zval
				for i, ival in enumerate(label_dict[label_keys[1]]):
					if len(label_keys) > 2:
						for j, jval in enumerate(label_dict[label_keys[2]]):
							label = "__".join([zval, ival, jval]).replace('.', 'p')
							#print(label)
							# CHECK THAT LABEL EXISTS FIRST
							if label in results_object[zlab].params:
								flux_array[z, i, j] = results_object[zlab].params[label].value
								#scale_errors = 1.0
								#if 'pacs' in key:
								#	scale_errors = 1e2
								#elif 'mips' in key:
								#	scale_errors = 1e1
								try:
									error_array[z, i, j] = results_object[zlab].params[label].stderr #* scale_errors
								except:
									error_array[z, i, j] = results_object[zlab].params[label].value
									print(key)
									print(label)
									print(results_object[zlab].params[label])
									#pdb.set_trace()
					else:
						label = "__".join([zval, ival]).replace('.', 'p')
						#print(label)
						if label in results_object[zlab].params:
							flux_array[z, i] = results_object[zlab].params[label].value
							error_array[z, i] = results_object[zlab].params[label].stderr

			z_bins = [i.replace('p', '.').split('_')[1:] for i in self.config_dict['catalog']['distance_labels']]
			z_mid = [(float(i[0]) + float(i[1]))/2 for i in z_bins]

			self.results_dict['maps_dict'][key]['results_df'] = {}
			if len(label_keys) > 2:
				self.results_dict['maps_dict'][key]['results_df']['flux_df'] = {}
				self.results_dict['maps_dict'][key]['results_df']['error_df'] = {}
				for j, jval in enumerate(label_dict[label_keys[2]]):
					self.results_dict['maps_dict'][key]['results_df']['flux_df'][jval] = \
						pd.DataFrame(flux_array[:, :, j],
									 columns=[label_dict[label_keys[1]]], index=z_mid) # , index = [label_dict[label_keys[0]]])
					self.results_dict['maps_dict'][key]['results_df']['error_df'][jval] = \
						pd.DataFrame(error_array[:, :, j],
									 columns=[label_dict[label_keys[1]]], index=z_mid)
			else:
				self.results_dict['maps_dict'][key]['results_df']['flux_df'] = \
					pd.DataFrame(flux_array[:, :],
								 columns=[label_dict[label_keys[1]]], index=z_mid)
				self.results_dict['maps_dict'][key]['results_df']['error_df'] = \
					pd.DataFrame(error_array[:, :],
								 columns=[label_dict[label_keys[1]]], index=z_mid)

			z_dict = {'flux_density': {}, 'std_error': {}, 'redshift': []}
			for z, zval in enumerate(results_object):
				z_dict['flux_density'][zval] = {}
				z_dict['std_error'][zval] = {}
				z_dict['redshift'].append(zval)
				if len(label_keys) > 2:
					for j, jval in enumerate(label_dict[label_keys[2]]):
						z_dict['flux_density'][zval][jval] = flux_array[z, :, j]
						z_dict['std_error'][zval][jval] = error_array[z, :, j]
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
						m_dict['flux_density'][ival][jval] = flux_array[:, i, j]
						m_dict['std_error'][ival][jval] = error_array[:, i, j]
				else:
					m_dict['flux_density'][ival] = flux_array[:, i]
					m_dict['std_error'][ival] = error_array[:, i]

			self.results_dict['maps_dict'][key][label_keys[0]] = z_dict
			self.results_dict['maps_dict'][key][label_keys[1]] = m_dict

			sed_flux_array[k, :, :] = flux_array
			sed_error_array[k, :, :] = error_array

		# Organize into SEDs
		if len(wavelengths) == 1:
			print("Skipping SED estimates because only single wavelength measured.")
			self.results_dict['SED_df'] = {'plot_sed': False}
		else:

			self.results_dict['SED_df'] = {'flux_density': {}, 'std_error': {}, 'SED': {}, 'LIR': {},
										   'wavelengths': wavelengths, 'plot_sed': True}
			for z, zlab in enumerate(label_dict[label_keys[0]]):
				if len(label_keys) > 2:
					for j, jlab in enumerate(label_dict[label_keys[2]]):
						if zlab not in self.results_dict['SED_df']['flux_density']:
							self.results_dict['SED_df']['flux_density'][zlab] = {}
							self.results_dict['SED_df']['std_error'][zlab] = {}
							self.results_dict['SED_df']['LIR'][zlab] = {}
							self.results_dict['SED_df']['SED'][zlab] = {}

						self.results_dict['SED_df']['flux_density'][zlab][jlab] = \
							pd.DataFrame(sed_flux_array[:, z, :, j], index=wavelengths, columns=label_dict[label_keys[1]])
						self.results_dict['SED_df']['std_error'][zlab][jlab] = \
							pd.DataFrame(sed_error_array[:, z, :, j], index=wavelengths, columns=label_dict[label_keys[1]])

						self.results_dict['SED_df']['LIR'][zlab][jlab] = {}
						self.results_dict['SED_df']['SED'][zlab][jlab] = {}
						for i, ilab in enumerate(label_dict[label_keys[1]]):
							tst_m = self.fast_sed_fitter(wavelengths, sed_flux_array[:, z, i, j], sed_error_array[:, z, i, j],
														 betain=beta_rj, redshiftin=z_mid[z])
							tst_LIR = self.fast_Lir(tst_m, z_mid[z])
							self.results_dict['SED_df']['LIR'][zlab][jlab][ilab] = tst_LIR.value
							self.results_dict['SED_df']['SED'][zlab][jlab][ilab] = tst_m
				else:
					self.results_dict['SED_df']['flux_density'][zlab] = \
						pd.DataFrame(sed_flux_array[:, z, :], index=wavelengths, columns=label_dict[label_keys[1]])
					self.results_dict['SED_df']['std_error'][zlab] = \
						pd.DataFrame(sed_error_array[:, z, :], index=wavelengths, columns=label_dict[label_keys[1]])

					for i, ilab in enumerate(label_dict[label_keys[1]]):
						tst_m = self.fast_sed_fitter(wavelengths, sed_flux_array[:, z, i], sed_error_array[:, z, i],
													 betain=beta_rj, redshiftin=z_mid[z])
						tst_LIR = self.fast_Lir(tst_m, z_mid[z])

						if zlab not in self.results_dict['SED_df']['LIR']:
							self.results_dict['SED_df']['LIR'][zlab] = {}
							self.results_dict['SED_df']['SED'][zlab] = {}

						self.results_dict['SED_df']['LIR'][zlab][ilab] = tst_LIR.value
						self.results_dict['SED_df']['SED'][zlab][ilab] = tst_m

		#pdb.set_trace()

	def plot_seds(self):
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
						for mlab in sed:
							axs[p, z].scatter(sed.index, sed[mlab])
							#axs[p, z].errorbar(sed.index, sed[mlab], std[mlab], 0, 'none')
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
								axs[p, z].plot(wv_array, sed_array[0], label=line_label)
								axs[p, z].legend(loc='upper right')
							else:
								axs[p, z].plot(wv_array, sed_array[0])

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
						#pdb.set_trace()
		else:
			print("Skipping SED plotting because only single wavelength measured.")

	def plot_flux_densities(self):
		wv_keys = list(self.results_dict['maps_dict'].keys())
		wlen = len(wv_keys)
		if len(self.config_dict['parameter_names']) == 3:
			plen = 2
			fig, axs = plt.subplots(plen, wlen, figsize=(22, 10))
			for iwv, wlab in enumerate(wv_keys):
				for ip, plab in enumerate(self.results_dict['maps_dict'][wlab]['results_df']['flux_df']):
					flux_df = self.results_dict['maps_dict'][wlab]['results_df']['flux_df'][plab]
					error_df = self.results_dict['maps_dict'][wlab]['results_df']['error_df'][plab]
					for mlab in flux_df:
						if wlen > 1:
							axs[ip, iwv].scatter(flux_df[mlab].index, flux_df[mlab].values * 1e3)
							axs[ip, iwv].errorbar(flux_df[mlab].index, flux_df[mlab].values * 1e3, error_df[mlab].values * 1e3,
												  label=mlab)
							if not ip:
								axs[ip, iwv].set_title(wlab)
							else:
								axs[ip, iwv].set_xlabel('redshift')
							if not iwv:
								axs[ip, iwv].set_ylabel('flux density (Jy)')
							axs[ip, iwv].set_yscale('log')
							#axs[ip, iwv].set_xlim([0., 8])
							axs[ip, iwv].set_ylim([1e-3, 5e1])
							if (ip == 1) & (iwv == 0):
								axs[ip, iwv].legend(loc='upper right')
						else:
							axs[ip].scatter(flux_df[mlab].index, flux_df[mlab].values * 1e3)
							axs[ip].errorbar(flux_df[mlab].index, flux_df[mlab].values * 1e3, error_df[mlab].values * 1e3,
												  label=mlab)
							if not ip:
								axs[ip].set_title(wlab)
							else:
								axs[ip].set_xlabel('redshift')
							if not iwv:
								axs[ip].set_ylabel('flux density (Jy)')
							axs[ip].set_yscale('log')
							#axs[ip].set_xlim([0., 8])
							axs[ip].set_ylim([1e-3, 5e1])
							if (ip == 1):
								axs[ip].legend(loc='upper right')
		else:
			plen = 1
			fig, axs = plt.subplots(plen, wlen, figsize=(22, 10))
			for iwv, wlab in enumerate(wv_keys):
				flux_df = self.results_dict['maps_dict'][wlab]['results_df']['flux_df']
				error_df = self.results_dict['maps_dict'][wlab]['results_df']['error_df']
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
						if not ip:
							axs[ip].set_ylabel('LIR (M_sun)')
						axs[ip].set_xlabel('redshift')
						axs[ip].set_ylabel('flux density (Jy)')
						axs[ip].set_ylim([9, 13.5])
						axs[ip].set_title(plab)
				else:
					axs.scatter(z_data_array, sed_lir_vs_z_array[:, im])
					axs.plot(z_data_array, sed_lir_vs_z_array[:, im], label=mlab)
					axs.set_ylabel('LIR (M_sun)')
					axs.set_xlabel('redshift')
					axs.set_ylabel('flux density (Jy)')
					axs.set_ylim([9, 13.5])
		else:
			print("Skipping SED plotting because only single wavelength measured.")
