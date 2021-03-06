import pdb
import os
import numpy as np
from astropy.io import fits

class Skymaps:
	'''
	This Class creates objects for a set of
	maps/noisemaps/beams/etc., at each Wavelength.

	Each map is defined through parameters in the config.ini file:
	- wavelength (float)
	- beam (dict); contains "fwhm" in arcsec and "area" in steradian**2.
	- color_correction (float)
	- path_map (str)
	- path_noise (str)
	'''

	maps_dict = {}

	def __init__(self):
		pass

	def import_maps(self):
		''' Import maps (and optionally noisemaps) described in config file.

		:return: Map dictionary stored in self.maps_dict.
		'''

		for imap in self.config_dict['maps']:
			map_params = self.config_dict['maps'][imap].copy()
			map_dict = self.import_map_dict(map_params)
			self.maps_dict[imap] = map_dict

	def import_map_dict(self, map_dict):
		''' Import maps described in config file and populate map_dict with parameters.

		:param map_dict:
		:return: populated map_dict
		'''

		file_map = self.parse_path(map_dict["path_map"])
		if 'path_noise' in map_dict:
			file_noise = self.parse_path(map_dict["path_noise"])
		else:
			file_noise = None
		wavelength = map_dict["wavelength"]
		psf = map_dict["beam"]
		beam_area = map_dict["beam"]["area"]
		color_correction = map_dict["color_correction"]

		#SPIRE Maps have Noise maps in the second extension.
		if file_noise is not None:
			if file_map == file_noise:
				header_ext_map = 1
				header_ext_noise = 2
			else:
				header_ext_map = 0
				header_ext_noise = 0
		else:
			header_ext_map = 0
			header_ext_noise = None

		if not os.path.isfile(file_map):
			file_map = os.path.join('..', file_map)
			file_noise = os.path.join('..', file_noise)

		if os.path.isfile(file_map):
			try:
				cmap, hd = fits.getdata(file_map, header_ext_map, header=True)
			except:
				cmap, hd = fits.getdata(file_map, 1, header=True)
			if file_noise is not None:
				try:
					cnoise, nhd = fits.getdata(file_noise, header_ext_noise, header=True)
				except:
					cnoise, nhd = fits.getdata(file_noise, 1, header=True)
		else:
			print("Files not found, check path in config file: "+file_map)
			pdb.set_trace()

		#GET MAP PIXEL SIZE
		if 'CD2_2' in hd:
			pix = hd['CD2_2'] * 3600.
		else:
			pix = hd['CDELT2'] * 3600.

		#READ BEAMS
		fwhm = psf["fwhm"]
		kern = self.gauss_kern(fwhm, np.floor(fwhm * 8.)/pix, pix)

		map_dict["map"] = self.clean_nans(cmap) * color_correction
		if beam_area != 1.0:
			map_dict["map"] *= beam_area * 1e6
		if file_noise is not None:
			map_dict["noise"] = self.clean_nans(cnoise, replacement_char=1e10) * color_correction
			if beam_area != 1.0:
				map_dict["noise"] *= beam_area * 1e6

		map_dict["header"] = hd
		map_dict["pixel_size"] = pix
		map_dict["psf"] = self.clean_nans(kern)

		if wavelength != None:
			map_dict["wavelength"] = wavelength

		if fwhm != None:
			map_dict["fwhm"] = fwhm

		return map_dict
