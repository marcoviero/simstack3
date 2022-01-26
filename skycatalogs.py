import pdb
import os
import json
import numpy as np
import pandas as pd

class Skycatalogs:
	'''
	This Class creates a catalog object containing the raw catalog and a split_table
	for which each category is split into indices, for example, if you have three redshift
	bins [1,2,3], the "redshift" column of split_dict will contain 0,1,2, and nans, where
	each object between z=1 and 2 has value 0; between z=1 and 2 has value 1... and objects
	outside that range have values nan.  Same is true for bins of "stellar_mass", "magnitude",
	or however the catalog is split.

	Required parameters are defined in the config.ini file
	- path (str); format CATSPATH PATH
	- file (str); format FILENAME
	- astrometry (dict); contains "ra", "dec", which indicate column names for positions in RA/DEC.
	- classification (dict); contains dicts for "split_type", "redshift", and optional categories, e.g.,
		"split_params", "stellar_mass", "magnitudes", etc. Specifically:
		-- "split_type" can be "label", "uvj", "nuvrj"
			> "label" is ..
			> "uvj" is ...
			> "nuvrj" is ..
		-- "split_params" can be
			> "labels"
		--
	'''
	def __init__(self, config_dict):

		self.config_dict = config_dict

	def import_catalog(self):

		self.catalog_dict = {}

		catalog_params = self.config_dict['catalog']
		path_catalog = os.path.join(self.parse_path(catalog_params['path']), catalog_params['file'])
		if os.path.isfile(path_catalog):
			self.catalog_dict['tables'] = {'full_table': pd.read_table(path_catalog, sep=',')}
		else:
			print("Catalog not found: "+path_catalog)
			pdb.set_trace()

		self.split_table_into_populations()

		# Remove full table from simstack_object (they're huge!)
		self.catalog_dict['tables'].pop('full_table')

	def split_table_into_populations(self):

		# Make new table starting with RA and DEC
		astrometry_keys = self.config_dict['catalog']['astrometry']
		self.catalog_dict['tables']['split_table'] = {}
		self.catalog_dict['tables']['split_table'] = pd.DataFrame(self.catalog_dict['tables']['full_table'][astrometry_keys.values()])
		self.catalog_dict['tables']['split_table'].rename(columns={astrometry_keys["ra"]: "ra", astrometry_keys["dec"]: "dec"}, inplace=True)

		# Split catalog by classification type
		split_dict = self.config_dict['catalog']['classification']
		split_type = split_dict.pop('split_type')

		# By labels means unique values inside columns (e.g., "CLASS" = [0,1,2])
		if 'labels' in split_type:
			self.separate_by_label(split_dict, self.catalog_dict['tables']['full_table'])

		# By uvj means it splits into star-forming and quiescent galaxies via the u-v/v-j method.
		if 'uvj' in split_type:
			self.separate_sf_qt_uvj(split_dict, self.catalog_dict['tables']['full_table'])

		# By nuvrj means it splits into star-forming and quiescent galaxies via the NUV-r/r-j method.
		if 'nuvrj' in split_type:
			self.separate_sf_qt_nuvrj(split_dict, self.catalog_dict['tables']['full_table'])

	def separate_by_label(self, split_dict, table, add_background=False):
		parameter_names = {}
		label_keys = list(split_dict.keys())
		for key in label_keys:
			if type(split_dict[key]['bins']) is str:
				bins = json.loads(split_dict[key]['bins'])
				parameter_names[key] = ["_".join([key, str(bins[i]), str(bins[i + 1])]) for i in range(len(bins[:-1]))]
			elif type(split_dict[key]['bins']) is dict:
				bins = len(split_dict[key]['bins'])
				parameter_names[key] = ["_".join([key, str(i)]) for i in range(bins)]
			else:
				bins = split_dict[key]['bins']
				parameter_names[key] = ["_".join([key, str(i)]) for i in range(bins)]
			# Categorize using pandas.cut.  So good.
			col = pd.cut(table[split_dict[key]['id']], bins=bins, labels=False)
			col.name = key  # Rename column to label
			# Add column to table
			self.catalog_dict['tables']['split_table'] = self.catalog_dict['tables']['split_table'].join(col)

		# Name Cube Layers (i.e., parameters)
		self.catalog_dict['tables']['parameter_labels'] = []
		for ipar in parameter_names[label_keys[0]]:
			for jpar in parameter_names[label_keys[1]]:
				if len(label_keys) > 2:
					for kpar in parameter_names[label_keys[2]]:
						if len(label_keys) > 3:
							for lpar in parameter_names[label_keys[3]]:
								pn = "__".join([ipar, jpar, kpar, lpar])
								self.catalog_dict['tables']['parameter_labels'].append(pn)
						else:
							pn = "__".join([ipar, jpar, kpar])
							self.catalog_dict['tables']['parameter_labels'].append(pn)
				else:
					pn = "__".join([ipar, jpar])
					self.catalog_dict['tables']['parameter_labels'].append(pn)
		if add_background:
			self.catalog_dict['tables']['parameter_labels'].append('background_layer')

		self.config_dict['parameter_names'] = parameter_names
		#pdb.set_trace()

	def separate_sf_qt_uvj(self, split_dict, table, zcut=10):

		uvkey = split_dict['split_params']["bins"]['U-V']
		vjkey = split_dict['split_params']["bins"]['V-J']
		zkey = split_dict['redshift']["id"]

		# Find quiescent galaxies using UVJ criteria
		ind_zlt1 = (table[uvkey] > 1.3) & (table[vjkey] < 1.5) & (table[zkey] < 1) & \
				   (table[uvkey] > (table[vjkey] * 0.88 + 0.69))
		ind_zgt1 = (table[uvkey] > 1.3) & (table[vjkey] < 1.5) & (table[zkey] >= 1) & \
				   (table[zkey] < zcut) & (table[uvkey] > (table[vjkey] * 0.88 + 0.59))

		# Add sfg column
		sfg = np.ones(len(table))
		sfg[ind_zlt1] = 0
		sfg[ind_zgt1] = 0
		class_label = split_dict['split_params']["id"]  # typically 'sfg', but can be anything.
		table[class_label] = sfg

		self.separate_by_label(split_dict, table)

	def separate_sf_qt_nuvrj(self, split_dict, table, zcut=10):

		uvrkey = split_dict['split_params']["bins"]['UV-R']
		rjkey = split_dict['split_params']["bins"]['R-J']
		zkey = split_dict['redshift']["id"]

		# Find quiescent galaxies using NUV-r/r-J criteria
		ind_nuvrj = (table[uvrkey] > (3*table[rjkey] + 1)) & (table[uvrkey] > 3.1) & (table[zkey] < zcut)

		# Add sfg column
		sfg = np.ones(len(table))
		sfg[ind_nuvrj] = 0
		class_label = split_dict['split_params']["id"]  # typically 'sfg', but can be anything.
		table[class_label] = sfg

		self.separate_by_label(split_dict, table)