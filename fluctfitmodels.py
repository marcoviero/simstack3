import numpy as np
from lmfit import Parameters, minimize, fit_report
from simstacktoolbox import SimstackToolbox
from scipy.ndimage.filters import gaussian_filter

c = 299792458.0 # m/s

class FluctFitModels(SimstackToolbox):
    def __init__(self):
        super().__init__()

    def direct_convolved_fit_A_Tdust_one_pop(self, params, X, y):

        v = params.valuesdict()  # .copy()
        A_model = v.pop('A_offset_sf')
        T_model = v.pop('T_offset_sf')
        Z_model = X[0]
        i = 0
        for ival in v:
            if 'A_' in ival:
                A_model += X[i] * v[ival]
            else:
                T_model += X[i] * v[ival]
                i += 1

        out_model = []
        for map_name in y:
            map_lambda = y[map_name]['wavelength']
            map_nu = c * 1.e6 / map_lambda
            map_coords = y[map_name]['map_coords']
            map_sky = y[map_name]['map'] - np.mean(y[map_name]['map'])

            map_model = np.zeros_like(map_sky)
            map_pixels = np.zeros_like(map_sky)

            S_model = self.get_flux_mJy(np.array([map_nu]), A_model, T_model / (1 + Z_model))
            map_model[map_coords[0], map_coords[1]] += S_model
            map_pixels[map_coords[0], map_coords[1]] += 1

            fwhm = y[map_name]['fwhm']
            pix = y[map_name]['pixel_size']
            kern = self.gauss_kern(fwhm, np.floor(fwhm * 10) / pix, pix)
            tmap = self.smooth_psf(map_model, kern)
            tmap -= np.mean(tmap)

            idx_fit = map_pixels != 0
            diff = map_sky - tmap
            out_model.extend(np.ravel(diff[idx_fit]))

        return out_model

    def direct_convolved_fit_A_Tdust_two_pop(self, params, X, y):
        v = params.valuesdict()  # .copy()
        A_model_sf = v.pop('A_offset_sf')
        T_model_sf = v.pop('T_offset_sf')
        A_model_qt = v.pop('A_offset_qt')
        T_model_qt = v.pop('T_offset_qt')
        idx_sf = X[-1] == 1
        idx_qt = X[-1] == 0
        Z_model_sf = X[0][idx_sf]
        Z_model_qt = X[0][idx_qt]

        i = 0
        for ival in v:
            if 'A_' in ival:
                if 'sf' in ival:
                    A_model_sf += X[i][idx_sf] * v[ival]
                else:
                    A_model_qt += X[i][idx_qt] * v[ival]
            else:
                if 'sf' in ival:
                    T_model_sf += X[i][idx_sf] * v[ival]
                else:
                    T_model_qt += X[i][idx_qt] * v[ival]
                    i += 1

        out_model = []
        for map_name in y:
            map_lambda = y[map_name]['wavelength']
            map_nu = c * 1.e6 / map_lambda
            map_coords_sf = y[map_name]['map_coords']['sf']
            map_coords_qt = y[map_name]['map_coords']['qt']
            map_sky = y[map_name]['map'] - np.mean(y[map_name]['map'])

            map_model = np.zeros_like(map_sky)
            map_pixels = np.zeros_like(map_sky)

            S_model_sf = self.get_flux_mJy(np.array([map_nu]), A_model_sf, T_model_sf / (1 + Z_model_sf))
            S_model_qt = self.get_flux_mJy(np.array([map_nu]), A_model_qt, T_model_qt / (1 + Z_model_qt))

            map_model[map_coords_sf[0], map_coords_sf[1]] += S_model_sf
            map_model[map_coords_qt[0], map_coords_qt[1]] += S_model_qt
            map_pixels[map_coords_sf[0], map_coords_sf[1]] += 1
            map_pixels[map_coords_qt[0], map_coords_qt[1]] += 1

            fwhm = y[map_name]['fwhm']
            pix = y[map_name]['pixel_size']
            kern = self.gauss_kern(fwhm, np.floor(fwhm * 10) / pix, pix)
            tmap = self.smooth_psf(map_model, kern)
            tmap -= np.mean(tmap)

            idx_fit = map_pixels != 0
            diff = map_sky - tmap
            out_model.extend(np.ravel(diff[idx_fit]))

        return out_model

    def direct_convolved_fit_A_Tdust_three_pop(self, params, X, y):

        v = params.valuesdict()  # .copy()
        A_model_sf = v.pop('A_offset_sf')
        T_model_sf = v.pop('T_offset_sf')
        A_model_qt = v.pop('A_offset_qt')
        T_model_qt = v.pop('T_offset_qt')
        A_model_agn = v.pop('A_offset_agn')
        T_model_agn = v.pop('T_offset_agn')

        idx_sf = X[-1] == 1
        idx_qt = X[-1] == 0
        idx_agn = X[-1] == 2
        Z_model_sf = X[0][idx_sf]
        Z_model_qt = X[0][idx_qt]
        Z_model_agn = X[0][idx_agn]

        i = 0
        for ival in v:
            if 'A_' in ival:
                if 'sf' in ival:
                    A_model_sf += X[i][idx_sf] * v[ival]
                elif 'qt' in ival:
                    A_model_qt += X[i][idx_qt] * v[ival]
                else:
                    A_model_agn += X[i][idx_agn] * v[ival]
            else:
                if 'sf' in ival:
                    T_model_sf += X[i][idx_sf] * v[ival]
                elif 'qt' in ival:
                    T_model_qt += X[i][idx_qt] * v[ival]
                else:
                    T_model_agn += X[i][idx_agn] * v[ival]
                    i += 1

        out_model = []
        for map_name in y:
            map_lambda = y[map_name]['wavelength']
            map_nu = c * 1.e6 / map_lambda
            map_coords_sf = y[map_name]['map_coords']['sf']
            map_coords_qt = y[map_name]['map_coords']['qt']
            map_coords_agn = y[map_name]['map_coords']['agn']
            map_sky = y[map_name]['map'] - np.mean(y[map_name]['map'])

            map_model = np.zeros_like(map_sky)
            map_pixels = np.zeros_like(map_sky)

            S_model_sf = self.get_flux_mJy(np.array([map_nu]), A_model_sf, T_model_sf / (1 + Z_model_sf))
            S_model_qt = self.get_flux_mJy(np.array([map_nu]), A_model_qt, T_model_qt / (1 + Z_model_qt))
            S_model_agn = self.get_flux_mJy(np.array([map_nu]), A_model_agn, T_model_agn / (1 + Z_model_agn))

            map_model[map_coords_sf[0], map_coords_sf[1]] += S_model_sf
            map_model[map_coords_qt[0], map_coords_qt[1]] += S_model_qt
            map_model[map_coords_agn[0], map_coords_agn[1]] += S_model_agn
            map_pixels[map_coords_sf[0], map_coords_sf[1]] += 1
            map_pixels[map_coords_qt[0], map_coords_qt[1]] += 1
            map_pixels[map_coords_agn[0], map_coords_agn[1]] += 1

            fwhm = y[map_name]['fwhm']
            pix = y[map_name]['pixel_size']
            kern = self.gauss_kern(fwhm, np.floor(fwhm * 10) / pix, pix)
            tmap = self.smooth_psf(map_model, kern)
            tmap -= np.mean(tmap)

            idx_fit = map_pixels != 0
            diff = map_sky - tmap
            out_model.extend(np.ravel(diff[idx_fit]))

        return out_model