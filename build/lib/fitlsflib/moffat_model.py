"""
Created on Thurs 7 July 2022

@author : minh.ngo
"""

from .lsf_model import LSF_MODEL
import json
from lmfit.models import MoffatModel, LinearModel
import numpy as np
import matplotlib.pyplot as plt
from .error import *
from .extract_lsf import extract_lsf

def fitted_moffat(wavelength, intensity):
    mod = MoffatModel()
    pars = mod.guess(intensity, x=wavelength)
    out = mod.fit(intensity, pars, x=wavelength, method='least_squares')
    intensity_appro = out.best_fit
    values = [(wavelength[j], intensity_appro[j]) for j in range(len(intensity))]
    dtype = [('wavelength', np.float32), ('intensity', np.float32)]
    listData = np.array(values, dtype)
    listData = np.sort(listData, order='wavelength')
    wave = np.asarray(listData[:]['wavelength'])
    inten = np.asarray(listData[:]['intensity'])
    return (wave, inten, out.params, rms_error(intensity, intensity_appro), max_relative_error(intensity, intensity_appro))

def intensity(wave, params_linear):
    A = params_linear["amplitude"][0] * wave + params_linear["amplitude"][1]
    mu = params_linear["center"][0] * wave + params_linear["center"][1]
    sigma = params_linear["sigma"][0] * wave + params_linear["sigma"][1]
    beta = params_linear["beta"][0] * wave + params_linear["beta"][1]
    y = A * (((wave-mu)/sigma)**2 + 1)**(-beta)
    return y

class MOFFAT_MODEL(LSF_MODEL):
    def __init__(self, params_linear=None, pose='sampled', slice=0, config='H', detID=1, nb_line=100, normal=True, flatfield=False) -> None:
        super().__init__(params_linear, pose, slice, config, detID, nb_line, normal, flatfield)

    @classmethod
    def from_json(obj, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        if data['name'] != 'MOFFAT_MODEL':
            raise NameError('IncompatibleModel')
        else:
            params_linear = data['params_linear']
            pose = data['pose']
            slice = data['slice'] 
            config = data['config'] 
            detID = data['detID'] 
            nb_line = data['nb_line']  
            normal = data['normal']  
            flatfield = data['flatfield'] 
            return obj(params_linear, pose, slice, config, detID, nb_line, normal, flatfield)

    def plot_estimated_curve(self):
        super().plot_estimated_curve(fitted_moffat)

    def calculate_parameters(self):
        listLines = []
        params = {'amplitude': [], 'center': [], 'sigma': [], 'beta': []}
        for nb_line in range(self._lineUp, self._lineDown+1):    
            map_wave, wavelength_line, image_cut, x_cor, y_cor, nb_slit, nb_l = extract_lsf(self.pose, 
                        self.config, self.slice, nb_line, self.detID, self.normal, self.flatfield) 
            listLines.append(wavelength_line)
            tup = fitted_moffat(map_wave-wavelength_line, image_cut)
            for key, val in params.items():
                val.append(tup[2][key].value)
    
        ## Plotting
        mod = LinearModel()
        fig, axes = plt.subplots(len(params), 1, figsize=(6, 16))
        plt.xlabel("wavelength")
        i = 0
        params_linear = {'amplitude': [], 'center': [], 'sigma': [], 'beta': []}
        for key, val in params.items():    
            pars = mod.guess(val, x=listLines)
            out = mod.fit(val, pars, x=listLines)
            params_linear[key].append(out.params["slope"].value)
            params_linear[key].append(out.params["intercept"].value)
            Y_pred = out.best_fit
            axes[i].set_ylabel(key)
            axes[i].scatter(listLines, val, marker="o", color="blue")    
            # Plotting the regression line
            axes[i].plot(listLines, Y_pred, color='red', markersize=10, linestyle='solid', label="Regression line")
            axes[i].set_title(f"Evolution of {key}")
            i += 1
        plt.legend()
        fig.suptitle("Evolution of 4 parameters of Moffat model", fontweight='bold')
        plt.show()
        self._params_linear = params_linear

    def plot_rms_error(self):
        super().plot_rms_error(fitted_moffat)

    def plot_intensity(self):
        super().plot_intensity(intensity) 

    def plot_intensity_rms_error(self):
        super().plot_intensity_rms_error(intensity)

    def write_file_json(self, filename):
        return super().write_file_json(filename)