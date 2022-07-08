"""
Created on Thurs 7 July 2022

@author : minh.ngo
"""

import json
import numpy as np
from .error import *
from .extract_lsf import extract_lsf
from scipy.optimize import *
import matplotlib.pyplot as plt
import time
from lmfit.models import LinearModel
from .lsf_model import LSF_MODEL

def gauss(x, A, mu, sigma):
    """
    Gaussian function

    Parameters
    -------------
    x : Any

    A : float
        Amplitude
    mu : float
        mean or centred point of LSF
    sigma : float
        sqrt of variance

    Returns
    ------------
    returned value : Any
                array_like or a number
    """
    return A * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2)

def fitted_gauss(wavelength, intensity):
    # Restore data of wavelength and intensity, after that we sort array by order of wavelength
    dtype=[('wavelength', np.float32), ('intensity', np.float32)]
    values = [(wavelength[i], intensity[i]) for i in range(len(intensity))]
    
    # Note : Use array_np.hstack()
    list_of_tuples = np.array(values, dtype=dtype)
    list_of_tuples = np.sort(list_of_tuples, order='wavelength')

    # Recast wavelength and intensity into numpy arrays so we can use their handy features
    wavelength_appro = np.asarray(list_of_tuples[:]['wavelength'])
    intensity_appro = np.asarray(list_of_tuples[:]['intensity'])
    intensity_ref = np.array(intensity_appro)

    # Executing curve_fit on data
    parameters, covariance = curve_fit(gauss, wavelength_appro, intensity_appro)
    intensity_appro = gauss(wavelength_appro, parameters[0], parameters[1], parameters[2])
    sq_error = rms_error(intensity_ref, intensity_appro)
    rel_error = max_relative_error(intensity_ref, intensity_appro)
    return (wavelength_appro, intensity_appro, parameters, sq_error, rel_error)

def intensity(wave, params_linear):
    """
    Parameters
    ------------
    wave : float ou array-like
    params_linear : dict
    """
    A = params_linear["Amplitude"][0] * wave + params_linear["Amplitude"][1]
    mu = params_linear["Mean"][0] * wave + params_linear["Mean"][1]
    sigma = params_linear["Sigma"][0] * wave + params_linear["Sigma"][1]
    y = A * 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-0.5*((wave-mu)/sigma)**2)
    return y

class GAUSSIAN_MODEL(LSF_MODEL):
    def __init__(self, params_linear=None, pose='sampled', slice=0, config='H', detID=1, nb_line=100, normal=True, flatfield=False) -> None:
        super().__init__(params_linear, pose, slice, config, detID, nb_line, normal, flatfield)

    @classmethod
    def from_json(obj, filename):
        with open(filename, 'r') as f:
            data = json.load(f)
        if data['name'] != 'GAUSSIAN_MODEL':
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
        super().plot_estimated_curve(fitted_gauss)

    def calculate_parameters(self):
        dic = {'Amplitude': [], 'Mean': [], 'Sigma': []}
        listLines = []
        start = time.process_time()
        for nb_line in range(self._lineUp, self._lineDown+1):
            tup = extract_lsf(self.pose, self.config, self.slice, nb_line, self.detID, flatfield=False)
            listLines.append(tup[1])
            tup_gauss = fitted_gauss(tup[0]-tup[1], tup[2])
            dic['Amplitude'].append(tup_gauss[2][0])
            dic['Mean'].append(tup_gauss[2][1])
            dic['Sigma'].append(tup_gauss[2][2])
        end = time.process_time()
        print("Time elapsed during calculation:", end-start, "seconds")

        # Plotting
        mod = LinearModel()
        fig, axes = plt.subplots(3,1,figsize=(6,16))
        plt.xlabel("wavelength")
        i = 0
        params_linear = []
        for key, value in dic.items():
            pars = mod.guess(value, x=listLines)
            out = mod.fit(value, pars, x=listLines)
            params_linear.append(out.params)
            Y_pred = out.best_fit
            axes[i].set_ylabel(key)
            axes[i].plot(listLines, value, color='blue', marker="o", linestyle="")
            ## Plot regression line
            axes[i].plot(listLines, Y_pred, color='red', markersize=10, linestyle='solid', label="Regression line")
            axes[i].set_title(f"Evolution of {key}")
            i += 1
        plt.legend()
        fig.suptitle(f"Evolution of 3 parameters Gaussian configuration {self.config} slice {self.slice} detID {self.detID}", fontsize=12, fontweight="bold")
        plt.savefig(f"Eval_Gauss_slice_{self.slice}_config_{self.config}_det_{self.detID}")
        plt.show()
        dic = {'Amplitude': None, 'Mean': None, 'Sigma': None}
        for i, key in enumerate(dic.keys()):
            dic[key] = [params_linear[i]["slope"].value, params_linear[i]["intercept"].value]
        self._params_linear = dic

    def plot_rms_error(self):
        super().plot_rms_error(fitted_gauss)

    def plot_intensity(self):
        super().plot_intensity(intensity) 

    def plot_intensity_rms_error(self):
        super().plot_intensity_rms_error(intensity)

    def write_file_json(self, filename):
        return super().write_file_json(filename)