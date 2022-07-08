"""
Created on Thurs 7 July 2022

@author : minh.ngo
"""

from .error import *
from . import choose_line
from .extract_lsf import extract_lsf
import matplotlib.pyplot as plt
import json
from numpyencoder import NumpyEncoder

class LSF_MODEL(object):
    def __init__(self, params_linear, pose, slice, config, detID, nb_line, normal, flatfield) -> None:
        self._params_linear = params_linear
        self.pose = pose
        self.slice = slice
        self.config = config
        self.detID = detID
        self.nb_line = nb_line
        self.normal = normal
        self.flatfield = flatfield
        self._lineUp = choose_line.choose_line_up(slice, pose, config, detID)
        self._lineDown = choose_line.choose_line_down(slice, pose, config, detID)
        self._tup = extract_lsf(pose, config, slice, nb_line, detID, normal, flatfield)

    def plot_estimated_curve(self, fitted_function):
        tup_model = fitted_function(self._tup[0]-self._tup[1], self._tup[2])
        fig = plt.figure()
        plt.plot(self._tup[0], self._tup[2], linestyle='', marker='+', label='Real data')
        plt.plot(tup_model[0]+self._tup[1], tup_model[1], label="Fitted curve")
        plt.legend()
        plt.title(f"Line {self.nb_line} configuration {self.config} slice {self.slice}")
        plt.show()
        self._rms_error = tup_model[3]
        self._mr_error = tup_model[4]

    def plot_rms_error(self, fitted_function):
        list_error = [] 
        lines = []
        for nb_line in range(self._lineUp, self._lineDown+1):  
            tup = extract_lsf(self.pose, self.config, self.slice, nb_line, self.detID, self.normal, self.flatfield)
            lines.append(tup[1])
            tup_fit = fitted_function(tup[0]-tup[1], tup[2])
            list_error.append(tup_fit[-2])

        fig = plt.figure()
        ax = plt.axes() 
        ax.set_xlabel("wavelength of lines")
        ax.set_ylabel("rms error")
        ax.plot(lines, list_error)
        plt.title(f"error LSF in slice {self.slice} detID {self.detID} configuration {self.config} from line {self._lineUp} to line {self._lineDown}")
        plt.show()

    def plot_intensity(self, intensity_function):
        """
        Plotting intensity after linearisation of parameters
        """
        params_linear = self._params_linear
        image_pred = intensity_function(self._tup[0]-self._tup[1], params_linear)
        fig = plt.figure()
        ax = plt.axes() 
        ax.set_xlabel("wavelength of lines")
        ax.set_ylabel("intensity")
        ax.plot(self._tup[0], self._tup[2], marker='+', linestyle='', label='Real data')
        ax.plot(self._tup[0], image_pred, marker='+', linestyle='', label='Estimated curve')
        plt.legend()
        plt.title(f"Evaluated data in slice {self.slice} configuration {self.config} line {self.nb_line}")
        plt.show()
        print("RMS error : ", rms_error(self._tup[2], image_pred))

    def plot_intensity_rms_error(self, intensity_function):
        """
        Plotting rms error after linearisation of parameters
        """
        params_linear = self._params_linear
        rms = []
        lines = []
        for nb_l in range(self._lineUp, self._lineDown+1):
            tup_data = extract_lsf(self.pose, self.config, self.slice, nb_l, self.detID, self.normal, self.flatfield)    
            image_pred = intensity_function(tup_data[0]-tup_data[1], params_linear)
            rms.append(rms_error(tup_data[2], image_pred))
            lines.append(tup_data[1])
        fig = plt.figure()
        plt.xlabel('wavelength of lines')
        plt.ylabel('RMS error')
        plt.plot(lines, rms)
        plt.title(f"RMS error in config {self.config} slice {self.slice} from line {self._lineUp} to line {self._lineDown}")
        plt.show()

    def write_file_json(self, filename):
        # Serializing json 
        classname = self.__class__.__name__
        dic = {'name': classname, 'params_linear': self._params_linear, 'pose': self.pose, 'slice': self.slice, 
                'config': self.config, 'detID': self.detID, 'nb_line': self.nb_line, 'normal': self.normal, 'flatfield': self.flatfield}
        json_object = json.dumps(dic, indent=4, cls=NumpyEncoder)
        
        with open(filename, "w") as outfile:
            outfile.write(json_object)