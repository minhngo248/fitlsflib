"""
Created on Thurs 7 July 2022

@author : minh.ngo
"""

## Imported libraries
import hpylib as hp
import numpy as np
from astropy.io import fits
import math
from . import normalize

def extract_lsf(pose, config, nb_slitlet, nb_line, detID, normal=True, flatfield=True, dirPath='../exposures'):
    """
    Extract all necessary informations for modelising the LSF

    Parameters
    ----------
    pose : string
            'sampled', 'oversampled'
    config : string
            'H', 'HK', 'Hhigh'
    nb_slitlet : int
             0 to 37
    nb_line : int
            0 to 254
    detID : int
            1 to 8
    normalize : boolean
            True or False
    flatfield : boolean
            True or False

    Returns
    --------------
    tuple : tuple
            tuple of information for measuring
            (map_wave, wavelength_line, image_cut, x_cor, y_cor, nb_slitlet, nb_line)
    """
    if pose == 'sampled':
        hdul = fits.open(f"{dirPath}/ARC-linspace256_CLEAR_20MAS_"+config+"_PRM.fits")
        hdul_flat = fits.open(f"{dirPath}/FLAT-CONT2_CLEAR_20MAS_" + config +"_PRM.fits")
    else:
        hdul = fits.open(f"{dirPath}/ARC-linspace256_CLEAR_20MAS_"+config+"_PRM_oversampled.fits")
        hdul_flat = fits.open(f"{dirPath}/FLAT-CONT2_CLEAR_20MAS_" + config +"_PRM_oversampled.fits")        
    
    hdul_lines = fits.open(f"{dirPath}/line_catalog_linspace256.fits")
    wavelength_line = hdul_lines[config].data[nb_line][0]
    hdul_lines.close()

    ## Step of calibration 
    # Image (intensity)
    if flatfield:
        image = hdul['CHIP'+str(detID)+'.DATA'].data/hdul_flat['CHIP'+str(detID)+'.DATA'].data
    else:
        image = hdul['CHIP'+str(detID)+'.DATA'].data
    hdul.close()
    hdul_flat.close()
    # Open table of wavelengths
    table_wave = hp.WAVECAL_TABLE.from_FITS(f"{dirPath}/WAVECAL_TABLE_20MAS_"+config+".fits", detID)

    # Open slitlet table
    obj = hp.SLITLET_TABLE.from_FITS(f"{dirPath}/SLITLET_TABLE_20MAS_"+config+".fits", detID)
    
    ## Get 3 columns left, center, right
    if pose == 'sampled':
        x_c = obj.get_xcenter(nb_slitlet, y=np.arange(4096))
        x_r = obj.get_xright(nb_slitlet, y=np.arange(4096))
        x_l = obj.get_xleft(nb_slitlet, y=np.arange(4096))
        array_wave_c = table_wave.get_lbda(nb_slitlet, x=x_c, y=np.arange(4096))
        array_wave_r = table_wave.get_lbda(nb_slitlet, x=x_r, y=np.arange(4096))
        array_wave_l = table_wave.get_lbda(nb_slitlet, x=x_l, y=np.arange(4096))
    else:
        x_c = obj.get_xcenter(nb_slitlet, y=(np.arange(12288)-1)/3)*3+1
        x_r = obj.get_xright(nb_slitlet, y=(np.arange(12288)-1)/3)*3+1
        x_l = obj.get_xleft(nb_slitlet, y=(np.arange(12288)-1)/3)*3+1
        array_wave_c = table_wave.get_lbda(nb_slitlet, x=(x_c-1)/3, y=(np.arange(12288)-1)/3)
        array_wave_r = table_wave.get_lbda(nb_slitlet, x=(x_r-1)/3, y=(np.arange(12288)-1)/3)
        array_wave_l = table_wave.get_lbda(nb_slitlet, x=(x_l-1)/3, y=(np.arange(12288)-1)/3)        

    ## Return 3 points
    ind = np.argmin(abs(array_wave_c - wavelength_line))
    point_c = (x_c[ind], ind)
    ind = np.argmin(abs(array_wave_r - wavelength_line))
    point_r = (x_r[ind], ind)
    ind = np.argmin(abs(array_wave_l - wavelength_line))
    point_l = (x_l[ind], ind)
    
    ## Meshgrid, choose rectangle
    down_y = np.min([point_l[1], point_c[1], point_r[1]])
    upper_y = np.max([point_l[1], point_c[1], point_r[1]])
    if pose == 'sampled':
        y_array = np.arange(down_y-4, upper_y+4+1)
        x_array = np.arange(math.ceil(point_l[0])+4, math.floor(point_r[0])-4)
    else:
        y_array = np.arange(down_y-12, upper_y+12+1)
        x_array = np.arange(math.ceil(point_l[0])+12, math.floor(point_r[0])-12)        
    x_cor, y_cor = np.meshgrid(x_array, y_array)
    median = abs(np.nanmean(np.diff(array_wave_c)))
    ## Cutted image
    if pose == 'sampled':
        map_wave = table_wave.get_lbda(nb_slitlet, x_cor, y_cor)
        mask = (abs(map_wave - wavelength_line) <= 4*median)
    else:
        map_wave = table_wave.get_lbda(nb_slitlet, (x_cor-1)/3, (y_cor-1)/3)
        mask = (abs(map_wave - wavelength_line) <= 12*median)
    image_cut = image[y_cor, x_cor]
    x_cor = x_cor[mask]
    y_cor = y_cor[mask]
    map_wave = map_wave[mask]
    image_cut = image_cut[mask]
    if normal:
        image_cut = normalize.normalize(image_cut)
    if pose == 'oversampled':
        x_cor = (x_cor-1)/3
        y_cor = (y_cor-1)/3
    ## Return
    return map_wave, wavelength_line, image_cut, x_cor, y_cor, nb_slitlet, nb_line