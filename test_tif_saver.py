# -*- coding: utf-8 -*-
"""
Created on 201102

@author: stepp

Application to read a h5 file and save a tif from it at 2Hz to test the
watchdog on lebpc20
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import h5py
from skimage import io
import os

data_path = 'C:/Users/stepp/Documents/data_raw/SmartMito/'  # nb: begin with /
print('data_path : ', data_path)
print()
input_data_filename1 = data_path + 'Mito.h5'  # Mito
input_data_filename2 = data_path + 'Drp1.h5'  # Drp1

# input_data_filename1 = data_path + 'cell_8_Pos1_mito_741.tif'  # Mito
# input_data_filename2 = data_path + 'cell_8_Pos1_drp1_741.tif'  # Drp1

# input_data_filename1 = data_path + 'cell_8_mito_1024.tif'  # Mito
# input_data_filename2 = data_path + 'cell_8_drp1_1024.tif'  # Drp1

input_data_filename1 = data_path + 'cell_8_mito.tif'  # Mito
input_data_filename2 = data_path + 'cell_8_drp1.tif'  # Drp1

input_data_filename1 = data_path + 's3_c6_mito.tif'  # Mito
input_data_filename2 = data_path + 's3_c6_drp1.tif'  # Drp1

input_data_filename1 = data_path + 's6_c12_p0_mito.tif'  # Mito
input_data_filename2 = data_path + 's6_c12_p0_drp1.tif'  # Drp1

# input_data = data_path + '180420_111.tif'

mito = []
drp = []

if 'input_data' in locals():
    mito = io.imread(input_data)
    drp = mito[1::2]
    mito = mito[0::2]
else:
    if input_data_filename1[-3:] == '.h5':
        hf = h5py.File(input_data_filename1, 'r')
        mito = hf.get('Mito')  # Mito
    else:
        mito = io.imread(input_data_filename1)

    if input_data_filename2[-3:] == '.h5':
        hf = h5py.File(input_data_filename2, 'r')
        drp = hf.get('Drp1')
    else:
        drp = io.imread(input_data_filename2)

print(drp.dtype)
print('Input : ', drp.shape)
print('Input : ', np.shape(drp)[0])

nas_path = '//lebnas1.epfl.ch/microsc125/Watchdog/python_saver/'

i = 0
for item in range(0, 100):
    # for item in range(1, 2002, 1000):  # [1, 208]:  # range(150, 210):
    t1 = time.perf_counter()
    print(item)
    mito_path = nas_path + 'img_channel000_position000_time' + \
        str((i*2)).zfill(9) + '_z000.tif'
    drp_path = nas_path + 'img_channel000_position000_time' + \
        str((i*2 + 1)).zfill(9) + '_z000.tif'

    # First write mito then drp as watchdog waits for drp image
    io.imsave(mito_path, mito[item, :, :].astype(np.uint16),
              check_contrast=False)
    time.sleep(0.05)
    io.imsave(drp_path, drp[item, :, :].astype(np.uint16),
              check_contrast=False)
    t2 = time.perf_counter()
    time.sleep(np.max([0, .5 - (t2 - t1)]))
    i = i + 1

time.sleep(3)

# Clear the folder completely to have the same situation always
print('deleting files')
for f in os.listdir(nas_path):
    # if not f.endswith(".tiff"):
    #     continue
    os.remove(os.path.join(nas_path, f))
    time.sleep(0.001)
print('Done')
