""" Take fixed framerate data and perform a simulation of ATS (adaptive temporal sampling) on
this data to see what performance would have been if ATS was used.

All the parameters are directly after the imports. As a default also the settings from the central
ATS_settings.json file can be used.

Created on Tue Jan  5 11:03:50 2021

@author: Willi Stepp
"""

import json
import os
import re
import shutil

import numpy as np
from skimage import io
from tensorflow import keras

from NNfeeder import prepareNNImages
from NNio import loadTifStack

#
#                          USER SETTINGS
#


stacks = ('C:/Users/stepp/Documents/02_Raw/Caulobacter_Zenodo/'
          '160613_ML2159_MTS-mCherry_FtsZ-GFP_80/'
          'SIM images/Series5_copy.ome.tiff')
stacks = [stacks]
# Comment if only taking one stack
# stacks = []
# for i in range(1, 16):
#     stacks.append(stack + str(i).zfill(2) + '.ome.tiff')
modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/GUI/model_Dora.h5'
# Should we get the settings from the central settings file?
extSettings = False
dataOrder = 0  # 0 for drp/foci first, 1 for mito/structure first

if not extSettings:
    # if extSettings is set to False, you can set the settings to be used here
    thresholdLow = 70
    thresholdUp = 90
    slowRate = 600  # in seconds
    # The fast frame rate is the rate of the original file for now.
    minFastFrames = 5  # minimal number of fast frames after switching to fast. Should be > 2
    #
    #               USER SETTTINGS END
    #

    # Save these settings for later documentation
    settings = {'lowerThreshold': thresholdLow, 'upperThreshold': thresholdUp,
                'slowRate': slowRate, 'minFastFrames': minFastFrames, 'dataOrder': dataOrder}
else:
    # if extSettings is True, load the settings from the json file
    with open('./ATS_settings.json') as file:
        settings = json.load(file)
        thresholdLow = settings['lowerThreshold']
        thresholdUp = settings['upperThreshold']
        slowRate = settings['slowRate']
        minFastFrames = settings['minFastFrames']
        settings['dataOrder'] = dataOrder

model = keras.models.load_model(modelPath, compile=True)
print('model loaded')

for stack in stacks:
    print(stack)
    # Make a new folder to put the output in
    newFolder = re.split(r'.tif', stack)[0] + '_ATS'
    if os.path.exists(newFolder):
        shutil.rmtree(newFolder)
    os.mkdir(newFolder)

    # Save the settings that were used for this run
    settingsFile = os.path.join(newFolder, 'ATSSim_settings.json')
    with open(settingsFile, 'w') as fp:
        json_string = json.dumps(settings, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        fp.write(json_string)

    # Make the file that simulates the NetworkWatchdog txt file output
    txtFile = os.path.join(newFolder, 'output.txt')
    file = open(txtFile, 'w')
    file.close()
    file = open(txtFile, 'a')

    DrpOrig, MitoOrig, DrpTimes, MitoTimes = loadTifStack(stack, dataOrder, outputElapsed=True)
    print('stack loaded')

    time = DrpTimes[0]
    frame = 0
    fastMode = False
    outputFrame = 0
    delay = slowRate
    fastCount = 0

    outputData = []
    print(DrpOrig.shape[0])
    while frame < DrpOrig.shape[0]-1:
        # make tiles for current frame
        inputData, positions = prepareNNImages(MitoOrig[frame, :, :], DrpOrig[frame, :, :], 128)

        # Run neural network
        outputPredict = model.predict_on_batch(inputData)
        inputSize = round(DrpOrig.shape[1]*56/81)

        # Stitch the tiles back together (~2ms 512x512)
        outputDataFull = np.zeros([inputSize, inputSize])
        mitoDataFull = np.zeros([inputSize, inputSize])
        drpDataFull = np.zeros([inputSize, inputSize])
        i = 0
        stitch = positions['stitch']
        stitch1 = None if stitch == 0 else -stitch
        for position in positions['px']:
            outputDataFull[position[0]+stitch:position[2]-stitch,
                           position[1]+stitch:position[3]-stitch] = \
                outputPredict[i, stitch:stitch1, stitch:stitch1, 0]
            mitoDataFull[position[0]+stitch:position[2]-stitch,
                         position[1]+stitch:position[3]-stitch] = \
                inputData[i, stitch:stitch1, stitch:stitch1, 0]
            drpDataFull[position[0]+stitch:position[2]-stitch,
                        position[1]+stitch:position[3]-stitch] = \
                inputData[i, stitch:stitch1, stitch:stitch1, 1]
            i = i + 1

        # Get the output data from the nn channel
        outputData.append(np.max(outputDataFull))

        # Decide if skipping frames

        mitoPath = (newFolder + '/img_channel000_position000_time' +
                    str((outputFrame*2 + 1)).zfill(9) + '_z000.tif')
        drpPath = (newFolder + '/img_channel000_position000_time' +
                   str((outputFrame*2)).zfill(9) + '_z000.tif')
        nnPath = (newFolder + '/img_channel000_position000_time' +
                  str((outputFrame*2 + 1)).zfill(9) + '_nn.tiff')
        io.imsave(mitoPath, MitoOrig[frame, :, :].astype(np.uint16),
                  check_contrast=False, imagej=True,
                  ijmetadata={'Info': json.dumps({'ElapsedTime-ms': MitoTimes[frame]})})
        io.imsave(drpPath, DrpOrig[frame, :, :].astype(np.uint16),
                  check_contrast=False, imagej=True,
                  ijmetadata={'Info': json.dumps({'ElapsedTime-ms': DrpTimes[frame]})})
        io.imsave(nnPath, outputDataFull.astype(np.uint8), check_contrast=False)

        # Write to output.txt file as networkWatchdog would
        file.write('%d, %d\n' % (outputFrame*2 + 1, outputData[-1]))

        # if outputData[-1] > thresholdUp:
        #     delay = 0
        # elif outputData[-1] < thresholdLow and fastMode:
        #     delay = slowRate

        # Decide in which mode to go on, do at least minFastFrames fast frames after high output
        if outputFrame > 0:
            if fastMode:
                fastCount = fastCount + 1
            if outputData[-2] > thresholdUp:
                print('FAST mode')
                fastMode = True
                fastCount = 0
            elif outputData[-2] < thresholdLow and fastMode and fastCount > minFastFrames - 1:
                print('SLOW mode')
                fastMode = False

        if fastMode:
            delay = 0
        else:
            delay = slowRate

        # Write iSIMmetadata File as Matlab would
        if outputFrame > 0:
            iSIMname = ('iSIMmetadata_Timing_' + str(int(time)).zfill(9) + '.txt')
            iSIMMetadataFile = os.path.join(newFolder, iSIMname)
            metaDataFile = open(iSIMMetadataFile, 'w')
            metaDataFile.write('%d\t%.3f\t%d' % (0, delay, 1))
            metaDataFile.close()

        print('fast frames:', fastCount)
        print(frame)
        # Decide which frame to take next
        if fastMode:
            frame = frame + 1
            time = DrpTimes[frame]
        else:
            # jump slowRate seconds and find which frame would be closest to this
            time = time + slowRate*1000
            oldFrame = frame
            frame = min(range(len(DrpTimes)), key=lambda i: abs(DrpTimes[i]-time))
            # check if it will jump at least one frame if not, jump one
            if oldFrame == frame:
                frame = frame + 1
            if frame - oldFrame == 1:
                print('Jumped one frame. slowRate might not be big enough.')

            time = DrpTimes[frame]

        outputFrame = outputFrame + 1
        print(time)
        print(outputData[-1])
        print(frame)
        print('\n')

    file.close()
