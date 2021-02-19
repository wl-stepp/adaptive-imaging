import h5py  # HDF5 data file management library
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras


def main():
    dataPath = 'W:/Watchdog/Model/'
    modelName = 'export_test'

    inputDataPath = dataPath + modelName + '/input_test.h5'
    hf = h5py.File(inputDataPath, 'r')
    inputTest = hf.get('input')  # Mito
    inputTest = np.array(inputTest)[:, :, :, :, 0]
    hf.close()

    outputDataPath = dataPath + modelName + '/output_test.h5'
    hf = h5py.File(outputDataPath, 'r')
    outputTest = hf.get('output')  # Mito
    outputTest = np.array(outputTest)[:, :, :, :, 0]
    hf.close()

    modelPath = dataPath + modelName +'.h5'
    model = keras.models.load_model(modelPath, compile=True)
    predictTest = np.zeros_like(outputTest)
    for part in range(0, inputTest.shape[0], 100):
        predictTest[part:part+100] = model.predict_on_batch(inputTest[part:part+100])

    print(outputTest.shape)

    # go through all the frames and compare output and prediction
    peakValue = 1000
    threshold = 1

    peaks = []
    for frame in range(outputTest.shape[0]):
        outputFrame = outputTest[frame]
        outputFrameCopy = outputFrame
        predictFrame = predictTest[frame]
        predictFrameCopy = predictFrame
        while peakValue > threshold:
            # get the momentary peak value
            peakValue = np.max(predictFrameCopy)
            maxPos = list(zip(*np.where(predictFrameCopy == peakValue)))
            # check what is there in the output
            peakOutput = np.max(outputFrame[maxPos[0]-2:maxPos[0]+3, maxPos[1]-2:maxPos[1]+3])
            print(peakValue)

if __name__ == '__main__':
    main()
