import os

import h5py  # HDF5 data file management library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from skimage import morphology
from tensorflow import keras


def main():
    dataPath = 'W:/Watchdog/Model/'
    collection = 'ParamSweep2'
    inputDataPath = dataPath + '/prep_data2.h5'
    hf = h5py.File(inputDataPath, 'r')
    input1 = hf.get('Mito')  # Mito
    input1 = np.array(input1).astype(np.float)
    input2 = hf.get('Drp1')  # Mito
    input2 = np.array(input2).astype(np.float)
    inputDataFull = np.stack((input1, input2), axis=3)
    print('Converting to tensor')

    outputData = hf.get('Proc')
    outputDataFull = np.array(outputData).astype(bool)
    hf.close()

    # Make the ouput data a little bigger, to not penalize to much on the correct location
    # for frame in range(outputData.shape[0]):
    #     outputData[frame] = morphology.binary_dilation(outputData[frame])
    #     outputData[frame] = morphology.binary_dilation(outputData[frame])

    print(inputDataFull.shape)
    print(outputDataFull.shape)

    pand = pd.DataFrame(columns=['model', 'filters', 'convs', 'batchSize', 'totalTruth',
                                 'totalPredict', 'maskPredict',
                                 'truePositive', 'falsePositive'])

    filters = [8, 16, 32]
    convs = [3, 5, 9, 11]
    batches = [8, 16, 32]
    for f in filters:
        for c in convs:
            for b in batches:
                modelName = ('f' + str(f).zfill(2) +
                             '_c' + str(c).zfill(2) +
                             '_b' + str(b).zfill(2))

                modelPath = os.path.join(dataPath, collection, (modelName + '.h5'))
                testSetPath = os.path.join(dataPath, collection, (modelName + '_labels.pkl'))

                model = keras.models.load_model(modelPath, compile=True)

                # Get the data that was not used for training
                labels = pd.read_pickle(testSetPath)
                inputData = inputDataFull[labels]
                inputData = tf.convert_to_tensor(inputData)
                outputData = outputDataFull[labels]
                # for part in range(0, inputTest.shape[0], 100):
                predictTest = np.zeros([len(labels), 128, 128, 1])
                step = 100
                for frame in np.arange(0, inputData.shape[0]+1, step):
                    predictTest[frame*step:frame*step+step] = model.predict_on_batch(
                        inputData[frame*step:frame*step+step])
                # Mask the prediction with the ground truth
                # Take the maximum, or what we take as input for ATS
                maxOutput = list()
                maxPredict = list()
                for frame in range(0, inputData.shape[0]):
                    maxPredict.append(np.max(predictTest[frame]))
                    maxOutput.append(np.max(outputData[frame]))
                maxPredict = np.array(maxPredict)
                maxOutput = np.array(maxOutput)
                predictTestTruePos = maxPredict[maxOutput]  # predictTest[outputData]

                totalPredict = np.sum(maxPredict)
                totalTruth = np.sum(maxOutput)
                maskPredict = np.sum(predictTestTruePos)
                truePositive = maskPredict/totalTruth
                falsePositive = (totalPredict - maskPredict)/totalTruth

                pand.loc[pand.shape[0]] = [modelName, f, c, b, totalTruth, totalPredict,
                                           maskPredict, truePositive, falsePositive]
                print(pand)
    pand.to_csv(os.path.join(dataPath, collection, 'evaluation.csv'))

    return


def visualize():
    dataPath = 'W:/Watchdog/Model/'
    collection = 'ParamSweep2'
    filePath = os.path.join(dataPath, collection, 'evaluation.csv')
    pand = pd.read_csv(filePath)
    print(pand)
    pand['truePositive'].plot(kind='bar')
    plt.show()

if __name__ == '__main__':
    visualize()
