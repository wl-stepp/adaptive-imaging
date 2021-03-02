import os

import h5py  # HDF5 data file management library
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.backends.backend_qt5agg import \
    FigureCanvasQTAgg as FigureCanvas
from PyQt5 import QtCore, QtWidgets
from skimage import io
from tensorflow import keras
from tqdm import tqdm

from NNfeeder import prepareNNImages

DataPath = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/'
Colors = {
    'ats':  (210/255, 42/255, 38/255),
    'slow': (2/255, 53/255, 62/255),
    'fast': (2/255, 147/255, 164/255)
}


def main(collection='paramSweep'):

    inputDataPath = DataPath + collection + '/prep_data' + collection[-1] + '.h5'
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

    try:
        iSIMdataPath = 'C:/Users/stepp/Documents/02_Raw/SmartMito/180420_130.tif'
        iSIMdata = io.imread(iSIMdataPath)
    except FileNotFoundError:
        iSIMdataPath = '//lebnas1.epfl.ch/microsc125/Watchdog/GUI/180420_130.tif'
        iSIMdata = io.imread(iSIMdataPath)



    # Make the ouput data a little bigger, to not penalize to much on the correct location
    # for frame in range(outputData.shape[0]):
    #     outputData[frame] = morphology.binary_dilation(outputData[frame])
    #     outputData[frame] = morphology.binary_dilation(outputData[frame])

    print(inputDataFull.shape)
    print(outputDataFull.shape)

    pand = pd.DataFrame(columns=['model', 'filters', 'convs', 'batchSize', 'totalTruth',
                                 'totalPredict', 'predictVariance', 'maskPredictThresh',
                                 'maskPredict',
                                 'truePositive', 'truePositiveThresh', 'falsePositive',
                                 'falsePositiveThresh', 'iSIMoutput'])

    # pand.loc[pand.shape[0]] = [modelName, f, c, b, totalTruth, totalPredict,
    #                             maskPredictThresh,
    #                             maskPredict, truePositive, truePositiveThresh,
    #                             falsePositive, falsePositiveThresh]

    filters = [8, 16, 32]
    convs = [3, 5, 7, 9]
    batches = [8, 16, 32]
    for f in filters:
        for c in convs:
            for b in batches:
                modelName = ('f' + str(f).zfill(2) +
                             '_c' + str(c).zfill(2) +
                             '_b' + str(b).zfill(2))

                modelPath = os.path.join(DataPath, collection, (modelName + '.h5'))
                testSetPath = os.path.join(DataPath, collection, (modelName + '_labels.pkl'))

                model = keras.models.load_model(modelPath, compile=True)

                # Get the data that was not used for training
                labels = pd.read_pickle(testSetPath)
                inputData = inputDataFull[labels]
                inputData = tf.convert_to_tensor(inputData)
                outputData = outputDataFull[labels]
                # for part in range(0, inputTest.shape[0], 100):
                predictTest = np.zeros([len(labels), 128, 128, 1])
                step = 100
                for frame in tqdm(np.arange(0, int(inputData.shape[0]/step))):
                    predictTest[frame*step:frame*step+step] = model.predict_on_batch(
                        inputData[frame*step:frame*step+step])

                # Predict on the actual iSIM data
                iSIMoutput = []
                for frame in tqdm(range(0, iSIMdata.shape[0], 2)):
                    iSIMdataprep = prepareNNImages(iSIMdata[frame], iSIMdata[frame+1], None)
                    iSIMoutput.append(np.max(model.predict(iSIMdataprep[0])))

                # Mask the prediction with the ground truth
                # Take the maximum, or what we take as input for ATS
                maxOutput = list()
                maxPredict = list()
                for frame in range(0, inputData.shape[0]):
                    maxPredict.append(np.max(predictTest[frame]))
                    maxOutput.append(np.max(outputData[frame]))
                maxPredict = np.array(maxPredict)

                predictVariance = np.var(maxPredict)

                maxOutput = np.array(maxOutput)
                predictTestTruePos = maxPredict[maxOutput]  # predictTest[outputData]

                totalPredict = np.sum(maxPredict)
                totalTruth = np.sum(maxOutput)
                maskPredict = np.sum(predictTestTruePos)
                maskPredictThresh = np.sum(predictTestTruePos > 0.85)
                truePositive = maskPredict/totalTruth
                truePositiveThresh = maskPredictThresh/totalTruth
                falsePositive = (totalPredict - maskPredict)/totalTruth
                falsePositiveThresh = (np.sum(maxPredict > 0.85) - maskPredictThresh)/totalTruth

                pand.loc[pand.shape[0]] = [modelName, f, c, b, totalTruth, totalPredict,
                                           predictVariance, maskPredictThresh,
                                           maskPredict, truePositive, truePositiveThresh,
                                           falsePositive, falsePositiveThresh, iSIMoutput]
                print(pand)
    pand.to_csv(os.path.join(DataPath, collection, 'evaluation.csv'))

    return


class TableWindow(QtWidgets.QMainWindow):
    """ A window that we can plot into that gives us rows """
    def __init__(self):
        self.qapp = QtWidgets.QApplication([])
        QtWidgets.QMainWindow.__init__(self)
        self.widget = QtWidgets.QWidget()
        self.setCentralWidget(self.widget)
        self.widget.setLayout(QtWidgets.QVBoxLayout())
        self.widget.layout().setContentsMargins(0, 0, 0, 0)
        self.widget.layout().setSpacing(0)

        self.headerBox = QtWidgets.QGroupBox()
        gridBox = QtWidgets.QGridLayout(self.headerBox)
        gridBox.addWidget(QtWidgets.QLabel('True/Flase Positives thresholded'), 0, 1)
        gridBox.addWidget(QtWidgets.QLabel('True/Flase Positives'), 0, 2)

        self.widget.layout().addWidget(self.headerBox)
        self.lines = 0

    def addLine(self, fig):
        canvas = FigureCanvas(fig)
        canvas.draw()
        canvas
        self.widget.layout().addWidget(canvas)

    def done(self):
        self.show()
        exit(self.qapp.exec_())

def visualizeDecision(collections=['paramSweep6', 'paramSweep7']):

    # Make line for one model
    modelName = 'f08_c03_b08'
    model = os.path.join(DataPath, collection, 'f08_c03_b08.h5')
    evalFilePath = os.path.join(DataPath, collection, 'evaluation.csv')
    pand = pd.read_csv(evalFilePath)
    print(pand)

    # Make a window that allows for esay adding of plot lines
    win = TableWindow()

    modelData = pand.loc[pand['model'] == 'f08_c03_b08']
    fig, axes = plt.subplots(ncols=4, nrows=1, figsize=(16, 3))

    axes[0].barh([0, 1],
                 [modelData['truePositiveThresh'][0], modelData['falsePositiveThresh'][0]],
                 1,
                 color=[Colors['slow'], Colors['ats']],
                 zorder = 10)


    axes[1].barh([0,1],
                [modelData['truePositive'][0], modelData['falsePositive'][0]],
                1,
                color=[Colors['slow'], Colors['ats']],
                alpha=0.5,
                zorder = 10)

    for i in [0, 1]:
        axes[i].set_xlim((0, 1))
        axes[i].set_xticks([0.05, 0.1, 0.15, 0.7, 0.8, 0.9])
        axes[i].set_xticklabels(['', '', '', '', '', ''])
        axes[i].set_yticks([])
        axes[i].set_frame_on(False)
        axes[i].grid(axis='x', zorder=1)

    axes[2].plot([1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1],
                 color=Colors['slow'])


    win.addLine(fig)
    win.done()




def visualize(collection='paramSweep'):
    """Function to visualize the overall outcome of a paramSweep of models for different parameters

    Args:
        collection (str, optional): Which collection to use for analysis. Defaults to 'paramSweep'.
    """
    dataPath = 'W:/Watchdog/Model/'
    filePath = os.path.join(dataPath, collection, 'evaluation.csv')
    pand = pd.read_csv(filePath)
    print(pand)

    axesList = []

    fig1 = plt.figure(figsize=(20,10))
    ax = fig1.add_subplot(331)
    axesList.append(ax)
    pand.sort_values('filters').plot(x='filters', y='truePositiveThresh', kind='bar', ax=ax, zorder = 10)
    ax.set_title('Sorted by filters')
    plt.ylabel('True Positive Thresholded')
    ax = fig1.add_subplot(332)
    axesList.append(ax)
    pand.sort_values('convs').plot(x='convs', y='truePositiveThresh', kind='bar', ax=ax, zorder = 10)
    ax.set_title('Sorted by firstConv')
    ax = fig1.add_subplot(333)
    axesList.append(ax)
    pand.sort_values('batchSize').plot(x='batchSize', y='truePositiveThresh', kind='bar', ax=ax, zorder = 10)
    ax.set_title('Sorted by batchSize')

    ax = fig1.add_subplot(334)
    axesList.append(ax)
    pand.sort_values('filters').plot(x='filters', y='totalPredict', kind='bar', ax=ax, zorder = 10)
    plt.ylabel('Total Prediction')
    ax = fig1.add_subplot(335)
    axesList.append(ax)
    pand.sort_values('convs').plot(x='convs', y='totalPredict', kind='bar', ax=ax, zorder = 10)
    ax = fig1.add_subplot(336)
    axesList.append(ax)
    pand.sort_values('batchSize').plot(x='batchSize', y='totalPredict', kind='bar', ax=ax, zorder = 10)

    ax = fig1.add_subplot(337)
    axesList.append(ax)
    pand.sort_values('filters').plot(x='filters', y='falsePositiveThresh', kind='bar', ax=ax, zorder = 10)
    plt.ylabel('False Positive Thresholded')
    ax = fig1.add_subplot(338)
    axesList.append(ax)
    pand.sort_values('convs').plot(x='convs', y='falsePositiveThresh', kind='bar', ax=ax, zorder = 10)
    ax = fig1.add_subplot(339)
    axesList.append(ax)
    pand.sort_values('batchSize').plot(x='batchSize', y='falsePositiveThresh', kind='bar', ax=ax, zorder = 10)

    for ax in axesList:
        ax.get_legend().remove()
        ax.grid(axis='y', zorder=1)
    plt.show()



if __name__ == '__main__':
    collection = 'paramSweep5'
    # visualizeDecision(['paramSweep5'])
    main(collection)
    # visualize(collection)
