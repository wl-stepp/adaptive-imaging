# -*- coding: utf-8 -*-
# pylint: skip-file
"""
Created on Fri Oct 23 11:01:27 2020

@author: stepp
"""

import os
import pickle
import re
import time

import h5py  # HDF5 data file management library
import matplotlib.pyplot as plt
import numpy as np
import psutil
import tensorflow as tf
from skimage import exposure, morphology, segmentation
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import (BatchNormalization, Conv2D, Conv3D,
                                            MaxPooling2D, MaxPooling3D,
                                            Reshape, UpSampling2D, concatenate)
from tqdm.keras import TqdmCallback  # Used for Progress bars during training


def prepareMitoDrp(name='Mito'):
    data_file = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/' + name + '.h5'
    hf = h5py.File(data_file, 'r')
    input_data1 = hf.get(name)  # Mito
    input_data1 = np.array(input_data1).astype(np.float)
    for frame in range(input_data1.shape[0]):
        scalingFactor = np.random.randint(30, 100)/100
        input_data1[frame] = exposure.rescale_intensity(
                    input_data1[frame],
                    (np.min(input_data1[frame]), np.max(input_data1[frame])),
                    out_range=(0, scalingFactor))
        if not frame % 100:
            printProgressBar(frame, input_data1.shape[0], printEnd='\r')
    input_data1 = input_data1.reshape(input_data1.shape[0], 128, 128, 1)
    hf.close()
    hf = h5py.File('//lebnas1.epfl.ch/microsc125/Watchdog/Model/prep_data.h5', 'a')
    try:
        del hf[name]
    except KeyError:
        print('could not delete will try to write directly')
    hf.create_dataset(name, data=input_data1)
    hf.close()

def prepareProc(threshold=150):
    data_file = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/' + 'Proc.h5'
    hf = h5py.File(data_file, 'r')
    output_data = hf.get('Proc')  # Mito
    # binarize with a certain threshold
    output_data = np.array(output_data).astype(np.float)
    hf.close()
    print('* data loaded *')
    output_data = output_data > threshold
    # double dilation to increase minimal spot size
    for frame in range(output_data.shape[0]):
        printProgressBar(frame, output_data.shape[0], printEnd='\r')
        output_data[frame] = morphology.binary_dilation(output_data[frame])
        output_data[frame] = morphology.binary_erosion(output_data[frame])
        output_data[frame] = morphology.binary_dilation(output_data[frame])
        output_data[frame] = morphology.skeletonize(output_data[frame])
        output_data[frame] = morphology.binary_dilation(output_data[frame])
        output_data[frame] = morphology.binary_dilation(output_data[frame])

        # check for something on the border of the image
        for x in range(output_data.shape[1]):
            if output_data[frame, x, 0]:
                output_data[frame] = segmentation.flood_fill(output_data[frame], (x, 0), 0)
            if output_data[frame, x, -1]:
                output_data[frame] = segmentation.flood_fill(
                    output_data[frame], (x, output_data.shape[2]-1), 0)
        for y in range(output_data.shape[2]):
            if output_data[frame, 0, y]:
                output_data[frame] = segmentation.flood_fill(output_data[frame], (0, y), 0)
            if output_data[frame, -1, y]:
                output_data[frame] = segmentation.flood_fill(
                    output_data[frame], (output_data.shape[1]-1, y), 0)


    # output_data = output_data.astype(np.float)
    # output_data = exposure.rescale_intensity(
    #             output_data,
    #             (100, np.max(output_data)),
    #             out_range=(0, 1))
    print(output_data.shape)
    plotTest = True
    if plotTest:
        fig = plt.figure()
        ax = fig.gca()
        image = ax.imshow(output_data[0], vmax=1)
        for frame in range(0, 3000, 100):
            image.set_data(output_data[frame].astype(np.uint8))
            plt.draw()
            plt.pause(1)
    output_data = output_data.reshape(output_data.shape[0], 128, 128, 1)
    hf = h5py.File('//lebnas1.epfl.ch/microsc125/Watchdog/Model/prep_data2.h5', 'a')
    try:
        del hf['Proc']
    except KeyError:
        print('Could not delete write directly')
    hf.create_dataset('Proc', data=output_data)
    hf.close()

def makeModel(input_data, output_data, nb_filters=32, firstConvSize=5, batch_size=16):
    # batch size for the training data set
    # Import data
    # Changed from the Colab version to load local data
    data_path = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/'  # nb: begin with /
    print('data_path : ', data_path, '\n')

    model_name = '/paramSweep2/temp_model'

    # Split data set into [test] and [train+valid] subsets using sklearn
    # train_test_split function

    # Prepare labels that are shuffled the same way that train_test_split will. Save these to
    # know which indices of frames where the test data for this model
    labels = np.arange(0, input_data.shape[0], 1)
    labels = shuffle(labels, random_state=data_split_state)[0:input_data.shape[0]*
                                                            data_set_test_trainvalid_ratio]

    data_set_test_trainvalid_ratio = 0.2
    data_split_state = None
    # Is this way because close images are very similar

    print('test:[train+valid] split ratio : ', data_set_test_trainvalid_ratio)
    print('data_split_state : ', data_split_state)
    print()

    input_train, input_test, output_train, output_test =  \
        train_test_split(input_data, output_data,
                         test_size=data_set_test_trainvalid_ratio,
                         random_state=data_split_state)

    print('input_data : ', input_data.shape, input_data.dtype)
    print('input_train : ', input_train.shape, input_train.dtype)
    print('input_test : ', input_test.shape, input_test.dtype)
    print()
    print('output_data : ', output_data.shape, output_data.dtype)
    print('output_train : ', output_train.shape, output_train.dtype)
    print('output_test : ', output_test.shape, output_test.dtype)

    optimizer_type = Adam(lr=0.5e-3)  # optimisation algorithm: Adam
    # Could do: Start with a higher value and then converge to a smaller value
    loss = 'binary_crossentropy'  # loss function to be minimised by the optimiser
    metrics = ['binary_accuracy']  # accuracy metric determined after ea epoch
    validtrain_split_ratio = 0.2
    # % of the seen dataset to be put aside for validation, rest is for training
    max_epochs = 20  # maxmimum number of epochs to be iterated
    batch_shuffle = True   # shuffle training data prior to; batching and each epoch

    model_version = '210219'

    if model_version == 'Dora':
        # Original version from Dora
        input_shape = (128, 128, 2, 1)
        inputs = Input(shape=input_shape)

        # encoder section

        print('* Start Encoder Section *')
        down0 = Conv3D(
            nb_filters, (firstConvSize, firstConvSize, 2), padding='same')(inputs)
        down0 = BatchNormalization()(down0)
        down0 = Activation('relu')(down0)
        down0 = Conv3D(
            nb_filters, (firstConvSize, firstConvSize, 2), padding='same')(down0)
        down0 = BatchNormalization()(down0)
        down0 = Activation('relu')(down0)
        down0_pool = MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(down0)
        down0_pool = Reshape((64, 64, nb_filters))(down0_pool)
        down0 = Reshape((128, 128, nb_filters*2))(down0)

        down1 = Conv2D(nb_filters*2, (3, 3), padding='same')(down0_pool)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1 = Conv2D(nb_filters*2, (3, 3), padding='same')(down1)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

        # center section

        print('* Start Center Section *')
        center = Conv2D(nb_filters*4, (3, 3), padding='same')(down1_pool)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        center = Conv2D(nb_filters*4, (3, 3), padding='same')(center)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)

        # decoder section with skip connections to the encoder section

        print('* Start Decoder Section *')
        up1 = UpSampling2D((2, 2))(center)
        up1 = concatenate([down1, up1], axis=3)
        up1 = Conv2D(nb_filters*2, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(nb_filters*2, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)

        up0 = UpSampling2D((2, 2))(up1)
        up0 = concatenate([down0, up0], axis=3)
        up0 = Conv2D(nb_filters, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)
        up0 = Conv2D(nb_filters, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)

        outputs = Conv2D(1, (1, 1), activation='relu')(up0)

    elif model_version == '210211':
        # Version optimized with help from MW, takes pictures of any size and makes
        # a binary decision if there is an event or not.
        input_shape = (None, None, 2)
        inputs = Input(shape=input_shape)

        # encoder section
        print('* Start Encoder Section *')

        down0 = Conv2D(
            nb_filters, (firstConvSize, firstConvSize), padding='same')(inputs)
        down0 = BatchNormalization()(down0)
        down0 = Activation('relu')(down0)
        down0 = Conv2D(
            nb_filters, (firstConvSize, firstConvSize), padding='same')(down0)
        down0 = BatchNormalization()(down0)
        down0 = Activation('relu')(down0)
        down0_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(down0)

        down1 = Conv2D(nb_filters*2, (3, 3), padding='same')(down0_pool)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1 = Conv2D(nb_filters*2, (3, 3), padding='same')(down1)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

        # center section

        print('* Start Center Section *')
        center = Conv2D(nb_filters*4, (3, 3), padding='same')(down1_pool)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        center = Conv2D(nb_filters*4, (3, 3), padding='same')(center)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)

        # decoder section with skip connections to the encoder section

        print('* Start Decoder Section *')
        up1 = UpSampling2D((2, 2))(center)
        up1 = concatenate([down1, up1], axis=3)
        up1 = Conv2D(nb_filters*2, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(nb_filters*2, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)

        up0 = UpSampling2D((2, 2))(up1)
        up0 = concatenate([down0, up0], axis=3)
        up0 = Conv2D(nb_filters, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)
        up0 = Conv2D(nb_filters, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(up0)  # was relu also before
        outputs.set_shape([None, None, None, 1])

    elif model_version == '210219':
        # deeper version with an additional layer
        input_shape = (None, None, 2)
        inputs = Input(shape=input_shape)

        # encoder section
        print('* Start Encoder Section *')

        down0 = Conv2D(
            nb_filters, (firstConvSize, firstConvSize), padding='same')(inputs)
        down0 = BatchNormalization()(down0)
        down0 = Activation('relu')(down0)
        down0 = Conv2D(
            nb_filters, (firstConvSize, firstConvSize), padding='same')(down0)
        down0 = BatchNormalization()(down0)
        down0 = Activation('relu')(down0)
        down0_pool = MaxPooling2D((2, 2), strides=(2, 2), padding='same')(down0)

        down1 = Conv2D(nb_filters*2, (3, 3), padding='same')(down0_pool)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1 = Conv2D(nb_filters*2, (3, 3), padding='same')(down1)
        down1 = BatchNormalization()(down1)
        down1 = Activation('relu')(down1)
        down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)

        down2 = Conv2D(nb_filters*2, (3, 3), padding='same')(down1_pool)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2 = Conv2D(nb_filters*2, (3, 3), padding='same')(down2)
        down2 = BatchNormalization()(down2)
        down2 = Activation('relu')(down2)
        down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)

        # center section
        print('* Start Center Section *')
        center = Conv2D(nb_filters*4, (3, 3), padding='same')(down1_pool)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)
        center = Conv2D(nb_filters*4, (3, 3), padding='same')(center)
        center = BatchNormalization()(center)
        center = Activation('relu')(center)

        # decoder section with skip connections to the encoder section
        print('* Start Decoder Section *')
        up2 = UpSampling2D((2, 2))(center)
        up2 = concatenate([down2, up2], axis=3)
        up2 = Conv2D(nb_filters*2, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)
        up2 = Conv2D(nb_filters*2, (3, 3), padding='same')(up2)
        up2 = BatchNormalization()(up2)
        up2 = Activation('relu')(up2)

        up1 = UpSampling2D((2, 2))(center)
        up1 = concatenate([down1, up1], axis=3)
        up1 = Conv2D(nb_filters*2, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)
        up1 = Conv2D(nb_filters*2, (3, 3), padding='same')(up1)
        up1 = BatchNormalization()(up1)
        up1 = Activation('relu')(up1)

        up0 = UpSampling2D((2, 2))(up1)
        up0 = concatenate([down0, up0], axis=3)
        up0 = Conv2D(nb_filters, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)
        up0 = Conv2D(nb_filters, (3, 3), padding='same')(up0)
        up0 = BatchNormalization()(up0)
        up0 = Activation('relu')(up0)

        outputs = Conv2D(1, (1, 1), activation='sigmoid')(up0)  # was relu also before
        outputs.set_shape([None, None, None, 1])

    print()
    print('* Compiling the network model *')
    print()

    t1 = time.perf_counter
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer_type, loss=loss, metrics=metrics)
    t2 = time.perf_counter

    # display a summary of the compiled neural network

    print(model.summary())
    print()
    print('* Training the compiled network *')
    print()

    t1 = time.perf_counter()
    history = model.fit(input_train, output_train,
                        batch_size=batch_size,
                        epochs=max_epochs,
                        validation_split=validtrain_split_ratio,
                        shuffle=batch_shuffle,
                        verbose=1)

    t2 = time.perf_counter()
    print()
    print('Training completed in ', (t2-t1)/60)
    print()

    # model loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss : ' + loss)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.pause(1)

    # model accuracy metric
    plt.figure()
    plt.plot(np.array(history.history[metrics[0]]))
    plt.plot(np.array(history.history['val_' + metrics[0]]))
    plt.title('Model accuracy metric : ' + metrics[0])
    plt.ylabel('Accuracy metric')
    plt.xlabel('Epoch')
    plt.legend(['Training', 'Validation'], loc='best')
    plt.pause(1)

    AverageLossVal = history.history['val_loss']
    AverageLossVal = sum(AverageLossVal[-5:])/len(AverageLossVal[-5:])

    AverageLossTrain = history.history['loss']
    AverageLossTrain = sum(AverageLossTrain[-5:])/len(AverageLossTrain[-5:])

    AverageAccTrain = history.history[metrics[0]]
    AverageAccTrain = sum(AverageAccTrain[-5:])/len(AverageAccTrain[-5:])

    AverageAccVal = history.history['val_' + metrics[0]]
    AverageAccVal = sum(AverageAccVal[-5:])/len(AverageAccVal[-5:])

    print('Average validation loss: ', AverageLossVal)
    print('Average training loss: ', AverageLossTrain)
    print('Average validation accuracy: ', AverageAccVal)
    print('Average training accuracy: ', AverageAccTrain)

    print('* Evaluating performance of trained network on the unseen dataset *')
    print()

    evaluate_model = model.evaluate(
        x=input_test, y=output_test, verbose=0,
        callbacks=[TqdmCallback(verbose=1)])
    loss_metric = evaluate_model[0]
    accuracy_metric = evaluate_model[1]

    print()
    print('Accuracy - ' + metrics[0] + ': %0.3f' % accuracy_metric)
    print('Loss - ' + loss + ': %0.3f' % loss_metric)

    # Save the compiled model to be used later
    print('* Save the Model for later use *')
    model_path = data_path + model_name
    model.save(model_path + '.h5')
    tf.keras.models.save_model(model, model_path)

    plotOutput = False
    if plotOutput:
        print('* Predicting the output of a given input from test set *')
        print()

        # for i in range(input_test.shape[0]):

        for i in range(3):
            test_id = i

            # create numpy array of required dimensions for network input
            input_predict = np.zeros(shape=(1, 128, 128, 2))

            # reshaping test input image
            input_predict[0, :, :, :] = input_test[test_id, :, :, :, 0]
            t1 = time.perf_counter()
            output_predict = model.predict(input_predict)
            t2 = time.perf_counter()

            print('time taken for prediction : ', t2-t1)

            print('test_id : ', test_id)
            print()

            # plot prediction example from test set
            fig = plt.figure()
            fig.add_subplot(2,2, 1)
            plt.imshow(input_test[test_id, :, :, 0], cmap='gray')
            plt.title('input_test [' + str(test_id) + ']')
            plt.grid(None)
            plt.xticks([])
            plt.yticks([])

            fig.add_subplot(2, 2, 2)
            plt.imshow(input_test[test_id, :, :, 1], cmap='gray')
            plt.title('input_test [' + str(test_id) + ']')
            plt.grid(None)
            plt.xticks([])
            plt.yticks([])

            fig.add_subplot(2, 2, 3)
            plt.imshow(output_predict[0, :, :, 0], cmap='gray', vmax=1)
            plt.title('output_predict')
            plt.grid(None)
            plt.xticks([])
            plt.yticks([])

            fig.add_subplot(2, 2, 4)
            plt.imshow(output_test[test_id, :, :, 0], cmap='gray',vmax=1)
            plt.title('output_test [' + str(test_id) + ']')
            plt.grid(None)
            plt.xticks([])
            plt.yticks([])
            plt.draw()
            plt.pause(2)
    return model, labels

def printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
                     length=100, fill='█', printEnd=None):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if printEnd is None:
        pprocName = psutil.Process(os.getppid()).name()
        isIDLE = bool(re.fullmatch('pyhtonw.exe', pprocName))
        if isIDLE:
            printEnd = '\n'
        else:
            printEnd = '\r'
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()


def main():
    print('* Importing data *')
    data_path = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/'  # nb: begin with /
    data_filename = data_path + 'prep_data2.h5'  # Mito
    hf = h5py.File(data_filename, 'r')
    input_data1 = hf.get('Mito')
    input_data1 = np.array(input_data1).astype(np.float)
    input_data2 = hf.get('Drp1')  # Drp1
    input_data2 = np.array(input_data2).astype(np.float)
    input_data = np.stack((input_data1, input_data2), axis=3)

    output_data = hf.get('Proc')
    output_data = np.array(output_data).astype(np.float)
    hf.close()

    print('\nInput : ', input_data.shape)
    print('Output : \n', output_data.shape)

    filters = [16, 32]  # 8
    convs = [3, 5, 9, 11]
    batches = [8, 16, 32]
    for f in filters:
        for c in convs:
            for b in batches:
                model, labels = makeModel(input_data, output_data, f, c, b)
                modelName = ('//lebnas1.epfl.ch/microsc125/Watchdog/Model/paramSweep2/f' +
                             str(f).zfill(2) + '_c' + str(c).zfill(2) + '_b' + str(b).zfill(2))
                model.save(modelName + '.h5')
                with open(modelName + '_labels.pkl', 'wb') as f:
                    pickle.dump(labels, f, pickle.HIGHEST_PROTOCOL)
                plt.close('all')


if __name__ == '__main__':
    main()
