# -*- coding: utf-8 -*-
# pylint: skip-file
"""
Created on Fri Oct 23 11:01:27 2020

@author: stepp
"""

import time

import h5py  # HDF5 data file management library
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras.layers import (BatchNormalization, Conv2D, Conv3D,
                                            MaxPooling2D, MaxPooling3D,
                                            Reshape, UpSampling2D, concatenate)
from tqdm.keras import TqdmCallback  # Used for Progress bars during training


def image_stack_blur(image_stack, sigma_x, sigma_y):

    import cv2

    image_size = image_stack.shape[1] * image_stack.shape[2]
    img = []

    for i in range(len(image_stack)):
        img.append(cv2.GaussianBlur(image_stack[i], ksize=(0, 0),
                                    sigmaX=sigma_x, sigmaY=sigma_y))

    img = np.array(img)
    img = img.reshape(len(img), image_size, 1)

    return(img)


# Import data
# Changed from the Colab version to load local data

data_path = 'E:/Watchdog/SmartMito/'  # nb: begin with /
print('data_path : ', data_path)
print()
input_data_filename1 = data_path + 'Mito.h5'  # Mito
input_data_filename2 = data_path + 'Drp1.h5'  # Drp1

output_class_filename = data_path + 'Proc.h5'
# output_class_onehot_filename = data_path + 'output_class_onehot.h5'
# output_class_names_filename = data_path + 'output_class_names.npy'

hf = h5py.File(input_data_filename1, 'r')
input_data1 = hf.get('Mito')  # Mito
input_data1 = np.array(input_data1)
input_data1 = input_data1.reshape(input_data1.shape[0], 128, 128, 1)

print('Input : ', input_data1.shape)
print('Input : ', input_data1.shape[0])


hf = h5py.File(input_data_filename2, 'r')
input_data2 = hf.get('Drp1')  # Drp1
input_data2 = np.array(input_data2)
input_data2 = input_data2.reshape(input_data2.shape[0], 128, 128, 1)

print('Input : ', input_data2.shape)
print('Input : ', input_data2.shape[0])


input_data = np.concatenate((input_data1, input_data2), axis=3)
input_data = input_data.reshape(input_data.shape[0], 128, 128, 2, 1)


hf = h5py.File(output_class_filename, 'r')
temp = hf.get('Proc')
output_data = np.array(temp)
print('Output : ', output_data.shape)
print('Output : ', output_data.shape[0])
output_data = output_data.reshape(output_data.shape[0], 128, 128, 1, 1)

print('* Importing data *')
print()
print('Input : ', input_data.shape)
print('Output : ', output_data.shape)
print()

# plot example

item_id = 5


plt.imshow(input_data[item_id, :, :, 0, 0], cmap='viridis')
plt.title('input_data [' + str(item_id) + ']')
plt.grid(None)
plt.xticks([])
plt.yticks([])
plt.draw()

plt.imshow(output_data[item_id, :, :, 0, 0], cmap='viridis')
plt.title('output_data [' + str(item_id) + ']')
plt.grid(None)
plt.xticks([])
plt.yticks([])
plt.draw()


# Split data set into [test] and [train+valid] subsets using sklearn
# train_test_split function

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
loss = 'mean_squared_error'  # loss function to be minimised by the optimiser
metrics = ['mean_absolute_error']  # accuracy metric determined after ea epoch
validtrain_split_ratio = 0.2
# % of the seen dataset to be put aside for validation, rest is for training
max_epochs = 20  # maxmimum number of epochs to be iterated
batch_size = 256   # batch size for the training data set
batch_shuffle = True   # shuffle training data prior to batching and each epoch

nb_filters = 8
firstConvSize = 9


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
                    verbose=0,
                    callbacks=[TqdmCallback(verbose=1)])

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
model_path = data_path + 'test_model'
tf.keras.models.save_model(model, model_path)

print('* Predicting the output of a given input from test set *')
print()

# for i in range(input_test.shape[0]):

for i in range(3):
    test_id = i

    # create numpy array of required dimensions for network input
    input_predict = np.zeros(shape=(1, 128, 128, 2, 1))

    # reshaping test input image
    input_predict[0, :, :, :, 0] = input_test[test_id, :, :, :, 0]
    t1 = time.perf_counter()
    output_predict = model.predict(input_predict)
    t2 = time.perf_counter()

    print('time taken for prediction : ', t2-t1)

    print('test_id : ', test_id)
    print()

    # plot prediction example from test set
    fig = plt.figure()
    fig.add_subplot(1, 4, 1)
    plt.imshow(input_test[test_id, :, :, 0, 0], cmap='gray')
    plt.title('input_test [' + str(test_id) + ']')
    plt.grid(None)
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(1, 4, 2)
    plt.imshow(input_test[test_id, :, :, 1, 0], cmap='gray')
    plt.title('input_test [' + str(test_id) + ']')
    plt.grid(None)
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(1, 4, 3)
    plt.imshow(output_predict[0, :, :, 0], cmap='gray')
    plt.title('output_predict')
    plt.grid(None)
    plt.xticks([])
    plt.yticks([])

    fig.add_subplot(1, 4, 4)
    plt.imshow(output_test[test_id, :, :, 0, 0], cmap='gray')
    plt.title('output_test [' + str(test_id) + ']')
    plt.grid(None)
    plt.xticks([])
    plt.yticks([])
    plt.draw()
    plt.pause(2)
