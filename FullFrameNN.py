import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile
# from kerassurgeon.operations import delete_layer, insert_layer
from tensorflow import keras

# from tensorflow.keras.engine import InputLayer

file = ('W:/iSIMstorage/Users/Willi/180420_drp_mito_Dora/'
        'sample1/sample2_cell_1_MMStack_Pos0_combine.ome.tif')
file1 = 'C:/Users/stepp/Documents/02_Raw/SmartMito/ATSSim_example/img_channel000_position000_time000000000_z000_prep.tif'
file2 = 'C:/Users/stepp/Documents/02_Raw/SmartMito/ATSSim_example/img_channel000_position000_time000000001_z000_prep.tif'
modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/test_model3.h5'

model = keras.models.load_model(modelPath, compile=True)

# model.summary()

# newInput = Input(shape=(None, None, 2), name='input_1')    # let us say this new InputLayer
# model._layers[0] = Input



model.summary()

with tifffile.TiffFile(file1) as tif:
    image1 = tif.pages[0].asarray()[356:684, 356:684]

with tifffile.TiffFile(file2) as tif:
    image2 = tif.pages[0].asarray()[356:684, 356:684]
    # image1 = np.random.rand(1024,1024)
    # image2 = np.random.rand(1024,1024)

print(image1.shape)
print(image2.shape)
image1 = image1.reshape(1, image1.shape[0], image1.shape[1], 1)
image2 = image2.reshape(1, image2.shape[0], image2.shape[1], 1)
stack = np.stack((image1, image2), 3)
print(stack.shape)

# new_model = change_model(model, new_input_shape=stack.shape)
output = model.predict(stack)

print(output.shape)

plt.figure()
plt.imshow(image1[0, :, :, 0])
plt.figure()
plt.imshow(image2[0, :, :, 0])
plt.figure()
plt.imshow(output[0, :, :, 0])
plt.show()
# newInput = Input(batch_shape=(None, 1024, 1024, 2), name='image_input')
# model = delete_layer(model, model._layers[0])
# model = insert_layer(model, model.layers[0], newInput)

# def change_model(model, new_input_shape=(None, 40, 40, 3)):
#     """ Change expected shape of input for model"""
#     # replace input shape of first layer
#     print(new_input_shape)
#     model._layers[0].batch_input_shape = new_input_shape

#     print('new:')
#     print(model._layers[0].batch_input_shape)
#     # rebuold model
#     new_model = keras.models.model_from_json(model.to_json())
#     new_model.summary()
#     # copy weights
#     for layer in new_model.layers:
#         layer.set_weights(model.get_layer(name=layer.name).get_weights())

#     return new_model
