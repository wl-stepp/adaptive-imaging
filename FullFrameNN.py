import numpy as np
import tifffile
# from kerassurgeon.operations import delete_layer, insert_layer
from tensorflow import keras
from tensorflow.keras.layers import Activation, Input
from tensorflow.keras.models import Model, clone_model

# from tensorflow.keras.engine import InputLayer

file = ('W:/iSIMstorage/Users/Willi/180420_drp_mito_Dora/'
        'sample1/sample2_cell_1_MMStack_Pos0_combine.ome.tif')
modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/test_model.h5'
model = keras.models.load_model(modelPath, compile=False)

# model.summary()

# newInput = Input(shape=(None, None, 2), name='input_1')    # let us say this new InputLayer
# model._layers[0] = Input

model.summary()



with tifffile.TiffFile(file) as tif:
    image1 = tif.pages[0].asarray()[0:1024, 0:1024]
    image2 = tif.pages[1].asarray()[0:1024, 0:1024]
    # image1 = np.random.rand(1024,1024)
    # image2 = np.random.rand(1024,1024)

print(image1.shape)
image1 = image1.reshape(1, image1.shape[0], image1.shape[1], 1)
image2 = image2.reshape(1, image2.shape[0], image2.shape[1], 1)
stack = np.stack((image1, image2), 3)
print(stack.shape)


# new_model = change_model(model, new_input_shape=stack.shape)
output = model.predict(stack)

print(output.shape)





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
