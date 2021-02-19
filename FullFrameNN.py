import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tifffile
# from kerassurgeon.operations import delete_layer, insert_layer
from tensorflow import keras

from SmartMicro.NNfeeder import prepareNNImages

# from tensorflow.keras.engine import InputLayer

file = ('W:/iSIMstorage/Users/Willi/180420_drp_mito_Dora/'
        'sample1/sample2_cell_1_MMStack_Pos0_combine.ome.tif')
fileDrp = ('C:/Users/stepp/Documents/02_Raw/SmartMito/ATSSim_example/'
         'img_channel000_position000_time000000000_z000_prep.tif')
fileMito = ('C:/Users/stepp/Documents/02_Raw/SmartMito/ATSSim_example/'
         'img_channel000_position000_time000000001_z000_prep.tif')
filePath = '/W:/iSIMstorage/Users/Willi/180420_drp_mito_Dora/sample1/sample1_cell_2_MMStack_Pos0_combine.ome_ATS/'
modelPath = '//lebnas1.epfl.ch/microsc125/Watchdog/Model/test_model4.h5'
modelPath2 = 'C:/Users/stepp/Documents/02_Raw/SmartMito/model_Dora.h5'

model = keras.models.load_model(modelPath, compile=True)

print(model.layers[0].input_shape[0][1])

nnOutput = []
for frame in range(0, 300, 2):
    fileDrp = filePath + 'img_channel000_position000_time' + str(frame).zfill(9) + '_z000.tif'  # _prep.tif'
    fileMito = filePath + 'img_channel000_position000_time' + str(frame+1).zfill(9) + '_z000.tif'  # _prep.tif'

    endFrame = 648
    with tifffile.TiffFile(fileMito) as tif:
        image1 = tif.pages[0].asarray()  # [128:endFrame, 128:146]
    with tifffile.TiffFile(fileDrp) as tif:
        image2 = tif.pages[0].asarray()  # [128:endFrame, 128:146]


    # image1 = image1.reshape(1, image1.shape[0], image1.shape[1], 1)
    # image2 = image2.reshape(1, image2.shape[0], image2.shape[1], 1)
    # stack = np.stack((image1, image2), 3)

    stack, pos = prepareNNImages(image1, image2, model)

    output = model.predict_on_batch(stack)
    output = output[0, :, :, 0]
    nnOutput.append(np.max(output))
    print(str(frame), end='\r')
    if not frame % 20:
        print(frame)


plt.figure()
plt.imshow(image1)
plt.figure()
plt.imshow(image2)
plt.figure()
plt.imshow(output)
plt.show()
plt.plot(nnOutput)
plt.show()
