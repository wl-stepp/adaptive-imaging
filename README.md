# adaptive-imaging
Tools for adaptive imaging in the LEB-EPFL

This project was used for the realization of adaptive temporal sampling (ATS) imaging of mitochondria and caulobacter. The two main modules are NetworkWatchdog.py and NNGui.py.

The overall idea of this project is to adapt the imaging speed of a microscope to the presence of events in the images. This is achieved by implementing a feedback loop from the images taken to the microscope control. NetworkWatchdog implements part of the feedback loop, while NNGui can be used to show the resulting data. The steps of the Feedback loop are:
* Micro-Manager saves the images to a network location
* The files are detected by NetworkWatchdog
* NetworkWatchdog computes a decision parameter
* A binary file with this information is written to the network location
* The Matlab program controlling the microscope reads that data

This will result in data with different framerates, which makes for an interesting problem for actually displaying the data. Hence NNGui.

As of 16.04.2021 this is still work in progress. The goal is to get the feedback loop closer to the actual microscope, by implementing the microscope control directly in python.

Some of the important settings are in ATS_settings.json. This was used to synchronize settings on different machines that were used to run the different files.

## NetworkWatchdog
This module looks for new files in a network location. If the files are tif files saved from Micro-Manager, the files will be prepared for the neural network. The neural network gives a heatmap of possible events in the data and the maximum of that value is saved to the binary file.

## NNGui
NNGui provides a GUI to look at the data produced by ATS. I loads a .h5 keras model for inference on the provided images.