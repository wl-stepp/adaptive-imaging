# -*- coding: utf-8 -*-
"""
This is a GUI that runs a neural network on the input data and displays
original data and the result in a scrollable form

Created on Mon Oct  5 12:18:48 2020

@author: stepp
"""

import tkinter as tk
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pathlib import Path
from skimage import io
import numpy as np
import time


class ImageProc(tk.Frame):

    # Defines settings upon initialization.
    def __init__(self, master=None):

        # parameters we want to send through Frame class
        tk.Frame.__init__(self, master)

        # reference to the master widget, which is the tk window
        self.master = master

        # initialization of imageProc frame
        self.init_imageProc()

    # here we create the initialization frame
    def init_imageProc(self):

        # set the title of master widget
        self.master.title("Image Processing")

        # allowing the widget to take the full space of the root window
        self.pack(fill=tk.BOTH, expand=1)
        self.create_widgets()
        # creating a menu instance
        
    def create_widgets(self):
        self.browse = tk.Button(self)
        self.browse["text"] = "Show Image"
        self.browse["command"] = self.loadImage
        self.browse.grid(row=0, column=0)
        
        self.frame = tk.Scrollbar(self, orient='horizontal', repeatdelay = 50)
        self.frame["command"] = self.setFrameText
        self.frame.grid(row=1, column=1, sticky='WE')
              
        self.frameLabel = tk.Label(self)
        self.frameLabel['text'] = 'Frame'
        self.frameLabel.grid(row=1, column=0)
        self.frameNum = 0 
        
        self.fig = [mpl.figure.Figure(figsize=(5, 5))]
        self.ax = [self.fig[0].add_subplot(111)]
        self.canvas = [FigureCanvasTkAgg(self.fig[0], self)]
        self.canvas[0].get_tk_widget().grid(row=0, column=1)
        self.canvas[0].draw()
        
        self.fig.append(mpl.figure.Figure(figsize=(5, 5)))
        self.ax.append(self.fig[1].add_subplot(111))
        self.canvas.append(FigureCanvasTkAgg(self.fig[1], self))
        self.canvas[1].get_tk_widget().grid(row=0, column=2)
        self.canvas[1].draw()
        
        self.fig.append(mpl.figure.Figure(figsize=(5, 5)))
        self.ax.append(self.fig[2].add_subplot(111))
        self.canvas.append(FigureCanvasTkAgg(self.fig[2], self))
        self.canvas[2].get_tk_widget().grid(row=0, column=3)
        self.canvas[2].draw()
        print(self.canvas)
        
    
    
    
    def loadImage(self):
        folder = 'C:/Users/stepp/Documents/data_raw/SmartMito/'
        mito_file = folder + 'cell_8_mito.tif'
        drp_file = folder + 'cell_8_drp1.tif'
        self.image = io.imread(mito_file)
        self.image_drp = io.imread(drp_file)
        print(self.image.shape)
        self.im = self.ax[0].imshow(self.image[0, :, :])  # for laterself.im.set_data(new_data)
        self.im_drp = self.ax[1].imshow(self.image_drp[0, :, :]) 
        # DrawingArea
        for canvas in self.canvas:
            canvas.draw()
        self.frameNum = 0
        
    def setFrameText(self, x, y, z=0):
        
        if x == 'scroll':
            if z == 'pages':
                y = 10
            self.frameNum = self.frameNum + int(y)
            self.frame.set(self.frameNum/self.image.shape[0], 
                (self.frameNum + 1)/self.image.shape[0])
        elif x == 'moveto':
            self.frameNum = int(float(y)*self.image.shape[0])
            self.frame.set(float(y), float(y) + 1/self.image.shape[0])
        
        
        self.im.set_data(self.image[self.frameNum, :, :])
        self.im_drp.set_data(self.image_drp[self.frameNum, :, :])
        self.frameLabel['text'] = str(self.frameNum)
        
        t1 = time.perf_counter()
        i = 0
        for figure in self.fig: 
            figure.canvas.blit(self.ax[i].bbox)
            i = i + 1
        #for canvas in self.canvas: canvas.draw()
        t2 = time.perf_counter()
        print((t2-t1)*1000)      
        
# root window created. Here, that would be the only window, but
# you can later have windows within windows.
root = tk.Tk()

root.geometry("1600x1024")

# creation of an instance
app = ImageProc(root)

# mainloop
root.mainloop()

print('Done')