"""This is a script to test out image slicing using he image_slicer module
        in Python. Attention, the newest version of Pillow was downgraded when
        installing image_slicer using pip
   """

from PIL import Image, ImageFont, ImageDraw
from skimage.io import imread
import time
import itertools
import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from matplotlib.axes import Axes


def calculatePixel(mn, overlap, targetSize, shape):
    xy = (int(mn[0]*(targetSize-overlap)), int(mn[1]*(targetSize-overlap)),
          int(mn[0]*(targetSize-overlap) + targetSize),
          int(mn[1]*(targetSize-overlap) + targetSize))

    # shift the last one if it goes over the edge of the image
    if mn[1]*(targetSize-overlap) + targetSize > shape[1]:
        shift = int(mn[1]*(targetSize-overlap) + targetSize) - shape[1]
        xy = (xy[0], xy[1] - shift, xy[2], xy[3] - shift)
        # print('Shifted vert for ', shift, 'pixel')

    if mn[0]*(targetSize-overlap) + targetSize > shape[0]:
        shift = int(mn[0]*(targetSize-overlap) + targetSize) - shape[0]
        xy = (xy[0] - shift, xy[1], xy[2] - shift, xy[3])
        # print('Shifted hor for ', shift, 'pixel')

    return xy


def getTilePositions(image, targetSize=128):
    """Generate tuples with the positions of tiles to split up an image with
    an overlap. Calculates the number of tiles in a way that allows for only
    full tiles to be needed.

    Args:
        filePath (PIL image): Image.open of a tiff file. Should be square and
        ideally from the geometric series (128, 256, 512, 1024, etc)
        targetSize (int, optional): target square size. Defaults to 128.

    Returns:
        [type]: [description]
    """

    # Check for the smallest overlap that gives a good result
    n = int(image.width/targetSize)+1
    cond = False

    while not cond and n < targetSize:
        overlap = (n*targetSize-image.width)/(n-1)
        if not overlap % 1:
            cond = True
        else:
            n = n + 1

    # For nxn tiles calculate the pixel positions considering the overlap
    a = [range(0, n)]*2
    positions = {'mn': tuple(itertools.product(*a)), 'px': [],
                 'overlap': overlap, 'stitch': int(overlap/2), 'n': n}

    for position in positions['mn']:
        position_xy = calculatePixel(position, overlap, targetSize)
        positions['px'].append(position_xy)

    return positions


def getTilePositions_v2(image, targetSize=128):
    """Generate tuples with the positions of tiles to split up an image with
    an overlap. Calculates the number of tiles in a way that allows for only
    full tiles to be needed.

    Args:
        filePath (PIL image): Image.open of a tiff file. Should be square and
        ideally from the geometric series (128, 256, 512, 1024, etc)
        targetSize (int, optional): target square size. Defaults to 128.

    Returns:
        [type]: [description]
    """
    # Check for the smallest overlap that gives a good result
    n = int(image.shape[0]/targetSize)+1
    cond = False
    min_overlap = 35

    while not cond and n < targetSize and n > 1:
        overlap = (n*targetSize-image.shape[0])/(n-1)
        if overlap % 2: overlap = overlap - 1
        if int(overlap) >= min_overlap:
            overlap = int(overlap)
            cond = True
        else:
            n = n + 1


    # For nxn tiles calculate the pixel positions considering the overlap
    a = [range(0, n)]*2
    positions = {'mn': tuple(itertools.product(*a)), 'px': [],
                 'overlap': overlap, 'stitch': int(overlap/2), 'n': n}

    for position in positions['mn']:
        position_xy = calculatePixel(position, overlap, targetSize,
                                     image.shape)
        positions['px'].append(position_xy)

    return positions


if __name__ == "__main__":
    path = '//lebnas1.epfl.ch/microsc125/Watchdog/test_image/Default'
    # fileName = 'img_channel000_position000_time000000000_z000.tif'
    fileName = 'moon.tif'
    filePath = path + '/' + fileName
    image = io.imread(filePath)

    positions = getTilePositions_v2(image, 128)

    fig = plt.figure()
    ax = plt.axes()
    plt.draw()
    image_1 = np.zeros(image.shape)
    stitch = positions['stitch']
    stitch1 = None if stitch == 0 else -stitch

    for position in positions['px']:
        setWidth = 0.5
        setColor = [0.7, 0.7, 0.7]
        Axes.axhline(ax, y=position[0], linewidth=setWidth, color=setColor)
        Axes.axvline(ax, x=position[1], linewidth=setWidth, color=setColor)
        Axes.axhline(ax, y=position[2], linewidth=setWidth, color=setColor)
        Axes.axvline(ax, x=position[3], linewidth=setWidth, color=setColor)

    n = positions['n']
    overlap = positions['overlap']
    print(n)
    print(n*(128-overlap)+overlap)

    plt.imshow(image)
    plt.pause(0.1)
    plt.show()
