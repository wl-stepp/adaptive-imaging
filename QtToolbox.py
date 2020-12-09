from matplotlib import cm
import numpy as np
from PyQt5.QtGui import qRgb
import matplotlib
import pyqtgraph


def getImageItemcolormap(name='hot', alpha=False):
    colormap = cm.get_cmap(name)
    colormap._init()
    colormap._lut = colormap._lut[:255]
    if alpha:
        a = np.linspace(0.3, 0.85, 255)
        colormap._lut[:, 3] = a
    lut = (colormap._lut * 255).view(np.ndarray).astype(int)
    # lut = [qRgb(i[0], i[1], i[2]) for i in lut]
    return lut


def getQtcolormap(name='hot'):
    colormap = cm.get_cmap(name)
    colormap._init()
    lut = (colormap._lut * 255).view(np.ndarray).astype(int)
    lut = [qRgb(i[0], i[1], i[2]) for i in lut]
    return lut


def getImageViewcolormap(name='hot'):
    """This can be used in a pyqtgrah ImageView to set a colormap that"s known from matplotlib

    Args:
        name (str, optional): [description]. Defaults to 'hot'.

    Returns:
        [type]: [description]
    """
    numColors = 5
    colormap = cm.get_cmap(name, numColors)
    colormap._init()
    lut = []
    pos = np.linspace(0, 1, num=numColors)
    a = (np.linspace(0, 1, num=numColors)*255).astype(int)
    a = np.append(a, [255, 255, 0])
    lut = (colormap._lut * 255).view(np.ndarray).astype(int)
    lut[:, 3] = a
    print(lut[:, 3])
    lut = pyqtgraph.ColorMap(pos, lut)
    return lut


if __name__ == '__main__':
    getPyQtcolormap('Reds')
