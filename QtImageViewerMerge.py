from PyQt5.QtCore import Qt, pyqtSignal, QT_VERSION_STR, QTimer, QRectF, QPointF
from PyQt5.QtWidgets import QWidget, QSlider, QPushButton, QLabel,\
    QGridLayout, QFileDialog, QProgressBar, QGroupBox, QApplication, QGraphicsScene, QGraphicsView
from PyQt5.QtGui import QColor, QBrush, QPen, QMovie, qRgb, QPixmap, QImage, QPicture, QPainter
import numpy as np
from QtToolbox import getImageItemcolormap
from qimage2ndarray import gray2qimage, array2qimage
from skimage import io
import sys
from QtImageViewer import QtImageViewer
import pyqtgraph as pg
from matplotlib import cm
import time


class CrossItem(pg.GraphicsObject):
    def __init__(self, rect, color='#CCCCCC', parent=None):

        super().__init__(parent)
        self._rect = rect
        self.color = color
        self.picture = QPicture()
        self._generate_picture()
        self.setZValue(100)

    @property
    def rect(self):
        return self._rect

    def _generate_picture(self, pos=[0, 0]):
        size = 5
        painter = QPainter(self.picture)
        painter.setPen(pg.mkPen(color=self.color))
        painter.setBrush(pg.mkBrush(color=self.color))
        painter.drawLine(pos[0]-size, pos[1]-size, pos[0]+size, pos[1]+size)
        painter.drawLine(pos[0]+size, pos[1]-size, pos[0]-size, pos[1]+size)
        painter.end()

    def setPosition(self, pos):
        pos = QPointF(pos[0][1]+0.5, pos[0][0]+0.5)
        self.setPos(pos)

    def paint(self, painter, option, widget=None):
        painter.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())


class QtImageViewerMerge(pg.GraphicsWindow):

    frameChanged = pyqtSignal([], [int])
    resizedEmitter = pyqtSignal([], [int])

    def __init__(self):
        pg.GraphicsWindow.__init__(self)

        # Image frame viewer.
        self.vb = self.addViewBox()
        self.vb.setAspectLocked()
        self.vb.invertY()
        self.Z = 0
        self.images = []
        self.imageItems = []

        self.cross = CrossItem(QRectF(0, 0, 1, 1))
        self.vb.addItem(self.cross)

        self.vb.sigRangeChanged.connect(self.resizedEvent)

    def setImage(self, img, pos=0):
        self.imageItems[pos].setImage(img)

    def addImage(self, img=None):
        if img is None:
            imgItem = pg.ImageItem()
        else:
            imgItem = pg.ImageItem(img)

        self.vb.addItem(imgItem)
        imgItem.setZValue(self.Z)
        self.Z = self.Z + 1
        self.imageItems.append(imgItem)
        imgItem.setOpts(axisOrder='row-major')
        return imgItem

    def setLookupTable(self, img, name='hot', transparency=False):
        img.setLookupTable(getImageItemcolormap(name, transparency))

    def resizedEvent(self):
        self.resizedEmitter.emit()


# pg.ViewBox.viewRange.__get__

if __name__ == '__main__':

    fname = ('C:/Users/stepp/Documents/data_raw/SmartMito/__short.tif')
    image_mitoOrig = io.imread(fname)
    image_drpOrig = image_mitoOrig[1]
    image_mitoOrig = image_mitoOrig[0]
    mito = image_mitoOrig/np.max(image_mitoOrig)
    drp = image_drpOrig/np.max(image_drpOrig)

    app = QApplication(sys.argv)

    # win = pg.GraphicsScene()
    # vb = win.addViewBox()
# make sure this image is on top
    # img2.setOpacity(0.5)
    # pyqtgraph.ColorMap()
    # viewer.setColorMap(getPyQtcolormap())
    win = pg.GraphicsWindow()

    # Show viewer and run application.
    viewer = QtImageViewerMerge()
    viewer2 = QtImageViewerMerge()

    grid = QGridLayout(win)
    grid.addWidget(viewer, 0, 0)
    grid.addWidget(viewer2, 0, 1)

    viewer.addImage()

    viewer.addImage(mito)
    viewer.setLookupTable(viewer.imageItems[0], 'viridis', False)
    viewer.setLookupTable(viewer.imageItems[1], 'hot', True)
    viewer.setImage(drp, 0)

    viewer2.addImage(mito)
    win.show()
    sys.exit(app.exec_())
