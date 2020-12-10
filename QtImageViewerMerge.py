from PyQt5.QtCore import Qt, pyqtSignal, QT_VERSION_STR, QTimer, QRectF, QPointF, pyqtSlot
from PyQt5.QtWidgets import QWidget, QSlider, QPushButton, QLabel,\
    QGridLayout, QFileDialog, QProgressBar, QGroupBox, QApplication, QGraphicsScene, QGraphicsView,\
    QHBoxLayout, QMainWindow, QFrame, QGraphicsGridLayout
from PyQt5.QtGui import QColor, QBrush, QPen, QMovie, qRgb, QPixmap, QImage, QPicture, QPainter
import PyQt5.QtGui as QtGui
import PyQt5.QtCore as QtCore
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


class QtImageViewerMerge(QMainWindow):  # GraphicsWindow):

    frameChanged = pyqtSignal([], [int])
    resizedEmitter = pyqtSignal([], [int])

    def __init__(self):
        QMainWindow.__init__(self)
        self.widget = QWidget()
        self.setCentralWidget(self.widget)
        self.glw = pg.GraphicsLayoutWidget()

        layout_box = QGridLayout(self.widget)
        layout_box.setContentsMargins(0, 0, 0, 0)
        layout_box.addWidget(self.glw)

        # Image frame viewer.
        self.vb = self.glw.addViewBox()
        self.vb.setAspectLocked()
        self.vb.invertY()
        self.cross = CrossItem(QRectF(0, 0, 1, 1))
        self.vb.addItem(self.cross)

        # add a Menu on top that can be toggled with a button
        self.toggleMenu = QFrame(self.widget)
        self.gridMenu = QGridLayout(self.toggleMenu)
        self.menuButton = QPushButton("...", self.widget)
        self.menuButton.setFixedSize(20, 20)
        self.gridMenu.addWidget(self.menuButton, 0, 0, Qt.AlignTop)

        self.Z = 0
        self.images = []
        self.imageItems = []
        self.saturationSliders = []
        self.opacitySliders = []
        self.qFrames = []
        self.numChannels = 0
        self.toggle = 0
        self.menuButton.clicked.connect(self.showMenu)
        # layout_box = QHBoxLayout(self.widget)
        # self.vb.sigRangeChanged.connect(self.resizedEvent)

    def setImage(self, img, pos=0):
        self.imageItems[pos]['ImageItem'].setImage(img)
        fullRange = self.imageItems[pos]['ImageItem'].quickMinMax()
        minRange = fullRange[0]
        maxRange = fullRange[1]
        self.saturationSliders[pos].vb.setYRange(minRange, maxRange)
        self.saturationSliders[pos].regions[0].setRegion((minRange, maxRange))

    def addImage(self, img=None):
        imgItem = pg.ImageItem()

        self.vb.addItem(imgItem)
        imgItem.setZValue(self.Z)
        self.Z = self.Z + 1
        imgStruct = {'ImageItem': imgItem, 'cm_name': None, 'transparency': False,
                     'opacity': (0.3, 0.85), 'saturation': 255}
        self.imageItems.append(imgStruct)

        imgItem.setOpts(axisOrder='row-major')
        if self.numChannels > 0:
            imgItem.setCompositionMode(QPainter.CompositionMode_Screen)
        self.addMenu()
        if img is not None:
            self.setImage(img, self.numChannels)
        self.numChannels = self.numChannels + 1
        return imgItem

    def setLUT(self, img, name='hot', transparency=False, opacity=(0.3, 0.85)):
        for i in range(0, len(self.imageItems)):
            if self.imageItems[i]['ImageItem'] == img:
                index = i
        print('lookup ', index, name)
        self.saturationSliders[index].gradient.loadPreset(name)

        self.updateImage(index)
        # self.imageItems[index]['cm_name'] = name
        # self.imageItems[index]['transparency'] = transparency
        # self.imageItems[index]['opacity'] = opacity
        # colormap = getImageItemcolormap(name, transparency, opacity)
        # img.setLookupTable(colormap)

    def resizedEvent(self):
        self.resizedEmitter.emit()

    def addMenu(self):
        print(self.numChannels)
        channel = self.numChannels
        thisMenu = QFrame(self.toggleMenu)
        self.qFrames.append(thisMenu)
        thisMenuLayout = QGridLayout(thisMenu)
        self.saturationSliders.append(LUTItemSimple(thisMenu))
        print('Channel    ', channel)
        self.saturationSliders[self.numChannels].gradientChanged.connect(
            lambda: self.updateImage(channel))
        self.saturationSliders[self.numChannels].levelChanged.connect(
            lambda: self.adjustLevel(channel))
        if channel > 0:
            self.saturationSliders[self.numChannels].regions[0].setRegion((100, 255))

        # self.saturationSliders[self.numChannels].setValue(255)
        thisMenuLayout.addWidget(self.saturationSliders[self.numChannels], 0, 0)

        if channel > 0:
            self.opacitySliders.append(QSlider(Qt.Vertical, self.widget))
            self.opacitySliders[self.numChannels].setRange(0, 255)
            self.opacitySliders[self.numChannels].setMaximumHeight(140)
            self.opacitySliders[self.numChannels].setMinimumWidth(20)

            self.opacitySliders[self.numChannels].setValue(int(0.85*255))
            thisMenuLayout.addWidget(self.opacitySliders[self.numChannels], 0, 1)
            self.opacitySliders[self.numChannels].valueChanged.connect(
                lambda value: self.adjustOpacity(value, channel))
        else:
            self.opacitySliders.append(0)

        self.gridMenu.addWidget(thisMenu, 0, self.numChannels+1)
        thisMenu.hide()

    def showMenu(self):
        if self.toggle == 0:
            # self.saturationSliders[0].setMinimumHeight(300)
            for frame in self.qFrames:
                frame.setVisible(True)

            self.toggle = 1
            self.toggleMenu.setMinimumHeight(200)
            self.toggleMenu.setMinimumWidth(130*self.numChannels+1)
        else:
            for frame in self.qFrames:
                frame.setVisible(False)
            self.toggle = 0
            self.toggleMenu.setFixedSize(40, 40)

    @pyqtSlot(int)
    def adjustOpacity(self, value, index):
        opacity = self.opacitySliders[index].value()/255
        self.imageItems[index]['ImageItem'].setOpacity(opacity)

    @pyqtSlot(int)
    def adjustLevel(self, index):
        self.imageItems[index]['ImageItem'].setLevels(
            self.saturationSliders[index].regions[0].getRegion())

    @pyqtSlot(int)
    def updateImage(self, index):
        print('INDEX  ', index)
        self.imageItems[index]['ImageItem'].setLookupTable(
            self.saturationSliders[index].getLookupTable())
        print('LUT set')


class GradientEditorWidget(pg.GraphicsView):

    def __init__(self, parent=None,  *args, **kargs):
        background = kargs.pop('background', 'default')
        pg.GraphicsView.__init__(self, parent, useOpenGL=False, background=background)
        self.item = pg.GradientEditorItem(*args, **kargs)
        self.setCentralItem(self.item)
        self.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.setMaximumWidth(15)
        self.setMaximumHeight(150)

    def sizeHint(self):
        return QtCore.QSize(115, 200)

    def __getattr__(self, attr):
        return getattr(self.item, attr)


class LUTItemSimple(QWidget):

    gradientChanged = pyqtSignal([], [int])
    levelChanged = pyqtSignal([], [int])

    def __init__(self, parent=None,  *args, **kargs):
        QWidget.__init__(self, parent)
        frame = QFrame(self)
        grid = QGridLayout(frame)
        self.setMinimumWidth(100)
        self.setMinimumHeight(150)

        self.gradient = GradientEditorWidget(frame)
        self.gradient.setOrientation('right')
        grid.addWidget(self.gradient, 0, 0)
        self.gradient.sigGradientChangeFinished.connect(self.gradientChange)
        self.gradient.loadPreset('inferno')

        self.glw = pg.GraphicsLayoutWidget(parent=frame)

        grid.addWidget(self.glw, 0, 1)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(0)

        self.vb = self.glw.addViewBox()
        self.vb.setZValue(100)
        self.vb.setMaximumHeight(150)
        self.vb.setMaximumWidth(30)
        self.vb.setFlag(self.gradient.ItemStacksBehindParent)
        self.vb.sigRangeChanged.connect(self.update)
        self.vb.setMouseEnabled(x=False, y=True)
        self.vb.setYRange(0, 255)
        self.regions = [
            pg.LinearRegionItem([0, 255], 'horizontal', swapMode='push')]
        for region in self.regions:
            region.setZValue(1000)
            self.vb.addItem(region)
            region.lines[0].addMarker('<|', 0.5)
            region.lines[1].addMarker('|>', 0.5)
            region.sigRegionChanged.connect(self.regionChanging)
            region.sigRegionChangeFinished.connect(self.regionChanged)
        # self.regions[0].setSpan(0.8, 0.8)

        self.levelMode = 'mono'

    def gradientChange(self):
        self.hideTicks()
        self.gradientChanged.emit()

    def hideTicks(self):
        ticks = self.gradient.listTicks()
        for tick in ticks:
            tick[0].hide()

    def getLookupTable(self, img=None, n=None, alpha=None):
        """Return a lookup table from the color gradient defined by this
        HistogramLUTItem.
        """
        if self.levelMode != 'mono':
            return None
        if n is None:
            if img is None:
                n = 256
            elif img.dtype == np.uint8:
                n = 256
            else:
                n = 512
        # if self.lut is None:
        self.lut = self.gradient.getLookupTable(n, alpha=alpha)
        return self.lut

    def regionChanging(self, value):
        self.levelChanged.emit()

    def regionChanged(self):
        self.levelChanged.emit()

# pg.ViewBox.viewRange.__get__


if __name__ == '__main__':

    fname = ('C:/Users/stepp/Documents/data_raw/SmartMito/__short.tif')
    image_mitoOrig = io.imread(fname)
    image_drpOrig = image_mitoOrig[1]
    image_mitoOrig = image_mitoOrig[0]
    mito = image_mitoOrig
    drp = image_drpOrig

    app = QApplication(sys.argv)

    # win = pg.GraphicsScene()
    # vb = win.addViewBox()
# make sure this image is on top
    # img2.setOpacity(0.5)
    # pyqtgraph.ColorMap()
    # viewer.setColorMap(getPyQtcolormap())

    # lutitem = LUTItemSimple()
    # lutitem.show()

    win = pg.GraphicsWindow()

    # Show viewer and run application.
    viewer = QtImageViewerMerge()
    viewer2 = QtImageViewerMerge()

    grid = QGridLayout(win)
    grid.addWidget(viewer, 0, 0)
    grid.addWidget(viewer2, 0, 1)

    img_drp = viewer.addImage(drp)
    img_mito = viewer.addImage(mito)

    viewer.setLUT(img_drp, 'viridis')
    viewer.setLUT(img_mito, 'inferno')
    # viewer.setImage(drp, 0)
    # viewer.setImage(mito, 1)

    viewer2.addImage(mito)
    win.show()

    sys.exit(app.exec_())
