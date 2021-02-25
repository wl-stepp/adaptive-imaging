import sys

import h5py
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtGui
from PyQt5.QtCore import QRectF, Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (QApplication, QGridLayout, QGroupBox,
                             QInputDialog, QPushButton, QSlider, QWidget)
from skimage import io

from SmartMicro.QtImageViewerMerge import QtImageViewerMerge


class QtImageViewerSeries(QWidget):
    """ Fork of ImageViewerMerge that can be used to see a series and also have a slider to see all
    """

    def __init__(self):
        QWidget.__init__(self)
        self.viewer = QtImageViewerMerge()
        self.grid = QGridLayout(self)
        # make a new slider that is much cooler
        self.slider = QNiceSlider()
        self.slider.sliderPressed.connect(self.startTimer)
        self.slider.sliderReleased.connect(self.stopTimer)
        self.slider.sliderChanged.connect(self.onTimer)

        self.grid.addWidget(self.viewer, 0, 0)
        self.grid.addWidget(self.slider, 1, 0)

        #get the timer ready for the slider
        self.timer = QTimer()
        self.timer.timeout.connect(self.onTimer)
        self.timer.setInterval(20)

        self.series = None

    def newSlider(self, ev):
        print(self.region.getRegion())
        print(ev)

    def onTimer(self, i=None):
        if i is None:
            i = int(self.slider.position)
        else:
            i = int(np.round(i))
        self.viewer.setImage(self.series[i])

    def loadSeries(self, filePath):
        if filePath[-3:] == '.h5':
            hf = h5py.File(filePath, 'r')
            if len(hf.keys()) > 1:
                item, ok = QInputDialog.getItem(
                    self, "select Series", "series", hf.keys(), 0, False)
            else:
                item = hf.keys()[0]
            self.series = hf.get(item)
            self.series = np.array(self.series).astype('float')

        else:
            self.series = io.imread(filePath)
        self.viewer.addImage(self.series[0])
        self.viewer.rangeChanged()
        self.viewer.resetRanges()
        self.slider.setSliderRange([-1, self.series.shape[0]-1])
        self.viewer.resetZoom()

    def startTimer(self):
        """ start Timer when slider is pressed """
        self.timer.start()

    def stopTimer(self):
        """ stop timer when slider is released"""
        self.timer.stop()

    def nextFrame(self):
        """ display next frame in all viewBoxes when '>' button is pressed """
        i = self.frameSlider.value()
        self.frameSlider.setValue(i + 1)
        self.onTimer()

    def prevFrame(self):
        """ display previous frame in all viewBoxes when '<' button is pressed """
        i = self.frameSlider.value()
        self.frameSlider.setValue(i - 1)
        self.onTimer()


class QNiceSlider(pg.GraphicsLayoutWidget):
    """ This is a slider that looks much nicer than QSlider and also allows for mouse wheel
    scrolling. It has the same Signals as QSlider to allow for easy replacement"""
    sliderChanged = pyqtSignal(float)
    sliderPressed = pyqtSignal()
    sliderReleased = pyqtSignal()

    def __init__(self):
        pg.GraphicsLayoutWidget.__init__(self)
        self.setMaximumHeight(100)
        self.setMinimumHeight(100)
        self.setSizePolicy(QtGui.QSizePolicy.Preferred, QtGui.QSizePolicy.Expanding)
        self.viewB = self.addViewBox()
        self.viewB.setMouseEnabled(y=False, x=False)

        self.buttonWidth = 1
        self.position = 0
        self.span = [self.position-self.buttonWidth/2, self.position+self.buttonWidth/2]
        self.orientation = 'vertical'
        self.button = self.SliderButton(self.span, self.orientation)
        self.sliderBackground = self.SliderBackground()
        self.sliderBackground.sliderClicked.connect(self.sliderBgClicked)
        self.button.buttonChanged.connect(self.buttonChange)
        self.button.buttonPressed.connect(self.sliderPressed)
        self.button.buttonReleased.connect(self.sliderReleased)
        self.viewB.addItem(self.button)
        self.viewB.addItem(self.sliderBackground)
        self.viewB.setYRange(0, 1)
        self.sliderRange = [-1, 100]
        self.setSliderRange(self.sliderRange)

    def setSliderRange(self, sliderRange=[-1, 100]):
        self.sliderRange = sliderRange
        self.viewB.setXRange(sliderRange[0], sliderRange[1])
        self.button.setsliderRange(sliderRange)
        self.button.width = (sliderRange[1]-sliderRange[0])/100
        self.sliderBackground.setSliderBackground(sliderRange)

    def wheelEvent(self, ev):
        wheelCorrect = 120
        ev.accept()
        delta = ev.angleDelta().y()
        if self.sliderRange[1] > self.position + delta/wheelCorrect > self.sliderRange[0]:
            self.position = self.position + delta/wheelCorrect
        elif self.position + delta/wheelCorrect > self.sliderRange[1]:
            self.position = self.sliderRange[1]
        elif self.position + delta/wheelCorrect < self.sliderRange[0]:
            self.position = self.sliderRange[0]

        self.button.updatePosition(self.position)
        self.sliderChanged.emit(self.position)

    def buttonChange(self, pos):
        self.position = pos
        self.sliderChanged.emit(self.position)

    def sliderBgClicked(self, pos):
        self.position = pos
        self.sliderChanged.emit(self.position)
        self.button.updatePosition(self.position)


    class SliderBackground(pg.GraphicsObject):
        """ The Background of the slider. """
        sliderClicked = pyqtSignal(float)

        def __init__(self):
            pg.GraphicsObject.__init__(self)
            self.setZValue(500)
            self.currentBrush = QtGui.QBrush(QtGui.QColor(30, 30, 30, 200))
            self.railHeight = 0.6
            self.boundingR = QRectF(-1, (1-self.railHeight)/2, 100, self.railHeight)

        def paint(self, p, *args):
            p.setBrush(self.currentBrush)
            p.setPen(pg.mkPen(None))
            p.drawRect(self.boundingR)

        def setSliderBackground(self, rect):
            self.boundingR.setLeft(rect[0])
            self.boundingR.setRight(rect[1])
            self.prepareGeometryChange()

        def boundingRect(self):
            br = self.boundingR
            br = br.normalized()
            return br

        def mousePressEvent(self, ev):
            self.sliderClicked.emit(ev.pos().x())

    class SliderButton(pg.GraphicsObject):
        """ The button for the slider. """
        buttonChanged = pyqtSignal(float)
        buttonPressed = pyqtSignal()
        buttonReleased = pyqtSignal()

        def __init__(self, span, orientation):
            pg.GraphicsObject.__init__(self)
            self.setZValue(1000)
            self.currentBrush = QtGui.QBrush(QtGui.QColor(255, 255, 255, 200))

            self.span = span
            self.width = span[1]-span[0]
            self.orientation = orientation
            self.sliderRange = [-1, 100]

        def setsliderRange(self, range):
            self.sliderRange = range

        def boundingRect(self, span=None):
            br = QRectF(self.viewRect())
            if span is None:
                rng = self.span
            else:
                rng = span

            if self.orientation == 'vertical':
                br.setLeft(rng[0])
                br.setRight(rng[1])
                length = br.height()
                br.setBottom(0)
                br.setTop(1)
            else:
                br.setTop(rng[0])
                br.setBottom(rng[1])
                length = br.width()
                br.setRight(br.left() + length * self.span[1])
                br.setLeft(br.left() + length * self.span[0])

            br = br.normalized()
            return br

        def paint(self, p, *args):
            p.setBrush(self.currentBrush)
            p.setPen(pg.mkPen(None))
            p.drawRect(self.boundingRect())

        def updatePosition(self, pos):
            width = self.span[1] - self.span[0]
            self.span = [pos-self.width/2, pos+self.width/2]
            self.prepareGeometryChange()
            self.buttonChanged.emit(pos)

        def mouseMoveEvent(self, ev):
            if self.sliderRange[1] > ev.lastPos().x() > self.sliderRange[0]:
                self.updatePosition(ev.lastPos().x())

        def mousePressEvent(self, ev):
            self.buttonPressed.emit()

        def mouseReleaseEvent(self, v):
            self.buttonReleased.emit()


def main():
    """ Method to test the Viewer in a QBoxLayout with 1 and 2 channels"""
    app = QApplication(sys.argv)
    viewer = QtImageViewerSeries()
    fname = ('W:/Watchdog/Model/prep_data3.h5')
    # fname = ('C:/Users/stepp/Documents/02_Raw/SmartMito/__short.tif')
    viewer.loadSeries(fname)
    viewer.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
