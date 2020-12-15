"""
Module that implements an interface for viewing and analysing adaptive temporal sampling
data that was generated using the iSIM.
"""

import sys

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtGui
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QApplication, QGridLayout, QWidget

from nnIO import loadElapsedTime, loadiSIMmetadata, loadNNData

# Adjust for different screen sizes
QApplication.setAttribute(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)



class RectItem(pg.GraphicsObject):
    """ Rectangle that can be added to a pg.ViewBox """
    def __init__(self, rect, color='#FFFFFF', parent=None):
        super().__init__(parent)
        self._rect = rect
        self.color = color
        self.picture = QtGui.QPicture()
        self.generatePicture()

    @property
    def rect(self):
        """ return the original rectangle object given on initialization """
        return self._rect

    def generatePicture(self):
        """ generate the Picture of te Rectangle using a QPainter """
        painter = QtGui.QPainter(self.picture)
        painter.setPen(pg.mkPen(color=self.color))
        painter.setBrush(pg.mkBrush(color=self.color))
        painter.drawRect(self.rect)
        painter.end()

    def paint(self, *painter):
        """ Not sure when this is called.
        Might be called by the ViewBox the rectangle is added to """
        painter[0].drawPicture(0, 0, self.picture)

    def boundingRect(self):
        """ return the QRectF bounding rectangle of the specified rectangle """
        return QtCore.QRectF(self.picture.boundingRect())


class KeyPressWindow(pg.PlotWidget):
    """ pg.PlotWidget that catches a key press on the keyboard. Used in SatsGUI to advance the
    GUI for presentation purposes """
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)


class SatsGUI(QWidget):
    """ Interface originally used to present the temporal component of adaptive temporal
    sampling data generated using NetworkWatchdog on Mito/Drp on the iSIM. Uses the
    KeyPressWindow to plot sequentially on a button press to allow for a 'presentation' style.
    Now mostly used in the NNGui to display the temporal information of the data taken. """

    frameChanged = pyqtSignal([], [int])

    def __init__(self):
        QWidget.__init__(self)

        self.folder = (
             'C:/Users/stepp/Documents/data_raw/SmartMito/' +
             'microM_test/cell_Int5s_30pc_488_50pc_561_5')
        # Windows for plotting
        self.plot = KeyPressWindow()
        self.plotItem1 = self.plot.plotItem
        self.frames = self.plotItem1.plot([])

        # Add second y axis
        self.plotItem2 = pg.ViewBox()
        self.plotItem1.showAxis('right')
        self.plotItem1.scene().addItem(self.plotItem2)
        self.plotItem1.getAxis('right').linkToView(self.plotItem2)
        self.plotItem2.setXLink(self.plotItem1)

        self.plotItem1.vb.sigResized.connect(self.updateViews)
        self.plot.sigKeyPress.connect(self.incrementView)

        self.nnPlotItem = pg.PlotCurveItem([], pen='w')
        self.nnframeScatter = pg.ScatterPlotItem([], symbol='o', pen=None)
        self.plotItem2.setZValue(100)
        self.plotItem2.addItem(self.nnPlotItem)
        self.plotItem2.addItem(self.nnframeScatter)

        self.frameratePlot = self.plot.plot([])
        pen = pg.mkPen(color='#FF0000', style=Qt.DashLine)
        self.thrLine1 = pg.InfiniteLine(pos=90, angle=0, pen=pen)
        self.thrLine2 = pg.InfiniteLine(pos=70, angle=0, pen=pen)
        self.thrLine1.hide()
        self.thrLine2.hide()
        self.plot.addItem(self.thrLine1)
        self.plot.addItem(self.thrLine2)

        self.scatter = pg.ScatterPlotItem(brush='#505050', size=8, pen=pg.mkPen(color='#303030'))
        self.nnline = pg.PlotCurveItem([], pen=pg.mkPen(color='#303030'))
        self.plot.addItem(self.scatter)
        self.plot.addItem(self.nnline)

        # adapt for presentation
        labelStyle = {'color': '#AAAAAA', 'font-size': '20pt'}
        self.plot.setLabel('right', 'Frame', **labelStyle)
        self.plot.setLabel('bottom', 'Time [s]', **labelStyle)
        self.plot.setLabel('left', 'NN output', **labelStyle)
        font = QtGui.QFont()
        font.setPixelSize(20)
        self.plot.getAxis("bottom").setStyle(tickFont=font, tickTextHeight=500)
        self.plot.getAxis("left").setStyle(tickFont=font)
        self.plot.getAxis("right").setStyle(tickFont=font)

        # Place these windows into the GUI
        grid = QGridLayout(self)
        grid.addWidget(self.plot, 0, 0)
        # grid.addWidget(self.Plot2, 1, 0)

        self.elapsed = None
        self.delay = None
        self.nnData = None
        self.nnframes = None
        self.nntimes = None
        self.inc = -1
        self.rects = []

    def loadData(self, folder, progress=None, app=None):
        """ load timing data using the methods in the nnIO module """
        self.elapsed = loadElapsedTime(folder, progress, app)
        self.elapsed.sort()
        self.elapsed = np.array(self.elapsed[0::2])/1000
        self.delay = loadiSIMmetadata(folder)
        self.nnData = loadNNData(folder)
        print(self.elapsed)
        print(self.delay)

    def updatePlot(self):
        """ update the plot when the 'A' key is pressed and advance plot. This is skipped
        over in NNGui. """
        if self.inc == 0:
            self.frames.setData(self.elapsed, np.zeros(len(self.elapsed)), symbol='o', pen=None)
            self.plotItem1.getAxis('left').hide()

        elif self.inc == 1:
            self.nnPlotItem.setData(x=self.elapsed, y=np.arange(0, len(self.elapsed)))

        elif self.inc == 2:
            self.delay = np.append(np.ones(5), self.delay)
            rectData = self.delay[5:len(self.elapsed)]
            # see where the delay value changes
            changes = np.where(np.roll(rectData, 1) != rectData)[0]
            # map this frame data to the elapsed time data
            changes = self.elapsed[changes+1]
            changes = np.insert(changes, 0, np.min(self.elapsed))
            changes = np.append(changes, np.max(self.elapsed))
            for i in range(1, len(changes)):
                color = '#202020' if i % 2 else '#101010'
                rectItem = RectItem(QtCore.QRectF(
                    changes[i-1], 0, changes[i]-changes[i-1], np.max(self.nnData[:, 1])), color)

                self.rects.append(rectItem)
                self.plot.addItem(rectItem)
                rectItem.setZValue(-100)

        elif self.inc == 3:
            self.nnPlotItem.hide()
            self.frames.hide()
            self.nnframeScatter.setData(self.elapsed, np.zeros(len(self.elapsed)))
            # self.plotItem2.setZValue(-1)
            self.plotItem1.getAxis('right').hide()
            self.plotItem1.getAxis('left').show()
            self.updateViews()

        elif self.inc == 4:
            # self.plot.plot(self.elapsed, self.delay[0:len(self.elapsed)]*150)
            self.nnframes = ((self.nnData[:, 0] - 1) / 2).astype(np.uint16)
            self.nntimes = self.elapsed[self.nnframes]
            self.nnline.setData(self.nntimes, self.nnData[:, 1])
            self.scatter.setData(self.nntimes, self.nnData[:, 1])

        elif self.inc == 5:
            self.thrLine1.show()
            self.thrLine2.show()

        elif self.inc == 10:
            self.deleteRects()

        self.plot.update()

    def updateViews(self):
        """ adjust the two views to keep in sync when one of them is moved """
        # view has resized; update auxiliary views to match
        self.plotItem2.setGeometry(self.plotItem1.vb.sceneBoundingRect())

        # need to re-update linked axes since this was called
        # incorrectly while views had different shapes.
        # (probably this should be handled in ViewBox.resizeEvent)
        self.plotItem2.linkedViewChanged(self.plotItem1.vb, self.plotItem2.XAxis)

    def incrementView(self, event):
        """ React to a press of 'A' and advance the presentation """
        if event.key() == 65:
            self.inc = self.inc + 1
            print(self.inc)
        self.updatePlot()

    def deleteRects(self):
        """ delete all rectangles in the scene. This is used when calling from NNGui to adjust for
        ATS vs normal stacks. """
        for rect in self.rects:
            self.plot.removeItem(rect)
            del rect
        self.rects = []


def main():
    "Presentation mode of the GUI that can be advanced clicked the A button on the keyboard."
    app = QApplication(sys.argv)
    gui = SatsGUI()
    gui.loadData('W:/Watchdog/microM_test/201208_cell_Int0s_30pc_488_50pc_561_band_5', None)
    gui.show()

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
