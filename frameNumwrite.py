""" Implements a class that is used to work together with NetworkWatchdog to skip files if it is
to slow to process all the images. """

import os
import re
import time

from watchdog.events import PatternMatchingEventHandler
from watchdog.observers import Observer

from BinOutput import writeBin


class FrameNumwrite():
    """ Class that writes the latest framenumber when microManager writes a new image in order
    to enable NetworkWatchdog to skip the file if it is working on an older one. """

    def __init__(self):
        patterns = ["*.tif"]
        ignorePatterns = ["*.txt", "*.tiff"]
        ignoreDirectories = True
        caseSensitive = True
        myEventHandler = PatternMatchingEventHandler(patterns, ignorePatterns,
                                                     ignoreDirectories,
                                                     caseSensitive)

        if os.environ['COMPUTERNAME'] == 'LEBPC20':
            self.modelPath = 'E:/Watchdog/SmartMito/model_Dora.h5'
        elif os.environ["COMPUTERNAME"] == 'LEBPC34':
            self.modelPath = (
                'C:/Users/stepp/Documents/data_raw/SmartMito/model_Dora.h5')

        self.frameNumOld = 100

        myEventHandler.on_deleted = self.onDeleted
        myEventHandler.on_created = self.onCreated

        path = "//lebnas1.epfl.ch/microsc125/Watchdog/"
        goRecursively = True
        self.myObserver = Observer()
        self.myObserver.schedule(myEventHandler, path, recursive=goRecursively)

        # Init the model by running it once
        self.myObserver.start()

    def onCreated(self, event):
        """ If a new file is written by microManager, get the frameNum and write it to a binary file
        that will be read by Network Watchdog to check for the latest file written """
        splitStr = re.split(r'img_channel\d+_position\d+_time',
                            os.path.basename(event.src_path))
        splitStr = re.split(r'_z\d+', splitStr[1])
        frameNum = int(splitStr[0])
        if frameNum == self.frameNumOld:
            return

        if frameNum % 2 and frameNum:
            writeBin(frameNum + 1, 0, os.path.dirname(self.modelPath))
            print(int((frameNum-1)/2), ' written')
            self.frameNumOld = frameNum

    def onDeleted(self):
        """ Signal if a file has been deleted, does not work very well. Used with TestTifSaver """
        writeBin(0, 0, os.path.dirname(self.modelPath))
        print(0, ' written')


if __name__ == "__main__":
    main()


def main():
    """ Main instance of the Module """

    writer = FrameNumwrite()

    print('All loaded, running...')
    # Keep running and let image update until Strg + C
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        writer.myObserver.stop()
        writer.myObserver.join()
