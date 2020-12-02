import time
from watchdog.observers import Observer
from watchdog.events import PatternMatchingEventHandler
import os
import re
from binOutput import write_bin


if __name__ == "__main__":
    patterns = ["*.tif"]
    ignore_patterns = ["*.txt", "*.tiff"]
    ignore_directories = True
    case_sensitive = True
    my_event_handler = PatternMatchingEventHandler(patterns, ignore_patterns,
                                                   ignore_directories,
                                                   case_sensitive)
    if os.environ['COMPUTERNAME'] == 'LEBPC20':
        modelPath = 'E:/Watchdog/SmartMito/model_Dora.h5'
    elif os.environ["COMPUTERNAME"] == 'LEBPC34':
        modelPath = 'C:/Users/stepp/Documents/data_raw/SmartMito/model_Dora.h5'

    global frameNumOld
    frameNumOld = 100


def on_created(event):
    global frameNumOld
    splitStr = re.split(r'img_channel\d+_position\d+_time',
                        os.path.basename(event.src_path))
    splitStr = re.split(r'_z\d+', splitStr[1])
    frameNum = int(splitStr[0])
    if frameNum == frameNumOld:
        return

    if frameNum % 2 and frameNum:
        write_bin(frameNum + 1, 0, os.path.dirname(modelPath))
        print(int((frameNum-1)/2), ' written')
        frameNumOld = frameNum


def on_deleted(event):
    write_bin(0, 0, os.path.dirname(modelPath))
    print(0, ' written')


my_event_handler.on_deleted = on_deleted
my_event_handler.on_created = on_created

path = "//lebnas1.epfl.ch/microsc125/Watchdog/"
go_recursively = True
my_observer = Observer()
my_observer.schedule(my_event_handler, path, recursive=go_recursively)


# Init the model by running it once
my_observer.start()

print('All loaded, running...')
# Keep running and let image update until Strg + C
try:
    while True:
        time.sleep(0.1)
except KeyboardInterrupt:
    my_observer.stop()
    my_observer.join()
