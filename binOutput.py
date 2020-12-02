# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 15:57:53 2020

@author: stepp

Application to write a value to a file in a remote location as fast as possible

"""

import time
import numpy
from datetime import datetime
import os.path as ospath

if __name__ == "__main__":
    path = "//lebnas1.epfl.ch/microsc125/Watchdog/"
    filename = 'binary_output.dat'
    fullFileDir = path + filename

    x = False
    x = bool(x)
    i = 0

    try:
        while True:
            t1 = time.perf_counter()
            f = open(fullFileDir, 'wb')
            f.write(bytearray(x))
            f.close()
            t2 = time.perf_counter()
            dt = t2-t1
            # print(dt)

            # This changes the value every some seconds
            if (time.perf_counter() % 3) < 1.5:
                x = False
                print('False False False')
            else:
                x = True
                print(x)
            t3 = time.perf_counter()
            time.sleep(0.5-(t3-t1))
    except KeyboardInterrupt:
        print('done')


def write_bin(x, print_time=0, path="//lebnas1.epfl.ch/microsc125/Watchdog/"):
    # Set the file location for the watched file here
    # path = "//lebnas1.epfl.ch/microsc125/Watchdog/"
    filename = 'binary_output.dat'
    fullFileDir = ospath.join(path, filename)
    f = open(fullFileDir, 'wb')
    f.write(bytearray(x))
    f.close()

    if print_time:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S.%f")
        print(f"{x} written at {current_time}")
