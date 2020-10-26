import threading
import queue
import time
import os
import png
import numpy
import math

BASE_PALETTE_4BIT = [[0,   0,   0],
                     [236,  94, 102],
                     [249, 144,  87],
                     [250, 199,  98],
                     [153, 199, 148],
                     [97, 179, 177],
                     [102, 153, 204],
                     [196, 148, 196],
                     [171, 120, 102],
                     [255, 255, 255],
                     [101, 115, 125],
                     [10,  10,  10],
                     [12,  12,  12],
                     [13,  13,  13],
                     [13,  13,  13],
                     [14,  14,  14]]

DAVIS_PALETTE_4BIT = [[0,   0,   0],
                      [128,   0,   0],
                      [0, 128,   0],
                      [128, 128,   0],
                      [0,   0, 128],
                      [128,   0, 128],
                      [0, 128, 128],
                      [128, 128, 128],
                      [64,   0,   0],
                      [191,   0,   0],
                      [64, 128,   0],
                      [191, 128,   0],
                      [64,   0, 128],
                      [191,   0, 128],
                      [64, 128, 128],
                      [191, 128, 128]]

class ReadSaveImage(object):
    def __init__(self):
        super(ReadSaveImage, self).__init__()

    def check_path(self, fullpath):
        path, filename = os.path.split(fullpath)
        if not os.path.exists(path):
            os.makedirs(path)
