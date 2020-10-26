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

class ReadSaveDAVISChallengeLabels(ReadSaveImage):
    def __init__(self, bpalette=DAVIS_PALETTE_4BIT, palette=None):
        super(ReadSaveDAVISChallengeLabels, self).__init__()
        self._palette = palette
        self._bpalette = bpalette
        self._width = 0
        self._height = 0

    @property
    def palette(self):
        return self._palette

    def save(self, image, path):
        self.check_path(path)

        if self._palette is None:
            palette = self._bpalette
        else:
            palette = self._palette

        bitdepth = int(math.log(len(palette))/math.log(2))

        height, width = image.shape
        file = open(path, 'wb')
        writer = png.Writer(width, height, palette=palette, bitdepth=bitdepth)
        writer.write(file, image)

    def read(self, path):
        try:
            reader = png.Reader(path)
            width, height, data, meta = reader.read()
            if self._palette is None:
                self._palette = meta['palette']
            image = numpy.vstack(data)
            self._height, self._width = image.shape
        except png.FormatError:
            image = numpy.zeros((self._height, self._width))
            self.save(image, path)

        return image

