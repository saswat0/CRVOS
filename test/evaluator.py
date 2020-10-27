import os
import time
import torch
import utils


class VOSEvaluator(object):
    def __init__(self, dataset, device='cuda', save=False):
        self._dataset = dataset
        self._device = device
        self._save = save
        self._imsavehlp = utils.ImageSaveHelper()
        if dataset.__class__.__name__ == 'DAVIS17V2':
            self._sdm = utils.ReadSaveDAVISChallengeLabels()

