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

    def read_video_part(self, video_part):
        images = video_part['images'].to(self._device)
        given_segannos = [seganno.to(self._device) if seganno is not None else None
                          for seganno in video_part['given_segannos']]
        segannos = video_part['segannos'].to(self._device) if video_part.get('segannos') is not None else None
        fnames = video_part['fnames']
        return images, given_segannos, segannos, fnames

