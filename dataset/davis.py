import os
import json
import glob
import random
from PIL import Image
from collections import OrderedDict

import torch
import numpy as np
from dataset import utils

class DAVIS17V2(torch.utils.data.Dataset):
    def __init__(self, root_path, version, image_set, image_read=None, anno_read=None,
                 samplelen=4, obj_selection=get_sample_all(), min_num_obj=1, start_frame='random'):
        self._min_num_objects = min_num_obj
        self._root_path = root_path
        self._version = version
        self._image_set = image_set
        self._image_read = image_read
        self._anno_read = anno_read
        self._seqlen = samplelen
        self._obj_selection = obj_selection
        self._start_frame = start_frame
        self._init_data()
        
    def _init_data(self):
        framework_path = os.path.join(os.path.dirname(__file__), '..')
        cache_path = os.path.join(framework_path, 'cache', 'davis17_v2_visible_objects_100px_threshold.json')

        if os.path.exists(cache_path):
            with open(cache_path, 'r') as f:
                self._visible_objects = json.load(f)
                self._visible_objects = {seqname: OrderedDict((int(idx), objlst) for idx, objlst in val.items())
                                         for seqname, val in self._visible_objects.items()}
        else:
            seqnames = os.listdir(os.path.join(self._root_path, 'JPEGImages', '480p'))
            
            self._visible_objects = {}
            for seqname in seqnames:
                anno_paths = sorted(glob.glob(self._full_anno_path(seqname, '*.png')))
                self._visible_objects[seqname] = OrderedDict(
                    (self._frame_name_to_idx(os.path.basename(path)),
                     get_anno_ids(path, dataset_utils.LabelToLongTensor(), 100))
                    for path in anno_paths)

            if not os.path.exists(os.path.dirname(cache_path)):
                os.makedirs(os.path.dirname(cache_path))
            with open(cache_path, 'w') as f:
                json.dump(self._visible_objects, f)
            print("Datafile {} was not found, creating it with {} sequences.".format(cache_path, len(self._visible_objects)))

        with open(os.path.join(self._root_path, 'ImageSets', self._version, self._image_set + '.txt'), 'r') as f:
            self._all_seqs = f.read().splitlines()

        self._nonempty_frame_ids = {seq: [frame_idx for frame_idx, obj_ids in lst.items() if len(obj_ids) >=
                                          self._min_num_objects]
                                    for seq, lst in self._visible_objects.items()}

        self._viable_seqs = [seq for seq in self._all_seqs if
                             len(self._nonempty_frame_ids[seq]) > 0
                             and len(self.get_image_frame_ids(seq)[min(self._nonempty_frame_ids[seq]):
                                                                   max(self._visible_objects[seq].keys()) + 1])
                             >= self._seqlen]
