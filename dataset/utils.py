import os
import sys
import numpy as np
import torch
import torch.utils.data

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

class LabelToLongTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            label = torch.from_numpy(pic).long()
        elif pic.mode == '1':
            label = torch.from_numpy(np.array(pic, np.uint8, copy=False)).long().view(1, pic.size[1], pic.size[0])
        else:
            label = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            if pic.mode == 'LA':
                label = label.view(pic.size[1], pic.size[0], 2)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()[0]
                label = label.view(1, label.size(0), label.size(1))
            else:
                label = label.view(pic.size[1], pic.size[0], -1)
                label = label.transpose(0, 1).transpose(0, 2).contiguous().long()
        return label
