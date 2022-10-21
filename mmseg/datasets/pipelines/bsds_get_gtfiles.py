import numpy as np
from PIL import Image



def get_bsds_gtfiles(filenames):
    edge = Image.open(filenames, 'r')
    label = np.array(edge, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label /= 255.
    label[label >= 0.3] = 1
    #label = torch.from_numpy(label).float()
    return label


def get_bsds_gtfiles_bythr(filenames,thr):
    edge = Image.open(filenames, 'r')
    label = np.array(edge, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:, :, 0]
    label /= 255.
    label[label >= thr] = 1
    #label = torch.from_numpy(label).float()
    return label
