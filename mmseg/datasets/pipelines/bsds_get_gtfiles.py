import numpy as np
from PIL import Image
import h5py
import os.path as osp
import cv2

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



def get_rind_gtfiles(filename):
    #filename = osp.join(results['img_prefix'],results['img_info']['filename'])
    #img = Image.open(filename).convert('RGB')
    #filename = osp.join(results['seg_prefix'], results['ann_info']['seg_map'])
    h = h5py.File(filename, 'r')
    edge = np.squeeze(h['label'][...])
    label = edge.astype(np.float32)
    label = label.transpose(1,2,0)
    #cv2.imwrite('depth.png', label[:, :, 1] * 255)
    #label = torch.from_numpy(label).float()
    return label
