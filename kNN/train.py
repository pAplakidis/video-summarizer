import os
import cv2
import numpy as np
import matplotlib.pythot as plt

import torch
from torch.utils.data import Dataset

from sklearn.neighbors import KNeighborsClassifier

W = 32
H = 32

class Dataset(Dataset):
  def __init__(self, base_dir):
    super(Dataset, self).__init__()

    # directories
    self.base_dir = base_dir
    self.input_frames = [np.zeros((3, W, H)) for _ in range(2)] # 2 consecutive frames for GRU

    print("Data from:")
    # for dir in sorted(os.listdir(base_dir)):
    #   prefix = self.base_dir+dir+"/"
    #   print(prefix)
    #   self.video_paths.append(prefix+"video.mp4")
    #   self.framepath_paths.append(prefix+"frame_paths.npy")
    #   self.desires_paths.append(prefix+"desires.npy")
    #   if self.combo:
    #     self.crossroads_paths.append(prefix+"crossroads.npy")

    # load and index actual data
    # self.caps = [cv2.VideoCapture(str(video_path)) for video_path in self.video_paths]
    # self.images = [[capid, framenum] for capid, cap in enumerate(self.caps) for framenum in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))-LOOKAHEAD+1)]
    # self.frame_paths = [np.load(fp) for fp in self.framepath_paths]
    # self.desires = [np.load(desires) for desires in self.desires_paths]
    # for i in range(len(self.desires)):
    #   self.desires[i] = one_hot_encode(self.desires[i])
    # if self.combo:
    #   self.crossroads = [np.load(crds) for crds in self.crossroads_paths]

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    capid, framenum = self.images[idx]
    path = self.frame_paths[capid][framenum]
    if np.isnan(path).any():
      path = np.zeros_like(path)

    return {"frame": self.images[idx]}
