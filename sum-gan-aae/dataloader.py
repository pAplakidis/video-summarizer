#!/usr/bin/env python3
import os
import h5py
import cv2
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader

class VideoDataset(Dataset):
  def __init__(self, base_dir):
    self.base_dir = base_dir
    self.video_dir = base_dir + "SumMe/videos/"
    self.video_list = [video for video in os.listdir(self.video_dir) if video.endswith(".mp4")]
    self.preprocessed_file = base_dir + "eccv16_dataset_summe_google_pool5.h5"
    self.hdf = h5py.File(self.preprocessed_file, 'r')

  def __len__(self):
    return len(self.video_list)

  def __getitem__(self, index):
    return {
      "features": torch.Tensor(np.array(self.hdf[f"video_{index+1}/features"])),
      "gtsummary": torch.Tensor(np.array(self.hdf[f"video_{index+1}/gtsummary"])),
      "video_name": str(np.array(self.hdf[f"video_{index+1}/video_name"]))[1:].replace("\'", "")
      }


if __name__ == "__main__":
  print("Hello")
  data_dir = "../data/"
  dataset = VideoDataset(data_dir)

  idx = 0
  feats, gt, video_name = dataset[idx]["features"],dataset[idx]["gtsummary"], dataset[idx]["video_name"]
  print(feats)
  print(gt)
  print(video_name)

  video_path = dataset.video_dir + video_name + ".mp4"
  print(video_path)

  cap = cv2.VideoCapture(video_path)
  while True:
      # Capture frame-by-frame
      ret, frame = cap.read()
      # if frame is read correctly ret is True
      if not ret:
          print("Can't receive frame (stream end?). Exiting ...")
          break
      # Our operations on the frame come here
      # Display the resulting frame
      cv2.imshow('frame', frame)
      if cv2.waitKey(1) == ord('q'):
          break

  # print(dataset[0].shape)
  # train_loader = DataLoader(train_set, batch_size=BS, shuffle=True, num_workers=N_WORKERS, collate_fn=custom_collate, pin_memory=True)
  dataloader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True)
