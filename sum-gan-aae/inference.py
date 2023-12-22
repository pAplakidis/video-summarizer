#!/usr/bin/env python3
import os
import cv2
import numpy as np
import h5py
import torch

from model import *
from util import *

# CHANGE THESE
base_dir = "../data/"
video_idx = 0
model_path = "./models/SUM-GAN-VAE.pth"


class SumGanVaeSummarizer:
  def __init__(self, base_dir, device):
    self.device = device

    self.video_dir = base_dir + "SumMe/videos/"
    self.video_list = [video for video in os.listdir(self.video_dir) if video.endswith(".mp4")]
    self.preprocessed_file = base_dir + "eccv16_dataset_summe_google_pool5.h5"

  def get_data(self,index):
    with h5py.File("../data/eccv16_dataset_summe_google_pool5.h5", 'r') as hdf:
      self.video_name = str(np.array(hdf[f"video_{index+1}/video_name"]))[1:].replace("\'", "")

      return{
        "features": torch.Tensor(np.array(hdf[f"video_{index+1}/features"])),
        "gtsummary": torch.Tensor(np.array(hdf[f"video_{index+1}/gtsummary"])),
        "video_name": self.video_name
      }

  def build_model(self):
    self.linear_compress = nn.Linear(INPUT_SIZE, HIDDEN_SIZE).to(self.device)
    self.summarizer = Summarizer(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(self.device)
    self.discriminator = Discriminator(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(self.device)
    self.model = nn.ModuleList([
      self.linear_compress, self.summarizer, self.discriminator
    ])
    self.model.eval()

  def load_model(self, path):
    self.model.load_state_dict(torch.load(path))
    print("[+] Loaded model from:", path)

  def extract_summary(self, img_feats):
    img_feats = self.linear_compress(img_feats.to(device).detach()).unsqueeze(1)
    scores = self.summarizer.s_lstm(img_feats).squeeze(1).detach().cpu().numpy()
    pred_keyframes = np.round(scores)  # TODO: use a threshold
    return pred_keyframes

  def show_summaries(self, pred_keyframes, gt_keyframes, video_name):
    video_path = self.video_dir + video_name + ".mp4"
    print(video_path)
    pred_frames, gt_frames = [], []
    
    # load frames into memory
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
      print(f"Processing frame {idx+1}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
      ret, frame = cap.read()
      if not ret:
        break

      # FIXME: video is frozen
      # frame = cv2.resize(frame, (W,H))
      try:
        if pred_keyframes[idx] == 1:
          # pred_frames.append(frame)
          cv2.imshow("Model Summary", frame)
        if gt_keyframes[idx] == 1:
          # gt_frames.append(frame)
          cv2.imshow("Ground-Truth Summary", frame)
        cv2.waitKey(1)
      except IndexError:
        pass
      idx += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device: ", device)

  summarizer = SumGanVaeSummarizer(base_dir, device)

  data = summarizer.get_data(video_idx)
  print(data)

  img_feats = data["features"]
  gt_summary = data["gtsummary"]
  video_name = data["video_name"]

  summarizer.build_model()
  summarizer.load_model(model_path)
  print(summarizer.model)

  pred_keyframes = summarizer.extract_summary(img_feats)
  summarizer.show_summaries(pred_keyframes, gt_summary, video_name)
