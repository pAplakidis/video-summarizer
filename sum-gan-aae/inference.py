#!/usr/bin/env python3
import os
import cv2
import numpy as np
from tqdm import tqdm
import h5py
import torch

from model import *
from util import *

# CHANGE THESE
video_idx = 0
base_dir = "../data/"
model_path = "./models/SUM-GAN-VAE.pth"

disp_W = 1920 // 2
disp_H = 1080 // 2


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
    # print("Predicted Scores:")
    # print(scores)
    # pred_keyframes = np.round(scores)
    # pred_keyframes = (scores >= THRESHOLD).astype(int)
    pred_keyframes = (scores > np.mean(scores)).astype(int)

    return pred_keyframes

  def show_summaries(self, pred_keyframes, gt_keyframes, video_name):
    video_path = self.video_dir + video_name + ".mp4"
    print(video_path)
    reduced_frames, pred_frames, gt_frames = [], [], []
    
    # load (and reduce) frames into memory
    cap = cv2.VideoCapture(video_path)
    idx = 0
    pbar = tqdm(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) + 1)
    while True:
      pbar.set_description(f"Processing frame {idx+1}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
      ret, frame = cap.read()
      if not ret:
        break

      if idx % 15 == 0:
        frame = cv2.resize(frame, (disp_W, disp_H))
        reduced_frames.append(frame)

      idx += 1
      pbar.update(1)
    pbar.close()
    cap.release()

    # predicted intervals => keyframes
    tmp_frames = []
    pred_keyframe_idxs = np.zeros_like(pred_keyframes)
    for idx, frame in enumerate(reduced_frames):
      if pred_keyframes[idx] == 1:
        tmp_frames.append(frame)
      elif pred_keyframes[idx] == 0 and len(tmp_frames) > 0:
        if len(tmp_frames) == 1:
          pred_keyframe_idxs[idx-1] = 1
        else:
          laplacian_scores = get_laplacian_scores(tmp_frames)
          max_idx = np.argmax(laplacian_scores)
          pred_keyframe_idxs[idx - len(tmp_frames) + max_idx] = 1
        tmp_frames = []
    print("After processing intervals:")
    print(pred_keyframe_idxs)
    print("Ground-Truth Indices:")
    print(gt_keyframes)

    # frame idxs => frames
    for idx, frame in enumerate(reduced_frames):
      if pred_keyframe_idxs[idx] == 1:
        pred_frames.append(frame)
      if gt_keyframes[idx] == 1:
        gt_frames.append(frame)

    print("Number of frames in predicted summary:", len(pred_frames))
    print("Number of frames in ground-truth summary:", len(gt_frames))

    for frame in pred_frames:
      cv2.imshow("Model-Predicted Summary", frame)
      cv2.waitKey(0)
    cv2.destroyAllWindows()

    for frame in gt_frames:
      cv2.imshow("Ground-Truth Summary", frame)
      cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pred_keyframe_idxs, gt_keyframes, pred_frames, gt_frames


def f1_score(pred_keyframes, gt_keyframes):
  matches = pred_keyframes & gt_keyframes
  precision = sum(matches) / sum(pred_keyframes) if sum(pred_keyframes) != 0 else 0.0
  recall = sum(matches) / sum(gt_keyframes) if sum(gt_keyframes) != 0 else 0.0
  f1_score = 2 * precision * recall * 100 / (precision + recall) if precision + recall != 0 else 0.0
  return f1_score

def IoU(pred_frames, gt_frames):
  if len(pred_frames) == 0:
    pred_frames, gt_frames = [], []
    return 0.0

  pred_frames = pred_frames.reshape(-1, disp_W * disp_H * 3)
  gt_frames = gt_frames.reshape(-1, disp_W * disp_H * 3)

  # calculate IoU
  intersection = np.sum(np.logical_and(pred_frames[:, None, :], gt_frames), axis=(1, 2))
  union = np.sum(np.logical_or(pred_frames[:, None, :], gt_frames), axis=(1, 2))
  iou = intersection / union

  # clear memory
  pred_frames, gt_frames = [], []
  return np.mean(iou)

def evaluate(pred_idxs, gt_idxs, pred_keyframes, gt_keyframes):
  print("F1 score:", f1_score(pred_idxs, gt_idxs))
  print("IoU score:", IoU(pred_keyframes, gt_keyframes))


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
  print("Model-Predicted Summary:")
  print(pred_keyframes)
  pred_keyframe_idxs, gt_keyframes, pred_frames, gt_frames = summarizer.show_summaries(pred_keyframes, gt_summary, video_name)

  # TODO: evaluate using fscore and IoU
  evaluate(pred_keyframe_idxs, gt_keyframes.detach().cpu().numpy().astype(int), np.array(pred_frames), np.array(gt_frames))
