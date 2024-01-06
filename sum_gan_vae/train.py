#!/usr/bin/env python3
import  os
import torch
from torch.utils.data import DataLoader

from dataloader import VideoDataset
from trainer import Trainer

BS = 1
DATA_DIR = "../data/"
MODEL_PATH = "models/SUM-GAN-VAE"


if __name__ == "__main__":
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("[+] Using device: ", device)

  dataloader = DataLoader(VideoDataset(DATA_DIR), batch_size=1, num_workers=8, pin_memory=True)
  trainer = Trainer(device, dataloader, MODEL_PATH)
  trainer.build_model()
  trainer.train()
