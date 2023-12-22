import cv2
import numpy as np
from tqdm import tqdm, trange
from datetime import date

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter

from model import Summarizer, Discriminator
from util import *


class Trainer:
  def __init__(self, device, dataloader, model_path, writer_path=None, eval_epoch=False):
    self.device = device
    self.model_path = model_path
    self.eval_epoch = eval_epoch

    if not writer_path:
      today = str(date.today())
      writer_path = "runs/" + today
    print("[+] Tensorboard output path:", writer_path)
    # self.writer = SummaryWriter(writer_path)
    self.writer = TensorboardWriter(writer_path)

    self.dataloader = dataloader

  def build_model(self):
    self.linear_compress = nn.Linear(INPUT_SIZE, HIDDEN_SIZE).to(self.device)
    self.summarizer = Summarizer(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(self.device)
    self.discriminator = Discriminator(input_size=HIDDEN_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(self.device)
    self.model = nn.ModuleList([
      self.linear_compress, self.summarizer, self.discriminator
    ])
    self.model.train()

  @staticmethod
  def save_checkpoint(state, path):
    torch.save(state, path)
    print("[+] Checkpoint saved at:", path)

  @staticmethod
  def freeze_model(module):
    for p in module.parameters():
      p.requires_grad = False

  # Loss Functions
  
  # L2 loss between original and regenerated features at cLSTM's last hidden layer
  def reconstruction_loss(self, h_origin, h_fake):
    return torch.norm(h_origin - h_fake, p=2)

  # KL( q(e|x) || N(0,1) )
  def prior_loss(self, mu, log_variance):
    return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

  # Summary-Length Regularization
  def sparsity_loss(self, scores):
    return torch.abs(torch.mean(scores) - SUMMARY_RATE)

  # Typical GAN loss + Classify uniformly scored features
  def gan_loss(self, original_prob, fake_prob, uniform_prob):
    return torch.mean(torch.log(original_prob) +
                      torch.log(1 - fake_prob) + torch.log(1 - uniform_prob))

  # Evaluation Metrics

  @staticmethod
  def f1_score(pred_keyframes, gt_keyframes):
    matches = pred_keyframes & gt_keyframes
    precision = sum(matches) / sum(pred_keyframes) if sum(pred_keyframes) != 0 else 0.0
    recall = sum(matches) / sum(gt_keyframes) if sum(gt_keyframes) != 0 else 0.0
    f1_score = 2 * precision * recall * 100 / (precision + recall) if precision + recall != 0 else 0.0
    return f1_score

  def IoU(self, pred_keyframes, gt_keyframes, video_name):
    video_path = self.dataloader.dataset.video_dir + video_name + ".mp4"
    pred_frames, gt_frames = [], []
    
    # load frames into memory
    cap = cv2.VideoCapture(video_path)
    idx = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        break

      frame = cv2.resize(frame, (W,H))
      try:
        if pred_keyframes[idx] == 1:
          pred_frames.append(frame)
        if gt_keyframes[idx] == 1:
          gt_frames.append(frame)
      except IndexError:
        pass
      idx += 1

    if len(pred_frames) == 0:
      pred_frames, gt_frames = [], []
      return 0.0

    pred_frames = np.array(pred_frames)
    gt_frames = np.array(gt_frames)
    pred_frames = pred_frames.reshape(-1, W * H * 3)
    gt_frames = gt_frames.reshape(-1, W * H * 3)

    # calculate IoU
    intersection = np.sum(np.logical_and(pred_frames[:, None, :], gt_frames), axis=(1, 2))
    union = np.sum(np.logical_or(pred_frames[:, None, :], gt_frames), axis=(1, 2))
    iou = intersection / union

    # clear memory
    pred_frames, gt_frames = [], []
    return np.mean(iou)

  def train(self, epochs=N_EPOCHS, lr=LR):
    self.s_e_optimizer = optim.Adam(
      list(self.summarizer.s_lstm.parameters()) +
      list(self.summarizer.vae.e_lstm.parameters()) +
      list(self.linear_compress.parameters()),
      lr=lr
    )
    self.d_optimizer = optim.Adam(
      list(self.summarizer.vae.d_lstm.parameters()) +
      list(self.linear_compress.parameters()),
      lr=lr
    )
    self.c_optimizer = optim.Adam(
      list(self.discriminator.parameters()) +
      list(self.linear_compress.parameters()),
      lr=lr
    )

    print("Training ...")
    step = 0
    for epoch_i in range(epochs):
      s_e_loss_history = []
      d_loss_history = []
      c_loss_history = []

      print(f"\n[=>] Epoch {epoch_i+1}/{epochs}")
      for i_batch, sample_batched in enumerate(t := tqdm(self.dataloader, desc=f"[=>] Epoch {epoch_i+1}/{epochs}")):
        # t.set_description(f"{i_batch} - {sample_batched.shape}")
        img_feats = sample_batched["features"].view(-1, INPUT_SIZE).to(self.device) # [batch_size, n_frames, frame_feats_vec=1024] => [n_frames, frame_feats_vec=1024]

        # train sLSTM and eLSTM
        t.write("[->] Training sLSTM and eLSTM ...")
        original_feats = self.linear_compress(img_feats.detach()).unsqueeze(1)  # [n_frames, 1, frame_feats]
        t.write(f"original feats: {original_feats.shape}")

        scores, h_mu, h_log_variance, generated_feats = self.summarizer(original_feats)
        # t.write(f"\nsLSTM out: {scores.shape}")
        # t.write(f"\nVAE out:\n{h_mu.shape}")
        # t.write(f"{h_log_variance.shape}")
        # t.write(f"{generated_feats.shape}")
        _, _, _, uniform_feats = self.summarizer(original_feats, uniform=True)

        h_origin, original_prob = self.discriminator(original_feats)
        h_fake, fake_prob = self.discriminator(generated_feats)
        h_uniform, uniform_prob = self.discriminator(uniform_feats)
        t.write(f'original_p: {original_prob.item():.3f}, fake_p: {fake_prob.item():.3f}, uniform_p: {uniform_prob.item():.3f}')

        reconstruction_loss = self.reconstruction_loss(h_origin, h_fake)
        prior_loss = self.prior_loss(h_mu, h_log_variance)
        sparsity_loss = self.sparsity_loss(scores)
        t.write(f'recon loss {reconstruction_loss.item():.3f}, prior loss: {prior_loss.item():.3f}, sparsity loss: {sparsity_loss.item():.3f}')

        s_e_loss = reconstruction_loss + prior_loss + sparsity_loss
        self.s_e_optimizer.zero_grad()
        s_e_loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        self.s_e_optimizer.step()
        s_e_loss_history.append(s_e_loss.data)

        # train dLSTM
        t.write("[->] Training dLSTM ...")
        original_feats = self.linear_compress(img_feats.detach()).unsqueeze(1)

        scores, h_mu, h_log_variance, generated_feats = self.summarizer(original_feats)
        _, _ , _, uniform_feats = self.summarizer(original_feats, uniform=True)

        h_origin, original_prob = self.discriminator(original_feats)
        h_fake, fake_prob = self.discriminator(generated_feats)
        h_uniform, uniform_prob = self.discriminator(uniform_feats)
        t.write(f'original_p: {original_prob.item():.3f}, fake_p: {fake_prob.item():.3f}, uniform_p: {uniform_prob.item():.3f}')

        reconstruction_loss = self.reconstruction_loss(h_origin, h_fake)
        gan_loss = self.gan_loss(original_prob, fake_prob, uniform_prob)
        t.write(f'recon loss {reconstruction_loss.item():.3f}, gan loss: {gan_loss.item():.3f}')

        d_loss = reconstruction_loss + gan_loss
        self.d_optimizer.zero_grad()
        d_loss.backward()
        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
        self.d_optimizer.step()

        d_loss_history.append(d_loss.data)

        # train cLSTM
        if i_batch > DISCRIMINATOR_SLOW_START:
          t.write("[->] Training cLSTM ...")
          original_feats = self.linear_compress(img_feats.detach()).unsqueeze(1)

          scores, h_mu, h_log_variance, generated_feats = self.summarizer(original_feats)
          _, _ , _, uniform_feats = self.summarizer(original_feats, uniform=True)

          h_origin, original_prob = self.discriminator(original_feats)
          h_fake, fake_prob = self.discriminator(generated_feats)
          h_uniform, uniform_prob = self.discriminator(uniform_feats)
          t.write(f'original_p: {original_prob.item():.3f}, fake_p: {fake_prob.item():.3f}, uniform_p: {uniform_prob.item():.3f}')

          # maximization
          c_loss = -1 * self.gan_loss(original_prob, fake_prob, uniform_prob)
          t.write(f'gan loss: {gan_loss.item():.3f}')

          self.c_optimizer.zero_grad()
          c_loss.backward()
          # gradient clipping
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), CLIP)
          self.c_optimizer.step()

          c_loss_history.append(c_loss.data)

        self.writer.update_loss(reconstruction_loss.item(), step, "recon_loss")
        self.writer.update_loss(prior_loss.data, step, 'prior_loss')
        self.writer.update_loss(sparsity_loss.data, step, 'sparsity_loss')
        self.writer.update_loss(gan_loss.data, step, 'gan_loss')

        # self.writer.update_loss(s_e_loss.data, step, 's_e_loss')
        # self.writer.update_loss(d_loss.data, step, 'd_loss')
        # self.writer.update_loss(c_loss.data, step, 'c_loss')

        self.writer.update_loss(original_prob.data, step, 'original_prob')
        self.writer.update_loss(fake_prob.data, step, 'fake_prob')
        self.writer.update_loss(uniform_prob.data, step, 'uniform_prob')

        step += 1
      
      s_e_loss = torch.stack(s_e_loss_history).mean()
      d_loss = torch.stack(d_loss_history).mean()
      c_loss = torch.stack(c_loss_history).mean()

      # plot
      t.write("Plotting ...")
      self.writer.update_loss(s_e_loss, epoch_i, 's_e_loss_epoch')
      self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
      self.writer.update_loss(c_loss, epoch_i, 'c_loss_epoch')

      checkpoint_path = str(self.model_path + f"_epoch-{epoch_i}.pth")
      t.write(f"[+] Saved parameters at {checkpoint_path}")
      torch.save(self.model.state_dict(), checkpoint_path)

      # self.evaluate_epoch(epoch_i)
      # self.model.train()

    save_path = str(self.model_path + ".pth")
    torch.save(self.model.state_dict(), save_path)
    print(f"[+] Saved parameters at {save_path}")

    self.evaluate_ground_truth()

  # TODO: evaluate using test/unseen videos (train/test split 80/20 ?)
  def evaluate_epoch(self, epoch_i):
    self.model.eval()

  def evaluate_ground_truth(self):
    self.model.eval()

    for i_batch, sample_batched in enumerate(t := tqdm(self.dataloader, desc="[=>] Eval")):
      img_feats = sample_batched["features"].view(-1, INPUT_SIZE).to(self.device) # [batch_size, n_frames, frame_feats_vec=1024] => [n_frames, frame_feats_vec=1024]
      gt_summary = sample_batched["gtsummary"].squeeze(0).detach().cpu().numpy()
      video_name = sample_batched["video_name"][0]

      img_feats = self.linear_compress(img_feats.detach()).unsqueeze(1)
      scores = self.summarizer.s_lstm(img_feats).squeeze(1).detach().cpu().numpy()
      pred_scores = np.round(scores)  # TODO: use a threshold

      # t.write(f"{pred_scores.astype(int)}")
      # t.write(f"{gt_summary.astype(int)}")
      f1_score = self.f1_score(pred_scores.astype(int), gt_summary.astype(int))
      t.write(f"F1 score: {f1_score}")
      iou_score = self.IoU(pred_scores.astype(int), gt_summary.astype(int), video_name)
      t.write(f"IoU score: {iou_score}")
