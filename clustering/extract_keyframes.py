#!/usr/bin/env python3
import os
import cv2
import h5py
import numpy as np
import imutils
from tqdm import trange

from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
import hdbscan
from hdbscan import flat

W = H = 224   # 32
disp_W, disp_H = 1920//2, 1080//2


class Dataset:
  def __init__(self, video_path):
    self.video_path = video_path
    self.video_name = video_path.split('/')[-1].split('.')[0]
    self.hdf = h5py.File("../data/eccv16_dataset_summe_google_pool5.h5", 'r')
    # TODO: dynamically get video index from video name
    self.video_idx = "video_1"  # CHANGE THIS

    # prepare frames and features
    self.frames = []
    self.frames_reduced = []
    self.feats = np.array(self.hdf[self.video_idx + "/features"])

    self.ground_truth_frames = []

  # read video file
  # TODO: add trange
  def load_video_frames(self):
    cap = cv2.VideoCapture(self.video_path)
    i = 0
    while True:
      ret, frame = cap.read()
      if not ret:
        break

      print("Loading frame: %d/%d"%(i, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))
      self.frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (disp_W, disp_H)))
      i += 1
    
    cap.release()
    return self.frames

  def get_ground_truth(self):
    self.ground_truth_idxs = np.array(self.hdf[self.video_idx + "/gtsummary"])
    self.ground_truth_frames = []
    for idx, f in enumerate(self.frames_reduced):
      if self.ground_truth_idxs[idx] == 1.:
        #self.ground_truth_frames.append(f)
        self.ground_truth_frames.append(idx)

    self.ground_truth_frames = np.array(self.ground_truth_frames)
    self.ground_truth_nclusters = self.ground_truth_frames.shape[0]
    print("Number of frames in Ground Truth Summary:", self.ground_truth_nclusters)

    # display ground-truth
    # for f in self.ground_truth_frames:
    #   cv2.imshow("ground-truth keyframe", f)
    #   cv2.waitKey(0)
    # cv2.destroyAllWindows()

  # get frames based on difference
  def extract_candidate_frames(self, threshold=20.):
    ret = []
    for i in range(1, len(self.frames)):
      print("Processing frame: %d/%d"%(i, len(self.frames)-1))
      if np.sum(np.absolute(self.frames[i] - self.frames[i-1]))/np.size(self.frames[i]) > threshold:
        ret.append(self.frames[i])

    print("[+] Number of candidate frames %d/%d"%(len(ret), len(self.frames)))
    return ret

  # get an image from video every 15 frames
  def reduce_frames(self):
    for i, f in enumerate(self.frames):
      if i % 15 == 0:
        self.frames_reduced.append(f)

    print("[+] Number of candidate frames %d/%d"%(len(self.frames_reduced), len(self.frames)))
    return self.frames_reduced

  # simply downscale and flatten image
  @staticmethod
  def image_to_feats(img, size=(W,H)):
    return cv2.resize(img, size).flatten()

  @staticmethod
  def get_color_histogram(img, bins=(8, 8, 8)):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

    if imutils.is_cv2():
      hist = cv2.normalize(hist)
    else:
      cv2.normalize(hist, hist)

    return hist.flatten()


def cluster(data):
  print("Clustering ...")
  clusterer = hdbscan.HDBSCAN(min_cluster_size=2,metric='manhattan').fit(data.feats)

  # clusterer = flat.HDBSCAN_flat(data.feats, data.ground_truth_nclusters, prediction_data=True)
  # flat.approximate_predict_flat(clusterer, data.feats, data.ground_truth_nclusters)
  
  # clusterer = KMeans(n_clusters=data.ground_truth_nclusters).fit(data.feats)  # TODO: first cluster is always empty

  labels = np.add(clusterer.labels_, 1)
  n_clusters = len(np.unique(clusterer.labels_))
  print("[+]", n_clusters, "clusters generated")

  # get each cluster's images' indices
  clusters_idx_array = []
  for i in np.arange(n_clusters):
    idx_array = np.where(labels == i)
    clusters_idx_array.append(idx_array)

  clusters_idx_array = np.array(clusters_idx_array)
  return clusters_idx_array

def get_laplacian_scores(dataset, n_images):
  variance_laplacians = []

  for img_idx in n_images:
    img = data.frames[n_images[img_idx]]
    # print(img)
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    variance_laplacian = cv2.Laplacian(img, cv2.CV_64F).var()
    variance_laplacians.append(variance_laplacian)

  return variance_laplacians

# calculate the scores of images in each cluster and return the best ones
def extract_keyframes(dataset, cluster_idx_array):
  ret = []

  clusters = np.arange(len(cluster_idx_array))
  for cluster in clusters:
    curr_row = cluster_idx_array[cluster][0]
    n_images = np.arange(len(curr_row))
    laplacian_scores = get_laplacian_scores(dataset, n_images)

    try:
      ret.append(curr_row[np.argmax(laplacian_scores)])
    except:
      # TODO: find a random orphaned image idx and add it here (?)
      ret.append(0) # since the first cluster is always empty, just add the first frame

  return ret

def f1_score(pred_keyframes, gt_keyframes):
  matches = pred_keyframes & gt_keyframes
  precision = sum(matches) / sum(pred_keyframes)
  recall = sum(matches) / sum(gt_keyframes)
  f1score = 2 * precision * recall * 100 / (precision + recall)
  return f1score

def IoU(pred_keyframes, gt_keyframes, video_path):
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

# TODO: user summaries?
# takes onehot encoded vectors of size=len(frames_reduced) from predictions and ground-truth
# and evaluates them using F1 score
def evaluate_summary(pred_keyframes, gt_keyframes, video_path):
  f1score = f1_score(pred_keyframes, gt_keyframes)
  iou = IoU(pred_keyframes, gt_keyframes, video_path)
  return f1score, iou


if __name__ == "__main__":
  video_path = "../data/SumMe/videos/Air_Force_One.mp4"
  # video_path = "../data/SumMe/videos/Base jumping.mp4"

  # load and prepare data
  data = Dataset(video_path)
  out_path = f"../data/OUT/{data.video_name}"
  data.load_video_frames()

  # video => candidate frames
  data.reduce_frames()
  data.get_ground_truth()

  # candidate frames => clusters of candidate frames
  cluster_idx_array = cluster(data)

  # clusters of candidate frames => final key frames
  keyframes = extract_keyframes(data, cluster_idx_array)

  # save keyframes
  if not os.path.exists(out_path):
    os.makedirs(out_path)

  for i in sorted(keyframes):
    print("Writing frame:", i)
    frame = cv2.cvtColor(data.frames_reduced[i], cv2.COLOR_BGR2RGB)
    cv2.imwrite(out_path+f"/{i}.jpg", frame)
    cv2.imshow("predicted keyframe", frame)
    cv2.waitKey(0)
  print("[+] Frames saved at", out_path)

  for i in data.ground_truth_frames:
    print("Writing frame:", i)
    frame = cv2.cvtColor(data.frames_reduced[i], cv2.COLOR_BGR2RGB)
    cv2.imshow("ground-truth keyframe", frame)
    cv2.waitKey(0)

  keyframes_onehot = np.zeros((len(data.frames_reduced)), dtype=int)
  for kf in keyframes:
    keyframes_onehot[kf] = 1.

  print("[+] Predicted Summary has %d keyframes"%len(keyframes))
  print("[+] Ground-Truth Summary has %d keyframes"%data.ground_truth_frames.shape[0])
  f1score, iou = evaluate_summary(keyframes_onehot, data.ground_truth_idxs.astype(int), video_path)
  print("[+] F1_score:", f1score, "%")
  print("[+] IoU score:", iou, "%")

  cv2.destroyAllWindows()
  data.hdf.close()
