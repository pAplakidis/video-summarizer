#!/usr/bin/env python3
import os
import cv2
import h5py
import numpy as np
import imutils
from sklearn.neighbors import KNeighborsClassifier
import hdbscan

W = H = 224   # 32
disp_W, disp_H = 1920//2, 1080//2

THRESH = 20.


class Dataset:
  def __init__(self, video_path):
    self.video_path = video_path
    self.video_name = video_path.split('/')[-1].split('.')[0]
    self.hdf = h5py.File("../data/eccv16_dataset_summe_google_pool5.h5", 'r')

    self.frames = []
    self.frames_reduced = []
    self.feats = np.array(self.hdf["video_1" + "/features"])  # TODO: dynamically get video index from video name

  # read video file
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

  # get frames based on difference
  def extract_candidate_frames(self, threshold=20.):
    ret = []
    for i in range(1, len(self.frames)):
      print("Processing frame: %d/%d"%(i, len(self.frames)-1))
      if np.sum(np.absolute(self.frames[i] - self.frames[i-1]))/np.size(self.frames[i]) > threshold:
        ret.append(self.frames[i])

    print("Number of candidate frames %d/%d"%(len(ret), len(self.frames)))
    return ret

  # get an image from video every 15 frames
  def reduce_frames(self):
    for i, f in enumerate(self.frames):
      if i % 15 == 0:
        self.frames_reduced.append(f)


    print("Number of candidate frames %d/%d"%(len(self.frames_reduced), len(self.frames)))
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


def cluster(feats):
  print("Clustering ...")
  Hdbascan = hdbscan.HDBSCAN(min_cluster_size=2,metric='manhattan').fit(feats)
  labels = np.add(Hdbascan.labels_, 1)
  n_clusters = len(np.unique(Hdbascan.labels_))
  print(n_clusters, "clusters generated")

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
      break

  return ret

if __name__ == "__main__":
  video_path = "../data/SumMe/videos/Air_Force_One.mp4"
  # video_path = "../data/SumMe/videos/Base jumping.mp4"

  # load and prepare data
  data = Dataset(video_path)
  data.load_video_frames()
  loader = Dataset(video_path)
  out_path = f"../data/OUT/{data.video_name}"

  # video => candidate frames
  data.reduce_frames()

  # candidate frames => clusters of candidate frames
  cluster_idx_array = cluster(data.feats)

  # clusters of candidate frames => final key frames
  keyframes = extract_keyframes(data, cluster_idx_array)

  # save keyframes
  if not os.path.exists(out_path):
    os.makedirs(out_path)

  # TODO: check if RGB is correct
  for i in sorted(keyframes):
    print("Writing frame:", i)
    cv2.imwrite(out_path+f"/{i}.jpg", data.frames[i])
    cv2.imshow("keyframe", data.frames_reduced[i])
    cv2.waitKey(0)
  print("Frames saved at", out_path)

  # TODO: score metrics => compare to ground-truth

  # cv2.destroyAllWindows()
  data.hdf.close()
