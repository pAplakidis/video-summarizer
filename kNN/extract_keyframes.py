#!/usr/bin/env python3
import os
import cv2
import numpy as np
import imutils
from sklearn.neighbors import KNeighborsClassifier
import hdbscan

W = H = 32  # TODO: 224
disp_W, disp_H = 1920//2, 1080//2

THRESH = 20.


# TODO: try adding a CNN
def image_to_feats(img, size=(W,H)):
  return cv2.resize(img, size).flatten()

def get_color_histogram(img, bins=(8, 8, 8)):
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])

  if imutils.is_cv2():
    hist = cv2.normalize(hist)
  else:
    cv2.normalize(hist, hist)

  return hist.flatten()

def load_video_frames(path):
  frames = []
  cap = cv2.VideoCapture(path)

  i = 0
  while True:
    ret, frame = cap.read()
    if not ret:
      break
    print("Loading frame: %d/%d"%(i, int(cap.get(cv2.CAP_PROP_FRAME_COUNT))))

    # cv2.imshow('frame', cv2.resize(frame, (disp_W, disp_H)))
    # if cv2.waitKey(1) == ord('q'):
    #   break
    # extract frames in RGB

    # frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (W, H)).flatten())
    frames.append(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (disp_W, disp_H)))

    i += 1
  
  cap.release()

  return frames

def extract_candidate_frames(frames, threshold):
  ret = []
  for i in range(1, len(frames)):
    print("Processing frame: %d/%d"%(i, len(frames)-1))
    if np.sum(np.absolute(frames[i] - frames[i-1]))/np.size(frames[i]) > threshold:
      ret.append(frames[i])

  return ret

def cluster(frames):
  print("Preprocessing frames ...")
  for i in range(len(frames)):
    frames[i] = image_to_feats(frames[i])

  print("Clustering ...")
  Hdbascan = hdbscan.HDBSCAN(min_cluster_size=2,metric='manhattan').fit(frames)
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

def get_laplacian_scores(frames, n_images):
  variance_laplacians = []

  for img_idx in n_images:
    img = frames[n_images[img_idx]].reshape((W, H, 3))
    # print(img)
    # print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    variance_laplacian = cv2.Laplacian(img, cv2.CV_64F).var()
    variance_laplacians.append(variance_laplacian)

  return variance_laplacians

# calculate the scores of images in each cluster and return the best ones
def extract_keyframes(frames, cluster_idx_array):
  ret = []

  clusters = np.arange(len(cluster_idx_array))
  for cluster in clusters:
    curr_row = cluster_idx_array[cluster][0]
    n_images = np.arange(len(curr_row))
    laplacian_scores = get_laplacian_scores(frames, n_images)

    try:
      ret.append(curr_row[np.argmax(laplacian_scores)])
    except:
      break

  return ret

if __name__ == "__main__":
  frames = load_video_frames("../data/SumMe/videos/Air_Force_One.mp4")

  # video => candidate frames => clusters of candidate frames => final key frames
  candidate_frames = extract_candidate_frames(frames, THRESH)
  print("Number of candidate frames %d/%d"%(len(candidate_frames), len(frames)))

  cluster_idx_array = cluster(candidate_frames)
  keyframes = extract_keyframes(candidate_frames, cluster_idx_array)

  # save keyframes
  for i in sorted(keyframes):
    print("Writing frame:", i)
    cv2.imwrite(f"../data/OUT/SumMe/Air_Force_One/{i}.jpg", frames[i])
    print("Frames saved at", "../data/OUT/SumMe/Air_Force_One")
    # cv2.imshow("keyframe", frames[i])
    # cv2.waitKey(0)

  # cv2.destroyAllWindows()
