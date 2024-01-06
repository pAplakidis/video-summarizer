import h5py
import numpy as np

with h5py.File("../data/eccv16_dataset_summe_google_pool5.h5", 'r') as hdf:
  print(hdf.keys())
  print(hdf["video_1"].keys())
  feats = np.array(hdf["video_2" + "/features"])
  print("number of frames:", np.array(hdf["video_2" + "/n_frames"]))
  print("number of steps:", np.array(hdf["video_2" + "/n_steps"]))
  print("video name:", np.array(hdf["video_2" + "/video_name"]))
  print("Features:", feats.shape)

  gtscore = np.array(hdf["video_2" + "/gtscore"])
  gtsummary = np.array(hdf["video_2" + "/gtsummary"])
  print("Ground Truth Score:", gtscore.shape)
  print(gtscore)
  print("Ground Truth Summary:", gtsummary.shape)
  print(gtsummary)
  print(gtsummary.shape)

  user_summary = np.array(hdf["video_2" + "/user_summary"])
  print("User Summary")
  print(user_summary)
  print(user_summary.shape)

  change_points = np.array(hdf["video_2" + "/change_points"])
  print(change_points)
