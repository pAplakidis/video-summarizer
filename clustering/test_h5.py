import h5py
import numpy as np

with h5py.File("../data/eccv16_dataset_summe_google_pool5.h5", 'r') as hdf:
  print(hdf.keys())
  print(hdf["video_1"].keys())
  feats = np.array(hdf["video_2" + "/features"])
  print(np.array(hdf["video_2" + "/n_frames"]))
  print(np.array(hdf["video_2" + "/n_steps"]))
  print(np.array(hdf["video_2" + "/video_name"]))
  print(feats.shape)
