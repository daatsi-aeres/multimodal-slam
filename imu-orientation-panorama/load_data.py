import pickle
import sys
import time
import numpy as np
import os

def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  
  # Try loading from .npz if available (much faster/lighter)
  root, ext = os.path.splitext(fname)
  npz_path = root + ".npz"
  
  if os.path.exists(npz_path):
    print(f"Loading optimized data from {npz_path}...")
    try:
      raw = np.load(npz_path, allow_pickle=True)
      if 'data' in raw.files and len(raw.files) == 1:
        d = raw['data']
      else:
        d = dict(raw)
      return d
    except Exception as e:
      print(f"Failed to load npz: {e}, falling back to pickle")

  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d

dataset="2"
cfile = "../data/trainset/cam/cam" + dataset + ".p"
ifile = "../data/trainset/imu/imuRaw" + dataset + ".p"
vfile = "../data/trainset/vicon/viconRot" + dataset + ".p"

ts = tic()
camd = read_data(cfile)
imud = read_data(ifile)
vicd = read_data(vfile)
toc(ts,"Data import")


