import torch

def find_min_max(loader):
  minn = 0
  maxx = 0
  for b in loader:
    if b.min() < minn:
      minn = b.min() # -18.6330
    if b.max() > maxx:
      maxx = b.max() # 19.5047
  return minn, maxx

def normalize_fn(loader):
    minn, maxx = find_min_max(loader)
    normalize_fn = lambda x: -1 + 2 * ((x - minn) / (maxx - minn))
    return normalize_fn 

def inverse_normalize_fn(loader):
    minn, maxx = find_min_max(loader)
    inverse_normalize_fn = lambda x: minn + (x + 1) * (maxx - minn) / 2
    return inverse_normalize_fn
