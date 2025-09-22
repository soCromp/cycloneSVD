import numpy as np 
import os 
import sys 
from utils.video_metrics import compute_fvd, compute_kvd
from utils.forecast_metrics import compute_rmse
import random

train_path = sys.argv[1] 
synth_path = sys.argv[2]
len_datapoint = 8 # number of frames per datapoint

def load_data(path):
    data = []
    for d in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, d)):
            assert len(os.listdir(os.path.join(path, d))) == len_datapoint 
            point = [] 
            for i in range(len_datapoint):
                point.append(np.load(os.path.join(path, d, f'{i}.npy')))
            data.append(np.stack(point, axis=0))
    return np.stack(data, axis = 0)


def get_fvd(train, synth, encoder=None):
    train_nonan = np.nan_to_num(train)
    synth_nonan = np.nan_to_num(synth)
    fvd, encoder = compute_fvd(train_nonan, synth_nonan, encoder=encoder)
    return fvd, encoder


def get_kvd(train, synth, encoder=None):
    train_nonan = np.nan_to_num(train)
    synth_nonan = np.nan_to_num(synth)
    kvd, _, _, _ = compute_kvd(train_nonan, synth_nonan, encoder=encoder)
    return kvd


def get_rmse(train, synth):
    if train.ndim == 4:
        train = train.unsqueeze(-1)
        synth = synth.unsqueeze(-1)
    _, T, H, W, V = train.shape


train = load_data(train_path)
synth = load_data(synth_path)

# syndices = random.sample(range(len(train)), len(train)//2)
# synth = train[syndices]
# train = np.delete(train, syndices, axis=0)

print('train', train.shape, 'synth', synth.shape)

fvd, encoder = get_fvd(train, synth)
kvd = get_kvd(train, synth, encoder=encoder)
print('fvd', fvd, 'kvd', kvd)
