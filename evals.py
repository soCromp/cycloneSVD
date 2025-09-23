import numpy as np 
import os 
import sys 
from utils.video_metrics import compute_fvd, compute_kvd
from utils.forecast_metrics import RmseAccumulator
import random

train_path = sys.argv[1] 
synth_path = sys.argv[2]
len_datapoint = 8 # number of frames per datapoint

def load_data(path):
    data = []
    names = []
    for d in sorted(os.listdir(path)):
        if os.path.isdir(os.path.join(path, d)):
            assert len(os.listdir(os.path.join(path, d))) == len_datapoint 
            names.append(d)
            point = [] 
            for i in range(len_datapoint):
                point.append(np.load(os.path.join(path, d, f'{i}.npy')).squeeze())
            data.append(np.stack(point, axis=0))
    return {'names': names, 'data': np.stack(data, axis = 0)}


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


def get_rmse(train, synth, batch_size=64):
    if train.ndim == 4:
        train = np.expand_dims(train, -1)
        synth = np.expand_dims(synth, -1)
    _, T, H, W, V = train.shape
    accumulator = RmseAccumulator(
        T, H, W, V,
        train.mean(), train.std(),
        standardize=False, var_weights=None,
    )
    n_splits = int(np.ceil(train.shape[0] / batch_size))
    for batch_synth, batch_train in zip(np.array_split(synth, n_splits), np.array_split(train, n_splits)):
        accumulator.update(batch_synth, batch_train)
    return accumulator.results()


train = load_data(train_path)
synth = load_data(synth_path)

train_match_data = []
for name in synth['names']:
    assert name in train['names'], f"{name} not in training set"
    train_match_data.append(train['data'][train['names'].index(name)])
train_match = {'names': synth['names'], 'data': np.stack(train_match_data, axis=0)}

# syndices = random.sample(range(len(train)), len(train)//2)
# synth = train[syndices]
# train = np.delete(train, syndices, axis=0)

print('train', train['data'].shape, 'synth', synth['data'].shape)

fvd, encoder = get_fvd(train['data'], synth['data'])
kvd = get_kvd(train['data'], synth['data'], encoder=encoder)
print('fvd', fvd, 'kvd', kvd)

rmse = get_rmse(train_match['data'], synth['data'])
print('rmse', rmse)
