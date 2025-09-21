import numpy as np 
import os 
import sys 
from utils.fvd import compute_fvd
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

def get_fvd(train, synth, model=None):
    train_nonan = np.nan_to_num(train)
    synth_nonan = np.nan_to_num(synth)
    fvd, encoder = compute_fvd(train_nonan, synth_nonan, )
    return fvd


train = load_data(train_path)
synth = load_data(synth_path)

# syndices = random.sample(range(len(train)), len(train)//2)
# synth = train[syndices]
# train = np.delete(train, syndices, axis=0)

print('train', train.shape, 'synth', synth.shape)

fvd = get_fvd(train, synth)
print(fvd)
