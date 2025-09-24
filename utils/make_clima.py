import os 
import xarray as xr 
from tqdm import tqdm

path = '/mnt/data/sonia/cyclone/0.5/slp'
fnames = sorted([fn for fn in os.listdir(path) if fn.endswith('.nc') and fn != 'clima.nc'])
n = len(fnames)

clima = xr.open_dataset(os.path.join(path, fnames[0])).mean(dim='time') * (1/n)

for fn in tqdm(fnames[1:]):
    clima += xr.open_dataset(os.path.join(path, fn)).mean(dim='time') * (1/n)

print(clima) 
clima.to_netcdf(os.path.join(path, 'clima.nc'))
