# %%
# nts: activate langchain_env 
import cdsapi
import logging
from tqdm import tqdm
import xarray as xr
import numpy as np
import pandas as pd
import os
import sys
import contextlib
import threading

trackspath='/home/sonia/mcms/tracker/1940-2010/era5/read_era5/out_era5_output_1940_2010.txt'
# trackspath='/home/sonia/mcms/tracker/2010-2024/era5/out_era5/era5/mcms_era5_2010_2024_tracks.txt'
use_slp = False # whether to include slp channel
threads = 32

# %%
# make dataframe of all tracks 
tracks = pd.read_csv(trackspath, sep=' ', header=None, 
        names=['year', 'month', 'day', 'hour', 'total_hrs', 'unk1', 'unk2', 'unk3', 'unk4', 'unk5', 'unk6', 
               'z1', 'z2', 'unk7', 'tid', 'sid'])
tracks = tracks.sort_values(by=['year', 'month', 'day', 'hour'])
tracks['lat'] = 90-tracks['unk1'].values/100
tracks['lon'] = tracks['unk2'].values/100
tracks = tracks[['year', 'month', 'day', 'hour', 'tid', 'sid', 'lat', 'lon']]


regmask = xr.open_dataset('/home/cyclone/regmask_0723_anl.nc')
reg_id = 110 # atlantic ocean

# %%
box = 32 # (box/2 from center in each direction)
if use_slp:
    file_year = 1940
    slp = xr.open_dataset('/home/cyclone/slp.1940.nc')
    slp_next = xr.open_dataset('/home/cyclone/slp.1941.nc')
    

def prep_point(df, client, thread=0):
    """make one training datapoint. df contains year/../hr, lat, lon of center"""
    boxes = []
    for _, frame in df.iterrows():
        year, month, day, hour = frame['year'], frame['month'], frame['day'], frame['hour']
        lat, lon = frame['lat'], frame['lon']
        if use_slp:
            # get the box
            if year==file_year:
                slp_box = slp.sel(time=f'{year}-{month:02d}-{day:02d}T{hour:02d}:00:00',
                                lat=slice(lat+box/4, lat-box/4), lon=slice(lon-box/4, lon+box/4))
                                # /4, because /2 for half box and /2 for grid resolution of 0.5 degrees
            elif year==file_year+1:
                slp_box = slp_next.sel(time=f'{year}-{month:02d}-{day:02d}T{hour:02d}:00:00', 
                                lat=slice(lat+box/4, lat-box/4), lon=slice(lon-box/4, lon+box/4))
            else:
                raise ValueError(f'Year {year} not supported, file year is {file_year}')
            slp_box = slp_box.slp.squeeze().values
        
        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                # "sea_surface_temperature"
            ],
            "year": [str(year)],
            "month": [str(month)],
            "day": [str(day)],
            "time": [f"{hour}:00"],
            'format': 'netcdf',
            "download_format": "unarchived",
            "area": [lat+box/4, lon-box/4, lat-box/4, lon+box/4],
            'grid': '0.5/0.5', 
        }
        # with suppress_output():
        out = client.retrieve('reanalysis-era5-single-levels', request, f'temp_{str(thread)}.nc') #.download()

        ds = xr.open_dataset(f'temp_{str(thread)}.nc')
        u = ds['u10'].squeeze().values[:box, :box] # deal with rounding things
        v = ds['v10'].squeeze().values[:box, :box]
        magnitude = np.sqrt(u**2 + v**2)
        boxes.append(magnitude)
        
    return boxes

# %%
sids = tracks[tracks['year'] <= 2010]['sid'].unique() # SIDS STARTING BEFORE 2010
# sids = tracks['sid'].unique()
RADIUS=6371 # Earth radius in km
outpath = '/home/cyclone/train/windmag_natlantic'
if not os.path.exists(outpath):
    os.makedirs(outpath)
readme = """32x32 of just wind magnitude in north atlantic ocean, 8 frames long, over [2010,2024]. Made on 18 June 2025"""
with open(f'{outpath}/README.txt', 'w') as f:
    f.write(readme)

def worker(sids_chunk, thread_id):
    # with suppress_output():
    client = cdsapi.Client()
    
    for i, sid in enumerate(sids_chunk):
        if i % 100 == 0:
            print(f'Thread {thread_id}: Processing sid {i}/{len(sids_chunk)}: {i/len(sids_chunk)*100:.2f}% complete')
        
        if os.path.exists(f'{outpath}/{sid}') and len(os.listdir(f'{outpath}/{sid}')) >= 8:
            continue #already processed
        sid_df = tracks[tracks['sid'] == sid]
        start_lat = sid_df['lat'].iloc[0]
        start_lon = sid_df['lon'].iloc[0]
        if len(sid_df) < 10:
            continue
        elif np.abs(start_lat) > 70:
            continue # starts poleward of 70 degrees
        elif start_lat < 0 or 110 not in regmask.sel(lono=start_lon, lato=start_lat, method='nearest')['regmaskoc'].values:
            continue # only get north atlantic ocean area

        sid_df = sid_df.sort_values(by=['tid'])
        
        # # check total distance traveled (sum of great circle)
        # lat1 = np.radians(sid_df['lat'].to_numpy()[:-1])
        # lon1 = np.radians(sid_df['lon'].to_numpy()[1])
        # lat2 = np.radians(sid_df['lat'].to_numpy()[1:])
        # lon2 = np.radians(sid_df['lon'].to_numpy()[1:])
        # dlat = lat2 - lat1
        # dlon = lon2 - lon1
        # a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        # c = 2 * np.arcsin(np.sqrt(a))
        # dist = np.sum(RADIUS * c)
        
        sid_df = sid_df.iloc[:8]  # only take the first 8 frames for debugging
        
        if use_slp and sid_df['year'].iloc[0] == file_year + 1: # starts in the next year
            slp = slp_next
            try:
                slp_next = xr.open_dataset(f'/home/cyclone/slp.{file_year+2}.nc')
            except:
                slp_next = None # reaching the end of our data
            file_year += 1 

        point = prep_point(sid_df, client, thread_id)
        os.makedirs(f'{outpath}/{sid}', exist_ok=True)
        for i, frame in enumerate(point):
            np.save(f'{outpath}/{sid}/{i}.npy', frame)

# %%
for i in range(threads):
    start = i * len(sids) // threads
    end = (i + 1) * len(sids) // threads
    sids_chunk = sids[start:end]
    print(start, end, sids_chunk.shape)
    thread = threading.Thread(target=worker, args=(sids_chunk, i))
    thread.start()
    # worker(sids_chunk, i)
    
for i in range(threads):
    thread.join()
print("All threads completed.")
