import diffusers
from transformers import CLIPVisionModelWithProjection, CLIPImageProcessor
import torch
import cv2
import numpy as np
import os
import random
from tqdm import tqdm
from PIL import Image 

model_path = '/home/sonia/cycloneSVD/windmag_atlanticpacific1e-6'
ckpt_name = 'checkpoint-50000'
real_data_path = '/home/cyclone/train/windmag_natlantic'

feature_extractor = CLIPImageProcessor.from_pretrained(os.path.join(model_path, 'feature_extractor'))
image_encoder = CLIPVisionModelWithProjection.from_pretrained(os.path.join(model_path, 'image_encoder'))
scheduler = diffusers.DDPMScheduler.from_pretrained(os.path.join(model_path, 'scheduler'))
unet = diffusers.UNetSpatioTemporalConditionModel.from_pretrained(os.path.join(model_path, ckpt_name, 'unet'))
vae = diffusers.AutoencoderKLTemporalDecoder.from_pretrained(os.path.join(model_path, 'vae'))
pipeline = diffusers.StableVideoDiffusionPipeline.from_pretrained(
    'stabilityai/stable-video-diffusion-img2vid',
    image_encoder=image_encoder,
    vae=vae,
    unet=unet,
    scheduler=scheduler,
    feature_extractor=feature_extractor,
)
width = height = 32
num_frames = 8
n = 50 # desired eval set size

outputdir = os.path.join(model_path, ckpt_name, 'synth') #FKA evals
if not os.path.exists(outputdir):
    os.makedirs(outputdir)
    
allexamples = os.listdir(real_data_path)
examples = random.sample(allexamples, n)

with open(os.path.join(outputdir, 'examples.txt'), 'w+') as f:
    f.write('\n'.join(examples))
preds = []

pipeline = pipeline.to('cuda')
for example in tqdm(examples):
    imgs = [] 
    for i in range(num_frames):
        frame_path = os.path.join(real_data_path, example, f'{i}.npy')
        img = Image.fromarray(np.load(frame_path))
        img_resized = img.resize((224,224))
        img_tensor = torch.from_numpy(np.array(img_resized)).float()
        img_normalized = img_tensor / 255
        img_normalized = img_normalized.unsqueeze(0).repeat([3,1,1]).unsqueeze(0)  
        imgs.append(img_normalized)
        
    sample = pipeline(
        imgs[0],
        height=height,
        width=width,
        num_frames=num_frames,
        decode_chunk_size=8,
        motion_bucket_id=127,
        fps=7,
        noise_aug_strength=0.02,
    ).frames[0]
    preds.append(sample)
    os.makedirs(os.path.join(outputdir, ckpt_name, example), exist_ok=True)
    for i in range(num_frames):
        frame_arr = np.array(sample[i])
        np.save(os.path.join(outputdir, ckpt_name, example, f'{i}.npy'), frame_arr)
