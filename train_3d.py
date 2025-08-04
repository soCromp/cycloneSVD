# https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training#train-the-model

import diffusers 
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from diffusers import DDPMPipeline, DiffusionPipeline, UNet3DConditionModel
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.backends.cuda import sdp_kernel
import torch.multiprocessing as mp
import os
from PIL import Image, ImageDraw
from dataclasses import dataclass, asdict
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import json
import random



config = {
    'image_size': 32,  # the generated image resolution
    'train_batch_size': 8,
    'eval_batch_size': 1,  # how many images to sample during evaluation
    'num_epochs': 250,
    'gradient_accumulation_steps': 8,
    'learning_rate': 1e-7,
    'lr_warmup_steps': 0,
    'save_image_epochs': 5,
    'save_model_epochs': 10,
    'output_dir': "video1e-7",  # the model name locally and on the HF Hub
    'seed': 0,
    'dataset': '/home/cyclone/train/windmag_natlantic',
    'channels': 1, # channels in the images
    'frames': 8,
    'continue': False,
    'img_model': 'img1e-7',
    'dtype': torch.float32,
}


class DummyDataset(Dataset):
    def __init__(self, dataset, 
                 width=1024, height=576, channels=3, sample_frames=25):
        """
        Args:
            num_samples (int): Number of samples in the dataset.
            channels (int): Number of channels, default is 3 for RGB.
        """
        # Define the path to the folder containing video frames
        self.base_folder = dataset
        self.folders = [f for f in os.listdir(self.base_folder) if os.path.isdir(os.path.join(self.base_folder, f))]
        self.num_samples = len(self.folders)
        self.channels = channels
        self.width = width
        self.height = height
        self.sample_frames = sample_frames
        
        # get min, max values for normalization
        self.min = np.inf
        self.max = -1 * np.inf
        for folder in self.folders:
            for i in range(sample_frames):
                frame = np.load(os.path.join(self.base_folder, folder, f'{i}.npy'))
                self.min = min(self.min, frame.min())
                self.max = max(self.max, frame.max())
                

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        chosen_folder = random.choice(self.folders)
        folder_path = os.path.join(self.base_folder, chosen_folder)
        frames = sorted(os.listdir(folder_path))[:self.sample_frames]

        # Initialize a tensor to store the pixel values (3 channels is baked into model)
        pixel_values = torch.empty((1, self.sample_frames, self.height, self.width))

        # Load and process each frame
        for i, frame_name in enumerate(frames):
            frame_path = os.path.join(folder_path, frame_name)
            # with Image.open(frame_path) as img:
            with Image.fromarray(np.load(frame_path)) as img:
                # Resize the image and convert it to a tensor
                img_resized = img.resize((self.width, self.height))
                img_tensor = torch.from_numpy(np.array(img_resized)).float()
                img_tensor[img_tensor.isnan()] = 0.0
                if img_tensor.isnan().sum()>0:
                    raise ValueError(
                        f"{img_tensor.isnan().sum()} NaN values found in the image tensor for frame {frame_name} in folder {chosen_folder}.")
                elif img_tensor.isinf().sum()>0:
                    raise ValueError(
                        f"Inf values found in the image tensor for frame {frame_name} in folder {chosen_folder}.")

                # Normalize the image by scaling pixel values to [-1, 1]
                img_normalized = 2 * (img_tensor - self.min) / (self.max - self.min) - 1
                img_normalized[img_normalized<-1] = -1 # in case of rounding errors
                img_normalized[img_normalized>1] = 1

                # Rearrange channels if necessary
                # if self.channels == 3:
                #     img_normalized = img_normalized.permute(
                #         2, 0, 1)  # For RGB images
                # elif self.channels == 1:
                #     img_normalized = img_normalized.unsqueeze(0).repeat([3,1,1])  
                #     img_normalized = img_normalized.mean(
                #         dim=2, keepdim=True)  # For grayscale images

                pixel_values[:, i, :, :] = img_normalized
        return {'pixel_values': pixel_values}
    

class MixDataset(Dataset):
    def __init__(self, dataset1, dataset2, width=32, height=32, channels=1, sample_frames=8, shared_step=None, choicefunc='uniform'):
        self.dataset1 = DummyDataset(dataset1, width=width, height=height, channels=channels, sample_frames=sample_frames)
        self.dataset2 = DummyDataset(dataset2, width=width, height=height, channels=channels, sample_frames=sample_frames)
        
        self.min = min(self.dataset1.min, self.dataset2.min)
        self.max = max(self.dataset1.max, self.dataset2.max)
        
        self._shared_step = shared_step
        if shared_step is None:
            self._shared_step = mp.Value('i', 0)  # 'i' == signed int
        if choicefunc == 'uniform':
            self.choicefunc = lambda f: np.random.choice([0,1])
        elif choicefunc == 'linear':
            self.choicefunc = lambda f: np.random.choice([0, 1], p=[f, 1-f])
            
        
    def __len__(self):
        return len(self.dataset1) + len(self.dataset2)
    
    
    def __getitem__(self, idx):
        pass


dataset = DummyDataset(dataset=config['dataset'], channels=config['channels'], sample_frames=config['frames'],
                       width=config['image_size'], height=config['image_size'])
dataloader = DataLoader(dataset, batch_size=config['train_batch_size'], shuffle=True, drop_last=True,) # otherwise crashes on last batch
config['data_norm_min'] = dataset.min 
config['data_norm_max'] = dataset.max 

def image_to_video_model(config, time_avg=True):
    # 1) load 2-D UNet and grab its config
    unet2d = diffusers.UNet2DConditionModel.from_pretrained(
        config['img_model'], subfolder='unet', revision='main')
    cfg = dict(unet2d.config)
    cfg['down_block_types'] = ['3D'.join(name.split('2D')) for name in cfg['down_block_types']]  
    cfg['up_block_types'] = ['3D'.join(name.split('2D')) for name in cfg['up_block_types']]  
    cfg['mid_block_type'] = '3D'.join(cfg['mid_block_type'].split('2D'))

    # 2) build a fresh 3-D UNet with matching hyper-params
    unet3d  = UNet3DConditionModel.from_config(cfg)
    print('Video model config', unet3d.config)

    # 3) copy 2-D weights → 3-D
    sd2 = unet2d.state_dict()
    sd3 = unet3d.state_dict()
    for k, w in sd2.items():
        w3d = sd3.get(k, np.asarray([]))
        if w.ndim == 4 and w3d.ndim==5:                                 # (O, I, H, W)
            w3 = w.unsqueeze(2).repeat(1, 1, config['frames'], 1, 1) # add time dimension
            if time_avg:                                # Ho et al., 2022
                w3 /= config['frames']
            sd3[k] = w3
        elif w.ndim > w3d.ndim:
            sd3[k] = w.squeeze()
        else: # no change
            sd3[k] = w
    unet3d.load_state_dict(sd3, strict=False)
    
    unet2d = None
    return unet3d

if not config.get('continue', False): 
    unet = image_to_video_model(config)
    noise_scheduler = diffusers.DDPMScheduler.from_pretrained(
        config['img_model'], subfolder='scheduler', revision='main')
    start_epoch = 0
else: # resume training
    unet = diffusers.UNet3DConditionModel.from_pretrained(
        config['output_dir'], subfolder="unet", revision="main")
    noise_scheduler = diffusers.DDPMScheduler.from_pretrained(
        config['output_dir'], subfolder='scheduler', revision='main')
    with open(os.path.join(config['output_dir'], 'train_log.txt'), 'r') as f:
        logs = f.readlines()
    start_epoch = int(logs[-1].split()[1][:-1]) + 1
    
unet = unet.to('cuda', dtype=config['dtype'])
cross_attention_dim = unet.config['cross_attention_dim']

optimizer = torch.optim.AdamW(
    unet.parameters(),
    lr=config['learning_rate'],
)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config['lr_warmup_steps'],
    num_training_steps=(len(dataloader) * config['num_epochs']),
)

# if args.dataset2 is not None:
#     shared_step = torch.multiprocessing.Value('i', 0)
#     train_dataset = MixDataset(dataset1=args.dataset, dataset2=args.dataset2, width=args.width, height=args.height,
#                                 channels=args.channels, sample_frames=args.num_frames, 
#                                 choicefunc=args.choice_func, max_train_steps=args.max_train_steps, shared_step=shared_step)
# else:


class CondDiffusionPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.unet = unet 
        self.scheduler = scheduler 
        
    @torch.no_grad()
    def __call__(self, input, num_frames=8, generator=None, encoder_hidden_states=None, **kwargs):
        """Input should be [batch_size, channels, 1, height, width], where frame=1 is the prompt frame"""
        device = self.unet.device
        batch_size = input.shape[0]
        sample  = torch.randn(
            batch_size, self.unet.config['out_channels'], num_frames, 32, 32,
            generator=generator, device=device
        )
        sample[:, :, 0, :, :] = input
        
        for t in self.scheduler.timesteps:
            eps = self.unet(sample, t, encoder_hidden_states=encoder_hidden_states).sample
            sample = self.scheduler.step(eps, t, sample).prev_sample
            sample[:, :, 0, :, :] = input
            
        return {"images": sample.cpu()}

def evaluate(samples, config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        torch.as_tensor(samples, dtype=config['dtype'], device='cuda'),
        num_frames=config['frames'],
        generator=torch.Generator(device='cuda').manual_seed(config['seed']), 
        encoder_hidden_states=torch.zeros((config['train_batch_size'], 1, cross_attention_dim),
                                          device='cuda')
        # Use a separate torch generator to avoid rewinding the random state of the main training loop
    )['images']
    
    # output from model is on [-1,1]  scale; convert to [0,255]
    images = 255/2 * ( 1+np.array(images) )
    
    test_dir = os.path.join(config['output_dir'], "samples")
    os.makedirs(test_dir, exist_ok=True)
    
    for i in range(len(images)):
        for t in range(config['frames']):
            frame = Image.fromarray(images[i, :, t, :, :].squeeze()).convert('P')
            frame.save(os.path.join(test_dir, f'{epoch:04d}_s{i:02d}_t{t:02d}.png'))
    
    # # output from model is on [-1,1]  scale; convert to [0,255]
    # images = [Image.fromarray(255/2*(np.array(images[i]).squeeze() + 1)) for i in range(config['eval_batch_size'])]

    # # Make a grid out of the images
    # image_grid = make_image_grid(images, rows=4, cols=4)

    # # Save the images
    # image_grid.save(f"{test_dir}/{epoch:04d}.png")
    
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        log_with="tensorboard",
        project_dir=os.path.join(config['output_dir'], "logs"),
    )
    
    if accelerator.is_main_process:
        os.makedirs(config['output_dir'], exist_ok=True)
        accelerator.init_trackers("train_example")
        
    with open(os.path.join(config['output_dir'], 'config.txt'), 'w') as f:
        str_config = {k:str(v) for k, v in config.items()}
        json.dump(str_config, f, indent=4)
        
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
        
    global_step = 0
    for epoch in range(start_epoch, config['num_epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        losses = []
        
        for step, batch in enumerate(train_dataloader):
            clean_images = torch.as_tensor(batch["pixel_values"], device='cuda', dtype=config['dtype'])
            zeros = torch.zeros((config['train_batch_size'], 1, cross_attention_dim), 
                                device=accelerator.device, dtype=config['dtype'])
            # print(clean_images.min(), clean_images.max())
            
            noise = torch.randn(clean_images.shape, device=clean_images.device, dtype=config['dtype'])
            bs = clean_images.shape[0]
            timesteps = torch.randint(
                0, noise_scheduler.config['num_train_timesteps'], (bs,), device=clean_images.device,
                dtype=torch.int64 # leave as is- don't change to config['dtype']
            )
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            batchlosses = []
            with accelerator.accumulate(model):
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=zeros.to(clean_images.device), 
                                    return_dict=False)[0]
                loss = F.mse_loss(noise_pred[:,:,1:,:,:], noise[:,:,1:,:,:]) # skip zeroth/prompt frame
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            losses.append(logs['loss'])
            global_step += 1
            # if True:
            #     break
            
        loss = np.mean(losses)
        with open(os.path.join(config['output_dir'], 'train_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch}, Step {global_step}, Loss: {loss}, LR: {logs['lr']}\n")
            
        # sample demo images, save model
        if accelerator.is_main_process:
            pipeline = CondDiffusionPipeline(unet=accelerator.unwrap_model(model).cuda(), scheduler=noise_scheduler)
            if (epoch + 1) % config['save_image_epochs'] == 0 or epoch == config['num_epochs'] - 1: # IMAGE
                # get just the first time step/prompt frame
                evaluate(batch["pixel_values"][:, :, 0, :, :], config, epoch, pipeline)
            if (epoch + 1) % config['save_model_epochs'] == 0 or epoch == config['num_epochs'] - 1: # MODEL
                pipeline.save_pretrained(config['output_dir'])
        

train_loop(config, unet, noise_scheduler, optimizer, dataloader, lr_scheduler)
    