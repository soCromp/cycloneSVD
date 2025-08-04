# %% [markdown]
# https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training#train-the-model

# %%
import diffusers 
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.utils import make_image_grid
from diffusers import DDPMPipeline, DiffusionPipeline
import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import os
from PIL import Image, ImageDraw
from dataclasses import dataclass, asdict
from accelerate import Accelerator
from tqdm.auto import tqdm
from pathlib import Path
import json


# %%
config = {
    'image_size': 32,  # the generated image resolution
    'train_batch_size': 16,
    'eval_batch_size': 16,  # how many images to sample during evaluation
    'num_epochs': 250,
    'gradient_accumulation_steps': 8,
    'learning_rate': 1e-7,
    'lr_warmup_steps': 500,
    'save_image_epochs': 10,
    'save_model_epochs': 10,
    # 'mixed_precision': "fp16",  # `no` for float32, `fp16` for automatic mixed precision
    'output_dir': "img1e-7",  # the model name locally and on the HF Hub
    'seed': 0,
    'dataset': '/home/cyclone/train/windmag_atlanticpacific',
    'continue': False,
}

# %%
cross_attention_dim = 768
if not config.get('continue', False):
    unet = diffusers.UNet2DConditionModel(
        sample_size        = 32,      # 32×32 tiles
        in_channels        = 1,       # wind magnitude only
        out_channels       = 1,
        block_out_channels = (32, 64, 128),   # 3 resolution scales: 32→16→8
        layers_per_block   = 2,
        down_block_types   = ("CrossAttnDownBlock2D",
                            "CrossAttnDownBlock2D",
                            "DownBlock2D"),
        up_block_types     = ("UpBlock2D",
                            "CrossAttnUpBlock2D",
                            "CrossAttnUpBlock2D"),
        cross_attention_dim= cross_attention_dim
    )
    last_epoch=0
else: # resume 
    unet = diffusers.UNet2DConditionModel.from_pretrained(
        config['output_dir'], subfolder="unet", revision="main"
    )
    with open(os.path.join(config['output_dir'], 'train_log.txt'), 'r') as f:
        logs = f.readlines()
    last_epoch = int(logs[-1].split()[1][:-1])

# %%
class DummyDataset(Dataset):
    def __init__(self, dataset, 
                 width=32, height=32, channels=3, sample_frames=8):
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

    def __len__(self):
        return self.sample_frames * len(self.folders)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index of the sample to return.

        Returns:
            dict: A dictionary containing the 'pixel_values' tensor of shape (16, channels, 320, 512).
        """
        # Randomly select a folder (representing a video) from the base folder
        folder_idx = idx // self.sample_frames
        frame_idx = idx % self.sample_frames
        frame_path = os.path.join(self.base_folder, self.folders[folder_idx], f'{frame_idx}.npy')
        
        # Initialize a tensor to store the pixel values (3 channels is baked into model)
        # pixel_values = torch.empty((self.sample_frames, 3, self.height, self.width))

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
            img_normalized = (img_tensor / img_tensor.max() * 2) -1.0

        return {'pixel_values': img_normalized.unsqueeze(0)}
    
dataset = DummyDataset(dataset=config['dataset'], channels=1)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# %%
optimizer = torch.optim.AdamW(unet.parameters(), lr=config['learning_rate'])
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config['lr_warmup_steps'],
    num_training_steps=(len(dataloader) * config['num_epochs']),
)

zeros = torch.zeros(config['train_batch_size'], 1, cross_attention_dim)

class CondDiffusionPipeline(DiffusionPipeline):
    def __init__(self, unet, scheduler):
        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler)
        self.unet = unet 
        self.scheduler = scheduler 
        
    @torch.no_grad()
    def __call__(self, batch_size=1, generator=None, encoder_hidden_states=None, **kwargs):
        device = self.unet.device
        sample  = torch.randn(
            batch_size, self.unet.config['out_channels'], 32, 32,
            generator=generator, device=device
        )
        
        for t in self.scheduler.timesteps:
            eps = self.unet(sample, t, encoder_hidden_states=encoder_hidden_states).sample
            sample = self.scheduler.step(eps, t, sample).prev_sample
            
        return {"images": sample.cpu()}
        

# %%
def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config['eval_batch_size'],
        generator=torch.Generator(device='cuda').manual_seed(config['seed']), 
        encoder_hidden_states=zeros.to('cuda'),
        # Use a separate torch generator to avoid rewinding the random state of the main training loop
    )['images']
    # print(images.shape, np.array(images[0]).squeeze().shape)
    
    # output from model is on [-1,1]  scale; convert to [0,255]
    images = [Image.fromarray(255/2*(np.array(images[i]).squeeze() + 1)) for i in range(config['eval_batch_size'])]

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config['output_dir'], "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

# %%
def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    accelerator = Accelerator(
        # mixed_precision=config['mixed_precision'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        log_with="tensorboard",
        project_dir=os.path.join(config['output_dir'], "logs"),
    )
    if accelerator.is_main_process:
        if config['output_dir'] is not None:
            os.makedirs(config['output_dir'], exist_ok=True)
        accelerator.init_trackers("train_example")
    
    with open(os.path.join(config['output_dir'], 'config.txt'), 'w') as f:
        json.dump(config, f, indent=4)
        
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    global_step = 0
    for epoch in range(last_epoch, config['num_epochs']):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        losses = []
        
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["pixel_values"]

            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]
            
            timesteps = torch.randint(
                0, noise_scheduler.config['num_train_timesteps'], (bs,), device=clean_images.device,
                dtype=torch.int64
            )
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)
            
            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, encoder_hidden_states=zeros.to(clean_images.device), return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
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
            
        loss = np.mean(losses)
        with open(os.path.join(config['output_dir'], 'train_log.txt'), 'a') as f:
            f.write(f"Epoch {epoch}, Step {global_step}, Loss: {loss}, LR: {logs['lr']}\n")
        
        # sample demo images, save model
        if accelerator.is_main_process:
            pipeline = CondDiffusionPipeline(unet=accelerator.unwrap_model(model).cuda(), scheduler=noise_scheduler)
            if (epoch + 1) % config['save_image_epochs'] == 0 or epoch == config['num_epochs'] - 1:
                evaluate(config, epoch, pipeline)
            if (epoch + 1) % config['save_model_epochs'] == 0 or epoch == config['num_epochs'] - 1:
                pipeline.save_pretrained(config['output_dir'])
            
noise_scheduler = diffusers.DDPMScheduler(num_train_timesteps=1000)
train_loop(config, unet, noise_scheduler, optimizer, dataloader, lr_scheduler)

