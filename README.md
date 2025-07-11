# SVD_Xtend

**Stable Video Diffusion Training Code 🚀**

pip install numpy torch==2.6.0 torchvision torchaudio xarray diffusers==0.32.2 tqdm transformers==4.50.0 pandas matplotlib notebook accelerate==1.5.2 opencv-python==4.11.0.86 einops==0.8.1

## Training code
```accelerate launch train_svd.py     --dataset /home/cyclone/train/windmag_atlanticpacific     --output_dir /home/sonia/cycloneSVD/windmag_atlanticpacific2     --per_gpu_batch_size=16 --gradient_accumulation_steps=1     --max_train_steps=50000     --channels=1     --width=32     --height=32     --checkpointing_steps=500 --checkpoints_total_limit=1     --learning_rate=1e-5 --lr_warmup_steps=0     --seed=123      --validation_steps=100     --num_frames=8     --mixed_precision="fp16" ```

