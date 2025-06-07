# CLEAR-VISION: Advanced Image Restoration with Variational Autoencoders


## 🎯 Overview

CLEAR-VISION is a state-of-the-art deep learning project that implements a Variational Autoencoder (VAE) for comprehensive image restoration and denoising. The system can handle multiple types of image corruptions including mask-based occlusions, Gaussian noise, salt-and-pepper noise, blur effects, JPEG compression artifacts, and combined corruption scenarios.

### Key Highlights
- **Multi-corruption handling**: Supports 6 different types of image corruptions
- **Advanced VAE architecture**: Custom VAE with residual blocks for superior reconstruction
- **Comprehensive evaluation**: Includes PSNR, SSIM, LPIPS, and FID metrics
- **Real-time processing**: Optimized for both CPU and GPU inference
- **Production-ready**: Complete training pipeline with checkpointing and visualization

## ✨ Features

### 🔧 Corruption Types Supported
- **Mask-based Occlusion**: Square mask removal and inpainting
- **Gaussian Noise**: Random noise corruption removal
- **Salt & Pepper Noise**: Impulse noise elimination
- **Gaussian Blur**: Motion and defocus blur correction
- **JPEG Artifacts**: Compression artifact removal
- **Combined Corruption**: Multiple simultaneous corruptions

### 🏗️ Architecture Features
- **Residual VAE**: Enhanced gradient flow with residual connections
- **Latent Space Manipulation**: 128-dimensional latent representation
- **Batch Normalization**: Stable training with batch normalization layers
- **SiLU Activation**: Modern activation functions for better performance

### 📊 Evaluation Metrics
- **PSNR**: Peak Signal-to-Noise Ratio for reconstruction quality
- **SSIM**: Structural Similarity Index for perceptual quality
- **LPIPS**: Learned Perceptual Image Patch Similarity
- **FID**: Fréchet Inception Distance for distribution matching



### Dataset Setup

The project is designed to work with the CelebA-HQ dataset:

```bash
# Download CelebA-HQ dataset (example path)
# Update the dataset path in the code accordingly
# Default path: '/kaggle/input/celebahq-resized-256x256/celeba_hq_256'
```

### Basic Usage

```python
import torch
from clear_vision_complete import *

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=128).to(device)

# Load dataset
train_loader, val_loader = load_dataset(max_images=1000)  # Limit for quick testing

# Train model
if train_loader and val_loader:
    train_losses, val_losses = train_vae(
        model, train_loader, val_loader, 
        num_epochs=50, lr=1e-4
    )

# Restore a single image
restored_results = restore_image(
    model, 
    'path/to/your/image.jpg', 
    corruption_type='combined'
)
```

## 📚 Detailed Usage

### Training Configuration

```python
# Advanced training configuration
model = VAE(latent_dim=128).to(device)

# Training parameters
training_config = {
    'num_epochs': 100,
    'learning_rate': 1e-4,
    'kld_weight': 0.001,  # KL divergence weight
    'batch_size': 16,     # Adjust based on GPU memory
    'save_dir': './checkpoints'
}

# Start training
train_losses, val_losses = train_vae(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    **training_config
)
```

### Model Evaluation

```python
# Comprehensive evaluation
eval_results = evaluate_model(
    model, 
    val_loader, 
    corruption_types=['mask', 'gaussian', 'blur', 'jpeg', 'combined']
)

# Print results
for corruption_type, metrics in eval_results.items():
    print(f"{corruption_type.upper()} Results:")
    print(f"  PSNR: {metrics['psnr']:.2f} dB")
    print(f"  SSIM: {metrics['ssim']:.4f}")
    print(f"  LPIPS: {metrics['lpips']:.4f}")
    print(f"  FID: {metrics['fid']:.2f}")
    print(f"  Inference Time: {metrics['inference_time_ms']:.2f} ms")
```

### Single Image Restoration

```python
# Restore individual images
corruption_types = ['mask', 'gaussian', 'salt_pepper', 'blur', 'jpeg']

for corruption in corruption_types:
    original, corrupted, restored, metrics = restore_image(
        model=model,
        image_path='sample_image.jpg',
        corruption_type=corruption,
        save_path=f'restored_{corruption}.png'
    )
    print(f"{corruption}: PSNR = {metrics['psnr']:.2f} dB")
```

### Loading Pretrained Models

```python
# Load a saved model
model = load_pretrained_model('./checkpoints/vae_epoch_100.pth')

# Use for inference
if model:
    results = restore_image(model, 'test_image.jpg', 'combined')
```

## 🛠️ Architecture Details

### VAE Model Structure

```
Encoder:
├── Conv2d(3→32) + BatchNorm + SiLU
├── ResidualBlock(32→32)
├── Conv2d(32→64) + BatchNorm + SiLU
├── ResidualBlock(64→64)
├── Conv2d(64→128) + BatchNorm + SiLU
├── ResidualBlock(128→128)
├── Conv2d(128→256) + BatchNorm + SiLU
├── ResidualBlock(256→256)
└── Latent Space Mapping (μ, σ)

Decoder:
├── Latent Space Reconstruction
├── ResidualBlock(256→256)
├── ConvTranspose2d(256→128) + BatchNorm + SiLU
├── ResidualBlock(128→128)
├── ConvTranspose2d(128→64) + BatchNorm + SiLU
├── ResidualBlock(64→64)
├── ConvTranspose2d(64→32) + BatchNorm + SiLU
├── ResidualBlock(32→32)
└── ConvTranspose2d(32→3) + Tanh
```

### Loss Function

```python
# VAE Loss = Reconstruction Loss + KL Divergence
loss = MSE(reconstructed, original) + β * KL_divergence(μ, σ)

# Where β (kld_weight) = 0.001 for balanced reconstruction
```

## 📊 Performance Benchmarks

### Typical Results on CelebA-HQ

| Corruption Type | PSNR (dB) | SSIM | LPIPS | FID | Inference (ms) |
|-----------------|-----------|------|-------|-----|----------------|
| Mask Removal    | 28.5      | 0.89 | 0.12  | 45.2| 15.3           |
| Gaussian Noise  | 26.8      | 0.85 | 0.15  | 52.1| 15.1           |
| Salt & Pepper   | 27.2      | 0.87 | 0.14  | 48.9| 15.2           |
| Blur Correction | 25.9      | 0.82 | 0.18  | 58.3| 15.4           |
| JPEG Artifacts  | 24.6      | 0.79 | 0.21  | 62.7| 15.8           |
| Combined        | 26.1      | 0.83 | 0.16  | 55.4| 15.5           |

### Hardware Requirements

- **Minimum**: 8GB RAM, GTX 1060 6GB
- **Recommended**: 16GB RAM, RTX 3070 8GB or better
- **CPU Training**: Possible but significantly slower
- **Inference**: Real-time on modern GPUs (~15ms per image)


## 🔬 Technical Implementation Details

### Corruption Functions

```python
# Available corruption types with parameters
corruption_params = {
    'mask': {'mask_size': 32, 'random_position': True},
    'gaussian': {'mean': 0.0, 'std': 0.1},
    'salt_pepper': {'salt_prob': 0.02, 'pepper_prob': 0.02},
    'blur': {'kernel_size': 7, 'sigma': 1.5},
    'jpeg': {'quality': 10},
    'combined': {'random_combination': True}
}
```

### Training Features

- **Automatic checkpointing** every 10 epochs
- **Learning rate scheduling** with ReduceLROnPlateau
- **Early stopping** capability
- **Real-time sample generation** during training
- **Comprehensive logging** of all metrics
- **GPU memory optimization** for large datasets

### Evaluation Pipeline

```python
# Evaluation includes:
1. PSNR calculation for reconstruction quality
2. SSIM for structural similarity
3. LPIPS for perceptual similarity
4. FID for distribution matching
5. Inference time measurement
6. Memory usage tracking
```

