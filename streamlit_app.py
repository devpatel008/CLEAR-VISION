# streamlit_app.py

import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2
import io
import time
import os
import matplotlib.pyplot as plt
from io import BytesIO

# Define the ResidualBlock and VAE model architecture exactly as in your notebook
class ResidualBlock(nn.Module):
    """Residual block for better gradient flow and feature learning."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection if dimensions change
        self.skip = nn.Sequential()
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.skip(residual)
        out = F.silu(out)
        return out

class VAE(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            ResidualBlock(32, 32),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            ResidualBlock(64, 64),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            ResidualBlock(128, 128),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.SiLU(),
            
            ResidualBlock(256, 256)
        )
        
        # For 128x128 input, feature map size is now 8x8
        self.encoder_output_size = 8
        encoder_flatten_size = 256 * self.encoder_output_size * self.encoder_output_size
        
        # Latent space mapping
        self.fc_mu = nn.Linear(encoder_flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(encoder_flatten_size, latent_dim)
        self.fc_decoder = nn.Linear(latent_dim, encoder_flatten_size)
        
        # Decoder
        self.decoder = nn.Sequential(
            ResidualBlock(256, 256),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            
            ResidualBlock(128, 128),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            
            ResidualBlock(64, 64),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            
            ResidualBlock(32, 32),
            
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1) # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def decode(self, z):
        x = self.fc_decoder(z)
        x = x.view(x.size(0), 256, self.encoder_output_size, self.encoder_output_size)
        x = self.decoder(x)
        return x
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mu, logvar

# Corruption functions from your notebook
def apply_mask(image, mask_size=32, offset_x=None, offset_y=None):
    """Apply a square mask to the image at a random position."""
    if len(image.shape) == 3:  # Add batch dimension if needed
        image = image.unsqueeze(0)
        
    batch_size, channels, height, width = image.shape
    corrupted_image = image.clone()
    
    # For each image in the batch
    for i in range(batch_size):
        # Determine mask position (random if not specified)
        if offset_x is None:
            offset_x = np.random.randint(0, width - mask_size)
        if offset_y is None:
            offset_y = np.random.randint(0, height - mask_size)
        
        # Apply mask (set to 0)
        corrupted_image[i, :, offset_y:offset_y+mask_size, offset_x:offset_x+mask_size] = 0.0
    
    return corrupted_image

def apply_gaussian_noise(image, mean=0.0, std=0.1):
    """Add Gaussian noise to image."""
    if len(image.shape) == 3:  # Add batch dimension if needed
        image = image.unsqueeze(0)
        
    noise = torch.randn_like(image) * std + mean
    corrupted_image = image + noise
    # Clip values to be in [-1, 1] range
    corrupted_image = torch.clamp(corrupted_image, -1, 1)
    return corrupted_image

def apply_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """Add salt and pepper noise to image."""
    if len(image.shape) == 3:  # Add batch dimension if needed
        image = image.unsqueeze(0)
        
    corrupted_image = image.clone()
    batch_size, channels, height, width = image.shape
    
    # For each image in the batch
    for i in range(batch_size):
        # Generate salt noise (white pixels)
        salt_mask = torch.rand(channels, height, width, device=image.device) < salt_prob
        corrupted_image[i][salt_mask] = 1.0
        
        # Generate pepper noise (black pixels)
        pepper_mask = torch.rand(channels, height, width, device=image.device) < pepper_prob
        corrupted_image[i][pepper_mask] = -1.0
    
    return corrupted_image

def apply_gaussian_blur(image, kernel_size=7, sigma=1.5):
    """Apply Gaussian blur to image."""
    if len(image.shape) == 3:  # Add batch dimension if needed
        image = image.unsqueeze(0)
    
    batch_size = image.shape[0]
    corrupted_image = image.clone()
    
    # Make sure kernel size is odd
    if kernel_size % 2 == 0:
        kernel_size += 1
    
    # Apply to each batch and channel
    for i in range(batch_size):
        for c in range(3):
            img_np = image[i, c].detach().cpu().numpy()
            blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
            corrupted_image[i, c] = torch.from_numpy(blurred).to(image.device)
    
    return corrupted_image

def apply_combined_corruption(image):
    """Apply a random combination of corruptions."""
    if len(image.shape) == 3:  # Add batch dimension if needed
        image = image.unsqueeze(0)
        
    corrupted_image = image.clone()
    
    # Apply 1-2 random corruptions
    num_corruptions = np.random.randint(1, 3)
    corruptions = [
        lambda x: apply_mask(x, mask_size=np.random.randint(20, 40)),
        lambda x: apply_gaussian_noise(x, std=np.random.uniform(0.05, 0.2)),
        lambda x: apply_salt_pepper_noise(x),
        lambda x: apply_gaussian_blur(x)
    ]
    
    selected_corruptions = np.random.choice(corruptions, num_corruptions, replace=False)
    
    for corruption_fn in selected_corruptions:
        corrupted_image = corruption_fn(corrupted_image)
    
    return corrupted_image

# Load pretrained model function
@st.cache_resource
def load_model(model_path):
    """Load the pretrained model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # Get latent dimension from checkpoint
        latent_dim = checkpoint.get('latent_dim', 128)
        
        # Initialize model
        model = VAE(latent_dim=latent_dim).to(device)
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, device

# Process image function
def process_image(model, image, corruption_type, device):
    """Process an image through the model."""
    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Apply corruption
    if corruption_type == "mask":
        corrupted = apply_mask(input_tensor)
    elif corruption_type == "gaussian_noise":
        corrupted = apply_gaussian_noise(input_tensor)
    elif corruption_type == "salt_pepper":
        corrupted = apply_salt_pepper_noise(input_tensor)
    elif corruption_type == "blur":
        corrupted = apply_gaussian_blur(input_tensor)
    else:  # combined
        corrupted = apply_combined_corruption(input_tensor)
    
    # Time the inference
    start_time = time.time()
    
    # Restore image
    with torch.no_grad():
        restored, _, _ = model(corrupted)
    
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Calculate metrics
    if input_tensor.shape[2:] != restored.shape[2:]:
        restored_resized = F.interpolate(restored, size=input_tensor.shape[2:], 
                                       mode='bilinear', align_corners=False)
    else:
        restored_resized = restored
        
    # Convert tensors for visualization
    def tensor_to_image(tensor):
        # Convert from [-1, 1] to [0, 1]
        img = (tensor.cpu().squeeze(0) + 1) / 2
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        return img
    
    original_img = tensor_to_image(input_tensor)
    corrupted_img = tensor_to_image(corrupted)
    restored_img = tensor_to_image(restored)
    
    # Calculate PSNR
    mse = ((original_img - restored_img) ** 2).mean()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    
    # Calculate SSIM (simplified version)
    def calculate_ssim(img1, img2):
        # Constants
        C1 = (0.01 * 1) ** 2
        C2 = (0.03 * 1) ** 2
        
        # Calculate mean, variance, covariance
        mu1 = np.mean(img1, axis=(0, 1))
        mu2 = np.mean(img2, axis=(0, 1))
        
        sigma1_sq = np.mean((img1 - mu1) ** 2, axis=(0, 1))
        sigma2_sq = np.mean((img2 - mu2) ** 2, axis=(0, 1))
        
        # Calculate covariance
        sigma12 = np.mean((img1 - mu1) * (img2 - mu2), axis=(0, 1))
        
        # SSIM formula
        numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
        denominator = (mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2)
        ssim = np.mean(numerator / denominator)
        
        return ssim
    
    ssim = calculate_ssim(original_img, restored_img)
    
    return {
        'original': original_img,
        'corrupted': corrupted_img, 
        'restored': restored_img,
        'psnr': psnr,
        'ssim': ssim,
        'inference_time': inference_time
    }

# Streamlit app
def main():
    st.set_page_config(page_title="CLEAR-VISION Image Restoration", layout="wide")
    
    st.title("CLEAR-VISION: Image Restoration App")
    st.write("""
    This app restores corrupted or degraded images using a Variational Autoencoder (VAE) model.
    Upload an image and select a corruption type to see the restoration results.
    """)
    
    # Sidebar
    st.sidebar.title("Settings")
    
    # Model selection
    model_path = st.sidebar.selectbox(
        "Select Model",
        ["./vers 3/checkpoints/vae_final_model.pth", "./vers 3/checkpoints/vae_epoch_50.pth", "./vers 3/checkpoints/vae_epoch_80.pth", "./vers 3/checkpoints/vae_epoch_10.pth"]
    )
    
    # Load the selected model
    model, device = load_model(model_path)
    
    if model is None:
        st.error("Failed to load model. Please check the model path.")
        return
    
    st.sidebar.write(f"Model loaded successfully. Running on: {device}")
    
    # Upload image
    uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    # Corruption type
    corruption_type = st.sidebar.selectbox(
        "Corruption Type",
        ["mask", "gaussian_noise", "salt_pepper", "blur", "combined"]
    )
    
    # Process image
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        
        # Create columns for original, corrupted and restored
        st.subheader("Image Restoration Results")
        col1, col2, col3 = st.columns(3)
        
        # Process the image
        results = process_image(model, image, corruption_type, device)
        
        # Display original image
        with col1:
            st.write("Original Image")
            st.image(results['original'], use_container_width=True)
        
        # Display corrupted image
        with col2:
            st.write(f"Corrupted Image ({corruption_type})")
            st.image(results['corrupted'], use_container_width=True)
        
        # Display restored image
        with col3:
            st.write("Restored Image")
            st.image(results['restored'], use_container_width=True)
        
        # Display metrics
        st.subheader("Image Quality Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("PSNR (dB)", f"{results['psnr']:.2f}")
            
        with col2:
            st.metric("SSIM", f"{results['ssim']:.4f}")
            
        with col3:
            st.metric("Inference Time (ms)", f"{results['inference_time']:.2f}")
        
        # Add download button for restored image
        buf = BytesIO()
        plt.imsave(buf, results['restored'], format='png')
        buf.seek(0)
        
        st.download_button(
            label="Download Restored Image",
            data=buf,
            file_name=f"restored_{uploaded_file.name}",
            mime="image/png"
        )
        
    else:
        # Display sample images when no file is uploaded
        st.info("Please upload an image to see restoration results.")

if __name__ == "__main__":
    main()
