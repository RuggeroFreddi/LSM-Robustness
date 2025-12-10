import os
import numpy as np
import torch.nn.functional as F
from torchvision import datasets, transforms
import torch


# === CONFIG ===
NUM_TIMESTEPS_RATE = 200
OUTPUT_RATE = "data/mnist_rate_encoded.npz"
NUM_SAMPLES = 6000


def convert_image_to_rate_code(image_tensor, num_timesteps=300):
    """
    Converts an image tensor into a rate-coded spike train based on pixel intensities.
    """
    # Downsample to 14x14 using average pooling
    downsampled_image = F.avg_pool2d(image_tensor, kernel_size=2, stride=2)
    downsampled_image = downsampled_image.squeeze(0)  

    min_intensity = downsampled_image.min().item()
    max_intensity = downsampled_image.max().item()

    # Normalize intensities to the range [0, 1]
    if max_intensity > min_intensity:
        normalized_image = (downsampled_image - min_intensity) / (max_intensity - min_intensity)
    else:
        normalized_image = downsampled_image

    # Flatten image and apply rate coding
    pixel_values = normalized_image.view(-1).numpy()  # Shape: [196]
    spike_train = np.random.rand(196, num_timesteps) < pixel_values[:, None]

    return spike_train.astype(np.uint8)

def convert_image_to_rate_code_edges(image_tensor, num_timesteps=300):
    """
    Converts an image tensor into a rate-coded spike train based on pixel intensities.
    Now: extract edges (Sobel) -> normalize [0,1] -> rate code.
    """
    # Downsample to 14x14 using average pooling
    downsampled_image = F.avg_pool2d(image_tensor, kernel_size=2, stride=2)
    downsampled_image = downsampled_image.squeeze(0)  # shape: (1, 14, 14) atteso

    # --- Edge extraction (Sobel) ---
    x = downsampled_image.unsqueeze(0)  # (N=1, C=1, H=14, W=14)
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=x.device).view(1,1,3,3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, sobel_x, padding=1)
    gy = F.conv2d(x, sobel_y, padding=1)
    edge_mag = torch.sqrt(gx * gx + gy * gy)  # (1,1,14,14)

    # --- Normalize edge magnitude to [0, 1] ---
    # (uso max-scaling; i bordi sono >=0 quindi min=0 in pratica)
    max_val = edge_mag.max()
    if max_val > 0:
        normalized_image = edge_mag / max_val
    else:
        normalized_image = edge_mag
    normalized_image = normalized_image.squeeze(0).squeeze(0)  # (14,14)

    # Flatten image and apply rate coding (come prima)
    pixel_values = normalized_image.view(-1).cpu().numpy()  # Shape: [196]
    spike_train = np.random.rand(196, num_timesteps) < pixel_values[:, None]

    return spike_train.astype(np.uint8)

def convert_image_to_rate_code_poisson(image_tensor, num_timesteps=300, max_rate_hz=100.0, dt=1e-3):
    """
    Converts an image tensor into a rate-coded spike train based on pixel intensities.
    """
    # Downsample to 14x14 using average pooling
    downsampled_image = F.avg_pool2d(image_tensor, kernel_size=2, stride=2)
    downsampled_image = downsampled_image.squeeze(0)

    # Normalize globally to [0,1] (assume MNIST already scaled; clamp for safety)
    normalized_image = downsampled_image.clamp(0.0, 1.0)

    # Flatten image and apply Poisson-like rate coding
    pixel_values = normalized_image.view(-1).cpu().numpy()  # Shape: [196]
    rates_hz = pixel_values * max_rate_hz                   # max firing rate mapping
    p = np.clip(rates_hz * dt, 0.0, 1.0)                    # per-timestep spike prob

    spike_train = np.random.rand(196, num_timesteps) < p[:, None]
    return spike_train.astype(np.uint8)

def encode_and_save_dataset(dataset, num_samples, num_timesteps, output_path, encoding_fn):
    encoded_images = np.zeros((num_samples, 196, num_timesteps), dtype=np.uint8)
    labels = np.zeros((num_samples,), dtype=np.uint8)

    for idx in range(num_samples):
        image, label = dataset[idx]
        encoded_images[idx] = encoding_fn(image, num_timesteps)
        labels[idx] = label

        if idx % 100 == 0:
            print(f"✅ Encoded {idx} images → {output_path}")

    np.savez_compressed(output_path, data=encoded_images, labels=labels)
    print(f"🎉 Saved to '{output_path}'")

    # === Display array shapes and stats for verification ===
    print("📐 Array shapes:")
    print("  - X (encoded images):", encoded_images.shape)
    print("  - y (labels):", labels.shape)

    total_spikes = encoded_images.sum()
    avg_spikes_per_sample = total_spikes / num_samples
    avg_spike_prob = total_spikes / (num_samples * 196 * num_timesteps)

    print(f"🔢 Average spikes per sample: {avg_spikes_per_sample:.2f}")
    print(f"📊 Average spike probability per pixel per timestep: {avg_spike_prob:.6f}")


transform = transforms.ToTensor()
mnist = datasets.MNIST(root="./data", train=True, download=True, transform=transform)

# === ENCODE DATASETS ===
encode_and_save_dataset(mnist, NUM_SAMPLES, NUM_TIMESTEPS_RATE, OUTPUT_RATE, convert_image_to_rate_code)
