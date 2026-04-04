#!/usr/bin/env python3
"""
Compute FID (Fréchet Inception Distance) for generated CIFAR-10 samples
saved by demo_cnn_rectified, using torchmetrics.image.fid.

Usage:
    python compute_fid.py [--samples-dir DIR] [--cifar-dir DIR]
                          [--real-images N] [--batch-size B]
                          [--device DEVICE] [--output PLOT.png]

Each epoch's samples are expected as:
    <samples-dir>/epoch_<k>_samples.bin
with header: int32 N, int32 C=3, int32 H=32, int32 W=32
followed by N*C*H*W float32 values in [-1,1], CHW per image.
"""

import argparse
import glob
import os
import re
import struct
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torchmetrics.image.fid import FrechetInceptionDistance


# ── Load binary samples from C++ ─────────────────────────────────────────────
def load_bin_samples(path):
    """Load raw float32 samples saved by the C++ programme.
    Returns numpy array of shape (N, 3, 32, 32) in [0, 1]."""
    with open(path, "rb") as f:
        header = struct.unpack("4i", f.read(16))
        n, c, h, w = header
        data = np.frombuffer(f.read(n * c * h * w * 4), dtype=np.float32)
    data = data.reshape(n, c, h, w)
    # Convert from [-1, 1] to [0, 1]
    data = np.clip((data + 1.0) * 0.5, 0.0, 1.0)
    return data


# ── Load real CIFAR-10 images ────────────────────────────────────────────────
def load_cifar10_real(cifar_dir, max_images=10000):
    """Load real CIFAR-10 training images from binary batches.
    Returns numpy array of shape (N, 3, 32, 32) in [0, 1]."""
    images = []
    for batch_idx in range(1, 6):
        path = os.path.join(cifar_dir, f"data_batch_{batch_idx}.bin")
        if not os.path.exists(path):
            raise FileNotFoundError(f"CIFAR-10 batch not found: {path}")
        with open(path, "rb") as f:
            buf = f.read()
        record_size = 1 + 3072  # 1 label + 3072 pixels
        n_records = len(buf) // record_size
        for i in range(n_records):
            offset = i * record_size
            pixels = np.frombuffer(buf[offset + 1 : offset + record_size],
                                   dtype=np.uint8).astype(np.float32)
            img = pixels.reshape(3, 32, 32) / 255.0
            images.append(img)
            if len(images) >= max_images:
                return np.array(images)
    return np.array(images)


def to_float_tensor(images_np):
    """Convert (N, 3, H, W) float32 in [0,1] to float tensor for torchmetrics FID."""
    return torch.from_numpy(images_np).float()


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Compute FID for rectified flow CIFAR-10 samples")
    parser.add_argument("--samples-dir", default="res/generated_cifar10",
                        help="Directory containing epoch_*_samples.bin files")
    parser.add_argument("--cifar-dir", default="data/CIFAR10/cifar-10-batches-bin",
                        help="Path to cifar-10-batches-bin")
    parser.add_argument("--real-images", type=int, default=10000,
                        help="Number of real images to use for reference statistics")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for Inception forward pass")
    parser.add_argument("--device", default="cuda",
                        help="Device for Inception model (cuda or cpu)")
    parser.add_argument("--output", default="res/generated_cifar10/fid_plot.png",
                        help="Output plot file")
    args = parser.parse_args()

    # Find all epoch sample files
    pattern = os.path.join(args.samples_dir, "epoch_*_samples.bin")
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"No sample files found matching {pattern}")
        sys.exit(1)

    # Extract epoch numbers
    epoch_files = []
    for f in files:
        m = re.search(r"epoch_(\d+)_samples\.bin", f)
        if m:
            epoch_files.append((int(m.group(1)), f))
    epoch_files.sort(key=lambda x: x[0])
    print(f"Found {len(epoch_files)} epoch sample files")

    device = args.device if torch.cuda.is_available() else "cpu"

    # Load real images and feed into FID once (shared across epochs)
    print(f"Loading {args.real_images} real CIFAR-10 images from {args.cifar_dir}...")
    real_images = load_cifar10_real(args.cifar_dir, max_images=args.real_images)
    real_tensor = to_float_tensor(real_images)
    print(f"  Loaded {len(real_images)} real images")

    # Compute FID for each epoch
    epochs = []
    fids = []
    for epoch_num, filepath in epoch_files:
        print(f"Epoch {epoch_num}: loading {filepath}...", flush=True)
        gen_images = load_bin_samples(filepath)
        gen_tensor = to_float_tensor(gen_images)

        fid_metric = FrechetInceptionDistance(feature=64, normalize=True).to(device)
        # Feed real images in batches
        for i in range(0, len(real_tensor), args.batch_size):
            fid_metric.update(real_tensor[i:i+args.batch_size].to(device), real=True)
        # Feed generated images in batches
        for i in range(0, len(gen_tensor), args.batch_size):
            fid_metric.update(gen_tensor[i:i+args.batch_size].to(device), real=False)

        fid_val = float(fid_metric.compute())
        print(f"  FID = {fid_val:.2f}", flush=True)
        epochs.append(epoch_num)
        fids.append(fid_val)
        del fid_metric

    # Save FID values to CSV
    csv_path = os.path.join(args.samples_dir, "fid_values.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,fid\n")
        for e, fid in zip(epochs, fids):
            f.write(f"{e},{fid:.4f}\n")
    print(f"\nFID values saved to {csv_path}")

    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, fids, "b-o", linewidth=2, markersize=4)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("FID", fontsize=14)
    plt.title("FID vs Epoch (Rectified Flow on CIFAR-10)", fontsize=16)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Plot saved to {args.output}")
    print(f"\nBest FID: {min(fids):.2f} at epoch {epochs[fids.index(min(fids))]}")


if __name__ == "__main__":
    main()
