<div align="center">

# 🚀 Diffusion from Scratch

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorBoard](https://img.shields.io/badge/TensorBoard-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/tensorboard)

**A PyTorch implementation of diffusion models for image generation**

</div>

-----

## ✨ Key Features

* Generate high-quality images from random noise
* Track progress with TensorBoard visualization
* Support for both DDPM and DDIM sampling methods
* Custom dataset compatibility
* Enhanced U-Net architecture with self-attention

## 🛠️ Installation

```bash
# Clone repository
git clone https://github.com/ORI-Muchim/diffusion_from_scratch
cd diffusion_from_scratch

# Install dependencies
pip install torch torchvision numpy matplotlib tensorboard pillow
```

## 🚀 Training

Train with MNIST (default):
```bash
python train.py --use_mnist
```

Train from existing [MNIST / custom dataset] model:
```bash
python train.py [--use_mnist / --data_dir path/to/images] --resume path/to/model
```

Train with custom dataset:
```bash
python train.py --data_dir path/to/images --image_size 64 --image_channels 3
```

### Key Parameters

* `--batch_size`: Training batch size (default: 64)
* `--epochs`: Number of training epochs (default: 100)
* `--lr`: Learning rate (default: 5e-5)
* `--image_size`: Target image size (default: 28)
* `--image_channels`: Number of channels (1=grayscale, 3=RGB)
* `--sampling_method`: Method to use for sampling (ddpm or ddim)
* `--use_self_conditioning`: Enable self-conditioning

## 🖼️ Generating Images

```bash
python inference.py --model_path runs/diffusion_model_[timestamp]/diffusion_model_final.pth --sampling_method ddim
```

### Key Parameters

* `--model_path`: Path to trained model (required)
* `--batch_size`: Number of images per batch (default: 16)
* `--sampling_method`: Sampling method (ddpm or ddim)
* `--ddim_steps`: Number of steps for DDIM sampling (default: 50)

## 📊 Monitoring

TensorBoard automatically launches during training:
```
TensorBoard is running at http://localhost:6006
```

The dashboard displays:
* Loss curves
* Generated image samples
* Denoising process visualization

## 📁 Project Structure

```
diffusion-from-scratch/
├── models.py           # Model architecture
├── hparams.py          # Hyperparameter settings
├── train.py            # Training script
├── inference.py        # Image generation script
├── custom_dataset.py   # Custom dataset handler
```

## 🖼️ Using Custom Datasets

1. **Prepare images** in a single directory

2. **Run training** with the data directory:
```bash
python train.py --data_dir /path/to/images --image_size 64 --image_channels 3
```

3. **Generate images** with the trained model:
```bash
python inference.py --model_path path/to/model.pth
```

## 💡 Tips for Better Results

* For complex datasets, increase model capacity with `--base_channels 128`
* Use more epochs (100-300) for higher quality
* For faster sampling with good quality, use DDIM: `--sampling_method ddim --ddim_steps 50`
* Adjust learning rate based on your dataset (`--lr 1e-5` to `5e-5`)
