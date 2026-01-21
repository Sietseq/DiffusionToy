# Diffusion Toy

![til](./gifs/DiffusionSample.gif)

Diffusion Toy is an experimental project designed to visualize how image generation works using diffusion models, specifically DDPM (Denoising Diffusion Probabilistic Models). The goal is to provide an intuitive, live view of the diffusion and denoising process rather than treating it as a black box.

The model is trained on the MNIST dataset, resulting in a small diffusion model that can generate images quickly enough to be explored interactively.

### Project Overview

- Implements a DDPM trained on MNIST digits
- Uses a U-Net–based CNN for denoising
- Visualizes the forward and reverse diffusion process in real time
- Mouse movement adds noise interactively
- Displays the current diffusion time step during denoising

This setup allows you to observe:

- How structured images emerge from Gaussian noise
- Why proper Gaussian noise is critical for diffusion models
- How insufficient noise leads to poor or unstable generations

### Interesting Observations

While the model successfully generates digits, it also produces non-digit symbols. Interestingly, these outputs still resemble plausible symbols rather than pure noise.

This behavior is reminiscent of how more advanced diffusion models sometimes struggle with text generation. The text may initially look real, but closer inspection reveals inconsistencies.

### Skills Learnt

- PyTorch
- UNet
- CNN
- Diffusion
- Noise schedulars

## Training

Much of the training code was based off:
[Diffusion Models Tutorial (YouTube)](https://www.youtube.com/watch?v=a4Yfz2FxXiY)

Some small tweaks were made:

- Added a learning rate scheduler.
- Configured to interface with grayscale instead of RGB.
- Experimented with different down and up channels.

## Viewing

The visualization is implemented using PyGame.

### How It Works

- The program starts from pure Gaussian noise
- The model gradually denoises the image step by step
- Moving the mouse adds forward noise
- A label shows the current diffusion time step

### Tips

- For best results, start around Time Step: 300
- Lower time steps often do not contain enough noise for proper reconstruction
- Adding too little noise highlights the importance of correct noise scheduling
