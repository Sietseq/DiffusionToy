import sys

import numpy as np
import pygame
from pygame.math import Vector2
import torch
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.cm as cm

import Unet


IMG_SIZE = 32

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        transforms.Lambda(lambda t: (t + 1) / 2),
        transforms.Lambda(lambda t: t.permute(1, 2, 0)),  # CHW -> HWC
        transforms.Lambda(lambda t: t * 255.0),
        transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    return reverse_transforms(image)


CMAP = cm.get_cmap("Blues")  # try: "viridis", "plasma", "inferno", "turbo"

def tensor_to_rgb_surface(x, size=(500, 500)):
    x = x.detach().cpu().squeeze()
    x = x.clamp(-1, 1)

    # normalize to [0,1]
    x01 = ((x + 1) * 0.5).numpy()

    # apply colormap
    rgba = CMAP(x01)

    # convert to uint8 RGB
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    # pygame wants (W,H,3)
    surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
    surf = pygame.transform.scale(surf, size)
    return surf


# -----------------------------
# Diffusion utilities
# -----------------------------

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Takes an image and a timestep as input and
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return (
        sqrt_alphas_cumprod_t.to(device) * x_0.to(device)
        + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device),
        noise.to(device),
    )


# -----------------------------
# Diffusion schedule (precompute)
# -----------------------------

T = 300
betas = linear_beta_schedule(timesteps=T)

alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)


# -----------------------------
# Sampling step
# -----------------------------

@torch.no_grad()
def sample_timestep(x, t):
    """
    Calls the model to predict the noise in the image and returns
    the denoised image.
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(index, image):
   t = torch.full((1,), index, device=device, dtype=torch.long)
   return sample_timestep(image, t)   # no clamp, no pygame, no wait

# -----------------------------
# App / main loop
# -----------------------------

# Pygame specific setup
pygame.init()
screen = pygame.display.set_mode((500, 500))
pygame.display.set_caption("Hello World")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 48)

# Create model
model = Unet.SimpleUnet()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.load_state_dict(torch.load("./model/model", weights_only=True))

# Initialize image settings
img = torch.randn((1, 1, IMG_SIZE, IMG_SIZE), device=device)
i = T - 1

# Keep track of mouse position
last_mouse_pos = Vector2(0, 0)

# Main game loop
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

    mouse_pos = Vector2(pygame.mouse.get_pos())

    # If mouse has been moving add forward diffusion
    if mouse_pos.distance_to(last_mouse_pos) > 5:
        if i < T - 2:
            i += 2
            t = torch.tensor([i], dtype=torch.long, device=device)
            img = forward_diffusion_sample(img, t, device)[0]

    # Else diffuse backwards
    else:
        if i > 0:
            img = sample_plot_image(i, img)
            i -= 1

    last_mouse_pos = mouse_pos

    # Clear screen
    screen.fill((0, 0, 0))

    # Draw image
    vis = img.clamp(-1, 1)
    surf = tensor_to_rgb_surface(vis, size=(500, 500))
    screen.blit(surf, (0, 0))

    # Draw index 
    text_surface = font.render(f"Time Step: {i:03d}", False, (255, 0, 0))
    screen.blit(text_surface, (10, 10))

    pygame.display.flip()
    clock.tick(60)