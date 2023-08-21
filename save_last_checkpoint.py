import torch

original = torch.load('outputs/checkpoint-48000/rng_state_7.pth')

new = {"model": original["model"]}
torch.save(new, 'outputs/checkpoint-48000/final-checkpoint-rng.pth')