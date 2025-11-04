"""
Utility functions for surrogate experiments.
"""
import torch
from utils.prepare_decoder import prepare_representation


def prepare_batch_data(sample, model_name, device):
    """
    Prepare batch data based on model type.
    
    Args:
        sample: Input sample from dataloader
        model_name: Name of the model ('HeartLang', etc.)
        device: PyTorch device
        
    Returns:
        Tuple of (x, inp) prepared for the model
    """
    if model_name == 'HeartLang':
        x = sample[0:3]
        sam = sample[0].to(device)
        x = [sam, x[1], x[2]]
        inp = sample[3].transpose(1, 2)
        # inp = torch.tensor(inp, dtype=torch.float32).to(device)
        inp = sam.to(device)
    else:
        x, inp = sample
        x = x.to(device)
        inp = inp.to(device)
    return x, inp


def forward_reconstruction(model, recon_decoder, x, model_name, layer, activations):
    """
    Forward pass through model and reconstruction decoder.
    
    Args:
        model: Pre-trained model
        recon_decoder: Reconstruction decoder
        x: Input data
        model_name: Name of the model
        layer: Layer to extract representations from
        activations: Model activations
        
    Returns:
        Reconstructed output
    """
    with torch.no_grad():
        z = prepare_representation(model, x, model_name, layer, activations)
    x_recon = recon_decoder(z)
        
    return x_recon


def print_surrogate_info(args):
    """Print surrogate experiment configuration."""
    print('-' * 50) 
    print(f'Model: {args.model}')
    print(f'Layer: {args.layer}')
    print(f'Dataset: {args.dataset}')
    print('-' * 50)
    print(f'Surrogate batch size: {args.surrogate_batch_size}')
    print(f'Surrogate epochs: {args.surrogate_epochs}')
    print(f'Surrogate # of dataset: {args.surrogate_ratio}')
    print(f'Surrogate LR: {args.surrogate_lr}')
    print('-' * 50)
    print(f'Decoder batch size: {args.batch_size}')
    print(f'Decoder epochs: {args.epochs}')
    print(f'Decoder # of dataset: {args.train_ratio}')
    print(f'Decoder LR: {args.lr}')
    print(f'Patience: {args.patience}')
    print('-' * 50)