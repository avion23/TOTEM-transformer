import torch
import numpy as np
import os
from vqvae import VQVAE
from data import load_dataset
from training import train_vqvae
from config import *

# Reduced max_samples for testing
MAX_SAMPLES = 50000  

def main():
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create save directory
    save_dir = 'models_test'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'logs'), exist_ok=True)
    
    # Data file path
    data_file = '../../trading/discountedRewards/data/xbt-test.csv'
    
    # Load and preprocess data with limited samples
    print(f"Loading data from {data_file}")
    train_loader, val_loader, test_loader, dataset_info = load_dataset(
        data_file,
        batch_size=BATCH_SIZE,
        max_samples=MAX_SAMPLES
    )
    
    # Create VQVAE model for multi-feature data
    in_channels = dataset_info['normalized_data'].shape[1]
    print(f"Creating VQVAE model with {in_channels} input channels")
    
    vqvae = VQVAE(
        in_channels=in_channels,
        embedding_dim=EMBEDDING_DIM,
        num_embeddings=CODEBOOK_SIZE,
        commitment_cost=COMMITMENT_COST,
        num_hiddens=NUM_HIDDENS,
        num_residual_layers=NUM_RESIDUAL_LAYERS,
        num_residual_hiddens=NUM_RESIDUAL_HIDDENS
    )
    
    # Train for 2 epochs only
    print("Training VQVAE model...")
    vqvae, metrics = train_vqvae(
        vqvae=vqvae,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=2,
        learning_rate=LR,
        save_dir=save_dir
    )
    
    print("Training completed successfully!")

if __name__ == '__main__':
    main()
