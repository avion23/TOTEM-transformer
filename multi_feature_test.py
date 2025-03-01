import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from vqvae import VQVAE, RevIN
from data import load_and_preprocess_data, normalize_data
from config import *


def train_multifeature_vqvae(data_norm, epochs=3):
    in_channels = data_norm.shape[1]
    device = DEVICE
    
    vqvae = VQVAE(
        in_channels=in_channels,
        embedding_dim=EMBEDDING_DIM, 
        num_embeddings=CODEBOOK_SIZE,
        commitment_cost=COMMITMENT_COST,
        num_hiddens=NUM_HIDDENS,
        num_residual_layers=NUM_RESIDUAL_LAYERS,
        num_residual_hiddens=NUM_RESIDUAL_HIDDENS
    ).to(device)
    
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=LR)
    mse_loss = torch.nn.MSELoss()
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
    
    print(f"Starting VQVAE training, data shape: {data_tensor.shape}")
    
    for epoch in range(epochs):
        start_time = time.time()
        vqvae.train()
        
        optimizer.zero_grad()
        # Pass normalize=False because data_norm is already normalized
x_recon, vq_loss, _, perplexity = vqvae(data_tensor, normalize=False)
        recon_loss = mse_loss(x_recon, data_tensor)
        loss = recon_loss + vq_loss
        
        loss.backward()
        optimizer.step()
        
        duration = time.time() - start_time
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}, Recon Loss: {recon_loss.item():.6f}, "
              f"VQ Loss: {vq_loss.item():.6f}, Perplexity: {perplexity.item():.2f}, Time: {duration:.2f}s")
    
    return vqvae


def analyze_tokens(vqvae, data_norm):
    device = next(vqvae.parameters()).device
    vqvae.eval()
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        x_recon, vq_loss, indices, perplexity = vqvae(data_tensor, normalize=False)
    
    recon = x_recon.squeeze().transpose(0, 1).cpu().numpy()
    indices = indices.squeeze().cpu().numpy()
    
    unique_tokens = np.unique(indices)
    
    print(f"\nToken Analysis:")
    print(f"Unique tokens: {len(unique_tokens)} out of {CODEBOOK_SIZE} ({len(unique_tokens)/CODEBOOK_SIZE*100:.2f}%)")
    print(f"Perplexity: {perplexity.item():.2f}")
    
    token_counts = np.bincount(indices, minlength=CODEBOOK_SIZE)
    sorted_indices = np.argsort(-token_counts)
    
    print(f"Top 15 tokens by usage:")
    total_tokens = len(indices)
    cumulative_pct = 0
    for i, token in enumerate(sorted_indices[:15]):
        pct = token_counts[token] / total_tokens * 100
        cumulative_pct += pct
        print(f"  {i+1}. Token {token}: {token_counts[token]} occurrences "
              f"({pct:.2f}%, cumulative: {cumulative_pct:.2f}%)")
    
    # Calculate unused tokens
    unused = np.sum(token_counts == 0)
    print(f"Unused tokens: {unused} ({unused/CODEBOOK_SIZE*100:.2f}%)")
    
    # Calculate reconstruction error
    mse = np.mean((data_norm - recon)**2)
    print(f"Reconstruction MSE: {mse:.6f}")
    
    # Analyze sequences
    print("\nSequence analysis:")
    sequences = []
    for i in range(len(indices) - 2):
        sequences.append(tuple(indices[i:i+3]))
    
    unique_sequences = set(sequences)
    print(f"Unique 3-token sequences: {len(unique_sequences)} out of {len(sequences)} total")
    print(f"Sequence diversity: {len(unique_sequences)/len(sequences)*100:.2f}%")
    
    return {
        'indices': indices,
        'reconstruction': recon,
        'mse': mse,
        'perplexity': perplexity.item(),
        'unique_tokens': len(unique_tokens),
        'unused_tokens': unused,
        'token_counts': token_counts
    }


def plot_reconstructions(data_norm, recon, features, n_samples=200):
    # Take a subset to visualize
    start_idx = np.random.randint(0, data_norm.shape[0] - n_samples) if data_norm.shape[0] > n_samples else 0
    data_subset = data_norm[start_idx:start_idx+n_samples]
    recon_subset = recon[start_idx:start_idx+n_samples]
    
    fig, axes = plt.subplots(len(features), 1, figsize=(12, 3*len(features)))
    if len(features) == 1:
        axes = [axes]
        
    for i, feature in enumerate(features):
        axes[i].plot(data_subset[:, i], label='Original', color='blue', alpha=0.7)
        axes[i].plot(recon_subset[:, i], label='Reconstruction', color='red', linestyle='--', alpha=0.7)
        axes[i].set_title(f'{feature} Original vs Reconstruction')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reconstructions.png')
    print("Reconstruction plot saved as 'reconstructions.png'")
    
    # Plot distribution of token usage
    plt.figure(figsize=(12, 6))
    token_counts = np.bincount(indices, minlength=CODEBOOK_SIZE)
    sorted_counts = np.sort(token_counts)[::-1]
    plt.bar(range(len(sorted_counts)), sorted_counts)
    plt.title('Token Usage Distribution')
    plt.xlabel('Token Rank')
    plt.ylabel('Count')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('token_distribution.png')
    print("Token distribution plot saved as 'token_distribution.png'")


if __name__ == "__main__":
    data_path = "../../trading/discountedRewards/data/xbt-test.csv"
    
    # Load and preprocess data
    raw_data, mean, std, features = load_and_preprocess_data(data_path, max_samples=20000)
    data_norm, _, _ = normalize_data(raw_data, mean, std)
    
    print(f"Loaded data with shape {data_norm.shape}, features: {features}")
    
    # Train or load VQVAE
    train_new = input("Train new VQVAE model? (y/n): ").lower() == 'y'
    
    if train_new:
        start_time = time.time()
        vqvae = train_multifeature_vqvae(data_norm, epochs=5)
        print(f"Training completed in {time.time() - start_time:.2f}s")
        
        # Save model
        torch.save(vqvae.state_dict(), "multifeature_vqvae.pt")
        print("Model saved as multifeature_vqvae.pt")
    else:
        vqvae = VQVAE(in_channels=data_norm.shape[1]).to(DEVICE)
        try:
            vqvae.load_state_dict(torch.load("multifeature_vqvae.pt", map_location=DEVICE))
            print("Loaded model from multifeature_vqvae.pt")
        except:
            print("Could not load model, training new one...")
            vqvae = train_multifeature_vqvae(data_norm, epochs=5)
            torch.save(vqvae.state_dict(), "multifeature_vqvae.pt")
    
    # Analyze model
    results = analyze_tokens(vqvae, data_norm)
    indices = results['indices']
    recon = results['reconstruction'] 
    
    # Plot reconstructions
    plot_reconstructions(data_norm, recon, features)
