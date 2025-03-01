import torch
import numpy as np
import os
from vqvae import VQVAE
from data import load_and_preprocess_data, normalize_data
from analysis import analyze_codebook, analyze_reconstruction, process_in_chunks, analyze_sequence_diversity, evaluate_vqvae
from config import *

np.random.seed(42)
torch.manual_seed(42)


def test_reconstruction(vqvae, data_norm, n_samples=100):
    """Test reconstruction quality of the VQVAE model."""
    device = next(vqvae.parameters()).device
    vqvae.eval()
    
    n_samples = min(n_samples, data_norm.shape[0])
    data_subset = data_norm
    
    data_tensor = torch.tensor(data_subset, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Use normalize=True to test both encoding+decoding with RevIN scale preservation
        x_recon, vq_loss, indices, perplexity = vqvae(data_tensor, normalize=True)
    
    recon = x_recon.squeeze().transpose(0, 1).cpu().numpy()
    
    metrics = analyze_reconstruction(data_subset, recon)
    
    print(f"Reconstruction MSE: {metrics['mse']:.6f}")
    print(f"Reconstruction RMSE: {metrics['rmse']:.6f}")
    
    if 'feature_metrics' in metrics:
        for i, feat_metrics in enumerate(metrics['feature_metrics']):
            print(f"Feature {i} - MSE: {feat_metrics['mse']:.6f}, RMSE: {feat_metrics['rmse']:.6f}")
    
    return metrics


def test_codebook_diversity(vqvae, data_norm):
    """Test codebook diversity and utilization."""
    device = next(vqvae.parameters()).device
    vqvae.eval()
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
    
    # Use normalize=False for codebook analysis since we want consistent token mappings
    results = process_in_chunks(vqvae, data_tensor, chunk_size=1000, normalize=False)
    indices = results['indices']
    
    metrics = analyze_codebook(indices, model=vqvae)
    
    print(f"\nCodebook Analysis:")
    print(f"Active tokens: {metrics['active_tokens']}/{metrics['codebook_size']} ({metrics['active_pct']:.2f}%)")
    print(f"Entropy: {metrics['entropy']:.4f} bits (normalized: {metrics['normalized_entropy']:.2f}%)")
    print(f"Gini coefficient: {metrics['gini_coefficient']:.4f}")
    
    print("\nTop 10 tokens by usage:")
    for i, (token_idx, count, pct) in enumerate(metrics['top_tokens']):
        print(f"  {i+1}. Token {token_idx}: {count} occurrences ({pct:.2f}%)")
    
    print("\nCumulative token usage:")
    print(f"  50% of data encoded using {metrics['tokens_50pct']} tokens")
    print(f"  80% of data encoded using {metrics['tokens_80pct']} tokens")
    print(f"  90% of data encoded using {metrics['tokens_90pct']} tokens")
    print(f"  95% of data encoded using {metrics['tokens_95pct']} tokens")
    
    return metrics


def test_sequence_diversity(vqvae, data_norm, seq_lengths=[2, 3, 4]):
    """Test sequence diversity in token sequences."""
    device = next(vqvae.parameters()).device
    vqvae.eval()
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
    
    # Use normalize=False for codebook analysis since we want consistent token mappings
    results = process_in_chunks(vqvae, data_tensor, chunk_size=1000, normalize=False)
    indices = results['indices']
    
    metrics = analyze_sequence_diversity(indices, seq_lengths=seq_lengths, model=vqvae)
    
    print("\nSequence Diversity Analysis:")
    
    for seq_len, seq_metrics in metrics.items():
        print(f"{seq_len.replace('_', ' ')}:")
        print(f"  Total sequences: {seq_metrics['total_sequences']}")
        print(f"  Unique sequences: {seq_metrics['unique_sequences']}")
        print(f"  Diversity: {seq_metrics['diversity']:.2f}%")
        print(f"  Normalized diversity: {seq_metrics['normalized_diversity']:.2f}%")
        print(f"  Sequence entropy: {seq_metrics['sequence_entropy']:.2f} bits")
        
        print(f"  Top 5 most common sequences:")
        for i, (seq, count, pct) in enumerate(seq_metrics['top_sequences']):
            print(f"    {i+1}. {seq}: {count} occurrences ({pct:.2f}%)")
    
    return metrics


def evaluate_model(vqvae, data_norm):
    """Perform comprehensive model evaluation."""
    print("=== VQVAE Model Evaluation ===")
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(next(vqvae.parameters()).device)
    
    all_metrics = evaluate_vqvae(vqvae, data_tensor, normalize=True, chunk_size=1000)
    
    print("\n=== Summary Statistics ===")
    print(f"Reconstruction MSE: {all_metrics['reconstruction']['mse']:.6f}")
    print(f"Active codebook tokens: {all_metrics['codebook']['active_tokens']} ({all_metrics['codebook']['active_pct']:.2f}%)")
    print(f"Codebook entropy: {all_metrics['codebook']['entropy']:.4f} bits")
    
    for seq_len, seq_metrics in all_metrics['sequence'].items():
        print(f"{seq_len.replace('_', ' ')} diversity: {seq_metrics['diversity']:.2f}%")
    
    return all_metrics


if __name__ == "__main__":
    data_path = "../../trading/discountedRewards/data/xbt-test.csv"
    
    if not os.path.exists(data_path):
        print(f"Error: Data file {data_path} not found")
        exit(1)
    
    try:
        raw_data, mean, std, features = load_and_preprocess_data(data_path, max_samples=10000)
        data_norm, _, _ = normalize_data(raw_data, mean, std)
        
        print(f"Loaded data with shape {data_norm.shape}, features: {features}")
        
        model_path = "multifeature_vqvae.pt"
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found")
            exit(1)
        
        vqvae = VQVAE(in_channels=data_norm.shape[1]).to(DEVICE)
        vqvae.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("Loaded model from multifeature_vqvae.pt")
        
        print("\n=== Reconstruction Test ===")
        recon_metrics = test_reconstruction(vqvae, data_norm)
        
        print("\n=== Codebook Diversity Analysis ===")
        codebook_metrics = test_codebook_diversity(vqvae, data_norm)
        
        print("\n=== Sequence Diversity Analysis ===")
        sequence_metrics = test_sequence_diversity(vqvae, data_norm)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
