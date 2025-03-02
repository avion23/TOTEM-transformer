import torch
import numpy as np
from config import *
from utils import to_device_and_dtype, clear_cache


def process_in_chunks(vqvae, data_tensor, chunk_size=1000, normalize=False):
    device = next(vqvae.parameters()).device
    vqvae.eval()
    
    total_indices = []
    total_recon = []
    revin_stats = []
    
    with torch.no_grad():
        for i in range(0, data_tensor.shape[2], chunk_size):
            end_idx = min(i + chunk_size, data_tensor.shape[2])
            chunk = to_device_and_dtype(data_tensor[:, :, i:end_idx])
            
            if normalize:
                chunk_norm = vqvae.revin.normalize(chunk)
                means = vqvae.revin.means
                stds = vqvae.revin.stds
                revin_stats.append((means, stds))
                x_recon, _, indices, _ = vqvae(chunk, normalize=False)
            else:
                x_recon, _, indices, _ = vqvae(chunk, normalize=False)
            
            total_indices.append(indices.cpu().numpy())
            total_recon.append(x_recon.cpu().numpy())
    
    indices = np.concatenate(total_indices, axis=1)
    
    if len(total_recon) > 0:
        recon = np.concatenate(total_recon, axis=2)
    else:
        recon = None
    
    return {
        'indices': indices,
        'reconstruction': recon,
        'revin_stats': revin_stats if normalize else None
    }


def analyze_codebook(indices, codebook_size=None, model=None):
    if model is not None:
        codebook_size = model.num_embeddings
    elif codebook_size is None:
        codebook_size = CODEBOOK_SIZE
    
    flat_indices = indices.flatten()
    
    token_counts = np.bincount(flat_indices, minlength=codebook_size)
    total_tokens = len(flat_indices)
    
    active_tokens = np.sum(token_counts > 0)
    active_pct = active_tokens / codebook_size * 100
    
    probs = token_counts / total_tokens
    nonzero_probs = probs[probs > 0]
    entropy = -np.sum(nonzero_probs * np.log2(nonzero_probs))
    max_entropy = np.log2(codebook_size)
    normalized_entropy = entropy / max_entropy * 100
    
    sorted_counts = np.sort(token_counts)[::-1]
    cumulative = np.cumsum(sorted_counts) / total_tokens * 100
    
    tokens_50pct = np.argmax(cumulative >= 50) + 1 if any(cumulative >= 50) else codebook_size
    tokens_80pct = np.argmax(cumulative >= 80) + 1 if any(cumulative >= 80) else codebook_size
    tokens_90pct = np.argmax(cumulative >= 90) + 1 if any(cumulative >= 90) else codebook_size
    tokens_95pct = np.argmax(cumulative >= 95) + 1 if any(cumulative >= 95) else codebook_size
    
    top_indices = np.argsort(-token_counts)[:10]
    top_tokens = [(idx, token_counts[idx], token_counts[idx]/total_tokens*100) 
                 for idx in top_indices]
    
    metrics = {
        'codebook_size': codebook_size,
        'active_tokens': active_tokens,
        'active_pct': active_pct,
        'unused_tokens': codebook_size - active_tokens,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'tokens_50pct': tokens_50pct,
        'tokens_80pct': tokens_80pct,
        'tokens_90pct': tokens_90pct,
        'tokens_95pct': tokens_95pct,
        'top_tokens': top_tokens
    }
    
    return metrics