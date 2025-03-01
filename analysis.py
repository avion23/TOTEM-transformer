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


def analyze_reconstruction(original, reconstruction, per_sample_stats=None):
    mse = np.mean((original - reconstruction)**2)
    rmse = np.sqrt(mse)
    
    if per_sample_stats is not None:
        denorm_orig = []
        denorm_recon = []
        
        for i in range(len(original)):
            mean, std = per_sample_stats[i]
            denorm_orig.append(original[i] * std + mean)
            denorm_recon.append(reconstruction[i] * std + mean)
            
        denorm_orig = np.array(denorm_orig)
        denorm_recon = np.array(denorm_recon)
        
        scale_error = np.mean((denorm_orig - denorm_recon)**2)
        scale_rmse = np.sqrt(scale_error)
    else:
        scale_error = None
        scale_rmse = None
        
    metrics = {
        'mse': mse,
        'rmse': rmse,
        'scale_error': scale_error,
        'scale_rmse': scale_rmse
    }
    
    if original.ndim > 1 and original.shape[1] > 1:
        feature_metrics = []
        for i in range(original.shape[1]):
            feat_mse = np.mean((original[:, i] - reconstruction[:, i])**2)
            feat_rmse = np.sqrt(feat_mse)
            
            if per_sample_stats is not None:
                feat_scale_error = np.mean((denorm_orig[:, i] - denorm_recon[:, i])**2)
                feat_scale_rmse = np.sqrt(feat_scale_error)
            else:
                feat_scale_error = None
                feat_scale_rmse = None
                
            feature_metrics.append({
                'mse': feat_mse,
                'rmse': feat_rmse,
                'scale_error': feat_scale_error,
                'scale_rmse': feat_scale_rmse
            })
        metrics['feature_metrics'] = feature_metrics
        
    return metrics


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
    
    sorted_norm = np.sort(probs)
    n = len(sorted_norm)
    index = np.arange(1, n + 1)
    gini = np.sum((2 * index - n - 1) * sorted_norm) / (n * np.sum(sorted_norm))
    
    metrics = {
        'codebook_size': codebook_size,
        'active_tokens': active_tokens,
        'active_pct': active_pct,
        'unused_tokens': codebook_size - active_tokens,
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'gini_coefficient': gini,
        'tokens_50pct': tokens_50pct,
        'tokens_80pct': tokens_80pct,
        'tokens_90pct': tokens_90pct,
        'tokens_95pct': tokens_95pct,
        'top_tokens': top_tokens
    }
    
    return metrics


def analyze_sequence_diversity(indices, seq_lengths=[2, 3, 4], codebook_size=None, model=None):
    if model is not None:
        codebook_size = model.num_embeddings
    elif codebook_size is None:
        codebook_size = CODEBOOK_SIZE
    
    flat_indices = indices.flatten()
    
    metrics = {}
    
    for seq_len in seq_lengths:
        sequences = []
        for i in range(len(flat_indices) - seq_len + 1):
            sequences.append(tuple(flat_indices[i:i+seq_len]))
        
        unique_sequences = set(sequences)
        total_sequences = len(sequences)
        
        diversity = len(unique_sequences) / total_sequences * 100
        max_possible = min(total_sequences, codebook_size ** seq_len)
        normalized_diversity = len(unique_sequences) / max_possible * 100
        
        sequence_counts = {}
        for seq in sequences:
            if seq in sequence_counts:
                sequence_counts[seq] += 1
            else:
                sequence_counts[seq] = 1
        
        sorted_seqs = sorted(sequence_counts.items(), key=lambda x: x[1], reverse=True)
        top_sequences = [(seq, count, count/total_sequences*100) 
                        for seq, count in sorted_seqs[:5]]
        
        seq_probs = np.array([count/total_sequences for _, count in sequence_counts.items()])
        seq_entropy = -np.sum(seq_probs * np.log2(seq_probs))
        
        metrics[f'length_{seq_len}'] = {
            'total_sequences': total_sequences,
            'unique_sequences': len(unique_sequences),
            'diversity': diversity,
            'normalized_diversity': normalized_diversity,
            'sequence_entropy': seq_entropy,
            'top_sequences': top_sequences
        }
    
    return metrics


def evaluate_vqvae(vqvae, data_tensor, normalize=True, chunk_size=1000):
    results = process_in_chunks(vqvae, data_tensor, chunk_size=chunk_size, normalize=normalize)
    indices = results['indices']
    recon = results['reconstruction']
    
    recon_metrics = analyze_reconstruction(
        data_tensor.cpu().numpy().transpose(0, 2, 1).squeeze(),
        recon.transpose(0, 2, 1).squeeze(),
        results['revin_stats']
    )
    
    codebook_metrics = analyze_codebook(indices, model=vqvae)
    
    sequence_metrics = analyze_sequence_diversity(indices, model=vqvae)
    
    clear_cache()
    
    return {
        'reconstruction': recon_metrics,
        'codebook': codebook_metrics,
        'sequence': sequence_metrics
    }
