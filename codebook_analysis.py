import torch
import numpy as np
from tqdm import tqdm
import math
from utils import to_device_and_dtype, clear_cache, is_finetune_mode
from config import *

def analyze_codebook_usage(vqvae, data_loader, device=DEVICE, sample_fraction=0.1, reset_threshold=0.001):
    vqvae.eval()
    
    # Pre-allocate the token counts tensor
    token_counts = torch.zeros(vqvae.num_embeddings, device=device)
    total_tokens = 0
    
    with torch.no_grad():
        # Use smaller sample fraction for efficiency
        max_samples = max(1, int(len(data_loader) * sample_fraction))
        sample_indices = np.random.choice(len(data_loader), max_samples, replace=False)
        
        for i, data in enumerate(tqdm(data_loader, desc="Analyzing codebook", total=max_samples)):
            if i not in sample_indices:
                continue
            
            x = data[0] if isinstance(data, (list, tuple)) else data
            if is_finetune_mode():
                x = x.to(device, torch.float32)
            else:
                x = to_device_and_dtype(x, device)
            
            # Process in smaller batches to reduce memory usage
            batch_size = min(16, x.size(0))
            for j in range(0, x.size(0), batch_size):
                batch = x[j:j+batch_size]
                _, _, indices, _ = vqvae(batch, normalize=False)
                
                # Use efficient tensor operations instead of item() calls
                unique_indices, counts = torch.unique(indices, return_counts=True)
                token_counts.index_add_(0, unique_indices, counts.float())
                total_tokens += indices.numel()
    
    # Convert to percentages in a vectorized way
    usage_pct = token_counts / max(1, total_tokens)
    
    # Find dead indices using efficient tensor operations
    dead_indices = torch.where(usage_pct < reset_threshold)[0].cpu().tolist()
    active_indices = torch.where(usage_pct >= reset_threshold)[0]
    
    # Calculate entropy
    probs = usage_pct[active_indices].cpu().numpy()
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    normalized_entropy = entropy / np.log2(vqvae.num_embeddings) * 100
    
    # Get top tokens
    top_indices = torch.argsort(usage_pct, descending=True)
    token_50pct = 0
    cumulative = 0
    for i, idx in enumerate(top_indices.cpu().numpy()):
        cumulative += usage_pct[idx].item()
        token_50pct = i + 1
        if cumulative >= 0.5:
            break
    
    # Format top token data for display
    top_tokens = []
    for i, idx in enumerate(top_indices[:20].cpu().numpy()):
        count = token_counts[idx].item()
        pct = usage_pct[idx].item() * 100
        top_tokens.append((int(idx), int(count), pct))
    
    print(f"\nAnalyzing codebook: Active tokens: {len(active_indices)}/{vqvae.num_embeddings}")
    
    metrics = {
        'codebook_size': vqvae.num_embeddings,
        'active_tokens': len(active_indices),
        'active_pct': len(active_indices) / vqvae.num_embeddings * 100,
        'unused_tokens': vqvae.num_embeddings - len(active_indices),
        'entropy': entropy,
        'normalized_entropy': normalized_entropy,
        'tokens_50pct': token_50pct,
        'top_tokens': top_tokens,
        'dead_indices': dead_indices
    }
    
    return len(active_indices), metrics