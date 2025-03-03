import torch
import os
from config import *

FLOAT_PRECISION = torch.float16 if USE_FLOAT16 and DEVICE != "cpu" else torch.float32

def is_finetune_mode():
    import sys
    return 'train_totem.py' in sys.argv[0] and '--mode' in sys.argv and 'finetune' in sys.argv

def to_device_and_dtype(tensor, device=DEVICE):
    if isinstance(tensor, torch.nn.Module):
        return tensor.to(device)
    
    dtype = FLOAT_PRECISION
    if is_finetune_mode():
        dtype = torch.float32
    
    return tensor.to(device, dtype)

def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if DEVICE == "mps" and torch.backends.mps.is_available():
        torch.mps.empty_cache()
    
    # Only invoke GC when really needed and less frequently
    if is_finetune_mode():
        import gc
        gc.collect()

def process_in_chunks(model, data_tensor, process_fn, chunk_size=16, threshold=32768):
    # Skip chunking for small tensors
    if data_tensor.shape[0] * data_tensor.shape[1] < threshold:
        return process_fn(model, data_tensor)
    
    # Pre-allocate result lists
    outputs = []
    vq_losses = []
    indices_list = []
    perplexities = []
    
    # Calculate optimal batch size based on tensor dimensions
    batch_size = min(chunk_size, max(1, threshold // data_tensor.shape[1]))
    
    # Process in chunks
    for i in range(0, data_tensor.shape[0], batch_size):
        end_idx = min(i + batch_size, data_tensor.shape[0])
        chunk = data_tensor[i:end_idx]
            
        # Process chunk
        with torch.no_grad():
            x_recon, vq_loss, indices, perplexity = process_fn(model, chunk)
        
        # Store results
        outputs.append(x_recon)
        vq_losses.append(vq_loss.detach())
        indices_list.append(indices)
        perplexities.append(perplexity)
    
    # Combine results
    combined_output = torch.cat(outputs, dim=0)
    combined_vq_loss = torch.stack(vq_losses).mean()
    combined_indices = torch.cat(indices_list, dim=0)
    
    # Calculate perplexity mean without item() calls
    with torch.no_grad():
        combined_perplexity = sum(perplexities) / len(perplexities)
    
    return combined_output, combined_vq_loss, combined_indices, combined_perplexity

def reset_codebook_entries(vqvae, dead_indices, train_loader, device=DEVICE):
    # Skip if no dead indices
    if len(dead_indices) == 0:
        return vqvae

    # Disable gradient for embedding
    vqvae.vq._embedding.requires_grad_(False)

    # Get sample data
    try:
        data_iter = iter(train_loader)
        data = next(data_iter)[0]
        if is_finetune_mode():
            data = data.to(device, torch.float32)
        else:
            data = to_device_and_dtype(data, device)
            
        # Limit batch size to reduce memory usage
        if data.size(0) > 16:
            data = data[:16]
    except StopIteration:
        return vqvae

    # Ensure float32 for embedding weights
    vqvae.vq._embedding.weight.data = vqvae.vq._embedding.weight.data.float()
    
    with torch.no_grad():
        # Generate encoder output
        z = vqvae.encoder(data).float()
        z_flattened = z.reshape(-1, vqvae.embedding_dim)
        
        # Calculate distance matrix once (major optimization)
        distances = torch.cdist(z_flattened, vqvae.vq._embedding.weight)
        
        # Reset dead codebook entries
        if len(dead_indices) > 0:
            # Find good candidates from encoder output
            min_distances, _ = torch.min(distances, dim=1)
            
            # Process all dead indices at once using vectorized operations
            # Find max distance indices for each dead index (in batches)
            batch_size = min(50, len(dead_indices))
            for i in range(0, len(dead_indices), batch_size):
                batch_indices = dead_indices[i:i+batch_size]
                max_distance_indices = torch.topk(min_distances, len(batch_indices))[1]
                
                # Update dead codebook entries with these values
                for j, dead_idx in enumerate(batch_indices):
                    vqvae.vq._embedding.weight.data[dead_idx] = z_flattened[max_distance_indices[j]].clone()

    # Re-enable gradient
    vqvae.vq._embedding.requires_grad_(True)
    return vqvae