import torch
import os
from config import *

FLOAT_PRECISION = torch.float16 if USE_FLOAT16 and DEVICE != "cpu" else torch.float32

def to_device_and_dtype(tensor, device=DEVICE):
    if isinstance(tensor, torch.nn.Module):
        return tensor.to(device)
    
    return tensor.to(device, FLOAT_PRECISION)

def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    import gc
    gc.collect()

def process_in_chunks(model, data_tensor, process_fn, chunk_size=1000, threshold=32768):
    if data_tensor.size(0) * data_tensor.size(1) < threshold:
        return process_fn(model, data_tensor)
    
    outputs = []
    vq_losses = []
    indices_list = []
    perplexities = []
    
    batch_size = max(1, threshold // data_tensor.size(1))
    
    for i in range(0, data_tensor.size(0), batch_size):
        end_idx = min(i + batch_size, data_tensor.size(0))
        chunk = data_tensor[i:end_idx]
        
        if chunk.size(0) * chunk.size(1) > threshold // 2:
            clear_cache()
            
        x_recon, vq_loss, indices, perplexity = process_fn(model, chunk)
        
        outputs.append(x_recon)
        if isinstance(vq_loss, torch.Tensor):
            vq_losses.append(vq_loss)
        else:
            vq_losses.append(torch.tensor(vq_loss, device=data_tensor.device, requires_grad=True))
        indices_list.append(indices)
        perplexities.append(perplexity)
    
    combined_output = torch.cat(outputs, dim=0)
    
    if all(isinstance(vl, torch.Tensor) for vl in vq_losses):
        combined_vq_loss = torch.stack(vq_losses).mean()
    else:
        vq_loss_val = sum(vl.item() if isinstance(vl, torch.Tensor) else vl for vl in vq_losses) / len(vq_losses)
        combined_vq_loss = torch.tensor(vq_loss_val, device=data_tensor.device, requires_grad=True)
    
    combined_indices = torch.cat(indices_list, dim=0)
    perplexity_val = sum(p.item() if isinstance(p, torch.Tensor) else p for p in perplexities) / len(perplexities)
    combined_perplexity = torch.tensor(perplexity_val, device=data_tensor.device)
    
    return combined_output, combined_vq_loss, combined_indices, combined_perplexity

def reset_codebook_entries(vqvae, dead_indices, source_indices=None, train_loader=None, device=DEVICE):
    if len(dead_indices) == 0:
        return vqvae
    
    vqvae.vq._embedding.requires_grad_(False)
    
    if train_loader is not None:
        try:
            data_iter = iter(train_loader)
            data = next(data_iter)[0]
            data = to_device_and_dtype(data, device)
            
            with torch.no_grad():
                z = vqvae.encoder(data)
                z_flattened = z.reshape(-1, vqvae.embedding_dim)
                
                for dead_idx in dead_indices:
                    distances = torch.cdist(z_flattened, vqvae.vq._embedding.weight)
                    max_distance_idx = torch.argmax(torch.min(distances, dim=1)[0])
                    source_weight = z_flattened[max_distance_idx].clone()
                    
                    vqvae.vq._embedding.weight.data[dead_idx] = source_weight
        except (StopIteration, RuntimeError) as e:
            print(f"Warning during train_loader processing: {e}")
            if source_indices is not None:
                for i, dead_idx in enumerate(dead_indices):
                    source_idx = source_indices[i % len(source_indices)]
                    source_weight = vqvae.vq._embedding.weight.data[source_idx].clone()
                    noise = torch.randn_like(source_weight) * 0.1
                    vqvae.vq._embedding.weight.data[dead_idx] = source_weight + noise
    elif source_indices is not None:
        for i, dead_idx in enumerate(dead_indices):
            source_idx = source_indices[i % len(source_indices)]
            source_weight = vqvae.vq._embedding.weight.data[source_idx].clone()
            noise = torch.randn_like(source_weight) * 0.1
            vqvae.vq._embedding.weight.data[dead_idx] = source_weight + noise
    
    vqvae.vq._embedding.requires_grad_(True)
    
    return vqvae