import torch
from tqdm import tqdm
import os
import json
from config import *
from utils import to_device_and_dtype, clear_cache, reset_codebook_entries, is_finetune_mode
import torch.amp as amp


class Logger:
    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.metrics = {}
        
    def log_epoch(self, epoch, metrics, prefix='train'):
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = {}
            if prefix not in self.metrics[key]:
                self.metrics[key][prefix] = []
            
            self.metrics[key][prefix].append(value)
        
        with open(os.path.join(self.log_dir, f'{prefix}_metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=2)
    
    def get_latest(self, key, prefix='train'):
        if key in self.metrics and prefix in self.metrics[key]:
            return self.metrics[key][prefix][-1]
        return None


def train_vqvae(vqvae, train_loader, val_loader=None, epochs=5, lr=LR, weight_decay=WEIGHT_DECAY, 
                grad_accumulation_steps=GRAD_ACCUMULATION_STEPS, save_dir='models', device=DEVICE):
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, 'logs'))
    
    use_amp = USE_FLOAT16 and device != "cpu" and not is_finetune_mode()
    
    optimizer = torch.optim.AdamW([
        {'params': vqvae.encoder.parameters(), 'lr': lr},
        {'params': vqvae.decoder.parameters(), 'lr': lr},
        {'params': vqvae.vq._embedding.weight, 'lr': lr * 5.0}
    ], weight_decay=weight_decay)
    
    scaler = amp.GradScaler(enabled=use_amp) 
    mse_loss = torch.nn.MSELoss()
    
    vqvae = vqvae.to(device)
    
    if is_finetune_mode():
        vqvae = vqvae.float()
        for param in vqvae.parameters():
            param.data = param.data.float()
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        vqvae.train()
        train_recon_loss = 0
        train_vq_loss = 0
        train_perplexity = 0
        train_batches = 0
        
        for i, data in enumerate(tqdm(train_loader, desc="Training")):
            x = data[0] if isinstance(data, (list, tuple)) else data
            if is_finetune_mode():
                x = x.to(device, torch.float32)
            else:
                x = to_device_and_dtype(x, device)
            
            with amp.autocast(device_type=DEVICE, enabled=use_amp):
                x_recon, vq_loss, _, perplexity = vqvae(x)
                recon_loss = mse_loss(x_recon, x)
                loss = recon_loss + vq_loss
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            
            if (i + 1) % grad_accumulation_steps == 0 or (i + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(vqvae.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Only clear cache periodically to reduce overhead
                if i % 100 == 0:
                    clear_cache()
            
            # Reduce item() calls by accumulating tensors
            with torch.no_grad():
                train_recon_loss += recon_loss.detach()
                train_vq_loss += vq_loss.detach() if isinstance(vq_loss, torch.Tensor) else vq_loss
                train_perplexity += perplexity.detach() if isinstance(perplexity, torch.Tensor) else perplexity
            train_batches += 1
        
        # Convert to Python floats at the end
        train_recon_loss = (train_recon_loss / train_batches).item()
        train_vq_loss = (train_vq_loss / train_batches).item()
        train_perplexity = (train_perplexity / train_batches).item()
        
        train_metrics = {
            'reconstruction_loss': train_recon_loss,
            'vq_loss': train_vq_loss,
            'perplexity': train_perplexity,
            'total_loss': train_recon_loss + train_vq_loss
        }
        
        logger.log_epoch(epoch, train_metrics, prefix='train')
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_metrics['total_loss']:.6f}, "
              f"Recon: {train_recon_loss:.6f}, "
              f"VQ: {train_vq_loss:.6f}, "
              f"Perplexity: {train_perplexity:.2f}")
        
        if val_loader is not None:
            vqvae.eval()
            val_recon_loss = 0
            val_vq_loss = 0
            val_perplexity = 0
            val_batches = 0
            
            with torch.no_grad():
                for data in tqdm(val_loader, desc="Validation"):
                    x = data[0] if isinstance(data, (list, tuple)) else data
                    if is_finetune_mode():
                        x = x.to(device, torch.float32)
                    else:
                        x = to_device_and_dtype(x, device)
                    
                    with amp.autocast(device_type=DEVICE, enabled=use_amp):
                        x_recon, vq_loss, _, perplexity = vqvae(x)
                        recon_loss = mse_loss(x_recon, x)
                    
                    val_recon_loss += recon_loss
                    val_vq_loss += vq_loss if isinstance(vq_loss, torch.Tensor) else torch.tensor(vq_loss, device=device)
                    val_perplexity += perplexity if isinstance(perplexity, torch.Tensor) else torch.tensor(perplexity, device=device)
                    val_batches += 1
            
            # Convert to Python floats at the end
            val_recon_loss = (val_recon_loss / val_batches).item()
            val_vq_loss = (val_vq_loss / val_batches).item()
            val_perplexity = (val_perplexity / val_batches).item()
            val_total_loss = val_recon_loss + val_vq_loss
            
            val_metrics = {
                'reconstruction_loss': val_recon_loss,
                'vq_loss': val_vq_loss,
                'perplexity': val_perplexity,
                'total_loss': val_total_loss
            }
            
            logger.log_epoch(epoch, val_metrics, prefix='val')
            
            print(f"Val Loss: {val_total_loss:.6f}, "
                  f"Recon: {val_recon_loss:.6f}, "
                  f"VQ: {val_vq_loss:.6f}, "
                  f"Perplexity: {val_perplexity:.2f}")
            
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                torch.save(vqvae.state_dict(), os.path.join(save_dir, 'best_vqvae.pt'))
                print(f"New best model saved: {val_total_loss:.6f}")
        else:
            # Save checkpoint for each epoch without validation
            torch.save(vqvae.state_dict(), os.path.join(save_dir, f'vqvae_epoch_{epoch+1}.pt'))
        
        # Check and reset dead indices
        from codebook_analysis import analyze_codebook_usage
        metrics = analyze_codebook_usage(vqvae, train_loader, device=device, sample_fraction=0.1)
        dead_indices = metrics[1]['dead_indices']
        if len(dead_indices) > 0:
            print(f"Found {len(dead_indices)} dead codebook entries. Resetting...")
            vqvae = reset_codebook_entries(vqvae, dead_indices, train_loader, device=device)
    
    return vqvae


def train_transformer(transformer, vqvae, train_loader, val_loader=None, context_length=CONTEXT_LENGTH, 
                     out_length=OUT_LENGTH, epochs=5, lr=LR, weight_decay=WEIGHT_DECAY, 
                     grad_accumulation_steps=GRAD_ACCUMULATION_STEPS, save_dir='models', device=DEVICE):
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, 'logs'))
    
    optimizer = torch.optim.AdamW(transformer.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = amp.GradScaler(enabled=USE_FLOAT16 and device != "cpu")
    
    transformer = transformer.to(device)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        transformer.train()
        train_loss = 0
        train_batches = 0
        
        for i, data in enumerate(tqdm(train_loader, desc="Training")):
            x = data[0] if isinstance(data, (list, tuple)) else data
            x = to_device_and_dtype(x, device)
            y = to_device_and_dtype(y, device)
            
            with amp.autocast(device_type=DEVICE, enabled=USE_FLOAT16 and device != "cpu"):
                logits, loss = transformer(x, y)
            
            loss = loss / grad_accumulation_steps
            scaler.scale(loss).backward()
            
            if (i + 1) % grad_accumulation_steps == 0 or (i + 1 == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(transformer.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                
                # Only clear cache less frequently
                if i % 100 == 0:
                    clear_cache()
            
            with torch.no_grad():
                train_loss += loss.detach() * grad_accumulation_steps
            train_batches += 1
        
        train_loss = (train_loss / train_batches).item()
        
        train_metrics = {
            'loss': train_loss
        }
        
        logger.log_epoch(epoch, train_metrics, prefix='train')
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.6f}")
        
        if val_loader is not None:
            transformer.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for data in tqdm(val_loader, desc="Validation"):
                    x, y = data
                    x = to_device_and_dtype(x, device)
                    y = to_device_and_dtype(y, device)
                    
                    with amp.autocast(device_type=DEVICE, enabled=USE_FLOAT16 and device != "cpu"):
                        logits, loss = transformer(x, y)
                    
                    val_loss += loss
                    val_batches += 1
            
            val_loss = (val_loss / val_batches).item()
            
            val_metrics = {
                'loss': val_loss
            }
            
            logger.log_epoch(epoch, val_metrics, prefix='val')
            
            print(f"Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(transformer.state_dict(), os.path.join(save_dir, 'best_transformer.pt'))
                print(f"New best model saved: {val_loss:.6f}")
        else:
            # Save checkpoint for each epoch without validation
            torch.save(transformer.state_dict(), os.path.join(save_dir, f'transformer_epoch_{epoch+1}.pt'))
    
    return transformer, logger.metrics


def finetune_vqvae(vqvae, train_loader, val_loader=None, epochs=10, lr=0.0001, reset_every=1,
                  device=DEVICE, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    
    vqvae = vqvae.float()
    for param in vqvae.parameters():
        param.requires_grad_(True)
        param.data = param.data.float()
    
    vqvae.vq._embedding.weight.requires_grad_(True)
    vqvae.vq._embedding.weight.data = vqvae.vq._embedding.weight.data.float()
    
    # Use a more efficient optimizer configuration
    optimizer = torch.optim.AdamW([
        {'params': vqvae.encoder.parameters(), 'lr': lr * 0.5},
        {'params': vqvae.decoder.parameters(), 'lr': lr * 0.5},
        {'params': vqvae.vq._embedding.weight, 'lr': lr * 5.0}
    ], weight_decay=WEIGHT_DECAY, eps=1e-5, betas=(0.9, 0.95))
    
    mse_loss = torch.nn.MSELoss()
    
    vqvae = vqvae.to(device)
    
    # Initial codebook analysis with reduced sample size
    from codebook_analysis import analyze_codebook_usage
    active_tokens, metrics = analyze_codebook_usage(vqvae, train_loader, device=device)
    dead_indices = metrics['dead_indices']
    if len(dead_indices) > 0:
        print(f"Found {len(dead_indices)} dead codebook entries. Resetting...")
        vqvae = reset_codebook_entries(vqvae, dead_indices, train_loader, device=device)
    
    for epoch in range(epochs):
        print(f"Fine-tuning epoch {epoch+1}/{epochs}")
        
        vqvae.train()
        train_loss = torch.tensor(0.0, device=device)
        train_recon_loss = torch.tensor(0.0, device=device)
        train_vq_loss = torch.tensor(0.0, device=device)
        train_perplexity = torch.tensor(0.0, device=device)
        train_batches = 0
        
        for i, data in enumerate(tqdm(train_loader, desc="Fine-tuning")):
            try:
                x = data[0] if isinstance(data, (list, tuple)) else data
                x = x.to(device, torch.float32)
                
                # Use smaller batch size to reduce memory pressure
                if x.size(0) > 32:
                    x = x[:32]
                
                optimizer.zero_grad()
                
                x_recon, vq_loss, _, perplexity = vqvae(x)
                recon_loss = mse_loss(x_recon, x)
                loss = recon_loss + vq_loss
                
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(vqvae.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                with torch.no_grad():
                    train_loss += loss.detach()
                    train_recon_loss += recon_loss.detach()
                    train_vq_loss += vq_loss.detach() if isinstance(vq_loss, torch.Tensor) else torch.tensor(vq_loss, device=device)
                    train_perplexity += perplexity.detach() if isinstance(perplexity, torch.Tensor) else torch.tensor(perplexity, device=device)
                train_batches += 1
                
                # Clear cache less frequently (every 500 batches)
                if i % 500 == 0 and i > 0:
                    clear_cache()
                
            except Exception as e:
                print(f"Error in batch: {e}")
                continue
        
        # Convert to Python floats at the end
        train_loss = (train_loss / max(1, train_batches)).item()
        train_recon_loss = (train_recon_loss / max(1, train_batches)).item()
        train_vq_loss = (train_vq_loss / max(1, train_batches)).item()
        train_perplexity = (train_perplexity / max(1, train_batches)).item()
        
        print(f"Loss: {train_loss:.6f}, Recon: {train_recon_loss:.6f}, "
              f"VQ: {train_vq_loss:.6f}, Perplexity: {train_perplexity:.2f}")
        
        # Check codebook usage with reduced sample size
        active_tokens, metrics = analyze_codebook_usage(vqvae, train_loader, device=device)
        dead_indices = metrics['dead_indices']
        if len(dead_indices) > 0:
            print(f"Found {len(dead_indices)} dead codebook entries. Resetting...")
            vqvae = reset_codebook_entries(vqvae, dead_indices, train_loader, device=device)
            
        clear_cache()
    
    torch.save(vqvae.state_dict(), os.path.join(save_dir, 'finetuned_vqvae.pt'))
    
    return vqvae


def create_token_dataset(vqvae, data_loader, context_length=CONTEXT_LENGTH, device=DEVICE):
    vqvae.eval()
    all_indices = []
    
    with torch.no_grad():
        for i, data in enumerate(tqdm(data_loader, desc="Tokenizing data")):
            x = data[0] if isinstance(data, (list, tuple)) else data
            x = to_device_and_dtype(x, device)
            
            # Process in smaller batches if needed
            if x.size(0) > 16:
                indices_batch = []
                for j in range(0, x.size(0), 16):
                    batch_slice = x[j:j+16]
                    indices_slice = vqvae.encode(batch_slice, normalize=False)
                    indices_batch.append(indices_slice.cpu())
                indices = torch.cat(indices_batch, dim=0)
            else:
                indices = vqvae.encode(x, normalize=False).cpu()
                
            all_indices.append(indices)
            
            # Clear cache periodically
            if i % 100 == 0 and i > 0:
                clear_cache()
    
    # Concatenate all indices
    concatenated_indices = torch.cat(all_indices, dim=0)
    flat_indices = concatenated_indices.reshape(-1)
    
    if len(flat_indices) < context_length + 2:
        raise ValueError(f"Not enough tokens ({len(flat_indices)}) to create sequences of length {context_length+1}. "
                         f"Need at least {context_length+2} tokens.")
    
    # Create sliding window sequences
    sequences = []
    for i in range(0, len(flat_indices) - context_length, max(1, len(flat_indices) // 10000)):  # Use stride to limit sequence count
        seq = flat_indices[i:i+context_length]
        target = flat_indices[i+1:i+context_length+1]
        sequences.append((seq, target))
    
    return sequences