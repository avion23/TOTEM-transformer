import torch
import numpy as np
from tqdm import tqdm
import os
import json
from config import *
from utils import to_device_and_dtype, clear_cache


class Logger:
    def __init__(self, log_dir='logs'):
        os.makedirs(log_dir, exist_ok=True)
        self.log_dir = log_dir
        self.metrics = {'train': {}, 'val': {}}
        
    def log(self, phase, metrics, epoch=None):
        for key, value in metrics.items():
            if key not in self.metrics[phase]:
                self.metrics[phase][key] = []
            
            self.metrics[phase][key].append(value)
            
            log_path = os.path.join(self.log_dir, f'{phase}_metrics_latest.json')
            
            full_metrics = self.metrics[phase].copy()
            full_metrics['current_epoch'] = epoch
            
            with open(log_path, 'w') as f:
                json.dump(full_metrics, f)
                
            if epoch % 10 == 0 or epoch == 0:
                checkpoint_path = os.path.join(self.log_dir, f'{phase}_metrics_epoch_{epoch}.json')
                with open(checkpoint_path, 'w') as f:
                    json.dump(full_metrics, f)
    
    def save(self):
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f)


def train_vqvae(vqvae, train_loader, val_loader=None, epochs=5, learning_rate=LR, 
                weight_decay=WEIGHT_DECAY, device=DEVICE, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, 'logs'))
    
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate, weight_decay=weight_decay)
    mse_loss = torch.nn.MSELoss()
    
    vqvae = vqvae.to(device)
    
    if USE_FLOAT16 and device != "cpu":
        vqvae = vqvae.to(torch.float16)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        vqvae.train()
        train_recon_loss = 0
        train_vq_loss = 0
        train_perplexity = 0
        train_batches = 0
        
        for data in tqdm(train_loader, desc="Training"):
            x = data[0] if isinstance(data, (list, tuple)) else data
            x = to_device_and_dtype(x, device).contiguous()
            
            optimizer.zero_grad()
            
            x_recon, vq_loss, _, perplexity = vqvae(x)
            recon_loss = mse_loss(x_recon, x)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            train_recon_loss += recon_loss.item()
            train_vq_loss += vq_loss.item()
            train_perplexity += perplexity.item()
            train_batches += 1
            
            if train_batches % 50 == 0:
                clear_cache()
        
        train_recon_loss /= train_batches
        train_vq_loss /= train_batches
        train_perplexity /= train_batches
        
        train_metrics = {
            'reconstruction_loss': train_recon_loss,
            'vq_loss': train_vq_loss,
            'total_loss': train_recon_loss + train_vq_loss,
            'perplexity': train_perplexity
        }
        
        logger.log('train', train_metrics, epoch)
        
        print(f"Train Loss: {train_metrics['total_loss']:.6f}, "
              f"Recon: {train_recon_loss:.6f}, VQ: {train_vq_loss:.6f}, "
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
                    x = to_device_and_dtype(x, device).contiguous()
                    
                    x_recon, vq_loss, _, perplexity = vqvae(x)
                    recon_loss = mse_loss(x_recon, x)
                    
                    val_recon_loss += recon_loss.item()
                    val_vq_loss += vq_loss.item()
                    val_perplexity += perplexity.item()
                    val_batches += 1
            
            val_recon_loss /= val_batches
            val_vq_loss /= val_batches
            val_perplexity /= val_batches
            val_total_loss = val_recon_loss + val_vq_loss
            
            val_metrics = {
                'reconstruction_loss': val_recon_loss,
                'vq_loss': val_vq_loss,
                'total_loss': val_total_loss,
                'perplexity': val_perplexity
            }
            
            logger.log('val', val_metrics, epoch)
            
            print(f"Val Loss: {val_total_loss:.6f}, "
                  f"Recon: {val_recon_loss:.6f}, VQ: {val_vq_loss:.6f}, "
                  f"Perplexity: {val_perplexity:.2f}")
            
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                torch.save(vqvae.state_dict(), os.path.join(save_dir, 'best_vqvae.pt'))
                print(f"Saved best model with val loss {best_val_loss:.6f}")
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save(vqvae.state_dict(), os.path.join(save_dir, f'vqvae_epoch_{epoch+1}.pt'))
        
        clear_cache()
    
    torch.save(vqvae.state_dict(), os.path.join(save_dir, 'final_vqvae.pt'))
    logger.save()
    
    return vqvae, logger.metrics


def train_transformer(transformer, train_loader, val_loader=None, epochs=5, learning_rate=LR,
                     weight_decay=WEIGHT_DECAY, device=DEVICE, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, 'logs'))
    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    transformer = transformer.to(device)
    
    if USE_FLOAT16 and device != "cpu":
        transformer = transformer.to(torch.float16)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        transformer.train()
        train_loss = 0
        train_batches = 0
        
        for data in tqdm(train_loader, desc="Training"):
            x, y = data
            x = to_device_and_dtype(x, device).contiguous()
            y = to_device_and_dtype(y, device).contiguous()
            
            optimizer.zero_grad()
            
            logits, loss = transformer(x, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if train_batches % 50 == 0:
                clear_cache()
        
        train_loss /= train_batches
        
        train_metrics = {
            'loss': train_loss
        }
        
        logger.log('train', train_metrics, epoch)
        
        print(f"Train Loss: {train_loss:.6f}")
        
        if val_loader is not None:
            transformer.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for data in tqdm(val_loader, desc="Validation"):
                    x, y = data
                    x = to_device_and_dtype(x, device).contiguous()
                    y = to_device_and_dtype(y, device).contiguous()
                    
                    logits, loss = transformer(x, y)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            val_loss /= val_batches
            
            val_metrics = {
                'loss': val_loss
            }
            
            logger.log('val', val_metrics, epoch)
            
            print(f"Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(transformer.state_dict(), os.path.join(save_dir, 'best_transformer.pt'))
                print(f"Saved best model with val loss {best_val_loss:.6f}")
        
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save(transformer.state_dict(), os.path.join(save_dir, f'transformer_epoch_{epoch+1}.pt'))
        
        clear_cache()
    
    torch.save(transformer.state_dict(), os.path.join(save_dir, 'final_transformer.pt'))
    logger.save()
    
    return transformer, logger.metrics


def reset_dead_codebook_entries(vqvae, data_loader, reset_threshold=0.01, device=DEVICE, sample_fraction=0.1):
    vqvae.eval()
    
    token_counts = torch.zeros(vqvae.num_embeddings, device=device)
    total_tokens = 0
    
    with torch.no_grad():
        sample_size = int(len(data_loader) * sample_fraction)
        sampled_batches = np.random.choice(len(data_loader), sample_size, replace=False)
        
        for i, data in enumerate(tqdm(data_loader, desc="Scanning codebook")):
            if i not in sampled_batches:
                continue
                
            x = data[0] if isinstance(data, (list, tuple)) else data
            x = to_device_and_dtype(x, device).contiguous()
                
            _, _, indices, _ = vqvae(x, normalize=False)
            
            for idx in range(vqvae.num_embeddings):
                token_counts[idx] += (indices == idx).sum().item()
            
            total_tokens += indices.numel()
    
    usage_pct = token_counts / total_tokens
    
    dead_indices = torch.where(usage_pct < reset_threshold)[0]
    n_dead = len(dead_indices)
    
    if n_dead > 0:
        print(f"Found {n_dead} dead codebook entries")
        
        top_indices = torch.argsort(usage_pct, descending=True)[:n_dead]
        
        vqvae.vq._embedding.weight.requires_grad = False
        
        for i, dead_idx in enumerate(dead_indices):
            source_idx = top_indices[i % len(top_indices)]
            
            source_weight = vqvae.vq._embedding.weight.data[source_idx].clone()
            noise = torch.randn_like(source_weight) * 0.1
            vqvae.vq._embedding.weight.data[dead_idx] = source_weight + noise
            
        vqvae.vq._embedding.weight.requires_grad = True
        
        print(f"Reset {n_dead} codebook entries")
    else:
        print("No dead codebook entries found")
    
    return n_dead, vqvae


def fine_tune_codebook(vqvae, train_loader, epochs=3, learning_rate=0.0001, 
                      reset_every=1, device=DEVICE, save_dir='models'):
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)
    mse_loss = torch.nn.MSELoss()
    
    vqvae = vqvae.to(device)
    
    if USE_FLOAT16 and device != "cpu":
        vqvae = vqvae.to(torch.float16)
    
    reset_dead_codebook_entries(vqvae, train_loader, device=device, sample_fraction=0.2)
    
    for epoch in range(epochs):
        print(f"Fine-tuning epoch {epoch+1}/{epochs}")
        
        vqvae.train()
        train_loss = 0
        train_recon_loss = 0
        train_vq_loss = 0
        train_perplexity = 0
        train_batches = 0
        
        for data in tqdm(train_loader, desc="Fine-tuning"):
            try:
                x = data[0] if isinstance(data, (list, tuple)) else data
                x = to_device_and_dtype(x, device).contiguous()
                
                optimizer.zero_grad()
                
                x_recon, vq_loss, _, perplexity = vqvae(x)
                recon_loss = mse_loss(x_recon, x)
                loss = recon_loss + vq_loss
                
                try:
                    loss.backward()
                    optimizer.step()
                except RuntimeError as e:
                    if "view size is not compatible" in str(e):
                        optimizer.zero_grad()
                        clear_cache()
                        
                        x_recon_safe = x_recon.clone().detach().contiguous()
                        x_safe = x.clone().detach().contiguous()
                        
                        recon_loss = torch.nn.functional.mse_loss(x_recon_safe, x_safe)
                        
                        if isinstance(vq_loss, torch.Tensor):
                            vq_loss_val = vq_loss.item()
                        else:
                            vq_loss_val = vq_loss
                            
                        total_loss = recon_loss * (1.0 + vq_loss_val / recon_loss.item())
                        
                        total_loss.backward()
                        optimizer.step()
                    else:
                        print(f"Unrecoverable error in backward pass: {e}")
                        optimizer.zero_grad()
                        clear_cache()
                        continue
                
                if isinstance(vq_loss, torch.Tensor):
                    vq_loss_item = vq_loss.item()
                else:
                    vq_loss_item = vq_loss
                    
                train_loss += loss.item()
                train_recon_loss += recon_loss.item()
                train_vq_loss += vq_loss_item
                train_perplexity += perplexity.item()
                train_batches += 1
                
                if train_batches % 50 == 0:
                    clear_cache()
                
            except RuntimeError as e:
                print(f"Error processing batch: {e}")
                if "view size is not compatible" in str(e):
                    try:
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        clear_cache()
                        
                        x = x.clone().detach().contiguous()
                        x_recon, vq_loss, _, perplexity = vqvae(x)
                        recon_loss = mse_loss(x_recon.contiguous(), x.contiguous())
                        
                        if isinstance(vq_loss, torch.Tensor):
                            loss = recon_loss + vq_loss
                            vq_loss_item = vq_loss.item()
                        else:
                            loss = recon_loss * (1.0 + vq_loss / recon_loss.item())
                            vq_loss_item = vq_loss
                        
                        try:
                            loss.backward()
                            optimizer.step()
                        except RuntimeError:
                            optimizer.zero_grad()
                            clear_cache()
                            continue
                        
                        train_loss += loss.item()
                        train_recon_loss += recon_loss.item()
                        train_vq_loss += vq_loss_item
                        train_perplexity += perplexity.item()
                        train_batches += 1
                    except Exception:
                        optimizer.zero_grad()
                        clear_cache()
                else:
                    optimizer.zero_grad()
                    clear_cache()
                continue
        
        train_loss /= max(1, train_batches)
        train_recon_loss /= max(1, train_batches)
        train_vq_loss /= max(1, train_batches)
        train_perplexity /= max(1, train_batches)
        
        print(f"Loss: {train_loss:.6f}, Recon: {train_recon_loss:.6f}, "
              f"VQ: {train_vq_loss:.6f}, Perplexity: {train_perplexity:.2f}")
        
        clear_cache()
    
    torch.save(vqvae.state_dict(), os.path.join(save_dir, 'vqvae_fine_tuned.pt'))
    
    return vqvae


def create_token_dataset(vqvae, data_loader, context_length=CONTEXT_LENGTH, device=DEVICE):
    vqvae.eval()
    all_indices = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Tokenizing data"):
            x = data[0] if isinstance(data, (list, tuple)) else data
            x = to_device_and_dtype(x, device).contiguous()
            
            if x.shape[2] > 1000:
                indices_list = []
                for i in range(0, x.shape[2], 1000):
                    end_idx = min(i + 1000, x.shape[2])
                    chunk = x[:, :, i:end_idx].contiguous()
                    indices = vqvae.encode(chunk, normalize=False)
                    indices_list.append(indices.cpu())
                indices = torch.cat(indices_list, dim=1).contiguous()
            else:
                indices = vqvae.encode(x, normalize=False).cpu()
            
            # Store the indices for this batch
            all_indices.append(indices)
    
    # First concatenate along batch dimension
    concatenated_indices = torch.cat(all_indices, dim=0).contiguous()
    
    # Now flatten to get a single sequence of tokens
    flat_indices = concatenated_indices.reshape(-1).contiguous()
    
    # Check if we have enough tokens for training
    if len(flat_indices) <= context_length + 1:
        raise ValueError(f"Not enough tokens ({len(flat_indices)}) to create sequences of length {context_length+1}. "
                         f"Need at least {context_length+2} tokens. Consider using a larger dataset "
                         f"or reducing the context_length parameter.")
    
    sequences = []
    for i in range(len(flat_indices) - context_length):
        seq = flat_indices[i:i+context_length].contiguous()
        target = flat_indices[i+1:i+context_length+1].contiguous()
        sequences.append((seq, target))
    
    return sequences