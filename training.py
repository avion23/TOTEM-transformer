import torch
import numpy as np
from tqdm import tqdm
import os
import json
from config import *


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
            
        # Update the latest metrics file instead of creating a new one for each epoch
        if epoch is not None:
            log_path = os.path.join(self.log_dir, f'{phase}_metrics_latest.json')
            
            # Save all metrics with epoch info
            full_metrics = self.metrics[phase].copy()
            full_metrics['current_epoch'] = epoch
            
            with open(log_path, 'w') as f:
                json.dump(full_metrics, f)
                
            # Also save a checkpoint version at specified intervals
            if epoch % 10 == 0 or epoch == 0:
                checkpoint_path = os.path.join(self.log_dir, f'{phase}_metrics_epoch_{epoch}.json')
                with open(checkpoint_path, 'w') as f:
                    json.dump(full_metrics, f)
    
    def save(self):
        with open(os.path.join(self.log_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f)


def train_vqvae(vqvae, train_loader, val_loader=None, epochs=5, learning_rate=LR, 
                weight_decay=WEIGHT_DECAY, device=DEVICE, save_dir='models'):
    """
    Train a VQVAE model.
    
    Args:
        vqvae: VQVAE model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        save_dir: Directory to save models
        
    Returns:
        Trained VQVAE model and training metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, 'logs'))
    
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate, weight_decay=weight_decay)
    mse_loss = torch.nn.MSELoss()
    
    vqvae = vqvae.to(device)
    
    # Enable float16 if configured
    if USE_FLOAT16 and device != "cpu":
        vqvae = vqvae.to(torch.float16)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        vqvae.train()
        train_recon_loss = 0
        train_vq_loss = 0
        train_perplexity = 0
        train_batches = 0
        
        for data in tqdm(train_loader, desc="Training"):
            x = data[0] if isinstance(data, (list, tuple)) else data
            x = x.to(device)
            
            # Convert to float16 if enabled
            if USE_FLOAT16 and device != "cpu":
                x = x.to(torch.float16)
            
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
        
        # Calculate average metrics
        train_recon_loss /= train_batches
        train_vq_loss /= train_batches
        train_perplexity /= train_batches
        
        train_metrics = {
            'reconstruction_loss': train_recon_loss,
            'vq_loss': train_vq_loss,
            'total_loss': train_recon_loss + train_vq_loss,
            'perplexity': train_perplexity
        }
        
        # Log training metrics
        logger.log('train', train_metrics, epoch)
        
        print(f"Train Loss: {train_metrics['total_loss']:.6f}, "
              f"Recon: {train_recon_loss:.6f}, VQ: {train_vq_loss:.6f}, "
              f"Perplexity: {train_perplexity:.2f}")
        
        # Validation
        if val_loader is not None:
            vqvae.eval()
            val_recon_loss = 0
            val_vq_loss = 0
            val_perplexity = 0
            val_batches = 0
            
            with torch.no_grad():
                for data in tqdm(val_loader, desc="Validation"):
                    x = data[0] if isinstance(data, (list, tuple)) else data
                    x = x.to(device)
                    
                    # Convert to float16 if enabled
                    if USE_FLOAT16 and device != "cpu":
                        x = x.to(torch.float16)
                    
                    x_recon, vq_loss, _, perplexity = vqvae(x)
                    recon_loss = mse_loss(x_recon, x)
                    
                    val_recon_loss += recon_loss.item()
                    val_vq_loss += vq_loss.item()
                    val_perplexity += perplexity.item()
                    val_batches += 1
            
            # Calculate average metrics
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
            
            # Log validation metrics
            logger.log('val', val_metrics, epoch)
            
            print(f"Val Loss: {val_total_loss:.6f}, "
                  f"Recon: {val_recon_loss:.6f}, VQ: {val_vq_loss:.6f}, "
                  f"Perplexity: {val_perplexity:.2f}")
            
            # Save best model
            if val_total_loss < best_val_loss:
                best_val_loss = val_total_loss
                torch.save(vqvae.state_dict(), os.path.join(save_dir, 'best_vqvae.pt'))
                print(f"Saved best model with val loss {best_val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save(vqvae.state_dict(), os.path.join(save_dir, f'vqvae_epoch_{epoch+1}.pt'))
        
        # Clear cache for MPS if available
        if device == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    # Save final model
    torch.save(vqvae.state_dict(), os.path.join(save_dir, 'final_vqvae.pt'))
    logger.save()
    
    return vqvae, logger.metrics


def train_transformer(transformer, train_loader, val_loader=None, epochs=5, learning_rate=LR,
                     weight_decay=WEIGHT_DECAY, device=DEVICE, save_dir='models'):
    """
    Train a transformer model on tokenized data.
    
    Args:
        transformer: Transformer model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data (optional)
        epochs: Number of training epochs
        learning_rate: Learning rate
        weight_decay: Weight decay
        device: Device to train on
        save_dir: Directory to save models
        
    Returns:
        Trained transformer model and training metrics
    """
    os.makedirs(save_dir, exist_ok=True)
    logger = Logger(os.path.join(save_dir, 'logs'))
    
    optimizer = torch.optim.Adam(transformer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    transformer = transformer.to(device)
    
    # Enable float16 if configured
    if USE_FLOAT16 and device != "cpu":
        transformer = transformer.to(torch.float16)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Training
        transformer.train()
        train_loss = 0
        train_batches = 0
        
        for data in tqdm(train_loader, desc="Training"):
            x, y = data
            x = x.to(device)
            y = y.to(device)
            
            # Convert to float16 if enabled
            if USE_FLOAT16 and device != "cpu":
                x = x.to(torch.float16)
                y = y.to(torch.float16)
            
            optimizer.zero_grad()
            
            logits, loss = transformer(x, y)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        # Calculate average metrics
        train_loss /= train_batches
        
        train_metrics = {
            'loss': train_loss
        }
        
        # Log training metrics
        logger.log('train', train_metrics, epoch)
        
        print(f"Train Loss: {train_loss:.6f}")
        
        # Validation
        if val_loader is not None:
            transformer.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for data in tqdm(val_loader, desc="Validation"):
                    x, y = data
                    x = x.to(device)
                    y = y.to(device)
                    
                    # Convert to float16 if enabled
                    if USE_FLOAT16 and device != "cpu":
                        x = x.to(torch.float16)
                        y = y.to(torch.float16)
                    
                    logits, loss = transformer(x, y)
                    
                    val_loss += loss.item()
                    val_batches += 1
            
            # Calculate average metrics
            val_loss /= val_batches
            
            val_metrics = {
                'loss': val_loss
            }
            
            # Log validation metrics
            logger.log('val', val_metrics, epoch)
            
            print(f"Val Loss: {val_loss:.6f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(transformer.state_dict(), os.path.join(save_dir, 'best_transformer.pt'))
                print(f"Saved best model with val loss {best_val_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            torch.save(transformer.state_dict(), os.path.join(save_dir, f'transformer_epoch_{epoch+1}.pt'))
        
        # Clear cache for MPS if available
        if device == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    # Save final model
    torch.save(transformer.state_dict(), os.path.join(save_dir, 'final_transformer.pt'))
    logger.save()
    
    return transformer, logger.metrics


def reset_dead_codebook_entries(vqvae, data_loader, reset_threshold=0.01, device=DEVICE):
    """
    Reset unused codebook entries with noisy copies of active ones.
    
    Args:
        vqvae: VQVAE model
        data_loader: DataLoader for data
        reset_threshold: Threshold for considering entries as dead (percentage)
        device: Device to run on
        
    Returns:
        Number of reset entries and updated model
    """
    vqvae.eval()
    
    token_counts = torch.zeros(vqvae.num_embeddings, device=device)
    total_tokens = 0
    
    with torch.no_grad():
        for data in data_loader:
            x = data[0] if isinstance(data, (list, tuple)) else data
            x = x.to(device)
            
            # Convert to float16 if enabled
            if USE_FLOAT16 and device != "cpu":
                x = x.to(torch.float16)
                
            _, _, indices, _ = vqvae(x, normalize=False)
            
            for idx in range(vqvae.num_embeddings):
                token_counts[idx] += (indices == idx).sum().item()
            
            total_tokens += indices.numel()
    
    usage_pct = token_counts / total_tokens
    
    dead_indices = torch.where(usage_pct < reset_threshold)[0]
    n_dead = len(dead_indices)
    
    if n_dead > 0:
        print(f"Found {n_dead} dead codebook entries")
        
        # Find most used tokens
        top_indices = torch.argsort(usage_pct, descending=True)[:n_dead]
        
        # Reset dead entries with noisy copies of active ones
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
    """
    Fine-tune VQVAE codebook to improve utilization.
    
    Args:
        vqvae: VQVAE model
        train_loader: DataLoader for training data
        epochs: Number of training epochs
        learning_rate: Learning rate
        reset_every: How often to reset dead entries (in epochs)
        device: Device to train on
        save_dir: Directory to save models
        
    Returns:
        Fine-tuned VQVAE model
    """
    os.makedirs(save_dir, exist_ok=True)
    
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)
    mse_loss = torch.nn.MSELoss()
    
    vqvae = vqvae.to(device)
    
    # Enable float16 if configured
    if USE_FLOAT16 and device != "cpu":
        vqvae = vqvae.to(torch.float16)
    
    for epoch in range(epochs):
        print(f"Fine-tuning epoch {epoch+1}/{epochs}")
        
        # Reset dead entries
        if epoch % reset_every == 0:
            reset_dead_codebook_entries(vqvae, train_loader, device=device)
        
        # Training
        vqvae.train()
        train_loss = 0
        train_recon_loss = 0
        train_vq_loss = 0
        train_perplexity = 0
        train_batches = 0
        
        for data in tqdm(train_loader, desc="Fine-tuning"):
            x = data[0] if isinstance(data, (list, tuple)) else data
            x = x.to(device)
            
            # Convert to float16 if enabled
            if USE_FLOAT16 and device != "cpu":
                x = x.to(torch.float16)
            
            optimizer.zero_grad()
            
            x_recon, vq_loss, _, perplexity = vqvae(x)
            recon_loss = mse_loss(x_recon, x)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_recon_loss += recon_loss.item()
            train_vq_loss += vq_loss.item()
            train_perplexity += perplexity.item()
            train_batches += 1
        
        # Calculate average metrics
        train_loss /= train_batches
        train_recon_loss /= train_batches
        train_vq_loss /= train_batches
        train_perplexity /= train_batches
        
        print(f"Loss: {train_loss:.6f}, Recon: {train_recon_loss:.6f}, "
              f"VQ: {train_vq_loss:.6f}, Perplexity: {train_perplexity:.2f}")
        
        # Clear cache for MPS if available
        if device == "mps" and hasattr(torch.mps, 'empty_cache'):
            torch.mps.empty_cache()
    
    # Save fine-tuned model
    torch.save(vqvae.state_dict(), os.path.join(save_dir, 'vqvae_fine_tuned.pt'))
    
    return vqvae


def create_token_dataset(vqvae, data_loader, context_length=CONTEXT_LENGTH, device=DEVICE):
    """
    Create a dataset of token indices for transformer training.
    
    Args:
        vqvae: Trained VQVAE model
        data_loader: DataLoader for data
        context_length: Context length for transformer
        device: Device to run on
        
    Returns:
        Tensor of token indices
    """
    vqvae.eval()
    all_indices = []
    
    with torch.no_grad():
        for data in tqdm(data_loader, desc="Tokenizing data"):
            x = data[0] if isinstance(data, (list, tuple)) else data
            x = x.to(device)
            
            # Convert to float16 if enabled
            if USE_FLOAT16 and device != "cpu":
                x = x.to(torch.float16)
            
            # Process in chunks if needed
            if x.shape[2] > 1000:
                indices_list = []
                for i in range(0, x.shape[2], 1000):
                    end_idx = min(i + 1000, x.shape[2])
                    chunk = x[:, :, i:end_idx]
                    indices = vqvae.encode(chunk, normalize=False)
                    indices_list.append(indices.cpu())
                indices = torch.cat(indices_list, dim=1)
            else:
                indices = vqvae.encode(x, normalize=False).cpu()
            
            all_indices.append(indices)
    
    all_indices = torch.cat(all_indices, dim=1).squeeze(0)
    
    # Create sequences for training
    sequences = []
    for i in range(len(all_indices) - context_length):
        seq = all_indices[i:i+context_length]
        target = all_indices[i+1:i+context_length+1]
        sequences.append((seq, target))
    
    return sequences