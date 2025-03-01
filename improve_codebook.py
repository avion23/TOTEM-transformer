import torch
import numpy as np
from vqvae import VQVAE
from data import load_and_preprocess_data, normalize_data
from config import *


def reset_dead_codes(vqvae, data_norm, reset_threshold=0.01):
    device = next(vqvae.parameters()).device
    vqvae.eval()
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, _, indices, _ = vqvae(data_tensor, normalize=False)
    
    indices = indices.squeeze().cpu().numpy()
    token_counts = np.bincount(indices, minlength=CODEBOOK_SIZE)
    usage_pct = token_counts / np.sum(token_counts)
    
    active_codes = np.sum(token_counts > 0)
    print(f"Active codes before reset: {active_codes}/{CODEBOOK_SIZE} ({active_codes/CODEBOOK_SIZE*100:.2f}%)")
    
    dead_codes = np.where(usage_pct < reset_threshold)[0]
    frequent_codes = np.argsort(-token_counts)[:len(dead_codes)]
    
    print(f"Found {len(dead_codes)} dead codes to reset")
    
    if len(dead_codes) > 0:
        print("Most used codes and their percentages:")
        for i, code in enumerate(frequent_codes[:5]):
            print(f"  {i+1}. Code {code}: {usage_pct[code]*100:.2f}%")
            
        print("\nLeast used codes:")
        for i, code in enumerate(dead_codes[:5]):
            print(f"  {i+1}. Code {code}: {usage_pct[code]*100:.2f}%")
        
        vqvae.vq._embedding.weight.requires_grad = False
        
        for i, dead_idx in enumerate(dead_codes):
            source_idx = frequent_codes[i % len(frequent_codes)]
            
            source_weight = vqvae.vq._embedding.weight.data[source_idx].clone()
            noise = torch.randn_like(source_weight) * 0.1
            vqvae.vq._embedding.weight.data[dead_idx] = source_weight + noise
            
            print(f"Reset code {dead_idx} using code {source_idx} with noise")
        
        vqvae.vq._embedding.weight.requires_grad = True
        
        return True, len(dead_codes)
    
    return False, 0


def fine_tune_vqvae(vqvae, data_norm, learning_rate=0.0001, epochs=5, batch_size=32):
    device = next(vqvae.parameters()).device
    
    data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).to(device)
    
    optimizer = torch.optim.Adam(vqvae.parameters(), lr=learning_rate)
    
    n_samples = data_tensor.shape[1]
    indices = np.arange(n_samples - 100)
    
    metrics = []
    
    print(f"Fine-tuning VQVAE for {epochs} epochs with batch size {batch_size}")
    
    for epoch in range(epochs):
        vqvae.train()
        
        np.random.shuffle(indices)
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_vq_loss = 0
        epoch_perplexity = 0
        
        batches = 0
        
        for i in range(0, len(indices), batch_size):
            if i + batch_size > len(indices):
                continue
                
            batch_indices = indices[i:i+batch_size]
            x = data_tensor[:, batch_indices].unsqueeze(0)
            
            optimizer.zero_grad()
            
            x_recon, vq_loss, _, perplexity = vqvae(x, normalize=False)
            
            recon_loss = torch.mean((x_recon - x) ** 2)
            loss = recon_loss + vq_loss
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_vq_loss += vq_loss.item()
            epoch_perplexity += perplexity.item()
            
            batches += 1
        
        epoch_loss /= batches
        epoch_recon_loss /= batches
        epoch_vq_loss /= batches
        epoch_perplexity /= batches
        
        metrics.append({
            'epoch': epoch + 1,
            'loss': epoch_loss,
            'recon_loss': epoch_recon_loss,
            'vq_loss': epoch_vq_loss,
            'perplexity': epoch_perplexity
        })
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.6f}, Recon: {epoch_recon_loss:.6f}, "
              f"VQ: {epoch_vq_loss:.6f}, Perplexity: {epoch_perplexity:.2f}")
              
        if (epoch + 1) % 2 == 0:
            was_reset, num_reset = reset_dead_codes(vqvae, data_norm)
            if was_reset:
                print(f"Reset {num_reset} codes after epoch {epoch+1}")
    
    vqvae.eval()
    return vqvae, metrics


if __name__ == "__main__":
    data_path = "../../trading/discountedRewards/data/xbt-test.csv"
    
    raw_data, mean, std, features = load_and_preprocess_data(data_path, max_samples=10000)
    data_norm, _, _ = normalize_data(raw_data, mean, std)
    
    print(f"Loaded data with shape {data_norm.shape}, features: {features}")
    
    try:
        vqvae = VQVAE(in_channels=data_norm.shape[1]).to(DEVICE)
        vqvae.load_state_dict(torch.load("multifeature_vqvae.pt", map_location=DEVICE))
        print("Loaded model from multifeature_vqvae.pt")
        
        was_reset, num_reset = reset_dead_codes(vqvae, data_norm)
        
        if was_reset:
            print(f"\nResetting {num_reset} dead codes and fine-tuning model")
            
            vqvae, metrics = fine_tune_vqvae(vqvae, data_norm, epochs=3)
            
            print("Saving improved model")
            torch.save(vqvae.state_dict(), "multifeature_vqvae_improved.pt")
            
            print("\nAnalyzing codebook utilization after improvements")
            data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                _, _, indices, perplexity = vqvae(data_tensor, normalize=False)
            
            indices = indices.squeeze().cpu().numpy()
            token_counts = np.bincount(indices, minlength=CODEBOOK_SIZE)
            active_codes = np.sum(token_counts > 0)
            
            print(f"Active codes after reset: {active_codes}/{CODEBOOK_SIZE} ({active_codes/CODEBOOK_SIZE*100:.2f}%)")
            print(f"Perplexity after reset: {perplexity.item():.2f}")
        else:
            print("No dead codes found, model is already well utilized")
        
    except Exception as e:
        print(f"Error: {e}")
