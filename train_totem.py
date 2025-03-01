import argparse
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from vqvae import VQVAE
from nanogpt import NanoGPT
from data import load_dataset
from training import train_vqvae, train_transformer, fine_tune_codebook, create_token_dataset
from analysis import analyze_codebook, analyze_reconstruction
from utils import to_device_and_dtype, clear_cache
from config import *

def main():
    parser = argparse.ArgumentParser(description='TOTEM Training')
    parser.add_argument('--data', type=str, default='../../trading/discountedRewards/data/xbt-test.csv')
    parser.add_argument('--mode', type=str, choices=['vqvae', 'transformer', 'finetune'], default='vqvae')
    parser.add_argument('--model_type', type=str, choices=['single', 'multi'], default='multi')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--vqvae_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='models')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_samples', type=int, default=None)
    
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)
    
    # Check if data file exists
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found")
        return
    
    # Load and preprocess data
    print(f"Loading data from {args.data}")
    train_loader, val_loader, test_loader, dataset_info = load_dataset(
        args.data,
        batch_size=args.batch_size,
        max_samples=args.max_samples
    )
    
    raw_data = dataset_info['raw_data']
    data_norm = dataset_info['normalized_data']
    mean = dataset_info['mean']
    std = dataset_info['std']
    features = dataset_info['features']
    
    print(f"Data shape: {data_norm.shape}, Features: {features}")
    
    # Number of input channels
    in_channels = 1 if args.model_type == 'single' else data_norm.shape[1]
    
    if args.mode == 'vqvae':
        # Create and train VQVAE
        vqvae = VQVAE(
            in_channels=in_channels,
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=CODEBOOK_SIZE,
            commitment_cost=COMMITMENT_COST,
            num_hiddens=NUM_HIDDENS,
            num_residual_layers=NUM_RESIDUAL_LAYERS,
            num_residual_hiddens=NUM_RESIDUAL_HIDDENS
        )
        
        print("Training VQVAE model...")
        vqvae, metrics = train_vqvae(
            vqvae=vqvae,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            save_dir=args.save_dir
        )
        
        # Evaluate model on test set
        vqvae.eval()
        data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).contiguous().unsqueeze(0)
        data_tensor = to_device_and_dtype(data_tensor)
        
        with torch.no_grad():
            # Process in chunks to avoid MPS limitations
            if data_tensor.shape[2] > 1000:
                chunks = []
                indices_list = []
                for i in range(0, data_tensor.shape[2], 1000):
                    end = min(i + 1000, data_tensor.shape[2])
                    chunk = data_tensor[:, :, i:end]
                    chunk_recon, _, chunk_indices, _ = vqvae(chunk, normalize=True)
                    chunks.append(chunk_recon.cpu())
                    indices_list.append(chunk_indices.cpu())
                x_recon = torch.cat(chunks, dim=2)
                indices = torch.cat(indices_list, dim=1)
            else:
                x_recon, _, indices, perplexity = vqvae(data_tensor, normalize=True)
                x_recon = x_recon.cpu()
                indices = indices.cpu()
        
        recon = x_recon.squeeze().transpose(0, 1).contiguous().numpy()
        indices = indices.squeeze().numpy()
        
        # Analyze reconstruction quality
        recon_metrics = analyze_reconstruction(data_norm, recon)
        print(f"Reconstruction MSE: {recon_metrics['mse']:.6f}")
        if recon_metrics['scale_error'] is not None:
            print(f"Scale error: {recon_metrics['scale_error']:.6f}")
        
        # Analyze codebook usage
        codebook_metrics = analyze_codebook(indices)
        print(f"Active tokens: {codebook_metrics['active_tokens']}/{CODEBOOK_SIZE}")
        print(f"Codebook entropy: {codebook_metrics['entropy']:.4f} bits")
        print(f"50% of data encoded with {codebook_metrics['tokens_50pct']} tokens")
        
    elif args.mode == 'transformer':
        # Load pretrained VQVAE
        if args.vqvae_path is None:
            args.vqvae_path = os.path.join(args.save_dir, 'best_vqvae.pt')
        
        if not os.path.exists(args.vqvae_path):
            print(f"Error: VQVAE model {args.vqvae_path} not found")
            return
        
        # Create VQVAE and load weights
        vqvae = VQVAE(in_channels=in_channels)
        vqvae = to_device_and_dtype(vqvae)
        
        try:
            vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=DEVICE))
            vqvae.eval()
            
            # Also save a copy of this VQVAE in the current save directory
            vqvae_save_path = os.path.join(args.save_dir, f'vqvae_{args.model_type}_used_for_transformer.pt')
            torch.save(vqvae.state_dict(), vqvae_save_path)
            print(f"Saved copy of VQVAE model to {vqvae_save_path}")
        except Exception as e:
            print(f"Error loading VQVAE model: {e}")
            return
        
        # Create token dataset
        token_sequences = create_token_dataset(vqvae, train_loader)
        
        # Create loaders for token sequences
        token_train_size = int(len(token_sequences) * 0.8)
        token_val_size = int(len(token_sequences) * 0.1)
        
        token_train = token_sequences[:token_train_size]
        token_val = token_sequences[token_train_size:token_train_size+token_val_size]
        token_test = token_sequences[token_train_size+token_val_size:]
        
        token_train_loader = DataLoader(token_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        token_val_loader = DataLoader(token_val, batch_size=args.batch_size, pin_memory=True)
        token_test_loader = DataLoader(token_test, batch_size=args.batch_size, pin_memory=True)
        
        # Create and train transformer
        transformer = NanoGPT(
            vocab_size=CODEBOOK_SIZE,
            model_dim=MODEL_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        )
        
        print("Training transformer model...")
        transformer, metrics = train_transformer(
            transformer=transformer,
            train_loader=token_train_loader,
            val_loader=token_val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            save_dir=args.save_dir
        )
        
    elif args.mode == 'finetune':
        # Load pretrained VQVAE
        if args.vqvae_path is None:
            args.vqvae_path = os.path.join(args.save_dir, 'best_vqvae.pt')
        
        if not os.path.exists(args.vqvae_path):
            print(f"Error: VQVAE model {args.vqvae_path} not found")
            return
        
        # Create VQVAE and load weights
        vqvae = VQVAE(in_channels=in_channels)
        vqvae = to_device_and_dtype(vqvae)
        vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=DEVICE))
        
        # Analyze codebook before fine-tuning
        vqvae.eval()
        data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).contiguous().unsqueeze(0)
        data_tensor = to_device_and_dtype(data_tensor)
        
        with torch.no_grad():
            # Process in chunks to avoid MPS limitations
            if data_tensor.shape[2] > 1000:
                indices_list = []
                for i in range(0, data_tensor.shape[2], 1000):
                    end = min(i + 1000, data_tensor.shape[2])
                    chunk = data_tensor[:, :, i:end]
                    _, _, chunk_indices, _ = vqvae(chunk, normalize=False)
                    indices_list.append(chunk_indices.cpu())
                indices = torch.cat(indices_list, dim=1)
            else:
                _, _, indices, _ = vqvae(data_tensor, normalize=False)
                indices = indices.cpu()
        
        indices = indices.squeeze().numpy()
        
        before_metrics = analyze_codebook(indices)
        print("Codebook usage before fine-tuning:")
        print(f"Active tokens: {before_metrics['active_tokens']}/{CODEBOOK_SIZE}")
        print(f"Codebook entropy: {before_metrics['entropy']:.4f} bits")
        
        # Fine-tune VQVAE
        print("Fine-tuning VQVAE codebook...")
        vqvae = fine_tune_codebook(
            vqvae=vqvae,
            train_loader=train_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            save_dir=args.save_dir
        )
        
        # Analyze codebook after fine-tuning
        vqvae.eval()
        
        with torch.no_grad():
            # Process in chunks to avoid MPS limitations
            if data_tensor.shape[2] > 1000:
                indices_list = []
                for i in range(0, data_tensor.shape[2], 1000):
                    end = min(i + 1000, data_tensor.shape[2])
                    chunk = data_tensor[:, :, i:end]
                    _, _, chunk_indices, _ = vqvae(chunk, normalize=False)
                    indices_list.append(chunk_indices.cpu())
                indices = torch.cat(indices_list, dim=1)
            else:
                _, _, indices, _ = vqvae(data_tensor, normalize=False)
                indices = indices.cpu()
        
        indices = indices.squeeze().numpy()
        
        after_metrics = analyze_codebook(indices)
        print("Codebook usage after fine-tuning:")
        print(f"Active tokens: {after_metrics['active_tokens']}/{CODEBOOK_SIZE}")
        print(f"Codebook entropy: {after_metrics['entropy']:.4f} bits")
        print(f"Improvement: {after_metrics['active_tokens'] - before_metrics['active_tokens']} more active tokens")

        # Clear MPS cache
        clear_cache()


if __name__ == '__main__':
    main()
