import argparse
import torch
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from vqvae import VQVAE
from nanogpt import NanoGPT
from data import load_dataset
from training import train_vqvae, train_transformer, fine_tune_codebook, create_token_dataset
from analysis import analyze_codebook
from utils import to_device_and_dtype, clear_cache
from config import *
from run_totem import TotemForecaster


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
    parser.add_argument('--grad_accumulation_steps', type=int, default=GRAD_ACCUMULATION_STEPS)
    parser.add_argument('--use_cached', action='store_true', help='Use cached preprocessed data')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'logs'), exist_ok=True)
    
    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found")
        return
    
    print(f"Loading data from {args.data}")
    train_loader, val_loader, test_loader, dataset_info = load_dataset(
        args.data,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        use_cached=args.use_cached
    )
    
    raw_data = dataset_info['raw_data']
    data_norm = dataset_info['normalized_data']
    mean = dataset_info['mean']
    std = dataset_info['std']
    features = dataset_info['features']
    
    print(f"Data shape: {data_norm.shape}, Features: {features}")
    
    in_channels = 1 if args.model_type == 'single' else data_norm.shape[1]
    
    if args.mode == 'vqvae':
        vqvae = VQVAE(
            in_channels=in_channels,
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=CODEBOOK_SIZE,
            commitment_cost=COMMITMENT_COST,
            num_hiddens=NUM_HIDDENS,
            num_residual_layers=NUM_RESIDUAL_LAYERS,
            num_residual_hiddens=NUM_RESIDUAL_HIDDENS
        ).to(DEVICE)
        
        if USE_FLOAT16 and DEVICE != "cpu":
            vqvae = vqvae.to(torch.float16)
        
        print("Training VQVAE model...")
        vqvae, metrics = train_vqvae(
            vqvae=vqvae,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            grad_accumulation_steps=args.grad_accumulation_steps
        )
        
        vqvae.eval()
        max_eval_samples = min(50000, data_norm.shape[0])
        eval_indices = np.random.choice(data_norm.shape[0], max_eval_samples, replace=False)
        eval_data = data_norm[eval_indices]
        
        data_tensor = torch.tensor(eval_data, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        data_tensor = to_device_and_dtype(data_tensor)
        
        with torch.no_grad():
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
        
        recon = x_recon.squeeze().transpose(0, 1).numpy()
        indices = indices.squeeze().numpy()
        
        mse = np.mean((eval_data - recon)**2)
        print(f"Reconstruction MSE: {mse:.6f}")
        
        codebook_metrics = analyze_codebook(indices)
        print(f"Active tokens: {codebook_metrics['active_tokens']}/{CODEBOOK_SIZE}")
        print(f"Codebook entropy: {codebook_metrics['entropy']:.4f} bits")
        print(f"50% of data encoded with {codebook_metrics['tokens_50pct']} tokens")
        
    elif args.mode == 'transformer':
        if args.vqvae_path is None:
            args.vqvae_path = os.path.join(args.save_dir, 'best_vqvae.pt')

        if not os.path.exists(args.vqvae_path):
            print(f"Error: VQVAE model {args.vqvae_path} not found")
            return
        
        forecaster = TotemForecaster(mode=args.model_type)
        try:
            forecaster.load_models(args.vqvae_path)
            vqvae = forecaster.vqvae
            vqvae_save_path = os.path.join(args.save_dir, f'vqvae_{args.model_type}_used_for_transformer.pt')
            torch.save(vqvae.state_dict(), vqvae_save_path)
            print(f"Saved copy of VQVAE model to {vqvae_save_path}")
        except Exception as e:
            print(f"Error loading VQVAE model: {e}")
            return
        
        token_sequences = create_token_dataset(vqvae, train_loader)
        
        token_train_size = int(len(token_sequences) * 0.8)
        token_val_size = int(len(token_sequences) * 0.1)
        
        token_train = token_sequences[:token_train_size]
        token_val = token_sequences[token_train_size:token_train_size+token_val_size]
        token_test = token_sequences[token_train_size+token_val_size:]
        
        token_train_loader = DataLoader(token_train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
        token_val_loader = DataLoader(token_val, batch_size=args.batch_size, pin_memory=True)
        token_test_loader = DataLoader(token_test, batch_size=args.batch_size, pin_memory=True)
        
        transformer = NanoGPT(
            vocab_size=CODEBOOK_SIZE,
            model_dim=MODEL_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(DEVICE)
        
        if USE_FLOAT16 and DEVICE != "cpu":
            transformer = transformer.to(torch.float16)
        
        print("Training transformer model...")
        transformer, metrics = train_transformer(
            transformer=transformer,
            train_loader=token_train_loader,
            val_loader=token_val_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            save_dir=args.save_dir,
            grad_accumulation_steps=args.grad_accumulation_steps
        )
        
    elif args.mode == 'finetune':
        if args.vqvae_path is None:
            args.vqvae_path = os.path.join(args.save_dir, 'best_vqvae.pt')
        
        if not os.path.exists(args.vqvae_path):
            print(f"Error: VQVAE model {args.vqvae_path} not found")
            return
        
        vqvae = VQVAE(in_channels=in_channels).to(DEVICE)
        
        if USE_FLOAT16 and DEVICE != "cpu":
            vqvae = vqvae.to(torch.float16)
            
        try:
            vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=DEVICE))
            print(f"Successfully loaded VQVAE from {args.vqvae_path}")
        except Exception as e:
            print(f"Error loading VQVAE: {e}")
            return
        
        print("Fine-tuning VQVAE codebook...")
        vqvae = fine_tune_codebook(
            vqvae=vqvae,
            train_loader=train_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            save_dir=args.save_dir
        )
        
        max_eval_samples = min(20000, data_norm.shape[0])
        eval_indices = np.random.choice(data_norm.shape[0], max_eval_samples, replace=False)
        eval_data = data_norm[eval_indices]
        
        data_tensor = torch.tensor(eval_data, dtype=torch.float32).transpose(0, 1).unsqueeze(0)
        data_tensor = to_device_and_dtype(data_tensor)
        
        with torch.no_grad():
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
        
        clear_cache()


if __name__ == '__main__':
    main()