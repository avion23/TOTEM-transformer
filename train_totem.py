import argparse
import torch
import numpy as np
import os
import sys
from vqvae import VQVAE
from nanogpt import NanoGPT
from data import create_dataloaders
from utils import to_device_and_dtype, reset_codebook_entries, clear_cache
from training import train_vqvae, train_transformer, finetune_vqvae
from codebook_analysis import analyze_codebook_usage
from config import *

def main():
    parser = argparse.ArgumentParser(description='Train TOTEM model')
    parser.add_argument('--mode', type=str, choices=['vqvae', 'transformer', 'finetune'], required=True)
    parser.add_argument('--data', type=str, default='../../trading/discountedRewards/data/xbt-test.csv')
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--weight_decay', type=float, default=WEIGHT_DECAY)
    parser.add_argument('--feature_mode', type=str, choices=['single', 'multi'], default='multi')
    parser.add_argument('--vqvae_path', type=str, default=None)
    parser.add_argument('--transformer_path', type=str, default=None)
    parser.add_argument('--save_dir', type=str, default='models')
    
    args = parser.parse_args()
    
    if args.mode == 'finetune' and args.vqvae_path is None:
        parser.error("--vqvae_path is required for finetune mode")
    
    if args.mode == 'transformer' and args.vqvae_path is None:
        parser.error("--vqvae_path is required for transformer mode")
        
    device = DEVICE
    
    if args.mode == 'finetune':
        print("Disabled float16 for fine-tuning to avoid type mismatches")
        float_precision = torch.float32
    else:
        float_precision = torch.float16 if USE_FLOAT16 and device != "cpu" else torch.float32
    
    in_channels = 1 if args.feature_mode == 'single' else 3
    
    print(f"Loading data from {args.data}")
    train_loader, val_loader, test_loader, dataset_info = create_dataloaders(
        args.data, 
        batch_size=args.batch_size,
        feature_mode=args.feature_mode
    )
    
    features = dataset_info.get('features', [])
    print(f"Using features: {features}")
    print(f"Data shape: {dataset_info['data_shape']}, Features: {features}")
    
    if args.mode in ['vqvae', 'finetune']:
        vqvae = VQVAE(
            in_channels=in_channels,
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=CODEBOOK_SIZE,
            commitment_cost=COMMITMENT_COST,
            num_hiddens=NUM_HIDDENS,
            num_residual_layers=NUM_RESIDUAL_LAYERS,
            num_residual_hiddens=NUM_RESIDUAL_HIDDENS
        ).to(device)
        
        if args.mode == 'vqvae' and float_precision == torch.float16:
            vqvae = vqvae.to(float_precision)
            
        if args.mode == 'finetune':
            try:
                state_dict = torch.load(args.vqvae_path, map_location=device)
                
                # Handle missing EMA buffers (added during upgrade)
                missing_keys = []
                model_dict = vqvae.state_dict()
                for k in model_dict.keys():
                    if k not in state_dict and k in ["vq._ema_cluster_size", "vq._ema_w"]:
                        missing_keys.append(k)
                        
                # Initialize missing EMA buffers
                if "vq._ema_cluster_size" in missing_keys:
                    vqvae.vq.register_buffer('_ema_cluster_size', torch.zeros(CODEBOOK_SIZE))
                    
                if "vq._ema_w" in missing_keys:
                    vqvae.vq.register_buffer('_ema_w', torch.zeros(CODEBOOK_SIZE, EMBEDDING_DIM))
                
                # Load with strict=False to allow missing keys
                vqvae.load_state_dict(state_dict, strict=False)
                print(f"Successfully loaded VQVAE from {args.vqvae_path}")
            except Exception as e:
                print(f"Error loading VQVAE: {e}")
                sys.exit(1)
            
            # Analyze codebook before fine-tuning
            analyze_codebook_usage(vqvae, train_loader, device)
            
            # Fine-tune
            finetune_vqvae(vqvae, train_loader, val_loader, epochs=args.epochs, lr=args.lr)
            
            # Analyze codebook after fine-tuning
            active_tokens, codebook_data = analyze_codebook_usage(vqvae, train_loader, device)
            print(f"Codebook usage after fine-tuning:")
            print(f"Active tokens: {active_tokens}/{CODEBOOK_SIZE}")
            print(f"Codebook entropy: {codebook_data['entropy']:.4f} bits")
            
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
                
            save_path = os.path.join(args.save_dir, 'finetuned_vqvae.pt')
            torch.save(vqvae.state_dict(), save_path)
            print(f"Fine-tuned VQVAE saved to {save_path}")
            
        elif args.mode == 'vqvae':
            # Train VQVAE from scratch
            vqvae = train_vqvae(
                vqvae, 
                train_loader, 
                val_loader, 
                epochs=args.epochs, 
                lr=args.lr, 
                weight_decay=args.weight_decay,
                device=device
            )
            
            if not os.path.exists(args.save_dir):
                os.makedirs(args.save_dir)
                
            save_path = os.path.join(args.save_dir, 'best_vqvae.pt')
            torch.save(vqvae.state_dict(), save_path)
            print(f"Best VQVAE model saved to {save_path}")
            
            # Save metadata
            metadata = {
                'codebook_size': CODEBOOK_SIZE,
                'embedding_dim': EMBEDDING_DIM,
                'in_channels': in_channels,
                'features': features,
                'dataset': os.path.basename(args.data),
                'steps': len(train_loader) * args.epochs
            }
            
            import json
            with open(os.path.join(args.save_dir, 'vqvae_metadata.json'), 'w') as f:
                json.dump(metadata, f)
            
    elif args.mode == 'transformer':
        # Load VQVAE model for tokenization
        vqvae = VQVAE(
            in_channels=in_channels,
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=CODEBOOK_SIZE,
            num_hiddens=NUM_HIDDENS,
            num_residual_layers=NUM_RESIDUAL_LAYERS,
            num_residual_hiddens=NUM_RESIDUAL_HIDDENS
        ).to(device)
        
        if float_precision == torch.float16:
            vqvae = vqvae.to(float_precision)
            
        try:
            vqvae.load_state_dict(torch.load(args.vqvae_path, map_location=device), strict=False)
            vqvae.eval()
            print(f"Successfully loaded VQVAE from {args.vqvae_path}")
        except Exception as e:
            print(f"Error loading VQVAE: {e}")
            sys.exit(1)
            
        # Create transformer model
        transformer = NanoGPT(
            vocab_size=CODEBOOK_SIZE,
            model_dim=MODEL_DIM,
            num_heads=NUM_HEADS,
            num_layers=NUM_LAYERS,
            dropout=DROPOUT
        ).to(device)
        
        if float_precision == torch.float16:
            transformer = transformer.to(float_precision)
            
        # Train transformer
        transformer = train_transformer(
            transformer,
            vqvae,
            train_loader,
            val_loader,
            context_length=CONTEXT_LENGTH,
            out_length=OUT_LENGTH,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            device=device
        )
        
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
            
        save_path = os.path.join(args.save_dir, 'best_transformer.pt')
        torch.save(transformer.state_dict(), save_path)
        print(f"Best transformer model saved to {save_path}")
        
        # Save metadata
        metadata = {
            'codebook_size': CODEBOOK_SIZE,
            'model_dim': MODEL_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'context_length': CONTEXT_LENGTH,
            'out_length': OUT_LENGTH,
            'vqvae_path': args.vqvae_path,
            'dataset': os.path.basename(args.data),
            'steps': len(train_loader) * args.epochs
        }
        
        import json
        with open(os.path.join(args.save_dir, 'transformer_metadata.json'), 'w') as f:
            json.dump(metadata, f)

if __name__ == '__main__':
    main()