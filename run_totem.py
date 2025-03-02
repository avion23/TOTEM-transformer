import argparse
import torch
import numpy as np
import pandas as pd
import os
from vqvae import VQVAE
from nanogpt import NanoGPT
from data import load_dataset
from analysis import analyze_codebook
from utils import to_device_and_dtype, clear_cache
from config import *

np.random.seed(42)
torch.manual_seed(42)

class TotemForecaster:
    def __init__(self, mode='single', device=DEVICE):
        self.mode = mode
        self.device = device
        self.in_channels = 1 if mode == 'single' else 3
        
        self.vqvae = VQVAE(
            in_channels=self.in_channels,
            embedding_dim=EMBEDDING_DIM,
            num_embeddings=CODEBOOK_SIZE,
            commitment_cost=COMMITMENT_COST,
            num_hiddens=NUM_HIDDENS,
            num_residual_layers=NUM_RESIDUAL_LAYERS,
            num_residual_hiddens=NUM_RESIDUAL_HIDDENS
        ).to(device)
        
        if USE_FLOAT16 and device != "cpu":
            self.vqvae = self.vqvae.to(torch.float16)
            
        self.transformer = None
        self.dataset_info = None
    
    def load_models(self, vqvae_path, transformer_path=None):
        if not os.path.exists(vqvae_path):
            raise FileNotFoundError(f"VQVAE model not found: {vqvae_path}")
        
        try:
            vqvae_state_dict = torch.load(vqvae_path, map_location=self.device, weights_only=True)
            self.vqvae.load_state_dict(vqvae_state_dict)
            self.vqvae.eval()
            print(f"Successfully loaded VQVAE from {vqvae_path}")
        except Exception as e:
            raise RuntimeError(f"Error loading VQVAE model: {e}")
        
        if transformer_path is not None and os.path.exists(transformer_path):
            try:
                state_dict = torch.load(transformer_path, map_location=self.device, weights_only=True)
            
                pos_emb_key = [k for k in state_dict.keys() if 'pos_embedding' in k][0]
                pos_emb_shape = state_dict[pos_emb_key].shape
                model_dim = pos_emb_shape[2]
                
                num_blocks = 0
                for key in state_dict.keys():
                    if 'blocks.' in key:
                        block_num = int(key.split('.')[1])
                        num_blocks = max(num_blocks, block_num + 1)
                
                heads_key = [k for k in state_dict.keys() if 'query.weight' in k][0]
                query_weight = state_dict[heads_key]
                head_size = query_weight.shape[0] // model_dim
                
                print(f"Creating transformer with {num_blocks} blocks, {model_dim} dim, {head_size} heads")
                
                self.transformer = NanoGPT(
                    vocab_size=CODEBOOK_SIZE,
                    model_dim=model_dim,
                    num_heads=head_size,
                    num_layers=num_blocks
                ).to(self.device)
                
                if USE_FLOAT16 and self.device != "cpu":
                    self.transformer = self.transformer.to(torch.float16)
                    
                self.transformer.load_state_dict(state_dict)
                self.transformer.eval()
                print(f"Successfully loaded transformer from {transformer_path}")
            except Exception as e:
                print(f"Warning: Error loading transformer model: {e}")
                print("Will proceed with only VQVAE for analysis")
                self.transformer = None
        else:
            if transformer_path is not None:
                print(f"Warning: Transformer model not found: {transformer_path}")
            print("Will proceed with only VQVAE for analysis")
            self.transformer = None
    
    def load_data(self, path, max_samples=5000):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Data file not found: {path}")
            
        _, _, _, dataset_info = load_dataset(path, max_samples=max_samples)
        self.dataset_info = dataset_info
        
        return dataset_info['normalized_data'], dataset_info['raw_data']
    
    def _prepare_data_tensor(self, data_norm):
        if self.mode == 'single':
            data_tensor = torch.tensor(data_norm, dtype=torch.float32).reshape(1, 1, -1).contiguous()
        else:
            data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).contiguous()
        
        return to_device_and_dtype(data_tensor)
    
    def tokenize(self, data_norm):
        data_tensor = self._prepare_data_tensor(data_norm)
        chunk_size = 1000
        total_indices = []
        
        with torch.no_grad():
            for i in range(0, data_tensor.shape[2], chunk_size):
                chunk = data_tensor[:, :, i:i+chunk_size].contiguous()
                indices = self.vqvae.encode(chunk, normalize=False)
                total_indices.append(indices.squeeze().cpu().numpy())
        
        return np.concatenate(total_indices)
    
    def forecast(self, indices, steps=30, temperature=1.0, top_k=None):
        if self.transformer is None:
            raise RuntimeError("Transformer model not loaded. Cannot forecast.")
            
        seq_len = min(CONTEXT_LENGTH, len(indices))
        seed = indices[-seq_len:]
        
        seed_tensor = torch.tensor(seed, dtype=torch.long).unsqueeze(0).to(self.device).contiguous()
        
        with torch.no_grad():
            forecast = self.transformer.generate(
                seed_tensor, 
                max_new_tokens=steps, 
                temperature=temperature,
                top_k=top_k
            )
        
        forecast_tokens = forecast[0, seq_len:].cpu().numpy()
        
        return forecast_tokens
    
    def decode_forecast(self, tokens):
        token_tensor = torch.tensor([tokens], dtype=torch.long).to(self.device).contiguous()
        
        revin_stats = {
            'mean': torch.tensor(self.dataset_info['mean'], dtype=torch.float32).to(self.device).contiguous(),
            'std': torch.tensor(self.dataset_info['std'], dtype=torch.float32).to(self.device).contiguous()
        }
        
        if USE_FLOAT16 and self.device != "cpu":
            revin_stats = {k: v.to(torch.float16) for k, v in revin_stats.items()}
        
        with torch.no_grad():
            values = self.vqvae.decode(token_tensor, revin_stats)
            values = values.squeeze().cpu().numpy()
        
        if self.mode == 'single':
            return values
        else:
            if len(values.shape) == 1:
                return values[0]
            else:
                return values[:, 0]
    
    def analyze_codebook_usage(self, data_norm):
        data_tensor = self._prepare_data_tensor(data_norm)
        chunk_size = 1000
        indices_list = []
        
        with torch.no_grad():
            for i in range(0, data_tensor.shape[2], chunk_size):
                chunk = data_tensor[:, :, i:i+chunk_size].contiguous()
                _, _, indices, _ = self.vqvae(chunk, normalize=False)
                indices_list.append(indices.cpu().numpy())
        
        all_indices = np.concatenate(indices_list, axis=1)
        
        metrics = analyze_codebook(all_indices, model=self.vqvae)
        
        print(f"\nCodebook Analysis:")
        print(f"Active tokens: {metrics['active_tokens']}/{metrics['codebook_size']} ({metrics['active_pct']:.2f}%)")
        print(f"Unused tokens: {metrics['unused_tokens']}/{metrics['codebook_size']} ({100-metrics['active_pct']:.2f}%)")
        print(f"Entropy: {metrics['entropy']:.2f} bits ({metrics['normalized_entropy']:.2f}% of max)")
        
        print("\nTop 10 tokens by usage:")
        for i, (idx, count, pct) in enumerate(metrics['top_tokens'][:10]):
            print(f"  {i+1}. Token {idx}: {pct:.2f}%")
        
        clear_cache()
            
        return metrics

    def analyze_reconstruction(self, data_norm, num_samples=None, feature_names=None):
        if num_samples is not None and num_samples < len(data_norm):
            sample_indices = np.random.choice(len(data_norm), num_samples, replace=False)
            data_subset = data_norm[sample_indices]
        else:
            data_subset = data_norm
        
        data_tensor = self._prepare_data_tensor(data_subset)
        
        with torch.no_grad():
            x_recon, _, _, _ = self.vqvae(data_tensor, normalize=True)
        
        original = data_tensor.cpu().numpy()
        recon = x_recon.cpu().numpy()
        
        mse = np.mean((original - recon)**2)
        print(f"\nReconstruction Analysis:")
        print(f"Mean Squared Error: {mse:.6f}")
        
        feature_mses = []
        if self.mode == 'multi' and original.shape[1] > 1:
            for i in range(original.shape[1]):
                feature_mse = np.mean((original[:, i] - recon[:, i])**2)
                feature_name = feature_names[i] if feature_names is not None and i < len(feature_names) else f"Feature {i}"
                print(f"{feature_name} MSE: {feature_mse:.6f}")
                feature_mses.append(feature_mse)
        
        return {
            'mse': mse,
            'feature_mses': feature_mses,
            'original': original,
            'reconstruction': recon
        }


def main():
    parser = argparse.ArgumentParser(description='TOTEM Forecaster')
    parser.add_argument('--data', type=str, default='../../trading/discountedRewards/data/xbt-test.csv')
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], default='multi')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--temps', type=float, nargs='+', default=[0.0, 0.5, 1.0])
    parser.add_argument('--vqvae', type=str, default=None)
    parser.add_argument('--transformer', type=str, default=None)
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--analyze', action='store_true', help='Analyze codebook usage')
    parser.add_argument('--analyze_recon', action='store_true', help='Analyze reconstruction quality')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples to use')
    parser.add_argument('--plot', action='store_true', help='Generate visualizations (requires matplotlib)')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
    
    args = parser.parse_args()
    
    if args.vqvae is None:
        args.vqvae = os.path.join('models', 'best_vqvae.pt')
    
    if args.transformer is None:
        args.transformer = os.path.join('models', 'best_transformer.pt')
    
    if args.plot and not os.path.exists(args.plot_dir):
        os.makedirs(args.plot_dir)
    
    print(f"Running in {args.mode}-feature mode")
    print(f"Using VQVAE: {args.vqvae}")
    print(f"Using Transformer: {args.transformer}")
    print(f"Device: {DEVICE}")
    
    try:
        forecaster = TotemForecaster(mode=args.mode)
        forecaster.load_models(args.vqvae, args.transformer)
        
        data_path = args.data
        if not os.path.exists(data_path):
            if os.path.exists('data'):
                data_files = [f for f in os.listdir('data') if f.endswith('.csv')]
                if data_files:
                    data_path = os.path.join('data', data_files[0])
                    print(f"Using data file: {data_path}")
                else:
                    print("No data files found in 'data' directory.")
                    return
            else:
                print(f"Error: Data file {data_path} not found")
                return
                
        data_norm, raw_data = forecaster.load_data(data_path, max_samples=args.samples)
        tokens = forecaster.tokenize(data_norm)
        
        feature_names = forecaster.dataset_info['features']
        print(f"Features: {feature_names}")
        
        if args.mode == 'single':
            price_data = raw_data
        else:
            price_data = raw_data[:, 0] if raw_data.ndim > 1 else raw_data
        
        print(f"Tokenized {len(tokens)} data points")
        
        if args.analyze or args.analyze_recon:
            if args.analyze:
                codebook_metrics = forecaster.analyze_codebook_usage(data_norm)
            
            if args.analyze_recon:
                recon_metrics = forecaster.analyze_reconstruction(data_norm, feature_names=feature_names)
                
                if args.plot:
                    try:
                        import matplotlib.pyplot as plt
                        
                        plt.figure(figsize=(12, 8))
                        
                        if args.mode == 'multi':
                            num_features = min(3, recon_metrics['original'].shape[1])
                            for i in range(num_features):
                                plt.subplot(num_features, 1, i+1)
                                feature_name = feature_names[i] if i < len(feature_names) else f"Feature {i}"
                                plt.plot(recon_metrics['original'][0, i, :200], 'b-', label=f'Original {feature_name}', linewidth=1.5)
                                plt.plot(recon_metrics['reconstruction'][0, i, :200], 'r--', label=f'Reconstruction', linewidth=1.5)
                                plt.legend(loc='upper right')
                                plt.grid(True, alpha=0.3)
                                plt.ylabel(feature_name)
                                if i == 0:
                                    plt.title('Original vs Reconstruction')
                                if i == num_features - 1:
                                    plt.xlabel('Time Steps')
                        else:
                            plt.plot(recon_metrics['original'][0, 0, :200], 'b-', label='Original', linewidth=1.5)
                            plt.plot(recon_metrics['reconstruction'][0, 0, :200], 'r--', label='Reconstruction', linewidth=1.5)
                            plt.legend(loc='upper right')
                            plt.grid(True, alpha=0.3)
                            plt.title('Original vs Reconstruction')
                            plt.xlabel('Time Steps')
                            plt.ylabel('Value')
                            
                        plt.tight_layout()
                        plot_path = os.path.join(args.plot_dir, 'reconstruction_plot.png')
                        plt.savefig(plot_path, dpi=300)
                        print(f"Saved reconstruction plot to {plot_path}")
                        plt.close()
                        
                        # Plot token distribution 
                        plt.figure(figsize=(10, 6))
                        token_counts = np.bincount(tokens.astype(np.int32), minlength=CODEBOOK_SIZE)
                        active_indices = np.where(token_counts > 0)[0]
                        active_counts = token_counts[active_indices]
                        plt.bar(active_indices, active_counts)
                        plt.title('Token Distribution')
                        plt.xlabel('Token ID')
                        plt.ylabel('Frequency')
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        token_plot_path = os.path.join(args.plot_dir, 'token_distribution.png')
                        plt.savefig(token_plot_path, dpi=300)
                        print(f"Saved token distribution plot to {token_plot_path}")
                        plt.close()
                        
                    except ImportError:
                        print("Matplotlib not available. Skipping plot generation.")
        
        if forecaster.transformer is not None:
            forecasts = []
            
            for temp in args.temps:
                print(f"\nGenerating forecast with temperature={temp}:")
                forecast_tokens = forecaster.forecast(indices=tokens, steps=args.steps, temperature=temp, top_k=args.top_k)
                forecast_prices = forecaster.decode_forecast(forecast_tokens)
                
                forecasts.append(forecast_prices)
                
                token_diversity = len(np.unique(forecast_tokens))
                print(f"Tokens: {forecast_tokens[:10]}... ({token_diversity} unique tokens)")
                print(f"Price forecast (first 5): {forecast_prices[:5].tolist()}")
                
                last_price = price_data[-1]
                change = forecast_prices[-1] - last_price
                perc = (change / last_price) * 100
                volatility = np.std(forecast_prices)
                
                print(f"Last actual price: ${last_price:.2f}")
                print(f"Final forecast: ${forecast_prices[-1]:.2f}")
                print(f"Change: ${change:.2f} ({perc:.2f}%)")
                print(f"Forecast volatility: ${volatility:.2f}")
                
                if args.plot and len(forecasts) > 0:
                    try:
                        import matplotlib.pyplot as plt
                        plt.figure(figsize=(12, 6))
                        
                        history_points = 50
                        if len(price_data) > history_points:
                            plt.plot(range(-history_points, 0), price_data[-history_points:], 'b-', label='Historical Prices', linewidth=1.5)
                        else:
                            plt.plot(range(-len(price_data), 0), price_data, 'b-', label='Historical Prices', linewidth=1.5)
                        
                        colors = ['r', 'g', 'c', 'm', 'y', 'k']
                        for i, (temp, forecast) in enumerate(zip(args.temps, forecasts)):
                            color = colors[i % len(colors)]
                            plt.plot(range(len(forecast)), forecast, linestyle='--', color=color, marker='o', markersize=3, 
                                     label=f'Forecast (temp={temp})')
                            
                        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
                        plt.title('Price Forecast')
                        plt.xlabel('Time Steps')
                        plt.ylabel('Price')
                        plt.legend()
                        plt.grid(True, alpha=0.3)
                        plt.tight_layout()
                        forecast_plot_path = os.path.join(args.plot_dir, 'forecast_plot.png')
                        plt.savefig(forecast_plot_path, dpi=300)
                        print(f"Saved forecast plot to {forecast_plot_path}")
                        plt.close()
                    except ImportError:
                        print("Matplotlib not available. Skipping plot generation.")
        
        clear_cache()
            
    except Exception as e:
        print(f"Error during forecast: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
