import argparse
import torch
import numpy as np
import os
from vqvae import VQVAE
from data import load_dataset
from analysis import analyze_codebook
from utils import to_device_and_dtype, clear_cache
from config import *

class CodebookAnalyzer:
    def __init__(self, mode='multi', device=DEVICE):
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

    def load_model(self, model_path):
        vqvae_state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.vqvae.load_state_dict(vqvae_state_dict)
        self.vqvae.eval()
        return self.vqvae

    def load_data(self, data_path, max_samples=5000):
        _, _, _, dataset_info = load_dataset(data_path, max_samples=max_samples)
        self.dataset_info = dataset_info
        data_norm = dataset_info['normalized_data']
        raw_data = dataset_info['raw_data']
        return data_norm, raw_data, dataset_info['features']

    def prepare_tensor(self, data_norm):
        if self.mode == 'single':
            data_tensor = torch.tensor(data_norm, dtype=torch.float32).reshape(1, 1, -1).contiguous()
        else:
            data_tensor = torch.tensor(data_norm, dtype=torch.float32).transpose(0, 1).unsqueeze(0).contiguous()
        return to_device_and_dtype(data_tensor)

    def evaluate_codebook(self, data_norm):
        data_tensor = self.prepare_tensor(data_norm)
        chunk_size = 1000
        indices_list = []
        
        with torch.no_grad():
            for i in range(0, data_tensor.shape[2], chunk_size):
                end_idx = min(i + chunk_size, data_tensor.shape[2])
                chunk = data_tensor[:, :, i:end_idx].contiguous()
                _, _, indices, _ = self.vqvae(chunk, normalize=False)
                indices_list.append(indices.cpu().numpy())
        
        all_indices = np.concatenate(indices_list, axis=1)
        metrics = analyze_codebook(all_indices, model=self.vqvae)
        return metrics, all_indices.flatten()

    def evaluate_reconstruction(self, data_norm, feature_names=None):
        data_tensor = self.prepare_tensor(data_norm)
        
        with torch.no_grad():
            x_recon, _, _, _ = self.vqvae(data_tensor, normalize=True)
        
        original = data_tensor.cpu().numpy()
        recon = x_recon.cpu().numpy()
        
        mse = np.mean((original - recon)**2)
        
        feature_mses = []
        if self.mode == 'multi' and original.shape[1] > 1:
            for i in range(original.shape[1]):
                feature_mse = np.mean((original[:, i] - recon[:, i])**2)
                feature_mses.append(feature_mse)
        
        return {
            'mse': mse,
            'feature_mses': feature_mses,
            'original': original,
            'reconstruction': recon
        }

    def generate_plots(self, codebook_metrics, token_indices, recon_metrics, feature_names, plot_dir='plots'):
        try:
            import matplotlib.pyplot as plt
            os.makedirs(plot_dir, exist_ok=True)
            
            # Token distribution plot
            plt.figure(figsize=(10, 6))
            token_counts = np.bincount(token_indices.astype(np.int32), minlength=CODEBOOK_SIZE)
            active_indices = np.where(token_counts > 0)[0]
            active_counts = token_counts[active_indices]
            plt.bar(active_indices, active_counts)
            plt.title('Token Distribution')
            plt.xlabel('Token ID')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            token_plot_path = os.path.join(plot_dir, 'token_distribution.png')
            plt.savefig(token_plot_path, dpi=300)
            plt.close()
            
            # Reconstruction plot
            plt.figure(figsize=(12, 8))
            
            if self.mode == 'multi':
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
            recon_plot_path = os.path.join(plot_dir, 'reconstruction_plot.png')
            plt.savefig(recon_plot_path, dpi=300)
            plt.close()
            
            # Cumulative distribution plot
            plt.figure(figsize=(10, 6))
            sorted_counts = np.sort(token_counts)[::-1]
            cumulative = np.cumsum(sorted_counts) / np.sum(sorted_counts) * 100
            plt.plot(range(len(cumulative)), cumulative, 'b-', linewidth=2)
            plt.axhline(y=50, color='r', linestyle='--')
            plt.axhline(y=90, color='g', linestyle='--')
            plt.title('Cumulative Token Usage')
            plt.xlabel('Number of tokens')
            plt.ylabel('Percentage of data encoded (%)')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            cumulative_plot_path = os.path.join(plot_dir, 'token_cumulative.png')
            plt.savefig(cumulative_plot_path, dpi=300)
            plt.close()
            
            return True
        except ImportError:
            print("Matplotlib not available. Skipping plot generation.")
            return False

def main():
    parser = argparse.ArgumentParser(description='Analyze VQVAE Codebook')
    parser.add_argument('--model', type=str, default='models/best_vqvae.pt', help='Path to VQVAE model')
    parser.add_argument('--data', type=str, required=True, help='Path to data CSV file')
    parser.add_argument('--mode', type=str, choices=['single', 'multi'], default='multi', help='Model mode')
    parser.add_argument('--samples', type=int, default=5000, help='Number of samples to analyze')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Directory to save plots')
    parser.add_argument('--no_plot', action='store_true', help='Skip plot generation')
    args = parser.parse_args()

    analyzer = CodebookAnalyzer(mode=args.mode, device=DEVICE)
    analyzer.load_model(args.model)
    print(f"Loaded model from {args.model}")
    
    data_norm, raw_data, feature_names = analyzer.load_data(args.data, args.samples)
    print(f"Loaded {len(data_norm)} samples from {args.data}")
    print(f"Features: {feature_names}")
    
    codebook_metrics, token_indices = analyzer.evaluate_codebook(data_norm)
    
    print("\nCodebook Analysis:")
    print(f"Active tokens: {codebook_metrics['active_tokens']}/{codebook_metrics['codebook_size']} ({codebook_metrics['active_pct']:.2f}%)")
    print(f"Unused tokens: {codebook_metrics['unused_tokens']}/{codebook_metrics['codebook_size']} ({100-codebook_metrics['active_pct']:.2f}%)")
    print(f"Entropy: {codebook_metrics['entropy']:.2f} bits ({codebook_metrics['normalized_entropy']:.2f}% of max)")
    print(f"50% of data encoded with {codebook_metrics['tokens_50pct']} tokens")
    print(f"90% of data encoded with {codebook_metrics['tokens_90pct']} tokens")
    
    print("\nTop 10 tokens by usage:")
    for i, (idx, count, pct) in enumerate(codebook_metrics['top_tokens'][:10]):
        print(f"  {i+1}. Token {idx}: {pct:.2f}%")
    
    recon_metrics = analyzer.evaluate_reconstruction(data_norm, feature_names)
    
    print("\nReconstruction Analysis:")
    print(f"Mean Squared Error: {recon_metrics['mse']:.6f}")
    
    if args.mode == 'multi' and len(recon_metrics['feature_mses']) > 0:
        for i, mse in enumerate(recon_metrics['feature_mses']):
            feature_name = feature_names[i] if i < len(feature_names) else f"Feature {i}"
            print(f"{feature_name} MSE: {mse:.6f}")
    
    if not args.no_plot:
        analyzer.generate_plots(codebook_metrics, token_indices, recon_metrics, feature_names, args.plot_dir)
        print(f"Generated plots in directory: {args.plot_dir}")

if __name__ == "__main__":
    main()
