import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
from utils import to_device_and_dtype, clear_cache, process_in_chunks, is_finetune_mode


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super().__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv1d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv1d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim):
        super().__init__()
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                out_channels=min(num_hiddens // 2, 8192), 
                                kernel_size=4,
                                stride=2, padding=1)
        self._conv_2 = nn.Conv1d(in_channels=min(num_hiddens // 2, 8192),
                                out_channels=min(num_hiddens, 16384),
                                kernel_size=4,
                                stride=2, padding=1)
        self._conv_3 = nn.Conv1d(in_channels=min(num_hiddens, 16384),
                                out_channels=min(num_hiddens, 16384),
                                kernel_size=3,
                                stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=min(num_hiddens, 16384),
                                            num_hiddens=min(num_hiddens, 16384),
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=min(num_residual_hiddens, 16384))
        self._pre_vq_conv = nn.Conv1d(in_channels=min(num_hiddens, 16384), 
                                     out_channels=embedding_dim,
                                     kernel_size=1, 
                                     stride=1)

    def forward(self, x):
        x = self._conv_1(x)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = self._residual_stack(x)
        x = self._pre_vq_conv(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super().__init__()
        self._conv_1 = nn.Conv1d(in_channels=in_channels,
                                out_channels=min(num_hiddens, 16384),
                                kernel_size=3,
                                stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=min(num_hiddens, 16384),
                                            num_hiddens=min(num_hiddens, 16384),
                                            num_residual_layers=num_residual_layers,
                                            num_residual_hiddens=min(num_residual_hiddens, 16384))
        
        self._conv_trans_1 = nn.ConvTranspose1d(in_channels=min(num_hiddens, 16384),
                                              out_channels=min(num_hiddens // 2, 8192),
                                              kernel_size=4,
                                              stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose1d(in_channels=min(num_hiddens // 2, 8192),
                                              out_channels=out_channels,
                                              kernel_size=4,
                                              stride=2, padding=1)

    def forward(self, x):
        x = self._conv_1(x)
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        x = self._conv_trans_2(x)
        
        return x


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings)
        self._embedding.weight.requires_grad_(True)
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.zeros(num_embeddings, embedding_dim))
        self._decay = 0.99
        self._epsilon = 1e-5
        
    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 1)
        input_shape = inputs.shape
        
        flat_input = inputs.reshape(-1, self._embedding_dim)
        
        # Compute distances efficiently
        distances = torch.cdist(flat_input, self._embedding.weight, p=2)
        
        # Find nearest codebook entry
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and compute loss
        quantized = torch.matmul(encodings, self._embedding.weight)
        quantized = quantized.view(input_shape)
        
        # Add noise during training (code diversity)
        if self.training:
            noise = torch.randn_like(quantized) * 0.01
            quantized = quantized + noise
        
        # Commitment loss with increased weight
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        # EMA update for codebook during fine-tuning
        if self.training and is_finetune_mode():
            with torch.no_grad():
                encodings_sum = encodings.sum(0)
                self._ema_cluster_size.data = self._ema_cluster_size * self._decay + (1 - self._decay) * encodings_sum
                
                # Reset dead entries
                n = torch.sum(self._ema_cluster_size < 1e-5)
                if n > 0:
                    # Find least used entries
                    indices = torch.argsort(self._ema_cluster_size)[:n]
                    # Reset to random centroids from the input batch
                    rand_indices = torch.randperm(flat_input.shape[0])[:n]
                    for i, idx in enumerate(indices):
                        self._embedding.weight.data[idx] = flat_input[rand_indices[i]]
                        self._ema_cluster_size.data[idx] = self._epsilon
                
                # Calculate updated weights
                dw = torch.matmul(encodings.t(), flat_input)
                self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw
                
                # Update embedding weights
                self._embedding.weight.data = self._ema_w / (self._ema_cluster_size.unsqueeze(1) + self._epsilon)
        
        # Straight-through estimator
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1)
        
        # Compute perplexity in a numerically stable way
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # Reshape indices for output
        encoding_indices = encoding_indices.reshape(input_shape[0], -1)
        
        return quantized, loss, encoding_indices, perplexity


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.means = None
        self.stds = None
    
    def normalize(self, x):
        self.means = x.mean(dim=2, keepdim=True).detach()
        self.stds = torch.sqrt(x.var(dim=2, keepdim=True, unbiased=False) + self.eps).detach()
        
        x_norm = (x - self.means) / self.stds
        return x_norm
    
    def denormalize(self, x, means=None, stds=None):
        means = means if means is not None else self.means
        stds = stds if stds is not None else self.stds
        
        if means is None or stds is None:
            raise ValueError("No normalization statistics available for denormalization")
            
        x = x * stds + means
        return x


class VQVAE(nn.Module):
    def __init__(self, in_channels=1, embedding_dim=EMBEDDING_DIM, num_embeddings=CODEBOOK_SIZE, 
                 commitment_cost=COMMITMENT_COST, num_hiddens=NUM_HIDDENS, 
                 num_residual_layers=NUM_RESIDUAL_LAYERS, num_residual_hiddens=NUM_RESIDUAL_HIDDENS):
        super().__init__()
        
        self.in_channels = in_channels
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.revin = RevIN(num_features=in_channels)
        
        self.encoder = Encoder(in_channels=in_channels,
                              num_hiddens=num_hiddens,
                              num_residual_layers=num_residual_layers,
                              num_residual_hiddens=num_residual_hiddens,
                              embedding_dim=embedding_dim)
        
        self.vq = VectorQuantizer(num_embeddings=num_embeddings,
                                 embedding_dim=embedding_dim,
                                 commitment_cost=commitment_cost)
        
        self.decoder = Decoder(in_channels=embedding_dim,
                              out_channels=in_channels,
                              num_hiddens=num_hiddens,
                              num_residual_layers=num_residual_layers,
                              num_residual_hiddens=num_residual_hiddens)
    
    def process_chunk(self, model, x_chunk):
        x_chunk = x_chunk.float()
        for module in [model.encoder, model.vq, model.decoder]:
            for param in module.parameters():
                param.data = param.data.float()
                    
        z = model.encoder(x_chunk)
        quantized, vq_loss, indices, perplexity = model.vq(z)
        x_recon = model.decoder(quantized)
        return x_recon, vq_loss, indices, perplexity
        
    def forward(self, x, normalize=True):
        # Handle normalization
        if normalize:
            x_norm = self.revin.normalize(x)
        else:
            x_norm = x
            
        # Ensure consistent type for all operations
        x_norm = x_norm.float()
        for module in [self.encoder, self.vq, self.decoder]:
            for param in module.parameters():
                param.data = param.data.float()
        
        # Limit batch size for processing
        if x_norm.size(0) > 16 and is_finetune_mode():
            return process_in_chunks(self, x_norm, self.process_chunk, chunk_size=16)
        
        # Standard forward pass
        z = self.encoder(x_norm)
        quantized, vq_loss, indices, perplexity = self.vq(z)
        x_recon = self.decoder(quantized)
        
        if normalize:
            x_recon = self.revin.denormalize(x_recon)
            
        return x_recon, vq_loss, indices, perplexity
    
    def encode(self, x, normalize=True):
        # Handle type and normalization
        if normalize:
            x_norm = self.revin.normalize(x)
        else:
            x_norm = x
            
        # Ensure consistent type
        x_norm = x_norm.float()
        for param in self.encoder.parameters():
            param.data = param.data.float()
        for param in self.vq.parameters():
            param.data = param.data.float()
        
        # Process in smaller batches if needed
        if x_norm.size(0) > 16:
            indices_list = []
            for i in range(0, x_norm.size(0), 16):
                batch = x_norm[i:i+16]
                z = self.encoder(batch)
                _, _, indices, _ = self.vq(z)
                indices_list.append(indices)
            return torch.cat(indices_list, dim=0)
        else:
            z = self.encoder(x_norm)
            _, _, indices, _ = self.vq(z)
            return indices
    
    def decode(self, indices, means=None, stds=None):
        device = indices.device
        batch_size = indices.size(0)
        
        # Ensure consistent type
        for param in self.decoder.parameters():
            param.data = param.data.float()
        for param in self.vq.parameters():
            param.data = param.data.float()
        
        # Convert indices to embeddings
        one_hot = F.one_hot(indices, num_classes=self.num_embeddings).float()
        one_hot = one_hot.view(batch_size, -1, self.num_embeddings)
        one_hot = one_hot.transpose(1, 2)
        
        # Get embeddings
        embeddings = self.vq._embedding.weight.to(device)
        quantized = torch.bmm(embeddings.expand(batch_size, -1, -1).transpose(1, 2), one_hot)
        quantized = quantized.transpose(1, 2)
        
        # Generate output
        x_recon = self.decoder(quantized)
        
        if means is not None and stds is not None:
            if isinstance(means, np.ndarray):
                means = torch.tensor(means, dtype=torch.float32).to(device).reshape(1, -1, 1)
            elif isinstance(means, torch.Tensor) and means.device != device:
                means = means.to(device)
                
            if isinstance(stds, np.ndarray):
                stds = torch.tensor(stds, dtype=torch.float32).to(device).reshape(1, -1, 1)
            elif isinstance(stds, torch.Tensor) and stds.device != device:
                stds = stds.to(device)
            
            x_recon = self.revin.denormalize(x_recon, means, stds)
            
        return x_recon
        
    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Ensure consistent type after moving to device
        for module in [self.encoder, self.decoder, self.vq]:
            module.float()
        return self