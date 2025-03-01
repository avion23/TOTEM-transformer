import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
from utils import to_device_and_dtype, clear_cache


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
        if x.size(0) * x.size(1) > 32768:
            outputs = []
            max_batch = max(1, 32768 // x.size(1))
            for i in range(0, x.size(0), max_batch):
                chunk = x[i:i+max_batch]
                outputs.append(self._forward_chunk(chunk))
            x = torch.cat(outputs, dim=0)
            return x
        else:
            return self._forward_chunk(x)

    def _forward_chunk(self, x):
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
        if x.size(0) * x.size(1) > 32768:
            outputs = []
            max_batch = max(1, 32768 // x.size(1))
            for i in range(0, x.size(0), max_batch):
                chunk = x[i:i+max_batch]
                outputs.append(self._forward_chunk(chunk))
            x = torch.cat(outputs, dim=0)
            return x
        else:
            return self._forward_chunk(x)

    def _forward_chunk(self, x):
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
        self._commitment_cost = commitment_cost
        
    def forward(self, inputs):
        if inputs.size(0) * inputs.size(2) * self._num_embeddings > 65536:
            return self._forward_chunked(inputs)
        else:
            return self._forward_standard(inputs)
    
    def _forward_standard(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.reshape(-1, self._embedding_dim)
        
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight)
        quantized = quantized.reshape(input_shape)
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        encoding_indices = encoding_indices.reshape(input_shape[0], -1)
        
        return quantized, loss, encoding_indices, perplexity
    
    def _forward_chunked(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        max_size = 65536 // self._num_embeddings
        chunk_size = max(1, max_size)
        
        all_quantized = []
        all_indices = []
        total_loss = 0
        total_samples = 0
        
        flat_input = inputs.reshape(-1, self._embedding_dim)
        num_chunks = (flat_input.size(0) + chunk_size - 1) // chunk_size
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i+1) * chunk_size, flat_input.size(0))
            chunk = flat_input[start_idx:end_idx]
            
            distances = (torch.sum(chunk**2, dim=1, keepdim=True) 
                      + torch.sum(self._embedding.weight**2, dim=1)
                      - 2 * torch.matmul(chunk, self._embedding.weight.t()))
            
            indices = torch.argmin(distances, dim=1).unsqueeze(1)
            encodings = torch.zeros(indices.shape[0], self._num_embeddings, device=inputs.device)
            encodings.scatter_(1, indices, 1)
            
            quantized_chunk = torch.matmul(encodings, self._embedding.weight)
            
            e_latent_loss = F.mse_loss(quantized_chunk.detach(), chunk)
            q_latent_loss = F.mse_loss(quantized_chunk, chunk.detach())
            loss = q_latent_loss + self._commitment_cost * e_latent_loss
            
            quantized_chunk = chunk + (quantized_chunk - chunk).detach()
            
            all_quantized.append(quantized_chunk)
            all_indices.append(indices)
            total_loss += loss.item() * chunk.size(0)
            total_samples += chunk.size(0)
            
            if i % 10 == 0 and i > 0:
                clear_cache()
        
        quantized = torch.cat(all_quantized)
        quantized = quantized.reshape(input_shape)
        indices = torch.cat(all_indices)
        avg_loss = total_loss / total_samples
        
        encodings = torch.zeros(indices.shape[0], self._num_embeddings, device=inputs.device)
        for i in range(indices.shape[0]):
            encodings[i, indices[i, 0]] = 1
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        quantized = quantized.permute(0, 2, 1).contiguous()
        indices = indices.reshape(input_shape[0], -1)
        
        return quantized, avg_loss, indices, perplexity


class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.means = None
        self.stds = None
        
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(self.num_features))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    
    def normalize(self, x):
        x = x.contiguous()
        self.means = x.mean(dim=2, keepdim=True).detach()
        self.stds = torch.sqrt(x.var(dim=2, keepdim=True, unbiased=False) + self.eps).detach()
        
        x_norm = (x - self.means) / self.stds
        
        if self.affine:
            x_norm = x_norm * self.affine_weight.reshape(1, -1, 1) + self.affine_bias.reshape(1, -1, 1)
            
        return x_norm
    
    def denormalize(self, x, means=None, stds=None):
        x = x.contiguous()
        if self.affine:
            x = (x - self.affine_bias.reshape(1, -1, 1)) / (self.affine_weight.reshape(1, -1, 1) + self.eps)
        
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
        
    def forward(self, x, normalize=True):
        x = x.contiguous()
        if normalize:
            x_norm = self.revin.normalize(x)
        else:
            x_norm = x
        
        if x_norm.size(0) * x_norm.size(1) > 16384:
            batch_size = max(1, 16384 // x_norm.size(1))
            outputs = []
            vq_losses = []
            indices_list = []
            perplexities = []
            
            for i in range(0, x_norm.size(0), batch_size):
                chunk = x_norm[i:i+batch_size]
                z = self.encoder(chunk)
                q, vq_loss, idx, perp = self.vq(z)
                recon = self.decoder(q)
                
                outputs.append(recon)
                vq_losses.append(vq_loss)
                indices_list.append(idx)
                perplexities.append(perp)
                clear_cache()
            
            x_recon = torch.cat(outputs, dim=0)
            vq_loss = sum(vq_losses) / len(vq_losses)
            indices = torch.cat(indices_list, dim=0)
            perplexity = sum(perplexities) / len(perplexities)
        else:
            z = self.encoder(x_norm)
            quantized, vq_loss, indices, perplexity = self.vq(z)
            x_recon = self.decoder(quantized)
        
        if normalize:
            x_recon = self.revin.denormalize(x_recon)
            
        return x_recon, vq_loss, indices, perplexity
    
    def encode(self, x, normalize=True):
        x = x.contiguous()
        if normalize:
            x_norm = self.revin.normalize(x)
        else:
            x_norm = x
        
        if x_norm.size(0) * x_norm.size(1) > 16384:
            batch_size = max(1, 16384 // x_norm.size(1))
            indices_list = []
            
            for i in range(0, x_norm.size(0), batch_size):
                chunk = x_norm[i:i+batch_size]
                z = self.encoder(chunk)
                _, _, idx, _ = self.vq(z)
                indices_list.append(idx)
                clear_cache()
            
            return torch.cat(indices_list, dim=0)
        else:
            z = self.encoder(x_norm)
            _, _, indices, _ = self.vq(z)
            return indices
    
    def decode(self, indices, means=None, stds=None):
        device = indices.device
        
        if indices.size(0) * indices.size(1) * self.num_embeddings > 65536:
            batch_size = max(1, 65536 // (indices.size(1) * self.num_embeddings))
            outputs = []
            
            for i in range(0, indices.size(0), batch_size):
                chunk_indices = indices[i:i+batch_size]
                
                encodings = torch.zeros(chunk_indices.shape[0], chunk_indices.shape[1], self.num_embeddings, device=device)
                for j in range(chunk_indices.shape[0]):
                    encodings[j].scatter_(1, chunk_indices[j].unsqueeze(1), 1)
                
                embedding_weight = self.vq._embedding.weight.to(device)
                quantized = torch.matmul(encodings, embedding_weight)
                quantized = quantized.permute(0, 2, 1).contiguous()
                
                chunk_recon = self.decoder(quantized)
                outputs.append(chunk_recon)
                clear_cache()
            
            x_recon = torch.cat(outputs, dim=0)
        else:
            embedding_weight = self.vq._embedding.weight.to(device)
            
            encodings = torch.zeros(indices.shape[0], indices.shape[1], self.num_embeddings, device=device)
            for i in range(indices.shape[0]):
                encodings[i].scatter_(1, indices[i].unsqueeze(1), 1)
            
            quantized = torch.matmul(encodings, embedding_weight)
            quantized = quantized.permute(0, 2, 1).contiguous()
            
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
    
    def get_codebook_stats(self):
        return {
            'codebook_size': self.num_embeddings,
            'embedding_dim': self.embedding_dim,
            'embeddings': self.vq._embedding.weight.detach().cpu().numpy()
        }
