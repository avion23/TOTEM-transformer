import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from config import *
from utils import to_device_and_dtype, clear_cache, process_in_chunks


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
        x = x.contiguous()
        
        def process_chunk(model, chunk):
            chunk = chunk.contiguous()
            chunk = model._conv_1(chunk)
            chunk = F.relu(chunk)
            
            chunk = model._conv_2(chunk)
            chunk = F.relu(chunk)
            
            chunk = model._conv_3(chunk)
            chunk = model._residual_stack(chunk)
            chunk = model._pre_vq_conv(chunk)
            
            return chunk.contiguous()
        
        return process_in_chunks(self, x, process_chunk, threshold=32768)


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
        x = x.contiguous()
        
        def process_chunk(model, chunk):
            chunk = chunk.contiguous()
            chunk = model._conv_1(chunk)
            chunk = model._residual_stack(chunk)
            
            chunk = model._conv_trans_1(chunk)
            chunk = F.relu(chunk)
            
            chunk = model._conv_trans_2(chunk)
            
            return chunk.contiguous()
        
        return process_in_chunks(self, x, process_chunk, threshold=32768)


class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1.0 / self._num_embeddings, 1.0 / self._num_embeddings)
        self._embedding.weight.requires_grad_(True)
        self._commitment_cost = commitment_cost
        
    def forward(self, inputs):
        inputs = inputs.contiguous()
        
        def process_chunk(model, chunk):
            return model._forward_standard(chunk)
        
        threshold = 65536 // self._num_embeddings
        return process_in_chunks(self, inputs, process_chunk, threshold=threshold)
    
    def _forward_standard(self, inputs):
        inputs = inputs.permute(0, 2, 1).contiguous()
        input_shape = inputs.shape
        
        flat_input = inputs.reshape(-1, self._embedding_dim).contiguous()
        
        distances = torch.cdist(flat_input, self._embedding.weight, p=2)
        
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        quantized = torch.matmul(encodings, self._embedding.weight).reshape(input_shape).contiguous()
        
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        quantized = quantized.permute(0, 2, 1).contiguous()
        
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        encoding_indices = encoding_indices.reshape(input_shape[0], -1).contiguous()
        
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
        x = x.contiguous()
        self.means = x.mean(dim=2, keepdim=True).detach()
        self.stds = torch.sqrt(x.var(dim=2, keepdim=True, unbiased=False) + self.eps).detach()
        
        x_norm = (x - self.means) / self.stds
        return x_norm.contiguous()
    
    def denormalize(self, x, means=None, stds=None):
        x = x.contiguous()
        means = means if means is not None else self.means
        stds = stds if stds is not None else self.stds
        
        if means is None or stds is None:
            raise ValueError("No normalization statistics available for denormalization")
            
        x = x * stds + means
        return x.contiguous()


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
    
    def _standard_forward(self, x_norm):
        z = self.encoder(x_norm)
        quantized, vq_loss, indices, perplexity = self.vq(z)
        x_recon = self.decoder(quantized)
        return x_recon, vq_loss, indices, perplexity
        
    def forward(self, x, normalize=True):
        x = x.contiguous()
        if normalize:
            x_norm = self.revin.normalize(x)
        else:
            x_norm = x.clone().contiguous()
        
        def process_chunk(model, chunk):
            try:
                return model._standard_forward(chunk)
            except RuntimeError as e:
                if "view size is not compatible" in str(e):
                    z = model.encoder(chunk)
                    z = z.contiguous()
                    quantized, vq_loss, indices, perplexity = model.vq(z)
                    x_recon = model.decoder(quantized)
                    return x_recon, vq_loss, indices, perplexity
                else:
                    raise e
        
        x_recon, vq_loss, indices, perplexity = process_in_chunks(self, x_norm, process_chunk, threshold=16384)
        
        if normalize:
            x_recon = self.revin.denormalize(x_recon)
            
        return x_recon, vq_loss, indices, perplexity
    
    def encode(self, x, normalize=True):
        x = x.contiguous()
        if normalize:
            x_norm = self.revin.normalize(x)
        else:
            x_norm = x.clone().contiguous()
        
        def process_chunk(model, chunk):
            z = model.encoder(chunk)
            _, _, indices, _ = model.vq(z)
            return indices
        
        indices = process_in_chunks(self, x_norm, process_chunk, threshold=16384)[2]
        return indices.contiguous()
    
    def decode(self, indices, means=None, stds=None):
        device = indices.device
        indices = indices.contiguous()
        
        def process_chunk(model, chunk_indices):
            one_hot = F.one_hot(chunk_indices, num_classes=model.num_embeddings).float()
            one_hot = one_hot.view(one_hot.size(0), -1, model.num_embeddings).contiguous()
            one_hot = one_hot.transpose(1, 2).contiguous()
            
            embedding_weight = model.vq._embedding.weight.to(device)
            quantized = torch.bmm(one_hot, embedding_weight.expand(one_hot.size(0), -1, -1))
            quantized = quantized.transpose(1, 2).contiguous()
            
            return model.decoder(quantized)
        
        threshold = 65536 // self.num_embeddings
        x_recon = process_in_chunks(self, indices, process_chunk, threshold=threshold)[0]
        
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
            
        return x_recon.contiguous()