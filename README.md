# TOTEM Transformer Implementation

## Quick Start

```bash
# Train VQVAE model
python3 train_totem.py --mode vqvae --model_type multi --epochs 10

# Fine-tune VQVAE to improve codebook utilization
python3 train_totem.py --mode finetune --vqvae_path models/best_vqvae.pt --epochs 3

# Train transformer on encoded tokens
python3 train_totem.py --mode transformer --vqvae_path models/vqvae_fine_tuned.pt --epochs 15

# Generate forecasts
python3 run_totem.py --vqvae models/vqvae_fine_tuned.pt --transformer models/best_transformer.pt --steps 30
```

## Key Components

- **VQVAE**: Encodes time series into discrete tokens
  - Improved encoder/decoder with residual connections
  - Scale-aware normalization with RevIN
  - Enhanced codebook utilization monitoring

- **NanoGPT**: Forecasts token sequences
  - Multi-head self-attention
  - Dropout for regularization
  - Top-K sampling support

- **Analysis Tools**: Evaluate model performance
  - Reconstruction quality metrics
  - Codebook utilization tracking
  - Sequence diversity analysis

## Advanced Usage

### Analyzing Codebook Utilization
```bash
python3 test_model.py
```

### Improving an Underutilized Codebook
```bash
python3 improve_codebook.py
```

### Multi-Feature Testing
```bash
python3 multi_feature_test.py
```

## Code Structure

- `vqvae.py`: Core VQVAE implementation with RevIN normalization
- `nanogpt.py`: Transformer model for sequence prediction
- `data.py`: Data loading and processing utilities
- `config.py`: Configuration parameters
- `analysis.py`: Analysis and visualization tools
- `training.py`: Training utilities for both VQVAE and Transformer
- `run_totem.py`: Main inference script
- `train_totem.py`: Unified training script
- `test_model.py`: Model evaluation utilities
- `improve_codebook.py`: Codebook improvement tools

## Improvements from Original Implementation

1. **Enhanced VQVAE Architecture**
   - Proper residual connections in encoder/decoder
   - ResidualStack modules for better feature extraction
   - Improved encoder/decoder with convolutional architecture

2. **Codebook Utilization**
   - Perplexity tracking to monitor codebook diversity
   - Codebook reset mechanism for dead entries
   - Detailed analysis tools for codebook usage

3. **Scale Preservation**
   - RevIN normalization for scale-aware compression
   - Enhanced normalization handling in encode/decode process
   - Denormalization support with stored statistics

4. **NanoGPT Improvements**
   - Enhanced transformer architecture with proper initialization
   - Improved attention mechanism with dropout
   - Better generation with temperature control and top-k sampling

5. **Reproducibility**
   - Fixed random seeds for consistent results
   - Proper chunking for handling large datasets
   - Consistent data processing across functions

6. **Error Handling**
   - Extensive error checking for file existence
   - Graceful handling of model loading issues
   - Detailed error reporting with stack traces

7. **Apple Silicon Optimization**
   - MPS backend support for Metal Performance Shaders
   - Float16 precision option for improved performance
   - Efficient memory management with cache clearing

## Performance Targets

When properly trained, the model should achieve:
- Active codebook utilization: >80% of tokens (ideally >90%)
- Perplexity: >50 (higher is better, max is codebook size)
- Reconstruction MSE: <0.05 for normalized data
- Scale preservation: <1% error in denormalized predictions
- Token diversity: Entropy >5 bits (for 256-token codebook)

## References

- Kim, T., & Kim, J. (2021). RevIN: Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift.
- TOTEM: Task-oriented Tokenization with Embedding Matrices - Original implementation
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning.
- Vaswani, A., et al. (2017). Attention is All You Need.