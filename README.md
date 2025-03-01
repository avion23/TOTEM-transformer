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

## Conceptual Overview

### Codebook Compression
TOTEM uses vector quantization (not Huffman coding) to compress time series data into discrete tokens. The VQVAE learns a codebook of embedding vectors that best represent patterns in your time series data:

1. The encoder converts segments of time series into continuous vectors
2. Vector quantization maps these vectors to the nearest vectors in the learned codebook using Euclidean distance
3. The resulting sequence of codebook indices (tokens) forms a compact representation
4. The decoder reconstructs the original time series from these tokens

This approach achieves significant data compression (configurable compression factor of 4x by default) while preserving essential patterns.

### Similar Concepts and Comparisons
The TOTEM approach can be better understood through comparison with related concepts:

- **Fourier Transforms**: While Fourier transforms decompose signals into predefined sine and cosine waves, TOTEM learns data-driven patterns that best represent your specific time series data.

- **Radial Basis Function (RBF) Networks**: RBF networks use Gaussian functions to approximate functions and surfaces. Similarly, TOTEM approximates time series with learned patterns, but instead of fixed Gaussian functions, it uses vectors from a learned codebook.

- **Bezier Curves**: While conceptually related in representing complex shapes with simpler components, Bezier curves focus on parametric interpolation for visualization, whereas TOTEM focuses on discrete representation for analysis and forecasting.

- **Word Embeddings (Word2Vec)**: TOTEM's codebook has some conceptual similarity to word embeddings, where patterns with similar meaning have similar representations. However, unlike continuous word embeddings where operations like "king - man + woman ≈ queen" are possible, TOTEM uses discrete tokens with simpler distance-based similarity measures.

### Representation and Pattern Similarity
TOTEM's approach is distinct from the above in key ways:

- Instead of using fixed basis functions, it learns data-driven patterns specific to your dataset
- The codebook vectors represent recurring temporal patterns found in your data
- Similar time series patterns map to the same or similar codebook vectors through Euclidean distance
- The discrete nature of tokens enables efficient transformer-based forecasting models

### Continuity and Smoothness
The TOTEM implementation maintains continuity in reconstructed time series, avoiding jumps at token boundaries:

- The decoder uses 1D convolutional and transpose convolutional layers with appropriate padding
- Residual connections help preserve information through the network
- The reconstruction loss optimizes for smooth transitions between tokens
- Larger receptive fields in the decoder ensure context awareness across token boundaries

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

## Practical Applications

TOTEM enables several time series analysis tasks:

- **Tokenization**: Convert continuous time series into discrete tokens for efficient representation
- **Forecasting**: Train transformer models on token sequences to predict future values
- **Pattern Analysis**: Analyze codebook usage to understand recurring patterns in your data
- **Anomaly Detection**: Identify unusual patterns through reconstruction error or token sequences
- **Compression**: Represent time series compactly for efficient storage or transmission

## Performance Targets

When properly trained, the model should achieve:
- Active codebook utilization: >80% of tokens (ideally >90%)
- Perplexity: >50 (higher is better, max is codebook size)
- Reconstruction MSE: <0.05 for normalized data
- Scale preservation: <1% error in denormalized predictions
- Token diversity: Entropy >5 bits (for 256-token codebook)

## Technical Implementation Notes

- Codebook similarity uses Euclidean distance, not cosine similarity
- Default codebook size is configurable (default: 256 vectors)
- Supports both single-feature and multi-feature time series
- The tokenization process handles arbitrary length time series through chunking
- The model uses a combination of reconstruction loss and commitment loss for training

## References

- Kim, T., & Kim, J. (2021). RevIN: Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift.
- TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis - Talukder et al. (2024)
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning.
- Vaswani, A., et al. (2017). Attention is All You Need.