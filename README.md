# TOTEM: Tokenized Time Series Embeddings for Forecasting

## Overview

TOTEM (Tokenized Time Series Embeddings) is a framework for time series analysis and forecasting that combines:
- A Vector Quantized Variational Autoencoder (VQVAE) to compress time series data into discrete tokens
- A Transformer model (NanoGPT) to forecast future token sequences

This approach enables efficient representation, pattern analysis, and accurate forecasting of time series data.

## Key Improvements

1. **Enhanced VQVAE Architecture:** Improved encoder/decoder with proper residual connections
2. **Codebook Utilization:** Perplexity tracking and reset mechanism for dead entries
3. **Scale Preservation:** RevIN normalization for scale-aware compression
4. **NanoGPT Improvements:** Enhanced transformer with proper initialization and attention mechanisms
5. **Reproducibility:** Fixed random seeds and chunking for large datasets
6. **Apple Silicon Optimization:** MPS backend support with float16 precision option

## Features

- **Time Series Tokenization:** Converts continuous time series into discrete tokens
- **Transformer-based Forecasting:** Predicts future token sequences
- **Codebook Analysis:** Provides insights into recurring patterns
- **Anomaly Detection:** Identifies unusual patterns through reconstruction error
- **Data Compression:** Achieves efficient storage and transmission
- **Multi-Feature Support:** Works with both single-variable and multi-variable time series

## Quick Start

1. **Install requirements:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Data Preparation:**
   - TOTEM expects a CSV file with columns: `midprice` (or `close`), `volume` (optional), `reward` (optional)
   ```csv
   midprice,volume,reward
   100.0,10,0.1
   100.1,12,0.2
   100.2,15,0.3
   ```

3. **Training and Forecasting:**
   ```bash
   DATA_PATH=path/to/your/data.csv

   # Train VQVAE model
   python3 train_totem.py --data $DATA_PATH --mode vqvae --model_type multi --epochs 10

   # Fine-tune VQVAE to improve codebook utilization
   python3 train_totem.py --data $DATA_PATH --mode finetune --vqvae_path models/best_vqvae.pt --epochs 3

   # Train transformer on encoded tokens
   python3 train_totem.py --data $DATA_PATH --mode transformer --vqvae_path models/vqvae_fine_tuned.pt --epochs 15

   # Generate forecasts
   python3 run_totem.py --data $DATA_PATH --vqvae models/vqvae_fine_tuned.pt --transformer models/best_transformer.pt --steps 30
   ```

## Conceptual Overview

### VQVAE: Learning a Discrete Representation

The VQVAE learns a codebook of embedding vectors that represent recurring patterns in time series data:

1. The encoder converts segments of time series into continuous vectors
2. Vector quantization maps these vectors to the nearest vectors in the codebook using Euclidean distance
3. The resulting sequence of codebook indices (tokens) forms a compact representation
4. The decoder reconstructs the original time series from these tokens

Unlike fixed basis functions (e.g., Fourier transforms), VQVAE *learns* the patterns that best represent your specific time series data. Similar patterns map to the same or similar codebook vectors, allowing the Transformer to learn dependencies between these patterns for forecasting.
### Similar Concepts and Comparisons
The TOTEM approach can be better understood through comparison with related concepts:

- **Fourier Transforms**: While Fourier transforms decompose signals into predefined sine and cosine waves, TOTEM learns data-driven patterns that best represent your specific time series data.

- **Radial Basis Function (RBF) Networks**: RBF networks use Gaussian functions to approximate functions and surfaces. Similarly, TOTEM approximates time series with learned patterns, but instead of fixed Gaussian functions, it uses vectors from a learned codebook.

- **Bezier Curves**: While conceptually related in representing complex shapes with simpler components, Bezier curves focus on parametric interpolation for visualization, whereas TOTEM focuses on discrete representation for analysis and forecasting.

- **Word Embeddings (Word2Vec)**: TOTEM's codebook has some conceptual similarity to word embeddings, where patterns with similar meaning have similar representations. However, unlike continuous word embeddings where operations like "king - man + woman ≈ queen" are possible, TOTEM uses discrete tokens with simpler distance-based similarity measures.

## Code Structure

- `vqvae.py`: Core VQVAE implementation with RevIN normalization
- `nanogpt.py`: Transformer model (NanoGPT) for sequence prediction
- `data.py`: Data loading and processing utilities
- `config.py`: Configuration parameters
- `analysis.py`: Analysis and visualization tools
- `training.py`: Training utilities for both VQVAE and Transformer
- `run_totem.py`: Main inference script for generating forecasts
- `train_totem.py`: Unified training script for VQVAE and Transformer

## Advanced Usage

### Analyzing Codebook Utilization

```bash
python3 codebook_analysis.py --data $DATA_PATH --model models/best_vqvae.pt --mode multi --plot_dir plots
```

### Improving an Underutilized Codebook

```bash
python3 train_totem.py --data $DATA_PATH --mode finetune --vqvae_path models/best_vqvae.pt --epochs 3
```

## Performance Targets

When properly trained, the model should achieve:
- Active codebook utilization: >80% of tokens (ideally >90%)
- Perplexity: >50 (higher is better, max is codebook size)
- Reconstruction MSE: <0.05 for normalized data
- Scale preservation: <1% error in denormalized predictions

## Troubleshooting

- **"CUDA out of memory" error:** Reduce the `BATCH_SIZE` in `config.py`
- **Poor reconstruction quality:** Increase the `NUM_HIDDENS` and `NUM_RESIDUAL_LAYERS` in `config.py`
- **Inactive codebook:** Finetune the VQVAE with the `--mode finetune` option
- **Apple Silicon issues:** Make sure `DEVICE="mps"` and `USE_FLOAT16=True` in `config.py`

## References

- Kim, T., & Kim, J. (2021). RevIN: Reversible Instance Normalization for Accurate Time-Series Forecasting Against Distribution Shift.
- van den Oord, A., Vinyals, O., & Kavukcuoglu, K. (2017). Neural Discrete Representation Learning.
- Vaswani, A., et al. (2017). Attention is All You Need.
