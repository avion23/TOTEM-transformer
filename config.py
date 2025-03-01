import torch

# Device selection - prioritize MPS for Apple Silicon
DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
USE_FLOAT16 = False  # Flag to enable/disable float16 precision

# TOTEM VQVAE settings
EMBEDDING_DIM = 64
CODEBOOK_SIZE = 256
COMMITMENT_COST = 0.5  # Increased from 0.25 to encourage better codebook usage
COMPRESSION_FACTOR = 4

# Encoder/Decoder architecture
NUM_HIDDENS = 128
NUM_RESIDUAL_LAYERS = 2
NUM_RESIDUAL_HIDDENS = 64

# NanoGPT settings
CONTEXT_LENGTH = 96
OUT_LENGTH = 24
MODEL_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 6
DROPOUT = 0.1

# Training settings
BATCH_SIZE = 16  # Reduced from 32
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4