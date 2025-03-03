import torch

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
USE_FLOAT16 = False  # Disabled float16 to prevent type mismatches

EMBEDDING_DIM = 64
CODEBOOK_SIZE = 256
COMMITMENT_COST = 1.5  # Increased from 0.75 to encourage better codebook usage
COMPRESSION_FACTOR = 4

NUM_HIDDENS = 128
NUM_RESIDUAL_LAYERS = 2
NUM_RESIDUAL_HIDDENS = 64

CONTEXT_LENGTH = 96
OUT_LENGTH = 24
MODEL_DIM = 128
NUM_HEADS = 4
NUM_LAYERS = 6
DROPOUT = 0.1

BATCH_SIZE = 32 if DEVICE in ["mps", "cuda"] else 16
EPOCHS = 5
LR = 1e-4
WEIGHT_DECAY = 1e-4

VQVAE_CHUNK_THRESHOLD = 16384
DECODER_CHUNK_THRESHOLD = 32768
VQ_CHUNK_THRESHOLD = 65536 // CODEBOOK_SIZE
CHUNK_SIZE = 1000
GRAD_ACCUMULATION_STEPS = 1