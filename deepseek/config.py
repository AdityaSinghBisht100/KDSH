import torch
import os

class Config:
    # BDH-GPU parameters from paper
    BDH_DIM = 256
    BDH_HEADS = 4
    BDH_NEURONS = 32768
    BDH_LAYERS = 6
    BDH_DROPOUT = 0.05
    BDH_VOCAB_SIZE = 256
    
    # Model parameters
    HIDDEN_SIZE = 768
    MAX_SEQ_LENGTH = 4096
    CHUNK_SIZE = 2000
    
    # Data paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    NOVEL_DIR = os.path.join(DATA_DIR, "novels")
    TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
    TEST_CSV = os.path.join(DATA_DIR, "test.csv")
    
    # Training
    BATCH_SIZE = 8
    LEARNING_RATE = 2e-5
    EPOCHS = 10
    
    # Inference
    SIMILARITY_THRESHOLD = 0.65
    CONTRADICTION_THRESHOLD = 0.25
    MIN_EVIDENCE_COUNT = 3
    
    # Device
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
config = Config()
