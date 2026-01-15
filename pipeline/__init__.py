# Pipeline Module
from .ingestion import ingest_novels
from .training import train_consistency_model
from .inference import predict_consistency, generate_predictions
