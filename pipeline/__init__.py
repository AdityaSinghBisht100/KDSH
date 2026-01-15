"""
Pipeline Package
"""
from .train import (
    pretrain_on_novels, 
    train_consistency_classifier, 
    NovelDataset, 
    ConsistencyDataset, 
    contrastive_finetune,
    train_sbert_consistency,
    train_with_rationale
)
from .inference import (
    generate_predictions, 
    predict_with_perplexity, 
    predict_with_classifier,
    generate_sbert_predictions
)

__all__ = [
    "pretrain_on_novels",
    "train_consistency_classifier",
    "contrastive_finetune",
    "NovelDataset",
    "ConsistencyDataset",
    "generate_predictions",
    "predict_with_perplexity",
    "predict_with_classifier",
    "train_sbert_consistency",
    "train_with_rationale",
    "generate_sbert_predictions"
]
