"""
Pipeline Package
"""
from .train import pretrain_on_novels, train_consistency_classifier, NovelDataset, ConsistencyDataset, contrastive_finetune
from .inference import generate_predictions, predict_with_perplexity, predict_with_classifier

__all__ = [
    "pretrain_on_novels",
    "train_consistency_classifier",
    "contrastive_finetune",
    "NovelDataset",
    "ConsistencyDataset",
    "generate_predictions",
    "predict_with_perplexity",
    "predict_with_classifier",
]
