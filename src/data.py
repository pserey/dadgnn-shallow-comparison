"""
Dataset loading and preprocessing for text classification experiments.
Supports ag_news, trec, and rotten_tomatoes datasets via Hugging Face.
"""

from typing import Dict, List, Tuple, Any
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np


DATASET_CONFIGS = {
    "ag_news": {
        "name": "ag_news",
        "text_column": "text",
        "label_column": "label",
        "num_classes": 4,
        "has_validation": False,
    },
    "trec": {
        "name": "trec",
        "text_column": "text", 
        "label_column": "coarse_label",
        "num_classes": 6,
        "has_validation": False,
    },
    "rotten_tomatoes": {
        "name": "rotten_tomatoes",
        "text_column": "text",
        "label_column": "label", 
        "num_classes": 2,
        "has_validation": True,
    }
}


def load_text_classification_dataset(
    dataset_name: str, 
    validation_split: float = 0.1, 
    seed: int = 42
) -> Tuple[List[str], List[int], List[str], List[int], List[str], List[int]]:
    """
    Load and split a text classification dataset.
    
    Args:
        dataset_name: Name of the dataset ('ag_news', 'trec', 'rotten_tomatoes')
        validation_split: Fraction of training data to use for validation if no val split exists
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_texts, train_labels, val_texts, val_labels, test_texts, test_labels)
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}. "
                        f"Supported: {list(DATASET_CONFIGS.keys())}")
    
    config = DATASET_CONFIGS[dataset_name]
    
    # Load dataset from Hugging Face
    dataset = load_dataset(config["name"])
    
    # Extract text and labels
    text_col = config["text_column"]
    label_col = config["label_column"]
    
    train_texts = dataset["train"][text_col]
    train_labels = dataset["train"][label_col]
    test_texts = dataset["test"][text_col]
    test_labels = dataset["test"][label_col]
    
    # Handle validation split
    if config["has_validation"] and "validation" in dataset:
        val_texts = dataset["validation"][text_col]
        val_labels = dataset["validation"][label_col]
    else:
        # Create validation split from training data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            train_texts, train_labels,
            test_size=validation_split,
            stratify=train_labels,
            random_state=seed
        )
    
    return (
        list(train_texts), list(train_labels),
        list(val_texts), list(val_labels), 
        list(test_texts), list(test_labels)
    )


def get_dataset_info(dataset_name: str) -> Dict[str, Any]:
    """Get metadata about a dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return DATASET_CONFIGS[dataset_name].copy()


def print_dataset_stats(
    train_texts: List[str], train_labels: List[int],
    val_texts: List[str], val_labels: List[int],
    test_texts: List[str], test_labels: List[int],
    dataset_name: str
) -> None:
    """Print dataset statistics."""
    config = DATASET_CONFIGS[dataset_name]
    
    print(f"\nDataset: {dataset_name}")
    print(f"Classes: {config['num_classes']}")
    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Label distribution
    print("\nLabel distribution:")
    for split_name, labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        unique, counts = np.unique(labels, return_counts=True)
        label_dist = ", ".join([f"class {label}: {count}" for label, count in zip(unique, counts)])
        print(f"  {split_name}: {label_dist}")
