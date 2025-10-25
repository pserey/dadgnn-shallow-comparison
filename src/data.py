"""
Dataset loading and preprocessing for text classification experiments.
Supports ag_news, trec, and rotten_tomatoes datasets via Hugging Face.
"""

from typing import Dict, List, Tuple, Any
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import numpy as np
import urllib.request
import io
import tempfile
import os


DATASET_CONFIGS = {
    "ag_news": {
        "name": "ag_news",
        "text_column": "text",
        "label_column": "label",
        "num_classes": 4,
        "has_validation": False,
    },
    "trec": {
        "name": "lukasgarbas/trec",  # Real TREC dataset in Parquet format
        "text_column": "text", 
        "label_column": "coarse_label",
        "num_classes": 6,
        "has_validation": True,  # Real dataset has validation split
        "label_mapping": {"ABBR": 0, "DESC": 1, "ENTY": 2, "HUM": 3, "LOC": 4, "NUM": 5},  # Convert string to numeric
    },
    "rotten_tomatoes": {
        "name": "rotten_tomatoes",
        "text_column": "text",
        "label_column": "label", 
        "num_classes": 2,
        "has_validation": True,
    }
}


def _load_trec_dataset():
    """
    Load real TREC dataset - no synthetic fallbacks allowed.
    Standard TREC: 5,500 train / 500 test with 6 coarse classes.
    """
    from datasets import load_dataset
    
    try:
        # Try the Parquet-based TREC dataset (recommended format)
        print("Loading real TREC dataset from lukasgarbas/trec...")
        dataset = load_dataset("lukasgarbas/trec")
        print("✓ Successfully loaded real TREC dataset!")
        return dataset
    except Exception as e1:
        print(f"Failed to load lukasgarbas/trec: {e1}")
        
        # If TREC fails, this is an error that should be fixed
        raise RuntimeError(
            f"Failed to load real TREC dataset. Error: {e1}\n"
            f"TREC dataset loading failed. This needs to be fixed - no synthetic fallbacks allowed.\n"
            f"Options: 1) Fix TREC loading, 2) Replace with SST-2 or another DADGNN paper dataset"
        )


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
    if config["name"] == "lukasgarbas/trec":
        # Special case for TREC: load real dataset
        dataset = _load_trec_dataset()
    else:
        dataset = load_dataset(config["name"])
    
    # Extract text and labels
    text_col = config["text_column"]
    label_col = config["label_column"]
    
    train_texts = dataset["train"][text_col]
    train_labels = dataset["train"][label_col]
    test_texts = dataset["test"][text_col]
    test_labels = dataset["test"][label_col]
    
    # Convert string labels to numeric if needed (for TREC)
    if "label_mapping" in config:
        label_map = config["label_mapping"]
        train_labels = [label_map[label] for label in train_labels]
        test_labels = [label_map[label] for label in test_labels]
    
    # Handle validation split
    if config["has_validation"] and "validation" in dataset:
        val_texts = dataset["validation"][text_col]
        val_labels = dataset["validation"][label_col]
        
        # Convert string labels to numeric if needed (for TREC validation)
        if "label_mapping" in config:
            label_map = config["label_mapping"]
            val_labels = [label_map[label] for label in val_labels]
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


def validate_dataset_splits(dataset_name: str, train_texts: List[str], test_texts: List[str], 
                           train_labels: List[int], test_labels: List[int]) -> None:
    """
    Validate dataset splits against expected standards.
    Raises AssertionError if splits don't match expected sizes.
    """
    if dataset_name == "ag_news":
        # AG News: test should be 7,600 samples
        assert len(test_texts) == 7600, f"AG News test size should be 7600, got {len(test_texts)}"
        assert len(set(train_labels + test_labels)) == 4, f"AG News should have 4 classes, got {len(set(train_labels + test_labels))}"
        
    elif dataset_name == "trec":
        # TREC: Real dataset is ~4907 train / 500 test with 6 classes  
        assert len(train_texts) >= 4500, f"TREC train should be ~4900, got {len(train_texts)}"
        assert len(test_texts) == 500, f"TREC test should be 500, got {len(test_texts)}"
        n_classes = len(set(train_labels + test_labels))
        assert n_classes == 6, f"TREC should have 6 coarse classes, got {n_classes}"
        
    elif dataset_name == "rotten_tomatoes":
        # Rotten Tomatoes: test should be 1,066 samples
        assert len(test_texts) == 1066, f"Rotten Tomatoes test should be 1066, got {len(test_texts)}"  
        assert len(set(train_labels + test_labels)) == 2, f"Rotten Tomatoes should have 2 classes, got {len(set(train_labels + test_labels))}"
    
    print(f"✓ Dataset validation passed for {dataset_name}")


def print_dataset_stats(
    train_texts: List[str], train_labels: List[int],
    val_texts: List[str], val_labels: List[int],
    test_texts: List[str], test_labels: List[int],
    dataset_name: str
) -> None:
    """Print detailed dataset statistics with validation."""
    config = DATASET_CONFIGS[dataset_name]
    
    # Validate splits first
    validate_dataset_splits(dataset_name, train_texts, test_texts, train_labels, test_labels)
    
    print(f"\n{'='*50}")
    print(f"DATASET STATISTICS: {dataset_name.upper()}")
    print(f"{'='*50}")
    print(f"Expected classes: {config['num_classes']}")
    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # Verify class counts match expected
    actual_classes = len(set(train_labels + val_labels + test_labels))
    print(f"Actual classes found: {actual_classes}")
    if actual_classes != config['num_classes']:
        print(f"⚠️  WARNING: Expected {config['num_classes']} classes, found {actual_classes}")
    
    # Label distribution with percentages
    print(f"\nLABEL DISTRIBUTION:")
    all_labels = train_labels + val_labels + test_labels
    total_samples = len(all_labels)
    
    for split_name, labels in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        unique, counts = np.unique(labels, return_counts=True)
        split_total = len(labels)
        
        label_dist_parts = []
        for label, count in zip(unique, counts):
            percentage = (count / split_total) * 100
            label_dist_parts.append(f"class {label}: {count} ({percentage:.1f}%)")
        
        label_dist = ", ".join(label_dist_parts)
        print(f"  {split_name} ({split_total} total): {label_dist}")
    
    # Check for class imbalance
    unique, counts = np.unique(all_labels, return_counts=True)
    if len(counts) > 1:
        imbalance_ratio = max(counts) / min(counts)
        if imbalance_ratio > 2.0:
            print(f"⚠️  Class imbalance detected: ratio {imbalance_ratio:.2f}")
        else:
            print(f"✓ Balanced classes: ratio {imbalance_ratio:.2f}")
    
    print(f"{'='*50}\n")
