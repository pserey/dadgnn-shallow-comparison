"""
Evaluation metrics and statistical analysis for model performance.
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import List, Tuple, Dict, Any
import scipy.stats as stats


def compute_accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Compute accuracy score."""
    return accuracy_score(y_true, y_pred)


def compute_macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    """Compute macro-averaged F1 score."""
    return f1_score(y_true, y_pred, average='macro')


def compute_confusion_matrix(y_true: List[int], y_pred: List[int]) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred)


def bootstrap_confidence_interval(
    y_true: List[int], 
    y_pred: List[int], 
    metric_func=accuracy_score,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_func: Function to compute metric (default: accuracy_score)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        random_state: Random seed
        
    Returns:
        Tuple of (metric_value, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    n_samples = len(y_true)
    
    # Original metric value
    original_score = metric_func(y_true, y_pred)
    
    # Bootstrap samples
    bootstrap_scores = []
    for _ in range(n_bootstrap):
        # Sample with replacement
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        bootstrap_score = metric_func(y_true[indices], y_pred[indices])
        bootstrap_scores.append(bootstrap_score)
    
    bootstrap_scores = np.array(bootstrap_scores)
    
    # Compute confidence interval
    alpha = 1 - confidence_level
    ci_lower = np.percentile(bootstrap_scores, (alpha/2) * 100)
    ci_upper = np.percentile(bootstrap_scores, (1 - alpha/2) * 100)
    
    return original_score, ci_lower, ci_upper


def bootstrap_accuracy_ci(
    y_true: List[int], 
    y_pred: List[int],
    n_bootstrap: int = 1000,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute 95% bootstrap confidence interval for accuracy.
    
    Returns:
        Tuple of (accuracy, ci_lower, ci_upper)
    """
    return bootstrap_confidence_interval(
        y_true, y_pred, 
        metric_func=accuracy_score,
        n_bootstrap=n_bootstrap,
        confidence_level=0.95,
        random_state=random_state
    )


def format_confusion_matrix(cm: np.ndarray, class_names: List[str] = None) -> str:
    """
    Format confusion matrix for pretty printing.
    
    Args:
        cm: Confusion matrix
        class_names: Optional class names for labeling
        
    Returns:
        Formatted string representation
    """
    if class_names is None:
        class_names = [f"Class {i}" for i in range(cm.shape[0])]
    
    # Calculate column widths
    max_width = max(len(str(cm.max())), max(len(name) for name in class_names))
    
    # Header
    header = "Confusion Matrix:\n"
    header += "True\\Pred" + " " * (max_width - 8)
    for name in class_names:
        header += f"{name:>{max_width+2}}"
    header += "\n"
    
    # Rows
    rows = []
    for i, name in enumerate(class_names):
        row = f"{name:<{max_width}}"
        for j in range(cm.shape[1]):
            row += f"{cm[i, j]:>{max_width+2}}"
        rows.append(row)
    
    return header + "\n".join(rows)


def evaluate_model_performance(
    y_true: List[int], 
    y_pred: List[int],
    dataset_name: str = "",
    model_name: str = "",
    class_names: List[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation with all metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels  
        dataset_name: Name of dataset for reporting
        model_name: Name of model for reporting
        class_names: Optional class names for confusion matrix
        
    Returns:
        Dictionary with all evaluation results
    """
    # Compute metrics
    accuracy = compute_accuracy(y_true, y_pred)
    macro_f1 = compute_macro_f1(y_true, y_pred)
    acc_score, acc_ci_lower, acc_ci_upper = bootstrap_accuracy_ci(y_true, y_pred)
    cm = compute_confusion_matrix(y_true, y_pred)
    
    results = {
        'dataset': dataset_name,
        'model': model_name,
        'accuracy': accuracy,
        'accuracy_ci_lower': acc_ci_lower,
        'accuracy_ci_upper': acc_ci_upper,
        'macro_f1': macro_f1,
        'confusion_matrix': cm,
        'n_samples': len(y_true),
        'n_classes': len(np.unique(y_true))
    }
    
    return results


def print_evaluation_results(results: Dict[str, Any], class_names: List[str] = None) -> None:
    """Print formatted evaluation results."""
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"Dataset: {results['dataset']}")
    print(f"Model: {results['model']}")
    print(f"Test samples: {results['n_samples']}")
    print(f"Classes: {results['n_classes']}")
    print(f"\nMETRICS:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  95% CI: ({results['accuracy_ci_lower']:.4f}, {results['accuracy_ci_upper']:.4f})")
    print(f"  Macro-F1: {results['macro_f1']:.4f}")
    
    print(f"\n{format_confusion_matrix(results['confusion_matrix'], class_names)}")
    print(f"{'='*60}\n")
