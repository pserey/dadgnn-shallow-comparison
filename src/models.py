"""
Shallow learning models and hyperparameter grids for text classification.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from typing import Dict, Any, Type
from sklearn.base import BaseEstimator

from src.vectorizer import get_default_vectorizer


# Model classes mapping
MODEL_CLASSES = {
    "logistic_regression": LogisticRegression,
    "linear_svm": LinearSVC, 
    "multinomial_nb": MultinomialNB,
}


# Hyperparameter grids for each model
HYPERPARAMETER_GRIDS = {
    "logistic_regression": {
        "classifier__C": [0.5, 1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
        "classifier__penalty": ["l2"],
        "classifier__max_iter": [2000],
    },
    "linear_svm": {
        "classifier__C": [0.5, 1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
        "classifier__loss": ["squared_hinge"],
    },
    "multinomial_nb": {
        "classifier__alpha": [0.1, 0.5, 1.0, 2.0],
    }
}


def create_model_pipeline(model_name: str) -> Pipeline:
    """
    Create a scikit-learn pipeline with TF-IDF vectorizer and classifier.
    
    Args:
        model_name: Name of the model ('logistic_regression', 'linear_svm', 'multinomial_nb')
        
    Returns:
        Pipeline with vectorizer and classifier
    """
    if model_name not in MODEL_CLASSES:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_CLASSES.keys())}")
    
    # Get model class and create instance with default parameters
    model_class = MODEL_CLASSES[model_name]
    
    # Set some reasonable defaults for reproducibility
    if model_name == "logistic_regression":
        classifier = model_class(random_state=42, solver='lbfgs', max_iter=2000)
    elif model_name == "linear_svm": 
        classifier = model_class(random_state=42)
    else:  # multinomial_nb
        classifier = model_class()
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', get_default_vectorizer()),
        ('classifier', classifier)
    ])
    
    return pipeline


def get_hyperparameter_grid(model_name: str) -> Dict[str, Any]:
    """
    Get hyperparameter grid for a given model.
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary with hyperparameter grid
    """
    if model_name not in HYPERPARAMETER_GRIDS:
        raise ValueError(f"No hyperparameter grid for model: {model_name}")
    
    return HYPERPARAMETER_GRIDS[model_name].copy()


def get_available_models() -> list:
    """Get list of available model names."""
    return list(MODEL_CLASSES.keys())


def get_model_display_name(model_name: str) -> str:
    """Get human-readable display name for a model.""" 
    display_names = {
        "logistic_regression": "TF-IDF + Logistic Regression",
        "linear_svm": "TF-IDF + Linear SVM",
        "multinomial_nb": "TF-IDF + Multinomial Naive Bayes"
    }
    return display_names.get(model_name, model_name)
