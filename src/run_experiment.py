"""
Command-line interface for running text classification experiments.

Usage:
    python -m src.run_experiment --dataset ag_news --model logistic_regression
"""

import argparse
import time
import json
import os
from datetime import datetime
from typing import Dict, Any
import numpy as np

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.base import clone

from src.data import (
    load_text_classification_dataset, 
    print_dataset_stats,
    get_dataset_info
)
from src.models import (
    create_model_pipeline,
    get_hyperparameter_grid, 
    get_available_models,
    get_model_display_name
)
from src.eval import evaluate_model_performance, print_evaluation_results
from src.vectorizer import log_vectorizer_config


def run_hyperparameter_search(
    pipeline, 
    param_grid: Dict[str, Any],
    train_texts, train_labels,
    val_texts, val_labels,
    cv_folds: int = 5,
    n_jobs: int = -1
) -> tuple:
    """
    Run hyperparameter search using GridSearchCV.
    
    Returns:
        Tuple of (best_estimator, best_params, cv_results)
    """
    print(f"\nRunning hyperparameter search with {cv_folds}-fold CV...")
    print(f"Parameter grid: {param_grid}")
    
    # Combine train and validation for hyperparameter search
    combined_texts = train_texts + val_texts
    combined_labels = train_labels + val_labels
    
    # Setup GridSearchCV
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='f1_macro',  # Select by Macro-F1 as specified
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
        n_jobs=n_jobs,
        verbose=1
    )
    
    # Fit grid search
    start_time = time.time()
    grid_search.fit(combined_texts, combined_labels)
    search_time = time.time() - start_time
    
    print(f"Hyperparameter search completed in {search_time:.2f} seconds")
    print(f"Best CV Macro-F1: {grid_search.best_score_:.4f}")
    print(f"Best parameters: {grid_search.best_params_}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.cv_results_


def train_final_model(best_estimator, train_texts, train_labels, val_texts, val_labels):
    """
    Retrain the best model on combined train+validation data.
    """
    print(f"\nRetraining best model on train+validation data...")
    
    # Combine train and validation data
    combined_texts = train_texts + val_texts
    combined_labels = train_labels + val_labels
    
    # Clone the best estimator and retrain
    final_model = clone(best_estimator)
    start_time = time.time()
    final_model.fit(combined_texts, combined_labels)
    train_time = time.time() - start_time
    
    print(f"Final model training completed in {train_time:.2f} seconds")
    
    return final_model


def run_experiment(dataset_name: str, model_name: str) -> Dict[str, Any]:
    """
    Run complete experiment: data loading, hyperparameter search, training, evaluation.
    
    Args:
        dataset_name: Name of dataset to use
        model_name: Name of model to train
        
    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*80}")
    print(f"STARTING EXPERIMENT: {dataset_name.upper()} + {get_model_display_name(model_name)}")
    print(f"{'='*80}")
    
    # Load dataset with validation
    print(f"Loading dataset: {dataset_name}")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
        load_text_classification_dataset(dataset_name)
    )
    
    # Print dataset statistics with validation
    print_dataset_stats(
        train_texts, train_labels, 
        val_texts, val_labels,
        test_texts, test_labels,
        dataset_name
    )
    
    # Log experiment configuration
    print(f"EXPERIMENT CONFIGURATION:")
    print(f"  Model: {get_model_display_name(model_name)}")
    print(f"  Dataset: {dataset_name}")
    print(f"  Train+Val combined: {len(train_texts) + len(val_texts)} samples")
    print(f"  Test (held-out): {len(test_texts)} samples")
    print(f"  Validation strategy: 5-fold stratified CV")
    print(f"  Selection metric: Macro-F1")
    print(f"  Final evaluation: Test set only")
    print()
    
    # Create model pipeline
    pipeline = create_model_pipeline(model_name)
    param_grid = get_hyperparameter_grid(model_name)
    
    # Validate pipeline configuration
    log_pipeline_validation(pipeline, param_grid)
    
    # Run hyperparameter search
    best_estimator, best_params, cv_results = run_hyperparameter_search(
        pipeline, param_grid,
        train_texts, train_labels,
        val_texts, val_labels
    )
    
    # Train final model on train+validation
    final_model = train_final_model(
        best_estimator, 
        train_texts, train_labels,
        val_texts, val_labels
    )
    
    # Evaluate on test set
    print(f"\nEvaluating on test set...")
    test_predictions = final_model.predict(test_texts)
    
    # Compute evaluation metrics
    results = evaluate_model_performance(
        test_labels, test_predictions,
        dataset_name=dataset_name,
        model_name=get_model_display_name(model_name)
    )
    
    # Add experiment metadata
    results['best_params'] = best_params
    results['model_name_key'] = model_name
    
    return results


def log_pipeline_validation(pipeline, param_grid: Dict[str, Any]) -> None:
    """Log pipeline configuration for reproducibility validation."""
    print(f"PIPELINE VALIDATION:")
    print(f"  Pipeline steps: {[step[0] for step in pipeline.steps]}")
    
    # Validate vectorizer is inside pipeline
    vectorizer_in_pipeline = any(step[0] == 'vectorizer' for step in pipeline.steps)
    classifier_in_pipeline = any(step[0] == 'classifier' for step in pipeline.steps)
    
    assert vectorizer_in_pipeline, "Vectorizer must be inside pipeline to prevent data leakage"
    assert classifier_in_pipeline, "Classifier must be inside pipeline"
    print(f"  ✓ Vectorizer inside pipeline (prevents leakage)")
    print(f"  ✓ Classifier inside pipeline")
    
    # Log TF-IDF configuration for consistency
    vectorizer = pipeline.named_steps['vectorizer']
    log_vectorizer_config(vectorizer)
    
    # Log parameter grid
    print(f"\nHYPERPARAMETER GRID:")
    print(f"  Grid size: {len(list(param_grid.keys()))} parameters")
    for param, values in param_grid.items():
        print(f"    {param}: {values}")
    print()


def save_results_to_file(results: Dict[str, Any], dataset_name: str, model_name: str) -> str:
    """
    Save experiment results to JSON file in results/ directory.
    
    Args:
        results: Dictionary with experiment results
        dataset_name: Name of dataset
        model_name: Model name key (e.g., 'logistic_regression')
        
    Returns:
        Path to saved file
    """
    # Create results directory if it doesn't exist
    os.makedirs("results/tables", exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/tables/{dataset_name}_{model_name}_{timestamp}.json"
    
    # Prepare results for JSON serialization (convert numpy arrays)
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, np.integer):
            serializable_results[key] = int(value)
        elif isinstance(value, np.floating):
            serializable_results[key] = float(value)
        else:
            serializable_results[key] = value
    
    # Add metadata
    serializable_results['experiment_timestamp'] = timestamp
    serializable_results['experiment_date'] = datetime.now().isoformat()
    
    # Save to file
    with open(filename, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    return filename


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run text classification experiment with shallow learning models"
    )
    parser.add_argument(
        '--dataset', 
        choices=['ag_news', 'trec', 'rotten_tomatoes'],
        required=True,
        help='Dataset to use for experiment'
    )
    parser.add_argument(
        '--model',
        choices=get_available_models(),
        required=True, 
        help='Model to train and evaluate'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.dataset not in ['ag_news', 'trec', 'rotten_tomatoes']:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    
    if args.model not in get_available_models():
        raise ValueError(f"Unsupported model: {args.model}")
    
    try:
        # Run experiment
        results = run_experiment(args.dataset, args.model)
        
        # Print results
        print_evaluation_results(results)
        
        # Print best hyperparameters
        print("BEST HYPERPARAMETERS:")
        for param, value in results['best_params'].items():
            print(f"  {param}: {value}")
        
        # Save results to file
        results_file = save_results_to_file(results, args.dataset, args.model)
        print(f"\nResults saved to: {results_file}")
        
        print(f"\nExperiment completed successfully!")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
