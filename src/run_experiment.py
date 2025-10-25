"""
Command-line interface for running text classification experiments.

Usage:
    python -m src.run_experiment --dataset ag_news --model logistic_regression
"""

import argparse
import time
from typing import Dict, Any

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
    print(f"Starting experiment: {dataset_name} + {get_model_display_name(model_name)}")
    
    # Load dataset
    print(f"\nLoading dataset: {dataset_name}")
    train_texts, train_labels, val_texts, val_labels, test_texts, test_labels = (
        load_text_classification_dataset(dataset_name)
    )
    
    # Print dataset statistics
    print_dataset_stats(
        train_texts, train_labels, 
        val_texts, val_labels,
        test_texts, test_labels,
        dataset_name
    )
    
    # Create model pipeline
    pipeline = create_model_pipeline(model_name)
    param_grid = get_hyperparameter_grid(model_name)
    
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
        
        print(f"\nExperiment completed successfully!")
        
    except Exception as e:
        print(f"Experiment failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
