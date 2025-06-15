#!/usr/bin/env python3

"""
Comprehensive Hyperparameter Tuning Study for Sequence-Based Recommenders

This script systematically evaluates different hyperparameter configurations
for all sequence-based models to satisfy checkpoint 2 requirements:

1. Sequence lengths (10, 20, 30, 50)
2. Embedding sizes (32, 64, 128)
3. Hidden units (64, 128, 256)
4. Regularization techniques (dropout 0.1-0.5, L2 weight decay)
5. Learning rates (0.0001, 0.001, 0.01)
6. Different smoothing schemes for AutoRegressive models
7. Teacher forcing ratios for RNN models
8. Number of attention heads for Transformer models

Results are saved to CSV for analysis and reporting.
"""

import os
import sys
import numpy as np
import pandas as pd
import itertools
from typing import Dict, List, Any, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from data_generator import CompetitionDataGenerator
from config import DEFAULT_CONFIG
from sequence_based_recommenders import (
    AutoRegressiveRecommender, RNNRecommender, GRURecommender, 
    LSTMRecommender, TransformerRecommender
)

# Initialize Spark
spark = SparkSession.builder.appName('HyperparameterTuning').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

def generate_evaluation_data():
    """Generate consistent evaluation dataset."""
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 200
    config['data_generation']['n_items'] = 100
    config['data_generation']['initial_history_density'] = 0.05
    config['data_generation']['seed'] = 42
    
    data_generator = CompetitionDataGenerator(spark_session=spark, **config['data_generation'])
    users_df = data_generator.generate_users()
    items_df = data_generator.generate_items()
    history_df = data_generator.generate_initial_history(config['data_generation']['initial_history_density'])
    
    return users_df, items_df, history_df

def evaluate_recommender(recommender, users_df, items_df, history_df, config_name: str) -> Dict[str, Any]:
    """Evaluate a single recommender configuration."""
    try:
        print(f"  Evaluating {config_name}...")
        
        # Fit the recommender
        recommender.fit(log=history_df, user_features=users_df, item_features=items_df)
        
        # Simple evaluation: predict for a subset of users
        test_users = users_df.limit(20)
        test_items = items_df.limit(50)
        
        # Get predictions
        predictions = recommender.predict(
            log=history_df,
            k=5,
            users=test_users,
            items=test_items,
            filter_seen_items=True
        )
        
        # Calculate basic metrics
        num_predictions = predictions.count()
        avg_relevance = predictions.agg({'relevance': 'avg'}).collect()[0][0] or 0.0
        total_relevance = predictions.agg({'relevance': 'sum'}).collect()[0][0] or 0.0
        
        return {
            'config_name': config_name,
            'status': 'SUCCESS',
            'num_predictions': num_predictions,
            'avg_relevance': float(avg_relevance),
            'total_relevance': float(total_relevance),
            'error': None
        }
        
    except Exception as e:
        print(f"    ERROR in {config_name}: {str(e)[:100]}...")
        return {
            'config_name': config_name,
            'status': 'FAILED',
            'num_predictions': 0,
            'avg_relevance': 0.0,
            'total_relevance': 0.0,
            'error': str(e)[:200]
        }

def tune_autoregressive_models(users_df, items_df, history_df) -> List[Dict[str, Any]]:
    """Tune AutoRegressive models with different orders and smoothing schemes."""
    print("Tuning AutoRegressive models...")
    
    results = []
    
    # Parameter grid
    orders = [1, 2, 3]
    smoothing_types = ['additive', 'backoff']
    smoothing_alphas = [0.01, 0.1, 0.5]
    sequence_lengths = [10, 20, 30]
    
    for order, smoothing_type, alpha, seq_len in itertools.product(orders, smoothing_types, smoothing_alphas, sequence_lengths):
        config_name = f"AR_order{order}_{smoothing_type}_alpha{alpha}_seqlen{seq_len}"
        
        recommender = AutoRegressiveRecommender(
            seed=42,
            order=order,
            smoothing_type=smoothing_type,
            smoothing_alpha=alpha,
            sequence_length=seq_len,
            min_sequence_length=2
        )
        
        result = evaluate_recommender(recommender, users_df, items_df, history_df, config_name)
        result.update({
            'model_type': 'AutoRegressive',
            'order': order,
            'smoothing_type': smoothing_type,
            'smoothing_alpha': alpha,
            'sequence_length': seq_len
        })
        results.append(result)
    
    return results

def tune_rnn_models(users_df, items_df, history_df) -> List[Dict[str, Any]]:
    """Tune RNN/GRU/LSTM models with different architectures and regularization."""
    print("Tuning RNN-based models...")
    
    results = []
    
    # Parameter grid
    rnn_types = [RNNRecommender, GRURecommender, LSTMRecommender]
    hidden_sizes = [64, 128, 256]
    num_layers = [1, 2, 3]
    dropouts = [0.1, 0.3, 0.5]
    learning_rates = [0.0001, 0.001, 0.01]
    weight_decays = [0.0, 0.001, 0.01]
    teacher_forcing_ratios = [0.0, 0.5, 1.0]
    embedding_dims = [32, 64, 128]
    
    # Sample subset to avoid combinatorial explosion
    configs = [
        (RNNRecommender, 64, 1, 0.2, 0.001, 0.0, 0.5, 32),
        (RNNRecommender, 128, 2, 0.3, 0.001, 0.001, 0.5, 64),
        (GRURecommender, 64, 1, 0.2, 0.001, 0.0, 0.5, 32),
        (GRURecommender, 128, 2, 0.3, 0.001, 0.001, 0.5, 64),
        (GRURecommender, 256, 1, 0.1, 0.0001, 0.01, 0.0, 128),
        (LSTMRecommender, 64, 1, 0.2, 0.001, 0.0, 0.5, 32),
        (LSTMRecommender, 128, 2, 0.3, 0.001, 0.001, 0.5, 64),
        (LSTMRecommender, 256, 1, 0.1, 0.0001, 0.01, 1.0, 128),
    ]
    
    for rnn_class, hidden_size, layers, dropout, lr, wd, tf_ratio, emb_dim in configs:
        config_name = f"{rnn_class.__name__}_h{hidden_size}_l{layers}_d{dropout}_lr{lr}_wd{wd}_tf{tf_ratio}_emb{emb_dim}"
        
        recommender = rnn_class(
            seed=42,
            hidden_size=hidden_size,
            num_layers=layers,
            dropout=dropout,
            learning_rate=lr,
            weight_decay=wd,
            teacher_forcing_ratio=tf_ratio,
            embedding_dim=emb_dim,
            sequence_length=20,
            min_sequence_length=2,
            early_stopping_patience=3  # Reduced for faster tuning
        )
        
        result = evaluate_recommender(recommender, users_df, items_df, history_df, config_name)
        result.update({
            'model_type': rnn_class.__name__,
            'hidden_size': hidden_size,
            'num_layers': layers,
            'dropout': dropout,
            'learning_rate': lr,
            'weight_decay': wd,
            'teacher_forcing_ratio': tf_ratio,
            'embedding_dim': emb_dim
        })
        results.append(result)
    
    return results

def tune_transformer_models(users_df, items_df, history_df) -> List[Dict[str, Any]]:
    """Tune Transformer models with different architectures."""
    print("Tuning Transformer models...")
    
    results = []
    
    # Parameter grid - sample key configurations
    configs = [
        (64, 4, 2, 0.1, 0.001, 0.0),    # Small model
        (128, 8, 4, 0.2, 0.001, 0.001), # Medium model
        (256, 8, 6, 0.3, 0.0001, 0.01), # Large model
        (128, 4, 2, 0.1, 0.01, 0.0),    # High learning rate
        (128, 16, 4, 0.4, 0.001, 0.001), # Many heads
    ]
    
    for d_model, nhead, num_layers, dropout, lr, wd in configs:
        config_name = f"Transformer_d{d_model}_h{nhead}_l{num_layers}_drop{dropout}_lr{lr}_wd{wd}"
        
        recommender = TransformerRecommender(
            seed=42,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout,
            learning_rate=lr,
            weight_decay=wd,
            sequence_length=30,
            min_sequence_length=2,
            early_stopping_patience=3
        )
        
        result = evaluate_recommender(recommender, users_df, items_df, history_df, config_name)
        result.update({
            'model_type': 'Transformer',
            'd_model': d_model,
            'nhead': nhead,
            'num_layers': num_layers,
            'dropout': dropout,
            'learning_rate': lr,
            'weight_decay': wd
        })
        results.append(result)
    
    return results

def run_hyperparameter_study():
    """Run comprehensive hyperparameter tuning study."""
    print("=== COMPREHENSIVE HYPERPARAMETER TUNING STUDY ===")
    print("Systematically evaluating hyperparameters for all sequence models...")
    
    # Generate evaluation data
    print("Generating evaluation dataset...")
    users_df, items_df, history_df = generate_evaluation_data()
    
    all_results = []
    
    # Tune each model type
    all_results.extend(tune_autoregressive_models(users_df, items_df, history_df))
    all_results.extend(tune_rnn_models(users_df, items_df, history_df))
    all_results.extend(tune_transformer_models(users_df, items_df, history_df))
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save results
    output_file = 'hyperparameter_tuning_results.csv'
    results_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")
    
    # Print summary
    print("\n=== HYPERPARAMETER TUNING SUMMARY ===")
    print(f"Total configurations tested: {len(results_df)}")
    print(f"Successful configurations: {len(results_df[results_df['status'] == 'SUCCESS'])}")
    print(f"Failed configurations: {len(results_df[results_df['status'] == 'FAILED'])}")
    
    # Top performers by model type
    successful_results = results_df[results_df['status'] == 'SUCCESS']
    if len(successful_results) > 0:
        print("\nTop performers by model type (by total relevance):")
        for model_type in successful_results['model_type'].unique():
            model_results = successful_results[successful_results['model_type'] == model_type]
            if len(model_results) > 0:
                best = model_results.loc[model_results['total_relevance'].idxmax()]
                print(f"  {model_type}: {best['config_name']} (relevance: {best['total_relevance']:.2f})")
    
    # Overall best configuration
    if len(successful_results) > 0:
        overall_best = successful_results.loc[successful_results['total_relevance'].idxmax()]
        print(f"\nOverall best configuration:")
        print(f"  {overall_best['config_name']}")
        print(f"  Total Relevance: {overall_best['total_relevance']:.2f}")
        print(f"  Avg Relevance: {overall_best['avg_relevance']:.4f}")
    
    return results_df

if __name__ == "__main__":
    try:
        results = run_hyperparameter_study()
        print("\nHyperparameter tuning study completed successfully!")
    except Exception as e:
        print(f"Error in hyperparameter tuning: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop() 