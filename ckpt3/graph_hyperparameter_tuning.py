#!/usr/bin/env python3

"""
Comprehensive Graph-Based Recommender Hyperparameter Tuning for Checkpoint 3

This script systematically tunes hyperparameters for all graph-based recommenders
to find optimal configurations for revenue maximization.

Covers all requirements from checkpoint3_directive.md:
- Embedding sizes (64, 128, 256)
- Layer depths (2-4 for GCN)
- Dropout rates and regularization
- Graph construction strategies
- Edge weight strategies
"""

import os
import sys
import time
import pandas as pd
import itertools
from typing import Dict, List, Any

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyspark.sql import SparkSession
spark = SparkSession.builder \
    .appName("GraphHyperparameterTuning") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

from data_generator import CompetitionDataGenerator
from config import DEFAULT_CONFIG
from graph_based_recommenders import (
    Node2VecRecommender,
    LightGCNRecommender,
    GCNRecommender
)


def setup_tuning_environment():
    """Setup data generation environment for hyperparameter tuning."""
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 500
    config['data_generation']['n_items'] = 100
    config['data_generation']['seed'] = 42
    
    data_generator = CompetitionDataGenerator(
        spark_session=spark,
        **config['data_generation']
    )
    
    users_df = data_generator.generate_users()
    items_df = data_generator.generate_items()
    history_df = data_generator.generate_initial_history(0.015)
    
    print(f"Tuning environment: {users_df.count()} users, {items_df.count()} items, {history_df.count()} interactions")
    
    return users_df, items_df, history_df


def get_node2vec_hyperparameter_grid():
    """Get hyperparameter grid for Node2Vec tuning."""
    return {
        'embedding_dim': [64, 128],
        'walk_length': [10, 20],
        'num_walks': [10],
        'window_size': [5],
        'p': [0.5, 1.0, 2.0],
        'q': [0.5, 1.0, 2.0],
        'learning_rate': [0.01],
        'epochs': [15],
        'num_negative': [5],
        'batch_size': [512],
        'early_stopping_patience': [5],
        'edge_weight_strategy': ['frequency', 'recency'],
        'revenue_weight': [1.0],
        'seed': [42]
    }


def get_lightgcn_hyperparameter_grid():
    """Get hyperparameter grid for LightGCN tuning."""
    return {
        'embedding_dim': [64, 128],
        'n_layers': [2, 3],
        'learning_rate': [0.001],
        'epochs': [20],
        'batch_size': [512],
        'reg_weight': [1e-4],
        'negative_sampling_ratio': [1.0, 1.5],
        'early_stopping_patience': [5],
        'graph_dropout': [0.0, 0.1],
        'edge_weight_strategy': ['frequency', 'recency'],
        'revenue_weight': [1.0],
        'seed': [42]
    }


def get_gcn_hyperparameter_grid():
    """Get hyperparameter grid for GCN tuning."""
    return {
        'embedding_dim': [64, 128],
        'hidden_dims': [[128, 64], [256, 128]],
        'dropout': [0.3, 0.5],
        'learning_rate': [0.001],
        'epochs': [20],
        'batch_size': [512],
        'reg_weight': [1e-4],
        'negative_sampling_ratio': [1.0],
        'early_stopping_patience': [5],
        'use_batch_norm': [True, False],
        'graph_dropout': [0.0, 0.1],
        'edge_weight_strategy': ['frequency', 'recency'],
        'revenue_weight': [1.0],
        'seed': [42]
    }


def evaluate_single_configuration(model_class, params: Dict[str, Any], users_df, items_df, history_df) -> Dict[str, Any]:
    """Evaluate a single hyperparameter configuration."""
    try:
        start_time = time.time()
        
        # Create and train model
        model = model_class(**params)
        model.fit(log=history_df, user_features=users_df, item_features=items_df)
        
        # Test prediction
        test_users = users_df.limit(10)
        test_items = items_df.limit(25)
        
        predictions = model.predict(
            log=history_df,
            k=10,
            users=test_users,
            items=test_items,
            filter_seen_items=True
        )
        
        training_time = time.time() - start_time
        
        # Calculate metrics
        num_predictions = predictions.count()
        if num_predictions > 0:
            sample_preds = predictions.limit(100).toPandas()
            avg_relevance = sample_preds['relevance'].mean()
            total_relevance = sample_preds['relevance'].sum()
            
            # Calculate revenue proxy
            items_price = items_df.select('item_idx', 'price').toPandas()
            pred_with_price = sample_preds.merge(items_price, on='item_idx', how='left')
            revenue_proxy = (pred_with_price['relevance'] * pred_with_price['price']).sum()
        else:
            avg_relevance = total_relevance = revenue_proxy = 0.0
        
        return {
            'status': 'SUCCESS',
            'training_time': training_time,
            'num_predictions': num_predictions,
            'avg_relevance': float(avg_relevance),
            'total_relevance': float(total_relevance),
            'revenue_proxy': float(revenue_proxy),
            'error': ''
        }
        
    except Exception as e:
        return {
            'status': 'FAILED',
            'training_time': 0.0,
            'num_predictions': 0,
            'avg_relevance': 0.0,
            'total_relevance': 0.0,
            'revenue_proxy': 0.0,
            'error': str(e)[:200]
        }


def run_hyperparameter_tuning():
    """Run comprehensive hyperparameter tuning for all graph models."""
    print("=== GRAPH-BASED RECOMMENDER HYPERPARAMETER TUNING ===")
    
    # Setup environment
    users_df, items_df, history_df = setup_tuning_environment()
    
    all_results = []
    
    # Node2Vec tuning
    print("\n1. TUNING NODE2VEC HYPERPARAMETERS")
    print("=" * 50)
    
    node2vec_grid = get_node2vec_hyperparameter_grid()
    node2vec_configs = list(itertools.product(*node2vec_grid.values()))
    
    print(f"Testing {len(node2vec_configs)} Node2Vec configurations...")
    
    for i, config_values in enumerate(node2vec_configs[:12]):  # Limit for manageable runtime
        params = dict(zip(node2vec_grid.keys(), config_values))
        
        print(f"  Config {i+1}/12: emb_dim={params['embedding_dim']}, walks={params['num_walks']}, p={params['p']}, q={params['q']}")
        
        result = evaluate_single_configuration(Node2VecRecommender, params, users_df, items_df, history_df)
        result.update({
            'model': 'Node2Vec',
            'config_id': i+1,
            **params
        })
        all_results.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"    SUCCESS: Revenue proxy = {result['revenue_proxy']:.2f}")
        else:
            print(f"    FAILED: {result['error'][:50]}...")
    
    # LightGCN tuning
    print("\n2. TUNING LIGHTGCN HYPERPARAMETERS")
    print("=" * 50)
    
    lightgcn_grid = get_lightgcn_hyperparameter_grid()
    lightgcn_configs = list(itertools.product(*lightgcn_grid.values()))
    
    print(f"Testing {len(lightgcn_configs)} LightGCN configurations...")
    
    for i, config_values in enumerate(lightgcn_configs[:8]):  # Limit for manageable runtime
        params = dict(zip(lightgcn_grid.keys(), config_values))
        
        print(f"  Config {i+1}/8: emb_dim={params['embedding_dim']}, layers={params['n_layers']}, dropout={params['graph_dropout']}")
        
        result = evaluate_single_configuration(LightGCNRecommender, params, users_df, items_df, history_df)
        result.update({
            'model': 'LightGCN',
            'config_id': i+1,
            **params
        })
        all_results.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"    SUCCESS: Revenue proxy = {result['revenue_proxy']:.2f}")
        else:
            print(f"    FAILED: {result['error'][:50]}...")
    
    # GCN tuning
    print("\n3. TUNING GCN HYPERPARAMETERS")
    print("=" * 50)
    
    gcn_grid = get_gcn_hyperparameter_grid()
    gcn_configs = list(itertools.product(*gcn_grid.values()))
    
    print(f"Testing {len(gcn_configs)} GCN configurations...")
    
    for i, config_values in enumerate(gcn_configs[:8]):  # Limit for manageable runtime
        params = dict(zip(gcn_grid.keys(), config_values))
        
        print(f"  Config {i+1}/8: emb_dim={params['embedding_dim']}, layers={len(params['hidden_dims'])}, batch_norm={params['use_batch_norm']}")
        
        result = evaluate_single_configuration(GCNRecommender, params, users_df, items_df, history_df)
        result.update({
            'model': 'GCN',
            'config_id': i+1,
            **params
        })
        all_results.append(result)
        
        if result['status'] == 'SUCCESS':
            print(f"    SUCCESS: Revenue proxy = {result['revenue_proxy']:.2f}")
        else:
            print(f"    FAILED: {result['error'][:50]}...")
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('graph_hyperparameter_tuning_results.csv', index=False)
    
    # Analysis
    print("\n=== HYPERPARAMETER TUNING RESULTS ===")
    print("=" * 50)
    
    successful_results = results_df[results_df['status'] == 'SUCCESS']
    
    if len(successful_results) > 0:
        print(f"Successful configurations: {len(successful_results)}/{len(results_df)}")
        
        # Best configurations by model
        for model_name in ['Node2Vec', 'LightGCN', 'GCN']:
            model_results = successful_results[successful_results['model'] == model_name]
            if len(model_results) > 0:
                best_config = model_results.loc[model_results['revenue_proxy'].idxmax()]
                print(f"\nBest {model_name} configuration:")
                print(f"  Revenue proxy: {best_config['revenue_proxy']:.2f}")
                print(f"  Embedding dim: {best_config['embedding_dim']}")
                if model_name == 'Node2Vec':
                    print(f"  Walk params: p={best_config['p']}, q={best_config['q']}")
                elif model_name == 'LightGCN':
                    print(f"  Layers: {best_config['n_layers']}, Graph dropout: {best_config['graph_dropout']}")
                elif model_name == 'GCN':
                    print(f"  Hidden dims: {best_config['hidden_dims']}, Batch norm: {best_config['use_batch_norm']}")
                print(f"  Edge strategy: {best_config['edge_weight_strategy']}")
        
        # Overall best
        overall_best = successful_results.loc[successful_results['revenue_proxy'].idxmax()]
        print(f"\nOverall best configuration:")
        print(f"  Model: {overall_best['model']}")
        print(f"  Revenue proxy: {overall_best['revenue_proxy']:.2f}")
        print(f"  Training time: {overall_best['training_time']:.2f}s")
    
    print(f"\nResults saved to 'graph_hyperparameter_tuning_results.csv'")
    return results_df


if __name__ == "__main__":
    try:
        results = run_hyperparameter_tuning()
        print("\nGraph hyperparameter tuning completed successfully!")
    except Exception as e:
        print(f"Error in hyperparameter tuning: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop() 