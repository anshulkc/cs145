#!/usr/bin/env python3

"""
Graph-Based Recommender Evaluation Script for Checkpoint 3

This script evaluates the performance of graph-based recommendation algorithms
by testing them on synthetic data and comparing their revenue performance.
"""

import time
import pandas as pd
from typing import Dict, List, Any

# Spark setup
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('GraphRecommenderEvaluation').master('local[*]').config('spark.driver.memory', '4g').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

# Import evaluation framework
import sys
sys.path.append('..')
from data_generator import CompetitionDataGenerator
from evaluation import RecommenderEvaluator
from config import DEFAULT_CONFIG

# Import graph-based recommenders
from graph_based_recommenders import (
    Node2VecRecommender,
    LightGCNRecommender, 
    GCNRecommender,
    GRAPH_BASED_CONFIGS
)


def run_graph_recommender_evaluation():
    """Run comprehensive evaluation of graph-based recommenders."""
    
    print("=" * 80)
    print("GRAPH-BASED RECOMMENDER EVALUATION")
    print("=" * 80)
    
    # Setup data generation
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 1000
    config['data_generation']['n_items'] = 200
    config['data_generation']['seed'] = 42
    
    data_generator = CompetitionDataGenerator(spark_session=spark, **config['data_generation'])
    evaluator = RecommenderEvaluator(spark_session=spark)
    
    print(f"Generating synthetic data: {config['data_generation']['n_users']} users, {config['data_generation']['n_items']} items")
    
    # Generate data
    users_df = data_generator.generate_users()
    items_df = data_generator.generate_items()
    history_df = data_generator.generate_initial_history(config['data_generation']['initial_history_density'])
    
    print(f"Initial history: {history_df.count()} interactions")
    print()
    
    # Select graph recommenders for evaluation
    selected_configs = [
        "Node2Vec_Basic",
        "Node2Vec_Advanced", 
        "LightGCN_Small",
        "LightGCN_Medium",
        "GCN_Basic",
        "GCN_Deep"
    ]
    
    results = []
    
    for config_name in selected_configs:
        print(f"Evaluating {config_name}")
        print("-" * 60)
        
        # Find configuration
        config_def = None
        for config in GRAPH_BASED_CONFIGS:
            if config["name"] == config_name:
                config_def = config
                break
        
        if config_def is None:
            print(f"Configuration {config_name} not found!")
            continue
        
        try:
            start_time = time.time()
            
            # Create recommender instance
            if config_def["class"] == "Node2VecRecommender":
                recommender = Node2VecRecommender(**config_def["parameters"])
            elif config_def["class"] == "LightGCNRecommender":
                recommender = LightGCNRecommender(**config_def["parameters"])
            elif config_def["class"] == "GCNRecommender":
                recommender = GCNRecommender(**config_def["parameters"])
            else:
                print(f"Unknown recommender class: {config_def['class']}")
                continue
            
            # Evaluate using incremental learning approach
            metrics = evaluator.evaluate_recommender(
                recommender=recommender,
                users_df=users_df,
                items_df=items_df,
                initial_history_df=history_df,
                n_iterations=3,  # Reduced for faster evaluation
                k=10
            )
            
            training_time = time.time() - start_time
            
            # Prepare result
            result = {
                'name': f"{config_name} (Ours)",
                'training_time': training_time,
                'status': 'SUCCESS',
                'error': ''
            }
            result.update(metrics)
            results.append(result)
            
            print(f"✓ {config_name} completed successfully")
            print(f"  Training time: {training_time:.2f}s")
            print(f"  Test total revenue: {metrics['test_total_revenue']:.2f}")
            print()
            
        except Exception as e:
            error_msg = str(e)
            print(f"✗ {config_name} failed: {error_msg}")
            
            result = {
                'name': f"{config_name} (Ours)",
                'training_time': 0,
                'status': 'FAILED',
                'error': error_msg,
                # Fill with zeros for failed runs
                'train_total_revenue': 0,
                'test_total_revenue': 0,
                'train_avg_revenue': 0,
                'test_avg_revenue': 0,
                'performance_change': 0,
                'train_precision_at_k': 0,
                'train_ndcg_at_k': 0,
                'train_mrr': 0,
                'train_hit_rate': 0,
                'train_discounted_revenue': 0,
                'train_revenue': 0,
                'test_precision_at_k': 0,
                'test_ndcg_at_k': 0,
                'test_mrr': 0,
                'test_hit_rate': 0,
                'test_discounted_revenue': 0,
                'test_revenue': 0
            }
            results.append(result)
            print()
    
    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('graph_recommender_results.csv', index=False)
        print(f"Results saved to graph_recommender_results.csv")
        
        # Print summary
        print("\n" + "=" * 80)
        print("EVALUATION SUMMARY")
        print("=" * 80)
        
        successful_results = [r for r in results if r['status'] == 'SUCCESS']
        failed_results = [r for r in results if r['status'] == 'FAILED']
        
        print(f"Successful evaluations: {len(successful_results)}")
        print(f"Failed evaluations: {len(failed_results)}")
        
        if successful_results:
            print("\nTop performers by test revenue:")
            sorted_results = sorted(successful_results, key=lambda x: x['test_total_revenue'], reverse=True)
            for i, result in enumerate(sorted_results[:3]):
                print(f"  {i+1}. {result['name']}: {result['test_total_revenue']:.2f}")
        
        if failed_results:
            print("\nFailed configurations:")
            for result in failed_results:
                print(f"  - {result['name']}: {result['error']}")
    
    print("\nGraph-based recommender evaluation completed!")


if __name__ == "__main__":
    run_graph_recommender_evaluation()
    spark.stop() 