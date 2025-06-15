#!/usr/bin/env python3

"""
Final Demonstration of Graph-Based Recommenders for Checkpoint 3
This script demonstrates that all three graph-based recommenders work correctly.
"""

import time
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Spark setup
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('GraphDemo').master('local[*]').config('spark.driver.memory', '4g').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

# Import required modules
from data_generator import CompetitionDataGenerator
from config import DEFAULT_CONFIG
from graph_based_recommenders import Node2VecRecommender, LightGCNRecommender, GCNRecommender


def run_final_demo():
    """Demonstrate all three graph-based recommenders working correctly."""
    
    print("=" * 60)
    print("CHECKPOINT 3: GRAPH-BASED RECOMMENDERS")
    print("FINAL DEMONSTRATION")
    print("=" * 60)
    
    # Setup test data
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 150
    config['data_generation']['n_items'] = 75
    config['data_generation']['seed'] = 42
    
    print("Generating test data...")
    data_generator = CompetitionDataGenerator(spark_session=spark, **config['data_generation'])
    
    users_df = data_generator.generate_users()
    items_df = data_generator.generate_items()
    history_df = data_generator.generate_initial_history(0.08)
    
    n_users = users_df.count()
    n_items = items_df.count()
    n_interactions = history_df.count()
    
    print(f"Generated: {n_users} users, {n_items} items, {n_interactions} interactions")
    print(f"Interaction density: {n_interactions / (n_users * n_items) * 100:.2f}%")
    print()
    
    # Test configurations
    recommenders = [
        {
            "name": "Node2Vec",
            "class": Node2VecRecommender,
            "params": {
                "embedding_dim": 64,
                "walk_length": 8,
                "num_walks": 10,
                "window_size": 4,
                "p": 1.0,
                "q": 1.0,
                "learning_rate": 0.01,
                "epochs": 20,
                "num_negative": 5,
                "batch_size": 256,
                "edge_weight_strategy": "frequency",
                "revenue_weight": 1.2,
                "seed": 42
            },
            "description": "Random walk-based embeddings for link prediction"
        },
        {
            "name": "LightGCN",
            "class": LightGCNRecommender,
            "params": {
                "embedding_dim": 64,
                "n_layers": 3,
                "learning_rate": 0.001,
                "epochs": 30,
                "batch_size": 512,
                "reg_weight": 1e-4,
                "negative_sampling_ratio": 1.5,
                "edge_weight_strategy": "purchase_amount",
                "revenue_weight": 1.1,
                "seed": 42
            },
            "description": "Simplified graph convolution for collaborative filtering"
        },
        {
            "name": "GCN",
            "class": GCNRecommender,
            "params": {
                "embedding_dim": 64,
                "hidden_dims": [128, 64],
                "dropout": 0.4,
                "learning_rate": 0.001,
                "epochs": 25,
                "batch_size": 512,
                "reg_weight": 1e-4,
                "negative_sampling_ratio": 2.0,
                "edge_weight_strategy": "recency",
                "revenue_weight": 1.3,
                "seed": 42
            },
            "description": "Graph Convolutional Networks for user-item prediction"
        }
    ]
    
    results = []
    test_users = users_df.limit(25)
    
    print("TRAINING AND TESTING ALL GRAPH RECOMMENDERS")
    print("=" * 50)
    
    for i, recommender_config in enumerate(recommenders, 1):
        name = recommender_config["name"]
        recommender_class = recommender_config["class"]
        params = recommender_config["params"]
        description = recommender_config["description"]
        
        print(f"\n{i}. {name}")
        print(f"   {description}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            
            print("   Initializing recommender...")
            recommender = recommender_class(**params)
            
            print("   Training graph model...")
            recommender.fit(history_df, users_df, items_df)
            
            print("   Generating recommendations...")
            recommendations = recommender.predict(
                log=history_df,
                k=10,
                users=test_users,
                items=items_df,
                user_features=users_df,
                item_features=items_df,
                filter_seen_items=True
            )
            
            training_time = time.time() - start_time
            rec_count = recommendations.count()
            
            # Calculate performance metrics
            if rec_count > 0:
                sample_recs = recommendations.limit(200).toPandas()
                avg_relevance = sample_recs['relevance'].mean()
                max_relevance = sample_recs['relevance'].max()
                min_relevance = sample_recs['relevance'].min()
                std_relevance = sample_recs['relevance'].std()
                revenue_score = avg_relevance * 50
            else:
                avg_relevance = max_relevance = min_relevance = std_relevance = revenue_score = 0.0
            
            status = "SUCCESS"
            error_msg = ""
            
            result = {
                'name': name,
                'status': status,
                'training_time': training_time,
                'recommendations_count': rec_count,
                'avg_relevance': avg_relevance,
                'max_relevance': max_relevance,
                'min_relevance': min_relevance,
                'std_relevance': std_relevance,
                'revenue_score': revenue_score,
                'error': error_msg
            }
            
            print(f"   {status}")
            print(f"   Training time: {training_time:.2f}s")
            print(f"   Recommendations: {rec_count}")
            print(f"   Revenue score: {revenue_score:.2f}")
            print(f"   Avg relevance: {avg_relevance:.4f} ± {std_relevance:.4f}")
            
        except Exception as e:
            error_msg = str(e)[:100] + "..." if len(str(e)) > 100 else str(e)
            status = "FAILED"
            
            result = {
                'name': name,
                'status': status,
                'training_time': 0,
                'recommendations_count': 0,
                'avg_relevance': 0.0,
                'max_relevance': 0.0,
                'min_relevance': 0.0,
                'std_relevance': 0.0,
                'revenue_score': 0.0,
                'error': error_msg
            }
            
            print(f"   {status}: {error_msg}")
        
        results.append(result)
    
    # Summary
    print("\n" + "=" * 40)
    print("EVALUATION SUMMARY")
    print("=" * 40)
    
    successful = [r for r in results if 'SUCCESS' in r['status']]
    failed = [r for r in results if 'FAILED' in r['status']]
    
    print(f"Successful: {len(successful)}/3")
    print(f"Failed: {len(failed)}/3")
    
    if successful:
        print("\nPerformance ranking:")
        sorted_results = sorted(successful, key=lambda x: x['revenue_score'], reverse=True)
        for i, result in enumerate(sorted_results, 1):
            print(f"   {i}. {result['name']}: {result['revenue_score']:.2f} revenue score")
    
    if failed:
        print("\nFailed configurations:")
        for result in failed:
            print(f"   - {result['name']}: {result['error']}")
    
    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv('graph_recommender_demo_results.csv', index=False)
        print(f"\nResults saved to: graph_recommender_demo_results.csv")
    
    success_rate = len(successful) / len(results)
    print(f"\nSuccess rate: {success_rate:.1%}")
    
    if success_rate == 1.0:
        print("\nAll three graph-based recommenders working successfully!")
        return True
    else:
        print(f"\nPartial success: {len(successful)}/3 recommenders working")
        return False


if __name__ == "__main__":
    success = run_final_demo()
    print(f"\nDemo completed!")
    spark.stop()
    exit(0 if success else 1) 