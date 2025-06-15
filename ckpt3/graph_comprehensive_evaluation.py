#!/usr/bin/env python3

"""
Fast Comprehensive Graph-Based Recommender Evaluation for Checkpoint 3

This script provides comprehensive evaluation of graph-based recommenders including:
- Graph-specific metrics (Link prediction AUC, embedding quality)
- Ablation studies (impact of graph construction choices)
- Comparison with baseline approaches
- All metrics required by checkpoint3_directive.md

Optimized for speed while maintaining comprehensive coverage.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from sklearn.metrics import roc_auc_score, ndcg_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

spark = SparkSession.builder \
    .appName("GraphComprehensiveEvaluation") \
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

# Import baseline recommenders for comparison
try:
    from sample_recommenders import (
        RandomRecommender,
        PopularityRecommender,
        ContentBasedRecommender
    )
    BASELINES_AVAILABLE = True
except ImportError:
    BASELINES_AVAILABLE = False
    print("Warning: Baseline recommenders not available")


def setup_evaluation_environment():
    """Setup fast evaluation environment."""
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 600
    config['data_generation']['n_items'] = 120
    config['data_generation']['seed'] = 42
    
    data_generator = CompetitionDataGenerator(
        spark_session=spark,
        **config['data_generation']
    )
    
    users_df = data_generator.generate_users()
    items_df = data_generator.generate_items()
    history_df = data_generator.generate_initial_history(0.02)
    
    print(f"Evaluation environment: {users_df.count()} users, {items_df.count()} items, {history_df.count()} interactions")
    
    return users_df, items_df, history_df


def get_optimized_graph_recommenders():
    """Get graph recommenders with optimized hyperparameters (fast configs)."""
    return {
        'Node2Vec_Optimized': Node2VecRecommender(
            embedding_dim=64,
            walk_length=15,
            num_walks=10,
            window_size=5,
            p=0.5,
            q=2.0,
            learning_rate=0.01,
            epochs=25,
            num_negative=5,
            batch_size=512,
            early_stopping_patience=5,
            edge_weight_strategy='recency',
            revenue_weight=1.0,
            seed=42
        ),
        'LightGCN_Optimized': LightGCNRecommender(
            embedding_dim=64,
            n_layers=2,
            learning_rate=0.001,
            epochs=30,
            batch_size=512,
            reg_weight=1e-4,
            negative_sampling_ratio=1.0,
            early_stopping_patience=5,
            graph_dropout=0.1,
            edge_weight_strategy='frequency',
            revenue_weight=1.0,
            seed=42
        ),
        'GCN_Optimized': GCNRecommender(
            embedding_dim=64,
            hidden_dims=[128, 64],
            dropout=0.3,
            learning_rate=0.001,
            epochs=30,
            batch_size=512,
            reg_weight=1e-4,
            negative_sampling_ratio=1.0,
            early_stopping_patience=5,
            use_batch_norm=True,
            graph_dropout=0.1,
            edge_weight_strategy='frequency',
            revenue_weight=1.0,
            seed=42
        )
    }


def get_baseline_recommenders():
    """Get baseline recommenders for comparison."""
    if not BASELINES_AVAILABLE:
        return {}
    
    return {
        'Random_Baseline': RandomRecommender(seed=42),
        'Popularity_Baseline': PopularityRecommender(alpha=1.0, seed=42),
        'ContentBased_Baseline': ContentBasedRecommender(similarity_threshold=0.0, seed=42)
    }


def calculate_link_prediction_auc(model, users_df, items_df, history_df) -> float:
    """Calculate link prediction AUC for graph models."""
    try:
        # Get positive interactions (sample for speed)
        positive_interactions = history_df.select('user_idx', 'item_idx').limit(500).toPandas()
        positive_pairs = [(row['user_idx'], row['item_idx']) for _, row in positive_interactions.iterrows()]
        
        # Sample negative interactions
        all_users = users_df.select('user_idx').limit(100).toPandas()['user_idx'].tolist()
        all_items = items_df.select('item_idx').limit(50).toPandas()['item_idx'].tolist()
        
        positive_set = set(positive_pairs)
        negative_pairs = []
        
        # Sample same number of negatives as positives (max 200)
        target_negatives = min(len(positive_pairs), 200)
        while len(negative_pairs) < target_negatives:
            user_idx = np.random.choice(all_users)
            item_idx = np.random.choice(all_items)
            if (user_idx, item_idx) not in positive_set:
                negative_pairs.append((user_idx, item_idx))
        
        # Get predictions for all pairs
        all_pairs = positive_pairs[:len(negative_pairs)] + negative_pairs
        labels = [1] * len(negative_pairs) + [0] * len(negative_pairs)
        
        if hasattr(model, '_predict_link_probabilities'):
            predictions = model._predict_link_probabilities(all_pairs)
            auc = roc_auc_score(labels, predictions)
            return float(auc)
        else:
            return 0.0
            
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        return 0.0


def calculate_embedding_quality_metrics(model) -> Dict[str, float]:
    """Calculate embedding quality metrics for graph models."""
    try:
        if hasattr(model, 'node_embeddings') and model.node_embeddings is not None:
            embeddings = model.node_embeddings
            
            # Calculate embedding statistics
            mean_norm = np.mean(np.linalg.norm(embeddings, axis=1))
            std_norm = np.std(np.linalg.norm(embeddings, axis=1))
            
            # Calculate embedding diversity (sample for speed)
            sample_size = min(50, len(embeddings))
            sample_indices = np.random.choice(len(embeddings), sample_size, replace=False)
            sample_embeddings = embeddings[sample_indices]
            
            pairwise_distances = []
            for i in range(min(sample_size, 20)):
                for j in range(i+1, min(sample_size, 20)):
                    dist = np.linalg.norm(sample_embeddings[i] - sample_embeddings[j])
                    pairwise_distances.append(dist)
            
            avg_distance = np.mean(pairwise_distances) if pairwise_distances else 0.0
            
            return {
                'embedding_mean_norm': float(mean_norm),
                'embedding_std_norm': float(std_norm),
                'embedding_avg_distance': float(avg_distance)
            }
        else:
            return {
                'embedding_mean_norm': 0.0,
                'embedding_std_norm': 0.0,
                'embedding_avg_distance': 0.0
            }
    except Exception as e:
        print(f"Error calculating embedding metrics: {e}")
        return {
            'embedding_mean_norm': 0.0,
            'embedding_std_norm': 0.0,
            'embedding_avg_distance': 0.0
        }


def calculate_comprehensive_metrics(predictions, items_df) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics."""
    try:
        if predictions.count() == 0:
            return {
                'precision_at_5': 0.0,
                'precision_at_10': 0.0,
                'ndcg_at_5': 0.0,
                'ndcg_at_10': 0.0,
                'mrr': 0.0,
                'hit_rate': 0.0,
                'total_revenue': 0.0,
                'discounted_revenue': 0.0
            }
        
        # Convert to pandas for easier processing (sample for speed)
        pred_df = predictions.limit(500).toPandas()
        items_price = items_df.select('item_idx', 'price').toPandas()
        pred_with_price = pred_df.merge(items_price, on='item_idx', how='left')
        
        # Group by user for ranking metrics
        user_groups = pred_with_price.groupby('user_idx')
        
        precision_5_scores = []
        precision_10_scores = []
        ndcg_5_scores = []
        ndcg_10_scores = []
        mrr_scores = []
        hit_rates = []
        
        for user_idx, group in user_groups:
            # Sort by relevance (descending)
            sorted_group = group.sort_values('relevance', ascending=False)
            
            # Binary relevance (threshold at median)
            relevance_threshold = sorted_group['relevance'].median()
            binary_relevance = (sorted_group['relevance'] >= relevance_threshold).astype(int)
            
            # Precision@K
            if len(binary_relevance) >= 5:
                precision_5_scores.append(binary_relevance.iloc[:5].mean())
            if len(binary_relevance) >= 10:
                precision_10_scores.append(binary_relevance.iloc[:10].mean())
            
            # NDCG@K
            if len(sorted_group) >= 5:
                try:
                    ndcg_5 = ndcg_score([binary_relevance.tolist()], [sorted_group['relevance'].tolist()], k=5)
                    ndcg_5_scores.append(ndcg_5)
                except:
                    ndcg_5_scores.append(0.0)
            if len(sorted_group) >= 10:
                try:
                    ndcg_10 = ndcg_score([binary_relevance.tolist()], [sorted_group['relevance'].tolist()], k=10)
                    ndcg_10_scores.append(ndcg_10)
                except:
                    ndcg_10_scores.append(0.0)
            
            # MRR
            first_relevant = binary_relevance.idxmax() if binary_relevance.sum() > 0 else -1
            if first_relevant != -1:
                rank = list(binary_relevance.index).index(first_relevant) + 1
                mrr_scores.append(1.0 / rank)
            else:
                mrr_scores.append(0.0)
            
            # Hit Rate
            hit_rates.append(1.0 if binary_relevance.sum() > 0 else 0.0)
        
        # Revenue calculations
        total_revenue = (pred_with_price['relevance'] * pred_with_price['price']).sum()
        
        # Discounted revenue (apply position-based discount)
        pred_with_price['position'] = pred_with_price.groupby('user_idx')['relevance'].rank(ascending=False)
        pred_with_price['discount'] = 1.0 / np.log2(pred_with_price['position'] + 1)
        discounted_revenue = (pred_with_price['relevance'] * pred_with_price['price'] * pred_with_price['discount']).sum()
        
        return {
            'precision_at_5': float(np.mean(precision_5_scores)) if precision_5_scores else 0.0,
            'precision_at_10': float(np.mean(precision_10_scores)) if precision_10_scores else 0.0,
            'ndcg_at_5': float(np.mean(ndcg_5_scores)) if ndcg_5_scores else 0.0,
            'ndcg_at_10': float(np.mean(ndcg_10_scores)) if ndcg_10_scores else 0.0,
            'mrr': float(np.mean(mrr_scores)) if mrr_scores else 0.0,
            'hit_rate': float(np.mean(hit_rates)) if hit_rates else 0.0,
            'total_revenue': float(total_revenue),
            'discounted_revenue': float(discounted_revenue)
        }
        
    except Exception as e:
        print(f"Error calculating comprehensive metrics: {e}")
        return {
            'precision_at_5': 0.0,
            'precision_at_10': 0.0,
            'ndcg_at_5': 0.0,
            'ndcg_at_10': 0.0,
            'mrr': 0.0,
            'hit_rate': 0.0,
            'total_revenue': 0.0,
            'discounted_revenue': 0.0
        }


def evaluate_single_recommender(name: str, recommender, users_df, items_df, history_df) -> Dict[str, Any]:
    """Evaluate a single recommender with comprehensive metrics."""
    print(f"Evaluating {name}...")
    
    try:
        start_time = time.time()
        
        # Fit the model
        recommender.fit(log=history_df, user_features=users_df, item_features=items_df)
        
        # Test prediction (smaller sample for speed)
        test_users = users_df.limit(15)
        test_items = items_df.limit(30)
        
        predictions = recommender.predict(
            log=history_df,
            k=10,
            users=test_users,
            items=test_items,
            filter_seen_items=True
        )
        
        training_time = time.time() - start_time
        
        # Calculate basic metrics
        num_predictions = predictions.count()
        avg_relevance = predictions.agg({'relevance': 'avg'}).collect()[0][0] or 0.0
        total_relevance = predictions.agg({'relevance': 'sum'}).collect()[0][0] or 0.0
        
        # Calculate revenue proxy
        items_price = items_df.select('item_idx', sf.col('price').alias('item_price'))
        predictions_with_price = predictions.join(items_price, 'item_idx', 'left')
        revenue_proxy = predictions_with_price.selectExpr('relevance * item_price as revenue').agg({'revenue': 'sum'}).collect()[0][0] or 0.0
        
        # Calculate comprehensive metrics
        comprehensive_metrics = calculate_comprehensive_metrics(predictions, items_df)
        
        # Calculate graph-specific metrics
        link_prediction_auc = calculate_link_prediction_auc(recommender, users_df, items_df, history_df)
        embedding_metrics = calculate_embedding_quality_metrics(recommender)
        
        metrics = {
            'name': name,
            'status': 'SUCCESS',
            'training_time': training_time,
            'num_predictions': num_predictions,
            'avg_relevance': float(avg_relevance),
            'total_relevance': float(total_relevance),
            'revenue_proxy': float(revenue_proxy),
            'link_prediction_auc': link_prediction_auc,
            **embedding_metrics,
            **comprehensive_metrics
        }
        
        print(f"  {name} - Revenue Proxy: {metrics['revenue_proxy']:.2f}, AUC: {metrics['link_prediction_auc']:.3f}")
        
        return metrics
        
    except Exception as e:
        print(f"  ERROR in {name}: {str(e)[:100]}...")
        return {
            'name': name,
            'status': 'FAILED',
            'error': str(e)[:200],
            'training_time': 0.0,
            'num_predictions': 0,
            'avg_relevance': 0.0,
            'total_relevance': 0.0,
            'revenue_proxy': 0.0,
            'link_prediction_auc': 0.0,
            'embedding_mean_norm': 0.0,
            'embedding_std_norm': 0.0,
            'embedding_avg_distance': 0.0,
            'precision_at_5': 0.0,
            'precision_at_10': 0.0,
            'ndcg_at_5': 0.0,
            'ndcg_at_10': 0.0,
            'mrr': 0.0,
            'hit_rate': 0.0,
            'total_revenue': 0.0,
            'discounted_revenue': 0.0
        }


def run_ablation_study(users_df, items_df, history_df) -> pd.DataFrame:
    """Run fast ablation study on graph construction choices."""
    print("\n=== ABLATION STUDY: GRAPH CONSTRUCTION CHOICES ===")
    
    ablation_results = []
    
    # Test different edge weight strategies
    edge_strategies = ['frequency', 'recency']
    
    for strategy in edge_strategies:
        print(f"Testing edge weight strategy: {strategy}")
        
        model = LightGCNRecommender(
            embedding_dim=64,
            n_layers=2,
            learning_rate=0.001,
            epochs=15,
            batch_size=512,
            edge_weight_strategy=strategy,
            seed=42
        )
        
        result = evaluate_single_recommender(f"LightGCN_{strategy}", model, users_df, items_df, history_df)
        result['ablation_type'] = 'edge_weight_strategy'
        result['ablation_value'] = strategy
        ablation_results.append(result)
    
    # Test different graph dropout rates
    dropout_rates = [0.0, 0.1]
    
    for dropout in dropout_rates:
        print(f"Testing graph dropout: {dropout}")
        
        model = GCNRecommender(
            embedding_dim=64,
            hidden_dims=[128, 64],
            dropout=0.3,
            learning_rate=0.001,
            epochs=15,
            batch_size=512,
            graph_dropout=dropout,
            seed=42
        )
        
        result = evaluate_single_recommender(f"GCN_dropout_{dropout}", model, users_df, items_df, history_df)
        result['ablation_type'] = 'graph_dropout'
        result['ablation_value'] = dropout
        ablation_results.append(result)
    
    return pd.DataFrame(ablation_results)


def create_performance_visualizations(results_df: pd.DataFrame):
    """Create comprehensive performance visualizations."""
    print("Creating performance visualizations...")
    
    successful_results = results_df[results_df['status'] == 'SUCCESS'].copy()
    
    if len(successful_results) == 0:
        print("No successful results to visualize")
        return
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Graph-Based Recommenders: Comprehensive Performance Analysis', fontsize=16, fontweight='bold')
    
    # 1. Revenue Performance
    ax1 = axes[0, 0]
    sns.barplot(data=successful_results, x='name', y='revenue_proxy', ax=ax1)
    ax1.set_title('Revenue Proxy Performance')
    ax1.set_xlabel('Model')
    ax1.set_ylabel('Revenue Proxy')
    ax1.tick_params(axis='x', rotation=45)
    
    # 2. Link Prediction AUC
    ax2 = axes[0, 1]
    sns.barplot(data=successful_results, x='name', y='link_prediction_auc', ax=ax2)
    ax2.set_title('Link Prediction AUC')
    ax2.set_xlabel('Model')
    ax2.set_ylabel('AUC')
    ax2.tick_params(axis='x', rotation=45)
    
    # 3. NDCG@5 vs Precision@5
    ax3 = axes[0, 2]
    ax3.scatter(successful_results['precision_at_5'], successful_results['ndcg_at_5'], s=100, alpha=0.7)
    for i, row in successful_results.iterrows():
        ax3.annotate(row['name'], (row['precision_at_5'], row['ndcg_at_5']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax3.set_title('NDCG@5 vs Precision@5')
    ax3.set_xlabel('Precision@5')
    ax3.set_ylabel('NDCG@5')
    ax3.grid(True, alpha=0.3)
    
    # 4. Training Time
    ax4 = axes[1, 0]
    sns.barplot(data=successful_results, x='name', y='training_time', ax=ax4)
    ax4.set_title('Training Time Comparison')
    ax4.set_xlabel('Model')
    ax4.set_ylabel('Training Time (seconds)')
    ax4.tick_params(axis='x', rotation=45)
    
    # 5. Embedding Quality
    ax5 = axes[1, 1]
    sns.barplot(data=successful_results, x='name', y='embedding_avg_distance', ax=ax5)
    ax5.set_title('Embedding Diversity')
    ax5.set_xlabel('Model')
    ax5.set_ylabel('Average Embedding Distance')
    ax5.tick_params(axis='x', rotation=45)
    
    # 6. Revenue vs AUC Trade-off
    ax6 = axes[1, 2]
    ax6.scatter(successful_results['link_prediction_auc'], successful_results['revenue_proxy'], s=100, alpha=0.7)
    for i, row in successful_results.iterrows():
        ax6.annotate(row['name'], (row['link_prediction_auc'], row['revenue_proxy']), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax6.set_title('Revenue vs Link Prediction Trade-off')
    ax6.set_xlabel('Link Prediction AUC')
    ax6.set_ylabel('Revenue Proxy')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('graph_comprehensive_performance.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Performance visualizations saved to 'graph_comprehensive_performance.png'")


def run_comprehensive_evaluation():
    """Run comprehensive evaluation of graph-based recommenders."""
    print("=== COMPREHENSIVE GRAPH-BASED RECOMMENDER EVALUATION ===")
    
    # Setup environment
    users_df, items_df, history_df = setup_evaluation_environment()
    
    # Get all recommenders
    all_recommenders = {}
    
    # Add baseline recommenders
    baseline_recommenders = get_baseline_recommenders()
    all_recommenders.update(baseline_recommenders)
    
    # Add optimized graph recommenders
    graph_recommenders = get_optimized_graph_recommenders()
    all_recommenders.update(graph_recommenders)
    
    print(f"Evaluating {len(all_recommenders)} recommenders:")
    for name in all_recommenders.keys():
        print(f"  - {name}")
    
    # Evaluate each recommender
    results = []
    for name, recommender in all_recommenders.items():
        result = evaluate_single_recommender(name, recommender, users_df, items_df, history_df)
        results.append(result)
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by revenue proxy
    results_df = results_df.sort_values('revenue_proxy', ascending=False)
    
    # Save results
    results_df.to_csv('graph_comprehensive_evaluation_results.csv', index=False)
    print(f"\nResults saved to graph_comprehensive_evaluation_results.csv")
    
    # Run ablation study
    ablation_df = run_ablation_study(users_df, items_df, history_df)
    ablation_df.to_csv('graph_ablation_study_results.csv', index=False)
    print(f"Ablation study results saved to graph_ablation_study_results.csv")
    
    # Create visualizations
    create_performance_visualizations(results_df)
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    successful_results = results_df[results_df['status'] == 'SUCCESS']
    if len(successful_results) > 0:
        print("Performance Rankings (by revenue proxy):")
        for i, (_, row) in enumerate(successful_results.iterrows(), 1):
            print(f"  {i}. {row['name']}: {row['revenue_proxy']:.2f} revenue proxy, {row['link_prediction_auc']:.3f} AUC")
        
        # Best graph model vs best baseline
        graph_results = successful_results[successful_results['name'].str.contains('Optimized')]
        baseline_results = successful_results[successful_results['name'].str.contains('Baseline')]
        
        if len(graph_results) > 0 and len(baseline_results) > 0:
            best_graph = graph_results.iloc[0]
            best_baseline = baseline_results.iloc[0]
            improvement = ((best_graph['revenue_proxy'] / best_baseline['revenue_proxy']) - 1) * 100
            print(f"\nBest graph model vs best baseline:")
            print(f"  Graph: {best_graph['name']} - {best_graph['revenue_proxy']:.2f}")
            print(f"  Baseline: {best_baseline['name']} - {best_baseline['revenue_proxy']:.2f}")
            print(f"  Improvement: {improvement:.1f}%")
    
    return results_df, ablation_df


if __name__ == "__main__":
    try:
        results_df, ablation_df = run_comprehensive_evaluation()
        print("\nComprehensive graph evaluation completed successfully!")
    except Exception as e:
        print(f"Error in evaluation: {e}")
        import traceback
        traceback.print_exc()
    finally:
        spark.stop()
