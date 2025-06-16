#!/usr/bin/env python3

"""
Comprehensive Graph Models Evaluation - Checkpoint 3

Evaluates all three graph-based models against baselines:
- GCN (Graph Convolutional Network)
- Node2Vec (Random walk embeddings)  
- LightGCN (Simplified GCN for collaborative filtering)

Results saved in ckpt3 directory for the final report.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from data_generator import CompetitionDataGenerator
from config import DEFAULT_CONFIG

# Import all graph models

from graph_based_recommenders import Node2VecRecommender, LightGCNRecommender, GCNRecommender

# Initialize Spark
spark = SparkSession.builder.appName('ComprehensiveGraphEval').master('local[*]').getOrCreate()
spark.sparkContext.setLogLevel('WARN')

def setup_evaluation_environment():
    """Setup evaluation environment."""
    config = DEFAULT_CONFIG.copy()
    config['data_generation']['n_users'] = 500
    config['data_generation']['n_items'] = 100
    config['data_generation']['seed'] = 42
    
    # Generate data
    data_generator = CompetitionDataGenerator(spark_session=spark, **config['data_generation'])
    users_df = data_generator.generate_users()
    items_df = data_generator.generate_items()
    history_df = data_generator.generate_initial_history(0.05)
    
    print(f"Dataset: {users_df.count()} users, {items_df.count()} items, {history_df.count()} interactions")
    
    return users_df, items_df, history_df

def get_baseline_recommenders():
    """Get baseline recommenders for comparison."""
    try:
        from sample_recommenders import RandomRecommender, PopularityRecommender
        return {
            'Random_Baseline': RandomRecommender(seed=42),
            'Popularity_Baseline': PopularityRecommender(alpha=1.0, seed=42)
        }
    except ImportError:
        print("Warning: Baseline recommenders not available")
        return {}

def get_graph_recommenders():
    """Get all graph recommenders with optimized parameters."""
    return {
        'GCN_Optimized': GCNRecommender(
            seed=42,
            embedding_dim=64,
            hidden_dims=[128, 64],
            dropout=0.3,
            learning_rate=0.01,
            epochs=30,
            batch_size=512,
            revenue_weight=0.1,
            negative_sampling_ratio=2.0,
            early_stopping_patience=8
        ),
        'Node2Vec_Optimized': Node2VecRecommender(
            seed=42,
            embedding_dim=64,
            walk_length=15,
            num_walks=10,
            window_size=5,
            p=0.5,
            q=2.0,
            learning_rate=0.01,
            epochs=20,
            num_negative=5,
            batch_size=512,
            early_stopping_patience=5,
            revenue_weight=0.1
        ),
        'LightGCN_Optimized': LightGCNRecommender(
            seed=42,
            embedding_dim=64,
            n_layers=2,
            learning_rate=0.005,
            epochs=25,
            batch_size=512,
            reg_weight=1e-5,
            negative_sampling_ratio=2.0,
            early_stopping_patience=5,
            revenue_weight=0.1
        )
    }

def evaluate_single_recommender(name: str, recommender, users_df, items_df, history_df):
    """Evaluate a single recommender."""
    print(f"\nEvaluating {name}...")
    
    try:
        # Fit the model
        recommender.fit(log=history_df, user_features=users_df, item_features=items_df)
        
        # Generate predictions
        predictions = recommender.predict(
            log=history_df,
            k=10,
            users=users_df,
            items=items_df,
            user_features=users_df,
            item_features=items_df,
            filter_seen_items=True
        )
        
        # Calculate metrics
        num_predictions = predictions.count()
        avg_relevance = predictions.agg({'relevance': 'avg'}).collect()[0][0] or 0.0
        
        # Calculate total revenue (relevance * item price)
        items_price = items_df.select('item_idx', sf.col('price').alias('item_price'))
        predictions_with_price = predictions.join(items_price, 'item_idx', 'left')
        total_revenue = predictions_with_price.agg({'item_price': 'sum'}).collect()[0][0] or 0.0
        
        # Calculate revenue proxy (relevance * price)
        revenue_proxy = predictions_with_price.selectExpr('relevance * item_price as revenue').agg({'revenue': 'sum'}).collect()[0][0] or 0.0
        
        # Calculate link prediction AUC for graph models
        auc_score = 0.0
        if hasattr(recommender, '_predict_link_probabilities'):
            auc_score = calculate_link_prediction_auc(recommender, users_df, items_df, history_df)
        
        result = {
            'name': name,
            'status': 'SUCCESS',
            'num_predictions': num_predictions,
            'avg_relevance': float(avg_relevance),
            'total_revenue': float(total_revenue),
            'revenue_proxy': float(revenue_proxy),
            'link_prediction_auc': float(auc_score)
        }
        
        print(f"  {name} - Revenue Proxy: {result['revenue_proxy']:.2f}, Total Revenue: {result['total_revenue']:.2f}, AUC: {result['link_prediction_auc']:.3f}")
        
        return result
        
    except Exception as e:
        print(f"  ERROR in {name}: {str(e)[:100]}...")
        return {
            'name': name,
            'status': 'FAILED',
            'error': str(e)[:200],
            'num_predictions': 0,
            'avg_relevance': 0.0,
            'total_revenue': 0.0,
            'revenue_proxy': 0.0,
            'link_prediction_auc': 0.0
        }

def calculate_link_prediction_auc(model, users_df, items_df, history_df) -> float:
    """Calculate link prediction AUC for graph models."""
    try:
        # Get positive interactions (sample for speed)
        positive_interactions = history_df.select('user_idx', 'item_idx').limit(100).toPandas()
        positive_pairs = [(row['user_idx'], row['item_idx']) for _, row in positive_interactions.iterrows()]
        
        # Sample negative interactions
        all_users = users_df.select('user_idx').limit(50).toPandas()['user_idx'].tolist()
        all_items = items_df.select('item_idx').limit(50).toPandas()['item_idx'].tolist()
        
        positive_set = set(positive_pairs)
        negative_pairs = []
        
        # Sample same number of negatives as positives
        target_negatives = len(positive_pairs)
        while len(negative_pairs) < target_negatives:
            user_idx = np.random.choice(all_users)
            item_idx = np.random.choice(all_items)
            if (user_idx, item_idx) not in positive_set:
                negative_pairs.append((user_idx, item_idx))
        
        # Get predictions for all pairs
        all_pairs = positive_pairs + negative_pairs
        labels = [1] * len(positive_pairs) + [0] * len(negative_pairs)
        
        predictions = model._predict_link_probabilities(all_pairs)
        
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, predictions)
        return float(auc)
        
    except Exception as e:
        print(f"Error calculating AUC: {e}")
        return 0.0

def create_comprehensive_visualizations(results_df: pd.DataFrame):
    """Create comprehensive performance visualizations."""
    # Filter successful results
    success_df = results_df[results_df['status'] == 'SUCCESS'].copy()
    
    if len(success_df) == 0:
        print("No successful results to plot")
        return
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Comprehensive Graph-Based Recommenders Evaluation', fontsize=16, fontweight='bold')
    
    # Colors: red for baselines, different colors for graph models
    colors = []
    for name in success_df['name']:
        if 'Baseline' in name:
            colors.append('red')
        elif 'GCN' in name:
            colors.append('green')
        elif 'Node2Vec' in name:
            colors.append('blue')
        elif 'LightGCN' in name:
            colors.append('purple')
        else:
            colors.append('gray')
    
    # 1. Revenue Proxy Comparison
    ax1 = axes[0, 0]
    bars1 = ax1.bar(success_df['name'], success_df['revenue_proxy'], color=colors)
    ax1.set_title('Revenue Proxy Comparison')
    ax1.set_ylabel('Revenue Proxy')
    ax1.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars1, success_df['revenue_proxy']):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(success_df['revenue_proxy'])*0.01,
                 f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 2. Total Revenue Comparison
    ax2 = axes[0, 1]
    bars2 = ax2.bar(success_df['name'], success_df['total_revenue'], color=colors)
    ax2.set_title('Total Revenue Comparison')
    ax2.set_ylabel('Total Revenue')
    ax2.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars2, success_df['total_revenue']):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(success_df['total_revenue'])*0.01,
                 f'{value:.0f}', ha='center', va='bottom', fontsize=8)
    
    # 3. Link Prediction AUC (graph models only)
    graph_df = success_df[~success_df['name'].str.contains('Baseline')].copy()
    if len(graph_df) > 0:
        ax3 = axes[1, 0]
        graph_colors = [c for name, c in zip(success_df['name'], colors) if 'Baseline' not in name]
        bars3 = ax3.bar(graph_df['name'], graph_df['link_prediction_auc'], color=graph_colors)
        ax3.set_title('Link Prediction AUC (Graph Models)')
        ax3.set_ylabel('AUC Score')
        ax3.tick_params(axis='x', rotation=45)
        for bar, value in zip(bars3, graph_df['link_prediction_auc']):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                     f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    # 4. Average Relevance
    ax4 = axes[1, 1]
    bars4 = ax4.bar(success_df['name'], success_df['avg_relevance'], color=colors)
    ax4.set_title('Average Relevance Score')
    ax4.set_ylabel('Average Relevance')
    ax4.tick_params(axis='x', rotation=45)
    for bar, value in zip(bars4, success_df['avg_relevance']):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(success_df['avg_relevance'])*0.01,
                 f'{value:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('ckpt3/comprehensive_graph_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nComprehensive visualization saved as 'ckpt3/comprehensive_graph_evaluation.png'")

def print_comprehensive_summary(results_df: pd.DataFrame):
    """Print comprehensive evaluation summary."""
    print("\n" + "="*80)
    print("CHECKPOINT 3: COMPREHENSIVE GRAPH MODELS EVALUATION SUMMARY")
    print("="*80)
    
    success_df = results_df[results_df['status'] == 'SUCCESS'].copy()
    
    if len(success_df) == 0:
        print("No successful evaluations to summarize")
        return
    
    # Sort by revenue proxy (primary metric)
    success_df = success_df.sort_values('revenue_proxy', ascending=False)
    
    print(f"\nRanking by Revenue Proxy:")
    for i, row in enumerate(success_df.itertuples(), 1):
        print(f"  {i}. {row.name}: {row.revenue_proxy:.2f}")
    
    print(f"\nRanking by Total Revenue:")
    revenue_sorted = success_df.sort_values('total_revenue', ascending=False)
    for i, row in enumerate(revenue_sorted.itertuples(), 1):
        print(f"  {i}. {row.name}: {row.total_revenue:.2f}")
    
    # Graph models vs baselines analysis
    baseline_names = [name for name in success_df['name'] if 'Baseline' in name]
    graph_names = [name for name in success_df['name'] if 'Baseline' not in name]
    
    if baseline_names and graph_names:
        print(f"\nGraph Models vs Baseline Performance:")
        best_baseline_revenue = success_df[success_df['name'].isin(baseline_names)]['revenue_proxy'].max()
        
        print(f"\nIndividual Graph Model Performance vs Best Baseline:")
        for graph_name in graph_names:
            graph_revenue = success_df[success_df['name'] == graph_name]['revenue_proxy'].iloc[0]
            if best_baseline_revenue > 0:
                improvement = ((graph_revenue - best_baseline_revenue) / best_baseline_revenue) * 100
                print(f"  {graph_name}: {improvement:+.1f}% vs best baseline")
    
    # Graph-specific metrics
    graph_df = success_df[~success_df['name'].str.contains('Baseline')].copy()
    if len(graph_df) > 0:
        print(f"\nGraph Model Link Prediction AUC:")
        auc_sorted = graph_df.sort_values('link_prediction_auc', ascending=False)
        for i, row in enumerate(auc_sorted.itertuples(), 1):
            print(f"  {i}. {row.name}: {row.link_prediction_auc:.3f}")

def run_comprehensive_evaluation():
    """Main evaluation function."""
    print("Checkpoint 3: Comprehensive Graph Models Evaluation")
    print("="*60)
    
    # Setup environment
    users_df, items_df, history_df = setup_evaluation_environment()
    
    # Get all recommenders
    baseline_recommenders = get_baseline_recommenders()
    graph_recommenders = get_graph_recommenders()
    
    all_recommenders = {**baseline_recommenders, **graph_recommenders}
    
    print(f"\nEvaluating {len(all_recommenders)} recommenders:")
    for i, name in enumerate(all_recommenders.keys(), 1):
        print(f"  {i}. {name}")
    
    # Evaluate each recommender
    results = []
    for name, recommender in all_recommenders.items():
        result = evaluate_single_recommender(name, recommender, users_df, items_df, history_df)
        results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Save results
    results_df.to_csv('ckpt3/comprehensive_graph_evaluation_results.csv', index=False)
    print(f"\nResults saved to 'ckpt3/comprehensive_graph_evaluation_results.csv'")
    
    # Create visualizations
    create_comprehensive_visualizations(results_df)
    
    # Print summary
    print_comprehensive_summary(results_df)
    
    print("\nComprehensive graph evaluation completed successfully!")
    return results_df

if __name__ == "__main__":
    # Ensure ckpt3 directory exists
    os.makedirs('ckpt3', exist_ok=True)
    results_df = run_comprehensive_evaluation() 