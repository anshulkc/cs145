"""
Comprehensive Comparison: Sequence-Based Recommenders vs Baselines

Compares sequence-based recommenders against existing baseline recommenders
using the CompetitionSimulator framework for consistency.

Includes primary and secondary metrics plus learning curves analysis.
"""

import os
import sys
# Import from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import time
import copy
import tempfile
import uuid

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SequenceBaselineComparison") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Import modules
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
from config import DEFAULT_CONFIG

# Baseline recommenders
from sample_recommenders import (
    RandomRecommender,
    PopularityRecommender,
    ContentBasedRecommender,
    SVMRecommender
)

# Our sequence recommenders
from sequence_based_recommenders import (
    AutoRegressiveRecommender,
    RNNRecommender,
    GRURecommender,
    TransformerRecommender,
    SEQUENCE_BASED_CONFIGS
)


def run_comprehensive_comparison():
    """Run comprehensive comparison using CompetitionSimulator framework."""
    print("=== Sequence-Based Recommenders vs Baselines ===")
    print("Using CompetitionSimulator Framework")
    print("=" * 60)
    
    # Setup configuration with higher interaction density for sequences
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['data_generation']['n_users'] = 1000
    config['data_generation']['n_items'] = 200
    config['data_generation']['seed'] = 42
    
    # Use higher density for better sequence data (as recommended by interaction_analysis.py)
    config['data_generation']['initial_history_density'] = 0.02  # Instead of default 0.001
    
    # Get simulation parameters
    train_iterations = config['simulation']['train_iterations']
    test_iterations = config['simulation']['test_iterations']
    
    print(f"Running simulation with {train_iterations} training iterations and {test_iterations} testing iterations")
    print(f"Using enhanced interaction density: {config['data_generation']['initial_history_density']}")
    
    # Initialize data generator
    data_generator = CompetitionDataGenerator(
        spark_session=spark,
        **config['data_generation']
    )
    
    # Generate data
    users_df = data_generator.generate_users()
    print(f"Generated {users_df.count()} users")
    
    items_df = data_generator.generate_items()
    print(f"Generated {items_df.count()} items")
    
    history_df = data_generator.generate_initial_history(
        config['data_generation']['initial_history_density']
    )
    print(f"Generated {history_df.count()} initial interactions")
    
    # Set up data generators for simulator
    user_generator, item_generator = data_generator.setup_data_generators()
    
    # Initialize recommenders
    recommenders = []
    recommender_names = []
    
    # Add baseline recommenders
    baseline_recommenders = [
        SVMRecommender(seed=42),
        RandomRecommender(seed=42),
        PopularityRecommender(alpha=1.0, seed=42),
        ContentBasedRecommender(similarity_threshold=0.0, seed=42)
    ]
    baseline_names = ["SVM", "Random", "Popularity", "ContentBased"]
    
    recommenders.extend(baseline_recommenders)
    recommender_names.extend(baseline_names)
    
    # Add sequence-based recommenders - select a subset for manageable runtime
    selected_configs = [
        "AutoRegressive_Order1",
        "AutoRegressive_Order2", 
        "AutoRegressive_Order3",
        "RNN_Basic",
        "GRU_Efficient",
        "Transformer_Small"
    ]
    
    for config_def in SEQUENCE_BASED_CONFIGS:
        if config_def["name"] not in selected_configs:
            continue
            
        if config_def["class"] == "AutoRegressiveRecommender":
            recommender = AutoRegressiveRecommender(**config_def["parameters"])
        elif config_def["class"] == "RNNRecommender":
            recommender = RNNRecommender(**config_def["parameters"])
        elif config_def["class"] == "GRURecommender":
            recommender = GRURecommender(**config_def["parameters"])
        elif config_def["class"] == "TransformerRecommender":
            recommender = TransformerRecommender(**config_def["parameters"])
        else:
            continue
            
        recommenders.append(recommender)
        recommender_names.append(f"{config_def['name']} (Ours)")
    
    print(f"\nEvaluating {len(recommenders)} recommenders:")
    for i, name in enumerate(recommender_names, 1):
        print(f"  {i}. {name}")
    
    # Initialize recommenders with initial history
    for recommender in recommenders:
        recommender.fit(log=data_generator.history_df, 
                        user_features=users_df, 
                        item_features=items_df)
    
    # Evaluate each recommender
    results = []
    
    for name, recommender in zip(recommender_names, recommenders):
        print(f"\nEvaluating {name}:")
        start_time = time.time()
        
        try:
            # Create unique temporary directory
            unique_id = str(uuid.uuid4())[:8]
            temp_dir = f"/tmp/sim_{name.replace(' ', '_').replace('(', '').replace(')', '')}_{unique_id}"
            
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            try:
                # Initialize simulator
                simulator = CompetitionSimulator(
                    user_generator=user_generator,
                    item_generator=item_generator,
                    data_dir=temp_dir,
                    log_df=data_generator.history_df,
                    conversion_noise_mean=config['simulation']['conversion_noise_mean'],
                    conversion_noise_std=config['simulation']['conversion_noise_std'],
                    spark_session=spark,
                    seed=config['data_generation']['seed']
                )
                
                # Run simulation
                train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
                    recommender=recommender,
                    train_iterations=train_iterations,
                    test_iterations=test_iterations,
                    user_frac=config['simulation']['user_fraction'],
                    k=config['simulation']['k'],
                    filter_seen_items=config['simulation']['filter_seen_items'],
                    retrain=config['simulation']['retrain']
                )
                
                # Calculate metrics
                train_avg_metrics = {}
                for metric_name in train_metrics[0].keys():
                    values = [metrics[metric_name] for metrics in train_metrics]
                    train_avg_metrics[f"train_{metric_name}"] = np.mean(values)
                
                test_avg_metrics = {}
                for metric_name in test_metrics[0].keys():
                    values = [metrics[metric_name] for metrics in test_metrics]
                    test_avg_metrics[f"test_{metric_name}"] = np.mean(values)
                
                total_time = time.time() - start_time
                
                # Calculate performance change
                performance_change = ((sum(test_revenue) / len(test_revenue)) / (sum(train_revenue) / len(train_revenue)) - 1) * 100
                
                # Store results
                result = {
                    "name": name,
                    "train_total_revenue": sum(train_revenue),
                    "test_total_revenue": sum(test_revenue),
                    "train_avg_revenue": np.mean(train_revenue),
                    "test_avg_revenue": np.mean(test_revenue),
                    "performance_change": performance_change,
                    "training_time": total_time,
                    **train_avg_metrics,
                    **test_avg_metrics
                }
                
                results.append(result)
                
                print(f"  Training revenue: ${sum(train_revenue):.2f}")
                print(f"  Testing revenue: ${sum(test_revenue):.2f}")
                print(f"  Performance change: {performance_change:.2f}%")
                print(f"  Training time: {total_time:.2f}s")
                
            finally:
                # Clean up temporary directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
        except Exception as e:
            print(f"  ERROR: {str(e)}")
            # Add failed result
            result = {
                "name": name,
                "train_total_revenue": 0,
                "test_total_revenue": 0,
                "train_avg_revenue": 0,
                "test_avg_revenue": 0,
                "performance_change": 0,
                "training_time": 0,
                "status": "FAILED",
                "error": str(e)
            }
            results.append(result)
    
    # Create results DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by test revenue
    results_df = results_df.sort_values('test_total_revenue', ascending=False)
    
    print(f"\n=== SEQUENCE RECOMMENDER RESULTS ===")
    print("=" * 50)
    
    # Print results table
    print("\nRankings by Test Revenue:")
    print("-" * 80)
    print(f"{'Rank':<4} {'Recommender':<30} {'Test Revenue':<12} {'Train Revenue':<12} {'Change':<8} {'Time':<8}")
    print("-" * 80)
    
    for i, (_, row) in enumerate(results_df.iterrows(), 1):
        name = row['name']
        if len(name) > 28:
            name = name[:25] + "..."
        
        status = row.get('status', 'SUCCESS')
        if status == 'FAILED':
            print(f"{i:<4} {name:<30} {'FAILED':<12} {'FAILED':<12} {'--':<8} {'--':<8}")
        else:
            print(f"{i:<4} {name:<30} ${row['test_total_revenue']:<11.2f} ${row['train_total_revenue']:<11.2f} {row['performance_change']:<7.1f}% {row['training_time']:<7.1f}s")
    
    # Save results
    results_df.to_csv('ckpt2/sequence_recommender_results.csv', index=False)
    print(f"\nResults saved to 'ckpt2/sequence_recommender_results.csv'")
    
    return results_df


def main():
    """Main execution function."""
    print("Starting Sequence-Based Recommender Analysis...")
    print("This may take several minutes due to neural network training...")
    
    try:
        results_df = run_comprehensive_comparison()
        
        print(f"\n🎯 ANALYSIS COMPLETE!")
        print("=" * 50)
        
        # Summary statistics
        valid_results = results_df[~results_df.get('status', 'SUCCESS').eq('FAILED')]
        
        if len(valid_results) > 0:
            best_performer = valid_results.iloc[0]
            our_models = valid_results[valid_results['name'].str.contains('Ours', na=False)]
            baselines = valid_results[~valid_results['name'].str.contains('Ours', na=False)]
            
            print(f"🏆 Best Performer: {best_performer['name']}")
            print(f"   Test Revenue: ${best_performer['test_total_revenue']:.2f}")
            
            if len(our_models) > 0 and len(baselines) > 0:
                our_avg = our_models['test_total_revenue'].mean()
                baseline_avg = baselines['test_total_revenue'].mean()
                improvement = ((our_avg / baseline_avg) - 1) * 100
                
                print(f"\n📊 Performance Summary:")
                print(f"   Our Models Average: ${our_avg:.2f}")
                print(f"   Baseline Average: ${baseline_avg:.2f}")
                print(f"   Improvement: {improvement:.1f}%")
        
        print(f"\n📁 Files Generated:")
        print(f"   - ckpt2/sequence_recommender_results.csv")
        
    except Exception as e:
        print(f"\n❌ Analysis failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 