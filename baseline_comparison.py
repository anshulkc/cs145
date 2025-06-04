"""
Comprehensive Comparison: Content-Based Recommenders vs Baselines

Compares content-based recommenders against existing baseline recommenders
using the CompetitionSimulator framework for consistency.

Includes primary and secondary metrics plus learning curves analysis.
"""

import os
import sys
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
    .appName("BaselineComparison") \
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

# Our recommenders
from content_based_recommenders import LogisticRegressionRecommender, KNeighborsRecommender, RandomForestRecommender


def run_comprehensive_comparison():
    """Run comprehensive comparison using CompetitionSimulator framework."""
    print("Comprehensive Baseline Comparison")
    print("Using CompetitionSimulator Framework")
    print("=" * 50)
    
    # Setup configuration
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['data_generation']['n_users'] = 1000
    config['data_generation']['n_items'] = 200
    config['data_generation']['seed'] = 42
    
    # Get simulation parameters
    train_iterations = config['simulation']['train_iterations']
    test_iterations = config['simulation']['test_iterations']
    
    print(f"Running simulation with {train_iterations} training iterations and {test_iterations} testing iterations")
    
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
    
    # Import configurations
    from content_based_recommenders import CONTENT_BASED_CONFIGS
    
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
    
    # Add content-based recommenders
    for config_def in CONTENT_BASED_CONFIGS:
        if config_def["class"] == "LogisticRegressionRecommender":
            recommender = LogisticRegressionRecommender(**config_def["parameters"])
        elif config_def["class"] == "KNeighborsRecommender":
            recommender = KNeighborsRecommender(**config_def["parameters"])
        elif config_def["class"] == "RandomForestRecommender":
            recommender = RandomForestRecommender(**config_def["parameters"])
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
                results.append({
                    "name": name,
                    "train_total_revenue": sum(train_revenue),
                    "test_total_revenue": sum(test_revenue),
                    "train_avg_revenue": np.mean(train_revenue),
                    "test_avg_revenue": np.mean(test_revenue),
                    "performance_change": performance_change,
                    "total_time": total_time,
                    "train_metrics": train_metrics,
                    "test_metrics": test_metrics,
                    "train_revenue": train_revenue,
                    "test_revenue": test_revenue,
                    **train_avg_metrics,
                    **test_avg_metrics
                })
                
                # Print summary
                print(f"  Training Phase - Total Revenue: {sum(train_revenue):.2f}")
                print(f"  Testing Phase - Total Revenue: {sum(test_revenue):.2f}")
                print(f"  Performance Change: {performance_change:.2f}%")
                print(f"  Total Time: {total_time:.1f}s")
                
            finally:
                # Cleanup
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                
        except Exception as e:
            total_time = time.time() - start_time
            print(f"  Failed: {e}")
            
            # Store failed result
            results.append({
                "name": name,
                "success": False,
                "error": str(e),
                "total_time": total_time
            })
    
    # Process results
    results_df = pd.DataFrame(results)
    
    # Filter successful results
    successful_results = [r for r in results if 'test_total_revenue' in r]
    failed_results = [r for r in results if 'success' in r and not r['success']]
    
    if successful_results:
        results_df = pd.DataFrame(successful_results)
        results_df = results_df.sort_values("test_total_revenue", ascending=False).reset_index(drop=True)
        
        # Display results
        print("\nComparison Results")
        print("=" * 50)
        
        print("\nTest Revenue Ranking:")
        print("-" * 30)
        for i, (_, row) in enumerate(results_df.iterrows(), 1):
            print(f"{i:2d}. {row['name']:<40} ${row['test_total_revenue']:>10,.2f}")
        
        summary_cols = ["name", "train_total_revenue", "test_total_revenue", 
                       "train_avg_revenue", "test_avg_revenue",
                       "train_precision_at_k", "test_precision_at_k",
                       "train_ndcg_at_k", "test_ndcg_at_k",
                       "train_mrr", "test_mrr"]
        available_cols = [col for col in summary_cols if col in results_df.columns]
        
        print(f"\nDetailed Metrics:")
        print(results_df[available_cols].to_string(index=False))
        
        # Analysis
        print(f"\nAnalysis:")
        best = results_df.iloc[0]
        our_results = results_df[results_df['name'].str.contains('(Ours)')]
        baseline_results = results_df[~results_df['name'].str.contains('(Ours)')]
        
        print(f"  Best Overall: {best['name']} - ${best['test_total_revenue']:,.2f}")
        
        if len(our_results) > 0 and len(baseline_results) > 0:
            our_avg = our_results['test_total_revenue'].mean()
            baseline_avg = baseline_results['test_total_revenue'].mean()
            improvement = (our_avg / baseline_avg - 1) * 100
            print(f"  Our methods average: ${our_avg:,.2f}")
            print(f"  Baseline methods average: ${baseline_avg:,.2f}")
            print(f"  Average improvement: {improvement:+.1f}%")
    
    # Display failures
    if failed_results:
        print(f"\nFailed Recommenders:")
        for result in failed_results:
            print(f"  {result['name']}: {result['error']}")
    
    return results


def evaluate_learning_curves():
    """
    Evaluate learning curves across training iterations.
    Tests how methods improve with more interaction data.
    """
    print("\nLearning Curves Analysis")
    print("-" * 30)
    
    # Setup data
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['data_generation']['n_users'] = 1000
    config['data_generation']['n_items'] = 200
    config['data_generation']['seed'] = 42
    
    data_generator = CompetitionDataGenerator(
        spark_session=spark,
        **config['data_generation']
    )
    
    users_df = data_generator.generate_users()
    items_df = data_generator.generate_items()
    history_df = data_generator.generate_initial_history(
        config['data_generation']['initial_history_density']
    )
    
    user_generator, item_generator = data_generator.setup_data_generators()
    
    # Test with different numbers of training iterations
    iteration_counts = [1, 3, 5, 7, 10]
    test_iterations = 3
    
    # Select top methods for learning curve analysis
    methods_to_test = [
        {
            "name": "LogisticRegression_L1",
            "class": LogisticRegressionRecommender,
            "params": {"seed": 42, "regularization": "l1", "C": 0.1, "revenue_weight": 1.0}
        },
        {
            "name": "RandomForest_Default", 
            "class": RandomForestRecommender,
            "params": {"seed": 42, "n_estimators": 50, "revenue_weight": 1.0}
        },
        {
            "name": "KNN_Euclidean",
            "class": KNeighborsRecommender,
            "params": {"seed": 42, "n_neighbors": 5, "metric": "euclidean", "revenue_weight": 1.0}
        }
    ]
    
    learning_curves = {}
    
    for method in methods_to_test:
        print(f"\nEvaluating learning curve for {method['name']}...")
        method_curves = {"train_iterations": [], "test_revenue": [], "test_precision": [], "test_ndcg": []}
        
        for train_iter in iteration_counts:
            try:
                # Create fresh recommender instance
                recommender = method["class"](**method["params"])
                
                # Initialize with base interactions
                recommender.fit(log=data_generator.history_df, 
                               user_features=users_df, 
                               item_features=items_df)
                
                # Create unique temporary directory
                unique_id = str(uuid.uuid4())[:8]
                temp_dir = f"/tmp/learn_{method['name']}_{train_iter}_{unique_id}"
                
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                
                try:
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
                        train_iterations=train_iter,
                        test_iterations=test_iterations,
                        user_frac=config['simulation']['user_fraction'],
                        k=config['simulation']['k'],
                        filter_seen_items=config['simulation']['filter_seen_items'],
                        retrain=config['simulation']['retrain']
                    )
                    
                    # Aggregate test metrics
                    avg_test_revenue = np.mean(test_revenue)
                    avg_test_precision = np.mean([m['precision_at_k'] for m in test_metrics])
                    avg_test_ndcg = np.mean([m['ndcg_at_k'] for m in test_metrics])
                    
                    method_curves["train_iterations"].append(train_iter)
                    method_curves["test_revenue"].append(avg_test_revenue)
                    method_curves["test_precision"].append(avg_test_precision)
                    method_curves["test_ndcg"].append(avg_test_ndcg)
                    
                    print(f"  {train_iter} iterations -> Revenue: ${avg_test_revenue:.2f}, Precision: {avg_test_precision:.3f}")
                    
                finally:
                    # Cleanup
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    
            except Exception as e:
                print(f"  Failed at {train_iter} iterations: {e}")
                continue
        
        learning_curves[method["name"]] = method_curves
    
    # Plot learning curves
    if learning_curves:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ["test_revenue", "test_precision", "test_ndcg"]
        titles = ["Test Revenue vs Training Iterations", "Test Precision@K vs Training Iterations", "Test NDCG@K vs Training Iterations"]
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            for method_name, curves in learning_curves.items():
                if len(curves["train_iterations"]) > 0:
                    axes[i].plot(curves["train_iterations"], curves[metric], marker='o', label=method_name)
            
            axes[i].set_xlabel("Training Iterations")
            axes[i].set_ylabel(metric.replace("test_", "").replace("_", " ").title())
            axes[i].set_title(title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("learning_curves_analysis.png", dpi=300, bbox_inches='tight')
        print("Learning curves plot saved as learning_curves_analysis.png")
        plt.close()
    
    return learning_curves


def main():
    """Main function to run comprehensive comparison with learning curves."""
    try:
        print("CS145 Checkpoint 1: Comprehensive Evaluation")
        print("=" * 50)
        print("Primary and secondary metrics with learning curves")
        
        # Run baseline comparison
        results = run_comprehensive_comparison()
        
        # Evaluate learning curves
        learning_results = evaluate_learning_curves()
        
        print("\nEvaluation complete")
        
    except Exception as e:
        print(f"Comparison failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main() 