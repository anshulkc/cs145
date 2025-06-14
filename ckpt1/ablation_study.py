"""
Ablation Study: Hyperparameter Impact Analysis

Systematic evaluation of hyperparameter impact on content-based recommendation methods.
Tests revenue weight, regularization strength, KNN neighbors, and Random Forest trees.
"""

import os
import sys
# Import from repo root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


import time
import shutil
import copy
import tempfile
import uuid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf

# Initialize Spark session
spark = SparkSession.builder \
    .appName("AblationStudy") \
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

from content_based_recommenders import (
    LogisticRegressionRecommender, 
    KNeighborsRecommender, 
    RandomForestRecommender
)


def setup_ablation_data():
    """Setup data generators for ablation study."""
    config = copy.deepcopy(DEFAULT_CONFIG)
    config['data_generation']['n_users'] = 500
    config['data_generation']['n_items'] = 100
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
    
    return user_generator, item_generator, data_generator, users_df, items_df, config


def evaluate_single_configuration(config_name, recommender_class, params, 
                                 user_gen, item_gen, data_gen, users_df, items_df, config):
    """Evaluate a single hyperparameter configuration."""
    try:
        # Create recommender
        recommender = recommender_class(**params)
        
        # Initialize
        recommender.fit(log=data_gen.history_df,
                       user_features=users_df,
                       item_features=items_df)
        
        # Create unique temporary directory
        unique_id = str(uuid.uuid4())[:8]
        temp_dir = f"/tmp/ablation_{config_name}_{unique_id}"
        
        # Ensure directory doesn't exist
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        
        try:
            simulator = CompetitionSimulator(
                user_generator=user_gen,
                item_generator=item_gen,
                data_dir=temp_dir,
                log_df=data_gen.history_df,
                conversion_noise_mean=config['simulation']['conversion_noise_mean'],
                conversion_noise_std=config['simulation']['conversion_noise_std'],
                spark_session=spark,
                seed=config['data_generation']['seed']
            )
            
            # Run simulation (shorter for ablation)
            train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
                recommender=recommender,
                train_iterations=3,
                test_iterations=2,
                user_frac=config['simulation']['user_fraction'],
                k=config['simulation']['k'],
                filter_seen_items=config['simulation']['filter_seen_items'],
                retrain=config['simulation']['retrain']
            )
            
            # Calculate metrics
            avg_test_revenue = np.mean(test_revenue)
            avg_test_precision = np.mean([m['precision_at_k'] for m in test_metrics])
            avg_test_ndcg = np.mean([m['ndcg_at_k'] for m in test_metrics])
            avg_test_mrr = np.mean([m['mrr'] for m in test_metrics])
            
            result = {
                "config": config_name,
                "test_revenue": avg_test_revenue,
                "test_precision_at_k": avg_test_precision,
                "test_ndcg_at_k": avg_test_ndcg,
                "test_mrr": avg_test_mrr,
                "success": True
            }
            
        finally:
            # Always clean up
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        
        return result
        
    except Exception as e:
        print(f"  Failed: {e}")
        return {
            "config": config_name,
            "success": False,
            "error": str(e)
        }


def revenue_weight_ablation():
    """Evaluate impact of revenue weight parameter."""
    print("\nRevenue Weight Ablation Study")
    print("-" * 40)
    
    user_gen, item_gen, data_gen, users_df, items_df, config = setup_ablation_data()
    
    revenue_weights = [1.0, 1.5, 2.0, 2.5]
    results = []
    
    for weight in revenue_weights:
        config_name = f"RevWeight_{weight}"
        print(f"Testing revenue weight: {weight}")
        
        result = evaluate_single_configuration(
            config_name=config_name,
            recommender_class=LogisticRegressionRecommender,
            params={"regularization": "l2", "C": 1.0, "revenue_weight": weight, "seed": 42},
            user_gen=user_gen, item_gen=item_gen, data_gen=data_gen,
            users_df=users_df, items_df=items_df, config=config
        )
        
        if result["success"]:
            print(f"  Revenue: ${result['test_revenue']:.2f}, Precision: {result['test_precision_at_k']:.3f}")
        
        results.append(result)
    
    return results


def regularization_strength_ablation():
    """Evaluate impact of regularization strength."""
    print("\nRegularization Strength Ablation Study")
    print("-" * 40)
    
    user_gen, item_gen, data_gen, users_df, items_df, config = setup_ablation_data()
    
    c_values = [0.01, 0.1, 1.0, 10.0]
    results = []
    
    for c_val in c_values:
        config_name = f"RegStrength_C_{c_val}"
        print(f"Testing regularization C: {c_val}")
        
        result = evaluate_single_configuration(
            config_name=config_name,
            recommender_class=LogisticRegressionRecommender,
            params={"regularization": "l2", "C": c_val, "revenue_weight": 1.0, "seed": 42},
            user_gen=user_gen, item_gen=item_gen, data_gen=data_gen,
            users_df=users_df, items_df=items_df, config=config
        )
        
        if result["success"]:
            print(f"  Revenue: ${result['test_revenue']:.2f}, Precision: {result['test_precision_at_k']:.3f}")
        
        results.append(result)
    
    return results


def knn_neighbors_ablation():
    """Evaluate impact of KNN neighbor count."""
    print("\nKNN Neighbors Ablation Study")
    print("-" * 40)
    
    user_gen, item_gen, data_gen, users_df, items_df, config = setup_ablation_data()
    
    k_values = [3, 5, 10, 20]
    results = []
    
    for k_val in k_values:
        config_name = f"KNN_k{k_val}"
        print(f"Testing KNN neighbors: {k_val}")
        
        result = evaluate_single_configuration(
            config_name=config_name,
            recommender_class=KNeighborsRecommender,
            params={"n_neighbors": k_val, "metric": "euclidean", "revenue_weight": 1.0, "seed": 42},
            user_gen=user_gen, item_gen=item_gen, data_gen=data_gen,
            users_df=users_df, items_df=items_df, config=config
        )
        
        if result["success"]:
            print(f"  Revenue: ${result['test_revenue']:.2f}, Precision: {result['test_precision_at_k']:.3f}")
        
        results.append(result)
    
    return results


def random_forest_trees_ablation():
    """Evaluate impact of Random Forest tree count."""
    print("\nRandom Forest Trees Ablation Study") 
    print("-" * 40)
    
    user_gen, item_gen, data_gen, users_df, items_df, config = setup_ablation_data()
    
    tree_counts = [25, 50, 100]
    results = []
    
    for n_trees in tree_counts:
        config_name = f"RF_trees{n_trees}"
        print(f"Testing Random Forest trees: {n_trees}")
        
        result = evaluate_single_configuration(
            config_name=config_name,
            recommender_class=RandomForestRecommender,
            params={"n_estimators": n_trees, "max_depth": None, "revenue_weight": 1.0, "seed": 42},
            user_gen=user_gen, item_gen=item_gen, data_gen=data_gen,
            users_df=users_df, items_df=items_df, config=config
        )
        
        if result["success"]:
            print(f"  Revenue: ${result['test_revenue']:.2f}, Precision: {result['test_precision_at_k']:.3f}")
        
        results.append(result)
    
    return results


def plot_ablation_results(revenue_results, reg_results, knn_results, rf_results):
    """Plot ablation study results."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    def filter_successful(results):
        return [r for r in results if r.get("success", False)]
    
    # Revenue weight ablation
    rev_success = filter_successful(revenue_results)
    if len(rev_success) > 0:
        rev_df = pd.DataFrame(rev_success)
        rev_weights = rev_df["config"].str.extract(r"RevWeight_(\d+\.?\d*)")[0].astype(float)
        axes[0,0].plot(rev_weights, rev_df["test_revenue"], marker='o', linewidth=2, markersize=8)
        axes[0,0].set_title("Revenue Weight Impact")
        axes[0,0].set_xlabel("Revenue Weight")
        axes[0,0].set_ylabel("Test Revenue")
        axes[0,0].grid(True, alpha=0.3)
    
    # Regularization strength ablation
    reg_success = filter_successful(reg_results)
    if len(reg_success) > 0:
        reg_df = pd.DataFrame(reg_success)
        c_values = reg_df["config"].str.extract(r"RegStrength_C_(\d+\.?\d*)")[0].astype(float)
        axes[0,1].semilogx(c_values, reg_df["test_revenue"], marker='o', linewidth=2, markersize=8)
        axes[0,1].set_title("Regularization Strength Impact")
        axes[0,1].set_xlabel("C Value (log scale)")
        axes[0,1].set_ylabel("Test Revenue")
        axes[0,1].grid(True, alpha=0.3)
    
    # KNN neighbors ablation
    knn_success = filter_successful(knn_results)
    if len(knn_success) > 0:
        knn_df = pd.DataFrame(knn_success)
        k_values = knn_df["config"].str.extract(r"KNN_k(\d+)")[0].astype(int)
        axes[1,0].plot(k_values, knn_df["test_revenue"], marker='o', linewidth=2, markersize=8)
        axes[1,0].set_title("KNN Neighbors Impact")
        axes[1,0].set_xlabel("Number of Neighbors (k)")
        axes[1,0].set_ylabel("Test Revenue")
        axes[1,0].grid(True, alpha=0.3)
    
    # Random Forest trees ablation
    rf_success = filter_successful(rf_results)
    if len(rf_success) > 0:
        rf_df = pd.DataFrame(rf_success)
        tree_counts = rf_df["config"].str.extract(r"RF_trees(\d+)")[0].astype(int)
        axes[1,1].plot(tree_counts, rf_df["test_revenue"], marker='o', linewidth=2, markersize=8)
        axes[1,1].set_title("Random Forest Trees Impact")
        axes[1,1].set_xlabel("Number of Trees")
        axes[1,1].set_ylabel("Test Revenue")
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("ablation_study_results.png", dpi=300, bbox_inches='tight')
    print("Ablation study plots saved as ablation_study_results.png")
    plt.close()


def analyze_ablation_results(revenue_results, reg_results, knn_results, rf_results):
    """Analyze and summarize ablation study findings."""
    print("\nAblation Study Analysis")
    print("-" * 40)
    
    def find_best_config(results, metric="test_revenue"):
        successful = [r for r in results if r.get("success", False)]
        if not successful:
            return None
        return max(successful, key=lambda x: x[metric])
    
    # Find best configurations
    best_rev = find_best_config(revenue_results)
    if best_rev:
        print(f"Best Revenue Weight: {best_rev['config']} -> ${best_rev['test_revenue']:.2f}")
    
    best_reg = find_best_config(reg_results)
    if best_reg:
        print(f"Best Regularization: {best_reg['config']} -> ${best_reg['test_revenue']:.2f}")
    
    best_knn = find_best_config(knn_results)
    if best_knn:
        print(f"Best KNN Config: {best_knn['config']} -> ${best_knn['test_revenue']:.2f}")
    
    best_rf = find_best_config(rf_results)
    if best_rf:
        print(f"Best Random Forest: {best_rf['config']} -> ${best_rf['test_revenue']:.2f}")
    
    # Summary table
    all_successful = []
    for results in [revenue_results, reg_results, knn_results, rf_results]:
        all_successful.extend([r for r in results if r.get("success", False)])
    
    if all_successful:
        results_df = pd.DataFrame(all_successful)
        print(f"\nComprehensive Ablation Results:")
        print("-" * 60)
        print(results_df[["config", "test_revenue", "test_precision_at_k", "test_ndcg_at_k"]].round(3).to_string(index=False))


def main():
    """Run ablation study."""
    try:
        print("CS145 Checkpoint 1: Ablation Study")
        print("=" * 40)
        print("Hyperparameter impact evaluation")
        
        # Run ablation studies
        revenue_results = revenue_weight_ablation()
        reg_results = regularization_strength_ablation()
        knn_results = knn_neighbors_ablation()
        rf_results = random_forest_trees_ablation()
        
        # Plot and analyze results
        plot_ablation_results(revenue_results, reg_results, knn_results, rf_results)
        analyze_ablation_results(revenue_results, reg_results, knn_results, rf_results)
        
        print("\nAblation study complete")
        
    except Exception as e:
        print(f"Ablation study failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        spark.stop()


if __name__ == "__main__":
    main() 