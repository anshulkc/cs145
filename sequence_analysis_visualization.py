import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import shutil
import time
import copy

from pyspark.sql import SparkSession
from pyspark.sql import functions as sf
from pyspark.sql import DataFrame, Window

# Set up plotting
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (14, 8)

# Initialize Spark session
spark = SparkSession.builder \
    .appName("SequenceRecSysVisualization") \
    .master("local[*]") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", "8") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Import competition modules
from data_generator import CompetitionDataGenerator
from simulator import CompetitionSimulator
from sample_recommenders import (
    RandomRecommender,
    PopularityRecommender,
    ContentBasedRecommender,
    SVMRecommender,
)
from config import DEFAULT_CONFIG, EVALUATION_METRICS

# Import sequence-based recommenders
from sequence_based_recommenders.transformer import TransformerRecommender
# Add other sequence recommenders as you implement them
# from sequence_based_recommenders.rnn import RNNRecommender
# from sequence_based_recommenders.lstm import LSTMRecommender


def run_sequence_recommender_analysis():
    """
    Run analysis specifically for sequence-based recommenders.
    """
    print("=== Sequence-Based Recommender Analysis ===")
    
    # Create dataset configuration
    config = DEFAULT_CONFIG.copy()
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
    
    # Initialize sequence-based recommenders
    sequence_recommenders = [
        TransformerRecommender(
            embed_dim=64,
            num_heads=4,
            num_layers=2,
            max_seq_len=50,
            dropout=0.1,
            lr=1e-3,
            n_epochs=3,
            seed=42
        ),
        # Add other sequence recommenders here as you implement them
        # RNNRecommender(hidden_size=64, num_layers=2, dropout=0.1, seed=42),
        # LSTMRecommender(hidden_size=128, num_layers=2, dropout=0.2, seed=42),
    ]
    
    sequence_names = [
        "Transformer",
        # "RNN",
        # "LSTM",
    ]
    
    # Add baseline recommenders for comparison
    baseline_recommenders = [
        RandomRecommender(seed=42),
        PopularityRecommender(alpha=1.0, seed=42),
    ]
    baseline_names = ["Random", "Popularity"]
    
    # Combine all recommenders
    all_recommenders = sequence_recommenders + baseline_recommenders
    all_names = sequence_names + baseline_names
    
    print(f"\nEvaluating {len(all_recommenders)} recommenders:")
    for i, name in enumerate(all_names, 1):
        print(f"  {i}. {name}")
    
    # Initialize recommenders with initial history
    for recommender in all_recommenders:
        recommender.fit(log=data_generator.history_df,
                        user_features=users_df,
                        item_features=items_df)
    
    # Evaluate each recommender
    results = []
    
    for name, recommender in zip(all_names, all_recommenders):
        print(f"\nEvaluating {name}:")
        start_time = time.time()
        
        try:
            # Clean up any existing simulator data directory
            simulator_data_dir = f"simulator_sequence_data_{name}"
            if os.path.exists(simulator_data_dir):
                shutil.rmtree(simulator_data_dir)
            
            # Initialize simulator
            simulator = CompetitionSimulator(
                user_generator=user_generator,
                item_generator=item_generator,
                data_dir=simulator_data_dir,
                log_df=data_generator.history_df,
                conversion_noise_mean=config['simulation']['conversion_noise_mean'],
                conversion_noise_std=config['simulation']['conversion_noise_std'],
                spark_session=spark,
                seed=config['data_generation']['seed']
            )
            
            # Run simulation with train-test split
            train_metrics, test_metrics, train_revenue, test_revenue = simulator.train_test_split(
                recommender=recommender,
                train_iterations=train_iterations,
                test_iterations=test_iterations,
                user_frac=config['simulation']['user_fraction'],
                k=config['simulation']['k'],
                filter_seen_items=config['simulation']['filter_seen_items'],
                retrain=config['simulation']['retrain']
            )
            
            # Calculate average metrics
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
                "type": "sequence" if name in sequence_names else "baseline",
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
            
            # Print summary for this recommender
            print(f"  Training Phase - Total Revenue: {sum(train_revenue):.2f}")
            print(f"  Testing Phase - Total Revenue: {sum(test_revenue):.2f}")
            print(f"  Performance Change: {performance_change:.2f}%")
            print(f"  Evaluation Time: {total_time:.2f}s")
            
        except Exception as e:
            print(f"  Error evaluating {name}: {str(e)}")
            continue
    
    # Convert to DataFrame for analysis
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_total_revenue", ascending=False).reset_index(drop=True)
    
    # Print summary table
    print("\n" + "="*80)
    print("SEQUENCE RECOMMENDER EVALUATION RESULTS")
    print("="*80)
    
    summary_cols = ["name", "type", "train_total_revenue", "test_total_revenue",
                   "performance_change", "total_time"]
    
    # Add metric columns that exist
    metric_cols = ["train_precision_at_k", "test_precision_at_k",
                   "train_ndcg_at_k", "test_ndcg_at_k",
                   "train_mrr", "test_mrr",
                   "train_discounted_revenue", "test_discounted_revenue"]
    
    existing_metric_cols = [col for col in metric_cols if col in results_df.columns]
    summary_cols.extend(existing_metric_cols)
    
    print(results_df[summary_cols].to_string(index=False))
    
    # Generate visualizations
    visualize_sequence_performance(results_df)
    
    return results_df


def visualize_sequence_performance(results_df):
    """
    Create visualizations specifically for sequence-based recommender performance.
    """
    plt.figure(figsize=(20, 12))
    
    # Separate sequence and baseline results
    sequence_results = results_df[results_df['type'] == 'sequence']
    baseline_results = results_df[results_df['type'] == 'baseline']
    
    # 1. Revenue Comparison
    plt.subplot(2, 3, 1)
    x = np.arange(len(results_df))
    width = 0.35
    
    colors = ['#2E8B57' if t == 'sequence' else '#CD5C5C' for t in results_df['type']]
    
    plt.bar(x - width/2, results_df['train_total_revenue'], width, 
            label='Training', alpha=0.7, color=colors)
    plt.bar(x + width/2, results_df['test_total_revenue'], width, 
            label='Testing', alpha=0.9, color=colors)
    
    plt.xlabel('Recommender')
    plt.ylabel('Total Revenue')
    plt.title('Revenue Comparison: Sequence vs Baseline')
    plt.xticks(x, results_df['name'], rotation=45)
    plt.legend()
    
    # 2. Performance Change
    plt.subplot(2, 3, 2)
    plt.bar(x, results_df['performance_change'], color=colors, alpha=0.7)
    plt.xlabel('Recommender')
    plt.ylabel('Performance Change (%)')
    plt.title('Train-to-Test Performance Change')
    plt.xticks(x, results_df['name'], rotation=45)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # 3. Evaluation Time
    plt.subplot(2, 3, 3)
    plt.bar(x, results_df['total_time'], color=colors, alpha=0.7)
    plt.xlabel('Recommender')
    plt.ylabel('Time (seconds)')
    plt.title('Evaluation Time Comparison')
    plt.xticks(x, results_df['name'], rotation=45)
    
    # 4. Revenue Trajectories
    plt.subplot(2, 3, 4)
    colors_dict = {'sequence': '#2E8B57', 'baseline': '#CD5C5C'}
    markers = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h']
    
    for i, (_, row) in enumerate(results_df.iterrows()):
        name = row['name']
        model_type = row['type']
        
        train_revenue = row['train_revenue']
        test_revenue = row['test_revenue']
        
        # Ensure revenues are lists
        if isinstance(train_revenue, (float, np.float64, int)):
            train_revenue = [train_revenue]
        if isinstance(test_revenue, (float, np.float64, int)):
            test_revenue = [test_revenue]
        
        # Plot trajectory
        iterations = list(range(len(train_revenue))) + \
                    list(range(len(train_revenue), len(train_revenue) + len(test_revenue)))
        revenues = list(train_revenue) + list(test_revenue)
        
        plt.plot(iterations, revenues, 
                marker=markers[i % len(markers)],
                color=colors_dict[model_type],
                linewidth=2 if model_type == 'sequence' else 1,
                alpha=0.8,
                label=name)
        
        # Add train/test separator
        if i == 0:
            plt.axvline(x=len(train_revenue)-0.5, color='k', 
                       linestyle='--', alpha=0.3, label='Train/Test Split')
    
    plt.xlabel('Iteration')
    plt.ylabel('Revenue')
    plt.title('Revenue Learning Curves')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 5. Ranking Metrics Comparison
    plt.subplot(2, 3, 5)
    ranking_metrics = ['precision_at_k', 'ndcg_at_k', 'mrr']
    test_metrics = [f'test_{m}' for m in ranking_metrics if f'test_{m}' in results_df.columns]
    
    if test_metrics:
        bar_positions = np.arange(len(test_metrics))
        bar_width = 0.8 / len(results_df)
        
        for i, (_, row) in enumerate(results_df.iterrows()):
            model_name = row['name']
            model_type = row['type']
            offsets = (i - len(results_df)/2 + 0.5) * bar_width
            metric_values = [row[m] for m in test_metrics]
            
            plt.bar(bar_positions + offsets, metric_values, bar_width, 
                   label=model_name, color=colors_dict[model_type], alpha=0.7)
        
        plt.xlabel('Metric')
        plt.ylabel('Value')
        plt.title('Test Phase Ranking Metrics')
        plt.xticks(bar_positions, [m.replace('test_', '').replace('_', ' ').title() 
                                  for m in test_metrics])
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # 6. Sequence Model Comparison (if multiple sequence models)
    plt.subplot(2, 3, 6)
    if len(sequence_results) > 1:
        seq_names = sequence_results['name'].tolist()
        seq_revenues = sequence_results['test_total_revenue'].tolist()
        
        plt.bar(range(len(seq_names)), seq_revenues, 
               color='#2E8B57', alpha=0.7)
        plt.xlabel('Sequence Model')
        plt.ylabel('Test Total Revenue')
        plt.title('Sequence Model Performance')
        plt.xticks(range(len(seq_names)), seq_names, rotation=45)
        
        # Add values on bars
        for i, v in enumerate(seq_revenues):
            plt.text(i, v + max(seq_revenues)*0.01, f'{v:.1f}', 
                    ha='center', va='bottom')
    else:
        # Show best sequence vs best baseline
        if len(sequence_results) > 0 and len(baseline_results) > 0:
            best_seq = sequence_results.iloc[0]
            best_baseline = baseline_results.iloc[0]
            
            names = [f"{best_seq['name']}\n(Sequence)", f"{best_baseline['name']}\n(Baseline)"]
            revenues = [best_seq['test_total_revenue'], best_baseline['test_total_revenue']]
            colors_comp = ['#2E8B57', '#CD5C5C']
            
            bars = plt.bar(names, revenues, color=colors_comp, alpha=0.7)
            plt.ylabel('Test Total Revenue')
            plt.title('Best Sequence vs Best Baseline')
            
            # Add values on bars
            for bar, value in zip(bars, revenues):
                plt.text(bar.get_x() + bar.get_width()/2, value + max(revenues)*0.01,
                        f'{value:.1f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('sequence_recommender_analysis.png', dpi=300, bbox_inches='tight')
    print(f"\nSequence recommender analysis saved to 'sequence_recommender_analysis.png'")


if __name__ == "__main__":
    print("Starting Sequence-Based Recommender Analysis...")
    results = run_sequence_recommender_analysis()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    
    # Print key findings
    if len(results) > 0:
        best_overall = results.iloc[0]
        sequence_results = results[results['type'] == 'sequence']
        
        print(f"\nBest Overall Performer: {best_overall['name']} ({best_overall['type']})")
        print(f"  Test Revenue: {best_overall['test_total_revenue']:.2f}")
        
        if len(sequence_results) > 0:
            best_sequence = sequence_results.iloc[0]
            print(f"\nBest Sequence Model: {best_sequence['name']}")
            print(f"  Test Revenue: {best_sequence['test_total_revenue']:.2f}")
            print(f"  Performance Change: {best_sequence['performance_change']:.2f}%") 