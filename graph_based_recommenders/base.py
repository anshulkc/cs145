# Base class for graph-based recommenders

import numpy as np
import pandas as pd
import networkx as nx
from typing import Optional, List, Dict, Tuple, Set
from abc import ABC, abstractmethod
from collections import defaultdict

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.sql.types import LongType
from sim4rec.utils import pandas_to_spark
import warnings
warnings.filterwarnings('ignore')


class GraphBasedRecommenderBase(ABC):
    """
    Base class for graph-based recommenders.
    Provides common functionality for graph construction, training, and prediction.
    """
    
    def __init__(self, seed=None, revenue_weight=1.0, edge_weight_strategy='frequency',
                 temporal_decay=0.95, min_interactions=1):
        """
        Args:
            seed: Random seed for reproducibility
            revenue_weight: Weight for revenue optimization in ranking
            edge_weight_strategy: Strategy for edge weights ('frequency', 'recency', 'purchase_amount')
            temporal_decay: Decay factor for older interactions (1.0 = no decay)
            min_interactions: Minimum interactions for including user/item in graph
        """
        self.seed = seed
        self.revenue_weight = revenue_weight
        self.edge_weight_strategy = edge_weight_strategy
        self.temporal_decay = temporal_decay
        self.min_interactions = min_interactions
        self.is_fitted = False
        
        # Graph data structures
        self.graph = None  # NetworkX bipartite graph
        self.user_nodes = set()  # Set of user node IDs
        self.item_nodes = set()  # Set of item node IDs
        self.user_vocab = {}  # user_idx -> node_id mapping
        self.item_vocab = {}  # item_idx -> node_id mapping
        self.reverse_user_vocab = {}  # node_id -> user_idx mapping
        self.reverse_item_vocab = {}  # node_id -> item_idx mapping
        self.n_users = 0
        self.n_items = 0
        
        # Interaction data
        self.interactions_df = None  # Processed interaction DataFrame
        
        if seed is not None:
            np.random.seed(seed)
    
    def _build_bipartite_graph(self, log: DataFrame, user_features: Optional[DataFrame] = None,
                              item_features: Optional[DataFrame] = None) -> None:
        """
        Build user-item bipartite graph from interaction log.
        
        Args:
            log: Interaction log with user_idx, item_idx, relevance columns
            user_features: User features (optional)
            item_features: Item features (optional)
        """
        if log is None or log.count() == 0:
            print("Warning: No interaction data available for graph construction")
            return
        
        # Add artificial timestamps if not present
        if '__iter' not in log.columns:
            log = log.withColumn('__iter', sf.lit(0))
        
        # Create timestamp based on iteration and relevance
        log = log.withColumn(
            'timestamp', 
            sf.col('__iter') * 1000000 + sf.col('relevance') * 1000
        )
        
        # Join with item features to get prices
        if item_features is not None:
            log_with_features = log.join(
                item_features.select('item_idx', 'price'), 
                on='item_idx', 
                how='left'
            )
        else:
            log_with_features = log.withColumn('price', sf.lit(1.0))
            
        # Add response column if not present
        if 'response' not in log_with_features.columns:
            log_with_features = log_with_features.withColumn('response', sf.col('relevance'))
        
        # Convert to pandas for graph processing
        self.interactions_df = log_with_features.select(
            'user_idx', 'item_idx', 'timestamp', 'price', 'response', 'relevance'
        ).toPandas()
        
        # Filter users and items by minimum interactions
        user_counts = self.interactions_df['user_idx'].value_counts()
        item_counts = self.interactions_df['item_idx'].value_counts()
        
        valid_users = set(user_counts[user_counts >= self.min_interactions].index)
        valid_items = set(item_counts[item_counts >= self.min_interactions].index)
        
        self.interactions_df = self.interactions_df[
            (self.interactions_df['user_idx'].isin(valid_users)) &
            (self.interactions_df['item_idx'].isin(valid_items))
        ]
        
        # Build vocabularies
        unique_users = sorted(self.interactions_df['user_idx'].unique())
        unique_items = sorted(self.interactions_df['item_idx'].unique())
        
        # Create node IDs: users get IDs 0 to n_users-1, items get IDs n_users to n_users+n_items-1
        self.n_users = len(unique_users)
        self.n_items = len(unique_items)
        
        for i, user_idx in enumerate(unique_users):
            node_id = i
            self.user_vocab[user_idx] = node_id
            self.reverse_user_vocab[node_id] = user_idx
            self.user_nodes.add(node_id)
        
        for i, item_idx in enumerate(unique_items):
            node_id = self.n_users + i
            self.item_vocab[item_idx] = node_id
            self.reverse_item_vocab[node_id] = item_idx
            self.item_nodes.add(node_id)
        
        print(f"Graph vocabulary: {self.n_users} users, {self.n_items} items")
        
        # Create bipartite graph
        self.graph = nx.Graph()
        
        # Add nodes with bipartite attribute
        for user_node in self.user_nodes:
            self.graph.add_node(user_node, bipartite=0)  # Users are partition 0
        for item_node in self.item_nodes:
            self.graph.add_node(item_node, bipartite=1)  # Items are partition 1
        
        # Calculate edge weights based on strategy
        edge_weights = self._calculate_edge_weights()
        
        # Add edges with weights
        for (user_idx, item_idx), weight in edge_weights.items():
            if user_idx in self.user_vocab and item_idx in self.item_vocab:
                user_node = self.user_vocab[user_idx]
                item_node = self.item_vocab[item_idx]
                self.graph.add_edge(user_node, item_node, weight=weight)
        
        print(f"Graph constructed: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def _calculate_edge_weights(self) -> Dict[Tuple[int, int], float]:
        """
        Calculate edge weights based on the specified strategy.
        
        Returns:
            Dictionary mapping (user_idx, item_idx) to edge weight
        """
        edge_weights = defaultdict(float)
        
        for _, row in self.interactions_df.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            key = (user_idx, item_idx)
            
            if self.edge_weight_strategy == 'frequency':
                # Weight by frequency of interactions
                edge_weights[key] += 1.0
                
            elif self.edge_weight_strategy == 'recency':
                # Weight by recency with temporal decay
                timestamp = row['timestamp']
                max_timestamp = self.interactions_df['timestamp'].max()
                recency_weight = self.temporal_decay ** ((max_timestamp - timestamp) / 1000000)
                edge_weights[key] += recency_weight
                
            elif self.edge_weight_strategy == 'purchase_amount':
                # Weight by purchase amount (response * price)
                purchase_weight = row['response'] * row['price']
                edge_weights[key] += purchase_weight
                
            else:  # Default to frequency
                edge_weights[key] += 1.0
        
        return dict(edge_weights)
    
    def _get_user_item_pairs_for_prediction(self, users: DataFrame, items: DataFrame,
                                          filter_seen_items: bool = True) -> List[Tuple[int, int]]:
        """
        Get user-item pairs for prediction.
        
        Args:
            users: User DataFrame
            items: Item DataFrame
            filter_seen_items: Whether to filter already seen items
            
        Returns:
            List of (user_idx, item_idx) tuples
        """
        user_item_pairs = []
        
        # Get user and item lists
        user_list = [row['user_idx'] for row in users.collect()]
        item_list = [row['item_idx'] for row in items.collect()]
        
        # Create seen items set if filtering
        seen_items = set()
        if filter_seen_items and self.interactions_df is not None:
            for _, row in self.interactions_df.iterrows():
                seen_items.add((int(row['user_idx']), int(row['item_idx'])))
        
        # Generate all user-item pairs
        for user_idx in user_list:
            for item_idx in item_list:
                pair = (user_idx, item_idx)
                if not filter_seen_items or pair not in seen_items:
                    # Only include pairs where both user and item are in graph
                    if user_idx in self.user_vocab and item_idx in self.item_vocab:
                        user_item_pairs.append(pair)
        
        return user_item_pairs
    
    def _apply_revenue_weighting(self, recommendations_df: DataFrame, items_df: DataFrame) -> DataFrame:
        """
        Apply revenue weighting to recommendations.
        
        Args:
            recommendations_df: DataFrame with user_idx, item_idx, relevance
            items_df: DataFrame with item_idx, price
            
        Returns:
            DataFrame with revenue-weighted relevance scores
        """
        # Join with item features to get prices
        recommendations_with_price = recommendations_df.join(
            items_df.select('item_idx', 'price'),
            on='item_idx',
            how='left'
        )
        
        # Apply revenue weighting: relevance = probability^(1/revenue_weight) * price^revenue_weight
        revenue_weighted = recommendations_with_price.withColumn(
            'relevance',
            sf.pow(sf.col('relevance'), 1.0 / self.revenue_weight) * 
            sf.pow(sf.col('price'), self.revenue_weight)
        )
        
        return revenue_weighted.drop('price')
    
    def _apply_position_bias(self, recommendations_df: DataFrame) -> DataFrame:
        """
        Apply position bias to ranking (discounted cumulative gain).
        
        Args:
            recommendations_df: DataFrame with user_idx, item_idx, relevance
            
        Returns:
            DataFrame with position-bias adjusted relevance scores
        """
        # Rank items by relevance for each user
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        ranked_recs = recommendations_df.withColumn("rank", sf.row_number().over(window))
        
        # Apply position bias: relevance = relevance / log2(rank + 1)
        position_biased = ranked_recs.withColumn(
            "relevance",
            sf.col("relevance") / sf.log2(sf.col("rank") + 1)
        )
        
        return position_biased.drop("rank")
    
    def _random_recommendations(self, users: DataFrame, items: DataFrame, k: int) -> DataFrame:
        """Generate random recommendations as fallback."""
        # Cross join users and items
        recs = users.crossJoin(items)
        
        # Add random relevance scores
        recs = recs.withColumn("relevance", sf.rand(seed=self.seed))
        
        # Rank items by relevance for each user
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs = recs.withColumn("rank", sf.row_number().over(window))
        
        # Filter top-k recommendations
        recs = recs.filter(sf.col("rank") <= k).drop("rank")
        
        return recs
    
    @abstractmethod
    def _train_model(self) -> None:
        """Train the graph-based model."""
        pass
    
    @abstractmethod
    def _predict_link_probabilities(self, user_item_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """
        Predict link probabilities for user-item pairs.
        
        Args:
            user_item_pairs: List of (user_idx, item_idx) tuples
            
        Returns:
            Array of link probabilities
        """
        pass
    
    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None) -> None:
        """Train the graph-based recommender model."""
        print("Building bipartite graph from interaction log...")
        self._build_bipartite_graph(log, user_features, item_features)
        
        if self.graph is None or self.graph.number_of_edges() == 0:
            print("Warning: No valid graph constructed")
            return
        
        print("Training graph model...")
        self._train_model()
        self.is_fitted = True
        print(f"Graph model trained on {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        """Generate recommendations for users."""
        if not self.is_fitted or self.graph is None:
            print("Warning: Model not fitted or no graph available. Using random recommendations.")
            return self._random_recommendations(users, items, k)
        
        # Get user-item pairs for prediction
        user_item_pairs = self._get_user_item_pairs_for_prediction(users, items, filter_seen_items)
        
        if len(user_item_pairs) == 0:
            print("Warning: No user-item pairs for prediction")
            return self._random_recommendations(users, items, k)
        
        # Predict link probabilities
        probabilities = self._predict_link_probabilities(user_item_pairs)
        
        # Create recommendations DataFrame
        recommendations_data = []
        for i, (user_idx, item_idx) in enumerate(user_item_pairs):
            recommendations_data.append({
                'user_idx': user_idx,
                'item_idx': item_idx,
                'relevance': float(probabilities[i])
            })
        
        recommendations_df = pandas_to_spark(pd.DataFrame(recommendations_data))
        
        # Cast to proper types
        recommendations_df = recommendations_df.withColumn("user_idx", sf.col("user_idx").cast(LongType()))
        recommendations_df = recommendations_df.withColumn("item_idx", sf.col("item_idx").cast(LongType()))
        
        # Apply revenue weighting if item features available
        if item_features is not None:
            recommendations_df = self._apply_revenue_weighting(recommendations_df, item_features)
        
        # Apply position bias
        recommendations_df = self._apply_position_bias(recommendations_df)
        
        # Rank and select top-k for each user
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        ranked_recs = recommendations_df.withColumn("rank", sf.row_number().over(window))
        final_recs = ranked_recs.filter(sf.col("rank") <= k).drop("rank")
        
        return final_recs 