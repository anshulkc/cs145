# Graph Convolutional Network (GCN) implementation for graph-based recommendations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
import networkx as nx
from collections import defaultdict
import random

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
        
        # Generate all user-item pairs (including cold start)
        for user_idx in user_list:
            for item_idx in item_list:
                pair = (user_idx, item_idx)
                if not filter_seen_items or pair not in seen_items:
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
        
        # Apply revenue weighting: relevance = probability * price^revenue_weight
        revenue_weighted = recommendations_with_price.withColumn(
            'relevance',
            sf.col('relevance') * sf.pow(sf.col('price'), self.revenue_weight)
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


class GCNLayer(nn.Module):
    """Single Graph Convolutional Layer."""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.5, use_batch_norm: bool = True):
        super(GCNLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Linear transformation
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout_layer = nn.Dropout(dropout)
        
        # Batch normalization
        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_dim)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
    
    def forward(self, features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of GCN layer.
        
        Args:
            features: Node features [n_nodes, input_dim]
            adjacency_matrix: Normalized adjacency matrix [n_nodes, n_nodes]
            
        Returns:
            Updated node features [n_nodes, output_dim]
        """
        # Apply linear transformation
        transformed = self.linear(features)
        
        # Graph convolution: A * X * W
        output = torch.sparse.mm(adjacency_matrix, transformed)
        
        # Apply batch normalization if enabled
        if self.use_batch_norm:
            output = self.batch_norm(output)
        
        # Apply ReLU activation and dropout
        output = F.relu(output)
        output = self.dropout_layer(output)
        
        return output


class GCNModel(nn.Module):
    """
    Graph Convolutional Network for user-item interaction prediction.
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int, hidden_dims: List[int] = [256, 128],
                 dropout: float = 0.5, use_batch_norm: bool = True):
        super(GCNModel, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Initial node embeddings
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        
        # Build GCN layers
        self.gcn_layers = nn.ModuleList()
        layer_dims = [embedding_dim] + hidden_dims
        
        for i in range(len(layer_dims) - 1):
            self.gcn_layers.append(GCNLayer(layer_dims[i], layer_dims[i + 1], dropout, use_batch_norm))
        
        # Final prediction layer
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1] * 2, hidden_dims[-1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[-1], 1)
        )
        
        # Store adjacency matrix
        self.adjacency_matrix = None
    
    def set_adjacency_matrix(self, adj_matrix: torch.Tensor):
        """Set the normalized adjacency matrix for graph convolution."""
        self.adjacency_matrix = adj_matrix
    
    def forward(self, user_indices=None, item_indices=None, compute_all=False):
        """
        Forward pass through GCN layers.
        
        Args:
            user_indices: User indices for prediction
            item_indices: Item indices for prediction
            compute_all: Whether to compute embeddings for all nodes
            
        Returns:
            User and item embeddings or predictions
        """
        if self.adjacency_matrix is None:
            raise ValueError("Adjacency matrix not set")
        
        # Get initial embeddings
        user_emb_0 = self.user_embeddings.weight
        item_emb_0 = self.item_embeddings.weight
        
        # Concatenate user and item embeddings
        all_features = torch.cat([user_emb_0, item_emb_0], dim=0)
        
        # Pass through GCN layers
        for gcn_layer in self.gcn_layers:
            all_features = gcn_layer(all_features, self.adjacency_matrix)
        
        # Split back to user and item embeddings
        user_embeddings = all_features[:self.n_users]
        item_embeddings = all_features[self.n_users:]
        
        if compute_all:
            return user_embeddings, item_embeddings
        
        if user_indices is not None and item_indices is not None:
            user_emb = user_embeddings[user_indices]
            item_emb = item_embeddings[item_indices]
            
            # Concatenate user and item embeddings for prediction
            combined = torch.cat([user_emb, item_emb], dim=-1)
            predictions = self.predictor(combined).squeeze(-1)
            
            return predictions
        
        return user_embeddings, item_embeddings


class GCNRecommender(GraphBasedRecommenderBase):
    """
    GCN-based graph recommender for collaborative filtering.
    Implements Graph Convolutional Networks for user-item interaction prediction.
    """
    
    def __init__(self, embedding_dim=128, hidden_dims=[256, 128], dropout=0.3, learning_rate=0.01,
                 epochs=100, batch_size=512, reg_weight=1e-5, negative_sampling_ratio=2.0, 
                 early_stopping_patience=15, use_batch_norm=True, graph_dropout=0.05, **kwargs):
        """
        Args:
            embedding_dim: Dimension of initial node embeddings
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate for regularization
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs  
            batch_size: Training batch size
            reg_weight: L2 regularization weight
            negative_sampling_ratio: Ratio of negative samples to positive samples
            early_stopping_patience: Patience for early stopping
            use_batch_norm: Whether to use batch normalization
            graph_dropout: Graph dropout rate (randomly remove edges)
        """
        # Set default revenue weight for graphs (lower than content-based)
        if 'revenue_weight' not in kwargs:
            kwargs['revenue_weight'] = 0.1
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.negative_sampling_ratio = negative_sampling_ratio
        self.early_stopping_patience = early_stopping_patience
        self.use_batch_norm = use_batch_norm
        self.graph_dropout = graph_dropout
        
        # Model components
        self.model = None
        self.adjacency_matrix = None
        self.positive_interactions = []
    
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """
        Build normalized adjacency matrix for GCN.
        
        Returns:
            Normalized adjacency matrix
        """
        if self.graph is None:
            raise ValueError("Graph not constructed")
        
        total_nodes = self.n_users + self.n_items
        
        # Build adjacency matrix indices and values
        row_indices = []
        col_indices = []
        values = []
        
        # Add self-loops (identity matrix)
        for i in range(total_nodes):
            row_indices.append(i)
            col_indices.append(i)
            values.append(1.0)
        
        # Add user-item edges (bidirectional)
        for user_node in self.user_nodes:
            for item_node in self.item_nodes:
                if self.graph.has_edge(user_node, item_node):
                    weight = self.graph[user_node][item_node].get('weight', 1.0)
                    
                    # User -> Item
                    row_indices.append(user_node)
                    col_indices.append(item_node)
                    values.append(weight)
                    
                    # Item -> User (symmetric)
                    row_indices.append(item_node)
                    col_indices.append(user_node)
                    values.append(weight)
        
        # Create sparse adjacency matrix
        indices = torch.LongTensor([row_indices, col_indices])
        values = torch.FloatTensor(values)
        adj_matrix = torch.sparse.FloatTensor(indices, values, (total_nodes, total_nodes))
        
        # Normalize adjacency matrix: D^(-1/2) * (A + I) * D^(-1/2)
        degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        degrees = torch.pow(degrees + 1e-8, -0.5)
        
        # Create degree matrix
        degree_indices = torch.arange(total_nodes).unsqueeze(0).repeat(2, 1)
        degree_values = degrees
        degree_matrix = torch.sparse.FloatTensor(degree_indices, degree_values, (total_nodes, total_nodes))
        
        # Normalize: D^(-1/2) * A * D^(-1/2)
        normalized_adj = torch.sparse.mm(torch.sparse.mm(degree_matrix, adj_matrix), degree_matrix)
        
        return normalized_adj
    
    def _prepare_training_data(self) -> List[Tuple[int, int, float]]:
        """Prepare positive and negative training samples."""
        training_data = []
        
        # Positive samples from interactions
        for _, row in self.interactions_df.iterrows():
            user_idx = int(row['user_idx'])
            item_idx = int(row['item_idx'])
            
            if user_idx in self.user_vocab and item_idx in self.item_vocab:
                user_node = self.user_vocab[user_idx]
                item_node = self.item_vocab[item_idx]
                
                # Convert to internal indices
                user_internal = user_node
                item_internal = item_node - self.n_users
                
                training_data.append((user_internal, item_internal, 1.0))
                self.positive_interactions.append((user_internal, item_internal))
        
        # Negative sampling
        positive_set = set(self.positive_interactions)
        num_negatives = int(len(self.positive_interactions) * self.negative_sampling_ratio)
        
        negatives_added = 0
        while negatives_added < num_negatives:
            user_internal = random.randint(0, self.n_users - 1)
            item_internal = random.randint(0, self.n_items - 1)
            
            if (user_internal, item_internal) not in positive_set:
                training_data.append((user_internal, item_internal, 0.0))
                negatives_added += 1
        
        print(f"Training data prepared: {len(self.positive_interactions)} positive, {num_negatives} negative samples")
        return training_data
    
    def _apply_graph_dropout(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        """Apply graph dropout by randomly removing edges."""
        if self.graph_dropout <= 0:
            return adj_matrix
        
        # Get indices and values from sparse matrix
        indices = adj_matrix._indices()
        values = adj_matrix._values()
        
        # Create dropout mask
        dropout_mask = torch.rand(values.size(0)) > self.graph_dropout
        
        # Apply mask
        kept_indices = indices[:, dropout_mask]
        kept_values = values[dropout_mask]
        
        # Create new sparse matrix
        dropped_adj = torch.sparse.FloatTensor(kept_indices, kept_values, adj_matrix.size())
        
        return dropped_adj
    
    def _train_model(self) -> None:
        """Train GCN model using binary cross-entropy loss."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("Warning: No graph available for training")
            return
        
        # Build adjacency matrix
        print("Building normalized adjacency matrix...")
        self.adjacency_matrix = self._build_adjacency_matrix()
        
        # Initialize model
        self.model = GCNModel(self.n_users, self.n_items, self.embedding_dim, 
                             self.hidden_dims, self.dropout, self.use_batch_norm)
        self.model.set_adjacency_matrix(self.adjacency_matrix)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.reg_weight)
        criterion = nn.BCEWithLogitsLoss()
        
        # Prepare training data
        training_data = self._prepare_training_data()
        
        print(f"Training GCN model for {self.epochs} epochs...")
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            num_batches = 0
            
            # Apply graph dropout by creating modified adjacency matrix
            if self.graph_dropout > 0:
                adj_matrix = self._apply_graph_dropout(self.adjacency_matrix)
                self.model.set_adjacency_matrix(adj_matrix)
            
            # Shuffle training data
            random.shuffle(training_data)
            
            # Mini-batch training
            for i in range(0, len(training_data), self.batch_size):
                batch_data = training_data[i:i + self.batch_size]
                if len(batch_data) == 0:
                    continue
                
                # Prepare batch tensors
                user_indices = torch.tensor([item[0] for item in batch_data], dtype=torch.long)
                item_indices = torch.tensor([item[1] for item in batch_data], dtype=torch.long)
                labels = torch.tensor([item[2] for item in batch_data], dtype=torch.float)
                
                # Forward pass
                optimizer.zero_grad()
                predictions = self.model(user_indices, item_indices)
                
                # Compute loss
                loss = criterion(predictions, labels)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            # Calculate average loss for this epoch
            avg_loss = total_loss / max(num_batches, 1)
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= self.early_stopping_patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {avg_loss:.4f}")
        
        print("GCN training completed")
    
    def _predict_link_probabilities(self, user_item_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """Predict link probabilities using GCN."""
        if self.model is None:
            print("Warning: No model available")
            return np.random.random(len(user_item_pairs))
        
        self.model.eval()
        probabilities = []
        
        # Cold start fallback score (lower than average training set)
        cold_start_score = 0.2  # Conservative score for cold start users/items
        
        # Process in batches for efficiency
        batch_size = 1000
        for i in range(0, len(user_item_pairs), batch_size):
            batch_pairs = user_item_pairs[i:i + batch_size]
            batch_user_indices = []
            batch_item_indices = []
            valid_mask = []
            
            for user_idx, item_idx in batch_pairs:
                if user_idx in self.user_vocab and item_idx in self.item_vocab:
                    user_node = self.user_vocab[user_idx]
                    item_node = self.item_vocab[item_idx]
                    
                    user_internal = user_node
                    item_internal = item_node - self.n_users
                    
                    if user_internal < self.n_users and item_internal < self.n_items:
                        batch_user_indices.append(user_internal)
                        batch_item_indices.append(item_internal)
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                else:
                    valid_mask.append(False)
            
            if len(batch_user_indices) > 0:
                with torch.no_grad():
                    user_tensor = torch.tensor(batch_user_indices, dtype=torch.long)
                    item_tensor = torch.tensor(batch_item_indices, dtype=torch.long)
                    batch_predictions = self.model(user_tensor, item_tensor)
                    batch_probabilities = torch.sigmoid(batch_predictions).cpu().numpy()
                
                # Map back to original order
                pred_idx = 0
                for is_valid in valid_mask:
                    if is_valid:
                        probabilities.append(batch_probabilities[pred_idx])
                        pred_idx += 1
                    else:
                        # Cold start case: use reasonable default score
                        probabilities.append(cold_start_score)
            else:
                # All cold start in this batch
                probabilities.extend([cold_start_score] * len(batch_pairs))
        
        return np.array(probabilities) 