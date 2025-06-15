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

from .base import GraphBasedRecommenderBase


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
    
    def __init__(self, embedding_dim=128, hidden_dims=[256, 128], dropout=0.5, learning_rate=0.001,
                 epochs=100, batch_size=1024, reg_weight=1e-4, negative_sampling_ratio=1.0, 
                 early_stopping_patience=10, use_batch_norm=True, graph_dropout=0.1, **kwargs):
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
                        probabilities.append(0.0)
            else:
                probabilities.extend([0.0] * len(batch_pairs))
        
        return np.array(probabilities) 