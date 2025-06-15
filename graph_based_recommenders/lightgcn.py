# LightGCN implementation for graph-based recommendations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import networkx as nx
from collections import defaultdict
import random

from .base import GraphBasedRecommenderBase


class LightGCNModel(nn.Module):
    """
    LightGCN model for collaborative filtering.
    Simplified GCN that removes feature transformation and nonlinear activation.
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int, n_layers: int = 3):
        super(LightGCNModel, self).__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Initialize user and item embeddings
        self.user_embeddings = nn.Embedding(n_users, embedding_dim)
        self.item_embeddings = nn.Embedding(n_items, embedding_dim)
        
        # Initialize embeddings with Xavier uniform
        nn.init.xavier_uniform_(self.user_embeddings.weight)
        nn.init.xavier_uniform_(self.item_embeddings.weight)
        
        # Store adjacency matrix info
        self.adjacency_matrix = None
        self.degree_matrix = None
    
    def set_adjacency_matrix(self, adj_matrix: torch.Tensor, degree_matrix: torch.Tensor):
        """Set the normalized adjacency matrix for graph convolution."""
        self.adjacency_matrix = adj_matrix
        self.degree_matrix = degree_matrix
    
    def forward(self, user_indices=None, item_indices=None, compute_all=False):
        """
        Forward pass through LightGCN layers.
        
        Args:
            user_indices: User indices for prediction
            item_indices: Item indices for prediction
            compute_all: Whether to compute embeddings for all nodes
            
        Returns:
            User and item embeddings
        """
        if self.adjacency_matrix is None:
            raise ValueError("Adjacency matrix not set")
        
        # Initial embeddings
        user_emb_0 = self.user_embeddings.weight
        item_emb_0 = self.item_embeddings.weight
        
        # Concatenate user and item embeddings
        all_embeddings = torch.cat([user_emb_0, item_emb_0], dim=0)
        embeddings_list = [all_embeddings]
        
        # Multi-layer graph convolution
        for layer in range(self.n_layers):
            # Graph convolution: A * embeddings
            all_embeddings = torch.sparse.mm(self.adjacency_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        
        # Average embeddings across all layers
        final_embeddings = torch.mean(torch.stack(embeddings_list, dim=0), dim=0)
        
        # Split back to user and item embeddings
        user_embeddings = final_embeddings[:self.n_users]
        item_embeddings = final_embeddings[self.n_users:]
        
        if compute_all:
            return user_embeddings, item_embeddings
        
        if user_indices is not None and item_indices is not None:
            user_emb = user_embeddings[user_indices]
            item_emb = item_embeddings[item_indices]
            return user_emb, item_emb
        
        return user_embeddings, item_embeddings
    
    def predict(self, user_embeddings: torch.Tensor, item_embeddings: torch.Tensor) -> torch.Tensor:
        """Predict interaction probability using inner product."""
        return torch.sum(user_embeddings * item_embeddings, dim=-1)


class LightGCNRecommender(GraphBasedRecommenderBase):
    """
    LightGCN-based graph recommender for collaborative filtering.
    Implements simplified graph convolution for user-item interaction prediction.
    """
    
    def __init__(self, embedding_dim=128, n_layers=3, learning_rate=0.001, epochs=100,
                 batch_size=1024, reg_weight=1e-4, negative_sampling_ratio=1.0, 
                 early_stopping_patience=10, graph_dropout=0.1, **kwargs):
        """
        Args:
            embedding_dim: Dimension of user/item embeddings
            n_layers: Number of graph convolution layers
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            batch_size: Training batch size
            reg_weight: L2 regularization weight
            negative_sampling_ratio: Ratio of negative samples to positive samples
            early_stopping_patience: Patience for early stopping
            graph_dropout: Graph dropout rate (randomly remove edges)
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.reg_weight = reg_weight
        self.negative_sampling_ratio = negative_sampling_ratio
        self.early_stopping_patience = early_stopping_patience
        self.graph_dropout = graph_dropout
        
        # Model components
        self.model = None
        self.adjacency_matrix = None
        self.positive_interactions = []
        
    def _build_adjacency_matrix(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build normalized adjacency matrix for LightGCN.
        
        Returns:
            Normalized adjacency matrix and degree matrix
        """
        if self.graph is None:
            raise ValueError("Graph not constructed")
        
        total_nodes = self.n_users + self.n_items
        
        # Build adjacency matrix indices and values
        row_indices = []
        col_indices = []
        values = []
        
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
        
        # Compute degree matrix for normalization
        degrees = torch.sparse.sum(adj_matrix, dim=1).to_dense()
        degrees = torch.pow(degrees + 1e-8, -0.5)  # Add small epsilon for numerical stability
        
        # Create degree matrix
        degree_indices = torch.arange(total_nodes).unsqueeze(0).repeat(2, 1)
        degree_values = degrees
        degree_matrix = torch.sparse.FloatTensor(degree_indices, degree_values, (total_nodes, total_nodes))
        
        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        normalized_adj = torch.sparse.mm(torch.sparse.mm(degree_matrix, adj_matrix), degree_matrix)
        
        return normalized_adj, degree_matrix
    
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
                
                # Convert to internal indices (users: 0 to n_users-1, items: 0 to n_items-1)
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
        """Train LightGCN model using BPR loss."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("Warning: No graph available for training")
            return
        
        # Build adjacency matrix
        print("Building normalized adjacency matrix...")
        self.adjacency_matrix, degree_matrix = self._build_adjacency_matrix()
        
        # Initialize model
        self.model = LightGCNModel(self.n_users, self.n_items, self.embedding_dim, self.n_layers)
        self.model.set_adjacency_matrix(self.adjacency_matrix, degree_matrix)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.reg_weight)
        criterion = nn.BCEWithLogitsLoss()
        
        # Prepare training data
        training_data = self._prepare_training_data()
        
        print(f"Training LightGCN model for {self.epochs} epochs...")
        
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
                self.model.set_adjacency_matrix(adj_matrix, degree_matrix)
            
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
                user_embeddings, item_embeddings = self.model(user_indices, item_indices)
                predictions = self.model.predict(user_embeddings, item_embeddings)
                
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
        
        print("LightGCN training completed")
    
    def _predict_link_probabilities(self, user_item_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """Predict link probabilities using LightGCN embeddings."""
        if self.model is None:
            print("Warning: No model available")
            return np.random.random(len(user_item_pairs))
        
        self.model.eval()
        probabilities = []
        
        # Get all embeddings
        with torch.no_grad():
            all_user_embeddings, all_item_embeddings = self.model(compute_all=True)
        
        for user_idx, item_idx in user_item_pairs:
            # Check if user and item are in vocabulary
            if user_idx not in self.user_vocab or item_idx not in self.item_vocab:
                probabilities.append(0.0)
                continue
            
            user_node = self.user_vocab[user_idx]
            item_node = self.item_vocab[item_idx]
            
            # Convert to internal indices
            user_internal = user_node
            item_internal = item_node - self.n_users
            
            # Check bounds
            if user_internal >= self.n_users or item_internal >= self.n_items:
                probabilities.append(0.0)
                continue
            
            # Get embeddings and compute prediction
            user_embed = all_user_embeddings[user_internal].unsqueeze(0)
            item_embed = all_item_embeddings[item_internal].unsqueeze(0)
            
            with torch.no_grad():
                prediction = self.model.predict(user_embed, item_embed)
                probability = torch.sigmoid(prediction).item()
            
            probabilities.append(probability)
        
        return np.array(probabilities) 