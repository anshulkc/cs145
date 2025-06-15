# Node2Vec implementation for graph-based recommendations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, Dict, Optional
import networkx as nx
from collections import defaultdict
import random

from .base import GraphBasedRecommenderBase


class Node2VecModel(nn.Module):
    """Node2Vec embedding model using skip-gram approach."""
    
    def __init__(self, vocab_size: int, embedding_dim: int):
        super(Node2VecModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Input and output embeddings
        self.input_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.output_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Initialize embeddings
        self.input_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
        self.output_embeddings.weight.data.uniform_(-0.5/embedding_dim, 0.5/embedding_dim)
    
    def forward(self, center_nodes, context_nodes, negative_nodes):
        """
        Args:
            center_nodes: Center node indices [batch_size]
            context_nodes: Context node indices [batch_size]  
            negative_nodes: Negative sample indices [batch_size, num_negative]
        """
        batch_size = center_nodes.size(0)
        num_negative = negative_nodes.size(1)
        
        # Get embeddings
        center_embed = self.input_embeddings(center_nodes)  # [batch_size, embed_dim]
        context_embed = self.output_embeddings(context_nodes)  # [batch_size, embed_dim]
        negative_embed = self.output_embeddings(negative_nodes)  # [batch_size, num_negative, embed_dim]
        
        # Positive score
        positive_score = torch.sum(center_embed * context_embed, dim=1)  # [batch_size]
        positive_loss = -torch.log(torch.sigmoid(positive_score))
        
        # Negative scores
        # center_embed: [batch_size, embed_dim] -> [batch_size, 1, embed_dim]
        center_embed_expanded = center_embed.unsqueeze(1)
        negative_scores = torch.sum(center_embed_expanded * negative_embed, dim=2)  # [batch_size, num_negative]
        negative_loss = -torch.sum(torch.log(torch.sigmoid(-negative_scores)), dim=1)  # [batch_size]
        
        # Total loss
        total_loss = positive_loss + negative_loss
        return total_loss.mean()
    
    def get_embeddings(self):
        """Get learned node embeddings."""
        return self.input_embeddings.weight.data.cpu().numpy()


class Node2VecRecommender(GraphBasedRecommenderBase):
    """
    Node2Vec-based graph recommender using random walks and skip-gram embeddings.
    Implements the Node2Vec algorithm for learning node representations in bipartite graphs.
    """
    
    def __init__(self, embedding_dim=128, walk_length=20, num_walks=10, window_size=5,
                 p=1.0, q=1.0, learning_rate=0.01, epochs=100, num_negative=5,
                 batch_size=512, **kwargs):
        """
        Args:
            embedding_dim: Dimension of node embeddings
            walk_length: Length of random walks
            num_walks: Number of walks per node
            window_size: Context window size for skip-gram
            p: Return parameter (controls return to previous node)
            q: In-out parameter (controls exploration vs exploitation)
            learning_rate: Learning rate for optimization
            epochs: Number of training epochs
            num_negative: Number of negative samples per positive sample
            batch_size: Training batch size
        """
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.window_size = window_size
        self.p = p
        self.q = q
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.num_negative = num_negative
        self.batch_size = batch_size
        
        # Model components
        self.model = None
        self.node_embeddings = None
        self.walks = []
        
    def _generate_random_walks(self) -> List[List[int]]:
        """Generate random walks using Node2Vec biased sampling."""
        walks = []
        nodes = list(self.graph.nodes())
        
        print(f"Generating {self.num_walks} random walks of length {self.walk_length} for {len(nodes)} nodes...")
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)  # Randomize starting order
            for node in nodes:
                walk = self._node2vec_walk(node)
                walks.append(walk)
        
        print(f"Generated {len(walks)} random walks")
        return walks
    
    def _node2vec_walk(self, start_node: int) -> List[int]:
        """Generate a single biased random walk starting from start_node."""
        walk = [start_node]
        
        while len(walk) < self.walk_length:
            current = walk[-1]
            neighbors = list(self.graph.neighbors(current))
            
            if len(neighbors) == 0:
                break
                
            if len(walk) == 1:
                # First step: uniform random choice
                next_node = random.choice(neighbors)
            else:
                # Biased choice based on p and q parameters
                prev_node = walk[-2]
                next_node = self._biased_choice(current, prev_node, neighbors)
            
            walk.append(next_node)
        
        return walk
    
    def _biased_choice(self, current: int, prev: int, neighbors: List[int]) -> int:
        """Make biased choice for next node based on p and q parameters."""
        weights = []
        
        for neighbor in neighbors:
            if neighbor == prev:
                # Return to previous node
                weight = 1.0 / self.p
            elif self.graph.has_edge(neighbor, prev):
                # Stay in local neighborhood  
                weight = 1.0
            else:
                # Explore new area
                weight = 1.0 / self.q
            weights.append(weight)
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(neighbors)
        
        weights = [w / total_weight for w in weights]
        
        # Sample based on weights
        r = random.random()
        cumsum = 0
        for i, weight in enumerate(weights):
            cumsum += weight
            if r <= cumsum:
                return neighbors[i]
        
        return neighbors[-1]  # Fallback
    
    def _generate_training_pairs(self) -> List[Tuple[int, int]]:
        """Generate (center, context) training pairs from random walks."""
        pairs = []
        
        for walk in self.walks:
            for i, center in enumerate(walk):
                # Define context window
                start = max(0, i - self.window_size)
                end = min(len(walk), i + self.window_size + 1)
                
                for j in range(start, end):
                    if i != j:  # Skip center node itself
                        context = walk[j]
                        pairs.append((center, context))
        
        print(f"Generated {len(pairs)} training pairs from walks")
        return pairs
    
    def _negative_sampling(self, batch_size: int) -> torch.Tensor:
        """Generate negative samples for training."""
        total_nodes = self.n_users + self.n_items
        negative_samples = torch.randint(0, total_nodes, (batch_size, self.num_negative))
        return negative_samples
    
    def _train_model(self) -> None:
        """Train Node2Vec embeddings using skip-gram with negative sampling."""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            print("Warning: No graph available for training")
            return
        
        # Generate random walks
        self.walks = self._generate_random_walks()
        if len(self.walks) == 0:
            print("Warning: No walks generated")
            return
        
        # Generate training pairs
        training_pairs = self._generate_training_pairs()
        if len(training_pairs) == 0:
            print("Warning: No training pairs generated")
            return
        
        # Initialize model
        total_nodes = self.n_users + self.n_items
        self.model = Node2VecModel(total_nodes, self.embedding_dim)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        print(f"Training Node2Vec model for {self.epochs} epochs...")
        
        # Training loop
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            # Shuffle training pairs
            random.shuffle(training_pairs)
            
            # Mini-batch training
            for i in range(0, len(training_pairs), self.batch_size):
                batch_pairs = training_pairs[i:i + self.batch_size]
                if len(batch_pairs) == 0:
                    continue
                
                # Prepare batch
                center_nodes = torch.tensor([pair[0] for pair in batch_pairs], dtype=torch.long)
                context_nodes = torch.tensor([pair[1] for pair in batch_pairs], dtype=torch.long)
                negative_nodes = self._negative_sampling(len(batch_pairs))
                
                # Forward pass
                optimizer.zero_grad()
                loss = self.model(center_nodes, context_nodes, negative_nodes)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            if (epoch + 1) % 20 == 0:
                avg_loss = total_loss / max(num_batches, 1)
                print(f"Epoch {epoch + 1}/{self.epochs}, Average Loss: {avg_loss:.4f}")
        
        # Store final embeddings
        self.node_embeddings = self.model.get_embeddings()
        print(f"Node2Vec training completed. Embeddings shape: {self.node_embeddings.shape}")
    
    def _predict_link_probabilities(self, user_item_pairs: List[Tuple[int, int]]) -> np.ndarray:
        """Predict link probabilities using dot product of embeddings."""
        if self.node_embeddings is None:
            print("Warning: No embeddings available")
            return np.random.random(len(user_item_pairs))
        
        probabilities = []
        
        for user_idx, item_idx in user_item_pairs:
            # Get node IDs
            if user_idx not in self.user_vocab or item_idx not in self.item_vocab:
                probabilities.append(0.0)
                continue
            
            user_node = self.user_vocab[user_idx]
            item_node = self.item_vocab[item_idx]
            
            # Get embeddings
            user_embed = self.node_embeddings[user_node]
            item_embed = self.node_embeddings[item_node]
            
            # Compute similarity (dot product)
            similarity = np.dot(user_embed, item_embed)
            
            # Convert to probability using sigmoid
            probability = 1.0 / (1.0 + np.exp(-similarity))
            probabilities.append(probability)
        
        return np.array(probabilities) 