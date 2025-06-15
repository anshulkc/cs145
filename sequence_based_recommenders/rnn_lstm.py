# RNN/GRU sequence-based recommenders

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from .base import SequenceBasedRecommenderBase


class SequenceDataset(Dataset):
    """Dataset for sequence training with price features."""
    
    def __init__(self, sequences, vocab_size, sequence_length, use_price_features=False):
        """
        Args:
            sequences: List of (item_sequence, price_sequence) tuples or just item sequences
            vocab_size: Size of item vocabulary
            sequence_length: Maximum sequence length
            use_price_features: Whether to include price features
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.use_price_features = use_price_features
        self.data = []
        
        for seq_data in sequences:
            if use_price_features and isinstance(seq_data, tuple):
                item_sequence, price_sequence = seq_data
            else:
                item_sequence = seq_data if not isinstance(seq_data, tuple) else seq_data[0]
                price_sequence = [1.0] * len(item_sequence)  # Default prices
            
            if len(item_sequence) < 2:
                continue
                
            # Create input-output pairs from sequence
            for i in range(1, len(item_sequence)):
                input_items = item_sequence[:i]
                input_prices = price_sequence[:i]
                target = item_sequence[i]
                
                # Pad or truncate sequences
                if len(input_items) > sequence_length:
                    input_items = input_items[-sequence_length:]
                    input_prices = input_prices[-sequence_length:]
                else:
                    pad_len = sequence_length - len(input_items)
                    input_items = [vocab_size] * pad_len + input_items
                    input_prices = [0.0] * pad_len + input_prices
                
                if use_price_features:
                    self.data.append((input_items, input_prices, target))
                else:
                    self.data.append((input_items, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if self.use_price_features:
            input_items, input_prices, target = self.data[idx]
            return (torch.LongTensor(input_items), 
                   torch.FloatTensor(input_prices), 
                   torch.LongTensor([target]))
        else:
            input_items, target = self.data[idx]
            return torch.LongTensor(input_items), torch.LongTensor([target])


class RNNModel(nn.Module):
    """RNN-based sequence model with price and category features."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.2, 
                 rnn_type='RNN', use_price_features=True, price_embedding_dim=8):
        """
        Args:
            vocab_size: Size of item vocabulary
            embedding_dim: Dimension of item embeddings
            hidden_size: Hidden size of RNN
            num_layers: Number of RNN layers
            dropout: Dropout probability
            rnn_type: Type of RNN ('RNN', 'GRU', 'LSTM')
            use_price_features: Whether to include price embeddings
            price_embedding_dim: Dimension of price embeddings
        """
        super(RNNModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        self.use_price_features = use_price_features
        self.price_embedding_dim = price_embedding_dim
        
        # Item embedding layer
        self.item_embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)
        
        # Price embedding (discretized price buckets)
        if use_price_features:
            self.price_embedding = nn.Embedding(100, price_embedding_dim)  # 100 price buckets
            rnn_input_dim = embedding_dim + price_embedding_dim
        else:
            rnn_input_dim = embedding_dim
        
        # RNN layer
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(rnn_input_dim, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(rnn_input_dim, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(rnn_input_dim, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Output layers
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, prices=None, hidden=None, teacher_forcing_ratio=0.5, target_seq=None):
        """
        Forward pass with optional teacher forcing.
        
        Args:
            x: Input sequences (batch_size, seq_len)
            prices: Price sequences (batch_size, seq_len) - optional
            hidden: Initial hidden state (optional)
            teacher_forcing_ratio: Probability of using teacher forcing during training
            target_seq: Target sequence for teacher forcing (batch_size, seq_len)
            
        Returns:
            Output logits and final hidden state
        """
        batch_size, seq_len = x.size()
        
        # Item embeddings
        item_embedded = self.item_embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # Combine with price embeddings if available
        if self.use_price_features and prices is not None:
            # Discretize prices into buckets (0-99)
            price_buckets = torch.clamp((prices * 10).long(), 0, 99)
            price_embedded = self.price_embedding(price_buckets)  # (batch_size, seq_len, price_embedding_dim)
            embedded = torch.cat([item_embedded, price_embedded], dim=-1)
        else:
            embedded = item_embedded
        
        if self.training and teacher_forcing_ratio > 0 and target_seq is not None:
            # Teacher forcing: use target sequence as input
            rnn_out, hidden = self.rnn(embedded, hidden)
            # Apply dropout and output layer to all timesteps
            rnn_out = self.dropout(rnn_out)
            logits = self.output(rnn_out)  # (batch_size, seq_len, vocab_size)
            return logits, hidden
        else:
            # Standard inference: use last timestep
            rnn_out, hidden = self.rnn(embedded, hidden)
            last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
            output = self.dropout(last_output)
            logits = self.output(output)  # (batch_size, vocab_size)
            return logits, hidden
    
    def predict_proba(self, x, prices=None, hidden=None):
        """Get prediction probabilities."""
        logits, hidden = self.forward(x, prices=prices, hidden=hidden)
        probs = torch.softmax(logits, dim=-1)
        return probs, hidden


class RNNRecommenderBase(SequenceBasedRecommenderBase):
    """Base class for RNN-based recommenders."""
    
    def __init__(self, seed=None, hidden_size=64, num_layers=1, dropout=0.2,
                 sequence_length=20, embedding_dim=32, revenue_weight=1.0,
                 min_sequence_length=2, rnn_type='RNN', learning_rate=0.001,
                 weight_decay=0.0, teacher_forcing_ratio=0.5, use_price_features=True,
                 early_stopping_patience=5):
        """
        Args:
            seed: Random seed for reproducibility
            hidden_size: Hidden size of RNN
            num_layers: Number of RNN layers
            dropout: Dropout probability
            sequence_length: Maximum sequence length
            embedding_dim: Dimension of item embeddings
            revenue_weight: Weight for revenue optimization
            min_sequence_length: Minimum sequence length for training
            rnn_type: Type of RNN ('RNN', 'GRU', 'LSTM')
            learning_rate: Learning rate for optimizer
            weight_decay: L2 regularization weight
            teacher_forcing_ratio: Probability of using teacher forcing
            use_price_features: Whether to use price embeddings
            early_stopping_patience: Patience for early stopping
        """
        super().__init__(seed=seed, sequence_length=sequence_length,
                         revenue_weight=revenue_weight, min_sequence_length=min_sequence_length)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.use_price_features = use_price_features
        self.early_stopping_patience = early_stopping_patience
        
        # Training parameters
        self.batch_size = 32
        self.epochs = 20
        
        # Model and training objects
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        print(f"Initialized {rnn_type} recommender with hidden_size={hidden_size}, num_layers={num_layers}, weight_decay={weight_decay}")
    
    def _train_model(self) -> None:
        """Train the RNN model using curated sequences with advanced features."""
        print(f"Training {self.rnn_type} model with weight_decay={self.weight_decay}, teacher_forcing={self.teacher_forcing_ratio}...")
        
        # Prepare training data with price features
        sequences = []
        for user_idx, sequence_data in self.user_sequences.items():
            item_sequence = []
            price_sequence = []
            
            for item_idx, price, _, _ in sequence_data:
                if item_idx in self.item_vocab:
                    item_sequence.append(self.item_vocab[item_idx])
                    price_sequence.append(float(price))
            
            if len(item_sequence) >= self.min_sequence_length:
                if self.use_price_features:
                    sequences.append((item_sequence, price_sequence))
                else:
                    sequences.append(item_sequence)
        
        if len(sequences) == 0:
            print("Warning: No valid sequences for training")
            return
        
        # Create dataset and dataloader
        dataset = SequenceDataset(sequences, self.vocab_size, self.sequence_length, self.use_price_features)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Training on {len(dataset)} sequence pairs")
        
        # Initialize model
        self.model = RNNModel(
            vocab_size=self.vocab_size + 1,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            rnn_type=self.rnn_type,
            use_price_features=self.use_price_features
        )
        
        # Initialize optimizer with weight decay
        self.optimizer = optim.Adam(self.model.parameters(), 
                                   lr=self.learning_rate, 
                                   weight_decay=self.weight_decay)
        
        # Early stopping variables
        best_loss = float('inf')
        patience_counter = 0
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_data in dataloader:
                if self.use_price_features:
                    batch_inputs, batch_prices, batch_targets = batch_data
                else:
                    batch_inputs, batch_targets = batch_data
                    batch_prices = None
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass with teacher forcing
                if self.model.training and self.teacher_forcing_ratio > 0:
                    # Create target sequence for teacher forcing
                    target_seq = batch_inputs[:, 1:].contiguous()  # Shift by one
                    outputs, _ = self.model(batch_inputs[:, :-1], 
                                          prices=batch_prices[:, :-1] if batch_prices is not None else None,
                                          teacher_forcing_ratio=self.teacher_forcing_ratio,
                                          target_seq=target_seq)
                    
                    # Reshape for loss calculation
                    if len(outputs.shape) == 3:  # Teacher forcing returns (batch, seq, vocab)
                        outputs = outputs[:, -1, :]  # Use last timestep
                else:
                    outputs, _ = self.model(batch_inputs, prices=batch_prices)
                
                # Calculate loss
                loss = self.criterion(outputs, batch_targets.squeeze())
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.4f}")
            
            # Early stopping check
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        self.model.eval()
        print("Training completed")
    
    def _predict_next_items(self, user_sequences: List[List[int]], candidate_items: List[int]) -> np.ndarray:
        """
        Predict probabilities for candidate items given user sequences.
        
        Args:
            user_sequences: List of item sequences for each user (vocab indices)
            candidate_items: List of candidate item indices (vocab indices)
            
        Returns:
            Array of shape (n_users, n_items) with prediction probabilities
        """
        if self.model is None:
            # Return uniform probabilities if model not trained
            n_users = len(user_sequences)
            n_items = len(candidate_items)
            return np.ones((n_users, n_items)) / n_items
        
        self.model.eval()
        predictions = []
        
        with torch.no_grad():
            for sequence in user_sequences:
                # Prepare input sequence
                input_seq = sequence.copy()
                
                # Pad or truncate (use vocab_size as padding token)
                if len(input_seq) > self.sequence_length:
                    input_seq = input_seq[-self.sequence_length:]
                else:
                    input_seq = [self.vocab_size] * (self.sequence_length - len(input_seq)) + input_seq
                
                # Convert to tensor
                input_tensor = torch.LongTensor(input_seq).unsqueeze(0)  # (1, seq_len)
                
                # Prepare price tensor if using price features
                if self.use_price_features:
                    # Use default prices for prediction (since we don't have actual prices)
                    price_tensor = torch.ones(1, self.sequence_length) * 0.5  # Default price
                else:
                    price_tensor = None
                
                # Get predictions
                probs, _ = self.model.predict_proba(input_tensor, prices=price_tensor)
                probs = probs.squeeze(0).numpy()  # (vocab_size,)
                
                # Extract probabilities for candidate items
                candidate_probs = []
                for item_idx in candidate_items:
                    if item_idx < len(probs):
                        candidate_probs.append(probs[item_idx])
                    else:
                        candidate_probs.append(0.0)
                
                predictions.append(candidate_probs)
        
        return np.array(predictions)


class RNNRecommender(RNNRecommenderBase):
    """RNN-based sequence recommender."""
    
    def __init__(self, **kwargs):
        super().__init__(rnn_type='RNN', **kwargs)


class GRURecommender(RNNRecommenderBase):
    """GRU-based sequence recommender."""
    
    def __init__(self, **kwargs):
        super().__init__(rnn_type='GRU', **kwargs)


class LSTMRecommender(RNNRecommenderBase):
    """LSTM-based sequence recommender."""
    
    def __init__(self, **kwargs):
        super().__init__(rnn_type='LSTM', **kwargs) 