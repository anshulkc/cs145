# RNN/LSTM/GRU sequence-based recommenders

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
from .base import SequenceBasedRecommenderBase


class SequenceDataset(Dataset):
    """Dataset for sequence training."""
    
    def __init__(self, sequences, vocab_size, sequence_length):
        """
        Args:
            sequences: List of item sequences
            vocab_size: Size of item vocabulary
            sequence_length: Maximum sequence length
        """
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.data = []
        
        for sequence in sequences:
            if len(sequence) < 2:
                continue
                
            # Create input-output pairs from sequence
            for i in range(1, len(sequence)):
                input_seq = sequence[:i]
                target = sequence[i]
                
                # Pad or truncate input sequence
                if len(input_seq) > sequence_length:
                    input_seq = input_seq[-sequence_length:]
                else:
                    input_seq = [vocab_size] * (sequence_length - len(input_seq)) + input_seq
                
                self.data.append((input_seq, target))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, target = self.data[idx]
        return torch.LongTensor(input_seq), torch.LongTensor([target])


class RNNModel(nn.Module):
    """RNN-based sequence model."""
    
    def __init__(self, vocab_size, embedding_dim, hidden_size, num_layers, dropout=0.2, rnn_type='RNN'):
        """
        Args:
            vocab_size: Size of item vocabulary
            embedding_dim: Dimension of item embeddings
            hidden_size: Hidden size of RNN
            num_layers: Number of RNN layers
            dropout: Dropout probability
            rnn_type: Type of RNN ('RNN', 'LSTM', 'GRU')
        """
        super(RNNModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size + 1, embedding_dim, padding_idx=vocab_size)  # padding token at end
        
        # RNN layer
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(embedding_dim, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_size, num_layers, 
                              batch_first=True, dropout=dropout if num_layers > 1 else 0)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_size, num_layers, 
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        else:
            raise ValueError(f"Unsupported RNN type: {rnn_type}")
        
        # Output layer
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, x, hidden=None):
        """
        Forward pass.
        
        Args:
            x: Input sequences (batch_size, seq_len)
            hidden: Initial hidden state (optional)
            
        Returns:
            Output logits and final hidden state
        """
        # Embedding
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # RNN
        rnn_out, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_len, hidden_size)
        
        # Use last timestep output
        last_output = rnn_out[:, -1, :]  # (batch_size, hidden_size)
        
        # Dropout and linear layer
        output = self.dropout(last_output)
        logits = self.output(output)  # (batch_size, vocab_size)
        
        return logits, hidden
    
    def predict_proba(self, x, hidden=None):
        """Get prediction probabilities."""
        logits, hidden = self.forward(x, hidden)
        probs = torch.softmax(logits, dim=-1)
        return probs, hidden


class RNNRecommenderBase(SequenceBasedRecommenderBase):
    """Base class for RNN-based recommenders."""
    
    def __init__(self, seed=None, hidden_size=64, num_layers=1, dropout=0.2,
                 sequence_length=20, embedding_dim=32, revenue_weight=1.0,
                 min_sequence_length=2, rnn_type='RNN'):
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
            rnn_type: Type of RNN ('RNN', 'LSTM', 'GRU')
        """
        super().__init__(seed=seed, sequence_length=sequence_length,
                         revenue_weight=revenue_weight, min_sequence_length=min_sequence_length)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10
        
        # Model and training objects
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        
        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        print(f"Initialized {rnn_type} recommender with hidden_size={hidden_size}, num_layers={num_layers}")
    
    def _train_model(self) -> None:
        """Train the RNN model using curated sequences."""
        print(f"Training {self.rnn_type} model...")
        
        # Prepare training data
        sequences = []
        for user_idx, sequence_data in self.user_sequences.items():
            # Convert to vocabulary indices
            item_sequence = [item_idx for item_idx, _, _, _ in sequence_data]
            vocab_sequence = []
            for item_idx in item_sequence:
                if item_idx in self.item_vocab:
                    vocab_sequence.append(self.item_vocab[item_idx])  # Keep 0-based indexing
            
            if len(vocab_sequence) >= self.min_sequence_length:
                sequences.append(vocab_sequence)
        
        if len(sequences) == 0:
            print("Warning: No valid sequences for training")
            return
        
        # Create dataset and dataloader (add 1 for padding token)
        dataset = SequenceDataset(sequences, self.vocab_size + 1, self.sequence_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Training on {len(dataset)} sequence pairs")
        
        # Initialize model (add 1 for padding token)
        self.model = RNNModel(
            vocab_size=self.vocab_size + 1,
            embedding_dim=self.embedding_dim,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            rnn_type=self.rnn_type
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_inputs, batch_targets in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs, _ = self.model(batch_inputs)
                
                # Calculate loss
                loss = self.criterion(outputs, batch_targets.squeeze())
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Update parameters
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0
            print(f"Epoch {epoch+1}/{self.epochs}, Average Loss: {avg_loss:.4f}")
        
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
                
                # Get predictions
                probs, _ = self.model.predict_proba(input_tensor)
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


class LSTMRecommender(RNNRecommenderBase):
    """LSTM-based sequence recommender."""
    
    def __init__(self, **kwargs):
        super().__init__(rnn_type='LSTM', **kwargs)


class GRURecommender(RNNRecommenderBase):
    """GRU-based sequence recommender."""
    
    def __init__(self, **kwargs):
        super().__init__(rnn_type='GRU', **kwargs) 