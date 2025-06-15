# Transformer-based sequence recommender

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional
import math
from .base import SequenceBasedRecommenderBase


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        return x + self.pe[:x.size(0), :]


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % nhead == 0
        
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        seq_len = query.size(1)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, seq_len, self.nhead, self.d_k).transpose(1, 2)
        
        # Attention
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        # Final linear layer
        output = self.w_o(attn_output)
        
        return output, attn_weights
    
    def attention(self, Q, K, V, mask=None):
        d_k = Q.size(-1)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, V)
        
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Transformer encoder block."""
    
    def __init__(self, d_model, nhead, d_ff, dropout=0.1):
        super(TransformerBlock, self).__init__()
        
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # Self-attention with residual connection
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x


class TransformerModel(nn.Module):
    """Transformer-based sequence model for recommendation."""
    
    def __init__(self, vocab_size, d_model, nhead, num_layers, sequence_length, dropout=0.1):
        super(TransformerModel, self).__init__()
        
        # vocab_size already includes +1 for padding token
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.sequence_length = sequence_length
        
        # Embedding layers (vocab_size already includes padding token)
        self.item_embedding = nn.Embedding(vocab_size, d_model, padding_idx=vocab_size-1)  # padding token at end
        self.pos_encoding = PositionalEncoding(d_model, sequence_length)
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, nhead, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        # Output layer (vocab_size already includes padding token)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input sequences (batch_size, seq_len)
            mask: Attention mask (optional)
        """
        batch_size, seq_len = x.size()
        
        # Embedding
        embedded = self.item_embedding(x) * math.sqrt(self.d_model)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        embedded = embedded.transpose(0, 1)  # (seq_len, batch_size, d_model)
        embedded = self.pos_encoding(embedded)
        embedded = embedded.transpose(0, 1)  # (batch_size, seq_len, d_model)
        
        # Create causal mask for training
        if mask is None and self.training:
            mask = self._generate_causal_mask(seq_len).to(x.device)
        
        # Pass through transformer blocks
        hidden = embedded
        for transformer_block in self.transformer_blocks:
            hidden = transformer_block(hidden, mask)
        
        # Layer normalization and dropout
        hidden = self.layer_norm(hidden)
        hidden = self.dropout(hidden)
        
        # Output projection
        logits = self.output(hidden)  # (batch_size, seq_len, vocab_size)
        
        return logits
    
    def _generate_causal_mask(self, size):
        """Generate causal mask to prevent attending to future positions."""
        mask = torch.tril(torch.ones(size, size))
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        return mask
    
    def predict_next(self, x):
        """Predict next item probabilities."""
        logits = self.forward(x)  # (batch_size, seq_len, vocab_size)
        last_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        probs = F.softmax(last_logits, dim=-1)
        return probs


class TransformerDataset(Dataset):
    """Dataset for transformer training."""
    
    def __init__(self, sequences, vocab_size, sequence_length):
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length
        self.data = []
        
        for sequence in sequences:
            if len(sequence) < 2:
                continue
                
            # Create input-target pairs
            for i in range(1, len(sequence)):
                input_seq = sequence[:i+1]  # Include target in input for teacher forcing
                
                # Pad or truncate
                if len(input_seq) > sequence_length:
                    input_seq = input_seq[-sequence_length:]
                else:
                    input_seq = [vocab_size] * (sequence_length - len(input_seq)) + input_seq
                
                # Create targets (shifted by one position)
                targets = input_seq[1:] + [vocab_size]  # Shift left and pad
                
                self.data.append((input_seq, targets))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_seq, targets = self.data[idx]
        return torch.LongTensor(input_seq), torch.LongTensor(targets)


class TransformerRecommender(SequenceBasedRecommenderBase):
    """Transformer-based sequence recommender."""
    
    def __init__(self, seed=None, d_model=128, nhead=8, num_layers=4,
                 sequence_length=30, dropout=0.3, revenue_weight=1.5,
                 min_sequence_length=2):
        """
        Args:
            seed: Random seed for reproducibility
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            sequence_length: Maximum sequence length
            dropout: Dropout probability
            revenue_weight: Weight for revenue optimization
            min_sequence_length: Minimum sequence length for training
        """
        super().__init__(seed=seed, sequence_length=sequence_length,
                         revenue_weight=revenue_weight, min_sequence_length=min_sequence_length)
        
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Training parameters
        self.learning_rate = 0.001
        self.batch_size = 32
        self.epochs = 10
        
        # Model and training objects
        self.model = None
        self.optimizer = None
        self.criterion = None  # Will be set in _train_model after vocab_size is determined
        
        # Set random seeds
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        print(f"Initialized Transformer recommender with d_model={d_model}, nhead={nhead}, num_layers={num_layers}")
    
    def _train_model(self) -> None:
        """Train the transformer model using curated sequences."""
        print("Training Transformer model...")
        
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
        
        # Create dataset and dataloader
        dataset = TransformerDataset(sequences, self.vocab_size, self.sequence_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Training on {len(dataset)} sequence pairs")
        
        # Initialize model (add 1 for padding token)
        self.model = TransformerModel(
            vocab_size=self.vocab_size + 1,
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            sequence_length=self.sequence_length,
            dropout=self.dropout
        )
        
        # Initialize optimizer and loss criterion
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.vocab_size)  # Ignore padding tokens
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.epochs):
            total_loss = 0
            num_batches = 0
            
            for batch_inputs, batch_targets in dataloader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                logits = self.model(batch_inputs)  # (batch_size, seq_len, vocab_size)
                
                # Reshape for loss calculation
                logits = logits.view(-1, self.vocab_size + 1)  # (batch_size * seq_len, vocab_size + 1)
                targets = batch_targets.view(-1)  # (batch_size * seq_len,)
                
                # Calculate loss
                loss = self.criterion(logits, targets)
                
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
                probs = self.model.predict_next(input_tensor)
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