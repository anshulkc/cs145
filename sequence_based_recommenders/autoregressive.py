# Autoregressive/N-gram sequence-based recommender

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
from .base import SequenceBasedRecommenderBase


class AutoRegressiveRecommender(SequenceBasedRecommenderBase):
    """
    Autoregressive (N-gram) recommender for sequence-based recommendation.
    
    Uses n-gram models to predict the next item in a sequence based on 
    the last n items in the user's interaction history.
    """
    
    def __init__(self, seed=None, order=2, smoothing_alpha=0.1, 
                 sequence_length=20, revenue_weight=1.0, min_sequence_length=2,
                 smoothing_type='additive', backoff_alpha=0.4):
        """
        Args:
            seed: Random seed for reproducibility
            order: N-gram order (1=unigram, 2=bigram, 3=trigram, etc.)
            smoothing_alpha: Smoothing parameter
            sequence_length: Maximum sequence length to consider
            revenue_weight: Weight for revenue optimization
            min_sequence_length: Minimum sequence length for training
            smoothing_type: Type of smoothing ('additive', 'kneser_ney', 'backoff')
            backoff_alpha: Backoff weight for interpolation
        """
        super().__init__(seed=seed, sequence_length=sequence_length, 
                         revenue_weight=revenue_weight, min_sequence_length=min_sequence_length)
        
        self.order = max(1, order)
        self.smoothing_alpha = smoothing_alpha
        self.smoothing_type = smoothing_type
        self.backoff_alpha = backoff_alpha
        
        # N-gram models for all orders (for backoff)
        self.ngram_counts = {}  # order -> {context -> Counter(next_item)}
        self.context_counts = {}  # order -> {context -> count}
        for o in range(1, self.order + 1):
            self.ngram_counts[o] = defaultdict(Counter)
            self.context_counts[o] = Counter()
        
        self.item_counts = Counter()
        self.total_items = 0
        
        print(f"Initialized AutoRegressive recommender with order={self.order}, smoothing={self.smoothing_type}")
    
    def _extract_ngrams(self, sequence: List[int]) -> Dict[int, List[Tuple[Tuple[int, ...], int]]]:
        """
        Extract n-grams of all orders from a sequence.
        
        Args:
            sequence: List of item indices
            
        Returns:
            Dict mapping order to list of (context, next_item) tuples
        """
        ngrams_by_order = {}
        
        for order in range(1, self.order + 1):
            ngrams = []
            for i in range(len(sequence)):
                if order == 1:
                    context = ()
                    next_item = sequence[i]
                    ngrams.append((context, next_item))
                else:
                    if i >= order - 1:
                        context = tuple(sequence[i - order + 1:i])
                        next_item = sequence[i]
                        ngrams.append((context, next_item))
            ngrams_by_order[order] = ngrams
        
        return ngrams_by_order
    
    def _train_model(self) -> None:
        """Train the n-gram model using curated sequences."""
        print(f"Training {self.order}-gram model with {self.smoothing_type} smoothing...")
        
        # Reset counters
        for order in range(1, self.order + 1):
            self.ngram_counts[order] = defaultdict(Counter)
            self.context_counts[order] = Counter()
        self.item_counts = Counter()
        self.total_items = 0
        
        # Process all user sequences
        for user_idx, sequence_data in self.user_sequences.items():
            item_sequence = [item_idx for item_idx, _, _, _ in sequence_data]
            
            vocab_sequence = []
            for item_idx in item_sequence:
                if item_idx in self.item_vocab:
                    vocab_sequence.append(self.item_vocab[item_idx])
            
            if len(vocab_sequence) < 2:
                continue
            
            # Extract n-grams for all orders
            ngrams_by_order = self._extract_ngrams(vocab_sequence)
            
            # Update counts for all orders
            for order, ngrams in ngrams_by_order.items():
                for context, next_item in ngrams:
                    self.ngram_counts[order][context][next_item] += 1
                    self.context_counts[order][context] += 1
                    if order == 1:  # Only count items once
                        self.item_counts[next_item] += 1
                        self.total_items += 1
        
        total_contexts = sum(len(self.ngram_counts[o]) for o in range(1, self.order + 1))
        print(f"Trained on {total_contexts} unique contexts across all orders")
        print(f"Total items: {self.total_items}")
        
        # Debug: Print example n-grams for highest order
        if len(self.ngram_counts[self.order]) > 0:
            print(f"Example {self.order}-grams:")
            for i, (context, next_items) in enumerate(list(self.ngram_counts[self.order].items())[:3]):
                top_items = next_items.most_common(3)
                print(f"  Context {context} -> {top_items}")
    
    def _get_ngram_probability(self, context: Tuple[int, ...], next_item: int) -> float:
        """
        Calculate n-gram probability with smoothing and backoff.
        
        Args:
            context: Context tuple (last n-1 items)
            next_item: Next item to predict
            
        Returns:
            Probability of next_item given context
        """
        order = len(context) + 1
        
        if self.smoothing_type == 'additive':
            return self._get_additive_smoothed_prob(order, context, next_item)
        elif self.smoothing_type == 'backoff':
            return self._get_backoff_prob(order, context, next_item)
        else:
            return self._get_additive_smoothed_prob(order, context, next_item)
    
    def _get_additive_smoothed_prob(self, order: int, context: Tuple[int, ...], next_item: int) -> float:
        """Additive (Laplace) smoothing."""
        if order > self.order:
            order = self.order
            context = context[-(order-1):] if order > 1 else ()
        
        ngram_count = self.ngram_counts[order][context][next_item]
        context_count = self.context_counts[order][context]
        
        if context_count == 0:
            return self._get_unigram_probability(next_item)
        
        smoothed_prob = (ngram_count + self.smoothing_alpha) / (context_count + self.smoothing_alpha * self.vocab_size)
        return smoothed_prob
    
    def _get_backoff_prob(self, order: int, context: Tuple[int, ...], next_item: int) -> float:
        """Backoff smoothing with interpolation."""
        if order > self.order:
            order = self.order
            context = context[-(order-1):] if order > 1 else ()
        
        if order == 1:
            return self._get_unigram_probability(next_item)
        
        ngram_count = self.ngram_counts[order][context][next_item]
        context_count = self.context_counts[order][context]
        
        if context_count == 0:
            # Back off to lower order
            lower_context = context[1:] if len(context) > 0 else ()
            return self._get_backoff_prob(order - 1, lower_context, next_item)
        
        # Interpolate with lower order
        higher_order_prob = ngram_count / context_count
        lower_context = context[1:] if len(context) > 0 else ()
        lower_order_prob = self._get_backoff_prob(order - 1, lower_context, next_item)
        
        interpolated_prob = self.backoff_alpha * higher_order_prob + (1 - self.backoff_alpha) * lower_order_prob
        return interpolated_prob
    
    def _get_unigram_probability(self, item: int) -> float:
        """
        Calculate unigram probability (fallback).
        
        Args:
            item: Item index
            
        Returns:
            Unigram probability of item
        """
        if self.total_items == 0:
            return 1.0 / self.vocab_size
        
        item_count = self.item_counts[item]
        # Apply smoothing
        smoothed_prob = (item_count + self.smoothing_alpha) / (self.total_items + self.smoothing_alpha * self.vocab_size)
        
        return smoothed_prob
    
    def _predict_next_items(self, user_sequences: List[List[int]], candidate_items: List[int]) -> np.ndarray:
        """
        Predict probabilities for candidate items given user sequences.
        
        Args:
            user_sequences: List of item sequences for each user (vocab indices)
            candidate_items: List of candidate item indices (vocab indices)
            
        Returns:
            Array of shape (n_users, n_items) with prediction probabilities
        """
        n_users = len(user_sequences)
        n_items = len(candidate_items)
        predictions = np.zeros((n_users, n_items))
        
        for i, sequence in enumerate(user_sequences):
            if len(sequence) == 0:
                # No sequence, use uniform distribution
                predictions[i, :] = 1.0 / n_items
                continue
            
            # Get context for prediction
            if self.order == 1:
                context = ()
            else:
                # Use last (order-1) items as context
                context_start = max(0, len(sequence) - self.order + 1)
                context = tuple(sequence[context_start:])
                
                # If sequence is shorter than required context, use what we have
                if len(context) < self.order - 1:
                    # Pad with a special token or use shorter context
                    context = tuple(sequence)
            
            # Predict probabilities for each candidate item
            for j, candidate_item in enumerate(candidate_items):
                prob = self._get_ngram_probability(context, candidate_item)
                predictions[i, j] = prob
        
        # Normalize probabilities (ensure they sum to 1 for each user)
        for i in range(n_users):
            prob_sum = predictions[i, :].sum()
            if prob_sum > 0:
                predictions[i, :] /= prob_sum
            else:
                predictions[i, :] = 1.0 / n_items
        
        return predictions
    
    def get_model_info(self) -> Dict:
        """Get information about the trained model."""
        return {
            'order': self.order,
            'smoothing_alpha': self.smoothing_alpha,
            'vocab_size': self.vocab_size,
            'num_contexts': len(self.ngram_counts),
            'total_ngrams': self.total_items,
            'is_fitted': self.is_fitted
        } 