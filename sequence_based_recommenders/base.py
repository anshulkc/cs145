# Base class for sequence-based recommenders

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict, Counter

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from pyspark.sql.types import LongType
from sim4rec.utils import pandas_to_spark
import warnings
warnings.filterwarnings('ignore')


class SequenceBasedRecommenderBase(ABC):
    """
    Base class for sequence-based recommenders.
    Provides common functionality for sequence curation, training, and prediction.
    """
    
    def __init__(self, seed=None, sequence_length=20, revenue_weight=1.0, 
                 min_sequence_length=2, temporal_decay=0.95):
        """
        Args:
            seed: Random seed for reproducibility
            sequence_length: Maximum sequence length to consider
            revenue_weight: Weight for revenue optimization in ranking
            min_sequence_length: Minimum sequence length for training
            temporal_decay: Decay factor for older interactions (1.0 = no decay)
        """
        self.seed = seed
        self.sequence_length = sequence_length
        self.revenue_weight = revenue_weight
        self.min_sequence_length = min_sequence_length
        self.temporal_decay = temporal_decay
        self.is_fitted = False
        
        # Sequence data structures
        self.user_sequences = {}  # user_idx -> [(item_idx, timestamp, price, response), ...]
        self.item_vocab = {}  # item_idx -> vocab_index mapping
        self.vocab_size = 0
        self.user_vocab = {}  # user_idx -> vocab_index mapping
        self.n_users = 0
        
        if seed is not None:
            np.random.seed(seed)
    
    def _curate_sequences(self, log: DataFrame, user_features: Optional[DataFrame] = None,
                         item_features: Optional[DataFrame] = None) -> None:
        """
        Curate user interaction sequences from log data.
        
        Args:
            log: Interaction log with user_idx, item_idx, relevance columns
            user_features: User features (optional)
            item_features: Item features (optional)
        """
        if log is None or log.count() == 0:
            print("Warning: No interaction data available for sequence curation")
            return
        
        # Add artificial timestamps if not present (using relevance as proxy for recency)
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
            
        # Add response column if not present (use relevance as proxy)
        if 'response' not in log_with_features.columns:
            log_with_features = log_with_features.withColumn('response', sf.col('relevance'))
        
        # Convert to pandas for sequence processing
        log_pd = log_with_features.select(
            'user_idx', 'item_idx', 'timestamp', 'price', 'response'
        ).toPandas()
        
        # Build item vocabulary
        unique_items = sorted(log_pd['item_idx'].unique())
        self.item_vocab = {item: idx for idx, item in enumerate(unique_items)}
        self.vocab_size = len(self.item_vocab)
        
        # Build user vocabulary
        unique_users = sorted(log_pd['user_idx'].unique())
        self.user_vocab = {user: idx for idx, user in enumerate(unique_users)}
        self.n_users = len(self.user_vocab)
        
        print(f"Vocabulary: {self.vocab_size} items, {self.n_users} users")
        
        # Group by user and sort by timestamp
        self.user_sequences = {}
        sequence_stats = {'total_sequences': 0, 'valid_sequences': 0, 'avg_length': 0}
        
        for user_idx, group in log_pd.groupby('user_idx'):
            # Sort by timestamp
            user_interactions = group.sort_values('timestamp')
            
            # Create sequence tuples: (item_idx, timestamp, price, response)
            sequence = []
            for _, row in user_interactions.iterrows():
                sequence.append((
                    int(row['item_idx']),
                    float(row['timestamp']),
                    float(row['price']),
                    float(row['response'])
                ))
            
            # Only keep sequences meeting minimum length requirement
            if len(sequence) >= self.min_sequence_length:
                # Truncate to maximum sequence length (keep most recent)
                if len(sequence) > self.sequence_length:
                    sequence = sequence[-self.sequence_length:]
                
                self.user_sequences[user_idx] = sequence
                sequence_stats['valid_sequences'] += 1
                sequence_stats['avg_length'] += len(sequence)
            
            sequence_stats['total_sequences'] += 1
        
        if sequence_stats['valid_sequences'] > 0:
            sequence_stats['avg_length'] /= sequence_stats['valid_sequences']
        
        print(f"Sequences curated: {sequence_stats['valid_sequences']}/{sequence_stats['total_sequences']} valid")
        print(f"Average sequence length: {sequence_stats['avg_length']:.2f}")
    
    def _get_sequence_features(self, user_idx: int) -> Optional[List[int]]:
        """
        Get item sequence for a user.
        
        Args:
            user_idx: User identifier
            
        Returns:
            List of item indices in temporal order, or None if no sequence
        """
        if user_idx not in self.user_sequences:
            return None
        
        sequence = self.user_sequences[user_idx]
        return [item_idx for item_idx, _, _, _ in sequence]
    
    def _get_weighted_sequence_features(self, user_idx: int) -> Optional[List[Tuple[int, float]]]:
        """
        Get weighted item sequence for a user with temporal decay.
        
        Args:
            user_idx: User identifier
            
        Returns:
            List of (item_idx, weight) tuples, or None if no sequence
        """
        if user_idx not in self.user_sequences:
            return None
        
        sequence = self.user_sequences[user_idx]
        weighted_sequence = []
        
        # Apply temporal decay - more recent items get higher weights
        for i, (item_idx, _, price, response) in enumerate(sequence):
            # Weight decreases exponentially with age (recent items have higher index)
            temporal_weight = self.temporal_decay ** (len(sequence) - 1 - i)
            
            # Also weight by response (purchased items get more weight)
            response_weight = 1.0 + response  # response is 0 or 1, so weight is 1.0 or 2.0
            
            final_weight = temporal_weight * response_weight
            weighted_sequence.append((item_idx, final_weight))
        
        return weighted_sequence
    
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
        
        # Apply revenue weighting: relevance = probability * (price^revenue_weight)
        revenue_weighted = recommendations_with_price.withColumn(
            'relevance',
            sf.col('relevance') * sf.pow(sf.col('price'), self.revenue_weight)
        )
        
        return revenue_weighted.select('user_idx', 'item_idx', 'relevance')
    
    def _apply_position_bias(self, recommendations_df: DataFrame) -> DataFrame:
        """
        Apply position bias with logarithmic discounting.
        
        Args:
            recommendations_df: DataFrame with user_idx, item_idx, relevance
            
        Returns:
            DataFrame with position-discounted relevance scores
        """
        # Add ranking within each user
        window = Window.partitionBy('user_idx').orderBy(sf.desc('relevance'))
        ranked_recs = recommendations_df.withColumn('rank', sf.row_number().over(window))
        
        # Apply logarithmic position discounting
        discounted_recs = ranked_recs.withColumn(
            'relevance',
            sf.col('relevance') / sf.log2(sf.col('rank') + 1)
        )
        
        return discounted_recs.select('user_idx', 'item_idx', 'relevance')
    
    def _random_recommendations(self, users: DataFrame, items: DataFrame, k: int) -> DataFrame:
        """
        Generate random recommendations as fallback.
        
        Args:
            users: User dataframe
            items: Item dataframe
            k: Number of recommendations per user
            
        Returns:
            DataFrame with random recommendations
        """
        # Cross join users and items
        recs = users.crossJoin(items)
        
        # Add random relevance scores
        recs = recs.withColumn('relevance', sf.rand(seed=self.seed))
        
        # Rank and filter top-k
        window = Window.partitionBy('user_idx').orderBy(sf.desc('relevance'))
        recs = recs.withColumn('rank', sf.row_number().over(window))
        recs = recs.filter(sf.col('rank') <= k).drop('rank')
        
        return recs.select('user_idx', 'item_idx', 'relevance')
    
    @abstractmethod
    def _train_model(self) -> None:
        """Train the sequence model using curated sequences."""
        pass
    
    @abstractmethod
    def _predict_next_items(self, user_sequences: List[List[int]], candidate_items: List[int]) -> np.ndarray:
        """
        Predict probabilities for candidate items given user sequences.
        
        Args:
            user_sequences: List of item sequences for each user
            candidate_items: List of candidate item indices
            
        Returns:
            Array of shape (n_users, n_items) with prediction probabilities
        """
        pass
    
    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None,
            item_features: Optional[DataFrame] = None) -> None:
        """Train the sequence recommender model."""
        if log is None:
            print("Warning: No interaction data available for training")
            return
        
        print("Curating sequences from interaction log...")
        self._curate_sequences(log, user_features, item_features)
        
        if len(self.user_sequences) == 0:
            print("Warning: No valid sequences found for training")
            return
        
        print("Training sequence model...")
        self._train_model()
        self.is_fitted = True
        print(f"Sequence model trained on {len(self.user_sequences)} user sequences")
    
    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        """Generate sequence-based recommendations for users."""
        if not self.is_fitted:
            print("Warning: Model not fitted. Using random recommendations.")
            return self._random_recommendations(users, items, k)
        
        # Get user and item lists
        users_pd = users.select('user_idx').toPandas()
        items_pd = items.select('item_idx').toPandas()
        
        user_list = users_pd['user_idx'].tolist()
        item_list = items_pd['item_idx'].tolist()
        
        # Get sequences for users
        user_sequences = []
        valid_users = []
        
        for user_idx in user_list:
            sequence = self._get_sequence_features(user_idx)
            if sequence is not None:
                # Convert item IDs to vocabulary indices
                vocab_sequence = []
                for item_idx in sequence:
                    if item_idx in self.item_vocab:
                        vocab_sequence.append(self.item_vocab[item_idx])
                
                if len(vocab_sequence) > 0:
                    user_sequences.append(vocab_sequence)
                    valid_users.append(user_idx)
        
        if len(user_sequences) == 0:
            print("Warning: No valid sequences found for users. Using random recommendations.")
            return self._random_recommendations(users, items, k)
        
        # Convert candidate items to vocabulary indices
        candidate_item_indices = []
        valid_items = []
        for item_idx in item_list:
            if item_idx in self.item_vocab:
                candidate_item_indices.append(self.item_vocab[item_idx])
                valid_items.append(item_idx)
        
        if len(candidate_item_indices) == 0:
            print("Warning: No valid candidate items. Using random recommendations.")
            return self._random_recommendations(users, items, k)
        
        # Get predictions
        try:
            predictions = self._predict_next_items(user_sequences, candidate_item_indices)
            
            # Create recommendations dataframe
            recommendations = []
            for i, user_idx in enumerate(valid_users):
                user_predictions = predictions[i]
                
                # Create user-item pairs with predictions
                for j, item_idx in enumerate(valid_items):
                    recommendations.append({
                        'user_idx': user_idx,
                        'item_idx': item_idx,
                        'relevance': float(user_predictions[j])
                    })
            
            if len(recommendations) == 0:
                print("Warning: No predictions generated. Using random recommendations.")
                return self._random_recommendations(users, items, k)
            
            # Convert to Spark DataFrame
            recommendations_df = pandas_to_spark(pd.DataFrame(recommendations))
            recommendations_df = recommendations_df.withColumn("user_idx", sf.col("user_idx").cast(LongType()))
            recommendations_df = recommendations_df.withColumn("item_idx", sf.col("item_idx").cast(LongType()))
            
            # Filter seen items if requested
            if filter_seen_items and log is not None and log.count() > 0:
                seen_pairs = log.select("user_idx", "item_idx")
                recommendations_df = recommendations_df.join(
                    seen_pairs,
                    on=["user_idx", "item_idx"],
                    how="left_anti"
                )
            
            # Apply revenue weighting if item features available
            if item_features is not None:
                recommendations_with_price = recommendations_df.join(
                    item_features.select('item_idx', 'price'),
                    on='item_idx',
                    how='left'
                )
                
                recommendations_df = recommendations_with_price.withColumn(
                    'relevance',
                    sf.col('relevance') * sf.pow(sf.col('price'), self.revenue_weight)
                ).select('user_idx', 'item_idx', 'relevance')
            
            # Rank and select top-k
            window = Window.partitionBy('user_idx').orderBy(sf.desc('relevance'))
            recommendations_df = recommendations_df.withColumn('rank', sf.row_number().over(window))
            recommendations_df = recommendations_df.filter(sf.col('rank') <= k).drop('rank')
            
            return recommendations_df.select('user_idx', 'item_idx', 'relevance')
            
        except Exception as e:
            print(f"Warning: Prediction failed with error: {e}. Using random recommendations.")
            return self._random_recommendations(users, items, k) 