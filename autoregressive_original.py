from collections import Counter, defaultdict
from typing import List, Sequence, Dict, Any, Optional, Tuple
import math
import json
import pickle

import pyspark.sql.functions as F
from pyspark.sql import Window # For window functions in Spark
from pyspark.sql.types import StructType, StructField, LongType, DoubleType # Ensure types are imported


class NGramAutoRegressiveRecommender:
    """
    Simple n‑gram (order‑n Markov) model for next‑item recommendation.
    Adapted for PySpark-based simulation environment.

    Parameters
    ----------
    n : int
        Order of the n‑gram (context length). n = 1 means context is the previous item.
    smoothing : float
        Add‑k (Laplace) smoothing parameter. smoothing = 1.0 corresponds to
        classic Laplace smoothing; 0 disables smoothing.
    revenue_weight : float, optional
        Weight for item prices in the scoring function. Defaults to 1.0.
    max_sequence_length : int, optional
        Maximum length of user interaction sequences to consider from history.
        If None or <= 0, all history is used. Defaults to 50.
    seed : int, optional
        Random seed for reproducibility, for API consistency (not actively used).
    """

    def __init__(self, n: int = 1, smoothing: float = 1.0, revenue_weight: float = 1.0, max_sequence_length: int = 50, seed: Optional[int] = None):
        if n < 1:
            raise ValueError("n (order) must be >= 1")
        if smoothing < 0:
            raise ValueError("smoothing must be non‑negative")
        self.n = n
        self.smoothing = smoothing
        self.revenue_weight = revenue_weight
        self.max_sequence_length = max_sequence_length if max_sequence_length and max_sequence_length > 0 else None
        self.seed = seed
        self.is_fitted = False

        # Mapping: context tuple -> Counter(next_item -> count)
        self.context_counts: Dict[Tuple[Any, ...], Counter] = defaultdict(Counter)

        # Vocabulary of items observed during training
        self.vocab: Counter = Counter()

        # Pre‑computed denominator for each context after fitting
        self._context_denoms: Dict[Tuple[Any, ...], float] = {}
        
        # Item prices: {item_id: price}
        self.item_prices: Dict[Any, float] = {}

    # --------------------------------------------------------------------- #
    #                               Training                                #
    # --------------------------------------------------------------------- #

    def fit(self, log: Any, user_features: Any, item_features: Any) -> None:
        """
        Fit the n‑gram model to historical interaction data from PySpark DataFrames.

        Parameters
        ----------
        log : pyspark.sql.DataFrame
            DataFrame of user interactions with columns ['user_idx', 'item_idx', 'relevance'].
        user_features : pyspark.sql.DataFrame
            DataFrame of user features (ignored by this model).
        item_features : pyspark.sql.DataFrame
            DataFrame of item features, must include ['item_idx', 'price'].
        """
        print("Schema of 'log' at the start of NGramAutoRegressiveRecommender.fit:")
        log.printSchema()

        # Defensive check and rename if 'user_id' exists and 'user_idx' does not
        log_columns = log.columns
        if 'user_id' in log_columns and 'user_idx' not in log_columns:
            print("Found 'user_id' but not 'user_idx' in log. Renaming 'user_id' to 'user_idx'.")
            log = log.withColumnRenamed('user_id', 'user_idx')
            print("Schema of 'log' after potential rename:")
            log.printSchema()
        elif 'user_idx' not in log_columns:
            # This case should ideally not be reached if the data generator is correct
            # or the above rename handles the common alias.
            print(f"CRITICAL: 'user_idx' is missing from log columns: {log_columns}. And 'user_id' is not present as an alternative.")
            # The subsequent code will likely fail if user_idx is truly missing.

        self.context_counts.clear()
        self.vocab.clear()
        self._context_denoms.clear()
        self.item_prices.clear()

        # Store item prices
        item_prices_collected = item_features.select("item_idx", "price").distinct().collect()
        for row in item_prices_collected:
            self.item_prices[row['item_idx']] = float(row['price'])

        # Process log to get sequences
        if self.max_sequence_length is not None:
            # Add row number to identify recent items if max_sequence_length is set
            log_with_total_count = log.withColumn("total_count", F.count("item_idx").over(Window.partitionBy("user_idx")))
            log_with_rn = log_with_total_count.withColumn("rn_asc", 
                F.row_number().over(Window.partitionBy("user_idx").orderBy("item_idx")))
            
            # Keep only rows that are part of the most recent 'max_sequence_length' items
            filtered_log = log_with_rn.filter(F.col("rn_asc") > (F.col("total_count") - self.max_sequence_length))
            
            user_sequences_df = filtered_log.groupBy("user_idx") \
                .agg(F.collect_list("item_idx").alias("sequence"))
        else:
            user_sequences_df = log.groupBy("user_idx") \
                .agg(F.collect_list("item_idx").alias("sequence"))

        sequences = [row['sequence'] for row in user_sequences_df.collect() if row['sequence']]

        # Train n-gram model
        for seq in sequences:
            if len(seq) == 0:
                continue
            self.vocab.update(seq)

            # Pad sequence for n-gram context
            padded_seq = [None] * (self.n - 1) + list(seq)
            
            # Extract n-grams
            for i in range(self.n - 1, len(padded_seq) - 1):
                context = tuple(padded_seq[i - self.n + 1 : i + 1])
                next_item = padded_seq[i + 1]
                self.context_counts[context][next_item] += 1

        # Pre-compute denominators for smoothing
        V = len(self.vocab)
        for ctx, counter in self.context_counts.items():
            denom = sum(counter.values()) + self.smoothing * V
            self._context_denoms[ctx] = denom

        self.is_fitted = True

    # --------------------------------------------------------------------- #
    #                             Inference                                 #
    # --------------------------------------------------------------------- #

    def _get_context(self, history: Sequence[Any]) -> Tuple[Any, ...]:
        """Get the n-gram context from history."""
        if not history:
            return tuple([None] * self.n)
        
        padding_needed = max(0, self.n - len(history))
        padded_history = [None] * padding_needed + list(history)
        return tuple(padded_history[-self.n:])


    def _predict_proba_single_candidate(self, context: Tuple[Any, ...], 
                                      candidate_item: Any) -> float:
        """Calculate probability for a single candidate item."""
        if not self.vocab:
            return 0.0

        counter = self.context_counts.get(context, Counter())
        count = counter.get(candidate_item, 0)
        
        V = len(self.vocab)
        denom = self._context_denoms.get(context, self.smoothing * V)
        
        if denom == 0:
            return 0.0
            
        return (count + self.smoothing) / denom

    def predict(self, log: Any, k: int, users: Any, items: Any, 
                user_features: Optional[Any] = None, 
                item_features: Optional[Any] = None, 
                filter_seen_items: bool = True) -> Any:
        """
        Generate recommendations for users.
        
        Parameters
        ----------
        log : pyspark.sql.DataFrame
            Interaction log with user_idx, item_idx, and relevance columns
        k : int
            Number of items to recommend
        users : pyspark.sql.DataFrame
            Users to recommend for
        items : pyspark.sql.DataFrame
            Candidate items
        user_features : pyspark.sql.DataFrame, optional
            User features (unused)
        item_features : pyspark.sql.DataFrame, optional
            Item features (prices used from self.item_prices)
        filter_seen_items : bool
            Whether to filter out already seen items
            
        Returns
        -------
        pyspark.sql.DataFrame
            Recommendations with columns ['user_idx', 'item_idx', 'relevance']
        """
        if not self.is_fitted:
            print("Model not fitted. Returning empty recommendations.")
            return self._create_empty_recommendations(users.sparkSession)

        # Get user histories
        user_histories = {}
        if log is not None:
            for user_id in users.select('user_idx').distinct().collect():
                user_id = user_id['user_idx']
                user_log = log.filter(F.col("user_idx") == user_id) \
                    .select("item_idx") \
                    .collect()
                
                history = [row['item_idx'] for row in user_log]
                if self.max_sequence_length:
                    history = history[-self.max_sequence_length:]
                user_histories[user_id] = history

        # Generate recommendations
        all_recs = []
        candidate_items = [row['item_idx'] for row in items.select('item_idx').distinct().collect()]
        
        for user_id in users.select('user_idx').distinct().collect():
            user_id = user_id['user_idx']
            history = user_histories.get(user_id, [])
            context = self._get_context(history)
            
            # Score each candidate item
            for item_id in candidate_items:
                if filter_seen_items and item_id in history:
                    continue
                    
                prob = self._predict_proba_single_candidate(context, item_id)
                price = self.item_prices.get(item_id, 0.0)
                score = price * prob * self.revenue_weight
                
                if score > 0:
                    all_recs.append({
                        'user_idx': user_id,
                        'item_idx': item_id,
                        'relevance': float(score)
                    })

        # Create recommendations DataFrame
        if not all_recs:
            return self._create_empty_recommendations(users.sparkSession)
            
        recs_df = users.sparkSession.createDataFrame(all_recs)
        
        # Rank and select top-k
        window_spec = Window.partitionBy("user_idx").orderBy(F.desc("relevance"))
        ranked_recs = recs_df.withColumn("rank", F.row_number().over(window_spec))
        top_k_recs = ranked_recs.filter(F.col("rank") <= k) \
            .select("user_idx", "item_idx", "relevance")
            
        return top_k_recs

    def _create_empty_recommendations(self, spark_session):
        """Create empty recommendations DataFrame with correct schema."""
        schema = StructType([
            StructField("user_idx", LongType(), True),
            StructField("item_idx", LongType(), True),
            StructField("relevance", DoubleType(), True)
        ])
        return spark_session.createDataFrame([], schema)

    # --------------------------------------------------------------------- #
    #                           Persistence                                 #
    # --------------------------------------------------------------------- #

    def save(self, path: str) -> None:
        """Serialize model to disk using pickle."""
        with open(path, "wb") as f:
            # Avoid trying to pickle Spark objects if any were accidentally stored
            # For this class, all members should be standard Python types
            pickle.dump(self.__dict__, f)

    def load(self, path: str) -> None:
        """Load model state from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.__dict__.update(state)

    # --------------------------------------------------------------------- #
    #                         Utility / Display                             #
    # --------------------------------------------------------------------- #

    def log_likelihood(self, sequences: Sequence[Sequence[Any]]) -> float:
        """
        Compute average log‑likelihood per token on given sequences.
        Used for debugging / evaluation.

        Returns
        -------
        float
            Mean log probability of the next items.
        """
        ll = 0.0
        n_tokens = 0
        V = len(self.vocab)
        if V == 0 and self.smoothing == 0: # Avoid division by zero
            return -float('inf') if sequences else 0.0


        for seq in sequences:
            if len(seq) == 0:
                continue
            
            padded_seq = [None] * (self.n -1) + list(seq) if self.n > 0 else list(seq)
            
            for i in range(self.n - 1, len(padded_seq) - 1):
                context = tuple(padded_seq[i - self.n + 1 : i + 1])
                next_item = padded_seq[i + 1]
                
                counter = self.context_counts.get(context, Counter())
                denom = self._context_denoms.get(context, self.smoothing * V) # Fallback if context not in _context_denoms
                if denom == 0: # Can happen if V=0 (no vocab) and smoothing=0
                    prob = 0.0
                else:
                    prob = (counter.get(next_item, 0) + self.smoothing) / denom
                
                ll += math.log(prob + 1e-12) # Fixed: Changed ‑ to -
                n_tokens += 1

        return ll / max(n_tokens, 1) if n_tokens > 0 else 0.0

    def __repr__(self):
        return f"NGramAutoRegressiveRecommender(n={self.n}, smoothing={self.smoothing}, revenue_weight={self.revenue_weight}, max_sequence_length={self.max_sequence_length})"

    # Deprecated/Internalized: predict_proba and original recommend
    # Kept for reference or potential internal use if refactored further.
    # The main API is now predict(self, user_id, candidate_items_df, k, log)

    def predict_proba(
        self,
        history: Sequence[Any],
        candidate_items: Optional[Sequence[Any]] = None,
    ) -> Dict[Any, float]:
        """
        Compute (smoothed) probabilities for candidate items given a history.
        This method is now primarily for internal use or detailed analysis.
        The main 'predict' method handles the simulation interaction.

        Parameters
        ----------
        history : list
            Ordered list of past items (most recent last).
        candidate_items : list, optional
            Items to score. If None, return probabilities for the entire vocab.

        Returns
        -------
        dict
            Mapping from item -> probability of being the next item.
        """
        if not self.vocab: # Model not fitted or no items seen
             # If candidate_items is None, behavior is undefined.
             # If candidate_items provided, all probs are 0.
            return {item: 0.0 for item in candidate_items} if candidate_items else {}


        ctx = self._get_context(history)
        
        # Define item set to score
        items_to_score = candidate_items if candidate_items is not None else list(self.vocab.keys())

        probs = {}
        for item in items_to_score:
            probs[item] = self._predict_proba_single_candidate(ctx, item)
            
        return probs