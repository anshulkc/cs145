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
            DataFrame of user interactions with columns ['user_idx', 'item_idx', 'timestamp'].
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

        # Define window spec for ordering interactions by user and timestamp
        user_window_spec = Window.partitionBy("user_idx").orderBy("timestamp")

        # Process log to get sequences
        if self.max_sequence_length is not None:
            # Add row number to identify recent items if max_sequence_length is set
            log_with_total_count = log.withColumn("total_count", F.count("item_idx").over(Window.partitionBy("user_idx")))
            log_with_rn = log_with_total_count.withColumn("rn_asc", F.row_number().over(user_window_spec))
            
            # Keep only rows that are part of the most recent 'max_sequence_length' items
            filtered_log = log_with_rn.filter(F.col("rn_asc") > (F.col("total_count") - self.max_sequence_length))
            
            user_sequences_df = filtered_log.groupBy("user_idx") \
                                .agg(F.sort_array(F.collect_list(F.struct("timestamp", "item_idx"))).alias("interactions")) \
                                .select(F.col("interactions.item_idx").alias("sequence"))
        else: # Use all history
            user_sequences_df = log.groupBy("user_idx") \
                                .agg(F.sort_array(F.collect_list(F.struct("timestamp", "item_idx"))).alias("interactions")) \
                                .select(F.col("interactions.item_idx").alias("sequence"))

        sequences = [row['sequence'] for row in user_sequences_df.collect() if row['sequence']]


        # Original fitting logic using the extracted sequences
        for seq in sequences:
            if len(seq) == 0:
                continue
            self.vocab.update(seq)

            # The n-gram model's context has length 'self.n'.
            # To form a context of length 'self.n' and predict the next item,
            # we need at least 'self.n' items in the history for that context,
            # plus one more item to be predicted.
            # Example: if n=1 (bigram: P(item_k | item_k-1)), context is 1 item.
            # We pad with 'None's to handle sequence starts.
            # A context (item_0) predicts item_1. padded_seq = [None]*(n-1) + seq
            # For n=1, padded_seq = seq. Iteration: (n-1) to len-1. context is padded_seq[i-n+1 : i+1]
            # if n=1: padded_seq[i:i+1] e.g. (seq[i])
            # next_item is padded_seq[i+1] e.g. seq[i+1]
            
            # Padding to handle start of sequences consistently with original logic
            # Context has length self.n.
            # (item_{t-n}, ..., item_{t-1}) -> item_t
            padded_seq = [None] * (self.n -1) + list(seq) if self.n > 0 else list(seq) # Ensure correct padding for n=0 logic if it existed. Here n>=1
            
            # Iterate to form (context, next_item) pairs
            # We need i to go up to len(padded_seq) - 2 to have a next_item
            # The context ends at index i. The next_item is at i+1.
            # Context starts at i - self.n + 1.
            # Range for i: self.n - 1 to len(padded_seq) - 2
            for i in range(self.n - 1, len(padded_seq) - 1):
                context = tuple(padded_seq[i - self.n + 1 : i + 1]) # Context of length n
                next_item = padded_seq[i + 1]
                self.context_counts[context][next_item] += 1
                
        # Pre‑compute denominators for speed at inference
        V = len(self.vocab)
        if V == 0 and self.smoothing > 0: # Handle empty vocab case to avoid division by zero if only smoothing is non-zero
             # Denominators will be self.smoothing * V, effectively V cannot be 0 here if smoothing >0
             # If V is 0, all probs should be 0 unless smoothing implies some default.
             # Let's ensure V is at least 1 for smoothing calculation if smoothing > 0 and V=0 initially.
             # This edge case (no items seen at all) means prediction is not really possible.
             # Probabilities will be 0 if V=0 as calculated in predict_proba.
             pass


        for ctx, counter in self.context_counts.items():
            # Denominator: sum of counts for this context + k * |Vocabulary|
            denom = sum(counter.values()) + self.smoothing * V 
            self._context_denoms[ctx] = denom
        
        # For contexts not seen, denom will be calculated on the fly in predict_proba
        # as (0 + self.smoothing * V) if that context is queried.

    # --------------------------------------------------------------------- #
    #                             Inference                                 #
    # --------------------------------------------------------------------- #

    def _get_context(self, history: Sequence[Any]) -> Tuple[Any, ...]:
        """Return the length‑n context tuple derived from the given history."""
        # If history is shorter than n, pad with Nones at the beginning.
        # Example: n=3. history=[item1, item2]. Context should be (None, item1, item2)
        # Example: n=3. history=[item1]. Context should be (None, None, item1)
        if not history:
            return tuple([None] * self.n)
        
        padding_needed = max(0, self.n - len(history))
        padded_history_list = [None] * padding_needed + list(history)
        return tuple(padded_history_list[-self.n:])


    def _predict_proba_single_candidate(self, history_context: Tuple[Any, ...], candidate_item: Any) -> float:
        """Compute smoothed probability for a single candidate item given a history context."""
        if not self.vocab: # Model not fitted or no items seen
            return 0.0

        counter = self.context_counts.get(history_context, Counter())
        count = counter.get(candidate_item, 0)
        
        V = len(self.vocab)
        # Denominator for this context. If context was seen, it's precomputed.
        # If context was not seen, its count sum is 0. Denom becomes smoothing * V.
        denom = self._context_denoms.get(history_context, self.smoothing * V)

        if denom == 0: # Can happen if V=0 and smoothing=0
            return 0.0
            
        prob = (count + self.smoothing) / denom
        return prob

    def predict(self, log: Any, k: int, users: Any, items: Any, user_features: Optional[Any] = None, item_features: Optional[Any] = None, filter_seen_items: bool = True, **kwargs: Any) -> Any:
        """
        Rank candidate items by expected revenue and return top-k recommendations for multiple users.
        Expected revenue = price * P(next_item | history) * revenue_weight.

        Parameters
        ----------
        log : pyspark.sql.DataFrame
            Full interaction log, used to get each user's most recent history for context.
        k : int
            Number of items to return per user.
        users : pyspark.sql.DataFrame
            DataFrame of user_idx for whom to make predictions.
        items : pyspark.sql.DataFrame
            DataFrame of candidate items with 'item_idx' (all items to consider for recommendation).
        user_features : pyspark.sql.DataFrame, optional
            User features (unused by this model in predict, but part of API).
        item_features : pyspark.sql.DataFrame, optional
            Item features (prices are used from self.item_prices, which is filled during fit).
        filter_seen_items : bool, optional
            Whether to filter out items already seen by the user. Defaults to True.
        **kwargs : dict
            Additional arguments (unused).

        Returns
        -------
        pyspark.sql.DataFrame
            Top-k recommendations with columns ['user_idx', 'item_idx', 'relevance'].
        """
        spark_session = users.sparkSession

        if not self.vocab: # No data seen during fit
            print("NGramAutoRegressiveRecommender: Vocab is empty. Returning empty recommendations.")
            schema = StructType([
                StructField("user_idx", LongType(), True),
                StructField("item_idx", LongType(), True),
                StructField("relevance", DoubleType(), True)
            ])
            return spark_session.createDataFrame([], schema)

        user_ids_list = [row['user_idx'] for row in users.select('user_idx').distinct().collect()]
        
        candidate_items_collected_rows = items.select("item_idx").distinct().collect()
        candidate_item_ids = [row['item_idx'] for row in candidate_items_collected_rows]

        all_recs_data = []

        for user_id in user_ids_list:
            user_history_sequence: List[Any] = []
            if log:
                # Filter log for the current user and order by timestamp
                user_log_df = log.filter(F.col("user_idx") == user_id).orderBy(F.col("timestamp"))
                
                # Collect item_idx to form the sequence
                full_user_sequence_from_log = [row['item_idx'] for row in user_log_df.select("item_idx").collect()]

                if self.max_sequence_length is not None:
                    user_history_sequence = full_user_sequence_from_log[-self.max_sequence_length:]
                else:
                    user_history_sequence = full_user_sequence_from_log
            
            history_context = self._get_context(user_history_sequence)
            
            user_recommendations = []
            for item_id_candidate in candidate_item_ids:
                prob = self._predict_proba_single_candidate(history_context, item_id_candidate)
                price = self.item_prices.get(item_id_candidate, 0.0) # Default to 0 if price not found
                
                score = price * prob * self.revenue_weight
                if score > 0 : # Only consider items with a positive score
                    user_recommendations.append({'user_idx': user_id, 'item_idx': item_id_candidate, 'relevance': float(score)})
            
            all_recs_data.extend(user_recommendations)

        if not all_recs_data:
            print("NGramAutoRegressiveRecommender: No recommendations generated. Returning empty DataFrame.")
            schema = StructType([
                StructField("user_idx", LongType(), True),
                StructField("item_idx", LongType(), True),
                StructField("relevance", DoubleType(), True)
            ])
            return spark_session.createDataFrame([], schema)

        # Create Spark DataFrame from the collected recommendations
        recs_df_schema = StructType([
            StructField("user_idx", LongType(), True),
            StructField("item_idx", LongType(), True),
            StructField("relevance", DoubleType(), True)
        ])
        recs_df = spark_session.createDataFrame(all_recs_data, schema=recs_df_schema)

        # Filter seen items if requested
        if filter_seen_items and log is not None:
            seen_items = log.select("user_idx", "item_idx").distinct()
            recs_df = recs_df.join(
                seen_items,
                on=["user_idx", "item_idx"],
                how="left_anti"
            )

        # Rank items by relevance for each user and select top-k
        window_spec = Window.partitionBy("user_idx").orderBy(F.desc("relevance"))
        ranked_recs_df = recs_df.withColumn("rank", F.row_number().over(window_spec))
        
        top_k_recs_df = ranked_recs_df.filter(F.col("rank") <= k).select("user_idx", "item_idx", "relevance")
        
        return top_k_recs_df

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