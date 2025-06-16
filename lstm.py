import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional, Tuple, Sequence
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType, DoubleType
import logging
from torch.utils.data import Dataset, DataLoader
import random
from collections import Counter
import numpy as np
import torch.nn.utils as nn_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO)

class LSTMRecommender:
    def __init__(self, input_size: int = 128, hidden_size: int = 256, num_layers: int = 2,
                 dropout: float = 0.3, max_sequence_length: int = 100, seed: Optional[int] = 42,
                 epochs: int = 10, batch_size: int = 128, lr: float = 5e-4, n_negatives: int = 5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_negatives = n_negatives
        self.seed = seed
        self.is_fitted = False

        # Initialize LSTM model
        self.model = nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size, 
                             num_layers=self.num_layers, dropout=self.dropout, 
                             batch_first=True)
        # Placeholders for embeddings and layers, initialized in fit()
        self.embedding = None
        self.fc = None
        self.user_embedding = None
        # Dropout for hidden states
        self.hidden_dropout = nn.Dropout(p=self.dropout)

    def fit(self, log: DataFrame, user_features: DataFrame, item_features: DataFrame) -> None:
        try:
            # 1. Sequence construction (ordered by timestamp if available)
            window_total = Window.partitionBy("user_idx")
            if 'timestamp' in log.columns:
                order_col = F.col("timestamp")
            else:
                order_col = F.col("item_idx")
            window_order = Window.partitionBy("user_idx").orderBy(order_col)
            if self.max_sequence_length:
                log_counts = log.withColumn("total_count", F.count("item_idx").over(window_total))
                log_rn = log_counts.withColumn("rn", F.row_number().over(window_order))
                filtered = log_rn.filter(F.col("rn") > (F.col("total_count") - self.max_sequence_length))
                seq_df = filtered.groupBy("user_idx").agg(F.collect_list("item_idx").alias("sequence"))
            else:
                seq_df = log.groupBy("user_idx").agg(F.collect_list("item_idx").alias("sequence"))

            seq_pairs = [(row['user_idx'], row['sequence']) for row in seq_df.collect() if row['sequence']]
            if not seq_pairs:
                logging.warning("No sequences available for training.")
                self.is_fitted = True
                return
            sequences = [s for (_, s) in seq_pairs]

            # 2. Vocabulary and user space
            item_ids = [r['item_idx'] for r in item_features.select("item_idx").collect()]
            vocab_size = max(item_ids) + 1
            # Derive number of users from user_features to cover all users
            user_id_rows = user_features.select("user_idx").distinct().collect()
            user_ids = [r['user_idx'] for r in user_id_rows]
            num_users = max(user_ids) + 1 if user_ids else 0

            # 3. Initialize embeddings and layers once
            if self.embedding is None or self.embedding.num_embeddings != vocab_size:
                self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.input_size)
                self.fc = nn.Linear(self.hidden_size, vocab_size)
            if num_users > 0 and (self.user_embedding is None or self.user_embedding.num_embeddings != num_users):
                self.user_embedding = nn.Embedding(num_embeddings=num_users, embedding_dim=self.hidden_size)

            # 4. Price and popularity distributions
            price_map = {r['item_idx']: r['price'] for r in item_features.select("item_idx", "price").collect()}
            item_pop = Counter(it for s in sequences for it in s)
            pop_weights = [float(item_pop.get(i, 1)) for i in range(vocab_size)]

            # 5. Sliding-window BPR dataset with popularity-based negatives
            class BPRDataset(Dataset):
                def __init__(self, seq_pairs, max_len, n_neg, vocab, price_map, pop_weights):
                    self.samples = []
                    self.pop_weights = pop_weights
                    for uid, s in seq_pairs:
                        for i in range(1, len(s)):
                            history = s[max(0, i - max_len):i]
                            pad = max_len - len(history)
                            inp = [0] * pad + history
                            pos_item = s[i]
                            negs = []
                            while len(negs) < n_neg:
                                neg = random.choices(range(vocab), weights=self.pop_weights, k=1)[0]
                                if neg not in s:
                                    negs.append(neg)
                            price = price_map.get(pos_item, 1.0)
                            self.samples.append((inp, uid, pos_item, negs, price))
                def __len__(self):
                    return len(self.samples)
                def __getitem__(self, idx):
                    inp, uid, pos_item, negs, price = self.samples[idx]
                    return (
                        torch.tensor(inp, dtype=torch.long),
                        torch.tensor(uid, dtype=torch.long),
                        torch.tensor(pos_item, dtype=torch.long),
                        torch.tensor(negs, dtype=torch.long),
                        torch.tensor(price, dtype=torch.float),
                    )

            dataset = BPRDataset(seq_pairs, self.max_sequence_length, self.n_negatives,
                                 vocab_size, price_map, pop_weights)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # 6. Setup device and optimizer
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.embedding.to(device)
            self.fc.to(device)
            self.user_embedding.to(device)
            optimizer = optim.Adam(
                list(self.model.parameters()) +
                list(self.embedding.parameters()) +
                list(self.fc.parameters()) +
                list(self.user_embedding.parameters()),
                lr=self.lr
            )
            # Scheduler to reduce LR on plateau of BPR loss
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

            # 7. BPR training loop
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0.0
                for inputs, uids, pos_items, neg_items, prices in dataloader:
                    inputs = inputs.to(device)
                    uids = uids.to(device)
                    pos_items = pos_items.to(device)
                    neg_items = neg_items.to(device)
                    prices = prices.to(device)
                    optimizer.zero_grad()
                    embeds = self.embedding(inputs)
                    outputs, (hn, _) = self.model(embeds)
                    hidden = hn[-1]  # (batch, hidden)
                    user_embs = self.user_embedding(uids)
                    hidden = hidden + user_embs
                    hidden = self.hidden_dropout(hidden)
                    logits = self.fc(hidden)  # (batch, vocab)
                    # sampled-softmax over pos + negs
                    candidates = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)  # (batch, n_neg+1)
                    cand_logits = torch.gather(logits, 1, candidates)
                    probs = torch.softmax(cand_logits, dim=1)
                    pos_probs = probs[:, 0]
                    # revenue-weighted loss: maximize p(pos)*price -> minimize -price*log(p)
                    loss = - (prices * torch.log(pos_probs + 1e-8)).mean()
                    loss.backward()
                    nn_utils.clip_grad_norm_(list(self.model.parameters()) +
                                             list(self.embedding.parameters()) +
                                             list(self.fc.parameters()) +
                                             list(self.user_embedding.parameters()),
                                             max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                logging.info(f"Epoch {epoch+1}/{self.epochs}, Revenue Loss: {avg_loss:.4f}")
                scheduler.step(avg_loss)

            self.is_fitted = True
            logging.info("Model BPR training completed.")
        except Exception as e:
            logging.error(f"Error during fitting: {e}")

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None,
                item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        # Only predict if fitted
        if not self.is_fitted:
            logging.warning("Model not fitted; returning empty recommendations.")
            return self._create_empty_recommendations(users.sparkSession)

        # Build and order sequences as before
        window_total = Window.partitionBy("user_idx")
        if 'timestamp' in log.columns:
            order_col = F.col("timestamp")
        else:
            order_col = F.col("item_idx")
        window_order = Window.partitionBy("user_idx").orderBy(order_col)
        if self.max_sequence_length:
            log_counts = log.withColumn("total_count", F.count("item_idx").over(window_total))
            log_rn = log_counts.withColumn("rn", F.row_number().over(window_order))
            filtered = log_rn.filter(F.col("rn") > (F.col("total_count") - self.max_sequence_length))
            seq_df = filtered.groupBy("user_idx").agg(F.collect_list("item_idx").alias("sequence"))
        else:
            seq_df = log.groupBy("user_idx").agg(F.collect_list("item_idx").alias("sequence"))

        seq_map = {row['user_idx']: row['sequence'] for row in seq_df.collect()}

        # Build price map once from items DataFrame
        price_map = {r['item_idx']: r['price'] for r in items.select("item_idx", "price").collect()}

        # Prepare for scoring
        device = next(self.model.parameters()).device
        user_rows = users.select("user_idx").distinct().collect()
        recs_list = []

        for row in user_rows:
            user_id = row['user_idx']
            seq = seq_map.get(user_id, [])
            # Truncate and pad
            if len(seq) > self.max_sequence_length:
                seq = seq[-self.max_sequence_length:]
            pad_len = self.max_sequence_length - len(seq)
            inp = [0] * pad_len + seq
            inp_tensor = torch.tensor([inp], dtype=torch.long, device=device)

            # Forward pass
            with torch.no_grad():
                embeds = self.embedding(inp_tensor)
                outputs, (hn, _) = self.model(embeds)
                hidden = hn[-1].squeeze(0)
                # Add user embedding if available
                if self.user_embedding is not None:
                    user_emb = self.user_embedding(torch.tensor([user_id], dtype=torch.long, device=device)).squeeze(0)
                    hidden = hidden + user_emb
                logits = self.fc(hidden)  # (vocab,)
                # Mask seen items
                if filter_seen_items:
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    for it in seq:
                        if 0 <= it < logits.size(0):
                            mask[it] = True
                    logits = logits.masked_fill(mask, -float('inf'))

                # Compute purchase probabilities
                probs = torch.sigmoid(logits).cpu().numpy()
                # Compute expected revenue = p * price
                prices = np.array([price_map.get(i, 0.0) for i in range(len(probs))], dtype=np.float32)
                exp_rev = probs * prices
                # Get top-k indices by expected revenue
                if len(exp_rev) > k:
                    topk = np.argpartition(-exp_rev, k)[:k]
                    # Sort these topk by exp_rev
                    topk = topk[np.argsort(-exp_rev[topk])]
                else:
                    topk = np.argsort(-exp_rev)
                # Append to recommendations list
                for idx in topk:
                    recs_list.append((user_id, int(idx), float(probs[idx])))

        # Create DataFrame
        schema = StructType([
            StructField("user_idx", LongType(), True),
            StructField("item_idx", LongType(), True),
            StructField("relevance", DoubleType(), True)
        ])
        return users.sparkSession.createDataFrame(recs_list, schema)

    def _create_empty_recommendations(self, spark_session):
        schema = StructType([
            StructField("user_idx", LongType(), True),
            StructField("item_idx", LongType(), True),
            StructField("relevance", DoubleType(), True)
        ])
        return spark_session.createDataFrame([], schema) 