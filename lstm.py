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

    def fit(self, log: DataFrame, user_features: DataFrame, item_features: DataFrame) -> None:
        try:
            # 1. Validate inputs
            if log is None or user_features is None or item_features is None:
                raise ValueError("Input DataFrames cannot be None")

            # 2. Build user sequences of item indices
            if self.max_sequence_length is not None:
                log_counts = log.withColumn("total_count", F.count("item_idx").over(Window.partitionBy("user_idx")))
                log_rn = log_counts.withColumn("rn", F.row_number().over(Window.partitionBy("user_idx").orderBy("item_idx")))
                filtered = log_rn.filter(F.col("rn") > (F.col("total_count") - self.max_sequence_length))
                seq_df = filtered.groupBy("user_idx").agg(F.collect_list("item_idx").alias("sequence"))
            else:
                seq_df = log.groupBy("user_idx").agg(F.collect_list("item_idx").alias("sequence"))

            sequences = [row['sequence'] for row in seq_df.collect() if row['sequence']]
            if not sequences:
                logging.warning("No sequences available for training.")
                self.is_fitted = True
                return

            # 3. Determine vocab size from item_features
            item_ids = [r['item_idx'] for r in item_features.select("item_idx").collect()]
            vocab_size = max(item_ids) + 1

            # 4. Initialize embedding and output layer
            self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.input_size)
            self.fc = nn.Linear(self.hidden_size, vocab_size)

            # 5. Build price map for revenue weighting
            price_map = {r['item_idx']: r['price'] for r in item_features.select("item_idx", "price").collect()}

            # 6. Prepare BPR dataset with negative sampling
            class BPRDataset(Dataset):
                def __init__(self, seqs, max_len, n_neg, vocab, price_map):
                    self.seqs = seqs
                    self.max_len = max_len
                    self.n_neg = n_neg
                    self.vocab = vocab
                    self.price_map = price_map
                def __len__(self):
                    return len(self.seqs)
                def __getitem__(self, idx):
                    s = self.seqs[idx]
                    if len(s) > self.max_len + 1:
                        s = s[-(self.max_len + 1):]
                    inp = s[:-1]
                    pos = s[-1]
                    pad = self.max_len - len(inp)
                    inp = [0]*pad + inp
                    # negative sampling
                    negs = []
                    for _ in range(self.n_neg):
                        neg = random.randrange(self.vocab)
                        while neg in s:
                            neg = random.randrange(self.vocab)
                        negs.append(neg)
                    price = self.price_map.get(pos, 1.0)
                    return (torch.tensor(inp, dtype=torch.long),
                            torch.tensor(pos, dtype=torch.long),
                            torch.tensor(negs, dtype=torch.long),
                            torch.tensor(price, dtype=torch.float))

            dataset = BPRDataset(sequences, self.max_sequence_length, self.n_negatives, vocab_size, price_map)
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

            # 7. Setup device, optimizer, and BPR training
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(device)
            self.embedding.to(device)
            self.fc.to(device)
            optimizer = optim.Adam(list(self.model.parameters()) +
                                   list(self.embedding.parameters()) +
                                   list(self.fc.parameters()),
                                   lr=self.lr)

            # 8. BPR training loop
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0.0
                for inputs, pos_items, neg_items, prices in dataloader:
                    optimizer.zero_grad()
                    inputs = inputs.to(device)
                    pos_items = pos_items.to(device)
                    neg_items = neg_items.to(device)
                    prices = prices.to(device)
                    embeds = self.embedding(inputs)
                    outputs, (hn, _) = self.model(embeds)
                    hidden = hn[-1]  # (batch, hidden)
                    logits = self.fc(hidden)  # (batch, vocab)
                    # BPR loss
                    pos_scores = logits.gather(1, pos_items.unsqueeze(1)).squeeze(1)
                    neg_scores = logits.gather(1, neg_items)
                    diff = pos_scores.unsqueeze(1) - neg_scores
                    loss = - (prices.unsqueeze(1) * torch.log(torch.sigmoid(diff) + 1e-8)).mean()
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                logging.info(f"Epoch {epoch+1}/{self.epochs}, BPR Loss: {avg_loss:.4f}")

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

        # Build user sequences of last interactions
        if self.max_sequence_length is not None:
            log_counts = log.withColumn("total_count", F.count("item_idx").over(Window.partitionBy("user_idx")))
            log_rn = log_counts.withColumn("rn", F.row_number().over(Window.partitionBy("user_idx").orderBy("item_idx")))
            filtered = log_rn.filter(F.col("rn") > (F.col("total_count") - self.max_sequence_length))
            seq_df = filtered.groupBy("user_idx").agg(F.collect_list("item_idx").alias("sequence"))
        else:
            seq_df = log.groupBy("user_idx").agg(F.collect_list("item_idx").alias("sequence"))

        seq_map = {row['user_idx']: row['sequence'] for row in seq_df.collect()}

        # Determine device for model
        device = next(self.model.parameters()).device

        # Collect unique users
        user_rows = users.select("user_idx").distinct().collect()
        recs_list = []

        # Score items for each user
        for row in user_rows:
            user_id = row['user_idx']
            seq = seq_map.get(user_id, [])
            # Truncate and pad sequence
            if len(seq) > self.max_sequence_length:
                seq = seq[-self.max_sequence_length:]
            pad_len = self.max_sequence_length - len(seq)
            inp = [0] * pad_len + seq
            inp_tensor = torch.tensor([inp], dtype=torch.long, device=device)

            # Forward pass
            with torch.no_grad():
                embeds = self.embedding(inp_tensor)
                outputs, (hn, _) = self.model(embeds)
                hidden = hn[-1].squeeze(0)  # Last layer hidden state
                logits = self.fc(hidden)     # Scores for all items
                # Mask out seen items if required
                if filter_seen_items:
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    for it in seq:
                        if 0 <= it < logits.size(0):
                            mask[it] = True
                    logits = logits.masked_fill(mask, -float('inf'))
                # Select top-k items
                topk_vals, topk_idxs = torch.topk(logits, k)
                topk_vals = topk_vals.cpu().numpy()
                topk_idxs = topk_idxs.cpu().numpy()
            for idx, score in zip(topk_idxs, topk_vals):
                recs_list.append((user_id, int(idx), float(score)))

        # Create Spark DataFrame of recommendations
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