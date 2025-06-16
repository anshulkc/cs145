import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Any, Optional
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils as nn_utils
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType, DoubleType
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(1)  # (max_len,1,d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (seq_len, batch, d_model)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TransformerRecommender:
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 3,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_sequence_length: int = 50,
        seed: Optional[int] = None,
        epochs: int = 10,
        batch_size: int = 128,
        lr: float = 5e-4,
        n_negatives: int = 5,
    ):
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_sequence_length = max_sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.n_negatives = n_negatives
        self.seed = seed
        self.is_fitted = False

        # Placeholders for modules initialized in fit()
        self.embedding = None
        self.pos_encoder = None
        self.transformer_encoder = None
        self.fc = None
        self.user_embedding = None
        self.hidden_dropout = nn.Dropout(p=self.dropout)

    def fit(self, log: DataFrame, user_features: DataFrame, item_features: DataFrame) -> None:
        try:
            # 1. Build sequences ordered by timestamp or item_idx
            window_total = Window.partitionBy("user_idx")
            order_col = F.col("timestamp") if 'timestamp' in log.columns else F.col("item_idx")
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

            # 2. Determine vocab and user space
            item_ids = [r['item_idx'] for r in item_features.select("item_idx").collect()]
            vocab_size = max(item_ids) + 1
            # Derive number of users from provided user_features and log sequences
            # Use user_features if available to cover all users, else fallback to log
            if user_features is not None and 'user_idx' in user_features.columns:
                feature_ids = [r['user_idx'] for r in user_features.select("user_idx").collect()]
                max_feature_id = max(feature_ids) if feature_ids else -1
            else:
                max_feature_id = -1
            log_user_ids = [uid for (uid, _) in seq_pairs]
            max_log_id = max(log_user_ids) if log_user_ids else -1
            num_users = max(max_feature_id, max_log_id) + 1

            # 3. Initialize modules once
            if self.embedding is None or self.embedding.num_embeddings != vocab_size:
                self.embedding = nn.Embedding(vocab_size, self.d_model)
                self.pos_encoder = PositionalEncoding(self.d_model, self.dropout, self.max_sequence_length)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    dropout=self.dropout,
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
                self.fc = nn.Linear(self.d_model, vocab_size)
            if num_users > 0 and (self.user_embedding is None or self.user_embedding.num_embeddings != num_users):
                self.user_embedding = nn.Embedding(num_users, self.d_model)

            # 4. Price & popularity
            price_map = {r['item_idx']: r['price'] for r in item_features.select("item_idx", "price").collect()}
            from collections import Counter
            item_pop = Counter(it for (_, s) in seq_pairs for it in s)
            pop_weights = [float(item_pop.get(i, 1)) for i in range(vocab_size)]

            # 5. Dataset with sliding-window + popularity negs
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

            # 6. Setup device, optimizer, scheduler
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.embedding.to(device)
            self.pos_encoder.to(device)
            self.transformer_encoder.to(device)
            self.fc.to(device)
            if self.user_embedding is not None:
                self.user_embedding.to(device)
            optimizer = optim.Adam(
                list(self.embedding.parameters()) +
                list(self.transformer_encoder.parameters()) +
                list(self.fc.parameters()) +
                (list(self.user_embedding.parameters()) if self.user_embedding is not None else []),
                lr=self.lr,
            )
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1)

            # 7. Revenue-weighted sampled-softmax training
            for epoch in range(self.epochs):
                self.transformer_encoder.train()
                total_loss = 0.0
                for inputs, uids, pos_items, neg_items, prices in dataloader:
                    inputs = inputs.to(device)
                    uids = uids.to(device)
                    pos_items = pos_items.to(device)
                    neg_items = neg_items.to(device)
                    prices = prices.to(device)
                    optimizer.zero_grad()
                    # Embed + positional
                    src = self.embedding(inputs).permute(1, 0, 2)  # (seq, batch, d_model)
                    src = self.pos_encoder(src)
                    memory = self.transformer_encoder(src)  # (seq, batch, d_model)
                    hidden = memory[-1]  # (batch, d_model)
                    if self.user_embedding is not None:
                        ue = self.user_embedding(uids).squeeze(0)
                        hidden = hidden + ue
                    hidden = self.hidden_dropout(hidden)
                    logits = self.fc(hidden.unsqueeze(0))  # shape (1, vocab)
                    logits = logits.squeeze(0)  # shape (vocab,)
                    # Sampled-softmax
                    candidates = torch.cat([pos_items.unsqueeze(1), neg_items], dim=1)
                    cand_logits = torch.gather(logits, 1, candidates)
                    probs = torch.softmax(cand_logits, dim=1)
                    pos_probs = probs[:, 0]
                    loss = - (prices * torch.log(pos_probs + 1e-8)).mean()
                    loss.backward()
                    nn_utils.clip_grad_norm_(
                        list(self.embedding.parameters()) +
                        list(self.transformer_encoder.parameters()) +
                        list(self.fc.parameters()) +
                        (list(self.user_embedding.parameters()) if self.user_embedding is not None else []),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    total_loss += loss.item()
                avg_loss = total_loss / len(dataloader)
                logging.info(f"Epoch {epoch+1}/{self.epochs}, Revenue Loss: {avg_loss:.4f}")
                scheduler.step(avg_loss)

            self.is_fitted = True
            logging.info("Transformer revenue-aware training completed.")
        except Exception as e:
            logging.error(f"Error during fit: {e}")

    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None,
                item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        if not self.is_fitted:
            logging.warning("Model not fitted; training model on first predict call.")
            try:
                if user_features is None or item_features is None:
                    raise ValueError("Missing user_features or item_features for on-the-fly training.")
                self.fit(log, user_features, item_features)
            except Exception as e:
                logging.error(f"Error during on-the-fly training in predict: {e}")
                # Fallback to baseline recommendations
                top_items = items.select("item_idx").limit(k)
                recs = users.select("user_idx").distinct().crossJoin(top_items)
                recs = recs.withColumn("relevance", F.lit(1.0))
                return recs

        window_total = Window.partitionBy("user_idx")
        order_col = F.col("timestamp") if 'timestamp' in log.columns else F.col("item_idx")
        window_order = Window.partitionBy("user_idx").orderBy(order_col)
        if self.max_sequence_length:
            log_counts = log.withColumn("total_count", F.count("item_idx").over(window_total))
            log_rn = log_counts.withColumn("rn", F.row_number().over(window_order))
            filtered = log_rn.filter(F.col("rn") > (F.col("total_count") - self.max_sequence_length))
            seq_df = filtered.groupBy("user_idx").agg(F.collect_list("item_idx").alias("sequence"))
        else:
            seq_df = log.groupBy("user_idx").agg(F.collect_list("item_idx").alias("sequence"))
        seq_map = {row['user_idx']: row['sequence'] for row in seq_df.collect()}

        price_map = {r['item_idx']: r['price'] for r in items.select("item_idx", "price").collect()}
        device = next(self.embedding.parameters()).device
        recs_list = []
        for row in users.select("user_idx").distinct().collect():
            uid = row['user_idx']
            seq = seq_map.get(uid, [])
            if len(seq) > self.max_sequence_length:
                seq = seq[-self.max_sequence_length:]
            pad = self.max_sequence_length - len(seq)
            inp = [0] * pad + seq
            inp_tensor = torch.tensor([inp], dtype=torch.long, device=device)
            # Forward pass and scoring without tracking gradients
            with torch.no_grad():
                src = self.embedding(inp_tensor).permute(1, 0, 2)
                src = self.pos_encoder(src)
                memory = self.transformer_encoder(src)
                hidden = memory[-1].squeeze(0)  # now (d_model,)
                if self.user_embedding is not None:
                    # Safely embed user if within trained range, else use zero vector
                    if uid < self.user_embedding.num_embeddings:
                        ue = self.user_embedding(torch.tensor([uid], dtype=torch.long, device=device)).squeeze(0)
                    else:
                        ue = torch.zeros(self.d_model, device=device)
                    hidden = hidden + ue
                hidden = self.hidden_dropout(hidden)
                logits = self.fc(hidden.unsqueeze(0))  # shape (1, vocab)
                logits = logits.squeeze(0)  # shape (vocab,)
                if filter_seen_items:
                    mask = torch.zeros_like(logits, dtype=torch.bool)
                    for it in seq:
                        if 0 <= it < logits.size(0):
                            mask[it] = True
                    logits = logits.masked_fill(mask, -float('inf'))

                probs = torch.sigmoid(logits).detach().cpu().numpy()
                prices = np.array([price_map.get(i, 0.0) for i in range(len(probs))], dtype=np.float32)
                exp_rev = probs * prices
                if len(exp_rev) > k:
                    topk = np.argpartition(-exp_rev, k)[:k]
                    topk = topk[np.argsort(-exp_rev[topk])]
                else:
                    topk = np.argsort(-exp_rev)
                # ensure indices are Python ints
                topk = [int(x) for x in topk]
                for idx in topk:
                    recs_list.append((uid, idx, float(probs[idx])))

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