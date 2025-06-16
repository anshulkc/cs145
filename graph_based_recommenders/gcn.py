"""GCN‑based recommender system optimised for **total revenue**.

Fully implements the ``GraphBasedRecommenderBase`` interface, so it can be
instantiated directly by the analysis script.
"""
from __future__ import annotations

import math
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import networkx as nx
from tqdm import trange

# Relative import assumes this file lives in the same package as `base.py`.
from .base import GraphBasedRecommenderBase  # type: ignore

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################################################
# Low‑level GCN layers
###########################################################################

class GCNLayer(nn.Module):
    """Single Graph Convolution layer with ReLU, dropout, and optional BatchNorm."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float, use_bn: bool):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_dim) if use_bn else None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:  # noqa: D401
        x = self.linear(x)
        x = torch.sparse.mm(adj, x)
        if self.bn is not None:
            x = self.bn(x)
        x = F.relu(x)
        return self.dropout(x)

###########################################################################
# Backbone
###########################################################################

class GCNBackbone(nn.Module):
    """Multi‑layer GCN; returns user and item embeddings."""

    def __init__(
        self,
        n_users: int,
        n_items: int,
        embed_dim: int,
        hidden_dims: List[int],
        dropout: float,
        use_bn: bool,
    ) -> None:
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.emb_user = nn.Embedding(n_users, embed_dim)
        self.emb_item = nn.Embedding(n_items, embed_dim)
        nn.init.xavier_uniform_(self.emb_user.weight)
        nn.init.xavier_uniform_(self.emb_item.weight)

        dims = [embed_dim] + hidden_dims
        self.layers = nn.ModuleList(
            [GCNLayer(dims[i], dims[i + 1], dropout, use_bn) for i in range(len(dims) - 1)]
        )
        self.adj: Optional[torch.Tensor] = None

    def set_adj(self, adj: torch.Tensor) -> None:  # noqa: D401
        self.adj = adj

    def forward(self) -> Tuple[torch.Tensor, torch.Tensor]:  # noqa: D401
        if self.adj is None:
            raise ValueError("Adjacency matrix not set – call set_adj first")
        x = torch.cat([self.emb_user.weight, self.emb_item.weight], dim=0)
        residual = x  # LightGCN‑style residual
        for layer in self.layers:
            x = layer(x, self.adj)
        x = x + residual
        users, items = torch.split(x, [self.n_users, self.n_items], dim=0)
        return users, items

###########################################################################
# Revenue‑aware GCN‑based recommender
###########################################################################

class GCNRecommender(GraphBasedRecommenderBase):
    """Revenue‑weighted GCN recommender (drop‑in replacement)."""

    def __init__(
        self,
        embedding_dim: int = 128,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        graph_dropout: float = 0.1,
        learning_rate: float = 3e-3,
        weight_decay: float = 1e-4,
        epochs: int = 200,
        batch_size: int = 1024,
        negative_sampling_ratio: int = 4,
        negative_sampling_strategy: str = "price",  # "uniform" | "price"
        loss_type: str = "bpr",  # "bpr" | "bce" | "weighted_bce"
        early_stopping_patience: int = 25,
        self_loop_weight: float = 1.0,
        use_batch_norm: bool = True,
        edge_weight_strategy: str = "purchase_amount",
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        # Reproducibility
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        super().__init__(edge_weight_strategy=edge_weight_strategy, **kwargs)

        # Hyper‑params
        self.embed_dim = embedding_dim
        self.hidden_dims = hidden_dims or [128, 128]
        self.dropout = dropout
        self.graph_dropout = graph_dropout
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.neg_ratio = negative_sampling_ratio
        self.neg_strategy = negative_sampling_strategy
        self.loss_type = loss_type
        self.patience = early_stopping_patience
        self.self_loop_weight = self_loop_weight
        self.use_batch_norm = use_batch_norm

        # Runtime containers
        self.model: Optional[GCNBackbone] = None
        self._optim: Optional[optim.Optimizer] = None
        self.price_map: Dict[int, float] = {}
        self._item_sampling_weights: Optional[np.ndarray] = None
        self.positive_interactions: List[Tuple[int, int]] = []
        self._adj_cache: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------
    # Training / public API
    # ------------------------------------------------------------------
    def fit(self, df_train: Optional[pd.DataFrame] = None, df_val: Optional[pd.DataFrame] | None = None, **kwargs) -> None:  # noqa: D401
        """Train model. Extra keyword arguments are accepted for compatibility and ignored."""
        # Resolve df_train if it was supplied via an alternate keyword (e.g., `log`)
        if df_train is None:
            df_train = kwargs.get("log") or kwargs.get("data") or kwargs.get("interactions")
        if df_train is None:
            raise ValueError("`df_train` (or equivalent) must be provided to `fit()`.")

        self._build_graph(df_train)
        self.price_map = (
            df_train.groupby("item_id")["price"].mean().to_dict() if "price" in df_train else {}
        )
        self._refresh_item_sampling_weights()

        self.model = GCNBackbone(
            n_users=self.n_users,
            n_items=self.n_items,
            embed_dim=self.embed_dim,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
            use_bn=self.use_batch_norm,
        ).to(DEVICE)
        self._optim = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_metric = -float("inf")
        patience_counter = 0
        for epoch in trange(self.epochs, desc="Training", leave=False):
            self.model.train()
            adj = self._build_adjacency(dropout_rate=self.graph_dropout)
            self.model.set_adj(adj)
            _ = self._train_one_epoch()

            if df_val is not None and (epoch + 1) % 5 == 0:
                val_metric = self._evaluate_revenue_ndcg(df_val, k=10)
                if val_metric > best_metric:
                    best_metric = val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                if patience_counter >= self.patience:
                    break

    def predict(self, user_ids: List[int], item_ids: List[int]) -> np.ndarray:  # noqa: D401
        self.model.eval()
        with torch.no_grad():
            self.model.set_adj(self._get_cached_adj())
            u_emb, i_emb = self.model()
            return (u_emb[user_ids] * i_emb[item_ids]).sum(dim=-1).cpu().numpy()

    def recommend(self, user_id: int, top_k: int = 10) -> List[int]:  # noqa: D401
        self.model.eval()
        with torch.no_grad():
            self.model.set_adj(self._get_cached_adj())
            u_emb, i_emb = self.model()
            scores = (u_emb[user_id].unsqueeze(0) * i_emb).sum(dim=-1)
            # Revenue weighting
            if self.price_map:
                price_vec = torch.tensor(
                    [self.price_map.get(i, 1.0) for i in range(self.n_items)], device=DEVICE
                )
                scores = scores * price_vec
            return torch.topk(scores, top_k).indices.cpu().tolist()

    # ------------------------------------------------------------------
    # Abstract interface bridges
    # ------------------------------------------------------------------
    def _train_model(self, df_train: pd.DataFrame, df_val: Optional[pd.DataFrame] | None = None, *_, **__) -> None:
        self.fit(df_train, df_val)

    def _predict_link_probabilities(self, user_ids: List[int], item_ids: List[int], *_, **__) -> np.ndarray:
        return self.predict(list(user_ids), list(item_ids))

    # ------------------------------------------------------------------
    # Graph helpers
    # ------------------------------------------------------------------
    def _build_graph(self, df: pd.DataFrame) -> None:
        build_fn = getattr(super(), "_build_graph_from_interactions", None)
        if build_fn is None:
            build_fn = getattr(super(), "build_graph_from_interactions", None)
        if build_fn is not None:
            build_fn(df)
        else:
            # Fallback: construct simple bipartite graph if base class lacks helper
            self.n_users = int(df["user_id"].max()) + 1
            self.n_items = int(df["item_id"].max()) + 1
            self.user_nodes = list(range(self.n_users))
            self.item_nodes = list(range(self.n_users, self.n_users + self.n_items))
            self.graph = nx.Graph()
            for row in df.itertuples(index=False):
                u = int(row.user_id)
                v = self.n_users + int(row.item_id)
                weight = getattr(row, "price", 1.0)
                self.graph.add_edge(u, v, weight=weight)
        self.positive_interactions = [
            (row.user_id, self.n_users + row.item_id) for row in df.itertuples(index=False)
        ]

    def _iterate_edges(self):
        for u, v in self.graph.edges:
            if v < self.n_users:
                continue
            yield u, v, self.graph[u][v].get("weight", 1.0)

    def _build_adjacency(self, dropout_rate: float = 0.0) -> torch.Tensor:
        total = self.n_users + self.n_items
        rows: List[int] = []
        cols: List[int] = []
        vals: List[float] = []
        # Self‑loops
        for i in range(total):
            rows.append(i)
            cols.append(i)
            vals.append(self.self_loop_weight)
        # Bidirectional edges
        for u, v, w in self._iterate_edges():
            if dropout_rate and random.random() < dropout_rate:
                continue
            rows.extend([u, v])
            cols.extend([v, u])
            vals.extend([w, w])
        indices = torch.tensor([rows, cols], dtype=torch.long, device=DEVICE)
        values = torch.tensor(vals, dtype=torch.float32, device=DEVICE)
        adj = torch.sparse.FloatTensor(indices, values, (total, total))
        deg = torch.sparse.sum(adj, dim=1).to_dense() + 1e-8
        deg_inv_sqrt = torch.pow(deg, -0.5)
        diag_idx = torch.stack([torch.arange(total, device=DEVICE)] * 2)
        D_inv_sqrt = torch.sparse.FloatTensor(diag_idx, deg_inv_sqrt, (total, total))
        return torch.sparse.mm(torch.sparse.mm(D_inv_sqrt, adj), D_inv_sqrt)

    # ------------------------------------------------------------------
    # Sampling helpers
    # ------------------------------------------------------------------
    def _refresh_item_sampling_weights(self):
        if self.neg_strategy == "price" and self.price_map:
            prices = np.array([self.price_map.get(i, 1.0) for i in range(self.n_items)], dtype=float)
            self._item_sampling_weights = prices / prices.sum()
        else:
            self._item_sampling_weights = None

    def _sample_negative_items(self, users: np.ndarray, n_samples: int) -> np.ndarray:
        neg_items = np.empty(n_samples, dtype=int)
        for idx in range(n_samples):
            u = users[idx % len(users)]
            pos_set = self.graph[u].keys()
            while True:
                if self._item_sampling_weights is None:
                    j = np.random.randint(0, self.n_items)
                else:
                    j = np.random.choice(self.n_items, p=self._item_sampling_weights)
                j_global = self.n_users + j
                if j_global not in pos_set:
                    break
            neg_items[idx] = j_global
        return neg_items

    # ------------------------------------------------------------------
    # Loss and training helpers
    # ------------------------------------------------------------------
    def _revenue_weight(self, item_idx: torch.Tensor) -> torch.Tensor:
        if not self.price_map:
            return torch.ones_like(item_idx, dtype=torch.float32, device=DEVICE)
        prices = torch.tensor([
            self.price_map.get(int(i.cpu()) - self.n_users, 1.0) for i in item_idx
        ], dtype=torch.float32, device=DEVICE)
        return prices / (prices.mean() + 1e-8)

    def _compute_loss(self, u, pos_i, neg_j):
        self.model.set_adj(self._get_cached_adj())
        user_emb, item_emb = self.model()
        u_vec = user_emb[u]
        pos_vec = item_emb[pos_i - self.n_users]
        neg_vec = item_emb[neg_j - self.n_users]
        s_pos = (u_vec * pos_vec).sum(dim=-1)
        s_neg = (u_vec * neg_vec).sum(dim=-1)

        if self.loss_type == "bpr":
            w = self._revenue_weight(pos_i)
            return -(w * F.logsigmoid(s_pos - s_neg)).mean()

        logits = torch.cat([s_pos, s_neg])
        labels = torch.cat([torch.ones_like(s_pos), torch.zeros_like(s_neg)])
        if self.loss_type == "weighted_bce":
            weights = torch.cat([self._revenue_weight(pos_i), self._revenue_weight(neg_j)])
            return F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        else:
            return F.binary_cross_entropy_with_logits(logits, labels)

    def _train_one_epoch(self) -> float:
        users, items = zip(*self.positive_interactions)
        users = np.array(users)
        items = np.array(items)
        perm = np.random.permutation(len(users))
        users, items = users[perm], items[perm]
        n_batches = math.ceil(len(users) / self.batch_size)
        epoch_loss = 0.0
        for b in range(n_batches):
            start, end = b * self.batch_size, (b + 1) * self.batch_size
            u_batch = torch.tensor(users[start:end], dtype=torch.long, device=DEVICE)
            pos_batch = torch.tensor(items[start:end], dtype=torch.long, device=DEVICE)
            neg_batch = self._sample_negative_items(u_batch.cpu().numpy(), len(u_batch) * self.neg_ratio)
            neg_batch = torch.tensor(neg_batch, dtype=torch.long, device=DEVICE)

            u_exp = u_batch.repeat_interleave(self.neg_ratio)
            pos_exp = pos_batch.repeat_interleave(self.neg_ratio)

            self._optim.zero_grad()
            loss = self._compute_loss(u_exp, pos_exp, neg_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self._optim.step()
            epoch_loss += loss.item()
        return epoch_loss / n_batches

    # ------------------------------------------------------------------
    # Validation metric
    # ------------------------------------------------------------------
    def _evaluate_revenue_ndcg(self, df_val: pd.DataFrame, k: int = 10) -> float:
        ndcgs = []
        for uid, group in df_val.groupby("user_id"):
            pos_items = group["item_id"].tolist()
            gains = group["price"].tolist() if "price" in group else [1.0] * len(pos_items)
            recs = self.recommend(uid, top_k=k)
            dcg = 0.0
            for idx, itm in enumerate(recs):
                if itm in pos_items:
                    gain = gains[pos_items.index(itm)]
                    dcg += gain / math.log2(idx + 2)
            ideal_gains = sorted(gains, reverse=True)[:k]
            idcg = sum(g / math.log2(idx + 2) for idx, g in enumerate(ideal_gains)) or 1.0
            ndcgs.append(dcg / idcg)
        return float(np.mean(ndcgs))

    # ------------------------------------------------------------------
    # Cached adjacency for inference
    # ------------------------------------------------------------------
    def _get_cached_adj(self) -> torch.Tensor:
        if self._adj_cache is None:
            self._adj_cache = self._build_adjacency(dropout_rate=0.0)
        return self._adj_cache
