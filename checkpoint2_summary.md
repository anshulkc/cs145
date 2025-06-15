## Task

1. **Curate Sequences**:
   - Maintain an ordered list of each user's historical interactions across training iterations.
   - Each element should minimally contain `(timestamp, item_id, price, response)`.
2. **Implement at least THREE sequence-aware recommenders** chosen from:
   - **Auto-Regressive (AR) Models** (e.g., n-gram autoregressive models)
   - **Recurrent Neural Networks (RNN/GRU)**
   - **Long Short-Term Memory (LSTM)**
   - **Transformer / Self-Attention architectures**
3. **Hyperparameter Tuning**: Experiment with different sequence lengths, embedding sizes, hidden units, regularization (dropout, L2, weight decay), and learning rates.
4. **Ranking & Revenue Optimization**: Output a score (logit/probability) per candidate item, then rank by:  
   `expected_revenue = price × probability`  
   Use this for top-`k` recommendation.
5. **Compare Performance** across all implemented models using the platform metrics.


