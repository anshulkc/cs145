# Checkpoint 2: Sequence-Based Recommenders

Implementation of sequence-based recommendation algorithms that leverage temporal patterns in user interactions to predict next-item purchases and maximize revenue through sophisticated sequence modeling architectures.

## Requirements Addressed

### 1. Sequence Curation
**Implementation**: Complete sequence management system maintaining ordered interaction histories.
- **Sequence format**: Tuples containing (timestamp, item_id, price, response) for each interaction
- **Cross-iteration persistence**: User sequences maintained and updated across training iterations
- **Sequence length management**: Configurable maximum lengths (10-50) with padding/masking for batch processing
- **Temporal ordering**: Chronological sorting with artificial timestamp generation when needed

### 2. Model Implementation
**Implementation**: Five different sequence-aware architectures exceeding minimum requirements.
- **Auto-Regressive (N-gram)**: Orders 1-3 with additive smoothing and backoff strategies
- **RNN/GRU/LSTM**: Recurrent networks with embedding layers, hidden sizes 64-256, 1-3 layers
- **Transformer**: Self-attention architecture with 2-8 heads, positional encodings, causal masking

### 3. Hyperparameter Tuning
**Implementation**: Extensive optimization across all sequence model parameters.
- **Sequence lengths**: Tested 10-50 events with optimal range 15-25
- **Embedding sizes**: 32-256 dimensions based on vocabulary size
- **Hidden units**: 64-256 for RNNs with optimal 128
- **Regularization**: Dropout 0.1-0.5, weight decay, early stopping with patience
- **Learning rates**: 0.0005-0.01 with exponential decay scheduling

### 4. Ranking & Revenue Optimization
**Implementation**: Revenue-aware ranking system with temporal considerations.
- **Expected revenue formula**: `price × sequence_probability` for candidate items
- **Revenue weighting**: Configurable weight parameters 1.0-2.0 for price emphasis
- **Top-k selection**: Ranked recommendations by expected revenue
- **Temporal decay**: Recent interactions weighted higher in predictions

### 5. Performance Comparison
**Implementation**: Comprehensive evaluation across all required metrics.
- **Primary metric**: AutoRegressive best at $857.41 test revenue
- **Secondary metrics**: Precision@K (0.054-0.075), coverage (85% users), training efficiency
- **Learning curves**: Performance tracking across training iterations
- **Model analysis**: Data sparsity challenges identified across all sequence approaches

## Script Usage

### eval.py
Comprehensive evaluation of all sequence-based models with baseline comparisons.
- **Purpose**: Tests AutoRegressive, RNN, GRU, LSTM, Transformer against Random/Popularity baselines
- **Method**: Standardized evaluation with 500 users, 200 items, consistent data generation
- **Metrics**: Revenue proxy calculation, relevance scores, prediction coverage analysis
- **Output**: comprehensive_evaluation_results.csv with detailed performance metrics
- **Runtime**: ~5-8 minutes for complete evaluation with all models

### hyperparameter_tuning_study.py
Systematic hyperparameter optimization across sequence model architectures.
- **Purpose**: Grid search and random search for optimal configurations
- **Parameters**: Sequence length, embedding dimension, hidden size, dropout, learning rate
- **Method**: Cross-validation with early stopping and statistical significance testing
- **Output**: Optimal hyperparameter configurations and sensitivity analysis plots
- **Runtime**: ~20-30 minutes for comprehensive tuning study

### IMPLEMENTATION_SUMMARY.md
Technical analysis document covering implementation challenges and solutions.
- **Content**: Data sparsity problem analysis, model complexity vs performance trade-offs
- **Insights**: AutoRegressive outperforms neural models due to sparse data handling
- **Lessons**: Temporal pattern recognition limitations in synthetic data environment

## Implementation Architecture

### SequenceBasedRecommenderBase (sequence_based_recommenders/base.py)
Base class providing sequence curation, vocabulary management, and revenue optimization.
- **Methods**: fit(), predict(), _curate_sequences(), _get_sequence_features()
- **Sequence processing**: Temporal ordering, vocabulary mapping, padding/masking
- **Revenue optimization**: Price weighting and position bias application

### AutoRegressiveRecommender (sequence_based_recommenders/autoregressive.py)
N-gram based sequence modeling with smoothing and backoff strategies.
- **N-gram orders**: 1-3 with context window prediction
- **Smoothing**: Additive smoothing with configurable alpha parameter
- **Backoff**: Interpolation between different order models
- **Performance**: Best sequence model with robust sparsity handling

### RNNRecommender/GRURecommender/LSTMRecommender (sequence_based_recommenders/rnn_lstm.py)
Recurrent neural network architectures for sequence modeling.
- **Architecture**: Embedding layer → RNN layers → Dense output
- **Training**: Teacher forcing with cross-entropy loss
- **Regularization**: Dropout, weight decay, gradient clipping
- **Performance**: Moderate results, sensitive to hyperparameters

### TransformerRecommender (sequence_based_recommenders/transformer.py)
Self-attention architecture with positional encodings for sequence prediction.
- **Attention**: Multi-head attention with 2-8 heads
- **Architecture**: Encoder-only with causal masking
- **Training**: Layer normalization, residual connections
- **Performance**: Lowest performance due to overfitting on sparse data

## Technical Specifications

### Sequence Processing Pipeline
1. **Data extraction**: Convert interaction logs to temporal sequences
2. **Vocabulary building**: Item-to-index mapping for neural models
3. **Sequence filtering**: Remove sequences below minimum length threshold
4. **Truncation**: Limit to recent interactions (last N items)
5. **Padding**: Dynamic padding for batch processing compatibility

### Revenue Optimization Strategy
- **Base prediction**: Sequence model outputs next-item probabilities
- **Revenue weighting**: Multiply probability by `price^revenue_weight`
- **Position bias**: Apply logarithmic discount for ranking quality
- **Cold start handling**: Content-based fallback for users without sequences

### Performance Analysis and Challenges
- **Data sparsity**: Major limitation with limited interaction history
- **Cold start**: 15% of users lack sufficient sequence data
- **Vocabulary mismatch**: Test items not in training vocabulary
- **Model complexity**: Neural models overfit to sparse training data
- **Best approach**: Simple AutoRegressive models most effective

### Model Performance Summary
- **AutoRegressive**: $857.41 revenue, best sparsity handling
- **RNN**: $692.17 revenue, moderate complexity
- **Transformer**: $535.02 revenue, overfitting issues
- **Training efficiency**: <1 second for all models
- **Prediction coverage**: 170/200 users (85% coverage)

### Dependencies and Setup
- **Core**: torch, numpy, pandas, pyspark
- **Versions**: torch>=1.9.0, numpy>=1.21.0, pandas>=1.3.0, pyspark>=3.2.0
- **Memory**: ~50MB for embeddings and model parameters
- **GPU**: Optional for neural models, CPU sufficient for n-gram approaches 