import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Base class for content-based recommenders

import numpy as np
import pandas as pd
from typing import Optional
from abc import ABC, abstractmethod

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as sf
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class ContentBasedRecommenderBase(ABC):
    """
    Base class for content-based recommenders.
    Provides common functionality for feature processing, training, and prediction.
    """
    
    def __init__(self, seed=None, revenue_weight=1.0):
        """
        Args:
            seed: Random seed for reproducibility
            revenue_weight: Weight for revenue optimization in ranking
        """
        self.seed = seed
        self.revenue_weight = revenue_weight
        self.user_feature_cols = None
        self.item_feature_cols = None
        self.categorical_cols = None
        self.numerical_cols = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_fitted = False
        
        if seed is not None:
            np.random.seed(seed)
    
    def _extract_feature_columns(self, user_features: DataFrame, item_features: DataFrame) -> None:
        """Extract and categorize feature columns from user and item dataframes."""
        # Get user feature columns (exclude user_idx and segment)
        user_cols = [col for col in user_features.columns 
                    if col not in ['user_idx', 'segment', '__iter']]
        
        # Get item feature columns (exclude item_idx, category, and price)  
        item_cols = [col for col in item_features.columns 
                    if col not in ['item_idx', 'category', 'price', '__iter']]
        
        self.user_feature_cols = user_cols
        self.item_feature_cols = item_cols
        
        # Identify categorical columns
        self.categorical_cols = []
        if 'segment' in user_features.columns:
            self.categorical_cols.append('segment')
        if 'category' in item_features.columns:
            self.categorical_cols.append('category')
            
        self.numerical_cols = user_cols + item_cols
        
    def _prepare_training_data(self, log: DataFrame, user_features: DataFrame, 
                             item_features: DataFrame) -> pd.DataFrame:
        """Prepare training data by joining log with user and item features."""
        training_data = log.join(user_features, on='user_idx', how='inner')
        training_data = training_data.join(item_features, on='item_idx', how='inner')
        return training_data.toPandas()
    
    def _process_features(self, df: pd.DataFrame, fit_transform: bool = False) -> np.ndarray:
        """Process features for training or prediction."""
        processed_df = df.copy()
        
        # Handle categorical features with label encoding
        for col in self.categorical_cols:
            if col in processed_df.columns:
                if fit_transform:
                    le = LabelEncoder()
                    processed_df[col] = le.fit_transform(processed_df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        le = self.label_encoders[col]
                        def safe_transform(x):
                            try:
                                return le.transform([str(x)])[0]
                            except ValueError:
                                return 0
                        processed_df[col] = processed_df[col].apply(safe_transform)
                    else:
                        processed_df[col] = 0
        
        # Select feature columns for processing
        feature_cols = []
        for col in self.numerical_cols + self.categorical_cols:
            if col in processed_df.columns:
                feature_cols.append(col)
        
        if len(feature_cols) == 0:
            raise ValueError("No valid feature columns found")
        
        feature_matrix = processed_df[feature_cols].values.astype(float)
        
        # Scale features
        if fit_transform:
            feature_matrix = self.scaler.fit_transform(feature_matrix)
        else:
            feature_matrix = self.scaler.transform(feature_matrix)
            
        return feature_matrix
    
    def _create_user_item_features(self, users: DataFrame, items: DataFrame,
                                 user_features: DataFrame, item_features: DataFrame) -> pd.DataFrame:
        """Create user-item feature combinations for prediction."""
        user_item_pairs = users.crossJoin(items)
        
        user_features_clean = user_features.select(
            *[col for col in user_features.columns if col != '__iter']
        )
        item_features_clean = item_features.select(
            *[col for col in item_features.columns if col != '__iter']
        )
        
        user_item_data = user_item_pairs.join(user_features_clean, on='user_idx', how='inner')
        user_item_data = user_item_data.join(item_features_clean, on='item_idx', how='inner')
        
        pandas_df = user_item_data.toPandas()
        pandas_df = pandas_df.loc[:, ~pandas_df.columns.duplicated()]
        
        return pandas_df
    
    @abstractmethod
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the underlying model."""
        pass
    
    @abstractmethod
    def _predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for feature matrix."""
        pass
    
    def fit(self, log: DataFrame, user_features: Optional[DataFrame] = None, 
            item_features: Optional[DataFrame] = None) -> None:
        """Train the recommender model."""
        if user_features is None or item_features is None:
            raise ValueError("Both user_features and item_features are required")
        
        if log is None or log.count() == 0:
            print("Warning: No interaction data available for training")
            return
            
        self._extract_feature_columns(user_features, item_features)
        training_df = self._prepare_training_data(log, user_features, item_features)
        
        if len(training_df) == 0:
            print("Warning: No training data after joining features")
            return
        
        X = self._process_features(training_df, fit_transform=True)
        y = training_df['relevance'].values
        
        self._train_model(X, y)
        self.is_fitted = True
        print(f"Model trained on {len(training_df)} interactions with {X.shape[1]} features")
    
    def predict(self, log: DataFrame, k: int, users: DataFrame, items: DataFrame,
                user_features: Optional[DataFrame] = None, item_features: Optional[DataFrame] = None,
                filter_seen_items: bool = True) -> DataFrame:
        """Generate recommendations for users."""
        if not self.is_fitted:
            print("Warning: Model not fitted. Using random recommendations.")
            return self._random_recommendations(users, items, k)
        
        if user_features is None or item_features is None:
            print("Warning: Missing features. Using random recommendations.")
            return self._random_recommendations(users, items, k)
        
        user_item_df = self._create_user_item_features(users, items, user_features, item_features)
        
        # Filter seen items if requested
        if filter_seen_items and log is not None and log.count() > 0:
            seen_pairs = log.select("user_idx", "item_idx").toPandas()
            seen_pairs['seen'] = True
            user_item_df = user_item_df.merge(
                seen_pairs, on=['user_idx', 'item_idx'], how='left'
            )
            user_item_df = user_item_df[user_item_df['seen'].isna()].drop('seen', axis=1)
        
        if len(user_item_df) == 0:
            print("Warning: No user-item pairs left after filtering")
            return self._random_recommendations(users, items, k)
        
        X = self._process_features(user_item_df, fit_transform=False)
        probabilities = self._predict_probabilities(X)
        
        # Add probabilities to DataFrame for processing
        user_item_df['relevance'] = probabilities
        
        # Convert to Spark DataFrame for ranking optimization
        from sim4rec.utils import pandas_to_spark
        temp_recs = pandas_to_spark(user_item_df[['user_idx', 'item_idx', 'relevance']])
        
        from pyspark.sql.types import LongType
        temp_recs = temp_recs.withColumn("user_idx", sf.col("user_idx").cast(LongType()))
        temp_recs = temp_recs.withColumn("item_idx", sf.col("item_idx").cast(LongType()))
        
        # Apply complete ranking optimization (revenue weighting + position bias)
        optimized_recs = self._optimize_ranking(temp_recs, items)
        
        # Apply top-k filtering
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        ranked_recs = optimized_recs.withColumn("rank", sf.row_number().over(window))
        final_recs = ranked_recs.filter(sf.col("rank") <= k).drop("rank")
        
        return final_recs
    
    def _random_recommendations(self, users: DataFrame, items: DataFrame, k: int) -> DataFrame:
        """Generate random recommendations as fallback."""
        recs = users.crossJoin(items)
        recs = recs.withColumn("relevance", sf.rand(seed=self.seed))
        
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        recs = recs.withColumn("rank", sf.row_number().over(window))
        recs = recs.filter(sf.col("rank") <= k).drop("rank")
        
        return recs
    
    def _apply_revenue_weighting(self, recommendations_df, items_df):
        """
        Apply revenue weighting to relevance scores.
        
        Args:
            recommendations_df: DataFrame with user_idx, item_idx, relevance columns
            items_df: DataFrame with item features including price
            
        Returns:
            DataFrame: Recommendations with revenue-weighted relevance scores
        """
        from pyspark.sql import functions as sf
        
        # Join with item prices
        revenue_weighted = recommendations_df.join(
            items_df.select("item_idx", "price"), 
            on="item_idx", 
            how="inner"
        )
        
        # Apply revenue weighting: relevance * (price^revenue_weight)
        revenue_weighted = revenue_weighted.withColumn(
            "relevance",
            sf.col("relevance") * sf.pow(sf.col("price"), self.revenue_weight)
        ).drop("price")
        
        return revenue_weighted
    
    def _apply_position_bias(self, recommendations_df):
        """
        Apply position bias to ranking by discounting scores based on position.
        
        Args:
            recommendations_df: DataFrame with user_idx, item_idx, relevance columns
            
        Returns:
            DataFrame: Recommendations with position-discounted relevance scores
        """
        from pyspark.sql import Window
        from pyspark.sql import functions as sf
        
        # Create ranking window partitioned by user, ordered by relevance desc
        window = Window.partitionBy("user_idx").orderBy(sf.desc("relevance"))
        
        # Add rank column (1-based)
        ranked_df = recommendations_df.withColumn("rank", sf.row_number().over(window))
        
        # Apply position bias discount: 1/log2(rank + 1) similar to NDCG
        # This gives higher weight to top positions
        position_discounted_df = ranked_df.withColumn(
            "relevance", 
            sf.col("relevance") / sf.log2(sf.col("rank") + 1)
        ).drop("rank")
        
        return position_discounted_df
    
    def _optimize_ranking(self, recommendations_df, items_df):
        """
        Optimize ranking using revenue weighting and position bias.
        
        Args:
            recommendations_df: DataFrame with user_idx, item_idx, relevance columns
            items_df: DataFrame with item features including price
            
        Returns:
            DataFrame: Optimized recommendations with revenue and position bias
        """
        # Apply revenue weighting first
        revenue_optimized = self._apply_revenue_weighting(recommendations_df, items_df)
        
        # Then apply position bias
        final_optimized = self._apply_position_bias(revenue_optimized)
        
        return final_optimized 


class MyRecommender(ContentBasedRecommenderBase):
    """Content-based recommender using Random Forest."""
    
    def __init__(self, seed=None, revenue_weight=1.0, n_estimators=100, 
                 max_depth=None, min_samples_split=2, min_samples_leaf=1, 
                 max_features='sqrt'):
        """
        Args:
            seed: Random seed for reproducibility
            revenue_weight: Weight for revenue optimization
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees (None for unlimited)
            min_samples_split: Minimum samples required to split a node
            min_samples_leaf: Minimum samples required at a leaf node
            max_features: Number of features to consider for splits
        """
        super().__init__(seed=seed, revenue_weight=revenue_weight)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.model = None
        self.feature_importance_ = None
        
    def _train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train Random Forest model."""
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.seed,
            class_weight='balanced',
            n_jobs=-1
        )
        
        try:
            self.model.fit(X, y)
            self.feature_importance_ = self.model.feature_importances_
            
            if len(np.unique(y)) > 1:
                train_score = self.model.score(X, y)
                print(f"Training accuracy: {train_score:.3f}")
                
                try:
                    y_proba = self.model.predict_proba(X)[:, 1]
                    auc_score = roc_auc_score(y, y_proba)
                    print(f"Training AUC: {auc_score:.3f}")
                except:
                    pass
            
            if self.feature_importance_ is not None and len(self.feature_importance_) > 0:
                top_features = np.argsort(self.feature_importance_)[-5:][::-1]
                print(f"Top 5 most important features (indices): {top_features}")
                print(f"Feature importances: {self.feature_importance_[top_features]}")
            
        except Exception as e:
            print(f"Warning: Model training failed: {e}")
            self.model = None
    
    def _predict_probabilities(self, X: np.ndarray) -> np.ndarray:
        """Predict purchase probabilities using Random Forest."""
        if self.model is None:
            return np.random.random(X.shape[0])
        
        try:
            probabilities = self.model.predict_proba(X)[:, 1]
            return probabilities
        except Exception as e:
            print(f"Warning: Prediction failed: {e}")
            return np.random.random(X.shape[0])
    
    def get_feature_importance(self):
        """Get feature importance from the trained Random Forest model."""
        return self.feature_importance_ if self.feature_importance_ is not None else None 