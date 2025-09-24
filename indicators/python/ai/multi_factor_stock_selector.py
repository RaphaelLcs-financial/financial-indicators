"""
Multi-Factor Intelligent Stock Selection Model
=============================================

A comprehensive AI-driven multi-factor stock selection system that combines
fundamental analysis, technical indicators, quantitative factors, and machine
learning to generate superior stock picks.

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class MultiFactorStockSelector:
    """
    Advanced multi-factor stock selection model using ensemble machine learning
    """

    def __init__(self, n_features: int = 50, n_clusters: int = 5):
        """
        Initialize multi-factor stock selector

        Args:
            n_features: Number of features to select
            n_clusters: Number of clusters for stock grouping
        """
        self.n_features = n_features
        self.n_clusters = n_clusters

        # Feature engineering
        self.feature_engineer = FeatureEngineer()

        # Machine learning models
        self.models = {
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'neural_network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1)
        }

        # Ensemble weights
        self.ensemble_weights = None

        # Feature importance
        self.feature_importance = None

        # Stock clustering
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

        # Scalers
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()

        # Risk models
        self.risk_model = RiskModel()

        # Performance tracking
        self.performance_history = []

    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer comprehensive features for stock selection

        Args:
            data: DataFrame with OHLCV data and fundamental data

        Returns:
            DataFrame with engineered features
        """
        features = self.feature_engineer.create_all_features(data)
        return features

    def prepare_features(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features and targets for model training

        Args:
            data: DataFrame with stock data

        Returns:
            Tuple of features and targets
        """
        # Engineer features
        features_df = self.engineer_features(data)

        # Create forward returns as target
        forward_returns = data['close'].pct_change(21).shift(-21)  # 21-day forward return

        # Remove rows with NaN
        valid_mask = ~(features_df.isnull().any(axis=1) | forward_returns.isnull())
        features_clean = features_df[valid_mask]
        targets_clean = forward_returns[valid_mask]

        # Feature selection
        if features_clean.shape[1] > self.n_features:
            features_clean = self.select_top_features(features_clean, targets_clean)

        return features_clean.values, targets_clean.values

    def select_top_features(self, features: pd.DataFrame, targets: pd.Series) -> pd.DataFrame:
        """
        Select top features using multiple methods

        Args:
            features: Feature matrix
            targets: Target values

        Returns:
            Selected features
        """
        # Method 1: Random Forest importance
        rf = RandomForestRegressor(n_estimators=50, random_state=42)
        rf.fit(features, targets)
        rf_importance = rf.feature_importances_

        # Method 2: Correlation with target
        correlations = features.corrwith(targets).abs()

        # Method 3: Mutual information (approximation)
        mi_scores = []
        for col in features.columns:
            mi = self.mutual_info_score(features[col], targets)
            mi_scores.append(mi)

        # Combine scores
        combined_scores = (rf_importance + correlations.values + np.array(mi_scores)) / 3

        # Select top features
        top_features_idx = np.argsort(combined_scores)[-self.n_features:]
        selected_features = features.iloc[:, top_features_idx]

        # Store feature importance
        self.feature_importance = pd.Series(combined_scores, index=features.columns)

        return selected_features

    def mutual_info_score(self, x: pd.Series, y: pd.Series, bins: int = 20) -> float:
        """
        Calculate mutual information score (simplified)

        Args:
            x: Feature series
            y: Target series
            bins: Number of bins for discretization

        Returns:
            Mutual information score
        """
        # Discretize continuous variables
        x_bins = pd.cut(x, bins=bins, labels=False)
        y_bins = pd.cut(y, bins=bins, labels=False)

        # Calculate joint probability
        joint_prob = pd.crosstab(x_bins, y_bins, normalize=True)

        # Calculate marginal probabilities
        x_marginal = joint_prob.sum(axis=1)
        y_marginal = joint_prob.sum(axis=0)

        # Calculate mutual information
        mi = 0
        for i in range(bins):
            for j in range(bins):
                if joint_prob.iloc[i, j] > 0:
                    mi += joint_prob.iloc[i, j] * np.log(
                        joint_prob.iloc[i, j] / (x_marginal.iloc[i] * y_marginal.iloc[j])
                    )

        return max(0, mi)

    def train_models(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Train ensemble of models

        Args:
            features: Feature matrix
            targets: Target values

        Returns:
            Dictionary of model performance metrics
        """
        # Scale features
        features_scaled = self.feature_scaler.fit_transform(features)
        targets_scaled = self.target_scaler.fit_transform(targets.reshape(-1, 1)).ravel()

        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)

        model_scores = {}

        for name, model in self.models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, features_scaled, targets_scaled,
                                      cv=tscv, scoring='neg_mean_squared_error')

            # Train final model
            model.fit(features_scaled, targets_scaled)

            # Store performance
            model_scores[name] = {
                'cv_score': np.mean(cv_scores),
                'cv_std': np.std(cv_scores)
            }

        # Calculate ensemble weights based on performance
        self.calculate_ensemble_weights(model_scores)

        # Train clustering model
        self.train_clustering(features_scaled)

        return model_scores

    def calculate_ensemble_weights(self, model_scores: Dict[str, Dict[str, float]]) -> None:
        """
        Calculate ensemble weights based on model performance

        Args:
            model_scores: Dictionary of model performance scores
        """
        # Extract CV scores
        cv_scores = [model_scores[name]['cv_score'] for name in self.models.keys()]

        # Convert to positive scores (lower MSE is better)
        positive_scores = [-score for score in cv_scores]

        # Normalize to get weights
        total_score = sum(positive_scores)
        self.ensemble_weights = np.array(positive_scores) / total_score

    def train_clustering(self, features: np.ndarray) -> None:
        """
        Train clustering model for stock grouping

        Args:
            features: Feature matrix
        """
        # Reduce dimensionality for clustering
        pca = PCA(n_components=min(10, features.shape[1]))
        features_pca = pca.fit_transform(features)

        # Train clustering
        self.kmeans.fit(features_pca)

    def predict_returns(self, features: np.ndarray) -> np.ndarray:
        """
        Predict returns using ensemble model

        Args:
            features: Feature matrix

        Returns:
            Predicted returns
        """
        # Scale features
        features_scaled = self.feature_scaler.transform(features)

        # Get predictions from each model
        predictions = []
        for name, model in self.models.items():
            pred = model.predict(features_scaled)
            predictions.append(pred)

        # Ensemble prediction
        ensemble_pred = np.average(predictions, weights=self.ensemble_weights, axis=0)

        # Inverse transform to original scale
        returns = self.target_scaler.inverse_transform(ensemble_pred.reshape(-1, 1)).ravel()

        return returns

    def generate_stock_signals(self, data: pd.DataFrame,
                            universe: List[str] = None) -> pd.DataFrame:
        """
        Generate stock selection signals

        Args:
            data: DataFrame with stock data
            universe: List of stocks to consider

        Returns:
            DataFrame with stock signals
        """
        if universe is None:
            universe = data['symbol'].unique().tolist()

        signals = []

        for symbol in universe:
            # Get stock data
            stock_data = data[data['symbol'] == symbol].copy()

            if len(stock_data) < 100:  # Need enough data
                continue

            # Prepare features
            features_df = self.engineer_features(stock_data)

            # Get latest features
            latest_features = features_df.iloc[-1:].fillna(0)

            # Predict returns
            predicted_return = self.predict_returns(latest_features.values)[0]

            # Calculate risk metrics
            risk_metrics = self.risk_model.calculate_stock_risk(stock_data)

            # Generate signal
            signal = self.generate_trading_signal(predicted_return, risk_metrics)

            # Get cluster assignment
            cluster = self.get_stock_cluster(latest_features.values)

            signals.append({
                'symbol': symbol,
                'predicted_return': predicted_return,
                'signal': signal,
                'risk_score': risk_metrics['risk_score'],
                'volatility': risk_metrics['volatility'],
                'sharpe_ratio': risk_metrics['sharpe_ratio'],
                'cluster': cluster,
                'timestamp': stock_data['timestamp'].iloc[-1]
            })

        signals_df = pd.DataFrame(signals)

        # Rank stocks
        signals_df = self.rank_stocks(signals_df)

        return signals_df

    def generate_trading_signal(self, predicted_return: float,
                              risk_metrics: Dict[str, float]) -> str:
        """
        Generate trading signal based on predicted return and risk

        Args:
            predicted_return: Predicted return
            risk_metrics: Risk metrics dictionary

        Returns:
            Trading signal (BUY, SELL, HOLD)
        """
        # Expected return threshold
        return_threshold = 0.02  # 2% monthly return

        # Risk threshold
        risk_threshold = 0.15    # 15% volatility

        # Risk-adjusted return
        risk_adjusted_return = predicted_return / max(risk_metrics['risk_score'], 0.01)

        if predicted_return > return_threshold and risk_adjusted_return > 0.5:
            return 'BUY'
        elif predicted_return < -return_threshold:
            return 'SELL'
        else:
            return 'HOLD'

    def get_stock_cluster(self, features: np.ndarray) -> int:
        """
        Get cluster assignment for stock

        Args:
            features: Feature vector

        Returns:
            Cluster assignment
        """
        # Reduce dimensionality
        pca = PCA(n_components=min(10, features.shape[1]))
        features_pca = pca.fit_transform(features)

        # Predict cluster
        cluster = self.kmeans.predict(features_pca)[0]

        return cluster

    def rank_stocks(self, signals_df: pd.DataFrame) -> pd.DataFrame:
        """
        Rank stocks based on multiple criteria

        Args:
            signals_df: DataFrame with signals

        Returns:
            Ranked DataFrame
        """
        # Calculate composite score
        signals_df['composite_score'] = (
            signals_df['predicted_return'] * 0.4 +
            (1 - signals_df['risk_score']) * 0.3 +
            signals_df['sharpe_ratio'] * 0.2 +
            (signals_df['signal'] == 'BUY').astype(int) * 0.1
        )

        # Rank by composite score
        signals_df['rank'] = signals_df['composite_score'].rank(ascending=False)

        # Sort by rank
        signals_df = signals_df.sort_values('rank')

        return signals_df

    def backtest_strategy(self, data: pd.DataFrame,
                         start_date: str = None,
                         end_date: str = None) -> Dict[str, Any]:
        """
        Backtest the stock selection strategy

        Args:
            data: Historical data
            start_date: Start date for backtest
            end_date: End date for backtest

        Returns:
            Backtest results
        """
        if start_date is None:
            start_date = data['timestamp'].min()
        if end_date is None:
            end_date = data['timestamp'].max()

        # Filter data
        backtest_data = data[
            (data['timestamp'] >= start_date) &
            (data['timestamp'] <= end_date)
        ]

        # Initialize portfolio
        portfolio = {
            'cash': 1000000,
            'positions': {},
            'transactions': [],
            'values': []
        }

        # Monthly rebalancing
        rebalance_dates = pd.date_range(start=start_date, end=end_date, freq='M')

        for date in rebalance_dates:
            # Get data up to current date
            current_data = backtest_data[backtest_data['timestamp'] <= date]

            # Generate signals
            signals = self.generate_stock_signals(current_data)

            # Select top 20 stocks
            top_stocks = signals.head(20)

            # Rebalance portfolio
            self.rebalance_portfolio(portfolio, top_stocks, date)

        # Calculate performance metrics
        performance = self.calculate_portfolio_performance(portfolio)

        return performance

    def rebalance_portfolio(self, portfolio: Dict, signals: pd.DataFrame, date: pd.Timestamp) -> None:
        """
        Rebalance portfolio based on signals

        Args:
            portfolio: Portfolio dictionary
            signals: Stock signals
            date: Rebalance date
        """
        # Get current positions
        current_positions = portfolio['positions'].copy()

        # Target positions (equal weight)
        target_stocks = signals[signals['signal'] == 'BUY']['symbol'].tolist()
        target_positions = {stock: portfolio['cash'] / len(target_stocks)
                          for stock in target_stocks}

        # Calculate trades
        for stock, target_value in target_positions.items():
            if stock in current_positions:
                # Adjust position
                current_value = current_positions[stock]
                trade_value = target_value - current_value

                if abs(trade_value) > 1000:  # Minimum trade size
                    portfolio['transactions'].append({
                        'date': date,
                        'stock': stock,
                        'action': 'BUY' if trade_value > 0 else 'SELL',
                        'value': abs(trade_value)
                    })

            else:
                # New position
                portfolio['transactions'].append({
                    'date': date,
                    'stock': stock,
                    'action': 'BUY',
                    'value': target_value
                })

        # Update positions
        portfolio['positions'] = target_positions
        portfolio['values'].append({
            'date': date,
            'total_value': portfolio['cash'] + sum(target_positions.values())
        })

    def calculate_portfolio_performance(self, portfolio: Dict) -> Dict[str, Any]:
        """
        Calculate portfolio performance metrics

        Args:
            portfolio: Portfolio dictionary

        Returns:
            Performance metrics
        """
        values = portfolio['values']

        if len(values) < 2:
            return {'error': 'Insufficient data for performance calculation'}

        # Calculate returns
        returns = []
        for i in range(1, len(values)):
            ret = (values[i]['total_value'] - values[i-1]['total_value']) / values[i-1]['total_value']
            returns.append(ret)

        returns = np.array(returns)

        # Performance metrics
        total_return = (values[-1]['total_value'] - values[0]['total_value']) / values[0]['total_value']
        annualized_return = (1 + total_return) ** (252 / len(returns)) - 1
        annualized_volatility = np.std(returns) * np.sqrt(252)
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        max_drawdown = self.calculate_max_drawdown(returns)

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_transactions': len(portfolio['transactions']),
            'final_value': values[-1]['total_value']
        }

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """
        Calculate maximum drawdown

        Args:
            returns: Array of returns

        Returns:
            Maximum drawdown
        """
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)


class FeatureEngineer:
    """
    Feature engineering for stock selection
    """

    def __init__(self):
        self.feature_functions = {
            'technical': self.create_technical_features,
            'fundamental': self.create_fundamental_features,
            'momentum': self.create_momentum_features,
            'volatility': self.create_volatility_features,
            'quality': self.create_quality_features,
            'value': self.create_value_features,
            'growth': self.create_growth_features
        }

    def create_all_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Create all features

        Args:
            data: Input data

        Returns:
            Feature DataFrame
        """
        all_features = []

        for category, func in self.feature_functions.items():
            try:
                features = func(data)
                if features is not None and not features.empty:
                    all_features.append(features)
            except Exception as e:
                print(f"Error creating {category} features: {e}")

        if all_features:
            return pd.concat(all_features, axis=1)
        else:
            return pd.DataFrame()

    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical indicators"""
        features = pd.DataFrame(index=data.index)

        # Price-based features
        features['price_momentum_5'] = data['close'].pct_change(5)
        features['price_momentum_20'] = data['close'].pct_change(20)
        features['price_momentum_60'] = data['close'].pct_change(60)

        # Moving averages
        features['ma_5'] = data['close'].rolling(5).mean()
        features['ma_20'] = data['close'].rolling(20).mean()
        features['ma_60'] = data['close'].rolling(60).mean()

        # MA crossovers
        features['ma_5_20_ratio'] = features['ma_5'] / features['ma_20']
        features['ma_20_60_ratio'] = features['ma_20'] / features['ma_60']

        # RSI
        features['rsi_14'] = self.calculate_rsi(data['close'], 14)
        features['rsi_30'] = self.calculate_rsi(data['close'], 30)

        # MACD
        macd, macd_signal, macd_hist = self.calculate_macd(data['close'])
        features['macd'] = macd
        features['macd_signal'] = macd_signal
        features['macd_hist'] = macd_hist

        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.calculate_bollinger_bands(data['close'])
        features['bb_upper'] = bb_upper
        features['bb_middle'] = bb_middle
        features['bb_lower'] = bb_lower
        features['bb_position'] = (data['close'] - bb_lower) / (bb_upper - bb_lower)

        return features

    def create_fundamental_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create fundamental features (if available)"""
        features = pd.DataFrame(index=data.index)

        # Check if fundamental data is available
        if 'pe_ratio' in data.columns:
            features['pe_ratio'] = data['pe_ratio']
            features['pe_percentile'] = data['pe_ratio'].rolling(252).rank(pct=True)

        if 'pb_ratio' in data.columns:
            features['pb_ratio'] = data['pb_ratio']
            features['pb_percentile'] = data['pb_ratio'].rolling(252).rank(pct=True)

        if 'dividend_yield' in data.columns:
            features['dividend_yield'] = data['dividend_yield']
            features['dividend_yield_percentile'] = data['dividend_yield'].rolling(252).rank(pct=True)

        if 'market_cap' in data.columns:
            features['market_cap'] = data['market_cap']
            features['market_cap_log'] = np.log(data['market_cap'])

        return features

    def create_momentum_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create momentum features"""
        features = pd.DataFrame(index=data.index)

        # Price momentum
        for period in [1, 3, 6, 12]:
            features[f'momentum_{period}m'] = data['close'].pct_change(period * 21)

        # Volume momentum
        features['volume_momentum'] = data['volume'].pct_change(20)
        features['volume_surge'] = data['volume'] / data['volume'].rolling(20).mean()

        # Relative momentum
        features['relative_strength'] = data['close'] / data['close'].rolling(60).mean()

        # Momentum acceleration
        features['momentum_acceleration'] = features['momentum_3m'] - features['momentum_6m']

        return features

    def create_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create volatility features"""
        features = pd.DataFrame(index=data.index)

        # Historical volatility
        returns = data['close'].pct_change()
        features['volatility_20'] = returns.rolling(20).std() * np.sqrt(252)
        features['volatility_60'] = returns.rolling(60).std() * np.sqrt(252)
        features['volatility_252'] = returns.rolling(252).std() * np.sqrt(252)

        # Volatility ratio
        features['volatility_ratio'] = features['volatility_20'] / features['volatility_60']

        # Volatility trend
        features['volatility_trend'] = features['volatility_20'].pct_change(20)

        # High-low volatility
        features['high_low_volatility'] = (data['high'] - data['low']) / data['close']
        features['high_low_volatility_ma'] = features['high_low_volatility'].rolling(20).mean()

        return features

    def create_quality_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create quality features"""
        features = pd.DataFrame(index=data.index)

        # Price stability
        returns = data['close'].pct_change()
        features['price_stability'] = 1 / (1 + returns.rolling(20).std())

        # Liquidity
        features['liquidity_ratio'] = data['volume'] * data['close'] / (data['high'] - data['low'])
        features['liquidity_ratio_ma'] = features['liquidity_ratio'].rolling(20).mean()

        # Trend stability
        features['trend_consistency'] = self.calculate_trend_consistency(data['close'])

        # Earnings stability (if available)
        if 'earnings' in data.columns:
            features['earnings_stability'] = 1 / (1 + data['earnings'].pct_change().rolling(4).std())

        return features

    def create_value_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create value features"""
        features = pd.DataFrame(index=data.index)

        # Price to earnings
        if 'pe_ratio' in data.columns:
            features['pe_inverse'] = 1 / data['pe_ratio']
            features['pe_rank'] = data['pe_ratio'].rolling(252).rank(pct=True)

        # Price to book
        if 'pb_ratio' in data.columns:
            features['pb_inverse'] = 1 / data['pb_ratio']
            features['pb_rank'] = data['pb_ratio'].rolling(252).rank(pct=True)

        # Dividend yield
        if 'dividend_yield' in data.columns:
            features['dividend_yield'] = data['dividend_yield']
            features['dividend_yield_rank'] = data['dividend_yield'].rolling(252).rank(pct=True)

        # Enterprise value features (if available)
        if 'ev_ebitda' in data.columns:
            features['ev_ebitda_inverse'] = 1 / data['ev_ebitda']

        return features

    def create_growth_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create growth features"""
        features = pd.DataFrame(index=data.index)

        # Revenue growth (if available)
        if 'revenue' in data.columns:
            features['revenue_growth_1y'] = data['revenue'].pct_change(4)
            features['revenue_growth_3y'] = data['revenue'].pct_change(12)

        # Earnings growth (if available)
        if 'earnings' in data.columns:
            features['earnings_growth_1y'] = data['earnings'].pct_change(4)
            features['earnings_growth_3y'] = data['earnings'].pct_change(12)

        # Price growth
        features['price_growth_1y'] = data['close'].pct_change(252)
        features['price_growth_3y'] = data['close'].pct_change(756)

        # Growth acceleration
        if 'revenue' in data.columns:
            features['revenue_growth_acceleration'] = features['revenue_growth_1y'] - features['revenue_growth_3y']

        return features

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        return macd, macd_signal, macd_hist

    def calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        ma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = ma + (std * std_dev)
        lower = ma - (std * std_dev)
        return upper, ma, lower

    def calculate_trend_consistency(self, prices: pd.Series, window: int = 20) -> pd.Series:
        """Calculate trend consistency"""
        returns = prices.pct_change()
        positive_days = (returns > 0).rolling(window).sum()
        total_days = returns.rolling(window).count()
        consistency = positive_days / total_days
        return consistency


class RiskModel:
    """
    Risk model for stock selection
    """

    def __init__(self):
        self.lookback_periods = [20, 60, 252]

    def calculate_stock_risk(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate comprehensive risk metrics for a stock

        Args:
            data: Stock data

        Returns:
            Risk metrics dictionary
        """
        returns = data['close'].pct_change().dropna()

        risk_metrics = {}

        # Volatility
        risk_metrics['volatility'] = returns.std() * np.sqrt(252)

        # Downside risk
        downside_returns = returns[returns < 0]
        risk_metrics['downside_risk'] = downside_returns.std() * np.sqrt(252)

        # Maximum drawdown
        risk_metrics['max_drawdown'] = self.calculate_max_drawdown(returns)

        # Value at Risk
        risk_metrics['var_95'] = np.percentile(returns, 5)
        risk_metrics['var_99'] = np.percentile(returns, 1)

        # Expected Shortfall
        risk_metrics['es_95'] = returns[returns <= risk_metrics['var_95']].mean()
        risk_metrics['es_99'] = returns[returns <= risk_metrics['var_99']].mean()

        # Beta (if market data available)
        if 'market_return' in data.columns:
            risk_metrics['beta'] = self.calculate_beta(returns, data['market_return'])
        else:
            risk_metrics['beta'] = 1.0

        # Sharpe ratio
        risk_free_rate = 0.02  # Assume 2% risk-free rate
        excess_return = returns.mean() * 252 - risk_free_rate
        risk_metrics['sharpe_ratio'] = excess_return / risk_metrics['volatility']

        # Sortino ratio
        risk_metrics['sortino_ratio'] = excess_return / risk_metrics['downside_risk']

        # Information ratio
        risk_metrics['information_ratio'] = returns.mean() / returns.std()

        # Composite risk score
        risk_metrics['risk_score'] = self.calculate_risk_score(risk_metrics)

        return risk_metrics

    def calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumprod(1 + returns)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        return np.min(drawdown)

    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta"""
        covariance = np.cov(stock_returns, market_returns)[0, 1]
        market_variance = np.var(market_returns)
        return covariance / market_variance if market_variance > 0 else 1.0

    def calculate_risk_score(self, risk_metrics: Dict[str, float]) -> float:
        """Calculate composite risk score"""
        # Normalize risk metrics
        vol_score = min(risk_metrics['volatility'] / 0.3, 1.0)  # 30% vol = max
        dd_score = min(abs(risk_metrics['max_drawdown']) / 0.5, 1.0)  # 50% drawdown = max
        var_score = min(abs(risk_metrics['var_95']) / 0.1, 1.0)  # 10% daily VaR = max

        # Weighted average
        risk_score = (vol_score * 0.4 + dd_score * 0.4 + var_score * 0.2)

        return risk_score


# Example usage
if __name__ == "__main__":
    # Create sample data
    dates = pd.date_range('2020-01-01', '2023-12-31')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']

    data = []
    for symbol in symbols:
        for date in dates:
            price = 100 + np.random.normal(0, 0.02) * (len(data) + 1)
            data.append({
                'timestamp': date,
                'symbol': symbol,
                'open': price * 0.99,
                'high': price * 1.02,
                'low': price * 0.98,
                'close': price,
                'volume': np.random.randint(1000000, 10000000),
                'pe_ratio': np.random.uniform(15, 35),
                'pb_ratio': np.random.uniform(2, 8),
                'dividend_yield': np.random.uniform(0.01, 0.03),
                'market_cap': np.random.uniform(1e11, 2e12)
            })

    df = pd.DataFrame(data)

    # Initialize and train model
    selector = MultiFactorStockSelector(n_features=30, n_clusters=3)

    # Train models
    features, targets = selector.prepare_features(df)
    model_scores = selector.train_models(features, targets)

    print("Model Training Scores:")
    for name, scores in model_scores.items():
        print(f"{name}: CV Score = {scores['cv_score']:.4f} (+/- {scores['cv_std']:.4f})")

    # Generate signals
    signals = selector.generate_stock_signals(df)
    print("\nStock Selection Signals:")
    print(signals.head(10))

    # Backtest
    backtest_results = selector.backtest_strategy(df, '2021-01-01', '2023-12-31')
    print("\nBacktest Results:")
    for key, value in backtest_results.items():
        print(f"{key}: {value}")

    print("\nFeature Importance (Top 10):")
    if selector.feature_importance is not None:
        print(selector.feature_importance.sort_values(ascending=False).head(10))