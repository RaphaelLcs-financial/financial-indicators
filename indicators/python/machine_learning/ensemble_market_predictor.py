"""
机器学习集成方法金融指标

本模块实现了基于机器学习集成方法的先进金融技术指标，结合多种算法的预测能力。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')


class EnsembleMarketPredictor:
    """
    集成学习市场预测器

    结合多种机器学习算法进行市场预测，提供集成决策和置信度评估
    """

    def __init__(self,
                 window_size: int = 100,
                 prediction_horizon: int = 5,
                 cv_folds: int = 5,
                 feature_selection_k: int = 20):
        """
        初始化集成市场预测器

        Args:
            window_size: 特征计算窗口大小
            prediction_horizon: 预测时间范围
            cv_folds: 交叉验证折数
            feature_selection_k: 选择最佳特征数量
        """
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.cv_folds = cv_folds
        self.feature_selection_k = feature_selection_k

        # 初始化基础模型
        self.base_models = {
            'random_forest': RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'gradient_boosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42
            ),
            'extra_trees': ExtraTreesRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ),
            'ridge': Ridge(alpha=1.0),
            'svr': SVR(kernel='rbf', C=1.0, gamma='scale')
        }

        # 初始化集成模型
        self.voting_regressor = VotingRegressor([
            (name, model) for name, model in self.base_models.items()
        ])

        # 特征选择器和标准化器
        self.feature_selector = SelectKBest(score_func=f_regression, k=feature_selection_k)
        self.scaler = RobustScaler()

        # 模型权重和性能记录
        self.model_weights = {}
        self.model_performance = {}
        self.feature_importance = {}

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建技术指标特征"""
        features = pd.DataFrame(index=data.index)

        # 价格特征
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # 移动平均特征
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = data['close'].rolling(window=period).mean()
            features[f'ema_{period}'] = data['close'].ewm(span=period).mean()
            features[f'price_sma_ratio_{period}'] = data['close'] / features[f'sma_{period}']

        # 波动率特征
        for period in [5, 10, 20]:
            features[f'volatility_{period}'] = data['close'].rolling(window=period).std()
            features[f'atr_{period}'] = self._calculate_atr(data, period)

        # 动量指标
        features['rsi'] = self._calculate_rsi(data['close'])
        features['stoch_k'], features['stoch_d'] = self._calculate_stochastic(data)
        features['macd'], features['macd_signal'] = self._calculate_macd(data['close'])

        # 成交量特征
        features['volume_sma'] = data['volume'].rolling(window=20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_sma']
        features['on_balance_volume'] = self._calculate_obv(data)

        # 价格变化特征
        features['price_change'] = data['close'].diff()
        features['price_acceleration'] = features['price_change'].diff()

        # 高低价差特征
        features['high_low_ratio'] = data['high'] / data['low']
        features['close_open_ratio'] = data['close'] / data['open']

        return features

    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """计算真实波幅"""
        high_low = data['high'] - data['low']
        high_close = np.abs(data['high'] - data['close'].shift())
        low_close = np.abs(data['low'] - data['close'].shift())

        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window=period).mean()

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_stochastic(self, data: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """计算随机指标"""
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()

        k_percent = 100 * ((data['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()

        return k_percent, d_percent

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """计算MACD指标"""
        exp1 = prices.ewm(span=fast).mean()
        exp2 = prices.ewm(span=slow).mean()
        macd = exp1 - exp2
        signal_line = macd.ewm(span=signal).mean()
        return macd, signal_line

    def _calculate_obv(self, data: pd.DataFrame) -> pd.Series:
        """计算成交量平衡指标"""
        obv = np.where(data['close'] > data['close'].shift(), data['volume'],
                      np.where(data['close'] < data['close'].shift(), -data['volume'], 0))
        return pd.Series(obv, index=data.index).cumsum()

    def _prepare_training_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """准备训练数据"""
        # 移除NaN值
        mask = ~(features.isna().any(axis=1) | target.isna())
        X = features[mask]
        y = target[mask]

        # 特征选择
        if len(X.columns) > self.feature_selection_k:
            X_selected = self.feature_selector.fit_transform(X, y)
            selected_features = X.columns[self.feature_selector.get_support()]
            X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)

        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

        return X, y

    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练所有模型

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            训练结果字典
        """
        # 创建特征和目标
        features = self._create_features(data)
        target = data['close'].shift(-self.prediction_horizon)  # 预测未来价格

        X, y = self._prepare_training_data(features, target)

        if len(X) < self.window_size:
            return {'error': 'Insufficient data for training'}

        # 时间序列交叉验证
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)

        results = {}

        # 训练基础模型
        for name, model in self.base_models.items():
            try:
                # 交叉验证
                cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')

                # 训练最终模型
                model.fit(X, y)

                # 预测性能
                y_pred = model.predict(X)
                mse = mean_squared_error(y, y_pred)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)

                results[name] = {
                    'cv_score_mean': cv_scores.mean(),
                    'cv_score_std': cv_scores.std(),
                    'mse': mse,
                    'mae': mae,
                    'r2': r2,
                    'model': model
                }

                # 特征重要性（对于树模型）
                if hasattr(model, 'feature_importances_'):
                    self.feature_importance[name] = dict(zip(X.columns, model.feature_importances_))

            except Exception as e:
                results[name] = {'error': str(e)}

        # 训练集成模型
        try:
            self.voting_regressor.fit(X, y)
            voting_pred = self.voting_regressor.predict(X)

            results['voting'] = {
                'mse': mean_squared_error(y, voting_pred),
                'mae': mean_absolute_error(y, voting_pred),
                'r2': r2_score(y, voting_pred),
                'model': self.voting_regressor
            }

        except Exception as e:
            results['voting'] = {'error': str(e)}

        # 更新模型权重基于性能
        self._update_model_weights(results)

        return results

    def _update_model_weights(self, results: Dict[str, Any]):
        """基于性能更新模型权重"""
        weights = {}
        total_score = 0

        for name, result in results.items():
            if 'error' not in result and 'r2' in result:
                # 使用R²得分作为权重
                score = max(0, result['r2'])
                weights[name] = score
                total_score += score

        # 归一化权重
        if total_score > 0:
            for name in weights:
                weights[name] /= total_score

        self.model_weights = weights

    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        进行预测

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            预测结果字典
        """
        if len(data) < self.window_size:
            return {'error': 'Insufficient data for prediction'}

        # 创建特征
        features = self._create_features(data)
        X, _ = self._prepare_training_data(features, pd.Series(dtype=float))

        if len(X) == 0:
            return {'error': 'No valid features for prediction'}

        predictions = {}
        confidence_scores = {}

        # 基础模型预测
        for name, model in self.base_models.items():
            try:
                pred = model.predict(X.iloc[[-1]])[0]
                predictions[name] = pred

                # 置信度基于模型权重
                confidence_scores[name] = self.model_weights.get(name, 0.2)

            except Exception as e:
                predictions[name] = None
                confidence_scores[name] = 0

        # 集成预测
        try:
            ensemble_pred = self.voting_regressor.predict(X.iloc[[-1]])[0]
            predictions['ensemble'] = ensemble_pred

            # 计算集成置信度
            ensemble_confidence = self._calculate_ensemble_confidence(predictions, confidence_scores)
            confidence_scores['ensemble'] = ensemble_confidence

        except Exception as e:
            predictions['ensemble'] = None
            confidence_scores['ensemble'] = 0

        return {
            'predictions': predictions,
            'confidence_scores': confidence_scores,
            'model_weights': self.model_weights,
            'feature_importance': self.feature_importance
        }

    def _calculate_ensemble_confidence(self, predictions: Dict[str, float],
                                    confidence_scores: Dict[str, float]) -> float:
        """计算集成预测的置信度"""
        valid_predictions = [p for p in predictions.values() if p is not None]

        if len(valid_predictions) < 2:
            return 0.3  # 低置信度

        # 计算预测的一致性
        predictions_array = np.array(valid_predictions)
        std_dev = np.std(predictions_array)
        mean_pred = np.mean(predictions_array)

        # 一致性得分（标准差越小，一致性越高）
        consistency_score = 1 / (1 + std_dev / abs(mean_pred)) if mean_pred != 0 else 0.5

        # 权重一致性得分
        weight_consistency = np.mean(list(confidence_scores.values()))

        # 综合置信度
        ensemble_confidence = 0.6 * consistency_score + 0.4 * weight_consistency

        return min(1.0, max(0.0, ensemble_confidence))

    def calculate_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算集成预测指标

        Args:
            data: 包含OHLCV数据的DataFrame

        Returns:
            包含指标值的DataFrame
        """
        results = self.predict(data)

        if 'error' in results:
            return pd.DataFrame()

        predictions = results['predictions']
        confidence_scores = results['confidence_scores']

        # 创建指标DataFrame
        indicators = pd.DataFrame(index=data.index[-1:], columns=[
            'ensemble_prediction', 'ensemble_confidence', 'prediction_signal'
        ])

        # 主预测
        indicators['ensemble_prediction'] = predictions.get('ensemble', 0)
        indicators['ensemble_confidence'] = confidence_scores.get('ensemble', 0)

        # 生成交易信号
        current_price = data['close'].iloc[-1]
        predicted_price = predictions.get('ensemble', current_price)

        price_change_pct = (predicted_price - current_price) / current_price
        confidence = confidence_scores.get('ensemble', 0)

        # 基于预测变化和置信度的信号
        if confidence > 0.7:
            if price_change_pct > 0.02:  # 预测上涨超过2%
                indicators['prediction_signal'] = 2  # 强烈买入
            elif price_change_pct < -0.02:  # 预测下跌超过2%
                indicators['prediction_signal'] = -2  # 强烈卖出
            else:
                indicators['prediction_signal'] = 0  # 持有
        elif confidence > 0.5:
            if price_change_pct > 0.01:
                indicators['prediction_signal'] = 1  # 买入
            elif price_change_pct < -0.01:
                indicators['prediction_signal'] = -1  # 卖出
            else:
                indicators['prediction_signal'] = 0  # 持有
        else:
            indicators['prediction_signal'] = 0  # 低置信度，持有

        return indicators


class AdaptiveEnsembleOptimizer:
    """
    自适应集成优化器

    动态调整模型组合和权重，适应市场状态变化
    """

    def __init__(self,
                 adaptation_window: int = 50,
                 performance_threshold: float = 0.6,
                 min_models: int = 3):
        """
        初始化自适应集成优化器

        Args:
            adaptation_window: 适应性调整窗口
            performance_threshold: 性能阈值
            min_models: 最小模型数量
        """
        self.adaptation_window = adaptation_window
        self.performance_threshold = performance_threshold
        self.min_models = min_models

        # 模型池
        self.model_pool = {
            'rf': RandomForestRegressor(n_estimators=50, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=50, random_state=42),
            'et': ExtraTreesRegressor(n_estimators=50, random_state=42),
            'ridge': Ridge(alpha=1.0),
            'lasso': Lasso(alpha=0.1),
            'knn': KNeighborsRegressor(n_neighbors=5),
            'dt': DecisionTreeRegressor(max_depth=5, random_state=42)
        }

        # 性能跟踪
        self.model_performance_history = {}
        self.current_weights = {}
        self.market_regime = 'unknown'

        # 初始化权重
        for model_name in self.model_pool:
            self.model_performance_history[model_name] = []
            self.current_weights[model_name] = 1.0 / len(self.model_pool)

    def detect_market_regime(self, data: pd.DataFrame) -> str:
        """检测市场状态"""
        returns = data['close'].pct_change().dropna()

        if len(returns) < 20:
            return 'unknown'

        # 计算市场特征
        volatility = returns.std() * np.sqrt(252)  # 年化波动率
        trend_strength = abs(returns.mean() * 252)  # 年化趋势强度

        # 基于波动率和趋势强度分类
        if volatility > 0.3:
            if trend_strength > 0.1:
                return 'high_volatility_trending'
            else:
                return 'high_volatility_sideways'
        else:
            if trend_strength > 0.1:
                return 'low_volatility_trending'
            else:
                return 'low_volatility_sideways'

    def optimize_ensemble(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        优化集成组合

        Args:
            data: 市场数据

        Returns:
            优化结果
        """
        # 检测市场状态
        current_regime = self.detect_market_regime(data)

        if current_regime != self.market_regime:
            # 市场状态变化，重新优化
            self.market_regime = current_regime
            return self._reoptimize_models(data)
        else:
            # 增量更新
            return self._update_weights(data)

    def _reoptimize_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """重新优化模型组合"""
        # 创建特征
        features = self._create_features(data)
        target = data['close'].shift(-5)  # 预测5期后价格

        # 移除NaN
        mask = ~(features.isna().any(axis=1) | target.isna())
        X = features[mask]
        y = target[mask]

        if len(X) < self.adaptation_window:
            return {'error': 'Insufficient data for optimization'}

        # 使用最近数据
        X_recent = X.tail(self.adaptation_window)
        y_recent = y.tail(self.adaptation_window)

        # 评估每个模型
        model_scores = {}

        for name, model in self.model_pool.items():
            try:
                # 时间序列交叉验证
                tscv = TimeSeriesSplit(n_splits=3)
                cv_scores = cross_val_score(model, X_recent, y_recent,
                                           cv=tscv, scoring='neg_mean_squared_error')

                model_scores[name] = cv_scores.mean()

                # 更新性能历史
                self.model_performance_history[name].append(cv_scores.mean())

                # 保持历史长度
                if len(self.model_performance_history[name]) > 100:
                    self.model_performance_history[name] = self.model_performance_history[name][-100:]

            except Exception as e:
                model_scores[name] = -np.inf

        # 选择最佳模型
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        selected_models = [model[0] for model in sorted_models[:self.min_models]]

        # 计算新权重
        selected_scores = [model_scores[model] for model in selected_models]
        weights = np.exp(selected_scores) / np.sum(np.exp(selected_scores))

        # 更新权重
        self.current_weights = {}
        for i, model in enumerate(selected_models):
            self.current_weights[model] = weights[i]

        return {
            'market_regime': self.market_regime,
            'selected_models': selected_models,
            'model_scores': model_scores,
            'new_weights': self.current_weights,
            'optimization_type': 'reoptimization'
        }

    def _update_weights(self, data: pd.DataFrame) -> Dict[str, Any]:
        """增量更新权重"""
        # 简单的性能衰减和更新
        for model_name in self.current_weights:
            # 轻微衰减现有权重
            self.current_weights[model_name] *= 0.95

        # 重新归一化
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for model_name in self.current_weights:
                self.current_weights[model_name] /= total_weight

        return {
            'market_regime': self.market_regime,
            'current_weights': self.current_weights,
            'optimization_type': 'incremental_update'
        }

    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """创建基础特征"""
        features = pd.DataFrame(index=data.index)

        # 基础特征
        features['returns'] = data['close'].pct_change()
        features['volatility'] = features['returns'].rolling(20).std()
        features['sma_20'] = data['close'].rolling(20).mean()
        features['sma_ratio'] = data['close'] / features['sma_20']

        return features

    def get_adaptive_prediction(self, data: pd.DataFrame) -> Dict[str, Any]:
        """获取自适应预测"""
        # 首先优化集成
        optimization_result = self.optimize_ensemble(data)

        if 'error' in optimization_result:
            return {'error': optimization_result['error']}

        # 使用当前权重进行预测
        features = self._create_features(data)
        X = features.dropna()

        if len(X) == 0:
            return {'error': 'No valid features for prediction'}

        predictions = {}
        weighted_sum = 0
        total_weight = 0

        for model_name, weight in self.current_weights.items():
            if weight > 0 and model_name in self.model_pool:
                try:
                    model = self.model_pool[model_name]
                    pred = model.predict(X.iloc[[-1]])[0]
                    predictions[model_name] = pred
                    weighted_sum += weight * pred
                    total_weight += weight
                except:
                    continue

        final_prediction = weighted_sum / total_weight if total_weight > 0 else 0

        return {
            'prediction': final_prediction,
            'model_predictions': predictions,
            'model_weights': self.current_weights,
            'market_regime': self.market_regime,
            'optimization_result': optimization_result
        }


def create_ensemble_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    创建集成学习特征

    Args:
        data: 包含OHLCV数据的DataFrame

    Returns:
        包含集成特征的新DataFrame
    """
    # 初始化指标计算器
    ensemble_predictor = EnsembleMarketPredictor()
    adaptive_optimizer = AdaptiveEnsembleOptimizer()

    # 计算指标
    ensemble_indicator = ensemble_predictor.calculate_indicator(data)

    # 获取自适应预测
    adaptive_result = adaptive_optimizer.get_adaptive_prediction(data)

    # 合并结果
    features = pd.DataFrame(index=data.index)

    if not ensemble_indicator.empty:
        features = pd.concat([features, ensemble_indicator], axis=1)

    if 'prediction' in adaptive_result:
        features['adaptive_prediction'] = adaptive_result['prediction']
        features['market_regime'] = adaptive_result.get('market_regime', 'unknown')

    return features


# 主要功能函数
def calculate_ensemble_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有集成学习指标

    Args:
        data: 包含OHLCV数据的DataFrame

    Returns:
        包含所有指标值的DataFrame
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("数据必须包含 'open', 'high', 'low', 'close', 'volume' 列")

    if len(data) < 100:
        raise ValueError("数据长度不足，至少需要100个数据点")

    return create_ensemble_features(data)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='D')

    # 模拟价格数据
    initial_price = 100
    returns = np.random.normal(0.001, 0.02, 500)
    prices = [initial_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    sample_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 500)
    }, index=dates)

    # 计算指标
    try:
        indicators = calculate_ensemble_indicators(sample_data)
        print("集成学习指标计算成功!")
        print(f"指标数量: {indicators.shape[1]}")
        print("最新指标值:")
        print(indicators.iloc[-1])

    except Exception as e:
        print(f"计算错误: {e}")