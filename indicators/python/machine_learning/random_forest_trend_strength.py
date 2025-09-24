"""
Random Forest Trend Strength Indicator
=====================================

基于随机森林算法的趋势强度指标，使用集成学习方法
来评估市场趋势的强度和持续性。该指标综合多个技术指标
作为特征，通过随机森林模型预测趋势方向和强度。

特点：
- 集成学习趋势分析
- 多特征融合
- 特征重要性评估
- 趋势持续性预测
- 异常趋势检测
- 模型置信度量化
- 自适应参数调整

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error


class RandomForestTrendStrength:
    """
    随机森林趋势强度指标

    使用随机森林算法分析市场趋势强度，
    提供多维度的趋势评估和预测能力。
    """

    def __init__(self, lookback_period: int = 50, n_estimators: int = 100,
                 trend_threshold: float = 0.02, confidence_threshold: float = 0.6):
        """
        初始化随机森林趋势强度指标

        Args:
            lookback_period: 回看周期，默认50
            n_estimators: 随机森林树数量，默认100
            trend_threshold: 趋势阈值，默认0.02
            confidence_threshold: 置信度阈值，默认0.6
        """
        self.lookback_period = lookback_period
        self.n_estimators = n_estimators
        self.trend_threshold = trend_threshold
        self.confidence_threshold = confidence_threshold
        self.name = f"Random Forest Trend Strength ({lookback_period})"
        self.category = "machine_learning"

        # 模型初始化
        self.regressor = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
        self.classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False

        # 特征重要性
        self.feature_importance = None

    def extract_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        提取特征矩阵

        Args:
            data: OHLCV数据

        Returns:
            特征矩阵
        """
        features = pd.DataFrame(index=data.index)

        # 价格特征
        features['close'] = data['close']
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # 移动平均特征
        for period in [5, 10, 20, 50]:
            features[f'sma_{period}'] = data['close'].rolling(period).mean()
            features[f'ema_{period}'] = self.calculate_ema(data['close'], period)

        # 趋势特征
        features['price_above_sma20'] = (data['close'] > features['sma_20']).astype(int)
        features['price_above_sma50'] = (data['close'] > features['sma_50']).astype(int)
        features['sma20_above_sma50'] = (features['sma_20'] > features['sma_50']).astype(int)

        # 动量特征
        features['momentum_5'] = data['close'] / data['close'].shift(5) - 1
        features['momentum_10'] = data['close'] / data['close'].shift(10) - 1
        features['momentum_20'] = data['close'] / data['close'].shift(20) - 1

        # 波动率特征
        features['volatility_5'] = features['returns'].rolling(5).std()
        features['volatility_10'] = features['returns'].rolling(10).std()
        features['volatility_20'] = features['returns'].rolling(20).std()
        features['atr'] = self.calculate_atr(data['high'], data['low'], data['close'])

        # 成交量特征
        if 'volume' in data.columns:
            features['volume_sma'] = data['volume'].rolling(20).mean()
            features['volume_ratio'] = data['volume'] / features['volume_sma']
            features['volume_trend'] = data['volume'].rolling(10).mean() / data['volume'].rolling(30).mean()

        # 技术指标特征
        features['rsi'] = self.calculate_rsi(data['close'])
        features['macd'] = self.calculate_macd(data['close'])['macd']
        features['macd_signal'] = self.calculate_macd(data['close'])['signal']
        features['stoch_k'] = self.calculate_stochastic(data['high'], data['low'], data['close'])['k']
        features['stoch_d'] = self.calculate_stochastic(data['high'], data['low'], data['close'])['d']

        # 趋势强度特征
        features['adx'] = self.calculate_adx(data['high'], data['low'], data['close'])
        features['cci'] = self.calculate_cci(data['high'], data['low'], data['close'])

        # 价格位置特征
        features['percentile_20'] = data['close'].rolling(20).rank(pct=True)
        features['percentile_50'] = data['close'].rolling(50).rank(pct=True)

        return features.fillna(0)

    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """
        准备训练数据

        Args:
            data: 原始数据

        Returns:
            特征矩阵，回归目标，分类目标
        """
        features = self.extract_features(data)

        # 计算未来收益作为回归目标
        future_returns = data['close'].shift(-5) / data['close'] - 1

        # 计算趋势方向作为分类目标
        trend_direction = pd.cut(future_returns,
                                bins=[-np.inf, -self.trend_threshold, self.trend_threshold, np.inf],
                                labels=[-1, 0, 1])

        return features, future_returns, trend_direction

    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练随机森林模型

        Args:
            data: 训练数据

        Returns:
            训练结果
        """
        # 准备训练数据
        X, y_reg, y_clf = self.prepare_training_data(data)

        # 移除包含NaN的行
        mask = ~(y_reg.isna() | y_clf.isna() | X.isna().any(axis=1))
        X_clean = X[mask]
        y_reg_clean = y_reg[mask]
        y_clf_clean = y_clf[mask]

        if len(X_clean) < self.lookback_period:
            return {'error': 'Insufficient training data'}

        # 标准化特征
        X_scaled = self.scaler.fit_transform(X_clean)

        # 训练回归模型（预测趋势强度）
        self.regressor.fit(X_scaled, y_reg_clean)

        # 训练分类模型（预测趋势方向）
        self.classifier.fit(X_scaled, y_clf_clean)

        # 获取特征重要性
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'regression_importance': self.regressor.feature_importances_,
            'classification_importance': self.classifier.feature_importances_
        })

        self.is_trained = True

        # 计算训练集性能
        reg_pred = self.regressor.predict(X_scaled)
        clf_pred = self.classifier.predict(X_scaled)

        reg_mse = mean_squared_error(y_reg_clean, reg_pred)
        clf_accuracy = accuracy_score(y_clf_clean, clf_pred)

        return {
            'training_samples': len(X_clean),
            'regression_mse': reg_mse,
            'classification_accuracy': clf_accuracy,
            'feature_importance': self.feature_importance.sort_values('regression_importance', ascending=False).head(10)
        }

    def predict_trend_strength(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        预测趋势强度

        Args:
            data: 输入数据

        Returns:
            预测结果
        """
        if not self.is_trained:
            return {'error': 'Models not trained'}

        # 提取特征
        features = self.extract_features(data)

        # 标准化特征
        X_scaled = self.scaler.transform(features)

        # 预测趋势强度和方向
        trend_strength = self.regressor.predict(X_scaled)
        trend_direction = self.classifier.predict(X_scaled)
        trend_proba = self.classifier.predict_proba(X_scaled)

        # 创建结果序列
        results = pd.DataFrame(index=data.index)
        results['trend_strength'] = trend_strength
        results['trend_direction'] = trend_direction
        results['confidence'] = np.max(trend_proba, axis=1)
        results['buy_proba'] = trend_proba[:, 2] if trend_proba.shape[1] > 2 else 0
        results['sell_proba'] = trend_proba[:, 0] if trend_proba.shape[1] > 0 else 0

        # 计算趋势持续性
        results['trend_persistence'] = self.calculate_trend_persistence(results['trend_direction'])

        # 计算趋势稳定性
        results['trend_stability'] = self.calculate_trend_stability(results['trend_strength'])

        # 生成交易信号
        results['signals'] = self.generate_signals(results)

        # 计算模型一致性
        results['model_consistency'] = self.calculate_model_consistency(results)

        return {
            'trend_strength': results['trend_strength'],
            'trend_direction': results['trend_direction'],
            'confidence': results['confidence'],
            'buy_probability': results['buy_proba'],
            'sell_probability': results['sell_proba'],
            'trend_persistence': results['trend_persistence'],
            'trend_stability': results['trend_stability'],
            'signals': results['signals'],
            'model_consistency': results['model_consistency']
        }

    def calculate_trend_persistence(self, trend_direction: pd.Series) -> pd.Series:
        """
        计算趋势持续性

        Args:
            trend_direction: 趋势方向序列

        Returns:
            趋势持续性指标
        """
        persistence = pd.Series(0.0, index=trend_direction.index)

        for i in range(1, len(trend_direction)):
            if trend_direction.iloc[i] == trend_direction.iloc[i-1]:
                persistence.iloc[i] = persistence.iloc[i-1] + 0.1
            else:
                persistence.iloc[i] = max(0, persistence.iloc[i-1] - 0.2)

        return persistence.clip(0, 1)

    def calculate_trend_stability(self, trend_strength: pd.Series) -> pd.Series:
        """
        计算趋势稳定性

        Args:
            trend_strength: 趋势强度序列

        Returns:
            趋势稳定性指标
        """
        # 计算趋势强度的滚动标准差
        volatility = trend_strength.rolling(window=10).std()
        stability = 1 / (1 + volatility * 10)
        return stability.fillna(0.5)

    def calculate_model_consistency(self, results: pd.DataFrame) -> pd.Series:
        """
        计算模型一致性

        Args:
            results: 预测结果

        Returns:
            模型一致性指标
        """
        # 检查回归和分类预测的一致性
        reg_direction = np.sign(results['trend_strength'])
        clf_direction = results['trend_direction']

        consistency = (reg_direction == clf_direction).astype(float)
        return consistency.rolling(window=5).mean().fillna(0.5)

    def generate_signals(self, results: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        Args:
            results: 预测结果

        Returns:
            交易信号
        """
        signals = pd.Series(0, index=results.index)

        # 强趋势信号
        strong_uptrend = (
            (results['trend_strength'] > self.trend_threshold) &
            (results['trend_direction'] == 1) &
            (results['confidence'] > self.confidence_threshold) &
            (results['trend_persistence'] > 0.5)
        )

        strong_downtrend = (
            (results['trend_strength'] < -self.trend_threshold) &
            (results['trend_direction'] == -1) &
            (results['confidence'] > self.confidence_threshold) &
            (results['trend_persistence'] > 0.5)
        )

        # 中等趋势信号
        moderate_uptrend = (
            (results['trend_strength'] > 0) &
            (results['trend_direction'] == 1) &
            (results['confidence'] > 0.5)
        )

        moderate_downtrend = (
            (results['trend_strength'] < 0) &
            (results['trend_direction'] == -1) &
            (results['confidence'] > 0.5)
        )

        # 趋势反转信号
        reversal_buy = (
            (results['trend_strength'] > 0) &
            (results['trend_direction'] == 1) &
            (results['trend_persistence'] < 0.3) &
            (results['buy_probability'] > 0.6)
        )

        reversal_sell = (
            (results['trend_strength'] < 0) &
            (results['trend_direction'] == -1) &
            (results['trend_persistence'] < 0.3) &
            (results['sell_probability'] > 0.6)
        )

        # 分配信号值
        signals[strong_uptrend] = 3
        signals[strong_downtrend] = -3
        signals[moderate_uptrend] = 2
        signals[moderate_downtrend] = -2
        signals[reversal_buy] = 1
        signals[reversal_sell] = -1

        return signals

    def get_trading_recommendations(self, predictions: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            predictions: 预测结果

        Returns:
            交易建议DataFrame
        """
        recommendations = pd.DataFrame(index=predictions['trend_strength'].index)

        # 基础预测信息
        recommendations['trend_strength'] = predictions['trend_strength']
        recommendations['trend_direction'] = predictions['trend_direction']
        recommendations['confidence'] = predictions['confidence']
        recommendations['signals'] = predictions['signals']

        # 趋势质量指标
        recommendations['trend_persistence'] = predictions['trend_persistence']
        recommendations['trend_stability'] = predictions['trend_stability']
        recommendations['model_consistency'] = predictions['model_consistency']

        # 概率信息
        recommendations['buy_probability'] = predictions['buy_probability']
        recommendations['sell_probability'] = predictions['sell_probability']

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 强上升趋势确认',
            2: '买入 - 中等上升趋势',
            1: '弱买入 - 趋势反转可能',
            0: '持有 - 趋势不明确',
            -1: '弱卖出 - 趋势反转可能',
            -2: '卖出 - 中等下降趋势',
            -3: '强烈卖出 - 强下降趋势确认'
        }
        recommendations['signal_description'] = predictions['signals'].map(signal_descriptions)

        # 仓位建议
        position_sizes = {
            3: 0.7, 2: 0.5, 1: 0.3, 0: 0.1, -1: 0.1, -2: 0.3, -3: 0.5
        }
        recommendations['position_size'] = predictions['signals'].map(position_sizes)

        # 置信度和稳定性调整
        confidence_adjustment = predictions['confidence']
        stability_adjustment = predictions['trend_stability']
        recommendations['position_size'] *= confidence_adjustment * stability_adjustment

        # 风险评估
        recommendations['risk_level'] = recommendations.apply(
            lambda x: 'low' if x['confidence'] > 0.7 and x['trend_stability'] > 0.7 else
                     'medium' if x['confidence'] > 0.5 else 'high',
            axis=1
        )

        # 持有期建议
        recommendations['holding_period'] = predictions['signals'].map({
            3: 'long_term', 2: 'medium_term', 1: 'short_term',
            0: 'neutral', -1: 'short_term', -2: 'medium_term', -3: 'long_term'
        })

        return recommendations

    # 辅助计算函数
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        alpha = 2 / (period + 1)
        ema = prices.copy()
        for i in range(1, len(prices)):
            ema.iloc[i] = alpha * prices.iloc[i] + (1 - alpha) * ema.iloc[i-1]
        return ema

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """计算平均真实波幅"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(window=14).mean().fillna(tr)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)

    def calculate_macd(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """计算MACD"""
        ema_12 = self.calculate_ema(prices, 12)
        ema_26 = self.calculate_ema(prices, 26)
        macd = ema_12 - ema_26
        signal = self.calculate_ema(macd, 9)
        return {'macd': macd, 'signal': signal}

    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """计算随机指标"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low)
        d = k.rolling(window=d_period).mean()
        return {'k': k.fillna(50), 'd': d.fillna(50)}

    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """计算平均趋向指数"""
        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算方向移动
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # 计算平滑值
        plus_di = 100 * pd.Series(plus_dm).ewm(span=period).mean() / pd.Series(tr).ewm(span=period).mean()
        minus_di = 100 * pd.Series(minus_dm).ewm(span=period).mean() / pd.Series(tr).ewm(span=period).mean()
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period).mean()

        return adx.fillna(25)

    def calculate_cci(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
        """计算商品通道指数"""
        tp = (high + low + close) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
        cci = (tp - sma_tp) / (0.015 * mad)
        return cci.fillna(0)


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    n = len(dates)

    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high_prices = close_prices + np.random.rand(n) * 2
    low_prices = close_prices - np.random.rand(n) * 2
    open_prices = close_prices + np.random.randn(n) * 0.3
    volumes = np.random.randint(1000000, 5000000, n)

    data = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)

    # 创建随机森林趋势强度指标
    rf_trend = RandomForestTrendStrength(
        lookback_period=50,
        n_estimators=100,
        trend_threshold=0.02,
        confidence_threshold=0.6
    )

    # 训练模型
    print("训练随机森林趋势强度模型...")
    training_result = rf_trend.train_models(data)
    print(f"训练结果: {training_result}")

    # 预测趋势强度
    print("\n预测趋势强度...")
    predictions = rf_trend.predict_trend_strength(data)

    if 'trend_strength' in predictions:
        print(f"最新趋势强度: {predictions['trend_strength'].iloc[-1]:.4f}")
        print(f"趋势方向: {predictions['trend_direction'].iloc[-1]}")
        print(f"预测置信度: {predictions['confidence'].iloc[-1]:.2f}")
        print(f"趋势持续性: {predictions['trend_persistence'].iloc[-1]:.2f}")
        print(f"趋势稳定性: {predictions['trend_stability'].iloc[-1]:.2f}")
        print(f"当前信号: {predictions['signals'].iloc[-1]}")

        # 获取交易建议
        recommendations = rf_trend.get_trading_recommendations(predictions)
        print("\n最新交易建议:")
        print(recommendations.iloc[-1])
    else:
        print(f"预测错误: {predictions['error']}")