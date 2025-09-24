"""
元学习优化器金融指标

本模块实现了基于元学习的自适应金融指标优化系统，能够从历史交易中学习并改进策略。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')


class MetaLearningOptimizer:
    """
    元学习优化器

    通过学习历史策略表现来优化当前策略选择和参数配置
    """

    def __init__(self,
                 meta_features_window: int = 50,
                 strategy_pool_size: int = 10,
                 adaptation_rate: float = 0.1):
        """
        初始化元学习优化器

        Args:
            meta_features_window: 元特征计算窗口
            strategy_pool_size: 策略池大小
            adaptation_rate: 适应率
        """
        self.meta_features_window = meta_features_window
        self.strategy_pool_size = strategy_pool_size
        self.adaptation_rate = adaptation_rate

        # 元学习模型
        self.meta_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.strategy_selector = RandomForestClassifier(n_estimators=50, random_state=42)

        # 特征处理
        self.feature_scaler = StandardScaler()
        self.feature_selector = SelectKBest(score_func=f_classif, k=20)

        # 策略池和性能历史
        self.strategy_pool = {}
        self.strategy_performance = {}
        self.meta_features_history = []
        self.strategy_selection_history = []

        # 市场状态分类器
        self.market_regime_classifier = SVC(kernel='rbf', probability=True)
        self.regime_encoder = LabelEncoder()

    def _extract_meta_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        提取元特征

        Args:
            data: 市场数据

        Returns:
            元特征DataFrame
        """
        meta_features = pd.DataFrame(index=data.index)

        # 基础市场特征
        returns = data['close'].pct_change()
        volatility = returns.rolling(window=20).std()
        trend = data['close'].rolling(window=20).mean()

        # 波动率特征
        meta_features['volatility_level'] = volatility * np.sqrt(252)
        meta_features['volatility_trend'] = volatility.pct_change()
        meta_features['volatility_regime'] = pd.qcut(volatility, q=5, labels=False)

        # 趋势特征
        meta_features['trend_strength'] = abs(trend.pct_change())
        meta_features['trend_direction'] = np.sign(trend.diff())
        meta_features['price_above_ma'] = (data['close'] > trend).astype(int)

        # 动量特征
        meta_features['momentum_5'] = data['close'].pct_change(5)
        meta_features['momentum_20'] = data['close'].pct_change(20)
        meta_features['momentum_divergence'] = meta_features['momentum_5'] - meta_features['momentum_20']

        # 成交量特征
        volume_sma = data['volume'].rolling(20).mean()
        meta_features['volume_trend'] = data['volume'] / volume_sma
        meta_features['volume_volatility'] = data['volume'].rolling(20).std()

        # 价格特征
        meta_features['price_range'] = (data['high'] - data['low']) / data['close']
        meta_features['gap_indicator'] = (data['open'] - data['close'].shift(1)) / data['close'].shift(1)

        # 技术指标
        meta_features['rsi'] = self._calculate_rsi(data['close'])
        meta_features['macd_signal'] = self._calculate_macd_signal(data['close'])
        meta_features['bb_position'] = self._calculate_bollinger_position(data)

        # 市场效率
        meta_features['market_efficiency'] = self._calculate_market_efficiency(data)
        meta_features['hurst_exponent'] = self._calculate_hurst_exponent(returns)

        # 风险特征
        meta_features['max_drawdown'] = self._calculate_max_drawdown(data['close'])
        meta_features['var_95'] = returns.rolling(50).quantile(0.05)

        # 相关性特征（如果有多个资产）
        if hasattr(data, 'close_related') and 'close_related' in data.columns:
            correlation = data['close'].corr(data['close_related'])
            meta_features['inter_asset_correlation'] = correlation

        return meta_features

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd_signal(self, prices: pd.Series) -> pd.Series:
        """计算MACD信号"""
        exp1 = prices.ewm(span=12).mean()
        exp2 = prices.ewm(span=26).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9).mean()
        return np.sign(macd - signal)

    def _calculate_bollinger_position(self, data: pd.DataFrame) -> pd.Series:
        """计算布林带位置"""
        sma = data['close'].rolling(20).mean()
        std = data['close'].rolling(20).std()
        upper_band = sma + (2 * std)
        lower_band = sma - (2 * std)

        position = (data['close'] - lower_band) / (upper_band - lower_band)
        return position.fillna(0.5)

    def _calculate_market_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """计算市场效率"""
        price_changes = abs(data['close'].diff())
        price_moves = abs(data['close'] - data['open'])

        efficiency = price_changes / (price_moves + 1e-8)
        return efficiency.rolling(20).mean()

    def _calculate_hurst_exponent(self, returns: pd.Series) -> pd.Series:
        """计算赫斯特指数"""
        hurst_values = []

        for i in range(len(returns)):
            if i < 50:
                hurst_values.append(0.5)
                continue

            window_returns = returns.iloc[max(0, i-50):i]
            if len(window_returns) < 20:
                hurst_values.append(0.5)
                continue

            try:
                # 简化的赫斯特指数计算
                lags = range(2, min(20, len(window_returns)//2))
                tau = [np.std(np.subtract(window_returns.iloc[lag:], window_returns.iloc[:-lag])) for lag in lags]

                if len(tau) > 1 and all(t > 0 for t in tau):
                    poly = np.polyfit(np.log(lags), np.log(tau), 1)
                    hurst = poly[0]
                    hurst_values.append(hurst)
                else:
                    hurst_values.append(0.5)
            except:
                hurst_values.append(0.5)

        return pd.Series(hurst_values, index=returns.index)

    def _calculate_max_drawdown(self, prices: pd.Series) -> pd.Series:
        """计算最大回撤"""
        rolling_max = prices.expanding().max()
        drawdown = (prices - rolling_max) / rolling_max
        return drawdown.rolling(50).min()

    def create_strategy_pool(self) -> Dict[str, Any]:
        """创建策略池"""
        strategies = {}

        # 趋势跟踪策略
        strategies['trend_following'] = {
            'type': 'trend',
            'parameters': {'ma_short': 10, 'ma_long': 30, 'threshold': 0.01},
            'description': '双均线趋势跟踪'
        }

        # 均值回归策略
        strategies['mean_reversion'] = {
            'type': 'mean_reversion',
            'parameters': {'period': 20, 'std_dev': 2.0},
            'description': '布林带均值回归'
        }

        # 动量策略
        strategies['momentum'] = {
            'type': 'momentum',
            'parameters': {'lookback': 10, 'threshold': 0.02},
            'description': '相对动量策略'
        }

        # 突破策略
        strategies['breakout'] = {
            'type': 'breakout',
            'parameters': {'period': 20, 'threshold': 0.015},
            'description': '价格突破策略'
        }

        # 波动率策略
        strategies['volatility'] = {
            'type': 'volatility',
            'parameters': {'period': 20, 'percentile': 80},
            'description': '波动率突破策略'
        }

        # 反转策略
        strategies['reversal'] = {
            'type': 'reversal',
            'parameters': {'period': 5, 'threshold': 0.03},
            'description': '短期价格反转策略'
        }

        # 成交量策略
        strategies['volume_based'] = {
            'type': 'volume',
            'parameters': {'volume_ma': 20, 'price_threshold': 0.01},
            'description': '成交量确认策略'
        }

        # 技术指标组合策略
        strategies['indicator_combo'] = {
            'type': 'indicator',
            'parameters': {'rsi_period': 14, 'rsi_overbought': 70, 'rsi_oversold': 30},
            'description': '技术指标组合策略'
        }

        # 机器学习策略
        strategies['ml_prediction'] = {
            'type': 'ml',
            'parameters': {'window': 50, 'prediction_horizon': 5},
            'description': '机器学习预测策略'
        }

        # 市场微观结构策略
        strategies['microstructure'] = {
            'type': 'microstructure',
            'parameters': {'tick_size': 0.01, 'imbalance_threshold': 0.6},
            'description': '市场微观结构策略'
        }

        return strategies

    def simulate_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """
        模拟策略表现

        Args:
            strategy: 策略定义
            data: 市场数据

        Returns:
            策略收益序列
        """
        returns = pd.Series(0.0, index=data.index)

        if strategy['type'] == 'trend':
            returns = self._simulate_trend_strategy(strategy, data)
        elif strategy['type'] == 'mean_reversion':
            returns = self._simulate_mean_reversion_strategy(strategy, data)
        elif strategy['type'] == 'momentum':
            returns = self._simulate_momentum_strategy(strategy, data)
        elif strategy['type'] == 'breakout':
            returns = self._simulate_breakout_strategy(strategy, data)
        elif strategy['type'] == 'volatility':
            returns = self._simulate_volatility_strategy(strategy, data)
        elif strategy['type'] == 'reversal':
            returns = self._simulate_reversal_strategy(strategy, data)
        elif strategy['type'] == 'volume':
            returns = self._simulate_volume_strategy(strategy, data)
        elif strategy['type'] == 'indicator':
            returns = self._simulate_indicator_strategy(strategy, data)
        elif strategy['type'] == 'ml':
            returns = self._simulate_ml_strategy(strategy, data)
        elif strategy['type'] == 'microstructure':
            returns = self._simulate_microstructure_strategy(strategy, data)
        else:
            returns = pd.Series(0.0, index=data.index)

        return returns

    def _simulate_trend_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """模拟趋势策略"""
        params = strategy['parameters']
        ma_short = data['close'].rolling(params['ma_short']).mean()
        ma_long = data['close'].rolling(params['ma_long']).mean()

        signals = pd.Series(0, index=data.index)
        signals[ma_short > ma_long * (1 + params['threshold'])] = 1
        signals[ma_short < ma_long * (1 - params['threshold'])] = -1

        returns = signals.shift(1) * data['close'].pct_change()
        return returns

    def _simulate_mean_reversion_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """模拟均值回归策略"""
        params = strategy['parameters']
        sma = data['close'].rolling(params['period']).mean()
        std = data['close'].rolling(params['period']).std()

        upper_band = sma + params['std_dev'] * std
        lower_band = sma - params['std_dev'] * std

        signals = pd.Series(0, index=data.index)
        signals[data['close'] > upper_band] = -1  # 卖出信号
        signals[data['close'] < lower_band] = 1   # 买入信号

        returns = signals.shift(1) * data['close'].pct_change()
        return returns

    def _simulate_momentum_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """模拟动量策略"""
        params = strategy['parameters']
        momentum = data['close'].pct_change(params['lookback'])

        signals = pd.Series(0, index=data.index)
        signals[momentum > params['threshold']] = 1
        signals[momentum < -params['threshold']] = -1

        returns = signals.shift(1) * data['close'].pct_change()
        return returns

    def _simulate_breakout_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """模拟突破策略"""
        params = strategy['parameters']
        high_max = data['high'].rolling(params['period']).max()
        low_min = data['low'].rolling(params['period']).min()

        signals = pd.Series(0, index=data.index)
        signals[data['close'] > high_max * (1 + params['threshold'])] = 1
        signals[data['close'] < low_min * (1 - params['threshold'])] = -1

        returns = signals.shift(1) * data['close'].pct_change()
        return returns

    def _simulate_volatility_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """模拟波动率策略"""
        params = strategy['parameters']
        volatility = data['close'].pct_change().rolling(params['period']).std()
        volatility_threshold = volatility.quantile(params['percentile'] / 100)

        signals = pd.Series(0, index=data.index)
        signals[volatility > volatility_threshold] = 1  # 高波动率买入

        returns = signals.shift(1) * data['close'].pct_change()
        return returns

    def _simulate_reversal_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """模拟反转策略"""
        params = strategy['parameters']
        price_change = data['close'].pct_change(params['period'])

        signals = pd.Series(0, index=data.index)
        signals[price_change > params['threshold']] = -1  # 大涨后卖出
        signals[price_change < -params['threshold']] = 1  # 大跌后买入

        returns = signals.shift(1) * data['close'].pct_change()
        return returns

    def _simulate_volume_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """模拟成交量策略"""
        params = strategy['parameters']
        volume_ma = data['volume'].rolling(params['volume_ma']).mean()
        price_change = data['close'].pct_change()

        signals = pd.Series(0, index=data.index)
        volume_signal = data['volume'] > volume_ma * 1.5  # 成交量放大
        price_signal = abs(price_change) > params['price_threshold']

        signals[volume_signal & price_signal & (price_change > 0)] = 1
        signals[volume_signal & price_signal & (price_change < 0)] = -1

        returns = signals.shift(1) * data['close'].pct_change()
        return returns

    def _simulate_indicator_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """模拟技术指标策略"""
        params = strategy['parameters']
        rsi = self._calculate_rsi(data['close'], params['rsi_period'])

        signals = pd.Series(0, index=data.index)
        signals[rsi < params['rsi_oversold']] = 1
        signals[rsi > params['rsi_overbought']] = -1

        returns = signals.shift(1) * data['close'].pct_change()
        return returns

    def _simulate_ml_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """模拟机器学习策略"""
        params = strategy['parameters']
        window = params['window']

        features = pd.DataFrame({
            'returns': data['close'].pct_change(),
            'volatility': data['close'].pct_change().rolling(10).std(),
            'momentum': data['close'].pct_change(5)
        })

        # 简单的机器学习信号
        target = np.sign(data['close'].shift(-params['prediction_horizon']) - data['close'])

        signals = pd.Series(0, index=data.index)
        if len(features) > window:
            # 使用简单的逻辑回归
            try:
                X = features.iloc[window:].fillna(0)
                y = target.iloc[window:].fillna(0)

                if len(X) > 20:
                    model = LogisticRegression(random_state=42)
                    model.fit(X, y)
                    pred = model.predict(X.iloc[[-1]])[0]
                    signals.iloc[-1] = pred
            except:
                pass

        returns = signals.shift(1) * data['close'].pct_change()
        return returns

    def _simulate_microstructure_strategy(self, strategy: Dict[str, Any], data: pd.DataFrame) -> pd.Series:
        """模拟市场微观结构策略"""
        params = strategy['parameters']
        # 简化的订单流不平衡
        buy_pressure = (data['close'] - data['open']) / (data['high'] - data['low'] + 1e-8)
        volume_pressure = data['volume'] / data['volume'].rolling(20).mean()

        signals = pd.Series(0, index=data.index)
        signals[(buy_pressure > params['imbalance_threshold']) & (volume_pressure > 1)] = 1
        signals[(buy_pressure < -params['imbalance_threshold']) & (volume_pressure > 1)] = -1

        returns = signals.shift(1) * data['close'].pct_change()
        return returns

    def train_meta_model(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        训练元学习模型

        Args:
            data: 历史市场数据

        Returns:
            训练结果
        """
        # 创建策略池
        self.strategy_pool = self.create_strategy_pool()

        # 提取元特征
        meta_features = self._extract_meta_features(data)

        # 为每个策略模拟表现
        strategy_returns = {}
        for name, strategy in self.strategy_pool.items():
            strategy_returns[name] = self.simulate_strategy(strategy, data)

        # 创建训练数据
        X_meta = []
        y_meta = []
        strategy_names = []

        for i in range(self.meta_features_window, len(meta_features)):
            # 当前市场状态的元特征
            current_features = meta_features.iloc[i]

            # 计算每个策略在未来窗口的表现
            future_window = min(20, len(data) - i)
            strategy_performances = {}

            for name, returns in strategy_returns.items():
                if i + future_window <= len(returns):
                    future_returns = returns.iloc[i:i+future_window]
                    performance = future_returns.mean() / (future_returns.std() + 1e-8)
                    strategy_performances[name] = performance

            # 选择最佳策略
            if strategy_performances:
                best_strategy = max(strategy_performances, key=strategy_performances.get)
                X_meta.append(current_features.values)
                y_meta.append(best_strategy)
                strategy_names.append(best_strategy)

        if len(X_meta) < 20:
            return {'error': 'Insufficient training data'}

        # 转换为numpy数组
        X_meta = np.array(X_meta)
        y_meta = np.array(y_meta)

        # 编码策略标签
        self.regime_encoder.fit(y_meta)
        y_encoded = self.regime_encoder.transform(y_meta)

        # 特征选择和标准化
        X_scaled = self.feature_scaler.fit_transform(X_meta)
        X_selected = self.feature_selector.fit_transform(X_scaled, y_encoded)

        # 训练元学习模型
        cv_scores = cross_val_score(self.meta_model, X_selected, y_encoded, cv=5)
        self.meta_model.fit(X_selected, y_encoded)

        # 训练策略选择器
        self.strategy_selector.fit(X_selected, y_encoded)

        return {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'n_samples': len(X_meta),
            'feature_importance': dict(zip(
                meta_features.columns[self.feature_selector.get_support()],
                self.meta_model.feature_importances_
            )),
            'strategy_distribution': pd.Series(y_meta).value_counts().to_dict()
        }

    def select_strategy(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        选择最佳策略

        Args:
            data: 当前市场数据

        Returns:
            策略选择结果
        """
        if not hasattr(self.meta_model, 'feature_importances_'):
            return {'error': 'Meta model not trained'}

        # 提取当前元特征
        meta_features = self._extract_meta_features(data)
        current_features = meta_features.iloc[-1:].values

        # 特征处理
        X_scaled = self.feature_scaler.transform(current_features)
        X_selected = self.feature_selector.transform(X_scaled)

        # 预测最佳策略
        strategy_probs = self.meta_model.predict_proba(X_selected)[0]
        strategy_names = self.regime_encoder.classes_

        # 获取推荐策略
        best_strategy_idx = np.argmax(strategy_probs)
        best_strategy = strategy_names[best_strategy_idx]
        confidence = strategy_probs[best_strategy_idx]

        # 获取策略详细信息
        strategy_info = self.strategy_pool.get(best_strategy, {})

        # 记录选择历史
        self.strategy_selection_history.append({
            'timestamp': data.index[-1],
            'selected_strategy': best_strategy,
            'confidence': confidence,
            'all_probabilities': dict(zip(strategy_names, strategy_probs))
        })

        return {
            'selected_strategy': best_strategy,
            'strategy_info': strategy_info,
            'confidence': confidence,
            'strategy_probabilities': dict(zip(strategy_names, strategy_probs)),
            'meta_features': meta_features.iloc[-1].to_dict(),
            'market_regime': self._classify_market_regime(current_features[0])
        }

    def _classify_market_regime(self, features: np.ndarray) -> str:
        """分类市场状态"""
        try:
            if hasattr(self.market_regime_classifier, 'classes_'):
                regime = self.market_regime_classifier.predict(features.reshape(1, -1))[0]
                return str(regime)
        except:
            pass

        return 'unknown'

    def adaptive_optimize(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        自适应优化

        Args:
            data: 市场数据

        Returns:
            优化结果
        """
        # 首先选择策略
        selection_result = self.select_strategy(data)

        if 'error' in selection_result:
            return selection_result

        # 获取选中的策略
        strategy_name = selection_result['selected_strategy']
        strategy = self.strategy_pool[strategy_name]

        # 参数自适应调整
        optimized_params = self._optimize_parameters(strategy, data)

        # 预测策略表现
        performance_prediction = self._predict_strategy_performance(strategy_name, data)

        return {
            'strategy_selection': selection_result,
            'optimized_parameters': optimized_params,
            'performance_prediction': performance_prediction,
            'recommendation': self._generate_recommendation(
                selection_result, optimized_params, performance_prediction
            )
        }

    def _optimize_parameters(self, strategy: Dict[str, Any], data: pd.DataFrame) -> Dict[str, float]:
        """优化策略参数"""
        original_params = strategy['parameters'].copy()
        optimized_params = {}

        # 基于市场状态调整参数
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)

        for param_name, param_value in original_params.items():
            if 'period' in param_name:
                # 高波动率时使用较短周期
                if volatility > 0.3:
                    optimized_params[param_name] = int(param_value * 0.8)
                elif volatility < 0.1:
                    optimized_params[param_name] = int(param_value * 1.2)
                else:
                    optimized_params[param_name] = param_value
            elif 'threshold' in param_name:
                # 高波动率时提高阈值
                if volatility > 0.3:
                    optimized_params[param_name] = param_value * 1.3
                else:
                    optimized_params[param_name] = param_value
            else:
                optimized_params[param_name] = param_value

        return optimized_params

    def _predict_strategy_performance(self, strategy_name: str, data: pd.DataFrame) -> Dict[str, float]:
        """预测策略表现"""
        # 基于历史相似性预测表现
        if not self.strategy_selection_history:
            return {'expected_return': 0.0, 'expected_risk': 0.1, 'sharpe_ratio': 0.0}

        # 获取该策略的历史表现
        similar_conditions = []
        for record in self.strategy_selection_history[-20:]:  # 最近20次
            if record['selected_strategy'] == strategy_name:
                similar_conditions.append(record['confidence'])

        if similar_conditions:
            avg_confidence = np.mean(similar_conditions)
            expected_return = avg_confidence * 0.02  # 假设最大月收益2%
            expected_risk = 0.1 * (1 - avg_confidence * 0.5)  # 置信度越高风险越低
            sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0
        else:
            expected_return = 0.01
            expected_risk = 0.1
            sharpe_ratio = 0.1

        return {
            'expected_return': expected_return,
            'expected_risk': expected_risk,
            'sharpe_ratio': sharpe_ratio
        }

    def _generate_recommendation(self, selection_result: Dict[str, Any],
                               optimized_params: Dict[str, float],
                               performance_prediction: Dict[str, float]) -> Dict[str, Any]:
        """生成推荐"""
        strategy_name = selection_result['selected_strategy']
        confidence = selection_result['confidence']
        expected_sharpe = performance_prediction['sharpe_ratio']

        # 生成建议
        recommendation = {
            'strategy': strategy_name,
            'action': 'execute' if confidence > 0.6 else 'monitor',
            'position_size': min(1.0, confidence * expected_sharpe * 2),
            'time_horizon': self._estimate_time_horizon(strategy_name),
            'risk_level': self._assess_risk_level(performance_prediction),
            'confidence_score': confidence,
            'optimization_notes': []
        }

        # 添加优化建议
        if confidence > 0.8:
            recommendation['optimization_notes'].append('高置信度，可考虑增加仓位')
        elif confidence < 0.4:
            recommendation['optimization_notes'].append('低置信度，建议等待更好机会')

        if expected_sharpe > 1.5:
            recommendation['optimization_notes'].append('预期夏普比率高，值得执行')
        elif expected_sharpe < 0.5:
            recommendation['optimization_notes'].append('预期风险收益比不佳，需谨慎')

        return recommendation

    def _estimate_time_horizon(self, strategy_name: str) -> str:
        """估计时间周期"""
        strategy_type = self.strategy_pool[strategy_name]['type']

        if strategy_type in ['trend', 'momentum']:
            return 'medium_term'  # 中期
        elif strategy_type in ['mean_reversion', 'reversal']:
            return 'short_term'   # 短期
        elif strategy_type in ['breakout', 'volatility']:
            return 'short_medium'  # 短中期
        else:
            return 'medium_term'

    def _assess_risk_level(self, performance_prediction: Dict[str, float]) -> str:
        """评估风险等级"""
        expected_risk = performance_prediction['expected_risk']
        sharpe_ratio = performance_prediction['sharpe_ratio']

        if expected_risk > 0.2 or sharpe_ratio < 0.3:
            return 'high'
        elif expected_risk > 0.15 or sharpe_ratio < 0.8:
            return 'medium'
        else:
            return 'low'

    def calculate_meta_learning_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算元学习指标

        Args:
            data: 市场数据

        Returns:
            元学习指标DataFrame
        """
        # 自适应优化
        optimization_result = self.adaptive_optimize(data)

        if 'error' in optimization_result:
            return pd.DataFrame()

        # 创建指标
        indicators = pd.DataFrame(index=data.index[-1:], columns=[
            'best_strategy_confidence', 'strategy_score', 'recommended_position_size',
            'risk_level_encoded', 'performance_prediction', 'meta_learning_signal'
        ])

        recommendation = optimization_result['recommendation']

        # 提取指标值
        indicators['best_strategy_confidence'] = recommendation['confidence_score']
        indicators['strategy_score'] = recommendation['confidence_score'] * performance_prediction.get('sharpe_ratio', 0)
        indicators['recommended_position_size'] = recommendation['position_size']
        indicators['risk_level_encoded'] = {'low': 1, 'medium': 2, 'high': 3}[recommendation['risk_level']]
        indicators['performance_prediction'] = performance_prediction.get('expected_return', 0)

        # 生成交易信号
        confidence = recommendation['confidence_score']
        action = recommendation['action']
        risk_level = recommendation['risk_level']

        if action == 'execute' and confidence > 0.7 and risk_level in ['low', 'medium']:
            indicators['meta_learning_signal'] = 1  # 执行信号
        elif action == 'monitor' or confidence < 0.5 or risk_level == 'high':
            indicators['meta_learning_signal'] = 0  # 观察信号
        else:
            indicators['meta_learning_signal'] = 0.5  # 中性信号

        return indicators


def create_meta_learning_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    创建元学习特征

    Args:
        data: 市场数据

    Returns:
        元学习特征DataFrame
    """
    meta_optimizer = MetaLearningOptimizer()
    indicators = meta_optimizer.calculate_meta_learning_indicator(data)
    return indicators


# 主要功能函数
def calculate_meta_learning_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有元学习指标

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

    return create_meta_learning_features(data)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')

    # 模拟价格数据
    initial_price = 100
    returns = np.random.normal(0.001, 0.02, 300)
    prices = [initial_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    sample_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 300)
    }, index=dates)

    # 计算指标
    try:
        indicators = calculate_meta_learning_indicators(sample_data)
        print("元学习指标计算成功!")
        print(f"指标数量: {indicators.shape[1]}")
        print("最新指标值:")
        print(indicators.iloc[-1])

    except Exception as e:
        print(f"计算错误: {e}")