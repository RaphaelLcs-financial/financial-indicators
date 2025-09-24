"""
Sentiment Momentum Indicator
============================

基于行为金融学的情绪动量指标，结合市场情绪、
投资者心理偏差和群体行为来预测市场走向。
该指标通过分析恐慌贪婪指数、媒体关注度、
社交媒体情绪等行为因子来识别市场极端状态。

特点：
- 行为偏差量化
- 群体情绪分析
- 极端状态识别
- 情绪动量计算
- 心理线建模
- 逆向操作信号
- 行为周期预测

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.stats import norm, zscore
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class SentimentMomentumIndicator:
    """
    情绪动量指标

    基于行为金融学理论分析市场情绪，
    识别投资者心理偏差和交易机会。
    """

    def __init__(self, sentiment_window: int = 20, extreme_threshold: float = 0.8,
                 reversal_window: int = 5, mean_reversion_factor: float = 0.3):
        """
        初始化情绪动量指标

        Args:
            sentiment_window: 情绪分析窗口，默认20
            extreme_threshold: 极端状态阈值，默认0.8
            reversal_window: 反转窗口，默认5
            mean_reversion_factor: 均值回归因子，默认0.3
        """
        self.sentiment_window = sentiment_window
        self.extreme_threshold = extreme_threshold
        self.reversal_window = reversal_window
        self.mean_reversion_factor = mean_reversion_factor
        self.name = f"Sentiment Momentum Indicator ({sentiment_window})"
        self.category = "behavioral"

    def calculate_fear_greed_index(self, data: pd.DataFrame) -> pd.Series:
        """
        计算恐慌贪婪指数

        Args:
            data: 市场数据

        Returns:
            恐慌贪婪指数
        """
        fear_greed_components = []

        # 价格动量 (25%)
        if 'close' in data.columns:
            price_momentum = data['close'].pct_change(self.sentiment_window).fillna(0)
            normalized_momentum = self.normalize_sentiment(price_momentum)
            fear_greed_components.append(normalized_momentum * 0.25)

        # 价格强度 (25%)
        if 'close' in data.columns and 'high' in data.columns and 'low' in data.columns:
            # 52周高位比例
            price_high = data['high'].rolling(252).max()
            current_price = data['close']
            price_strength = (current_price - price_high) / (price_high + 1e-10)
            normalized_strength = self.normalize_sentiment(price_strength)
            fear_greed_components.append(normalized_strength * 0.25)

        # 波动率 (15%)
        if 'close' in data.columns:
            volatility = data['close'].pct_change().rolling(20).std().fillna(0)
            # 低波动率 = 贪婪，高波动率 = 恐慌
            volatility_sentiment = -self.normalize_sentiment(volatility)
            fear_greed_components.append(volatility_sentiment * 0.15)

        # 成交量 (15%)
        if 'volume' in data.columns:
            volume_ma = data['volume'].rolling(30).mean()
            volume_ratio = data['volume'] / (volume_ma + 1e-10)
            volume_sentiment = self.normalize_sentiment(volume_ratio)
            fear_greed_components.append(volume_sentiment * 0.15)

        # 安全需求 (20%)
        if 'close' in data.columns:
            # 使用价格相对移动平均的位置作为安全需求代理
            ma_200 = data['close'].rolling(200).mean()
            safe_haven_demand = (data['close'] - ma_200) / (ma_200 + 1e-10)
            safe_haven_sentiment = self.normalize_sentiment(safe_haven_demand)
            fear_greed_components.append(safe_haven_sentiment * 0.20)

        # 综合恐慌贪婪指数
        if fear_greed_components:
            fear_greed_index = pd.Series(0.0, index=data.index)
            for component in fear_greed_components:
                fear_greed_index += component
            return fear_greed_index.fillna(50)
        else:
            return pd.Series(50.0, index=data.index)

    def calculate_investor_psychology_line(self, data: pd.DataFrame) -> pd.Series:
        """
        计算投资者心理线

        Args:
            data: 市场数据

        Returns:
            投资者心理线
        """
        if 'close' not in data.columns:
            return pd.Series(50.0, index=data.index)

        # 计算上涨天数比例
        price_changes = data['close'].diff().fillna(0)
        up_days = (price_changes > 0).astype(int)

        # 计算滚动心理线
        psychology_line = up_days.rolling(window=self.sentiment_window).mean() * 100

        return psychology_line.fillna(50)

    def calculate_herd_behavior_index(self, data: pd.DataFrame) -> pd.Series:
        """
        计算羊群行为指数

        Args:
            data: 市场数据

        Returns:
            羊群行为指数
        """
        herd_components = []

        # 价格同步性
        if 'close' in data.columns:
            # 计算价格变动的相关性
            returns = data['close'].pct_change().fillna(0)
            volatility = returns.rolling(self.sentiment_window).std().fillna(0)
            avg_volatility = volatility.rolling(252).mean().fillna(0)

            # 低波动且高相关 = 羊群行为
            synchronization = 1 / (1 + volatility / (avg_volatility + 1e-10))
            herd_components.append(synchronization * 0.4)

        # 成交量集中度
        if 'volume' in data.columns:
            volume_concentration = self.calculate_volume_concentration(data['volume'])
            herd_components.append(volume_concentration * 0.3)

        # 价格动量一致性
        if 'close' in data.columns:
            momentum_consistency = self.calculate_momentum_consistency(data['close'])
            herd_components.append(momentum_consistency * 0.3)

        if herd_components:
            herd_index = pd.Series(0.0, index=data.index)
            for component in herd_components:
                herd_index += component
            return herd_index.fillna(0.5)
        else:
            return pd.Series(0.5, index=data.index)

    def calculate_volume_concentration(self, volume: pd.Series) -> pd.Series:
        """
        计算成交量集中度

        Args:
            volume: 成交量序列

        Returns:
            成交量集中度指标
        """
        # 计算赫芬达尔指数
        rolling_volume = volume.rolling(self.sentiment_window)
        volume_sum = rolling_volume.sum()
        volume_squared_sum = rolling_volume.apply(lambda x: (x**2).sum())

        herfindahl_index = volume_squared_sum / (volume_sum**2 + 1e-10)
        concentration = herfindahl_index * self.sentiment_window  # 标准化

        return concentration.fillna(1/self.sentiment_window)

    def calculate_momentum_consistency(self, prices: pd.Series) -> pd.Series:
        """
        计算动量一致性

        Args:
            prices: 价格序列

        Returns:
            动量一致性指标
        """
        returns = prices.pct_change().fillna(0)

        # 计算不同时间窗口的动量符号一致性
        momentum_signs = []
        windows = [5, 10, 20]

        for window in windows:
            window_momentum = returns.rolling(window).sum()
            momentum_signs.append(np.sign(window_momentum))

        # 计算一致性（符号相同比例）
        if len(momentum_signs) >= 2:
            consistency = pd.Series(0.0, index=prices.index)
            for i in range(len(momentum_signs[0])):
                if i < len(momentum_signs[1]):
                    signs_at_i = [ms.iloc[i] for ms in momentum_signs if not np.isnan(ms.iloc[i])]
                    if len(signs_at_i) >= 2:
                        # 计算符号一致性
                        positive_ratio = np.sum(np.array(signs_at_i) > 0) / len(signs_at_i)
                        consistency.iloc[i] = max(positive_ratio, 1-positive_ratio) * 2 - 1

            return consistency.fillna(0)
        else:
            return pd.Series(0.0, index=prices.index)

    def calculate_behavioral_biases(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算行为偏差指标

        Args:
            data: 市场数据

        Returns:
            行为偏差指标字典
        """
        biases = {}

        if 'close' not in data.columns:
            return biases

        prices = data['close']
        returns = prices.pct_change().fillna(0)

        # 损失厌恶偏差
        loss_aversion = self.calculate_loss_aversion_bias(returns)
        biases['loss_aversion'] = loss_aversion

        # 过度自信偏差
        overconfidence = self.calculate_overconfidence_bias(returns, data.get('volume', pd.Series()))
        biases['overconfidence'] = overconfidence

        # 锚定效应偏差
        anchoring = self.calculate_anchoring_bias(prices)
        biases['anchoring'] = anchoring

        # 羊群效应偏差
        herding = self.calculate_herding_bias(returns)
        biases['herding'] = herding

        return biases

    def calculate_loss_aversion_bias(self, returns: pd.Series) -> pd.Series:
        """
        计算损失厌恶偏差

        Args:
            returns: 收益率序列

        Returns:
            损失厌恶偏差指标
        """
        # 分离盈利和亏损
        gains = returns[returns > 0]
        losses = returns[returns < 0]

        # 计算平均盈利和平均亏损
        avg_gain = gains.rolling(self.sentiment_window).mean()
        avg_loss = abs(losses.rolling(self.sentiment_window).mean())

        # 损失厌恶比率（通常为2-2.5）
        loss_aversion_ratio = avg_loss / (avg_gain + 1e-10)

        # 标准化为偏差指标
        loss_aversion_bias = (loss_aversion_ratio - 2.0) / 2.0  # 0为中性，正值为偏差

        return loss_aversion_bias.fillna(0)

    def calculate_overconfidence_bias(self, returns: pd.Series, volume: pd.Series) -> pd.Series:
        """
        计算过度自信偏差

        Args:
            returns: 收益率序列
            volume: 成交量序列

        Returns:
            过度自信偏差指标
        """
        # 高成交量但低波动 = 过度自信
        volatility = returns.rolling(self.sentiment_window).std().fillna(0)

        if not volume.empty:
            volume_ma = volume.rolling(self.sentiment_window).mean().fillna(0)
            volume_ratio = volume / (volume_ma + 1e-10)

            # 过度自信：交易活跃但波动率低
            overconfidence = volume_ratio / (1 + volatility * 100)
        else:
            overconfidence = pd.Series(0.0, index=returns.index)

        return overconfidence.fillna(1.0)

    def calculate_anchoring_bias(self, prices: pd.Series) -> pd.Series:
        """
        计算锚定效应偏差

        Args:
            prices: 价格序列

        Returns:
            锚定效应偏差指标
        """
        # 使用历史高点作为锚点
        historical_high = prices.rolling(252).max().fillna(prices)
        current_price = prices

        # 计算相对于锚点的偏差
        anchoring_deviation = (current_price - historical_high) / (historical_high + 1e-10)

        # 标准化
        anchoring_bias = -self.normalize_sentiment(anchoring_deviation)  # 价格低于锚点 = 正偏差

        return anchoring_bias.fillna(0)

    def calculate_herding_bias(self, returns: pd.Series) -> pd.Series:
        """
        计算羊群效应偏差

        Args:
            returns: 收益率序列

        Returns:
            羊群效应偏差指标
        """
        # 计算市场离散度
        market_volatility = returns.rolling(self.sentiment_window).std().fillna(0)
        individual_volatility = returns.rolling(5).std().fillna(0)

        # 羊群效应：个体波动率接近市场波动率
        herding_intensity = 1 / (1 + abs(individual_volatility - market_volatility) / (market_volatility + 1e-10))

        return herding_intensity.fillna(0.5)

    def normalize_sentiment(self, series: pd.Series) -> pd.Series:
        """
        标准化情绪指标到0-100范围

        Args:
            series: 原始序列

        Returns:
            标准化后的情绪指标
        """
        if len(series) < 10:
            return pd.Series(50.0, index=series.index)

        # 使用滚动分位数标准化
        rolling_min = series.rolling(self.sentiment_window).min()
        rolling_max = series.rolling(self.sentiment_window).max()

        normalized = (series - rolling_min) / (rolling_max - rolling_min + 1e-10)
        normalized = normalized * 100

        return normalized.fillna(50)

    def analyze_sentiment_momentum(self, data: pd.DataFrame, external_sentiment: pd.DataFrame = None) -> Dict[str, Any]:
        """
        分析情绪动量

        Args:
            data: 市场数据
            external_sentiment: 外部情绪数据（可选）

        Returns:
            情绪动量分析结果
        """
        results = {}

        # 基础情绪指标
        fear_greed = self.calculate_fear_greed_index(data)
        results['fear_greed_index'] = fear_greed

        psychology_line = self.calculate_investor_psychology_line(data)
        results['psychology_line'] = psychology_line

        herd_behavior = self.calculate_herd_behavior_index(data)
        results['herd_behavior_index'] = herd_behavior

        # 行为偏差分析
        behavioral_biases = self.calculate_behavioral_biases(data)
        results.update(behavioral_biases)

        # 外部情绪整合
        if external_sentiment is not None:
            external_metrics = self.integrate_external_sentiment(external_sentiment)
            results.update(external_metrics)

        # 计算情绪动量
        sentiment_momentum = self.calculate_sentiment_momentum_components(results)
        results.update(sentiment_momentum)

        # 计算情绪极值
        extreme_points = self.identify_sentiment_extremes(results)
        results.update(extreme_points)

        # 生成交易信号
        signals = self.generate_behavioral_signals(results)
        results['signals'] = signals

        return results

    def integrate_external_sentiment(self, external_sentiment: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        整合外部情绪数据

        Args:
            external_sentiment: 外部情绪数据

        Returns:
            外部情绪指标
        """
        external_metrics = {}

        # 媒体关注度
        if 'media_sentiment' in external_sentiment.columns:
            media_sentiment = external_sentiment['media_sentiment']
            external_metrics['media_sentiment'] = self.normalize_sentiment(media_sentiment)

        # 社交媒体情绪
        if 'social_sentiment' in external_sentiment.columns:
            social_sentiment = external_sentiment['social_sentiment']
            external_metrics['social_sentiment'] = self.normalize_sentiment(social_sentiment)

        # 搜索趋势
        if 'search_trends' in external_sentiment.columns:
            search_trends = external_sentiment['search_trends']
            external_metrics['search_intensity'] = self.normalize_sentiment(search_trends)

        return external_metrics

    def calculate_sentiment_momentum_components(self, sentiment_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        计算情绪动量成分

        Args:
            sentiment_data: 情绪数据

        Returns:
            情绪动量指标
        """
        momentum_components = {}

        # 恐慌贪婪动量
        if 'fear_greed_index' in sentiment_data:
            fg_momentum = sentiment_data['fear_greed_index'].diff(self.reversal_window)
            momentum_components['fear_greed_momentum'] = fg_momentum.fillna(0)

        # 心理线动量
        if 'psychology_line' in sentiment_data:
            psych_momentum = sentiment_data['psychology_line'].diff(self.reversal_window)
            momentum_components['psychology_momentum'] = psych_momentum.fillna(0)

        # 羊群行为动量
        if 'herd_behavior_index' in sentiment_data:
            herd_momentum = sentiment_data['herd_behavior_index'].diff(self.reversal_window)
            momentum_components['herd_momentum'] = herd_momentum.fillna(0)

        # 综合情绪动量
        momentum_series = []
        for component in momentum_components.values():
            momentum_series.append(component)

        if momentum_series:
            composite_momentum = pd.Series(0.0, index=momentum_series[0].index)
            for component in momentum_series:
                composite_momentum += component
            composite_momentum /= len(momentum_series)
            momentum_components['composite_sentiment_momentum'] = composite_momentum

        return momentum_components

    def identify_sentiment_extremes(self, sentiment_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        识别情绪极值点

        Args:
            sentiment_data: 情绪数据

        Returns:
            情绪极值指标
        """
        extremes = {}

        # 恐慌贪婪极值
        if 'fear_greed_index' in sentiment_data:
            fg_index = sentiment_data['fear_greed_index']
            extreme_fear = fg_index < 20
            extreme_greed = fg_index > 80
            extremes['extreme_fear'] = extreme_fear.astype(int)
            extremes['extreme_greed'] = extreme_greed.astype(int)

        # 心理线极值
        if 'psychology_line' in sentiment_data:
            psych_line = sentiment_data['psychology_line']
            oversold = psych_line < 25
            overbought = psych_line > 75
            extremes['oversold'] = oversold.astype(int)
            extremes['overbought'] = overbought.astype(int)

        # 情绪背离
        if 'fear_greed_index' in sentiment_data and 'composite_sentiment_momentum' in sentiment_data:
            fg_index = sentiment_data['fear_greed_index']
            momentum = sentiment_data['composite_sentiment_momentum']

            # 价格与情绪背离
            bearish_divergence = (fg_index > 60) & (momentum < 0)
            bullish_divergence = (fg_index < 40) & (momentum > 0)

            extremes['bearish_divergence'] = bearish_divergence.astype(int)
            extremes['bullish_divergence'] = bullish_divergence.astype(int)

        return extremes

    def generate_behavioral_signals(self, sentiment_data: Dict[str, pd.Series]) -> pd.Series:
        """
        生成行为交易信号

        Args:
            sentiment_data: 情绪数据

        Returns:
            交易信号
        """
        if 'fear_greed_index' not in sentiment_data:
            return pd.Series()

        fg_index = sentiment_data['fear_greed_index']
        signals = pd.Series(0, index=fg_index.index)

        # 极端恐惧信号（逆向买入）
        extreme_fear = fg_index < 20
        fear_region = fg_index < 40

        # 极端贪婪信号（逆向卖出）
        extreme_greed = fg_index > 80
        greed_region = fg_index > 60

        # 情绪动量确认
        momentum_confirmation = False
        if 'composite_sentiment_momentum' in sentiment_data:
            momentum = sentiment_data['composite_sentiment_momentum']
            positive_momentum = momentum > momentum.std()
            negative_momentum = momentum < -momentum.std()
            momentum_confirmation = True

        # 行为偏差确认
        bias_confirmation = False
        if 'loss_aversion' in sentiment_data and 'herding' in sentiment_data:
            loss_aversion = sentiment_data['loss_aversion']
            herding = sentiment_data['herding']

            # 高损失厌恶 + 高羊群效应 = 极端情绪
            extreme_biases = (loss_aversion > 0.5) & (herding > 0.7)
            bias_confirmation = True

        # 心理线确认
        psych_confirmation = False
        if 'psychology_line' in sentiment_data:
            psych_line = sentiment_data['psychology_line']
            extreme_psych_oversold = psych_line < 20
            extreme_psych_overbought = psych_line > 80
            psych_confirmation = True

        # 生成信号
        # 强烈买入信号：极端恐惧 + 心理超卖 + 正向动量
        strong_buy = extreme_fear
        if psych_confirmation:
            strong_buy &= extreme_psych_oversold
        if momentum_confirmation:
            strong_buy &= positive_momentum

        # 买入信号：恐惧区域
        buy = fear_region & ~strong_buy

        # 强烈卖出信号：极端贪婪 + 心理超买 + 负向动量
        strong_sell = extreme_greed
        if psych_confirmation:
            strong_sell &= extreme_psych_overbought
        if momentum_confirmation:
            strong_sell &= negative_momentum

        # 卖出信号：贪婪区域
        sell = greed_region & ~strong_sell

        # 背离信号
        if 'bullish_divergence' in sentiment_data:
            signals[sentiment_data['bullish_divergence'] == 1] = 1
        if 'bearish_divergence' in sentiment_data:
            signals[sentiment_data['bearish_divergence'] == 1] = -1

        # 分配信号强度
        signals[strong_buy] = 3
        signals[buy] = 2
        signals[sell] = -2
        signals[strong_sell] = -3

        return signals

    def get_trading_recommendations(self, sentiment_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            sentiment_analysis: 情绪分析结果

        Returns:
            交易建议DataFrame
        """
        # 找到一个指标序列作为索引
        index_keys = [k for k, v in sentiment_analysis.items() if isinstance(v, pd.Series) and not v.empty]
        if not index_keys:
            return pd.DataFrame()

        index = sentiment_analysis[index_keys[0]].index
        recommendations = pd.DataFrame(index=index)

        # 添加情绪指标
        for key, value in sentiment_analysis.items():
            if isinstance(value, pd.Series):
                recommendations[key] = value

        # 交易信号
        recommendations['signals'] = sentiment_analysis.get('signals', pd.Series(0, index=index))

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 极端恐惧，逆向机会',
            2: '买入 - 恐惧区域，情绪低迷',
            1: '弱买入 - 情绪背离',
            0: '持有 - 情绪中性',
            -1: '弱卖出 - 情绪背离',
            -2: '卖出 - 贪婪区域，情绪高涨',
            -3: '强烈卖出 - 极端贪婪，风险极高'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 情绪状态分类
        def classify_sentiment_state(row):
            fg_index = row.get('fear_greed_index', 50)
            psych_line = row.get('psychology_line', 50)
            herd_behavior = row.get('herd_behavior_index', 0.5)

            if fg_index < 25 and psych_line < 30:
                return 'extreme_panic'
            elif fg_index > 75 and psych_line > 70:
                return 'extreme_euphoria'
            elif herd_behavior > 0.8:
                return 'herding_behavior'
            elif abs(fg_index - 50) < 10:
                return 'neutral_sentiment'
            else:
                return 'normal_sentiment'

        recommendations['sentiment_state'] = recommendations.apply(classify_sentiment_state, axis=1)

        # 仓位建议
        position_map = {3: 0.8, 2: 0.5, 1: 0.3, 0: 0.2, -1: 0.1, -2: 0.1, -3: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 情绪动量调整
        if 'composite_sentiment_momentum' in recommendations.columns:
            momentum_adjustment = 1 + recommendations['composite_sentiment_momentum'] * 0.1
            recommendations['position_size'] *= momentum_adjustment.clip(0.5, 1.5)

        # 策略建议
        def get_behavioral_strategy(sentiment_state, signal_strength):
            if sentiment_state in ['extreme_panic', 'extreme_euphoria']:
                return 'contrarian'  # 逆向策略
            elif sentiment_state == 'herding_behavior':
                return 'mean_reversion'  # 均值回归
            elif abs(signal_strength) >= 2:
                return 'momentum'  # 趋势跟随
            else:
                return 'neutral'  # 中性策略

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_behavioral_strategy(x['sentiment_state'], abs(x['signals'])),
            axis=1
        )

        # 风险评估
        def assess_behavioral_risk(row):
            risk_score = 0

            # 极端情绪风险
            if row['sentiment_state'] in ['extreme_panic', 'extreme_euphoria']:
                risk_score += 0.4

            # 羊群行为风险
            if row.get('herd_behavior_index', 0) > 0.8:
                risk_score += 0.3

            # 行为偏差风险
            if row.get('loss_aversion', 0) > 0.6:
                risk_score += 0.3

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['risk_level'] = recommendations.apply(assess_behavioral_risk, axis=1)

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    n = len(dates)

    # 生成价格数据（包含一些情绪驱动的模式）
    base_trend = np.cumsum(np.random.randn(n) * 0.5)
    sentiment_cycles = 10 * np.sin(2 * np.pi * np.arange(n) / 50)  # 情绪周期
    prices = 100 + base_trend + sentiment_cycles + np.random.randn(n) * 2
    volumes = np.random.lognormal(12, 0.5, n).astype(int)

    data = pd.DataFrame({
        'open': prices + np.random.randn(n) * 0.5,
        'high': prices + np.random.rand(n) * 3,
        'low': prices - np.random.rand(n) * 3,
        'close': prices,
        'volume': volumes
    }, index=dates)

    # 创建情绪动量指标
    sentiment_indicator = SentimentMomentumIndicator(sentiment_window=20)

    # 分析情绪动量
    print("分析市场情绪动量...")
    sentiment_results = sentiment_indicator.analyze_sentiment_momentum(data)

    # 显示主要结果
    print(f"恐慌贪婪指数: {sentiment_results['fear_greed_index'].iloc[-1]:.1f}")
    print(f"心理线: {sentiment_results['psychology_line'].iloc[-1]:.1f}")
    print(f"羊群行为指数: {sentiment_results['herd_behavior_index'].iloc[-1]:.3f}")

    # 显示行为偏差
    bias_keys = ['loss_aversion', 'overconfidence', 'anchoring', 'herding']
    for bias in bias_keys:
        if bias in sentiment_results:
            print(f"{bias}偏差: {sentiment_results[bias].iloc[-1]:.3f}")

    # 显示情绪极值
    extreme_keys = ['extreme_fear', 'extreme_greed', 'oversold', 'overbought']
    print("\n情绪极值状态:")
    for extreme in extreme_keys:
        if extreme in sentiment_results:
            status = "是" if sentiment_results[extreme].iloc[-1] == 1 else "否"
            print(f"{extreme}: {status}")

    print(f"当前信号: {sentiment_results['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = sentiment_indicator.get_trading_recommendations(sentiment_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['fear_greed_index', 'psychology_line', 'signal_description',
                                          'sentiment_state', 'strategy', 'position_size', 'risk_level']])