"""
Investor Psychology Cycle Analyzer
===================================

基于投资者心理周期的行为金融学指标，识别市场参与者的
心理状态变化和情绪波动，预测市场转折点和趋势变化。
该指标通过分析投资者情绪周期、心理偏差和群体行为
来捕捉市场的非理性波动。

特点：
- 心理周期相位识别
- 情绪极值检测
- 群体恐慌/贪婪量化
- 心理偏差累积效应
- 市场情绪动能
- 非理性繁荣识别
- 逆向投资信号

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.stats import norm, zscore
from scipy.signal import find_peaks, butter, filtfilt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


class InvestorPsychologyCycle:
    """
    投资者心理周期分析器

    分析市场参与者的心理状态变化，
    识别情绪周期和非理性行为模式。
    """

    def __init__(self, psychology_windows: List[int] = [10, 20, 50],
                 sentiment_threshold: float = 0.7,
                 cycle_length: int = 42):
        """
        初始化投资者心理周期分析器

        Args:
            psychology_windows: 心理分析窗口大小，默认[10,20,50]
            sentiment_threshold: 情绪阈值，默认0.7
            cycle_length: 心理周期长度，默认42天
        """
        self.psychology_windows = psychology_windows
        self.sentiment_threshold = sentiment_threshold
        self.cycle_length = cycle_length
        self.name = f"Investor Psychology Cycle ({psychology_windows[0]})"
        self.category = "behavioral"

    def calculate_market_sentiment_cycle(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算市场情绪周期

        Args:
            data: 市场数据，包含价格、成交量等

        Returns:
            情绪周期指标
        """
        sentiment_indicators = {}

        if 'close' not in data.columns:
            return sentiment_indicators

        prices = data['close']

        # 1. 价格动能情绪
        for window in self.psychology_windows:
            # 价格变化率
            price_change = prices.pct_change(window).fillna(0)
            sentiment_indicators[f'price_momentum_sentiment_{window}'] = self.normalize_sentiment(price_change)

            # 价格加速
            price_acceleration = price_change.diff().fillna(0)
            sentiment_indicators[f'price_acceleration_sentiment_{window}'] = self.normalize_sentiment(price_acceleration)

        # 2. 成交量情绪
        if 'volume' in data.columns:
            volume = data['volume']

            # 成交量异常
            volume_zscore = zscore(volume.fillna(volume.mean()))
            volume_sentiment = np.tanh(volume_zscore * 0.5)  # 压缩到[-1,1]
            sentiment_indicators['volume_sentiment'] = pd.Series(volume_sentiment, index=prices.index)

            # 价量背离
            for window in self.psychology_windows:
                price_change_window = prices.pct_change(window).fillna(0)
                volume_change = volume.pct_change(window).fillna(0)

                # 价量背离程度
                divergence = np.sign(price_change_window) * volume_change
                sentiment_indicators[f'price_volume_divergence_{window}'] = self.normalize_sentiment(divergence)

        # 3. 波动率情绪
        # 计算历史波动率
        returns = prices.pct_change().fillna(0)
        volatility = returns.rolling(window=20).std().fillna(0)

        # 波动率异常情绪
        volatility_zscore = zscore(volatility.fillna(volatility.mean()))
        volatility_sentiment = -np.tanh(volatility_zscore * 0.5)  # 高波动率通常对应恐慌情绪
        sentiment_indicators['volatility_sentiment'] = pd.Series(volatility_sentiment, index=prices.index)

        return sentiment_indicators

    def calculate_psychology_cycle_phase(self, data: pd.DataFrame) -> pd.Series:
        """
        计算心理周期相位

        Args:
            data: 市场数据

        Returns:
            心理周期相位序列
        """
        if 'close' not in data.columns:
            return pd.Series()

        prices = data['close']

        # 使用多个指标计算综合情绪
        sentiment_components = []

        # 1. 价格位置情绪
        max_window = max(self.psychology_windows)
        price_high = prices.rolling(window=max_window).max()
        price_low = prices.rolling(window=max_window).min()
        price_position = (prices - price_low) / (price_high - price_low + 1e-10)
        price_position_sentiment = (price_position - 0.5) * 2  # 转换到[-1,1]
        sentiment_components.append(price_position_sentiment.fillna(0))

        # 2. 趋势强度情绪
        returns = prices.pct_change().fillna(0)
        trend_strength = abs(returns.rolling(window=20).mean())
        trend_sentiment = np.sign(returns.rolling(window=20).mean()) * trend_strength
        sentiment_components.append(trend_sentiment.fillna(0))

        # 3. 动量情绪
        momentum = prices.pct_change(10).fillna(0)
        momentum_sentiment = self.normalize_sentiment(momentum)
        sentiment_components.append(momentum_sentiment)

        # 综合情绪指数
        if sentiment_components:
            composite_sentiment = pd.Series(0.0, index=prices.index)
            weights = [0.4, 0.4, 0.2]

            for i, component in enumerate(sentiment_components):
                composite_sentiment += component * weights[i]

            # 计算情绪周期相位
            cycle_phase = self.calculate_cycle_phase(composite_sentiment)
            return cycle_phase
        else:
            return pd.Series(0.0, index=prices.index)

    def calculate_cycle_phase(self, sentiment_series: pd.Series) -> pd.Series:
        """
        计算情绪周期相位

        Args:
            sentiment_series: 情绪序列

        Returns:
            周期相位序列
        """
        # 使用Hilbert变换计算瞬时相位
        try:
            # 创建解析信号
            analytic_signal = self.hilbert_transform(sentiment_series.values)

            # 计算瞬时相位
            instantaneous_phase = np.angle(analytic_signal)

            # 归一化到[0, 2π]
            phase = (instantaneous_phase + np.pi) / (2 * np.pi)

            return pd.Series(phase, index=sentiment_series.index)
        except:
            # 如果Hilbert变换失败，使用简化方法
            normalized_sentiment = (sentiment_series - sentiment_series.min()) / (sentiment_series.max() - sentiment_series.min() + 1e-10)
            phase = normalized_sentiment * 2 * np.pi
            return pd.Series(phase, index=sentiment_series.index)

    def hilbert_transform(self, signal: np.ndarray) -> np.ndarray:
        """
        简化的Hilbert变换实现

        Args:
            signal: 输入信号

        Returns:
            解析信号
        """
        from scipy.fft import fft, ifft, fftfreq

        n = len(signal)

        # FFT
        fft_signal = fft(signal)

        # 创建Hilbert变换器
        freqs = fftfreq(n)

        # Hilbert变换核
        hilbert_kernel = np.zeros(n)
        hilbert_kernel[1:n//2] = 2j
        hilbert_kernel[n//2+1:] = -2j

        # 应用Hilbert变换
        hilbert_fft = fft_signal * hilbert_kernel

        # 逆FFT
        analytic_signal = ifft(hilbert_fft)

        return analytic_signal

    def detect_extreme_psychology_states(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测极端心理状态

        Args:
            data: 市场数据

        Returns:
            极端心理状态指标
        """
        extreme_states = {}

        # 计算情绪周期
        sentiment_cycle = self.calculate_market_sentiment_cycle(data)
        cycle_phase = self.calculate_psychology_cycle_phase(data)

        if sentiment_cycle:
            # 综合情绪指数
            composite_sentiment = self.calculate_composite_sentiment(sentiment_cycle)

            # 极端贪婪状态
            extreme_greed = composite_sentiment > self.sentiment_threshold
            extreme_states['extreme_greed'] = extreme_greed.astype(int)

            # 极端恐慌状态
            extreme_fear = composite_sentiment < -self.sentiment_threshold
            extreme_states['extreme_fear'] = extreme_fear.astype(int)

            # 情绪过热状态
            emotional_overheating = composite_sentiment.rolling(window=5).mean() > self.sentiment_threshold * 0.8
            extreme_states['emotional_overheating'] = emotional_overheating.astype(int)

            # 情绪冰点状态
            emotional_freezing = composite_sentiment.rolling(window=5).mean() < -self.sentiment_threshold * 0.8
            extreme_states['emotional_freezing'] = emotional_freezing.astype(int)

        # 周期相位极端状态
        if not cycle_phase.empty:
            # 周期顶部（情绪最高点）
            cycle_top = (cycle_phase > 0.4) & (cycle_phase < 0.6)
            extreme_states['cycle_top'] = cycle_top.astype(int)

            # 周期底部（情绪最低点）
            cycle_bottom = (cycle_phase > 0.9) | (cycle_phase < 0.1)
            extreme_states['cycle_bottom'] = cycle_bottom.astype(int)

        return extreme_states

    def calculate_psychological_bias_metrics(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算心理偏差指标

        Args:
            data: 市场数据

        Returns:
            心理偏差指标
        """
        bias_metrics = {}

        if 'close' not in data.columns:
            return bias_metrics

        prices = data['close']
        returns = prices.pct_change().fillna(0)

        # 1. 损失厌恶偏差
        # 下跌时的反应强度与上涨时的反应强度对比
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        if len(positive_returns) > 0 and len(negative_returns) > 0:
            loss_aversion_ratio = abs(negative_returns.mean()) / positive_returns.mean()
            loss_aversion_sentiment = -np.tanh((loss_aversion_ratio - 1) * 2)  # 损失厌恶程度

            # 滚动计算
            rolling_loss_aversion = pd.Series(0.0, index=prices.index)
            for i in range(20, len(returns)):
                window_returns = returns.iloc[i-20:i]
                pos_ret = window_returns[window_returns > 0]
                neg_ret = window_returns[window_returns < 0]

                if len(pos_ret) > 0 and len(neg_ret) > 0:
                    ratio = abs(neg_ret.mean()) / pos_ret.mean()
                    rolling_loss_aversion.iloc[i] = -np.tanh((ratio - 1) * 2)

            bias_metrics['loss_aversion_bias'] = rolling_loss_aversion

        # 2. 过度自信偏差
        # 成交量异常增加时的价格持续性
        if 'volume' in data.columns:
            volume = data['volume']
            volume_zscore = zscore(volume.fillna(volume.mean()))
            high_volume_periods = volume_zscore > 1.0

            # 高成交量期间的价格动量持续性
            price_momentum = returns.rolling(window=5).mean().fillna(0)
            overconfidence = high_volume_periods.astype(int) * price_momentum
            bias_metrics['overconfidence_bias'] = self.normalize_sentiment(overconfidence)

        # 3. 锚定效应
        # 价格对历史高点的依赖程度
        price_high_52w = prices.rolling(window=252).max()
        anchoring_effect = prices / price_high_52w
        bias_metrics['anchoring_bias'] = self.normalize_sentiment(anchoring_effect.fillna(0) - 0.5)

        # 4. 羊群效应
        # 价格与成交量的同步性
        if 'volume' in data.columns:
            volume = data['volume']
            price_volume_corr = returns.rolling(window=20).corr(volume.pct_change().fillna(0))
            herding_effect = price_volume_corr.fillna(0)
            bias_metrics['herding_bias'] = herding_effect

        return bias_metrics

    def calculate_psychology_momentum(self, data: pd.DataFrame) -> pd.Series:
        """
        计算心理动能

        Args:
            data: 市场数据

        Returns:
            心理动能序列
        """
        if 'close' not in data.columns:
            return pd.Series()

        # 计算情绪周期
        sentiment_cycle = self.calculate_market_sentiment_cycle(data)

        if not sentiment_cycle:
            return pd.Series(0.0, index=data.index)

        # 综合情绪指数
        composite_sentiment = self.calculate_composite_sentiment(sentiment_cycle)

        # 计算情绪动能
        sentiment_momentum = composite_sentiment.diff().fillna(0)

        # 计算情绪加速度
        sentiment_acceleration = sentiment_momentum.diff().fillna(0)

        # 心理动能 = 情绪方向 * 情绪强度 * 情绪加速度
        psychology_momentum = np.sign(composite_sentiment) * abs(sentiment_momentum) * (1 + sentiment_acceleration)

        # 标准化
        psychology_momentum = self.normalize_sentiment(psychology_momentum)

        return psychology_momentum

    def normalize_sentiment(self, series: pd.Series) -> pd.Series:
        """
        标准化情绪指标

        Args:
            series: 输入序列

        Returns:
            标准化后的序列
        """
        if series.empty:
            return series

        # 使用双曲正切函数压缩到[-1,1]
        zscore_series = zscore(series.fillna(0))
        normalized = np.tanh(zscore_series * 0.5)

        return pd.Series(normalized, index=series.index)

    def calculate_composite_sentiment(self, sentiment_indicators: Dict[str, pd.Series]) -> pd.Series:
        """
        计算综合情绪指数

        Args:
            sentiment_indicators: 各类情绪指标

        Returns:
            综合情绪指数
        """
        if not sentiment_indicators:
            return pd.Series()

        # 找到共同的索引
        common_index = None
        for indicator in sentiment_indicators.values():
            if not indicator.empty:
                if common_index is None:
                    common_index = indicator.index
                else:
                    common_index = common_index.intersection(indicator.index)

        if common_index is None or len(common_index) == 0:
            return pd.Series()

        # 计算加权平均
        composite_sentiment = pd.Series(0.0, index=common_index)
        total_weight = 0

        for name, indicator in sentiment_indicators.items():
            if not indicator.empty and len(indicator) > 0:
                # 根据指标类型分配权重
                if 'price_momentum' in name:
                    weight = 0.3
                elif 'volume' in name:
                    weight = 0.25
                elif 'volatility' in name:
                    weight = 0.2
                elif 'divergence' in name:
                    weight = 0.15
                else:
                    weight = 0.1

                aligned_indicator = indicator.reindex(common_index).fillna(0)
                composite_sentiment += aligned_indicator * weight
                total_weight += weight

        if total_weight > 0:
            composite_sentiment /= total_weight

        return composite_sentiment.fillna(0)

    def analyze_investor_psychology(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析投资者心理

        Args:
            data: 市场数据

        Returns:
            投资者心理分析结果
        """
        results = {}

        # 基础情绪周期
        sentiment_cycle = self.calculate_market_sentiment_cycle(data)
        results.update(sentiment_cycle)

        # 心理周期相位
        cycle_phase = self.calculate_psychology_cycle_phase(data)
        results['cycle_phase'] = cycle_phase

        # 极端心理状态
        extreme_states = self.detect_extreme_psychology_states(data)
        results.update(extreme_states)

        # 心理偏差指标
        bias_metrics = self.calculate_psychological_bias_metrics(data)
        results.update(bias_metrics)

        # 心理动能
        psychology_momentum = self.calculate_psychology_momentum(data)
        if not psychology_momentum.empty:
            results['psychology_momentum'] = psychology_momentum

        # 计算综合情绪指数
        if sentiment_cycle:
            composite_sentiment = self.calculate_composite_sentiment(sentiment_cycle)
            results['composite_sentiment'] = composite_sentiment

        # 生成交易信号
        results['signals'] = self.generate_psychology_signals(results)

        return results

    def generate_psychology_signals(self, psychology_analysis: Dict[str, Any]) -> pd.Series:
        """
        基于心理分析生成交易信号

        Args:
            psychology_analysis: 心理分析结果

        Returns:
            交易信号
        """
        if 'composite_sentiment' not in psychology_analysis:
            return pd.Series()

        composite_sentiment = psychology_analysis['composite_sentiment']
        signals = pd.Series(0, index=composite_sentiment.index)

        # 极端情绪信号
        extreme_greed = psychology_analysis.get('extreme_greed', pd.Series(0, index=composite_sentiment.index))
        extreme_fear = psychology_analysis.get('extreme_fear', pd.Series(0, index=composite_sentiment.index))

        # 逆向投资信号
        reverse_buy = extreme_fear.astype(bool)
        reverse_sell = extreme_greed.astype(bool)

        # 心理动能确认
        psychology_momentum = psychology_analysis.get('psychology_momentum', pd.Series(0, index=composite_sentiment.index))
        momentum_confirm_buy = (psychology_momentum > 0) & (composite_sentiment < -0.3)
        momentum_confirm_sell = (psychology_momentum < 0) & (composite_sentiment > 0.3)

        # 周期相位信号
        cycle_phase = psychology_analysis.get('cycle_phase', pd.Series(0, index=composite_sentiment.index))
        cycle_bottom = psychology_analysis.get('cycle_bottom', pd.Series(0, index=composite_sentiment.index))
        cycle_top = psychology_analysis.get('cycle_top', pd.Series(0, index=composite_sentiment.index))

        # 心理偏差确认
        loss_aversion = psychology_analysis.get('loss_aversion_bias', pd.Series(0, index=composite_sentiment.index))
        herding_bias = psychology_analysis.get('herding_bias', pd.Series(0, index=composite_sentiment.index))

        # 综合信号逻辑
        # 强烈买入信号：极端恐慌 + 心理动能反转 + 周期底部
        strong_buy = reverse_buy & momentum_confirm_buy & cycle_bottom.astype(bool)

        # 买入信号：恐慌情绪 + 心理动能改善
        buy_signal = reverse_buy | momentum_confirm_buy

        # 强烈卖出信号：极端贪婪 + 心理动能转弱 + 周期顶部
        strong_sell = reverse_sell & momentum_confirm_sell & cycle_top.astype(bool)

        # 卖出信号：贪婪情绪 + 心理动能减弱
        sell_signal = reverse_sell | momentum_confirm_sell

        # 心理偏差调整
        high_loss_aversion = loss_aversion < -0.5
        high_herding = herding_bias > 0.5

        # 在高损失厌恶和高羊群效应时，信号减弱
        signals[strong_buy & ~high_loss_aversion] = 3
        signals[buy_signal & ~high_loss_aversion] = 2
        signals[strong_sell & ~high_herding] = -3
        signals[sell_signal & ~high_herding] = -2

        return signals

    def get_trading_recommendations(self, psychology_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            psychology_analysis: 心理分析结果

        Returns:
            交易建议DataFrame
        """
        # 找到一个指标序列作为索引
        index_keys = [k for k, v in psychology_analysis.items() if isinstance(v, pd.Series) and not v.empty]
        if not index_keys:
            return pd.DataFrame()

        index = psychology_analysis[index_keys[0]].index
        recommendations = pd.DataFrame(index=index)

        # 添加心理分析指标
        for key, value in psychology_analysis.items():
            if isinstance(value, pd.Series):
                recommendations[key] = value

        # 交易信号
        recommendations['signals'] = psychology_analysis.get('signals', pd.Series(0, index=index))

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 极端恐慌中的逆向机会',
            2: '买入 - 恐慌情绪中的反转机会',
            0: '持有 - 心理状态平衡',
            -2: '卖出 - 贪婪情绪中的风险',
            -3: '强烈卖出 - 极度贪婪后的调整'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 心理周期阶段
        def classify_psychology_cycle(row):
            phase = row.get('cycle_phase', 0)
            sentiment = row.get('composite_sentiment', 0)

            if phase > 0.4 and phase < 0.6 and sentiment > 0.5:
                return 'euphoria_phase'  # 亢奋期
            elif phase > 0.7 and sentiment > 0:
                return 'anxiety_phase'  # 焦虑期
            elif phase > 0.8 and sentiment < 0:
                return 'panic_phase'  # 恐慌期
            elif phase < 0.2 and sentiment < 0:
                return 'despair_phase'  # 绝望期
            elif phase < 0.3 and sentiment > 0:
                return 'hope_phase'  # 希望期
            else:
                return 'normal_psychology'

        recommendations['psychology_cycle_phase'] = recommendations.apply(classify_psychology_cycle, axis=1)

        # 仓位建议
        position_map = {3: 0.6, 2: 0.4, 0: 0.2, -2: 0.1, -3: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 心理偏差调整
        loss_aversion = recommendations.get('loss_aversion_bias', 0)
        overconfidence = recommendations.get('overconfidence_bias', 0)

        # 损失厌恶时减少仓位
        loss_aversion_adjustment = 1 + loss_aversion * 0.5  # 损失厌恶为负，所以减少仓位
        recommendations['position_size'] *= loss_aversion_adjustment

        # 过度自信时进一步减少仓位
        overconfidence_adjustment = 1 - abs(overconfidence) * 0.3
        recommendations['position_size'] *= overconfidence_adjustment

        # 确保仓位在合理范围内
        recommendations['position_size'] = recommendations['position_size'].clip(0, 1)

        # 策略建议
        def get_psychology_strategy(cycle_phase, signal_strength):
            if cycle_phase in ['euphoria_phase', 'anxiety_phase']:
                return 'contrarian'  # 逆向策略
            elif cycle_phase in ['panic_phase', 'despair_phase']:
                return 'value_investing'  # 价值投资
            elif cycle_phase == 'hope_phase':
                return 'momentum'  # 动量策略
            else:
                return 'balanced'

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_psychology_strategy(x['psychology_cycle_phase'], abs(x['signals'])),
            axis=1
        )

        # 心理风险评估
        def assess_psychology_risk(row):
            risk_score = 0

            # 极端情绪风险
            if abs(row.get('composite_sentiment', 0)) > 0.8:
                risk_score += 0.4

            # 心理偏差风险
            if abs(row.get('loss_aversion_bias', 0)) > 0.6:
                risk_score += 0.3

            if abs(row.get('overconfidence_bias', 0)) > 0.6:
                risk_score += 0.2

            # 羊群效应风险
            if abs(row.get('herding_bias', 0)) > 0.5:
                risk_score += 0.1

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['risk_level'] = recommendations.apply(assess_psychology_risk, axis=1)

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成模拟市场数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')

    # 生成价格序列（带有心理周期）
    base_price = 100
    price_changes = np.random.randn(500) * 0.02

    # 添加心理周期影响
    psychology_cycle = np.sin(np.linspace(0, 4*np.pi, 500)) * 0.01
    price_changes += psychology_cycle

    # 构建价格
    prices = [base_price]
    for change in price_changes:
        prices.append(prices[-1] * (1 + change))

    # 生成成交量（与情绪相关）
    base_volume = 1000000
    volume_changes = np.random.lognormal(0, 0.3, 500)
    volume_changes *= (1 + psychology_cycle * 2)  # 情绪影响成交量
    volumes = base_volume * volume_changes

    # 创建数据框
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices[1:],
        'volume': volumes
    }).set_index('date')

    # 创建投资者心理周期分析器
    ipc = InvestorPsychologyCycle(psychology_windows=[10, 20, 50])

    # 分析投资者心理
    print("分析投资者心理...")
    psychology_results = ipc.analyze_investor_psychology(market_data)

    # 显示主要指标
    print(f"综合情绪指数: {psychology_results['composite_sentiment'].iloc[-1]:.3f}")
    print(f"心理周期相位: {psychology_results['cycle_phase'].iloc[-1]:.3f}")
    print(f"心理动能: {psychology_results['psychology_momentum'].iloc[-1]:.3f}")

    if 'extreme_greed' in psychology_results:
        print(f"极端贪婪状态: {psychology_results['extreme_greed'].iloc[-1]}")
        print(f"极端恐慌状态: {psychology_results['extreme_fear'].iloc[-1]}")

    if 'loss_aversion_bias' in psychology_results:
        print(f"损失厌恶偏差: {psychology_results['loss_aversion_bias'].iloc[-1]:.3f}")
        print(f"过度自信偏差: {psychology_results['overconfidence_bias'].iloc[-1]:.3f}")

    print(f"当前信号: {psychology_results['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = ipc.get_trading_recommendations(psychology_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['composite_sentiment', 'cycle_phase', 'psychology_cycle_phase',
                                          'signal_description', 'strategy', 'position_size', 'risk_level']])