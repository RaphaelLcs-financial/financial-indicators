"""
Volume Weighted MACD (VW MACD)
================================

成交量加权的MACD指标，在传统MACD的基础上引入成交量权重，
使MACD信号更能反映真实的市场强度和资金流向。VW MACD能够
更好地识别趋势强度、发现潜在的背离，并提供更可靠的交易信号。

特点：
- 成交量加权计算
- 动态信号线平滑
- 多重背离检测
- 趋势强度评估
- 成交量确认机制
- 信号质量评分

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional


class VolumeWeightedMACD:
    """
    Volume Weighted MACD - 成交量加权MACD

    通过成交量加权改进传统MACD，提供更准确的趋势信号和
    市场强度评估。
    """

    def __init__(self, fast_period: int = 12, slow_period: int = 26,
                 signal_period: int = 9, volume_smoothing: int = 10):
        """
        初始化成交量加权MACD

        Args:
            fast_period: 快速EMA周期，默认12
            slow_period: 慢速EMA周期，默认26
            signal_period: 信号线周期，默认9
            volume_smoothing: 成交量平滑周期，默认10
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signal_period = signal_period
        self.volume_smoothing = volume_smoothing
        self.name = f"VW MACD ({fast_period}, {slow_period}, {signal_period})"
        self.category = "momentum"

    def calculate_vw_macd(self, prices: Union[List[float], pd.Series],
                          volume: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        计算成交量加权MACD

        Args:
            prices: 价格序列
            volume: 成交量序列

        Returns:
            包含所有指标的字典
        """
        prices_series = pd.Series(prices)
        volume_series = pd.Series(volume)

        # 计算成交量权重
        volume_weights = self.calculate_volume_weights(volume_series)

        # 计算成交量加权EMA
        vw_ema_fast = self.calculate_volume_weighted_ema(prices_series, volume_weights, self.fast_period)
        vw_ema_slow = self.calculate_volume_weighted_ema(prices_series, volume_weights, self.slow_period)

        # 计算MACD线
        macd_line = vw_ema_fast - vw_ema_slow

        # 计算信号线
        signal_line = self.calculate_volume_weighted_ema(macd_line, volume_weights, self.signal_period)

        # 计算直方图
        histogram = macd_line - signal_line

        # 计算成交量确认指标
        volume_confirmation = self.calculate_volume_confirmation(volume_series, macd_line)

        # 计算MACD动量
        macd_momentum = self.calculate_macd_momentum(macd_line)

        # 计算趋势强度
        trend_strength = self.calculate_trend_strength(macd_line, signal_line)

        # 生成交易信号
        signals = self.generate_signals(macd_line, signal_line, histogram, volume_confirmation)

        # 计算信号质量
        signal_quality = self.calculate_signal_quality(signals, volume_confirmation, trend_strength)

        # 识别背离
        divergence = self.calculate_divergence(prices_series, macd_line, signal_line)

        return {
            'vw_macd': macd_line,
            'vw_signal': signal_line,
            'vw_histogram': histogram,
            'volume_weights': volume_weights,
            'volume_confirmation': volume_confirmation,
            'macd_momentum': macd_momentum,
            'trend_strength': trend_strength,
            'signals': signals,
            'signal_quality': signal_quality,
            'divergence': divergence
        }

    def calculate_volume_weights(self, volume: pd.Series) -> pd.Series:
        """
        计算成交量权重

        Args:
            volume: 成交量序列

        Returns:
            成交量权重序列
        """
        # 计算成交量移动平均
        volume_ma = volume.rolling(window=self.volume_smoothing).mean()

        # 计算成交量比率
        volume_ratio = volume / volume_ma

        # 标准化权重
        volume_weights = volume_ratio / volume_ratio.rolling(window=20).mean()

        return volume_weights.fillna(1.0)

    def calculate_volume_weighted_ema(self, prices: pd.Series, volume_weights: pd.Series,
                                    period: int) -> pd.Series:
        """
        计算成交量加权EMA

        Args:
            prices: 价格序列
            volume_weights: 成交量权重
            period: 计算周期

        Returns:
            成交量加权EMA序列
        """
        # 传统EMA权重
        alpha = 2 / (period + 1)

        # 计算成交量加权EMA
        vw_ema = pd.Series(0.0, index=prices.index)

        for i in range(len(prices)):
            if i == 0:
                vw_ema.iloc[i] = prices.iloc[i]
            else:
                # 结合EMA权重和成交量权重
                weight_factor = alpha * volume_weights.iloc[i]
                vw_ema.iloc[i] = (weight_factor * prices.iloc[i] +
                                 (1 - weight_factor) * vw_ema.iloc[i-1])

        return vw_ema

    def calculate_volume_confirmation(self, volume: pd.Series, macd_line: pd.Series) -> pd.Series:
        """
        计算成交量确认指标

        Args:
            volume: 成交量序列
            macd_line: MACD线序列

        Returns:
            成交量确认序列 (0-100)
        """
        # 计算成交量强度
        volume_ma = volume.rolling(window=20).mean()
        volume_strength = volume / volume_ma

        # 计算MACD变化强度
        macd_change = macd_line.diff().abs()
        macd_change_ma = macd_change.rolling(window=10).mean()
        macd_intensity = macd_change / macd_change_ma

        # 成交量确认指标
        confirmation = volume_strength * macd_intensity

        # 平滑处理
        confirmation = confirmation.rolling(window=5).mean()

        # 标准化到0-100
        min_val = confirmation.rolling(window=50).min()
        max_val = confirmation.rolling(window=50).max()
        normalized = (confirmation - min_val) / (max_val - min_val) * 100

        return normalized.fillna(50)

    def calculate_macd_momentum(self, macd_line: pd.Series) -> pd.Series:
        """
        计算MACD动量

        Args:
            macd_line: MACD线序列

        Returns:
            MACD动量序列
        """
        # 计算MACD变化率
        macd_change = macd_line.diff()

        # 计算动量
        momentum = macd_change.rolling(window=5).mean()

        return momentum.fillna(0)

    def calculate_trend_strength(self, macd_line: pd.Series, signal_line: pd.Series) -> pd.Series:
        """
        计算趋势强度

        Args:
            macd_line: MACD线序列
            signal_line: 信号线序列

        Returns:
            趋势强度序列 (0-100)
        """
        # 计算MACD与信号线的距离
        distance = abs(macd_line - signal_line)

        # 计算MACD斜率
        macd_slope = macd_line.diff().rolling(window=5).mean()

        # 计算趋势强度
        trend_strength = (
            (distance * 10) +  # 距离权重
            (macd_slope.abs() * 1000)  # 斜率权重
        )

        # 标准化到0-100
        trend_strength = trend_strength.clip(0, 100)

        return trend_strength.fillna(50)

    def generate_signals(self, macd_line: pd.Series, signal_line: pd.Series,
                        histogram: pd.Series, volume_confirmation: pd.Series) -> pd.Series:
        """
        生成交易信号

        Args:
            macd_line: MACD线序列
            signal_line: 信号线序列
            histogram: 直方图序列
            volume_confirmation: 成交量确认

        Returns:
            信号序列 (2=强买入, 1=买入, 0=持有, -1=卖出, -2=强卖出)
        """
        signals = pd.Series(0, index=macd_line.index)

        # MACD线与信号线交叉
        macd_above_signal = macd_line > signal_line
        macd_cross_above = macd_above_signal & ~macd_above_signal.shift(1)
        macd_cross_below = ~macd_above_signal & macd_above_signal.shift(1)

        # 直方图变化
        histogram_increasing = histogram > histogram.shift(1)
        histogram_decreasing = histogram < histogram.shift(1)

        # 零轴穿越
        macd_above_zero = macd_line > 0
        macd_cross_zero_above = macd_above_zero & ~macd_above_zero.shift(1)
        macd_cross_zero_below = ~macd_above_zero & macd_above_zero.shift(1)

        # 成交量确认
        volume_confirmed = volume_confirmation > 50

        # 强买入信号：MACD上穿信号线+成交量确认+直方图增长
        strong_buy = (
            macd_cross_above &
            volume_confirmed &
            histogram_increasing &
            macd_above_zero
        )

        # 买入信号：MACD上穿信号线或上穿零轴
        buy = (
            (macd_cross_above & volume_confirmed) |
            (macd_cross_zero_above & volume_confirmed)
        )

        # 强卖出信号：MACD下穿信号线+成交量确认+直方图减少
        strong_sell = (
            macd_cross_below &
            volume_confirmed &
            histogram_decreasing &
            ~macd_above_zero
        )

        # 卖出信号：MACD下穿信号线或下穿零轴
        sell = (
            (macd_cross_below & volume_confirmed) |
            (macd_cross_zero_below & volume_confirmed)
        )

        signals[strong_buy] = 2
        signals[buy] = 1
        signals[strong_sell] = -2
        signals[sell] = -1

        return signals

    def calculate_signal_quality(self, signals: pd.Series, volume_confirmation: pd.Series,
                               trend_strength: pd.Series) -> pd.Series:
        """
        计算信号质量

        Args:
            signals: 信号序列
            volume_confirmation: 成交量确认
            trend_strength: 趋势强度

        Returns:
            信号质量序列 (0-100)
        """
        # 基础质量评分
        base_quality = abs(signals) * 50  # 信号强度基础分

        # 成交量确认加分
        volume_bonus = (volume_confirmation - 50) * 0.5

        # 趋势强度加分
        trend_bonus = (trend_strength - 50) * 0.3

        # 综合质量评分
        signal_quality = base_quality + volume_bonus + trend_bonus

        # 限制范围
        signal_quality = signal_quality.clip(0, 100)

        return signal_quality.fillna(50)

    def calculate_divergence(self, prices: pd.Series, macd_line: pd.Series,
                           signal_line: pd.Series) -> Dict[str, pd.Series]:
        """
        计算背离

        Args:
            prices: 价格序列
            macd_line: MACD线序列
            signal_line: 信号线序列

        Returns:
            背离检测结果
        """
        divergence = {
            'regular_bullish': pd.Series(0, index=prices.index),
            'regular_bearish': pd.Series(0, index=prices.index),
            'hidden_bullish': pd.Series(0, index=prices.index),
            'hidden_bearish': pd.Series(0, index=prices.index)
        }

        # 识别价格和MACD的极值点
        for i in range(30, len(prices)):
            # 常规多头背离（价格创新低，MACD未创新低）
            if (self.is_lower_low(prices, i) and
                not self.is_lower_low(macd_line, i) and
                macd_line.iloc[i] < 0):
                divergence['regular_bullish'].iloc[i] = 1

            # 常规空头背离（价格创新高，MACD未创新高）
            elif (self.is_higher_high(prices, i) and
                  not self.is_higher_high(macd_line, i) and
                  macd_line.iloc[i] > 0):
                divergence['regular_bearish'].iloc[i] = 1

            # 隐藏多头背离（价格创较高低点，MACD创较低低点）
            elif (self.is_higher_low(prices, i) and
                  self.is_lower_low(macd_line, i) and
                  macd_line.iloc[i] > 0):
                divergence['hidden_bullish'].iloc[i] = 1

            # 隐藏空头背离（价格创较低高点，MACD创较高高点）
            elif (self.is_lower_high(prices, i) and
                  self.is_higher_high(macd_line, i) and
                  macd_line.iloc[i] < 0):
                divergence['hidden_bearish'].iloc[i] = 1

        return divergence

    def is_lower_low(self, series: pd.Series, index: int, lookback: int = 10) -> bool:
        """检查是否为更低低点"""
        if index < lookback:
            return False
        recent_values = series.iloc[index-lookback:index+1]
        return series.iloc[index] == recent_values.min()

    def is_higher_high(self, series: pd.Series, index: int, lookback: int = 10) -> bool:
        """检查是否为更高高点"""
        if index < lookback:
            return False
        recent_values = series.iloc[index-lookback:index+1]
        return series.iloc[index] == recent_values.max()

    def is_higher_low(self, series: pd.Series, index: int, lookback: int = 10) -> bool:
        """检查是否为较高低点"""
        if index < lookback:
            return False
        return (series.iloc[index] > series.iloc[index-lookback] and
                series.iloc[index] < series.iloc[index-1])

    def is_lower_high(self, series: pd.Series, index: int, lookback: int = 10) -> bool:
        """检查是否为较低高点"""
        if index < lookback:
            return False
        return (series.iloc[index] < series.iloc[index-lookback] and
                series.iloc[index] > series.iloc[index-1])

    def get_comprehensive_analysis(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        获取综合分析结果

        Args:
            data: 包含OHLCV数据的字典

        Returns:
            综合分析结果
        """
        prices = data['close']
        volume = data['volume']

        # 计算VW MACD
        vw_macd_data = self.calculate_vw_macd(prices, volume)

        # 计算多时间框架分析
        mtf_analysis = self.calculate_mtf_analysis(prices, volume)

        # 计算市场情绪
        market_sentiment = self.calculate_market_sentiment(vw_macd_data)

        # 生成高级信号
        advanced_signals = self.generate_advanced_signals(vw_macd_data, mtf_analysis)

        # 计算动能指标
        momentum_metrics = self.calculate_momentum_metrics(vw_macd_data)

        return {
            'vw_macd': vw_macd_data['vw_macd'],
            'vw_signal': vw_macd_data['vw_signal'],
            'vw_histogram': vw_macd_data['vw_histogram'],
            'volume_confirmation': vw_macd_data['volume_confirmation'],
            'trend_strength': vw_macd_data['trend_strength'],
            'signals': advanced_signals,
            'signal_quality': vw_macd_data['signal_quality'],
            'divergence': vw_macd_data['divergence'],
            'mtf_analysis': mtf_analysis,
            'market_sentiment': market_sentiment,
            'momentum_metrics': momentum_metrics
        }

    def calculate_mtf_analysis(self, prices: pd.Series, volume: pd.Series) -> Dict[str, Any]:
        """
        计算多时间框架分析

        Args:
            prices: 价格序列
            volume: 成交量序列

        Returns:
            多时间框架分析结果
        """
        # 不同周期的VW MACD
        periods = [
            {'fast': 8, 'slow': 17, 'signal': 6},
            {'fast': 12, 'slow': 26, 'signal': 9},
            {'fast': 16, 'slow': 35, 'signal': 12}
        ]

        mtf_results = {}

        for i, period_config in enumerate(periods):
            # 临时修改参数
            original_fast = self.fast_period
            original_slow = self.slow_period
            original_signal = self.signal_period

            self.fast_period = period_config['fast']
            self.slow_period = period_config['slow']
            self.signal_period = period_config['signal']

            # 计算该周期的VW MACD
            temp_data = self.calculate_vw_macd(prices, volume)

            mtf_results[f'timeframe_{i+1}'] = {
                'macd': temp_data['vw_macd'],
                'signal': temp_data['vw_signal'],
                'histogram': temp_data['vw_histogram'],
                'trend': 'bullish' if temp_data['vw_macd'].iloc[-1] > temp_data['vw_signal'].iloc[-1] else 'bearish'
            }

            # 恢复原始参数
            self.fast_period = original_fast
            self.slow_period = original_slow
            self.signal_period = original_signal

        # 计算多时间框架共识
        trends = [result['trend'] for result in mtf_results.values()]
        bullish_count = trends.count('bullish')
        bearish_count = trends.count('bearish')

        if bullish_count >= 2:
            consensus = 'bullish'
        elif bearish_count >= 2:
            consensus = 'bearish'
        else:
            consensus = 'mixed'

        return {
            'timeframes': mtf_results,
            'consensus': consensus,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count
        }

    def calculate_market_sentiment(self, vw_macd_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        计算市场情绪

        Args:
            vw_macd_data: VW MACD数据

        Returns:
            市场情绪指标
        """
        macd = vw_macd_data['vw_macd']
        signal = vw_macd_data['vw_signal']
        histogram = vw_macd_data['vw_histogram']
        volume_conf = vw_macd_data['volume_confirmation']

        # 计算情绪指标
        latest_macd = macd.iloc[-1]
        latest_signal = signal.iloc[-1]
        latest_histogram = histogram.iloc[-1]
        latest_volume_conf = volume_conf.iloc[-1]

        # 情绪评分
        sentiment_score = 0

        # MACD位置
        if latest_macd > latest_signal:
            sentiment_score += 30
        else:
            sentiment_score -= 30

        # 零轴位置
        if latest_macd > 0:
            sentiment_score += 20
        else:
            sentiment_score -= 20

        # 直方图趋势
        if latest_histogram > 0:
            sentiment_score += 20
        else:
            sentiment_score -= 20

        # 成交量确认
        sentiment_score += (latest_volume_conf - 50) * 0.3

        # 标准化情绪评分
        sentiment_score = max(-100, min(100, sentiment_score))

        # 情绪分类
        if sentiment_score > 50:
            sentiment = 'very_bullish'
        elif sentiment_score > 20:
            sentiment = 'bullish'
        elif sentiment_score > -20:
            sentiment = 'neutral'
        elif sentiment_score > -50:
            sentiment = 'bearish'
        else:
            sentiment = 'very_bearish'

        return {
            'sentiment': sentiment,
            'score': sentiment_score,
            'confidence': latest_volume_conf / 100
        }

    def generate_advanced_signals(self, vw_macd_data: Dict[str, pd.Series],
                                 mtf_analysis: Dict[str, Any]) -> pd.Series:
        """
        生成高级交易信号

        Args:
            vw_macd_data: VW MACD数据
            mtf_analysis: 多时间框架分析

        Returns:
            高级信号序列
        """
        signals = pd.Series(0, index=vw_macd_data['vw_macd'].index)

        # 基础信号
        base_signals = vw_macd_data['signals']

        # 多时间框架确认
        mtf_confirmed = (
            (mtf_analysis['consensus'] in ['bullish', 'bearish']) &
            (base_signals != 0)
        )

        # 背离确认
        has_divergence = (
            (vw_macd_data['divergence']['regular_bullish'] == 1) |
            (vw_macd_data['divergence']['regular_bearish'] == 1) |
            (vw_macd_data['divergence']['hidden_bullish'] == 1) |
            (vw_macd_data['divergence']['hidden_bearish'] == 1)
        )

        # 信号质量过滤
        high_quality = vw_macd_data['signal_quality'] > 60

        # 高级买入信号
        advanced_buy = (
            (base_signals > 0) &
            (mtf_confirmed | has_divergence) &
            high_quality
        )

        # 高级卖出信号
        advanced_sell = (
            (base_signals < 0) &
            (mtf_confirmed | has_divergence) &
            high_quality
        )

        # 背离信号
        divergence_buy = (vw_macd_data['divergence']['regular_bullish'] == 1)
        divergence_sell = (vw_macd_data['divergence']['regular_bearish'] == 1)

        signals[advanced_buy] = 2
        signals[advanced_sell] = -2
        signals[divergence_buy] = 1
        signals[divergence_sell] = -1

        return signals

    def calculate_momentum_metrics(self, vw_macd_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        计算动能指标

        Args:
            vw_macd_data: VW MACD数据

        Returns:
            动能指标字典
        """
        macd = vw_macd_data['vw_macd']
        histogram = vw_macd_data['vw_histogram']

        # 计算MACD动能
        macd_momentum = macd.diff().rolling(window=5).mean().iloc[-1]

        # 计算直方图动能
        histogram_momentum = histogram.diff().rolling(window=5).mean().iloc[-1]

        # 计算动能强度
        momentum_strength = np.sqrt(macd_momentum**2 + histogram_momentum**2)

        # 计算动能方向
        momentum_direction = 'bullish' if (macd_momentum + histogram_momentum) > 0 else 'bearish'

        return {
            'macd_momentum': macd_momentum,
            'histogram_momentum': histogram_momentum,
            'momentum_strength': momentum_strength,
            'momentum_direction': momentum_direction
        }

    def get_trading_recommendations(self, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            analysis: 综合分析结果

        Returns:
            交易建议DataFrame
        """
        recommendations = pd.DataFrame(index=analysis['vw_macd'].index)

        # 信号描述
        recommendations['signal'] = analysis['signals']
        recommendations['signal_description'] = analysis['signals'].map({
            2: '强烈买入 - 多时间框架确认',
            1: '买入 - 背离或动能转强',
            0: '持有 - 观望',
            -1: '卖出 - 背离或动能转弱',
            -2: '强烈卖出 - 多时间框架确认'
        })

        # 市场情绪
        recommendations['market_sentiment'] = analysis['market_sentiment']['sentiment']
        recommendations['sentiment_score'] = analysis['market_sentiment']['score']

        # 多时间框架共识
        recommendations['mtf_consensus'] = analysis['mtf_analysis']['consensus']

        # 仓位建议
        recommendations['position_size'] = analysis['signals'].map({
            2: 0.7,   # 强买入：70%仓位
            1: 0.4,   # 买入：40%仓位
            0: 0.3,   # 持有：30%仓位
            -1: 0.1,  # 卖出：10%仓位
            -2: 0.0   # 强卖出：0仓位
        })

        # 信号质量调整
        quality_adjustment = analysis['signal_quality'] / 100
        recommendations['position_size'] *= quality_adjustment

        # 止损止盈建议
        current_price = analysis['vw_macd'].index.to_series().map(lambda x: analysis['vw_macd'].loc[x])
        recommendations['stop_loss'] = current_price * 0.96
        recommendations['take_profit'] = current_price * 1.08

        # 信号强度
        recommendations['signal_quality'] = analysis['signal_quality']

        # 动能指标
        recommendations['momentum_direction'] = analysis['momentum_metrics']['momentum_direction']
        recommendations['momentum_strength'] = analysis['momentum_metrics']['momentum_strength']

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    volume = np.random.randint(1000000, 5000000, len(dates))

    data = {
        'close': pd.Series(prices, index=dates),
        'volume': pd.Series(volume, index=dates)
    }

    # 创建成交量加权MACD指标
    vw_macd = VolumeWeightedMACD()

    # 计算指标
    results = vw_macd.calculate_vw_macd(data['close'], data['volume'])

    print("Volume Weighted MACD 结果:")
    print(f"VW MACD: {results['vw_macd'].iloc[-1]:.4f}")
    print(f"VW Signal: {results['vw_signal'].iloc[-1]:.4f}")
    print(f"VW Histogram: {results['vw_histogram'].iloc[-1]:.4f}")
    print(f"成交量确认: {results['volume_confirmation'].iloc[-1]:.2f}")
    print(f"趋势强度: {results['trend_strength'].iloc[-1]:.2f}")
    print(f"信号质量: {results['signal_quality'].iloc[-1]:.2f}")
    print(f"当前信号: {results['signals'].iloc[-1]}")

    # 检查背离
    divergence = results['divergence']
    print(f"\n背离状态:")
    print(f"常规多头背离: {divergence['regular_bullish'].iloc[-1]}")
    print(f"常规空头背离: {divergence['regular_bearish'].iloc[-1]}")
    print(f"隐藏多头背离: {divergence['hidden_bullish'].iloc[-1]}")
    print(f"隐藏空头背离: {divergence['hidden_bearish'].iloc[-1]}")

    # 获取综合分析
    comprehensive = vw_macd.get_comprehensive_analysis(data)
    print("\n综合分析结果:")
    print(f"市场情绪: {comprehensive['market_sentiment']['sentiment']}")
    print(f"情绪评分: {comprehensive['market_sentiment']['score']:.2f}")
    print(f"多时间框架共识: {comprehensive['mtf_analysis']['consensus']}")
    print(f"动能方向: {comprehensive['momentum_metrics']['momentum_direction']}")
    print(f"动能强度: {comprehensive['momentum_metrics']['momentum_strength']:.4f}")
    print(f"高级信号: {comprehensive['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = vw_macd.get_trading_recommendations(comprehensive)
    print("\n最新交易建议:")
    print(recommendations.iloc[-1])