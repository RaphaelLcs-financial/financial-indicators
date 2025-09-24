"""
Woodies CCI with Volume Confirmation
===================================

Woodies CCI的改进版本，结合成交量确认机制。传统的Woodies CCI使用
两条不同周期的CCI线来识别趋势和信号，本版本增加了成交量确认、
动态参数调整和多时间框架分析，提高信号的可靠性。

特点：
- 双CCI线系统（快线+慢线）
- 成交量确认机制
- 零线背离检测
- 趋势线突破识别
- 动态参数调整
- 多时间框架确认

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional


class WoodiesCCIVolume:
    """
    Woodies CCI with Volume Confirmation - 伍迪CCI成交量确认版

    改进的Woodies CCI指标，结合成交量确认和多时间框架分析，
    提供更可靠的交易信号。
    """

    def __init__(self, fast_period: int = 14, slow_period: int = 30,
                 volume_period: int = 20, trendline_period: int = 50):
        """
        初始化Woodies CCI成交量确认指标

        Args:
            fast_period: 快速CCI周期，默认14
            slow_period: 慢速CCI周期，默认30
            volume_period: 成交量确认周期，默认20
            trendline_period: 趋势线周期，默认50
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.volume_period = volume_period
        self.trendline_period = trendline_period
        self.name = f"Woodies CCI Volume ({fast_period}, {slow_period})"
        self.category = "momentum"

    def calculate_woodies_cci_volume(self, high: Union[List[float], pd.Series],
                                   low: Union[List[float], pd.Series],
                                   close: Union[List[float], pd.Series],
                                   volume: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        计算Woodies CCI成交量确认指标

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列

        Returns:
            包含所有指标的字典
        """
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        volume_series = pd.Series(volume)

        # 计算TP (Typical Price)
        tp = (high_series + low_series + close_series) / 3

        # 计算双CCI
        cci_fast = self.calculate_cci(tp, high_series, low_series, close_series, self.fast_period)
        cci_slow = self.calculate_cci(tp, high_series, low_series, close_series, self.slow_period)

        # 计算成交量确认
        volume_confirmation = self.calculate_volume_confirmation(volume_series, close_series)

        # 计算趋势线
        trendline = self.calculate_trendline(cci_slow)

        # 计算零线背离
        divergence = self.calculate_divergence(close_series, cci_slow)

        # 计算信号强度
        signal_strength = self.calculate_signal_strength(cci_fast, cci_slow, volume_confirmation)

        # 生成交易信号
        signals = self.generate_woodies_signals(cci_fast, cci_slow, volume_confirmation, trendline)

        # 识别Woodies模式
        patterns = self.identify_woodies_patterns(cci_fast, cci_slow, trendline)

        return {
            'cci_fast': cci_fast,
            'cci_slow': cci_slow,
            'volume_confirmation': volume_confirmation,
            'trendline': trendline,
            'divergence': divergence,
            'signal_strength': signal_strength,
            'signals': signals,
            'patterns': patterns
        }

    def calculate_cci(self, tp: pd.Series, high: pd.Series, low: pd.Series,
                     close: pd.Series, period: int) -> pd.Series:
        """
        计算CCI

        Args:
            tp: 典型价格序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: CCI周期

        Returns:
            CCI序列
        """
        # 计算简单移动平均
        sma = tp.rolling(window=period).mean()

        # 计算平均偏差
        mad = tp.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))

        # 计算CCI
        cci = (tp - sma) / (0.015 * mad)

        return cci

    def calculate_volume_confirmation(self, volume: pd.Series, close: pd.Series) -> pd.Series:
        """
        计算成交量确认指标

        Args:
            volume: 成交量序列
            close: 价格序列

        Returns:
            成交量确认序列 (0-100)
        """
        # 计算成交量比率
        volume_ma = volume.rolling(window=self.volume_period).mean()
        volume_ratio = volume / volume_ma

        # 计算价格变化
        price_change = close.pct_change().abs()

        # 计算成交量强度
        volume_strength = volume_ratio * (1 + price_change * 10)

        # 平滑处理
        volume_confirmation = volume_strength.rolling(window=5).mean()

        # 标准化到0-100
        min_val = volume_confirmation.rolling(window=50).min()
        max_val = volume_confirmation.rolling(window=50).max()
        normalized = (volume_confirmation - min_val) / (max_val - min_val) * 100

        return normalized.fillna(50)

    def calculate_trendline(self, cci_slow: pd.Series) -> pd.Series:
        """
        计算CCI趋势线

        Args:
            cci_slow: 慢速CCI序列

        Returns:
            趋势线序列
        """
        # 使用线性回归计算趋势线
        trendline = cci_slow.rolling(window=self.trendline_period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] * len(x) + np.polyfit(range(len(x)), x, 1)[1]
            if len(x) > 1 else x.iloc[-1]
        )

        return trendline

    def calculate_divergence(self, prices: pd.Series, cci: pd.Series) -> pd.Series:
        """
        计算零线背离

        Args:
            prices: 价格序列
            cci: CCI序列

        Returns:
            背离标记序列 (1=多头背离, -1=空头背离, 0=无背离)
        """
        divergence = pd.Series(0, index=prices.index)

        # 识别价格和CCI的极值点
        price_highs = self.find_peaks(prices, 20)
        price_lows = self.find_troughs(prices, 20)
        cci_highs = self.find_peaks(cci, 20)
        cci_lows = self.find_troughs(cci, 20)

        # 检测多头背离（价格创新低，CCI未创新低）
        for i in range(len(prices)):
            if i >= 40:  # 需要足够的历史数据
                recent_price_low = price_lows.iloc[i-20:i]
                recent_cci_low = cci_lows.iloc[i-20:i]

                if len(recent_price_low) > 0 and len(recent_cci_low) > 0:
                    if (recent_price_low.iloc[-1] < recent_price_low.iloc[-2] and
                        recent_cci_low.iloc[-1] > recent_cci_low.iloc[-2]):
                        divergence.iloc[i] = 1  # 多头背离

        # 检测空头背离（价格创新高，CCI未创新高）
        for i in range(len(prices)):
            if i >= 40:
                recent_price_high = price_highs.iloc[i-20:i]
                recent_cci_high = cci_highs.iloc[i-20:i]

                if len(recent_price_high) > 0 and len(recent_cci_high) > 0:
                    if (recent_price_high.iloc[-1] > recent_price_high.iloc[-2] and
                        recent_cci_high.iloc[-1] < recent_cci_high.iloc[-2]):
                        divergence.iloc[i] = -1  # 空头背离

        return divergence

    def find_peaks(self, series: pd.Series, window: int) -> pd.Series:
        """
        识别峰值

        Args:
            series: 序列
            window: 窗口大小

        Returns:
            峰值标记
        """
        peaks = pd.Series(0, index=series.index)
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window].max():
                peaks.iloc[i] = series.iloc[i]
        return peaks

    def find_troughs(self, series: pd.Series, window: int) -> pd.Series:
        """
        识别谷值

        Args:
            series: 序列
            window: 窗口大小

        Returns:
            谷值标记
        """
        troughs = pd.Series(0, index=series.index)
        for i in range(window, len(series) - window):
            if series.iloc[i] == series.iloc[i-window:i+window].min():
                troughs.iloc[i] = series.iloc[i]
        return troughs

    def calculate_signal_strength(self, cci_fast: pd.Series, cci_slow: pd.Series,
                                 volume_confirmation: pd.Series) -> pd.Series:
        """
        计算信号强度

        Args:
            cci_fast: 快速CCI
            cci_slow: 慢速CCI
            volume_confirmation: 成交量确认

        Returns:
            信号强度序列 (0-100)
        """
        # 计算CCI偏离度
        cci_deviation = abs(cci_fast - cci_slow)

        # 计算CCI强度
        cci_strength = (cci_deviation / 100).clip(0, 1) * 100

        # 综合信号强度
        signal_strength = (cci_strength * 0.7 + volume_confirmation * 0.3)

        return signal_strength.fillna(50)

    def generate_woodies_signals(self, cci_fast: pd.Series, cci_slow: pd.Series,
                                 volume_confirmation: pd.Series, trendline: pd.Series) -> pd.Series:
        """
        生成Woodies CCI交易信号

        Args:
            cci_fast: 快速CCI
            cci_slow: 慢速CCI
            volume_confirmation: 成交量确认
            trendline: 趋势线

        Returns:
            信号序列 (2=强买入, 1=买入, 0=持有, -1=卖出, -2=强卖出)
        """
        signals = pd.Series(0, index=cci_fast.index)

        # Woodies CCI核心规则
        # 1. 零线交叉
        fast_above_zero = cci_fast > 0
        slow_above_zero = cci_slow > 0

        # 2. CCI快慢线关系
        fast_above_slow = cci_fast > cci_slow

        # 3. 趋势线突破
        above_trendline = cci_fast > trendline

        # 4. 成交量确认
        volume_confirmed = volume_confirmation > 50

        # 强买入信号：双线在零线上方且快线突破趋势线，成交量确认
        strong_buy = (
            fast_above_zero &
            slow_above_zero &
            fast_above_slow &
            above_trendline &
            volume_confirmed &
            (cci_fast > 100)
        )

        # 买入信号：快线上穿零线或趋势线
        buy = (
            (fast_above_zero & ~fast_above_zero.shift(1)) |  # 零线上穿
            (above_trendline & ~above_trendline.shift(1))     # 趋势线上穿
        ) & volume_confirmed

        # 强卖出信号：双线在零线下方且快线跌破趋势线，成交量确认
        strong_sell = (
            ~fast_above_zero &
            ~slow_above_zero &
            ~fast_above_slow &
            ~above_trendline &
            volume_confirmed &
            (cci_fast < -100)
        )

        # 卖出信号：快线下穿零线或趋势线
        sell = (
            (~fast_above_zero & fast_above_zero.shift(1)) |  # 零线下穿
            (~above_trendline & above_trendline.shift(1))   # 趋势线下穿
        ) & volume_confirmed

        signals[strong_buy] = 2
        signals[buy] = 1
        signals[strong_sell] = -2
        signals[sell] = -1

        return signals

    def identify_woodies_patterns(self, cci_fast: pd.Series, cci_slow: pd.Series,
                                 trendline: pd.Series) -> Dict[str, pd.Series]:
        """
        识别Woodies CCI特定模式

        Args:
            cci_fast: 快速CCI
            cci_slow: 慢速CCI
            trendline: 趋势线

        Returns:
            模式识别结果
        """
        patterns = {}

        # 1. 零线拒绝模式
        patterns['zero_line_rejection'] = self.identify_zero_line_rejection(cci_fast, cci_slow)

        # 2. 趋势线突破模式
        patterns['trendline_breakout'] = self.identify_trendline_breakout(cci_fast, trendline)

        # 3. 快慢线交叉模式
        patterns['crossover_pattern'] = self.identify_crossover_pattern(cci_fast, cci_slow)

        # 4. 极值反转模式
        patterns['extreme_reversal'] = self.identify_extreme_reversal(cci_fast)

        return patterns

    def identify_zero_line_rejection(self, cci_fast: pd.Series, cci_slow: pd.Series) -> pd.Series:
        """
        识别零线拒绝模式

        Args:
            cci_fast: 快速CCI
            cci_slow: 慢速CCI

        Returns:
            零线拒绝标记
        """
        rejection = pd.Series(0, index=cci_fast.index)

        # 识别接近零线后反弹
        for i in range(2, len(cci_fast)):
            # 上方拒绝
            if (cci_fast.iloc[i-2] > 0 and cci_fast.iloc[i-1] < 10 and
                cci_fast.iloc[i] > 20 and cci_slow.iloc[i] > 0):
                rejection.iloc[i] = 1

            # 下方拒绝
            elif (cci_fast.iloc[i-2] < 0 and cci_fast.iloc[i-1] > -10 and
                  cci_fast.iloc[i] < -20 and cci_slow.iloc[i] < 0):
                rejection.iloc[i] = -1

        return rejection

    def identify_trendline_breakout(self, cci_fast: pd.Series, trendline: pd.Series) -> pd.Series:
        """
        识别趋势线突破模式

        Args:
            cci_fast: 快速CCI
            trendline: 趋势线

        Returns:
            趋势线突破标记
        """
        breakout = pd.Series(0, index=cci_fast.index)

        # 识别趋势线突破
        above_trend = cci_fast > trendline
        below_trend = cci_fast < trendline

        breakout[above_trend & ~above_trend.shift(1)] = 1    # 向上突破
        breakout[below_trend & ~below_trend.shift(1)] = -1   # 向下突破

        return breakout

    def identify_crossover_pattern(self, cci_fast: pd.Series, cci_slow: pd.Series) -> pd.Series:
        """
        识别快慢线交叉模式

        Args:
            cci_fast: 快速CCI
            cci_slow: 慢速CCI

        Returns:
            交叉模式标记
        """
        crossover = pd.Series(0, index=cci_fast.index)

        fast_above_slow = cci_fast > cci_slow

        crossover[fast_above_slow & ~fast_above_slow.shift(1)] = 1    # 金叉
        crossover[~fast_above_slow & fast_above_slow.shift(1)] = -1   # 死叉

        return crossover

    def identify_extreme_reversal(self, cci_fast: pd.Series) -> pd.Series:
        """
        识别极值反转模式

        Args:
            cci_fast: 快速CCI

        Returns:
            极值反转标记
        """
        reversal = pd.Series(0, index=cci_fast.index)

        # 识别极值后反转
        reversal[(cci_fast > 200) & (cci_fast.shift(1) > 200)] = -1  # 超买反转
        reversal[(cci_fast < -200) & (cci_fast.shift(1) < -200)] = 1  # 超卖反转

        return reversal

    def get_comprehensive_analysis(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        获取综合分析结果

        Args:
            data: 包含OHLCV数据的字典

        Returns:
            综合分析结果
        """
        # 计算Woodies CCI
        woodies_data = self.calculate_woodies_cci_volume(
            data['high'], data['low'], data['close'], data['volume']
        )

        # 计算多时间框架确认
        mtf_confirmation = self.calculate_mtf_confirmation(data)

        # 计算市场情绪
        market_sentiment = self.calculate_market_sentiment(woodies_data)

        # 生成高级信号
        advanced_signals = self.generate_advanced_signals(woodies_data, mtf_confirmation)

        # 计算支撑阻力位
        support_resistance = self.calculate_support_resistance(data['close'], woodies_data)

        return {
            'cci_fast': woodies_data['cci_fast'],
            'cci_slow': woodies_data['cci_slow'],
            'volume_confirmation': woodies_data['volume_confirmation'],
            'trendline': woodies_data['trendline'],
            'divergence': woodies_data['divergence'],
            'signal_strength': woodies_data['signal_strength'],
            'signals': advanced_signals,
            'patterns': woodies_data['patterns'],
            'mtf_confirmation': mtf_confirmation,
            'market_sentiment': market_sentiment,
            'support_resistance': support_resistance
        }

    def calculate_mtf_confirmation(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        计算多时间框架确认

        Args:
            data: 市场数据

        Returns:
            多时间框架确认结果
        """
        close = data['close']

        # 计算不同周期的CCI
        cci_periods = [10, 20, 30, 50]
        mtf_cci = {}

        for period in cci_periods:
            tp = (data['high'] + data['low'] + close) / 3
            cci = self.calculate_cci(tp, data['high'], data['low'], close, period)
            mtf_cci[f'cci_{period}'] = cci

        # 计算多时间框架信号
        mtf_signals = pd.Series(0, index=close.index)

        # 多时间框架一致性
        bullish_count = sum(cci.iloc[-1] > 0 for cci in mtf_cci.values())
        bearish_count = sum(cci.iloc[-1] < 0 for cci in mtf_cci.values())

        if bullish_count >= 3:
            mtf_signals.iloc[-1] = 1
        elif bearish_count >= 3:
            mtf_signals.iloc[-1] = -1

        return {
            'mtf_cci': mtf_cci,
            'mtf_signals': mtf_signals,
            'consensus': 'bullish' if bullish_count >= 3 else 'bearish' if bearish_count >= 3 else 'neutral'
        }

    def calculate_market_sentiment(self, woodies_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        计算市场情绪

        Args:
            woodies_data: Woodies CCI数据

        Returns:
            市场情绪指标
        """
        cci_fast = woodies_data['cci_fast']
        cci_slow = woodies_data['cci_slow']

        # 计算情绪指标
        latest_cci_fast = cci_fast.iloc[-1]
        latest_cci_slow = cci_slow.iloc[-1]
        latest_volume_conf = woodies_data['volume_confirmation'].iloc[-1]

        # 市场情绪评分
        if latest_cci_fast > 100 and latest_cci_slow > 0:
            sentiment = 'strong_bullish'
            score = 0.8
        elif latest_cci_fast > 0 and latest_cci_slow > 0:
            sentiment = 'bullish'
            score = 0.6
        elif latest_cci_fast < -100 and latest_cci_slow < 0:
            sentiment = 'strong_bearish'
            score = 0.2
        elif latest_cci_fast < 0 and latest_cci_slow < 0:
            sentiment = 'bearish'
            score = 0.4
        else:
            sentiment = 'neutral'
            score = 0.5

        # 考虑成交量确认
        if latest_volume_conf > 60:
            score += 0.1
        elif latest_volume_conf < 40:
            score -= 0.1

        return {
            'sentiment': sentiment,
            'score': max(0, min(1, score)),
            'confidence': latest_volume_conf / 100
        }

    def generate_advanced_signals(self, woodies_data: Dict[str, pd.Series],
                                 mtf_confirmation: Dict[str, pd.Series]) -> pd.Series:
        """
        生成高级交易信号

        Args:
            woodies_data: Woodies CCI数据
            mtf_confirmation: 多时间框架确认

        Returns:
            高级信号序列
        """
        signals = pd.Series(0, index=woodies_data['cci_fast'].index)

        # 基础信号
        base_signals = woodies_data['signals']

        # 多时间框架确认
        mtf_confirmed = mtf_confirmation['mtf_signals'] == base_signals

        # 信号强度过滤
        strong_signal = woodies_data['signal_strength'] > 60

        # 背离确认
        divergence_confirmed = woodies_data['divergence'] != 0

        # 高级买入信号
        advanced_buy = (
            (base_signals > 0) &
            mtf_confirmed &
            strong_signal &
            (woodies_data['volume_confirmation'] > 50)
        )

        # 高级卖出信号
        advanced_sell = (
            (base_signals < 0) &
            mtf_confirmed &
            strong_signal &
            (woodies_data['volume_confirmation'] > 50)
        )

        # 背离信号
        divergence_buy = (woodies_data['divergence'] == 1) & (woodies_data['cci_fast'] < 0)
        divergence_sell = (woodies_data['divergence'] == -1) & (woodies_data['cci_fast'] > 0)

        signals[advanced_buy] = 2
        signals[advanced_sell] = -2
        signals[divergence_buy] = 1
        signals[divergence_sell] = -1

        return signals

    def calculate_support_resistance(self, prices: pd.Series,
                                   woodies_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        计算支撑阻力位

        Args:
            prices: 价格序列
            woodies_data: Woodies CCI数据

        Returns:
            支撑阻力位
        """
        # 基于CCI极值的支撑阻力
        cci_highs = self.find_peaks(woodies_data['cci_fast'], 20)
        cci_lows = self.find_troughs(woodies_data['cci_fast'], 20)

        # 找到对应的价格水平
        support_levels = pd.Series(0, index=prices.index)
        resistance_levels = pd.Series(0, index=prices.index)

        for i in range(len(prices)):
            if i >= 20:
                # 支撑位
                recent_lows = cci_lows.iloc[i-20:i]
                if len(recent_lows[recent_lows != 0]) > 0:
                    support_levels.iloc[i] = prices.iloc[recent_lows[recent_lows != 0].index[-1]]

                # 阻力位
                recent_highs = cci_highs.iloc[i-20:i]
                if len(recent_highs[recent_highs != 0]) > 0:
                    resistance_levels.iloc[i] = prices.iloc[recent_highs[recent_highs != 0].index[-1]]

        return {
            'support_levels': support_levels,
            'resistance_levels': resistance_levels
        }

    def get_trading_recommendations(self, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            analysis: 综合分析结果

        Returns:
            交易建议DataFrame
        """
        recommendations = pd.DataFrame(index=analysis['cci_fast'].index)

        # 信号描述
        recommendations['signal'] = analysis['signals']
        recommendations['signal_description'] = analysis['signals'].map({
            2: '强烈买入 - 多时间框架确认',
            1: '买入 - 背离反转',
            0: '持有 - 观望',
            -1: '卖出 - 背离反转',
            -2: '强烈卖出 - 多时间框架确认'
        })

        # 市场情绪
        recommendations['market_sentiment'] = analysis['market_sentiment']['sentiment']
        recommendations['sentiment_score'] = analysis['market_sentiment']['score']

        # 仓位建议
        recommendations['position_size'] = analysis['signals'].map({
            2: 0.7,   # 强买入：70%仓位
            1: 0.4,   # 买入：40%仓位
            0: 0.2,   # 持有：20%仓位
            -1: 0.1,  # 卖出：10%仓位
            -2: 0.0   # 强卖出：0仓位
        })

        # 止损止盈建议
        current_price = analysis['cci_fast'].index.to_series().map(lambda x: analysis['cci_fast'].loc[x])
        recommendations['stop_loss'] = current_price * 0.95
        recommendations['take_profit'] = current_price * 1.08

        # 信号强度
        recommendations['signal_strength'] = analysis['signal_strength']

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    n = len(dates)

    # 生成OHLCV数据
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    volume = np.random.randint(1000000, 5000000, n)

    data = {
        'high': pd.Series(high, index=dates),
        'low': pd.Series(low, index=dates),
        'close': pd.Series(close, index=dates),
        'volume': pd.Series(volume, index=dates)
    }

    # 创建Woodies CCI成交量确认指标
    woodies_cci = WoodiesCCIVolume()

    # 计算指标
    results = woodies_cci.calculate_woodies_cci_volume(
        data['high'], data['low'], data['close'], data['volume']
    )

    print("Woodies CCI with Volume Confirmation 结果:")
    print(f"快速CCI: {results['cci_fast'].iloc[-1]:.2f}")
    print(f"慢速CCI: {results['cci_slow'].iloc[-1]:.2f}")
    print(f"成交量确认: {results['volume_confirmation'].iloc[-1]:.2f}")
    print(f"趋势线: {results['trendline'].iloc[-1]:.2f}")
    print(f"信号强度: {results['signal_strength'].iloc[-1]:.2f}")
    print(f"当前信号: {results['signals'].iloc[-1]}")

    # 获取综合分析
    comprehensive = woodies_cci.get_comprehensive_analysis(data)
    print("\n综合分析结果:")
    print(f"市场情绪: {comprehensive['market_sentiment']['sentiment']}")
    print(f"情绪评分: {comprehensive['market_sentiment']['score']:.2f}")
    print(f"高级信号: {comprehensive['signals'].iloc[-1]}")
    print(f"多时间框架共识: {comprehensive['mtf_confirmation']['consensus']}")

    # 获取交易建议
    recommendations = woodies_cci.get_trading_recommendations(comprehensive)
    print("\n最新交易建议:")
    print(recommendations.iloc[-1])