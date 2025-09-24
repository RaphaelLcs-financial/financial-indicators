"""
Fisher Transform (费雪变换) - Python实现
将价格转换为高斯正态分布，产生更清晰的买卖信号。

由John Ehlers开发，近年来在算法交易中变得越来越流行。
能够将非正态分布的价格数据转换为接近正态分布。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class FisherTransform:
    """费雪变换指标"""

    def __init__(self):
        self.name = "Fisher Transform"
        self.category = "momentum"

    @staticmethod
    def normalize_price(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 10) -> pd.Series:
        """
        标准化价格

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 周期

        返回:
            标准化价格序列
        """
        # 计算最高价的最高值和最低价的最低值
        max_high = high.rolling(window=period).max()
        min_low = low.rolling(window=period).min()

        # 标准化价格到-1到1之间
        normalized = pd.Series(0.0, index=close.index)

        for i in range(len(close)):
            if max_high.iloc[i] != min_low.iloc[i]:
                normalized.iloc[i] = 0.5 * 2 * ((close.iloc[i] - min_low.iloc[i]) /
                                                (max_high.iloc[i] - min_low.iloc[i]) - 0.5) + 0.5 * normalized.iloc[i-1]
                # 限制在-0.999到0.999之间，避免log(0)错误
                normalized.iloc[i] = max(-0.999, min(0.999, normalized.iloc[i]))

        return normalized

    @staticmethod
    def fisher_transform(normalized: pd.Series) -> pd.Series:
        """
        费雪变换

        参数:
            normalized: 标准化价格序列

        返回:
            费雪变换值序列
        """
        fisher = pd.Series(0.0, index=normalized.index)

        for i in range(len(normalized)):
            if pd.notna(normalized.iloc[i]):
                fisher.iloc[i] = 0.5 * np.log((1 + normalized.iloc[i]) / (1 - normalized.iloc[i]))

        return fisher

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series],
                  period: int = 10,
                  smoothing: int = 4) -> Dict[str, pd.Series]:
        """
        计算Fisher Transform指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 标准化周期，默认10
            smoothing: 平滑周期，默认4

        返回:
            包含Fisher Transform和信号线的字典
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        # 标准化价格
        normalized = FisherTransform.normalize_price(high, low, close, period)

        # 计算费雪变换
        fisher = FisherTransform.fisher_transform(normalized)

        # 计算信号线（Fisher的移动平均）
        signal_line = fisher.rolling(window=smoothing).mean()

        return {
            'fisher': fisher,
            'signal_line': signal_line,
            'normalized': normalized
        }

    @staticmethod
    def generate_signals(fisher_data: Dict[str, pd.Series],
                        overbought: float = 1.5,
                        oversold: float = -1.5) -> pd.Series:
        """
        生成Fisher Transform交易信号

        参数:
            fisher_data: Fisher Transform数据
            overbought: 超买阈值，默认1.5
            oversold: 超卖阈值，默认-1.5

        返回:
            交易信号序列
        """
        fisher = fisher_data['fisher']
        signal_line = fisher_data['signal_line']

        signals = pd.Series(0, index=fisher.index)

        for i in range(1, len(fisher)):
            # Fisher线上穿信号线 - 买入信号
            if fisher.iloc[i] > signal_line.iloc[i] and fisher.iloc[i-1] <= signal_line.iloc[i-1]:
                if fisher.iloc[i] < oversold:  # 超卖区域
                    signals.iloc[i] = 2  # 强买入
                else:
                    signals.iloc[i] = 1  # 普通买入

            # Fisher线下穿信号线 - 卖出信号
            elif fisher.iloc[i] < signal_line.iloc[i] and fisher.iloc[i-1] >= signal_line.iloc[i-1]:
                if fisher.iloc[i] > overbought:  # 超买区域
                    signals.iloc[i] = -2  # 强卖出
                else:
                    signals.iloc[i] = -1  # 普通卖出

            # 超买超卖反转信号
            elif fisher.iloc[i] > overbought and fisher.iloc[i-1] <= overbought:
                signals.iloc[i] = -1  # 超买反转卖出

            elif fisher.iloc[i] < oversold and fisher.iloc[i-1] >= oversold:
                signals.iloc[i] = 1  # 超卖反转买入

        return signals

    @staticmethod
    def divergence_detection(prices: pd.Series,
                            fisher_data: Dict[str, pd.Series],
                            period: int = 14) -> Dict[str, pd.Series]:
        """
        Fisher Transform背离检测

        参数:
            prices: 价格序列
            fisher_data: Fisher Transform数据
            period: 检测周期

        返回:
            背离信号
        """
        fisher = fisher_data['fisher']

        # 检测价格和Fisher的高点低点
        price_highs = prices.rolling(window=period).max()
        price_lows = prices.rolling(window=period).min()
        fisher_highs = fisher.rolling(window=period).max()
        fisher_lows = fisher.rolling(window=period).min()

        bullish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence = pd.Series(0, index=prices.index)

        # 看涨背离：价格创新低，Fisher没有创新低
        bullish_divergence[(prices == price_lows) & (fisher > fisher_lows) & (fisher < -1)] = 1

        # 看跌背离：价格创新高，Fisher没有创新高
        bearish_divergence[(prices == price_highs) & (fisher < fisher_highs) & (fisher > 1)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }

    @staticmethod
    def cycle_analysis(fisher_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Fisher Transform周期分析

        参数:
            fisher_data: Fisher Transform数据

        返回:
            周期分析结果
        """
        fisher = fisher_data['fisher']

        # 检测周期性转折点
        peaks = pd.Series(0, index=fisher.index)
        troughs = pd.Series(0, index=fisher.index)

        for i in range(2, len(fisher)-2):
            # 检测峰值
            if (fisher.iloc[i] > fisher.iloc[i-1] and
                fisher.iloc[i] > fisher.iloc[i+1] and
                fisher.iloc[i] > fisher.iloc[i-2] and
                fisher.iloc[i] > fisher.iloc[i+2]):
                peaks.iloc[i] = 1

            # 检测谷底
            elif (fisher.iloc[i] < fisher.iloc[i-1] and
                  fisher.iloc[i] < fisher.iloc[i+1] and
                  fisher.iloc[i] < fisher.iloc[i-2] and
                  fisher.iloc[i] < fisher.iloc[i+2]):
                troughs.iloc[i] = 1

        return {
            'peaks': peaks,
            'troughs': troughs
        }

    @staticmethod
    def adaptive_fisher(high: pd.Series, low: pd.Series, close: pd.Series,
                       volatility_period: int = 20) -> Dict[str, pd.Series]:
        """
        自适应Fisher Transform

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volatility_period: 波动率周期

        返回:
            自适应Fisher Transform结果
        """
        # 计算波动率
        true_range = high - low
        volatility = true_range.rolling(window=volatility_period).std()

        # 根据波动率调整周期
        adaptive_period = pd.Series(10, index=close.index)
        high_vol_threshold = volatility.quantile(0.75)
        low_vol_threshold = volatility.quantile(0.25)

        adaptive_period[volatility > high_vol_threshold] = 8   # 高波动率，缩短周期
        adaptive_period[volatility < low_vol_threshold] = 12  # 低波动率，延长周期

        # 计算自适应Fisher Transform
        fisher_results = []
        for i in range(len(close)):
            if i >= adaptive_period.iloc[i]:
                # 使用滚动窗口计算
                window_high = high.iloc[i-int(adaptive_period.iloc[i])+1:i+1]
                window_low = low.iloc[i-int(adaptive_period.iloc[i])+1:i+1]
                window_close = close.iloc[i-int(adaptive_period.iloc[i])+1:i+1]

                window_data = FisherTransform.calculate(window_high, window_low, window_close,
                                                      int(adaptive_period.iloc[i]))
                fisher_results.append(window_data['fisher'].iloc[-1])
            else:
                fisher_results.append(np.nan)

        fisher_series = pd.Series(fisher_results, index=close.index)
        signal_line = fisher_series.rolling(window=4).mean()

        return {
            'adaptive_fisher': fisher_series,
            'signal_line': signal_line,
            'adaptive_period': adaptive_period
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算Fisher Transform
    fisher_calculator = FisherTransform()
    fisher_data = fisher_calculator.calculate(high, low, close)

    # 生成信号
    signals = fisher_calculator.generate_signals(fisher_data)

    # 背离检测
    divergence = fisher_calculator.divergence_detection(close, fisher_data)

    # 周期分析
    cycle_analysis = fisher_calculator.cycle_analysis(fisher_data)

    # 自适应Fisher Transform
    adaptive_fisher = fisher_calculator.adaptive_fisher(high, low, close)

    print("Fisher Transform (费雪变换) 计算完成！")
    print(f"Fisher值: {fisher_data['fisher'].iloc[-1]:.4f}")
    print(f"信号线: {fisher_data['signal_line'].iloc[-1]:.4f}")
    print(f"信号: {signals.iloc[-1]}")
    print(f"看涨背离: {divergence['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence['bearish_divergence'].iloc[-1]}")
    print(f"自适应Fisher: {adaptive_fisher['adaptive_fisher'].iloc[-1]:.4f}")
    print(f"自适应周期: {adaptive_fisher['adaptive_period'].iloc[-1]:.0f}")