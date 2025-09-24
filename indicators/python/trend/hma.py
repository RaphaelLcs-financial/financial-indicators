"""
HMA (Hull Moving Average) - 赫尔移动平均线
一个创新的移动平均线，显著减少滞后性，同时保持平滑度。

由Alan Hull在2005年开发，近年来变得越来越流行。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class HMA:
    """赫尔移动平均线"""

    def __init__(self):
        self.name = "Hull Moving Average"
        self.category = "trend"

    @staticmethod
    def wma(prices: Union[List[float], pd.Series], period: int) -> pd.Series:
        """
        计算加权移动平均线

        参数:
            prices: 价格序列
            period: 计算周期

        返回:
            WMA值序列
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        weights = np.arange(1, period + 1)
        weights = weights / weights.sum()

        return prices.rolling(window=period).apply(
            lambda x: np.dot(x, weights), raw=True
        )

    @staticmethod
    def calculate(prices: Union[List[float], pd.Series], period: int = 16) -> pd.Series:
        """
        计算HMA

        参数:
            prices: 价格序列
            period: 计算周期，默认16

        返回:
            HMA值序列
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        # HMA计算公式
        # HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))

        # 计算两个不同周期的WMA
        wma_half = HMA.wma(prices, half_period)
        wma_full = HMA.wma(prices, period)

        # 计算2*WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full

        # 对结果进行WMA平滑
        hma = HMA.wma(raw_hma, sqrt_period)

        return hma

    @staticmethod
    def hma_slope(hma: Union[List[float], pd.Series], period: int = 3) -> pd.Series:
        """
        计算HMA斜率

        参数:
            hma: HMA序列
            period: 斜率计算周期

        返回:
            斜率序列
        """
        if isinstance(hma, list):
            hma = pd.Series(hma)

        return hma.diff(period)

    @staticmethod
    def hma_crossover(fast_hma: Union[List[float], pd.Series],
                      slow_hma: Union[List[float], pd.Series]) -> pd.Series:
        """
        HMA交叉策略

        参数:
            fast_hma: 快速HMA
            slow_hma: 慢速HMA

        返回:
            交叉信号序列
        """
        if isinstance(fast_hma, list):
            fast_hma = pd.Series(fast_hma)
        if isinstance(slow_hma, list):
            slow_hma = pd.Series(slow_hma)

        signals = pd.Series(0, index=fast_hma.index)

        # 金叉买入信号
        signals[(fast_hma > slow_hma) & (fast_hma.shift(1) <= slow_hma.shift(1))] = 1

        # 死叉卖出信号
        signals[(fast_hma < slow_hma) & (fast_hma.shift(1) >= slow_hma.shift(1))] = -1

        return signals

    @staticmethod
    def hma_color_change(hma: Union[List[float], pd.Series],
                        prices: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        HMA颜色变化策略（基于价格与HMA的关系）

        参数:
            hma: HMA序列
            prices: 价格序列

        返回:
            包含趋势和信号的字典
        """
        if isinstance(hma, list):
            hma = pd.Series(hma)
        if isinstance(prices, list):
            prices = pd.Series(prices)

        # 确定趋势方向
        trend = pd.Series(0, index=prices.index)
        trend[prices > hma] = 1    # 上升趋势
        trend[prices < hma] = -1   # 下降趋势

        # 生成信号
        signals = pd.Series(0, index=prices.index)

        # 价格上穿HMA - 买入信号
        signals[(prices > hma) & (prices.shift(1) <= hma.shift(1))] = 1

        # 价格下穿HMA - 卖出信号
        signals[(prices < hma) & (prices.shift(1) >= hma.shift(1))] = -1

        return {
            'trend': trend,
            'signals': signals
        }

    @staticmethod
    def hma_bands(prices: Union[List[float], pd.Series],
                  period: int = 16,
                  multiplier: float = 1.5) -> Dict[str, pd.Series]:
        """
        HMA带状指标

        参数:
            prices: 价格序列
            period: HMA周期
            multiplier: 带宽倍数

        返回:
            包含HMA带状数据的字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        # 计算HMA
        hma_line = HMA.calculate(prices, period)

        # 计算价格与HMA的偏差
        deviation = prices - hma_line

        # 计算偏差的标准差
        deviation_std = deviation.rolling(window=period).std()

        # 计算上下带
        upper_band = hma_line + (deviation_std * multiplier)
        lower_band = hma_line - (deviation_std * multiplier)

        return {
            'hma': hma_line,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'mid_band': (upper_band + lower_band) / 2
        }

    @staticmethod
    def adaptive_hma(prices: Union[List[float], pd.Series],
                     base_period: int = 16,
                     min_period: int = 6,
                     max_period: int = 50) -> Dict[str, pd.Series]:
        """
        自适应HMA - 根据市场波动性调整周期

        参数:
            prices: 价格序列
            base_period: 基础周期
            min_period: 最小周期
            max_period: 最大周期

        返回:
            自适应HMA数据
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        # 计算波动率
        returns = prices.pct_change()
        volatility = returns.rolling(window=20).std()

        # 将波动率标准化到0-1范围
        volatility_min = volatility.rolling(window=50).min()
        volatility_max = volatility.rolling(window=50).max()
        normalized_volatility = (volatility - volatility_min) / (volatility_max - volatility_min)

        # 根据波动率调整周期
        adaptive_period = base_period + (normalized_volatility * (max_period - min_period))

        # 对每个点计算HMA
        adaptive_hma = pd.Series(0.0, index=prices.index)

        for i in range(len(prices)):
            if i < max_period:
                adaptive_hma.iloc[i] = np.nan
            else:
                period = int(adaptive_period.iloc[i])
                period = max(min_period, min(max_period, period))
                adaptive_hma.iloc[i] = HMA.calculate(prices.iloc[:i+1], period).iloc[-1]

        return {
            'adaptive_hma': adaptive_hma,
            'adaptive_period': adaptive_period,
            'volatility': volatility
        }

    @staticmethod
    def hma_divergence(prices: Union[List[float], pd.Series],
                       hma: Union[List[float], pd.Series],
                       period: int = 20) -> Dict[str, pd.Series]:
        """
        HMA背离分析

        参数:
            prices: 价格序列
            hma: HMA序列
            period: 分析周期

        返回:
            背离信号字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)
        if isinstance(hma, list):
            hma = pd.Series(hma)

        # 计算价格和HMA的高点低点
        price_highs = prices.rolling(window=period).max()
        price_lows = prices.rolling(window=period).min()
        hma_highs = hma.rolling(window=period).max()
        hma_lows = hma.rolling(window=period).min()

        # 检测看涨背离（价格创新低，HMA没有创新低）
        bullish_divergence = pd.Series(0, index=prices.index)
        bullish_divergence[(prices == price_lows) & (hma > hma_lows)] = 1

        # 检测看跌背离（价格创新高，HMA没有创新高）
        bearish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence[(prices == price_highs) & (hma < hma_highs)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    prices = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算HMA
    hma_calculator = HMA()
    hma_16 = hma_calculator.calculate(prices, 16)
    hma_32 = hma_calculator.calculate(prices, 32)

    # HMA交叉策略
    crossover_signals = hma_calculator.hma_crossover(hma_16, hma_32)

    # HMA颜色变化策略
    color_change = hma_calculator.hma_color_change(hma_16, prices)

    # HMA带状指标
    hma_bands = hma_calculator.hma_bands(prices)

    # 自适应HMA
    adaptive_hma = hma_calculator.adaptive_hma(prices)

    # 背离分析
    divergence = hma_calculator.hma_divergence(prices, hma_16)

    print("HMA (赫尔移动平均线) 指标计算完成！")
    print(f"HMA(16): {hma_16.iloc[-1]:.2f}")
    print(f"HMA(32): {hma_32.iloc[-1]:.2f}")
    print(f"交叉信号: {crossover_signals.iloc[-1]}")
    print(f"趋势方向: {color_change['trend'].iloc[-1]}")
    print(f"颜色变化信号: {color_change['signals'].iloc[-1]}")
    print(f"自适应HMA: {adaptive_hma['adaptive_hma'].iloc[-1]:.2f}")
    print(f"看涨背离: {divergence['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence['bearish_divergence'].iloc[-1]}")