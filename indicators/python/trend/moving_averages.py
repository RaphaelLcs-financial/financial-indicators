"""
移动平均线指标实现
Moving Averages Indicators Implementation

包含多种移动平均线的Python实现：
- SMA: 简单移动平均线
- EMA: 指数移动平均线
- WMA: 加权移动平均线
- HMA: 赫尔移动平均线
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class MovingAverages:
    """移动平均线指标集合"""

    def __init__(self):
        pass

    @staticmethod
    def sma(prices: Union[List[float], pd.Series], period: int = 20) -> pd.Series:
        """
        简单移动平均线 (Simple Moving Average)

        参数:
            prices: 价格序列
            period: 计算周期

        返回:
            SMA值序列
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(prices: Union[List[float], pd.Series], period: int = 20) -> pd.Series:
        """
        指数移动平均线 (Exponential Moving Average)

        参数:
            prices: 价格序列
            period: 计算周期

        返回:
            EMA值序列
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def wma(prices: Union[List[float], pd.Series], period: int = 20) -> pd.Series:
        """
        加权移动平均线 (Weighted Moving Average)

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
    def hma(prices: Union[List[float], pd.Series], period: int = 20) -> pd.Series:
        """
        赫尔移动平均线 (Hull Moving Average)
        减少滞后性的移动平均线

        参数:
            prices: 价格序列
            period: 计算周期

        返回:
            HMA值序列
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        # 计算WMA
        def _wma(series, p):
            weights = np.arange(1, p + 1)
            weights = weights / weights.sum()
            return series.rolling(window=p).apply(
                lambda x: np.dot(x, weights), raw=True
            )

        # HMA计算公式
        wma_half = _wma(prices, period // 2)
        wma_full = _wma(prices, period)
        raw_hma = 2 * wma_half - wma_full
        hma = _wma(raw_hma, int(np.sqrt(period)))

        return hma

    @staticmethod
    def sma_crossover(prices: Union[List[float], pd.Series],
                     fast_period: int = 10, slow_period: int = 20) -> Dict[str, pd.Series]:
        """
        双均线交叉策略

        参数:
            prices: 价格序列
            fast_period: 快线周期
            slow_period: 慢线周期

        返回:
            包含快线、慢线和信号的字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        fast_ma = MovingAverages.sma(prices, fast_period)
        slow_ma = MovingAverages.sma(prices, slow_period)

        # 生成信号
        signals = pd.Series(0, index=prices.index)
        signals[fast_ma > slow_ma] = 1   # 买入信号
        signals[fast_ma < slow_ma] = -1  # 卖出信号

        return {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'signals': signals
        }

    @staticmethod
    def ma_envelope(prices: Union[List[float], pd.Series],
                   period: int = 20,
                   envelope_pct: float = 0.02) -> Dict[str, pd.Series]:
        """
        移动平均线包络线

        参数:
            prices: 价格序列
            period: 计算周期
            envelope_pct: 包络线百分比

        返回:
            包含中轨、上轨、下轨的字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        middle_band = MovingAverages.sma(prices, period)
        upper_band = middle_band * (1 + envelope_pct)
        lower_band = middle_band * (1 - envelope_pct)

        return {
            'middle': middle_band,
            'upper': upper_band,
            'lower': lower_band
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100)) + 100

    # 创建移动平均线实例
    ma_calculator = MovingAverages()

    # 计算各种移动平均线
    sma_20 = ma_calculator.sma(prices, 20)
    ema_20 = ma_calculator.ema(prices, 20)
    wma_20 = ma_calculator.wma(prices, 20)
    hma_20 = ma_calculator.hma(prices, 20)

    # 双均线交叉策略
    crossover_signals = ma_calculator.sma_crossover(prices, 10, 20)

    # 包络线策略
    envelope = ma_calculator.ma_envelope(prices, 20, 0.02)

    print("移动平均线计算完成！")
    print(f"SMA(20) 最新值: {sma_20.iloc[-1]:.2f}")
    print(f"EMA(20) 最新值: {ema_20.iloc[-1]:.2f}")
    print(f"HMA(20) 最新值: {hma_20.iloc[-1]:.2f}")
    print(f"当前信号: {crossover_signals['signals'].iloc[-1]}")