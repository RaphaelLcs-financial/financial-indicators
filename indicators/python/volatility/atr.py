"""
ATR (平均真实范围) 指标实现
Average True Range (ATR) Implementation

ATR是一个波动性指标，用于衡量市场的波动性。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class ATR:
    """平均真实范围指标"""

    def __init__(self):
        pass

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series],
                  period: int = 14) -> pd.Series:
        """
        计算ATR指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认14

        返回:
            ATR值序列
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算真实范围 (True Range)
        tr1 = high - low  # 当日最高价与最低价的差
        tr2 = abs(high - close.shift(1))  # 当日最高价与前一日收盘价的差的绝对值
        tr3 = abs(low - close.shift(1))   # 当日最低价与前一日收盘价的差的绝对值

        # 真实范围是三个值中的最大值
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算ATR（真实范围的移动平均）
        atr = tr.rolling(window=period).mean()

        return atr

    @staticmethod
    def atr_position_close(high: Union[List[float], pd.Series],
                         low: Union[List[float], pd.Series],
                         close: Union[List[float], pd.Series],
                         atr_period: int = 14,
                         multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """
        ATR止损位计算

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            atr_period: ATR周期
            multiplier: ATR倍数

        返回:
            包含止损位的字典
        """
        if isinstance(close, list):
            close = pd.Series(close)

        atr = ATR.calculate(high, low, close, atr_period)

        # 计算多头止损位
        long_stop = close - (atr * multiplier)

        # 计算空头止损位
        short_stop = close + (atr * multiplier)

        return {
            'atr': atr,
            'long_stop': long_stop,
            'short_stop': short_stop
        }

    @staticmethod
    def atr_channels(high: Union[List[float], pd.Series],
                    low: Union[List[float], pd.Series],
                    close: Union[List[float], pd.Series],
                    atr_period: int = 14,
                    multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        ATR通道指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            atr_period: ATR周期
            multiplier: 通道倍数

        返回:
            ATR通道数据
        """
        if isinstance(close, list):
            close = pd.Series(close)

        atr = ATR.calculate(high, low, close, atr_period)

        # 计算通道上下轨
        upper_channel = close + (atr * multiplier)
        lower_channel = close - (atr * multiplier)

        # 计算通道中线
        middle_channel = (upper_channel + lower_channel) / 2

        return {
            'atr': atr,
            'upper_channel': upper_channel,
            'middle_channel': middle_channel,
            'lower_channel': lower_channel
        }

    @staticmethod
    def volatility_breakout(high: Union[List[float], pd.Series],
                          low: Union[List[float], pd.Series],
                          close: Union[List[float], pd.Series],
                          atr_period: int = 14,
                          breakout_multiplier: float = 1.5) -> pd.Series:
        """
        ATR波动性突破策略

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            atr_period: ATR周期
            breakout_multiplier: 突破倍数

        返回:
            突破信号序列
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        atr = ATR.calculate(high, low, close, atr_period)

        # 计算前一日收盘价
        prev_close = close.shift(1)

        # 计算突破价格
        breakout_upper = prev_close + (atr * breakout_multiplier)
        breakout_lower = prev_close - (atr * breakout_multiplier)

        breakout_signals = pd.Series(0, index=close.index)

        # 向上突破
        breakout_signals[(high > breakout_upper) & (high.shift(1) <= breakout_upper.shift(1))] = 1

        # 向下突破
        breakout_signals[(low < breakout_lower) & (low.shift(1) >= breakout_lower.shift(1))] = -1

        return breakout_signals

    @staticmethod
    def atr_trend_filter(close: Union[List[float], pd.Series],
                         high: Union[List[float], pd.Series],
                         low: Union[List[float], pd.Series],
                         atr_period: int = 14) -> pd.Series:
        """
        ATR趋势过滤器

        参数:
            close: 收盘价序列
            high: 最高价序列
            low: 最低价序列
            atr_period: ATR周期

        返回:
            趋势信号序列 (1=上升趋势, -1=下降趋势, 0=盘整)
        """
        if isinstance(close, list):
            close = pd.Series(close)

        atr = ATR.calculate(high, low, close, atr_period)

        # 计算价格移动平均
        price_ma = close.rolling(window=atr_period).mean()

        # 计算ATR的移动平均
        atr_ma = atr.rolling(window=atr_period).mean()

        trend_signals = pd.Series(0, index=close.index)

        # 上升趋势判断
        uptrend_condition = (close > price_ma) & (atr < atr_ma)
        trend_signals[uptrend_condition] = 1

        # 下降趋势判断
        downtrend_condition = (close < price_ma) & (atr > atr_ma)
        trend_signals[downtrend_condition] = -1

        return trend_signals


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算ATR
    atr_calculator = ATR()
    atr_values = atr_calculator.calculate(high, low, close)

    # ATR止损位
    stop_loss_data = atr_calculator.atr_position_close(high, low, close)

    # ATR通道
    channel_data = atr_calculator.atr_channels(high, low, close)

    # 波动性突破
    breakout_signals = atr_calculator.volatility_breakout(high, low, close)

    # 趋势过滤器
    trend_signals = atr_calculator.atr_trend_filter(close, high, low)

    print("ATR指标计算完成！")
    print(f"ATR值: {atr_values.iloc[-1]:.4f}")
    print(f"多头止损位: {stop_loss_data['long_stop'].iloc[-1]:.2f}")
    print(f"空头止损位: {stop_loss_data['short_stop'].iloc[-1]:.2f}")
    print(f"通道上轨: {channel_data['upper_channel'].iloc[-1]:.2f}")
    print(f"通道下轨: {channel_data['lower_channel'].iloc[-1]:.2f}")
    print(f"突破信号: {breakout_signals.iloc[-1]}")
    print(f"趋势信号: {trend_signals.iloc[-1]}")