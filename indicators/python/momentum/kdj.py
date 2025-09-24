"""
KDJ (随机指标) 指标实现
Stochastic Oscillator (KDJ) Implementation

KDJ是一个动量指标，用于比较特定时间段内的收盘价与价格范围的关系。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class KDJ:
    """KDJ随机指标"""

    def __init__(self):
        pass

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series],
                  k_period: int = 9,
                  d_period: int = 3,
                  j_period: int = 3) -> Dict[str, pd.Series]:
        """
        计算KDJ指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            k_period: K值周期，默认9
            d_period: D值周期，默认3
            j_period: J值周期，默认3

        返回:
            包含K、D、J值的字典
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算RSV（未成熟随机值）
        low_min = low.rolling(window=k_period).min()
        high_max = high.rolling(window=k_period).max()
        rsv = (close - low_min) / (high_max - low_min) * 100

        # 计算K值（RSV的移动平均）
        k_values = pd.Series(50.0, index=close.index)  # 初始化K值为50
        for i in range(k_period, len(close)):
            k_values.iloc[i] = (2 * k_values.iloc[i-1] + rsv.iloc[i]) / 3

        # 计算D值（K值的指数移动平均）
        d_values = pd.Series(50.0, index=close.index)  # 初始化D值为50
        for i in range(k_period, len(close)):
            d_values.iloc[i] = (2 * d_values.iloc[i-1] + k_values.iloc[i]) / 3

        # 计算J值
        j_values = 3 * k_values - 2 * d_values

        return {
            'k': k_values,
            'd': d_values,
            'j': j_values,
            'rsv': rsv
        }

    @staticmethod
    def generate_signals(kdj_data: Dict[str, pd.Series],
                        oversold: float = 20,
                        overbought: float = 80) -> pd.Series:
        """
        生成KDJ交易信号

        参数:
            kdj_data: KDJ数据
            oversold: 超卖阈值，默认20
            overbought: 超买阈值，默认80

        返回:
            信号序列 (1=买入, -1=卖出, 0=持有)
        """
        k = kdj_data['k']
        d = kdj_data['d']
        j = kdj_data['j']

        signals = pd.Series(0, index=k.index)

        # 金叉买入信号（K线上穿D线）
        signals[(k > d) & (k.shift(1) <= d.shift(1)) & (k < oversold)] = 1

        # 死叉卖出信号（K线下穿D线）
        signals[(k < d) & (k.shift(1) >= d.shift(1)) & (k > overbought)] = -1

        # J值超卖买入
        signals[(j < 0) & (j.shift(1) >= 0)] = 1

        # J值超买卖出
        signals[(j > 100) & (j.shift(1) <= 100)] = -1

        return signals

    @staticmethod
    def divergence_analysis(prices: Union[List[float], pd.Series],
                           kdj_data: Dict[str, pd.Series],
                           period: int = 14) -> Dict[str, pd.Series]:
        """
        KDJ背离分析

        参数:
            prices: 价格序列
            kdj_data: KDJ数据
            period: 分析周期

        返回:
            背离信号字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        k = kdj_data['k']
        j = kdj_data['j']

        # 计算价格和指标的趋势
        price_trend = prices.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])
        k_trend = k.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])
        j_trend = j.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])

        # 检测看涨背离
        bullish_divergence = pd.Series(0, index=prices.index)
        bullish_divergence[(price_trend < 0) & (k_trend > 0)] = 1
        bullish_divergence[(price_trend < 0) & (j_trend > 0)] = 1

        # 检测看跌背离
        bearish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence[(price_trend > 0) & (k_trend < 0)] = -1
        bearish_divergence[(price_trend > 0) & (j_trend < 0)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }

    @staticmethod
    def multiple_timeframe_signals(high_df: pd.DataFrame,
                                 low_df: pd.DataFrame,
                                 close_df: pd.DataFrame,
                                 timeframes: List[int] = [9, 14, 21]) -> pd.Series:
        """
        多时间框架KDJ信号

        参数:
            high_df: 最高价DataFrame
            low_df: 最低价DataFrame
            close_df: 收盘价DataFrame
            timeframes: 时间周期列表

        返回:
            综合信号序列
        """
        all_signals = []

        for timeframe in timeframes:
            kdj_data = KDJ.calculate(high_df[timeframe], low_df[timeframe], close_df[timeframe], timeframe)
            signals = KDJ.generate_signals(kdj_data)
            all_signals.append(signals)

        # 综合信号（多数时间框架一致）
        combined_signals = pd.Series(0, index=all_signals[0].index)

        for i in range(len(all_signals[0])):
            signal_sum = sum(sig.iloc[i] for sig in all_signals)
            if signal_sum >= 2:  # 至少2个时间框架买入信号
                combined_signals.iloc[i] = 1
            elif signal_sum <= -2:  # 至少2个时间框架卖出信号
                combined_signals.iloc[i] = -1

        return combined_signals


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算KDJ
    kdj_calculator = KDJ()
    kdj_data = kdj_calculator.calculate(high, low, close)

    # 生成交易信号
    signals = kdj_calculator.generate_signals(kdj_data)

    # 背离分析
    divergence_signals = kdj_calculator.divergence_analysis(close, kdj_data)

    print("KDJ指标计算完成！")
    print(f"K值: {kdj_data['k'].iloc[-1]:.2f}")
    print(f"D值: {kdj_data['d'].iloc[-1]:.2f}")
    print(f"J值: {kdj_data['j'].iloc[-1]:.2f}")
    print(f"当前信号: {signals.iloc[-1]}")
    print(f"看涨背离: {divergence_signals['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence_signals['bearish_divergence'].iloc[-1]}")