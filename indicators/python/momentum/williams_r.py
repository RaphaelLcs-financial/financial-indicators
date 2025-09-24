"""
威廉指标 (Williams %R) 指标实现
Williams Percent Range Implementation

威廉指标是一个动量指标，用于衡量价格是否处于超买或超卖状态。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class WilliamsR:
    """威廉指标"""

    def __init__(self):
        pass

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series],
                  period: int = 14) -> pd.Series:
        """
        计算威廉指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认14

        返回:
            威廉指标值序列
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算周期内的最高价和最低价
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()

        # 计算威廉指标
        williams_r = -100 * (highest_high - close) / (highest_high - lowest_low)

        return williams_r

    @staticmethod
    def generate_signals(williams_r: Union[List[float], pd.Series],
                        oversold: float = -80,
                        overbought: float = -20) -> pd.Series:
        """
        生成威廉指标交易信号

        参数:
            williams_r: 威廉指标序列
            oversold: 超卖阈值，默认-80
            overbought: 超买阈值，默认-20

        返回:
            信号序列 (1=买入, -1=卖出, 0=持有)
        """
        if isinstance(williams_r, list):
            williams_r = pd.Series(williams_r)

        signals = pd.Series(0, index=williams_r.index)

        # 超卖买入信号
        signals[(williams_r < oversold) & (williams_r.shift(1) >= oversold)] = 1

        # 超买卖出信号
        signals[(williams_r > overbought) & (williams_r.shift(1) <= overbought)] = -1

        return signals

    @staticmethod
    def divergence_analysis(prices: Union[List[float], pd.Series],
                           williams_r: Union[List[float], pd.Series],
                           period: int = 14) -> Dict[str, pd.Series]:
        """
        威廉指标背离分析

        参数:
            prices: 价格序列
            williams_r: 威廉指标序列
            period: 分析周期

        返回:
            背离信号字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)
        if isinstance(williams_r, list):
            williams_r = pd.Series(williams_r)

        # 计算价格和指标的趋势
        price_trend = prices.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])
        wr_trend = williams_r.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])

        # 检测看涨背离（价格下跌，威廉指标上涨）
        bullish_divergence = pd.Series(0, index=prices.index)
        bullish_divergence[(price_trend < 0) & (wr_trend > 0)] = 1

        # 检测看跌背离（价格上涨，威廉指标下跌）
        bearish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence[(price_trend > 0) & (wr_trend < 0)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }

    @staticmethod
    def failure_swing(williams_r: Union[List[float], pd.Series],
                      overbought: float = -20,
                      oversold: float = -80) -> Dict[str, pd.Series]:
        """
        威廉指标失败摆动信号

        参数:
            williams_r: 威廉指标序列
            overbought: 超买阈值
            oversold: 超卖阈值

        返回:
            失败摆动信号字典
        """
        if isinstance(williams_r, list):
            williams_r = pd.Series(williams_r)

        # 看涨失败摆动：指标超卖后未能达到-20
        bullish_failure = pd.Series(0, index=williams_r.index)
        bullish_failure[(williams_r < oversold) & (williams_r.shift(1) >= oversold)] = 1
        bullish_failure[(williams_r > overbought) & (williams_r.shift(1) <= overbought)] = 0  # 取消信号

        # 看跌失败摆动：指标超买后未能达到-80
        bearish_failure = pd.Series(0, index=williams_r.index)
        bearish_failure[(williams_r > overbought) & (williams_r.shift(1) <= overbought)] = -1
        bearish_failure[(williams_r < oversold) & (williams_r.shift(1) >= oversold)] = 0  # 取消信号

        return {
            'bullish_failure': bullish_failure,
            'bearish_failure': bearish_failure
        }

    @staticmethod
    def williams_r_strategy(high: Union[List[float], pd.Series],
                           low: Union[List[float], pd.Series],
                           close: Union[List[float], pd.Series],
                           period: int = 14,
                           oversold: float = -80,
                           overbought: float = -20) -> Dict[str, pd.Series]:
        """
        完整的威廉指标交易策略

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期
            oversold: 超卖阈值
            overbought: 超买阈值

        返回:
            策略结果字典
        """
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算威廉指标
        wr_values = WilliamsR.calculate(high, low, close, period)

        # 生成基本信号
        basic_signals = WilliamsR.generate_signals(wr_values, oversold, overbought)

        # 背离分析
        divergence_signals = WilliamsR.divergence_analysis(close, wr_values)

        # 失败摆动分析
        failure_swing_signals = WilliamsR.failure_swing(wr_values, overbought, oversold)

        # 综合信号
        combined_signals = pd.Series(0, index=close.index)

        # 多种信号确认买入
        buy_conditions = (
            (basic_signals == 1) |
            (divergence_signals['bullish_divergence'] == 1) |
            (failure_swing_signals['bullish_failure'] == 1)
        )
        combined_signals[buy_conditions] = 1

        # 多种信号确认卖出
        sell_conditions = (
            (basic_signals == -1) |
            (divergence_signals['bearish_divergence'] == -1) |
            (failure_swing_signals['bearish_failure'] == -1)
        )
        combined_signals[sell_conditions] = -1

        return {
            'williams_r': wr_values,
            'basic_signals': basic_signals,
            'divergence_signals': divergence_signals,
            'failure_swing_signals': failure_swing_signals,
            'combined_signals': combined_signals
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算威廉指标
    wr_calculator = WilliamsR()
    wr_values = wr_calculator.calculate(high, low, close)

    # 生成交易信号
    signals = wr_calculator.generate_signals(wr_values)

    # 背离分析
    divergence_signals = wr_calculator.divergence_analysis(close, wr_values)

    # 失败摆动
    failure_swing_signals = wr_calculator.failure_swing(wr_values)

    # 完整策略
    strategy_results = wr_calculator.williams_r_strategy(high, low, close)

    print("威廉指标计算完成！")
    print(f"威廉指标值: {wr_values.iloc[-1]:.2f}")
    print(f"基本信号: {signals.iloc[-1]}")
    print(f"看涨背离: {divergence_signals['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence_signals['bearish_divergence'].iloc[-1]}")
    print(f"看涨失败摆动: {failure_swing_signals['bullish_failure'].iloc[-1]}")
    print(f"看跌失败摆动: {failure_swing_signals['bearish_failure'].iloc[-1]}")
    print(f"综合信号: {strategy_results['combined_signals'].iloc[-1]}")