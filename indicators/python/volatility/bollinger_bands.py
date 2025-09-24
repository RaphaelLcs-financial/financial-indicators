"""
布林带 (Bollinger Bands) 指标实现
Bollinger Bands Implementation

布林带是一个波动性指标，由三条线组成：中轨（简单移动平均线）、上轨和下轨。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class BollingerBands:
    """布林带指标"""

    def __init__(self):
        pass

    @staticmethod
    def calculate(prices: Union[List[float], pd.Series],
                  period: int = 20,
                  std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        计算布林带

        参数:
            prices: 价格序列
            period: 移动平均周期，默认20
            std_dev: 标准差倍数，默认2.0

        返回:
            包含上轨、中轨、下轨的字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        # 计算中轨（简单移动平均）
        middle_band = prices.rolling(window=period).mean()

        # 计算标准差
        rolling_std = prices.rolling(window=period).std()

        # 计算上轨和下轨
        upper_band = middle_band + (rolling_std * std_dev)
        lower_band = middle_band - (rolling_std * std_dev)

        # 计算带宽
        bandwidth = (upper_band - lower_band) / middle_band * 100

        # 计算%b指标
        percent_b = (prices - lower_band) / (upper_band - lower_band) * 100

        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band,
            'bandwidth': bandwidth,
            'percent_b': percent_b
        }

    @staticmethod
    def generate_signals(bb_data: Dict[str, pd.Series],
                         prices: Union[List[float], pd.Series]) -> pd.Series:
        """
        生成布林带交易信号

        参数:
            bb_data: 布林带数据
            prices: 价格序列

        返回:
            信号序列 (1=买入, -1=卖出, 0=持有)
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        upper = bb_data['upper']
        lower = bb_data['lower']
        middle = bb_data['middle']
        percent_b = bb_data['percent_b']

        signals = pd.Series(0, index=prices.index)

        # 突破策略
        # 价格突破下轨买入
        signals[(prices < lower) & (prices.shift(1) >= lower.shift(1))] = 1

        # 价格突破上轨卖出
        signals[(prices > upper) & (prices.shift(1) <= upper.shift(1))] = -1

        # 回归策略
        # 从下轨回归到中轨买入
        signals[(prices > lower) & (prices.shift(1) <= lower.shift(1)) & (prices < middle)] = 1

        # 从上轨回归到中轨卖出
        signals[(prices < upper) & (prices.shift(1) >= upper.shift(1)) & (prices > middle)] = -1

        return signals

    @staticmethod
    def squeeze_detection(bb_data: Dict[str, pd.Series],
                         squeeze_threshold: float = 0.1) -> pd.Series:
        """
        检测布林带收缩（挤压）

        参数:
            bb_data: 布林带数据
            squeeze_threshold: 挤压阈值

        返回:
            挤压信号序列
        """
        bandwidth = bb_data['bandwidth']

        squeeze_signals = pd.Series(0, index=bandwidth.index)

        # 带宽极窄，可能是盘整期
        squeeze_signals[bandwidth < squeeze_threshold] = 1

        return squeeze_signals

    @staticmethod
    def volatility_breakout(bb_data: Dict[str, pd.Series],
                          prices: Union[List[float], pd.Series],
                          lookback_period: int = 20) -> pd.Series:
        """
        波动性突破策略

        参数:
            bb_data: 布林带数据
            prices: 价格序列
            lookback_period: 回看周期

        返回:
            突破信号序列
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        bandwidth = bb_data['bandwidth']

        # 计算带宽的移动平均
        avg_bandwidth = bandwidth.rolling(window=lookback_period).mean()

        breakout_signals = pd.Series(0, index=prices.index)

        # 带宽突然扩大，可能是突破信号
        breakout_signals[(bandwidth > avg_bandwidth * 1.5) & (bandwidth.shift(1) <= avg_bandwidth.shift(1) * 1.5)] = 1

        return breakout_signals

    @staticmethod
    def bollinger_bandit_strategy(prices: Union[List[float], pd.Series],
                                 period: int = 20,
                                 std_dev: float = 2.0) -> Dict[str, pd.Series]:
        """
        完整的布林带交易策略

        参数:
            prices: 价格序列
            period: 移动平均周期
            std_dev: 标准差倍数

        返回:
            策略结果字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        # 计算布林带
        bb_data = BollingerBands.calculate(prices, period, std_dev)

        # 生成交易信号
        signals = BollingerBands.generate_signals(bb_data, prices)

        # 检测挤压
        squeeze_signals = BollingerBands.squeeze_detection(bb_data)

        # 检测突破
        breakout_signals = BollingerBands.volatility_breakout(bb_data, prices)

        return {
            'bb_data': bb_data,
            'signals': signals,
            'squeeze': squeeze_signals,
            'breakout': breakout_signals
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100)) + 100

    # 计算布林带
    bb_calculator = BollingerBands()
    bb_data = bb_calculator.calculate(prices)

    # 生成交易信号
    signals = bb_calculator.generate_signals(bb_data, prices)

    # 完整策略
    strategy_results = bb_calculator.bollinger_bandit_strategy(prices)

    print("布林带指标计算完成！")
    print(f"上轨: {bb_data['upper'].iloc[-1]:.2f}")
    print(f"中轨: {bb_data['middle'].iloc[-1]:.2f}")
    print(f"下轨: {bb_data['lower'].iloc[-1]:.2f}")
    print(f"带宽: {bb_data['bandwidth'].iloc[-1]:.2f}%")
    print(f"%b指标: {bb_data['percent_b'].iloc[-1]:.2f}%")
    print(f"当前信号: {signals.iloc[-1]}")