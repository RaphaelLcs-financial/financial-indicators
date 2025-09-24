"""
MACD (移动平均收敛发散) 指标实现
Moving Average Convergence Divergence (MACD) Implementation

MACD是一个趋势跟踪动量指标，显示价格的两个移动平均线之间的关系。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class MACD:
    """MACD指标"""

    def __init__(self):
        pass

    @staticmethod
    def calculate(prices: Union[List[float], pd.Series],
                  fast_period: int = 12,
                  slow_period: int = 26,
                  signal_period: int = 9) -> Dict[str, pd.Series]:
        """
        计算MACD指标

        参数:
            prices: 价格序列
            fast_period: 快速EMA周期，默认12
            slow_period: 慢速EMA周期，默认26
            signal_period: 信号线EMA周期，默认9

        返回:
            包含MACD线、信号线和直方图的字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        # 计算快速和慢速EMA
        ema_fast = prices.ewm(span=fast_period, adjust=False).mean()
        ema_slow = prices.ewm(span=slow_period, adjust=False).mean()

        # 计算MACD线
        macd_line = ema_fast - ema_slow

        # 计算信号线
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()

        # 计算直方图
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def generate_signals(macd_data: Dict[str, pd.Series]) -> pd.Series:
        """
        生成MACD交易信号

        参数:
            macd_data: 包含MACD数据的字典

        返回:
            信号序列 (1=买入, -1=卖出, 0=持有)
        """
        macd = macd_data['macd']
        signal = macd_data['signal']
        histogram = macd_data['histogram']

        signals = pd.Series(0, index=macd.index)

        # 金叉买入信号
        signals[(macd > signal) & (macd.shift(1) <= signal.shift(1))] = 1

        # 死叉卖出信号
        signals[(macd < signal) & (macd.shift(1) >= signal.shift(1))] = -1

        # 直方图背离信号
        signals[(histogram > 0) & (histogram.shift(1) < 0)] = 1  # 直方图由负转正
        signals[(histogram < 0) & (histogram.shift(1) > 0)] = -1  # 直方图由正转负

        return signals

    @staticmethod
    def divergence_analysis(prices: Union[List[float], pd.Series],
                           macd_data: Dict[str, pd.Series],
                           period: int = 20) -> Dict[str, pd.Series]:
        """
        MACD背离分析

        参数:
            prices: 价格序列
            macd_data: MACD数据
            period: 分析周期

        返回:
            背离信号字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        macd = macd_data['macd']

        # 计算价格和MACD的趋势
        price_trend = prices.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])
        macd_trend = macd.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])

        # 检测看涨背离（价格下跌，MACD上涨）
        bullish_divergence = pd.Series(0, index=prices.index)
        bullish_divergence[(price_trend < 0) & (macd_trend > 0)] = 1

        # 检测看跌背离（价格上涨，MACD下跌）
        bearish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence[(price_trend > 0) & (macd_trend < 0)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }

    @staticmethod
    def zero_line_crossover(macd: Union[List[float], pd.Series]) -> pd.Series:
        """
        零轴交叉信号

        参数:
            macd: MACD线序列

        返回:
            零轴交叉信号
        """
        if isinstance(macd, list):
            macd = pd.Series(macd)

        signals = pd.Series(0, index=macd.index)

        # 向上穿越零轴
        signals[(macd > 0) & (macd.shift(1) <= 0)] = 1

        # 向下穿越零轴
        signals[(macd < 0) & (macd.shift(1) >= 0)] = -1

        return signals


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100)) + 100

    # 计算MACD
    macd_calculator = MACD()
    macd_data = macd_calculator.calculate(prices)

    # 生成交易信号
    signals = macd_calculator.generate_signals(macd_data)

    # 零轴交叉分析
    zero_cross_signals = macd_calculator.zero_line_crossover(macd_data['macd'])

    # 背离分析
    divergence_signals = macd_calculator.divergence_analysis(prices, macd_data)

    print("MACD指标计算完成！")
    print(f"MACD线: {macd_data['macd'].iloc[-1]:.4f}")
    print(f"信号线: {macd_data['signal'].iloc[-1]:.4f}")
    print(f"直方图: {macd_data['histogram'].iloc[-1]:.4f}")
    print(f"当前信号: {signals.iloc[-1]}")
    print(f"零轴交叉信号: {zero_cross_signals.iloc[-1]}")