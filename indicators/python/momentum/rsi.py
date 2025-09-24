"""
RSI (相对强弱指数) 指标实现
Relative Strength Index (RSI) Implementation

RSI是一个动量指标，用于衡量价格变动的速度和变化，识别超买超卖条件。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class RSI:
    """相对强弱指数指标"""

    def __init__(self):
        pass

    @staticmethod
    def calculate(prices: Union[List[float], pd.Series], period: int = 14) -> pd.Series:
        """
        计算RSI指标

        参数:
            prices: 价格序列
            period: 计算周期，默认14

        返回:
            RSI值序列
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        # 计算价格变化
        delta = prices.diff()

        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 使用Wilder的平滑方法计算平均收益和平均损失
        # 第一个值使用简单移动平均
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        # 从第period+1个值开始使用Wilder平滑
        for i in range(period, len(gain)):
            avg_gain.iloc[i] = (avg_gain.iloc[i-1] * (period - 1) + gain.iloc[i]) / period
            avg_loss.iloc[i] = (avg_loss.iloc[i-1] * (period - 1) + loss.iloc[i]) / period

        # 计算相对强度
        rs = avg_gain / avg_loss

        # 计算RSI
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def generate_signals(rsi: Union[List[float], pd.Series],
                        oversold: float = 30,
                        overbought: float = 70) -> pd.Series:
        """
        生成RSI交易信号

        参数:
            rsi: RSI值序列
            oversold: 超卖阈值，默认30
            overbought: 超买阈值，默认70

        返回:
            信号序列 (1=买入, -1=卖出, 0=持有)
        """
        if isinstance(rsi, list):
            rsi = pd.Series(rsi)

        signals = pd.Series(0, index=rsi.index)

        # 超卖买入信号
        signals[rsi < oversold] = 1

        # 超买卖出信号
        signals[rsi > overbought] = -1

        return signals

    @staticmethod
    def divergence(prices: Union[List[float], pd.Series],
                  rsi: Union[List[float], pd.Series],
                  period: int = 14) -> Dict[str, pd.Series]:
        """
        检测价格与RSI的背离

        参数:
            prices: 价格序列
            rsi: RSI序列
            period: 检测周期

        返回:
            包含背离信号的字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)
        if isinstance(rsi, list):
            rsi = pd.Series(rsi)

        # 计算价格高点和低点
        price_highs = prices.rolling(window=period).max()
        price_lows = prices.rolling(window=period).min()

        # 计算RSI高点和低点
        rsi_highs = rsi.rolling(window=period).max()
        rsi_lows = rsi.rolling(window=period).min()

        # 检测看涨背离（价格创新低，RSI没有创新低）
        bullish_divergence = pd.Series(0, index=prices.index)
        bullish_divergence[(prices == price_lows) & (rsi > rsi_lows)] = 1

        # 检测看跌背离（价格创新高，RSI没有创新高）
        bearish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence[(prices == price_highs) & (rsi < rsi_highs)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    prices = np.cumsum(np.random.randn(100)) + 100

    # 计算RSI
    rsi_calculator = RSI()
    rsi_values = rsi_calculator.calculate(prices, period=14)

    # 生成交易信号
    signals = rsi_calculator.generate_signals(rsi_values)

    # 检测背离
    divergence_signals = rsi_calculator.divergence(prices, rsi_values)

    print("RSI指标计算完成！")
    print(f"当前RSI值: {rsi_values.iloc[-1]:.2f}")
    print(f"当前信号: {signals.iloc[-1]}")
    print(f"看涨背离: {divergence_signals['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence_signals['bearish_divergence'].iloc[-1]}")