"""
QQE (Quantitative Qualitative Estimation) - 定量定性估计
一个较新的指标，结合RSI和ATR，提供更准确的买卖信号。

近年来在TradingView社区非常流行，被认为是RSI的改良版。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class QQE:
    """定量定性估计指标"""

    def __init__(self):
        self.name = "Quantitative Qualitative Estimation"
        self.category = "momentum"

    @staticmethod
    def calculate_rsi(prices: Union[List[float], pd.Series], period: int = 14) -> pd.Series:
        """计算RSI"""
        if isinstance(prices, list):
            prices = pd.Series(prices)

        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def calculate_wilder_ma(data: pd.Series, period: int) -> pd.Series:
        """计算Wilder移动平均"""
        wilder_ma = pd.Series(0.0, index=data.index)
        wilder_ma.iloc[period-1] = data.iloc[period-1]

        for i in range(period, len(data)):
            wilder_ma.iloc[i] = (wilder_ma.iloc[i-1] * (period-1) + data.iloc[i]) / period

        return wilder_ma

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series]],
                  close: Union[List[float], pd.Series],
                  rsi_period: int = 14,
                  smooth_period: int = 5,
                  atr_period: int = 14,
                  factor: float = 4.236) -> Dict[str, pd.Series]:
        """
        计算QQE指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            rsi_period: RSI周期
            smooth_period: 平滑周期
            atr_period: ATR周期
            factor: QQE因子 (黄金比例)

        返回:
            QQE指标数据
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算RSI
        rsi = QQE.calculate_rsi(close, rsi_period)

        # 计算RSI的平滑移动平均
        rsi_ma = QQE.calculate_wilder_ma(rsi, smooth_period)

        # 计算ATR
        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=atr_period).mean()

        # 计算QQE的快速线和慢速线
        fast_line = rsi_ma.rolling(window=smooth_period).mean()
        slow_line = rsi_ma.rolling(window=smooth_period*2).mean()

        # 计算QQE的带宽
        bandwidth = atr * factor / close * 100

        # 计算QQE的上下通道
        upper_channel = fast_line + bandwidth
        lower_channel = fast_line - bandwidth

        return {
            'rsi': rsi,
            'rsi_ma': rsi_ma,
            'fast_line': fast_line,
            'slow_line': slow_line,
            'upper_channel': upper_channel,
            'lower_channel': lower_channel,
            'bandwidth': bandwidth
        }

    @staticmethod
    def generate_signals(qqe_data: Dict[str, pd.Series]) -> pd.Series:
        """
        生成QQE交易信号

        参数:
            qqe_data: QQE指标数据

        返回:
            交易信号序列
        """
        fast_line = qqe_data['fast_line']
        slow_line = qqe_data['slow_line']
        upper_channel = qqe_data['upper_channel']
        lower_channel = qqe_data['lower_channel']

        signals = pd.Series(0, index=fast_line.index)

        for i in range(1, len(fast_line)):
            # 快线上穿慢线 - 买入信号
            if fast_line.iloc[i] > slow_line.iloc[i] and fast_line.iloc[i-1] <= slow_line.iloc[i-1]:
                signals.iloc[i] = 1
            # 快线下穿慢线 - 卖出信号
            elif fast_line.iloc[i] < slow_line.iloc[i] and fast_line.iloc[i-1] >= slow_line.iloc[i-1]:
                signals.iloc[i] = -1

            # 价格突破上通道 - 强买入
            elif fast_line.iloc[i] > upper_channel.iloc[i]:
                signals.iloc[i] = 2
            # 价格突破下通道 - 强卖出
            elif fast_line.iloc[i] < lower_channel.iloc[i]:
                signals.iloc[i] = -2

        return signals

    @staticmethod
    def divergence_detection(prices: pd.Series,
                             qqe_data: Dict[str, pd.Series],
                             period: int = 20) -> Dict[str, pd.Series]:
        """
        QQE背离检测

        参数:
            prices: 价格序列
            qqe_data: QQE数据
            period: 检测周期

        返回:
            背离信号
        """
        rsi = qqe_data['rsi']

        # 检测价格和RSI的高点低点
        price_highs = prices.rolling(window=period).max()
        price_lows = prices.rolling(window=period).min()
        rsi_highs = rsi.rolling(window=period).max()
        rsi_lows = rsi.rolling(window=period).min()

        bullish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence = pd.Series(0, index=prices.index)

        # 看涨背离：价格创新低，RSI没有创新低
        bullish_divergence[(prices == price_lows) & (rsi > rsi_lows)] = 1

        # 看跌背离：价格创新高，RSI没有创新高
        bearish_divergence[(prices == price_highs) & (rsi < rsi_highs)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算QQE
    qqe_calculator = QQE()
    qqe_data = qqe_calculator.calculate(high, low, close)

    # 生成信号
    signals = qqe_calculator.generate_signals(qqe_data)

    # 背离检测
    divergence = qqe_calculator.divergence_detection(close, qqe_data)

    print("QQE (定量定性估计) 指标计算完成！")
    print(f"快速线: {qqe_data['fast_line'].iloc[-1]:.2f}")
    print(f"慢速线: {qqe_data['slow_line'].iloc[-1]:.2f}")
    print(f"上通道: {qqe_data['upper_channel'].iloc[-1]:.2f}")
    print(f"下通道: {qqe_data['lower_channel'].iloc[-1]:.2f}")
    print(f"信号: {signals.iloc[-1]}")
    print(f"看涨背离: {divergence['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence['bearish_divergence'].iloc[-1]}")