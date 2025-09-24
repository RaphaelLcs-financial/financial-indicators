"""
OBV (能量潮) 指标实现
On Balance Volume (OBV) Implementation

OBV是一个成交量指标，通过成交量的累积变化来预测价格走势。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class OBV:
    """能量潮指标"""

    def __init__(self):
        self.name = "On Balance Volume"
        self.category = "volume"

    @staticmethod
    def calculate(prices: Union[List[float], pd.Series],
                  volume: Union[List[float], pd.Series]) -> pd.Series:
        """
        计算OBV指标

        参数:
            prices: 价格序列
            volume: 成交量序列

        返回:
            OBV值序列
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)
        if isinstance(volume, list):
            volume = pd.Series(volume)

        # 计算价格变化
        price_change = prices.diff()

        # 初始化OBV序列
        obv = pd.Series(0.0, index=prices.index)

        # 计算OBV
        for i in range(1, len(prices)):
            if price_change.iloc[i] > 0:  # 上涨日
                obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
            elif price_change.iloc[i] < 0:  # 下跌日
                obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
            else:  # 平盘日
                obv.iloc[i] = obv.iloc[i-1]

        return obv

    @staticmethod
    def obv_ma_cross(obv: Union[List[float], pd.Series],
                    ma_period: int = 20) -> Dict[str, pd.Series]:
        """
        OBV移动平均线交叉策略

        参数:
            obv: OBV序列
            ma_period: 移动平均周期

        返回:
            包含OBV、移动平均线和信号的字典
        """
        if isinstance(obv, list):
            obv = pd.Series(obv)

        # 计算OBV的移动平均
        obv_ma = obv.rolling(window=ma_period).mean()

        # 生成交叉信号
        signals = pd.Series(0, index=obv.index)
        signals[(obv > obv_ma) & (obv.shift(1) <= obv_ma.shift(1))] = 1  # 金叉
        signals[(obv < obv_ma) & (obv.shift(1) >= obv_ma.shift(1))] = -1  # 死叉

        return {
            'obv': obv,
            'obv_ma': obv_ma,
            'signals': signals
        }

    @staticmethod
    def divergence_analysis(prices: Union[List[float], pd.Series],
                           obv: Union[List[float], pd.Series],
                           period: int = 20) -> Dict[str, pd.Series]:
        """
        OBV背离分析

        参数:
            prices: 价格序列
            obv: OBV序列
            period: 分析周期

        返回:
            背离信号字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)
        if isinstance(obv, list):
            obv = pd.Series(obv)

        # 计算价格和OBV的趋势
        price_trend = prices.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])
        obv_trend = obv.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])

        # 检测看涨背离（价格下跌，OBV上涨）
        bullish_divergence = pd.Series(0, index=prices.index)
        bullish_divergence[(price_trend < 0) & (obv_trend > 0)] = 1

        # 检测看跌背离（价格上涨，OBV下跌）
        bearish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence[(price_trend > 0) & (obv_trend < 0)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }

    @staticmethod
    def breakout_detection(obv: Union[List[float], pd.Series],
                          period: int = 20) -> pd.Series:
        """
        OBV突破检测

        参数:
            obv: OBV序列
            period: 突破检测周期

        返回:
            突破信号序列
        """
        if isinstance(obv, list):
            obv = pd.Series(obv)

        # 计算OBV的最高价和最低价
        obv_high = obv.rolling(window=period).max()
        obv_low = obv.rolling(window=period).min()

        breakout_signals = pd.Series(0, index=obv.index)

        # 向上突破
        breakout_signals[(obv > obv_high) & (obv.shift(1) <= obv_high.shift(1))] = 1

        # 向下突破
        breakout_signals[(obv < obv_low) & (obv.shift(1) >= obv_low.shift(1))] = -1

        return breakout_signals

    @staticmethod
    def volume_weighted_obv(prices: Union[List[float], pd.Series],
                           volume: Union[List[float], pd.Series],
                           period: int = 20) -> pd.Series:
        """
        成交量加权OBV

        参数:
            prices: 价格序列
            volume: 成交量序列
            period: 计算周期

        返回:
            加权OBV序列
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)
        if isinstance(volume, list):
            volume = pd.Series(volume)

        # 计算价格变化百分比
        price_change_pct = prices.pct_change()

        # 计算加权OBV
        weighted_obv = pd.Series(0.0, index=prices.index)

        for i in range(1, len(prices)):
            if price_change_pct.iloc[i] > 0:
                weighted_obv.iloc[i] = weighted_obv.iloc[i-1] + (volume.iloc[i] * (1 + price_change_pct.iloc[i]))
            elif price_change_pct.iloc[i] < 0:
                weighted_obv.iloc[i] = weighted_obv.iloc[i-1] - (volume.iloc[i] * (1 - price_change_pct.iloc[i]))
            else:
                weighted_obv.iloc[i] = weighted_obv.iloc[i-1]

        return weighted_obv


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    prices = np.cumsum(np.random.randn(n_periods)) + 100
    volume = np.random.randint(1000000, 5000000, n_periods)

    # 计算OBV
    obv_calculator = OBV()
    obv_values = obv_calculator.calculate(prices, volume)

    # OBV移动平均线交叉
    ma_cross_signals = obv_calculator.obv_ma_cross(obv_values)

    # 背离分析
    divergence_signals = obv_calculator.divergence_analysis(prices, obv_values)

    # 突破检测
    breakout_signals = obv_calculator.breakout_detection(obv_values)

    # 加权OBV
    weighted_obv = obv_calculator.volume_weighted_obv(prices, volume)

    print("OBV指标计算完成！")
    print(f"OBV值: {obv_values.iloc[-1]:.0f}")
    print(f"OBV移动平均: {ma_cross_signals['obv_ma'].iloc[-1]:.0f}")
    print(f"交叉信号: {ma_cross_signals['signals'].iloc[-1]}")
    print(f"看涨背离: {divergence_signals['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence_signals['bearish_divergence'].iloc[-1]}")
    print(f"突破信号: {breakout_signals.iloc[-1]}")
    print(f"加权OBV: {weighted_obv.iloc[-1]:.0f}")