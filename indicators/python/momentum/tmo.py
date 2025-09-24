"""
TMO (True Momentum Oscillator) - 真实动量振荡器
一个较新的动量指标，结合了RSI和Stochastic的优点，提供更准确的动量测量。

这个指标是由TradingView社区开发的，在2020年后开始流行。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class TMO:
    """真实动量振荡器"""

    def __init__(self):
        self.name = "True Momentum Oscillator"
        self.category = "momentum"

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series],
                  length: int = 14,
                  smooth: int = 3) -> Dict[str, pd.Series]:
        """
        计算TMO指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            length: 计算周期，默认14
            smooth: 平滑周期，默认3

        返回:
            包含TMO线、信号线和直方图的字典
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算最高价和最低价的移动平均
        highest_high = high.rolling(window=length).max()
        lowest_low = low.rolling(window=length).min()

        # 计算TMO原始值
        tmo_raw = ((close - lowest_low) / (highest_high - lowest_low)) * 100

        # 平滑处理
        tmo_line = tmo_raw.rolling(window=smooth).mean()
        signal_line = tmo_line.rolling(window=smooth).mean()

        # 计算直方图
        histogram = tmo_line - signal_line

        return {
            'tmo_line': tmo_line,
            'signal_line': signal_line,
            'histogram': histogram
        }

    @staticmethod
    def generate_signals(tmo_data: Dict[str, pd.Series],
                        overbought: float = 80,
                        oversold: float = 20) -> pd.Series:
        """
        生成TMO交易信号

        参数:
            tmo_data: TMO指标数据
            overbought: 超买阈值，默认80
            oversold: 超卖阈值，默认20

        返回:
            信号序列 (1=买入, -1=卖出, 0=持有)
        """
        tmo_line = tmo_data['tmo_line']
        signal_line = tmo_data['signal_line']
        histogram = tmo_data['histogram']

        signals = pd.Series(0, index=tmo_line.index)

        # TMO线上穿信号线 - 买入
        signals[(tmo_line > signal_line) & (tmo_line.shift(1) <= signal_line.shift(1))] = 1

        # TMO线下穿信号线 - 卖出
        signals[(tmo_line < signal_line) & (tmo_line.shift(1) >= signal_line.shift(1))] = -1

        # 超卖区域买入确认
        signals[(tmo_line < oversold) & (tmo_line > signal_line)] = 1

        # 超买区域卖出确认
        signals[(tmo_line > overbought) & (tmo_line < signal_line)] = -1

        # 直方图背离信号
        signals[(histogram > 0) & (histogram.shift(1) < 0)] = 1  # 直方图由负转正
        signals[(histogram < 0) & (histogram.shift(1) > 0)] = -1  # 直方图由正转负

        return signals

    @staticmethod
    def divergence_analysis(prices: Union[List[float], pd.Series],
                           tmo_data: Dict[str, pd.Series],
                           period: int = 20) -> Dict[str, pd.Series]:
        """
        TMO背离分析

        参数:
            prices: 价格序列
            tmo_data: TMO数据
            period: 分析周期

        返回:
            背离信号字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        tmo_line = tmo_data['tmo_line']

        # 计算价格和TMO的趋势
        price_trend = prices.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])
        tmo_trend = tmo_line.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])

        # 检测看涨背离（价格下跌，TMO上涨）
        bullish_divergence = pd.Series(0, index=prices.index)
        bullish_divergence[(price_trend < 0) & (tmo_trend > 0) & (tmo_line < 30)] = 1

        # 检测看跌背离（价格上涨，TMO下跌）
        bearish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence[(price_trend > 0) & (tmo_trend < 0) & (tmo_line > 70)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }

    @staticmethod
    def tmo_zones(tmo_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        TMO区域分析

        参数:
            tmo_data: TMO数据

        返回:
            区域信号字典
        """
        tmo_line = tmo_data['tmo_line']

        # 定义不同区域
        extreme_oversold = pd.Series(0, index=tmo_line.index)
        oversold = pd.Series(0, index=tmo_line.index)
        neutral = pd.Series(0, index=tmo_line.index)
        overbought = pd.Series(0, index=tmo_line.index)
        extreme_overbought = pd.Series(0, index=tmo_line.index)

        # 极度超卖区域
        extreme_oversold[tmo_line <= 10] = 1

        # 超卖区域
        oversold[(tmo_line > 10) & (tmo_line <= 30)] = 1

        # 中性区域
        neutral[(tmo_line > 30) & (tmo_line <= 70)] = 1

        # 超买区域
        overbought[(tmo_line > 70) & (tmo_line <= 90)] = 1

        # 极度超买区域
        extreme_overbought[tmo_line > 90] = 1

        return {
            'extreme_oversold': extreme_oversold,
            'oversold': oversold,
            'neutral': neutral,
            'overbought': overbought,
            'extreme_overbought': extreme_overbought
        }

    @staticmethod
    def advanced_strategy(high: Union[List[float], pd.Series],
                         low: Union[List[float], pd.Series],
                         close: Union[List[float], pd.Series],
                         length: int = 14,
                         smooth: int = 3) -> Dict[str, Any]:
        """
        高级TMO策略

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            length: 计算周期
            smooth: 平滑周期

        返回:
            完整策略结果
        """
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算TMO指标
        tmo_data = TMO.calculate(high, low, close, length, smooth)

        # 生成基本信号
        basic_signals = TMO.generate_signals(tmo_data)

        # 背离分析
        divergence_signals = TMO.divergence_analysis(close, tmo_data)

        # 区域分析
        zone_signals = TMO.tmo_zones(tmo_data)

        # 多重确认信号
        confirmed_signals = pd.Series(0, index=close.index)

        # 强买入信号：背离 + 超卖 + 信号线金叉
        strong_buy = (
            (divergence_signals['bullish_divergence'] == 1) |
            (zone_signals['extreme_oversold'] == 1)
        ) & (basic_signals == 1)
        confirmed_signals[strong_buy] = 2  # 强买入信号

        # 强卖出信号：背离 + 超买 + 信号线死叉
        strong_sell = (
            (divergence_signals['bearish_divergence'] == -1) |
            (zone_signals['extreme_overbought'] == 1)
        ) & (basic_signals == -1)
        confirmed_signals[strong_sell] = -2  # 强卖出信号

        # 普通信号
        normal_signals = (basic_signals != 0) & (confirmed_signals == 0)
        confirmed_signals[normal_signals] = basic_signals[normal_signals]

        return {
            'tmo_data': tmo_data,
            'basic_signals': basic_signals,
            'divergence_signals': divergence_signals,
            'zone_signals': zone_signals,
            'confirmed_signals': confirmed_signals
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算TMO指标
    tmo_calculator = TMO()
    tmo_data = tmo_calculator.calculate(high, low, close)

    # 生成交易信号
    signals = tmo_calculator.generate_signals(tmo_data)

    # 背离分析
    divergence_signals = tmo_calculator.divergence_analysis(close, tmo_data)

    # 区域分析
    zone_signals = tmo_calculator.tmo_zones(tmo_data)

    # 高级策略
    advanced_results = tmo_calculator.advanced_strategy(high, low, close)

    print("TMO (True Momentum Oscillator) 指标计算完成！")
    print(f"TMO线: {tmo_data['tmo_line'].iloc[-1]:.2f}")
    print(f"信号线: {tmo_data['signal_line'].iloc[-1]:.2f}")
    print(f"直方图: {tmo_data['histogram'].iloc[-1]:.2f}")
    print(f"基本信号: {signals.iloc[-1]}")
    print(f"看涨背离: {divergence_signals['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence_signals['bearish_divergence'].iloc[-1]}")
    print(f"确认信号: {advanced_results['confirmed_signals'].iloc[-1]}")