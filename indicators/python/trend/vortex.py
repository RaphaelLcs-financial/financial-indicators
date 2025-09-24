"""
Vortex Indicator (涡旋指标) - Python实现
一种相对较新的技术指标，用于识别趋势的开始和反转。

由Etienne Botes和Douglas Siepman在2010年开发，近年来变得越来越流行。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class Vortex:
    """涡旋指标"""

    def __init__(self):
        self.name = "Vortex Indicator"
        self.category = "trend"

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series],
                  period: int = 14) -> Dict[str, pd.Series]:
        """
        计算Vortex指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认14

        返回:
            包含VIPlus, VIMinus和振荡器的字典
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        n = len(high)

        # 计算真实范围
        true_range = pd.Series(0.0, index=high.index)
        for i in range(1, n):
            tr = max(
                high.iloc[i] - low.iloc[i],
                abs(high.iloc[i] - close.iloc[i-1]),
                abs(low.iloc[i] - close.iloc[i-1])
            )
            true_range.iloc[i] = tr

        # 计算VM+和VM-
        vm_plus = pd.Series(0.0, index=high.index)
        vm_minus = pd.Series(0.0, index=high.index)

        for i in range(1, n):
            vm_plus.iloc[i] = abs(high.iloc[i] - low.iloc[i-1])
            vm_minus.iloc[i] = abs(low.iloc[i] - high.iloc[i-1])

        # 计算周期总和
        vi_plus = pd.Series(0.0, index=high.index)
        vi_minus = pd.Series(0.0, index=high.index)

        for i in range(period, n):
            sum_tr = true_range.iloc[i-period+1:i+1].sum()
            sum_vm_plus = vm_plus.iloc[i-period+1:i+1].sum()
            sum_vm_minus = vm_minus.iloc[i-period+1:i+1].sum()

            if sum_tr > 0:
                vi_plus.iloc[i] = sum_vm_plus / sum_tr
                vi_minus.iloc[i] = sum_vm_minus / sum_tr

        # 计算振荡器
        oscillator = vi_plus - vi_minus

        return {
            'vi_plus': vi_plus,
            'vi_minus': vi_minus,
            'oscillator': oscillator,
            'true_range': true_range
        }

    @staticmethod
    def generate_signals(vortex_data: Dict[str, pd.Series]) -> pd.Series:
        """
        生成Vortex交易信号

        参数:
            vortex_data: Vortex指标数据

        返回:
            信号序列 (1=买入, -1=卖出, 0=持有)
        """
        vi_plus = vortex_data['vi_plus']
        vi_minus = vortex_data['vi_minus']

        signals = pd.Series(0, index=vi_plus.index)

        for i in range(1, len(vi_plus)):
            # VI+上穿VI- - 买入信号
            if vi_plus.iloc[i] > vi_minus.iloc[i] and vi_plus.iloc[i-1] <= vi_minus.iloc[i-1]:
                signals.iloc[i] = 1
            # VI+下穿VI- - 卖出信号
            elif vi_plus.iloc[i] < vi_minus.iloc[i] and vi_plus.iloc[i-1] >= vi_minus.iloc[i-1]:
                signals.iloc[i] = -1

        return signals

    @staticmethod
    def divergence_analysis(prices: Union[List[float], pd.Series],
                           vortex_data: Dict[str, pd.Series],
                           period: int = 20) -> Dict[str, pd.Series]:
        """
        Vortex背离分析

        参数:
            prices: 价格序列
            vortex_data: Vortex数据
            period: 分析周期

        返回:
            背离信号字典
        """
        if isinstance(prices, list):
            prices = pd.Series(prices)

        vi_plus = vortex_data['vi_plus']
        vi_minus = vortex_data['vi_minus']

        # 计算价格和指标的趋势
        price_trend = prices.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])
        vi_trend = vi_plus.rolling(window=period).apply(lambda x: x.iloc[-1] - x.iloc[0])

        # 检测看涨背离（价格下跌，VI上涨）
        bullish_divergence = pd.Series(0, index=prices.index)
        bullish_divergence[(price_trend < 0) & (vi_trend > 0) & (vi_plus < 0.5)] = 1

        # 检测看跌背离（价格上涨，VI下跌）
        bearish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence[(price_trend > 0) & (vi_trend < 0) & (vi_plus > 1.5)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }

    @staticmethod
    def trend_strength(vortex_data: Dict[str, pd.Series]) -> pd.Series:
        """
        计算Vortex趋势强度

        参数:
            vortex_data: Vortex数据

        返回:
            趋势强度序列
        """
        vi_plus = vortex_data['vi_plus']
        vi_minus = vortex_data['vi_minus']

        # 计算趋势强度
        strength = pd.Series(0.0, index=vi_plus.index)

        for i in range(len(vi_plus)):
            if vi_plus.iloc[i] > vi_minus.iloc[i]:
                strength.iloc[i] = vi_plus.iloc[i] / (vi_plus.iloc[i] + vi_minus.iloc[i])
            else:
                strength.iloc[i] = -vi_minus.iloc[i] / (vi_plus.iloc[i] + vi_minus.iloc[i])

        return strength


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算Vortex指标
    vortex_calculator = Vortex()
    vortex_data = vortex_calculator.calculate(high, low, close)

    # 生成交易信号
    signals = vortex_calculator.generate_signals(vortex_data)

    # 背离分析
    divergence = vortex_calculator.divergence_analysis(close, vortex_data)

    # 趋势强度
    strength = vortex_calculator.trend_strength(vortex_data)

    print("Vortex Indicator (涡旋指标) 计算完成！")
    print(f"VI+: {vortex_data['vi_plus'].iloc[-1]:.4f}")
    print(f"VI-: {vortex_data['vi_minus'].iloc[-1]:.4f}")
    print(f"振荡器: {vortex_data['oscillator'].iloc[-1]:.4f}")
    print(f"信号: {signals.iloc[-1]}")
    print(f"趋势强度: {strength.iloc[-1]:.4f}")
    print(f"看涨背离: {divergence['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence['bearish_divergence'].iloc[-1]}")