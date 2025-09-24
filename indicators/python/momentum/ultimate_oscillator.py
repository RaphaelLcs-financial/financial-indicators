"""
Ultimate Oscillator (终极振荡器) - Python实现
结合多个时间框架的动量指标，减少虚假信号。

由Larry Williams开发，是一个多时间框架的动量振荡器。
近年来在市场分析中重新获得关注，特别是在识别超买超卖方面。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class UltimateOscillator:
    """终极振荡器"""

    def __init__(self):
        self.name = "Ultimate Oscillator"
        self.category = "momentum"

    @staticmethod
    def calculate_true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
        """
        计算真实范围

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        返回:
            真实范围序列
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    @staticmethod
    def calculate_buying_pressure(low: pd.Series, close: pd.Series, prev_close: pd.Series) -> pd.Series:
        """
        计算买入压力

        参数:
            low: 最低价序列
            close: 收盘价序列
            prev_close: 前一日收盘价序列

        返回:
            买入压力序列
        """
        return close - np.minimum(low, prev_close)

    @staticmethod
    def calculate_average(bp: pd.Series, tr: pd.Series, period: int) -> pd.Series:
        """
        计算平均值

        参数:
            bp: 买入压力序列
            tr: 真实范围序列
            period: 周期

        返回:
            平均值序列
        """
        bp_sum = bp.rolling(window=period).sum()
        tr_sum = tr.rolling(window=period).sum()

        average = pd.Series(0.0, index=bp.index)
        for i in range(len(tr_sum)):
            if tr_sum.iloc[i] > 0:
                average.iloc[i] = bp_sum.iloc[i] / tr_sum.iloc[i]

        return average

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series],
                  period1: int = 7,
                  period2: int = 14,
                  period3: int = 28) -> Dict[str, pd.Series]:
        """
        计算Ultimate Oscillator指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period1: 短期周期，默认7
            period2: 中期周期，默认14
            period3: 长期周期，默认28

        返回:
            包含Ultimate Oscillator及其组件的字典
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算真实范围和买入压力
        true_range = UltimateOscillator.calculate_true_range(high, low, close)
        prev_close = close.shift(1)
        buying_pressure = UltimateOscillator.calculate_buying_pressure(low, close, prev_close)

        # 计算三个时间框架的平均值
        avg1 = UltimateOscillator.calculate_average(buying_pressure, true_range, period1)
        avg2 = UltimateOscillator.calculate_average(buying_pressure, true_range, period2)
        avg3 = UltimateOscillator.calculate_average(buying_pressure, true_range, period3)

        # 计算Ultimate Oscillator
        total_periods = period1 + period2 + period3
        ultimate_osc = pd.Series(0.0, index=close.index)

        for i in range(len(close)):
            if i >= period3 - 1:  # 确保有足够的数据
                ultimate_osc.iloc[i] = (4 * avg1.iloc[i] + 2 * avg2.iloc[i] + avg3.iloc[i]) / total_periods * 100

        return {
            'ultimate_osc': ultimate_osc,
            'avg1': avg1,
            'avg2': avg2,
            'avg3': avg3,
            'buying_pressure': buying_pressure,
            'true_range': true_range
        }

    @staticmethod
    def generate_signals(ultimate_data: Dict[str, pd.Series],
                        overbought: float = 70,
                        oversold: float = 30,
                        divergence_lookback: int = 7) -> pd.Series:
        """
        生成Ultimate Oscillator交易信号

        参数:
            ultimate_data: Ultimate Oscillator数据
            overbought: 超买阈值，默认70
            oversold: 超卖阈值，默认30
            divergence_lookback: 背离回看周期，默认7

        返回:
            交易信号序列
        """
        uo = ultimate_data['ultimate_osc']
        close_prices = ultimate_data.get('close_prices', pd.Series(0, index=uo.index))

        signals = pd.Series(0, index=uo.index)

        for i in range(1, len(uo)):
            # 基本超买超卖信号
            if uo.iloc[i] < oversold and uo.iloc[i-1] >= oversold:
                signals.iloc[i] = 1  # 买入信号
            elif uo.iloc[i] > overbought and uo.iloc[i-1] <= overbought:
                signals.iloc[i] = -1  # 卖出信号

            # Larry Williams的3个买入信号
            if i >= divergence_lookback:
                # 买入信号1: 看涨背离
                price_low_idx = i - divergence_lookback + np.argmin(close_prices.iloc[i-divergence_lookback+1:i+1])
                uo_low_idx = i - divergence_lookback + np.argmin(uo.iloc[i-divergence_lookback+1:i+1])

                if (price_low_idx < uo_low_idx and
                    uo.iloc[i] < oversold and
                    uo.iloc[i] > uo.iloc[i-1]):
                    signals.iloc[i] = 1

                # 买入信号2: UO低于30，然后上升超过50
                if uo.iloc[i-2] < oversold and uo.iloc[i-1] < oversold and uo.iloc[i] > 50:
                    signals.iloc[i] = 1

                # 买入信号3: UO低于50，然后上升超过60
                if uo.iloc[i-2] < 50 and uo.iloc[i-1] < 50 and uo.iloc[i] > 60:
                    signals.iloc[i] = 1

            # Larry Williams的卖出信号
            if i >= divergence_lookback:
                # 卖出信号1: 看跌背离
                price_high_idx = i - divergence_lookback + np.argmax(close_prices.iloc[i-divergence_lookback+1:i+1])
                uo_high_idx = i - divergence_lookback + np.argmax(uo.iloc[i-divergence_lookback+1:i+1])

                if (price_high_idx < uo_high_idx and
                    uo.iloc[i] > overbought and
                    uo.iloc[i] < uo.iloc[i-1]):
                    signals.iloc[i] = -1

                # 卖出信号2: UO高于70，然后下降
                if uo.iloc[i-2] > overbought and uo.iloc[i-1] > overbought and uo.iloc[i] < uo.iloc[i-1]:
                    signals.iloc[i] = -1

        return signals

    @staticmethod
    def divergence_detection(prices: pd.Series,
                            ultimate_data: Dict[str, pd.Series],
                            period: int = 14) -> Dict[str, pd.Series]:
        """
        Ultimate Oscillator背离检测

        参数:
            prices: 价格序列
            ultimate_data: Ultimate Oscillator数据
            period: 检测周期

        返回:
            背离信号
        """
        uo = ultimate_data['ultimate_osc']

        # 检测价格和UO的高点低点
        price_highs = prices.rolling(window=period).max()
        price_lows = prices.rolling(window=period).min()
        uo_highs = uo.rolling(window=period).max()
        uo_lows = uo.rolling(window=period).min()

        bullish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence = pd.Series(0, index=prices.index)

        # 看涨背离：价格创新低，UO没有创新低
        bullish_divergence[(prices == price_lows) & (uo > uo_lows) & (uo < 30)] = 1

        # 看跌背离：价格创新高，UO没有创新高
        bearish_divergence[(prices == price_highs) & (uo < uo_highs) & (uo > 70)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }

    @staticmethod
    def momentum_analysis(ultimate_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        动量分析

        参数:
            ultimate_data: Ultimate Oscillator数据

        返回:
            动量分析结果
        """
        uo = ultimate_data['ultimate_osc']

        # 计算动量指标
        momentum = uo.diff()
        momentum_ma = momentum.rolling(window=5).mean()

        # 计算动量强度
        momentum_strength = pd.Series('中性', index=uo.index)
        momentum_direction = pd.Series('无方向', index=uo.index)

        for i in range(len(momentum)):
            if pd.notna(momentum.iloc[i]):
                # 动量强度判断
                if momentum.iloc[i] > 2:
                    momentum_strength.iloc[i] = '强动量'
                elif momentum.iloc[i] > 0.5:
                    momentum_strength.iloc[i] = '中等动量'
                elif momentum.iloc[i] > -0.5:
                    momentum_strength.iloc[i] = '中性'
                elif momentum.iloc[i] > -2:
                    momentum_strength.iloc[i] = '弱动量'
                else:
                    momentum_strength.iloc[i] = '极弱动量'

                # 动量方向判断
                if momentum.iloc[i] > 0:
                    momentum_direction.iloc[i] = '向上'
                elif momentum.iloc[i] < 0:
                    momentum_direction.iloc[i] = '向下'
                else:
                    momentum_direction.iloc[i] = '无方向'

        return {
            'momentum': momentum,
            'momentum_ma': momentum_ma,
            'momentum_strength': momentum_strength,
            'momentum_direction': momentum_direction
        }

    @staticmethod
    def zone_analysis(ultimate_data: Dict[str, pd.Series],
                    zone1: float = 30,
                    zone2: float = 50,
                    zone3: float = 70) -> Dict[str, pd.Series]:
        """
        区域分析

        参数:
            ultimate_data: Ultimate Oscillator数据
            zone1: 区域1阈值，默认30
            zone2: 区域2阈值，默认50
            zone3: 区域3阈值，默认70

        返回:
            区域分析结果
        """
        uo = ultimate_data['ultimate_osc']

        zones = {
            'extreme_oversold': pd.Series(0, index=uo.index),
            'oversold': pd.Series(0, index=uo.index),
            'neutral': pd.Series(0, index=uo.index),
            'overbought': pd.Series(0, index=uo.index),
            'extreme_overbought': pd.Series(0, index=uo.index)
        }

        for i in range(len(uo)):
            if pd.notna(uo.iloc[i]):
                uo_val = uo.iloc[i]
                if uo_val < 20:
                    zones['extreme_oversold'].iloc[i] = 1
                elif uo_val < zone1:
                    zones['oversold'].iloc[i] = 1
                elif uo_val < zone2:
                    zones['neutral'].iloc[i] = 1
                elif uo_val < zone3:
                    zones['overbought'].iloc[i] = 1
                else:
                    zones['extreme_overbought'].iloc[i] = 1

        return zones


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算Ultimate Oscillator
    uo_calculator = UltimateOscillator()
    uo_data = uo_calculator.calculate(high, low, close)

    # 添加价格数据到结果中（用于背离检测）
    uo_data['close_prices'] = close

    # 生成信号
    signals = uo_calculator.generate_signals(uo_data)

    # 背离检测
    divergence = uo_calculator.divergence_detection(close, uo_data)

    # 动量分析
    momentum_analysis = uo_calculator.momentum_analysis(uo_data)

    # 区域分析
    zone_analysis = uo_calculator.zone_analysis(uo_data)

    print("Ultimate Oscillator (终极振荡器) 计算完成！")
    print(f"UO值: {uo_data['ultimate_osc'].iloc[-1]:.2f}")
    print(f"短期平均值: {uo_data['avg1'].iloc[-1]:.4f}")
    print(f"中期平均值: {uo_data['avg2'].iloc[-1]:.4f}")
    print(f"长期平均值: {uo_data['avg3'].iloc[-1]:.4f}")
    print(f"信号: {signals.iloc[-1]}")
    print(f"看涨背离: {divergence['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence['bearish_divergence'].iloc[-1]}")
    print(f"动量强度: {momentum_analysis['momentum_strength'].iloc[-1]}")
    print(f"动量方向: {momentum_analysis['momentum_direction'].iloc[-1]}")