"""
Woodie's CCI (伍迪商品通道指数) - Python实现
传统CCI指标的变种，使用不同的参数和解释方法。

由Ken Wood开发的CCI交易系统，在交易员社区中广受欢迎。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class WoodiesCCI:
    """伍迪CCI指标"""

    def __init__(self):
        self.name = "Woodie's CCI"
        self.category = "momentum"
        self.cci_period = 14
        self.turbo_period = 6

    @staticmethod
    def calculate_typical_price(high: Union[List[float], pd.Series],
                              low: Union[List[float], pd.Series],
                              close: Union[List[float], pd.Series]) -> pd.Series:
        """
        计算典型价格

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        返回:
            典型价格序列
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        return (high + low + close) / 3

    @staticmethod
    def calculate_sma(data: Union[List[float], pd.Series], period: int) -> pd.Series:
        """
        计算简单移动平均

        参数:
            data: 数据序列
            period: 周期

        返回:
            移动平均序列
        """
        if isinstance(data, list):
            data = pd.Series(data)

        return data.rolling(window=period).mean()

    @staticmethod
    def calculate_mean_deviation(data: Union[List[float], pd.Series],
                               sma: pd.Series,
                               period: int) -> pd.Series:
        """
        计算平均偏差

        参数:
            data: 数据序列
            sma: 移动平均序列
            period: 周期

        返回:
            平均偏差序列
        """
        if isinstance(data, list):
            data = pd.Series(data)

        mean_deviation = pd.Series(0.0, index=data.index)

        for i in range(period-1, len(data)):
            if pd.notna(sma.iloc[i]):
                deviations = [abs(data.iloc[j] - sma.iloc[i]) for j in range(i-period+1, i+1)]
                mean_deviation.iloc[i] = sum(deviations) / period

        return mean_deviation

    @staticmethod
    def calculate_cci(typical_prices: pd.Series, period: int) -> pd.Series:
        """
        计算CCI值

        参数:
            typical_prices: 典型价格序列
            period: 计算周期

        返回:
            CCI值序列
        """
        sma = WoodiesCCI.calculate_sma(typical_prices, period)
        mean_deviation = WoodiesCCI.calculate_mean_deviation(typical_prices, sma, period)

        cci = pd.Series(0.0, index=typical_prices.index)

        for i in range(len(typical_prices)):
            if pd.notna(mean_deviation.iloc[i]) and mean_deviation.iloc[i] != 0:
                cci.iloc[i] = (typical_prices.iloc[i] - sma.iloc[i]) / (0.015 * mean_deviation.iloc[i])

        return cci

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        计算Woodie's CCI完整系统

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        返回:
            包含所有Woodie's CCI组件的数据
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算典型价格
        typical_prices = WoodiesCCI.calculate_typical_price(high, low, close)

        # 主CCI
        main_cci = WoodiesCCI.calculate_cci(typical_prices, 14)

        # Turbo CCI (较快的CCI)
        turbo_cci = WoodiesCCI.calculate_cci(typical_prices, 6)

        # 计算CCI的移动平均线
        main_cci_ma = WoodiesCCI.calculate_sma(main_cci, 9)
        turbo_cci_ma = WoodiesCCI.calculate_sma(turbo_cci, 9)

        # 计算零线交叉
        zero_line_cross = WoodiesCCI.calculate_zero_line_cross(main_cci)

        # 计算趋势线
        trend_line = WoodiesCCI.calculate_trend_line(high, low, close)

        return {
            'main_cci': main_cci,
            'turbo_cci': turbo_cci,
            'main_cci_ma': main_cci_ma,
            'turbo_cci_ma': turbo_cci_ma,
            'zero_line_cross': zero_line_cross,
            'trend_line': trend_line
        }

    @staticmethod
    def calculate_zero_line_cross(cci: pd.Series) -> pd.Series:
        """
        计算零线交叉

        参数:
            cci: CCI值序列

        返回:
            零线交叉信号序列
        """
        signals = pd.Series(0, index=cci.index)

        for i in range(1, len(cci)):
            if pd.notna(cci.iloc[i]) and pd.notna(cci.iloc[i-1]):
                if cci.iloc[i] > 0 and cci.iloc[i-1] <= 0:
                    signals.iloc[i] = 1  # 向上穿越零线
                elif cci.iloc[i] < 0 and cci.iloc[i-1] >= 0:
                    signals.iloc[i] = -1  # 向下穿越零线

        return signals

    @staticmethod
    def calculate_trend_line(high: pd.Series,
                             low: pd.Series,
                             close: pd.Series) -> pd.Series:
        """
        计算趋势线

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        返回:
            趋势值序列
        """
        period = 34
        trend_line = pd.Series(0.0, index=close.index)

        for i in range(period-1, len(close)):
            highest_high = max(high.iloc[i-period+1:i+1])
            lowest_low = min(low.iloc[i-period+1:i+1])
            trend_line.iloc[i] = (highest_high + lowest_low) / 2

        return trend_line

    @staticmethod
    def generate_signals(woodies_data: Dict[str, pd.Series]) -> pd.Series:
        """
        生成Woodie's CCI交易信号

        参数:
            woodies_data: Woodie's CCI数据

        返回:
            交易信号序列
        """
        main_cci = woodies_data['main_cci']
        turbo_cci = woodies_data['turbo_cci']
        main_cci_ma = woodies_data['main_cci_ma']
        turbo_cci_ma = woodies_data['turbo_cci_ma']
        trend_line = woodies_data['trend_line']

        signals = pd.Series(0, index=main_cci.index)

        for i in range(len(main_cci)):
            signal = 0

            # Woodie's CCI 规则1: CCI穿越+100线
            if main_cci.iloc[i] > 100 and main_cci.iloc[i-1] <= 100:
                signal = 1
            # Woodie's CCI 规则2: CCI穿越-100线
            elif main_cci.iloc[i] < -100 and main_cci.iloc[i-1] >= -100:
                signal = -1
            # Woodie's CCI 规则3: 零线交叉确认
            elif (main_cci.iloc[i] > 0 and main_cci.iloc[i-1] <= 0 and
                  turbo_cci.iloc[i] > turbo_cci_ma.iloc[i]):
                signal = 1
            # Woodie's CCI 规则4: 趋势线突破
            elif (pd.notna(trend_line.iloc[i]) and main_cci.iloc[i] > 0 and
                  main_cci.iloc[i] > trend_line.iloc[i]):
                signal = 1
            # Woodie's CCI 规则5: CCI与趋势背离
            elif (main_cci.iloc[i] < 0 and turbo_cci.iloc[i] < turbo_cci_ma.iloc[i]):
                signal = -1

            signals.iloc[i] = signal

        return signals

    @staticmethod
    def woodies_zones(woodies_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        Woodie's CCI区域分析

        参数:
            woodies_data: Woodie's CCI数据

        返回:
            区域信号字典
        """
        main_cci = woodies_data['main_cci']

        # 定义Woodie's CCI特定区域
        zones = {
            'extreme_long': pd.Series(0, index=main_cci.index),
            'long': pd.Series(0, index=main_cci.index),
            'neutral': pd.Series(0, index=main_cci.index),
            'short': pd.Series(0, index=main_cci.index),
            'extreme_short': pd.Series(0, index=main_cci.index)
        }

        for i in range(len(main_cci)):
            cci_val = main_cci.iloc[i]
            if pd.notna(cci_val):
                if cci_val > 200:
                    zones['extreme_long'].iloc[i] = 1
                elif cci_val > 100:
                    zones['long'].iloc[i] = 1
                elif cci_val > -100:
                    zones['neutral'].iloc[i] = 1
                elif cci_val > -200:
                    zones['short'].iloc[i] = 1
                else:
                    zones['extreme_short'].iloc[i] = 1

        return zones


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算Woodie's CCI
    woodies_calculator = WoodiesCCI()
    woodies_data = woodies_calculator.calculate(high, low, close)

    # 生成交易信号
    signals = woodies_calculator.generate_signals(woodies_data)

    # 区域分析
    zones = woodies_calculator.woodies_zones(woodies_data)

    print("Woodie's CCI (伍迪商品通道指数) 计算完成！")
    print(f"主CCI: {woodies_data['main_cci'].iloc[-1]:.2f}")
    print(f"Turbo CCI: {woodies_data['turbo_cci'].iloc[-1]:.2f}")
    print(f"信号: {signals.iloc[-1]}")
    print(f"零线交叉: {woodies_data['zero_line_cross'].iloc[-1]}")
    print(f"在极多头区域: {zones['extreme_long'].iloc[-1]}")
    print(f"在多头区域: {zones['long'].iloc[-1]}")
    print(f"在中性区域: {zones['neutral'].iloc[-1]}")