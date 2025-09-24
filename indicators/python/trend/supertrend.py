"""
Supertrend (超级趋势) 指标实现
一个较新的趋势跟踪指标，结合ATR和价格趋势，提供清晰的买卖信号。

这个指标在2020年后变得非常流行，特别适合趋势交易。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class Supertrend:
    """超级趋势指标"""

    def __init__(self):
        self.name = "Supertrend"
        self.category = "trend"

    @staticmethod
    def atr(high: Union[List[float], pd.Series],
            low: Union[List[float], pd.Series],
            close: Union[List[float], pd.Series],
            period: int = 10) -> pd.Series:
        """
        计算平均真实范围

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期

        返回:
            ATR值序列
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        prev_close = close.shift(1)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 使用Wilder平滑计算ATR
        atr = tr.rolling(window=period).mean()
        for i in range(period, len(tr)):
            atr.iloc[i] = (atr.iloc[i-1] * (period - 1) + tr.iloc[i]) / period

        return atr

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series],
                  period: int = 10,
                  multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """
        计算Supertrend指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: ATR周期，默认10
            multiplier: ATR倍数，默认3.0

        返回:
            包含上轨、下轨、Supertrend线和信号的字典
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算ATR
        atr = Supertrend.atr(high, low, close, period)

        # 计算基本轨道
        hl2 = (high + low) / 2
        basic_upperband = hl2 + (multiplier * atr)
        basic_lowerband = hl2 - (multiplier * atr)

        # 计算最终轨道
        final_upperband = pd.Series(0.0, index=close.index)
        final_lowerband = pd.Series(0.0, index=close.index)

        for i in range(len(close)):
            if i == 0:
                final_upperband.iloc[i] = basic_upperband.iloc[i]
                final_lowerband.iloc[i] = basic_lowerband.iloc[i]
            else:
                # 上轨调整逻辑
                if basic_upperband.iloc[i] < final_upperband.iloc[i-1] or close.iloc[i-1] > final_upperband.iloc[i-1]:
                    final_upperband.iloc[i] = basic_upperband.iloc[i]
                else:
                    final_upperband.iloc[i] = final_upperband.iloc[i-1]

                # 下轨调整逻辑
                if basic_lowerband.iloc[i] > final_lowerband.iloc[i-1] or close.iloc[i-1] < final_lowerband.iloc[i-1]:
                    final_lowerband.iloc[i] = basic_lowerband.iloc[i]
                else:
                    final_lowerband.iloc[i] = final_lowerband.iloc[i-1]

                    if final_lowerband.iloc[i] > final_lowerband.iloc[i-1] or close.iloc[i-1] < final_lowerband.iloc[i-1]:
                        final_lowerband.iloc[i] = final_lowerband.iloc[i-1]

        # 计算Supertrend线
        supertrend = pd.Series(0.0, index=close.index)

        for i in range(len(close)):
            if i == 0:
                supertrend.iloc[i] = final_upperband.iloc[i]
            else:
                if supertrend.iloc[i-1] == final_upperband.iloc[i-1]:
                    if close.iloc[i] > final_upperband.iloc[i]:
                        supertrend.iloc[i] = final_lowerband.iloc[i]
                    else:
                        supertrend.iloc[i] = final_upperband.iloc[i]
                else:
                    if close.iloc[i] < final_lowerband.iloc[i]:
                        supertrend.iloc[i] = final_upperband.iloc[i]
                    else:
                        supertrend.iloc[i] = final_lowerband.iloc[i]

        # 计算趋势信号
        trend = pd.Series(1, index=close.index)  # 1=上升趋势, -1=下降趋势

        for i in range(len(close)):
            if supertrend.iloc[i] == final_upperband.iloc[i]:
                trend.iloc[i] = -1  # 下降趋势
            else:
                trend.iloc[i] = 1   # 上升趋势

        return {
            'upperband': final_upperband,
            'lowerband': final_lowerband,
            'supertrend': supertrend,
            'trend': trend,
            'atr': atr
        }

    @staticmethod
    def generate_signals(supertrend_data: Dict[str, pd.Series],
                         close: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        生成Supertrend交易信号

        参数:
            supertrend_data: Supert指标数据
            close: 收盘价序列

        返回:
            包含买卖信号的字典
        """
        if isinstance(close, list):
            close = pd.Series(close)

        supertrend = supertrend_data['supertrend']
        upperband = supertrend_data['upperband']
        lowerband = supertrend_data['lowerband']

        signals = pd.Series(0, index=close.index)
        trend_direction = pd.Series(0, index=close.index)

        for i in range(len(close)):
            if i == 0:
                trend_direction.iloc[i] = 0
                signals.iloc[i] = 0
            else:
                # 确定趋势方向
                if supertrend.iloc[i] == upperband.iloc[i]:
                    trend_direction.iloc[i] = -1  # 下降趋势
                else:
                    trend_direction.iloc[i] = 1   # 上升趋势

                # 生成信号
                if trend_direction.iloc[i] != trend_direction.iloc[i-1]:
                    if trend_direction.iloc[i] == 1:
                        signals.iloc[i] = 1   # 买入信号（趋势反转向上）
                    else:
                        signals.iloc[i] = -1  # 卖出信号（趋势反转向下）
                else:
                    signals.iloc[i] = 0   # 保持信号

        return {
            'signals': signals,
            'trend_direction': trend_direction
        }

    @staticmethod
    def advanced_supertrend(high: Union[List[float], pd.Series],
                            low: Union[List[float], pd.Series],
                            close: Union[List[float], pd.Series],
                            period: int = 10,
                            multiplier: float = 3.0) -> Dict[str, pd.Series]:
        """
        高级Supertrend指标，包含多重时间框架确认

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: ATR周期
            multiplier: ATR倍数

        返回:
            高级Supertrend数据
        """
        # 计算基础Supertrend
        basic_result = Supertrend.calculate(high, low, close, period, multiplier)

        # 计算短周期Supertrend用于确认
        short_result = Supertrend.calculate(high, low, close, period//2, multiplier*0.8)

        # 计算长周期Supertrend用于趋势过滤
        long_result = Supertrend.calculate(high, low, close, period*2, multiplier*1.2)

        # 生成信号
        basic_signals = Supertrend.generate_signals(basic_result, close)
        short_signals = Supertrend.generate_signals(short_result, close)
        long_signals = Supertrend.generate_signals(long_result, close)

        # 多重确认信号
        confirmed_signals = pd.Series(0, index=close.index)

        # 强买入信号：三个时间框架都显示买入
        strong_buy = (basic_signals['signals'] == 1) & (short_signals['signals'] == 1) & (long_signals['trend_direction'] == 1)
        confirmed_signals[strong_buy] = 2

        # 普通买入信号：基础信号 + 至少一个确认
        normal_buy = (basic_signals['signals'] == 1) & ((short_signals['trend_direction'] == 1) | (long_signals['trend_direction'] == 1))
        confirmed_signals[normal_buy & ~strong_buy] = 1

        # 强卖出信号：三个时间框架都显示卖出
        strong_sell = (basic_signals['signals'] == -1) & (short_signals['signals'] == -1) & (long_signals['trend_direction'] == -1)
        confirmed_signals[strong_sell] = -2

        # 普通卖出信号：基础信号 + 至少一个确认
        normal_sell = (basic_signals['signals'] == -1) & ((short_signals['trend_direction'] == -1) | (long_signals['trend_direction'] == -1))
        confirmed_signals[normal_sell & ~strong_sell] = -1

        return {
            'basic_supertrend': basic_result,
            'short_supertrend': short_result,
            'long_supertrend': long_result,
            'confirmed_signals': confirmed_signals,
            'basic_signals': basic_signals['signals'],
            'trend_strength': basic_signals['trend_direction']
        }

    @staticmethod
    def supertrend_with_volatility_filter(high: Union[List[float], pd.Series],
                                         low: Union[List[float], pd.Series],
                                         close: Union[List[float], pd.Series],
                                         period: int = 10,
                                         multiplier: float = 3.0,
                                         volatility_period: int = 20) -> Dict[str, pd.Series]:
        """
        带波动率过滤的Supertrend

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: ATR周期
            multiplier: ATR倍数
            volatility_period: 波动率计算周期

        返回:
            带过滤的Supertrend数据
        """
        # 计算基础Supertrend
        supertrend_result = Supertrend.calculate(high, low, close, period, multiplier)

        # 计算波动率
        returns = close.pct_change()
        volatility = returns.rolling(window=volatility_period).std() * np.sqrt(252)

        # 计算波动率百分位
        volatility_percentile = volatility.rolling(window=volatility_period).rank(pct=True)

        # 生成波动率过滤信号
        volatility_filter = pd.Series(0, index=close.index)
        volatility_filter[volatility_percentile > 0.7] = 1  # 高波动率
        volatility_filter[volatility_percentile < 0.3] = -1  # 低波动率

        # 基础信号
        basic_signals = Supertrend.generate_signals(supertrend_result, close)

        # 波动率过滤后的信号
        filtered_signals = basic_signals['signals'].copy()

        # 在高波动率时，加强信号
        filtered_signals[(basic_signals['signals'] == 1) & (volatility_filter == 1)] = 2  # 强买入
        filtered_signals[(basic_signals['signals'] == -1) & (volatility_filter == 1)] = -2  # 强卖出

        # 在低波动率时，减弱信号
        filtered_signals[(basic_signals['signals'] != 0) & (volatility_filter == -1)] = 0  # 过滤掉低波动率信号

        return {
            'supertrend_data': supertrend_result,
            'volatility': volatility,
            'volatility_filter': volatility_filter,
            'basic_signals': basic_signals['signals'],
            'filtered_signals': filtered_signals
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算Supertrend
    supertrend_calculator = Supertrend()
    supertrend_data = supertrend_calculator.calculate(high, low, close)

    # 生成交易信号
    signals = supertrend_calculator.generate_signals(supertrend_data, close)

    # 高级Supertrend
    advanced_result = supertrend_calculator.advanced_supertrend(high, low, close)

    # 带波动率过滤的Supertrend
    volatility_filtered = supertrend_calculator.supertrend_with_volatility_filter(high, low, close)

    print("Supertrend (超级趋势) 指标计算完成！")
    print(f"Supertrend线: {supertrend_data['supertrend'].iloc[-1]:.2f}")
    print(f"上轨: {supertrend_data['upperband'].iloc[-1]:.2f}")
    print(f"下轨: {supertrend_data['lowerband'].iloc[-1]:.2f}")
    print(f"基本信号: {signals['signals'].iloc[-1]}")
    print(f"趋势方向: {signals['trend_direction'].iloc[-1]}")
    print(f"确认信号: {advanced_result['confirmed_signals'].iloc[-1]}")
    print(f"过滤后信号: {volatility_filtered['filtered_signals'].iloc[-1]}")