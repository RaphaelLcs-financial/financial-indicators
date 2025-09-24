"""
ADX (Average Directional Index) - 平均趋向指数
趋势强度指标，用于衡量价格趋势的强度，不关注趋势方向。

由Welles Wilder开发，是判断趋势强度的重要工具。
近年来在趋势跟踪策略中重新获得关注。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class ADX:
    """平均趋向指数"""

    def __init__(self):
        self.name = "Average Directional Index"
        self.category = "trend"

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
    def calculate_directional_movement(high: pd.Series, low: pd.Series) -> Dict[str, pd.Series]:
        """
        计算方向性移动

        参数:
            high: 最高价序列
            low: 最低价序列

        返回:
            包含+DM和-DM的字典
        """
        up_move = high.diff()
        down_move = low.diff().abs()

        plus_dm = pd.Series(0.0, index=high.index)
        minus_dm = pd.Series(0.0, index=high.index)

        # +DM: 今日最高价高于昨日最高价，且大于今日最低价与昨日最低价的差
        plus_dm[(up_move > down_move) & (up_move > 0)] = up_move

        # -DM: 今日最低价低于昨日最低价，且大于今日最高价与昨日最高价的差
        minus_dm[(down_move > up_move) & (down_move > 0)] = down_move

        return {
            'plus_dm': plus_dm,
            'minus_dm': minus_dm
        }

    @staticmethod
    def calculate_wilder_ma(data: pd.Series, period: int) -> pd.Series:
        """
        计算Wilder移动平均（平滑移动平均）

        参数:
            data: 数据序列
            period: 周期

        返回:
            Wilder移动平均序列
        """
        wilder_ma = pd.Series(0.0, index=data.index)
        if len(data) >= period:
            wilder_ma.iloc[period-1] = data.iloc[period-1]

            for i in range(period, len(data)):
                wilder_ma.iloc[i] = (wilder_ma.iloc[i-1] * (period-1) + data.iloc[i]) / period

        return wilder_ma

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series],
                  period: int = 14) -> Dict[str, pd.Series]:
        """
        计算ADX指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: 计算周期，默认14

        返回:
            包含ADX、+DI、-DI的字典
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)

        # 计算真实范围
        true_range = ADX.calculate_true_range(high, low, close)

        # 计算方向性移动
        dm = ADX.calculate_directional_movement(high, low)
        plus_dm = dm['plus_dm']
        minus_dm = dm['minus_dm']

        # 计算平滑真实范围
        atr = ADX.calculate_wilder_ma(true_range, period)

        # 计算平滑方向性移动
        smooth_plus_dm = ADX.calculate_wilder_ma(plus_dm, period)
        smooth_minus_dm = ADX.calculate_wilder_ma(minus_dm, period)

        # 计算+DI和-DI
        plus_di = pd.Series(0.0, index=high.index)
        minus_di = pd.Series(0.0, index=high.index)

        for i in range(len(atr)):
            if atr.iloc[i] > 0:
                plus_di.iloc[i] = (smooth_plus_dm.iloc[i] / atr.iloc[i]) * 100
                minus_di.iloc[i] = (smooth_minus_dm.iloc[i] / atr.iloc[i]) * 100

        # 计算DI差值
        di_diff = (plus_di - minus_di).abs()
        di_sum = plus_di + minus_di

        # 计算DX
        dx = pd.Series(0.0, index=high.index)
        for i in range(len(di_sum)):
            if di_sum.iloc[i] > 0:
                dx.iloc[i] = (di_diff.iloc[i] / di_sum.iloc[i]) * 100

        # 计算ADX
        adx = ADX.calculate_wilder_ma(dx, period)

        return {
            'adx': adx,
            'plus_di': plus_di,
            'minus_di': minus_di,
            'dx': dx,
            'atr': atr
        }

    @staticmethod
    def generate_signals(adx_data: Dict[str, pd.Series],
                        adx_threshold: float = 25,
                        di_threshold: float = 25) -> pd.Series:
        """
        生成ADX交易信号

        参数:
            adx_data: ADX指标数据
            adx_threshold: ADX阈值，默认25
            di_threshold: DI阈值，默认25

        返回:
            交易信号序列
        """
        adx = adx_data['adx']
        plus_di = adx_data['plus_di']
        minus_di = adx_data['minus_di']

        signals = pd.Series(0, index=adx.index)

        for i in range(1, len(adx)):
            # 确保有足够的ADX值
            if pd.notna(adx.iloc[i]) and adx.iloc[i] >= adx_threshold:
                # +DI上穿-DI，且ADX强劲 - 买入信号
                if (plus_di.iloc[i] > minus_di.iloc[i] and
                    plus_di.iloc[i-1] <= minus_di.iloc[i-1] and
                    plus_di.iloc[i] > di_threshold):
                    signals.iloc[i] = 1

                # -DI上穿+DI，且ADX强劲 - 卖出信号
                elif (minus_di.iloc[i] > plus_di.iloc[i] and
                      minus_di.iloc[i-1] <= plus_di.iloc[i-1] and
                      minus_di.iloc[i] > di_threshold):
                    signals.iloc[i] = -1

                # 趋势减弱信号
                elif adx.iloc[i] < adx_threshold and adx.iloc[i-1] >= adx_threshold:
                    signals.iloc[i] = 0  # 平仓信号

        return signals

    @staticmethod
    def trend_strength_analysis(adx_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        ADX趋势强度分析

        参数:
            adx_data: ADX指标数据

        返回:
            趋势强度分析结果
        """
        adx = adx_data['adx']
        plus_di = adx_data['plus_di']
        minus_di = adx_data['minus_di']

        # 定义趋势强度等级
        strength = pd.Series('无趋势', index=adx.index)
        trend_direction = pd.Series('盘整', index=adx.index)

        for i in range(len(adx)):
            if pd.notna(adx.iloc[i]):
                # 趋势强度判断
                if adx.iloc[i] < 20:
                    strength.iloc[i] = '无趋势'
                elif adx.iloc[i] < 25:
                    strength.iloc[i] = '弱趋势'
                elif adx.iloc[i] < 40:
                    strength.iloc[i] = '中等趋势'
                elif adx.iloc[i] < 60:
                    strength.iloc[i] = '强趋势'
                else:
                    strength.iloc[i] = '极强趋势'

                # 趋势方向判断
                if plus_di.iloc[i] > minus_di.iloc[i]:
                    trend_direction.iloc[i] = '上升趋势'
                elif minus_di.iloc[i] > plus_di.iloc[i]:
                    trend_direction.iloc[i] = '下降趋势'
                else:
                    trend_direction.iloc[i] = '盘整'

        return {
            'strength': strength,
            'trend_direction': trend_direction
        }

    @staticmethod
    def divergence_detection(prices: pd.Series,
                            adx_data: Dict[str, pd.Series],
                            period: int = 20) -> Dict[str, pd.Series]:
        """
        ADX背离检测

        参数:
            prices: 价格序列
            adx_data: ADX数据
            period: 检测周期

        返回:
            背离信号
        """
        adx = adx_data['adx']
        plus_di = adx_data['plus_di']
        minus_di = adx_data['minus_di']

        # 检测价格和ADX的高点低点
        price_highs = prices.rolling(window=period).max()
        price_lows = prices.rolling(window=period).min()
        adx_highs = adx.rolling(window=period).max()
        adx_lows = adx.rolling(window=period).min()

        # 趋势强度背离
        adx_divergence = pd.Series(0, index=prices.index)

        # 价格创新高但ADX没有创新高 - 趋势减弱
        adx_divergence[(prices == price_highs) & (adx < adx_highs) & (adx > 25)] = -1

        # 价格创新低但ADX没有创新低 - 趋势减弱
        adx_divergence[(prices == price_lows) & (adx > adx_lows) & (adx > 25)] = -1

        return {
            'adx_divergence': adx_divergence
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100

    # 计算ADX
    adx_calculator = ADX()
    adx_data = adx_calculator.calculate(high, low, close)

    # 生成信号
    signals = adx_calculator.generate_signals(adx_data)

    # 趋势强度分析
    strength_analysis = adx_calculator.trend_strength_analysis(adx_data)

    # 背离检测
    divergence = adx_calculator.divergence_detection(close, adx_data)

    print("ADX (平均趋向指数) 计算完成！")
    print(f"ADX: {adx_data['adx'].iloc[-1]:.2f}")
    print(f"+DI: {adx_data['plus_di'].iloc[-1]:.2f}")
    print(f"-DI: {adx_data['minus_di'].iloc[-1]:.2f}")
    print(f"信号: {signals.iloc[-1]}")
    print(f"趋势强度: {strength_analysis['strength'].iloc[-1]}")
    print(f"趋势方向: {strength_analysis['trend_direction'].iloc[-1]}")
    print(f"ADX背离: {divergence['adx_divergence'].iloc[-1]}")