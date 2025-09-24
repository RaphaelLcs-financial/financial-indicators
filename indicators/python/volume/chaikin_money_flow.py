"""
Chaikin Money Flow (蔡金资金流) - Python实现
结合价格和成交量的资金流指标，用于识别买入和卖出压力。

由Marc Chaikin开发，是资金流分析的重要工具。
近年来在量化交易和机构分析中越来越受重视。
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any

class ChaikinMoneyFlow:
    """蔡金资金流指标"""

    def __init__(self):
        self.name = "Chaikin Money Flow"
        self.category = "volume"

    @staticmethod
    def calculate_money_flow_multiplier(high: pd.Series,
                                       low: pd.Series,
                                       close: pd.Series) -> pd.Series:
        """
        计算资金流乘数

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        返回:
            资金流乘数序列
        """
        return ((close - low) - (high - close)) / (high - low)

    @staticmethod
    def calculate_money_flow_volume(multiplier: pd.Series,
                                   volume: pd.Series) -> pd.Series:
        """
        计算资金流成交量

        参数:
            multiplier: 资金流乘数序列
            volume: 成交量序列

        返回:
            资金流成交量序列
        """
        return multiplier * volume

    @staticmethod
    def calculate(high: Union[List[float], pd.Series],
                  low: Union[List[float], pd.Series],
                  close: Union[List[float], pd.Series],
                  volume: Union[List[float], pd.Series],
                  period: int = 20) -> Dict[str, pd.Series]:
        """
        计算Chaikin Money Flow指标

        参数:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
            period: 计算周期，默认20

        返回:
            包含CMF及其组件的字典
        """
        if isinstance(high, list):
            high = pd.Series(high)
        if isinstance(low, list):
            low = pd.Series(low)
        if isinstance(close, list):
            close = pd.Series(close)
        if isinstance(volume, list):
            volume = pd.Series(volume)

        # 计算资金流乘数
        mfm = ChaikinMoneyFlow.calculate_money_flow_multiplier(high, low, close)

        # 计算资金流成交量
        mfv = ChaikinMoneyFlow.calculate_money_flow_volume(mfm, volume)

        # 计算CMF
        cmf = pd.Series(0.0, index=close.index)

        for i in range(period-1, len(close)):
            sum_mfv = mfv.iloc[i-period+1:i+1].sum()
            sum_volume = volume.iloc[i-period+1:i+1].sum()

            if sum_volume > 0:
                cmf.iloc[i] = sum_mfv / sum_volume

        return {
            'cmf': cmf,
            'money_flow_multiplier': mfm,
            'money_flow_volume': mfv
        }

    @staticmethod
    def generate_signals(cmf_data: Dict[str, pd.Series],
                        overbought: float = 0.25,
                        oversold: float = -0.25,
                        centerline: float = 0) -> pd.Series:
        """
        生成Chaikin Money Flow交易信号

        参数:
            cmf_data: CMF数据
            overbought: 超买阈值，默认0.25
            oversold: 超卖阈值，默认-0.25
            centerline: 中心线，默认0

        返回:
            交易信号序列
        """
        cmf = cmf_data['cmf']

        signals = pd.Series(0, index=cmf.index)

        for i in range(1, len(cmf)):
            # 基本CMF信号
            if cmf.iloc[i] > overbought and cmf.iloc[i-1] <= overbought:
                signals.iloc[i] = -1  # 卖出信号
            elif cmf.iloc[i] < oversold and cmf.iloc[i-1] >= oversold:
                signals.iloc[i] = 1   # 买入信号

            # 中心线交叉信号
            elif cmf.iloc[i] > centerline and cmf.iloc[i-1] <= centerline:
                signals.iloc[i] = 1   # 买入信号
            elif cmf.iloc[i] < centerline and cmf.iloc[i-1] >= centerline:
                signals.iloc[i] = -1  # 卖出信号

            # 极值信号
            elif cmf.iloc[i] > 0.5:
                signals.iloc[i] = -2  # 强卖出
            elif cmf.iloc[i] < -0.5:
                signals.iloc[i] = 2   # 强买入

        return signals

    @staticmethod
    def divergence_detection(prices: pd.Series,
                            cmf_data: Dict[str, pd.Series],
                            period: int = 20) -> Dict[str, pd.Series]:
        """
        CMF背离检测

        参数:
            prices: 价格序列
            cmf_data: CMF数据
            period: 检测周期

        返回:
            背离信号
        """
        cmf = cmf_data['cmf']

        # 检测价格和CMF的高点低点
        price_highs = prices.rolling(window=period).max()
        price_lows = prices.rolling(window=period).min()
        cmf_highs = cmf.rolling(window=period).max()
        cmf_lows = cmf.rolling(window=period).min()

        bullish_divergence = pd.Series(0, index=prices.index)
        bearish_divergence = pd.Series(0, index=prices.index)

        # 看涨背离：价格创新低，CMF没有创新低
        bullish_divergence[(prices == price_lows) & (cmf > cmf_lows) & (cmf < 0)] = 1

        # 看跌背离：价格创新高，CMF没有创新高
        bearish_divergence[(prices == price_highs) & (cmf < cmf_highs) & (cmf > 0)] = -1

        return {
            'bullish_divergence': bullish_divergence,
            'bearish_divergence': bearish_divergence
        }

    @staticmethod
    def trend_analysis(cmf_data: Dict[str, pd.Series],
                      period: int = 10) -> Dict[str, pd.Series]:
        """
        CMF趋势分析

        参数:
            cmf_data: CMF数据
            period: 分析周期

        返回:
            趋势分析结果
        """
        cmf = cmf_data['cmf']

        # 计算CMF的移动平均
        cmf_ma = cmf.rolling(window=period).mean()

        # 计算CMF的趋势斜率
        trend_slope = pd.Series(0.0, index=cmf.index)
        for i in range(period-1, len(cmf)):
            x = np.arange(period)
            y = cmf.iloc[i-period+1:i+1].values
            if len(y) > 0:
                slope = np.polyfit(x, y, 1)[0]
                trend_slope.iloc[i] = slope

        # 趋势强度判断
        trend_strength = pd.Series('无趋势', index=cmf.index)
        for i in range(len(trend_slope)):
            if pd.notna(trend_slope.iloc[i]):
                if trend_slope.iloc[i] > 0.01:
                    trend_strength.iloc[i] = '强上升趋势'
                elif trend_slope.iloc[i] > 0.005:
                    trend_strength.iloc[i] = '上升趋势'
                elif trend_slope.iloc[i] > -0.005:
                    trend_strength.iloc[i] = '盘整'
                elif trend_slope.iloc[i] > -0.01:
                    trend_strength.iloc[i] = '下降趋势'
                else:
                    trend_strength.iloc[i] = '强下降趋势'

        return {
            'cmf_ma': cmf_ma,
            'trend_slope': trend_slope,
            'trend_strength': trend_strength
        }

    @staticmethod
    def accumulation_distribution_analysis(cmf_data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        累积/派发分析

        参数:
            cmf_data: CMF数据

        返回:
            累积派发分析结果
        """
        cmf = cmf_data['cmf']

        # 累积/派发信号
        accumulation = pd.Series(0, index=cmf.index)
        distribution = pd.Series(0, index=cmf.index)

        for i in range(len(cmf)):
            if pd.notna(cmf.iloc[i]):
                if cmf.iloc[i] > 0.1:
                    accumulation.iloc[i] = 1
                elif cmf.iloc[i] < -0.1:
                    distribution.iloc[i] = 1

        # 累积/派发比率
        cum_accumulation = accumulation.rolling(window=20).sum()
        cum_distribution = distribution.rolling(window=20).sum()

        ad_ratio = pd.Series(0.0, index=cmf.index)
        total_signals = cum_accumulation + cum_distribution

        for i in range(len(total_signals)):
            if total_signals.iloc[i] > 0:
                ad_ratio.iloc[i] = cum_accumulation.iloc[i] / total_signals.iloc[i]

        return {
            'accumulation': accumulation,
            'distribution': distribution,
            'ad_ratio': ad_ratio
        }

    @staticmethod
    def money_flow_index_analysis(cmf_data: Dict[str, pd.Series],
                                 prices: pd.Series,
                                 period: int = 14) -> Dict[str, pd.Series]:
        """
        资金流指数分析

        参数:
            cmf_data: CMF数据
            prices: 价格序列
            period: 分析周期

        返回:
            资金流指数分析结果
        """
        cmf = cmf_data['cmf']
        mfv = cmf_data['money_flow_volume']

        # 正资金流和负资金流
        positive_mfv = mfv.where(mfv > 0, 0)
        negative_mfv = mfv.where(mfv < 0, 0).abs()

        # 计算正负资金流的周期总和
        positive_mf_sum = positive_mfv.rolling(window=period).sum()
        negative_mf_sum = negative_mfv.rolling(window=period).sum()

        # 计算资金流指数
        mfi = pd.Series(50.0, index=cmf.index)

        for i in range(len(mfi)):
            if negative_mf_sum.iloc[i] > 0:
                money_ratio = positive_mf_sum.iloc[i] / negative_mf_sum.iloc[i]
                mfi.iloc[i] = 100 - (100 / (1 + money_ratio))

        return {
            'money_flow_index': mfi,
            'positive_mf_sum': positive_mf_sum,
            'negative_mf_sum': negative_mf_sum
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    n_periods = 100
    high = np.cumsum(np.random.randn(n_periods)) + 105
    low = np.cumsum(np.random.randn(n_periods)) + 95
    close = np.cumsum(np.random.randn(n_periods)) + 100
    volume = np.random.randint(1000000, 5000000, n_periods)

    # 计算Chaikin Money Flow
    cmf_calculator = ChaikinMoneyFlow()
    cmf_data = cmf_calculator.calculate(high, low, close, volume)

    # 生成信号
    signals = cmf_calculator.generate_signals(cmf_data)

    # 背离检测
    divergence = cmf_calculator.divergence_detection(close, cmf_data)

    # 趋势分析
    trend_analysis = cmf_calculator.trend_analysis(cmf_data)

    # 累积派发分析
    ad_analysis = cmf_calculator.accumulation_distribution_analysis(cmf_data)

    # 资金流指数分析
    mfi_analysis = cmf_calculator.money_flow_index_analysis(cmf_data, close)

    print("Chaikin Money Flow (蔡金资金流) 计算完成！")
    print(f"CMF值: {cmf_data['cmf'].iloc[-1]:.4f}")
    print(f"信号: {signals.iloc[-1]}")
    print(f"看涨背离: {divergence['bullish_divergence'].iloc[-1]}")
    print(f"看跌背离: {divergence['bearish_divergence'].iloc[-1]}")
    print(f"趋势强度: {trend_analysis['trend_strength'].iloc[-1]}")
    print(f"累积派发比率: {ad_analysis['ad_ratio'].iloc[-1]:.4f}")
    print(f"资金流指数: {mfi_analysis['money_flow_index'].iloc[-1]:.2f}")