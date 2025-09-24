"""
Money Flow Index (MFI) 资金流量指标实现

MFI是"带成交量的RSI"，结合价格和成交量的动量指标。
它用于识别超买超卖区域和背离信号，是验证趋势强度的有效工具。

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional


class MoneyFlowIndex:
    """
    Money Flow Index (MFI) 资金流量指标

    MFI通过以下方式分析资金流向：
    - 计算典型价格和资金流量
    - 分析正负资金流量的比率
    - 识别超买超卖区域
    - 发现价格与MFI的背离
    """

    def __init__(self, period: int = 14):
        """
        初始化MFI指标

        Args:
            period: 计算周期
        """
        self.period = period

        # 计算结果存储
        self.mfi_values = None
        self.typical_prices = None
        self.money_flow = None
        self.positive_money_flow = None
        self.negative_money_flow = None
        self.money_flow_ratio = None

    def calculate(self,
                 high: Union[np.ndarray, pd.Series],
                 low: Union[np.ndarray, pd.Series],
                 close: Union[np.ndarray, pd.Series],
                 volume: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        计算MFI指标

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列

        Returns:
            包含MFI和相关指标的字典
        """
        # 转换为numpy数组
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        volume = np.asarray(volume)

        # 数据验证
        if len(high) != len(low) != len(close) != len(volume):
            raise ValueError("所有输入序列必须具有相同的长度")

        if len(high) < self.period:
            return {}

        # 计算典型价格
        typical_prices = (high + low + close) / 3.0

        # 计算资金流量
        money_flow = typical_prices * volume

        # 初始化数组
        positive_money_flow = np.zeros(len(high))
        negative_money_flow = np.zeros(len(high))
        money_flow_ratio = np.full(len(high), np.nan)
        mfi_values = np.full(len(high), np.nan)

        # 计算正负资金流量
        for i in range(1, len(high)):
            if typical_prices[i] > typical_prices[i-1]:
                positive_money_flow[i] = money_flow[i]
                negative_money_flow[i] = 0
            elif typical_prices[i] < typical_prices[i-1]:
                positive_money_flow[i] = 0
                negative_money_flow[i] = money_flow[i]
            else:
                positive_money_flow[i] = 0
                negative_money_flow[i] = 0

        # 计算MFI
        for i in range(self.period, len(high)):
            # 计算正负资金流量的14周期总和
            pos_sum = np.sum(positive_money_flow[i-self.period+1:i+1])
            neg_sum = np.sum(negative_money_flow[i-self.period+1:i+1])

            if neg_sum == 0:
                money_flow_ratio[i] = float('inf')
                mfi_values[i] = 100
            else:
                money_flow_ratio[i] = pos_sum / neg_sum
                mfi_values[i] = 100 - (100 / (1 + money_flow_ratio[i]))

        # 存储结果
        self.mfi_values = mfi_values
        self.typical_prices = typical_prices
        self.money_flow = money_flow
        self.positive_money_flow = positive_money_flow
        self.negative_money_flow = negative_money_flow
        self.money_flow_ratio = money_flow_ratio

        return {
            'mfi': mfi_values,
            'typical_price': typical_prices,
            'money_flow': money_flow,
            'positive_money_flow': positive_money_flow,
            'negative_money_flow': negative_money_flow,
            'money_flow_ratio': money_flow_ratio
        }

    def get_signals(self, high: Union[np.ndarray, pd.Series], low: Union[np.ndarray, pd.Series],
                   close: Union[np.ndarray, pd.Series], volume: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        生成交易信号

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列

        Returns:
            交易信号字典
        """
        if self.mfi_values is None:
            raise ValueError("请先调用calculate方法计算MFI")

        signals = np.zeros(len(close))
        divergence_signals = np.zeros(len(close))

        # 超买超卖信号
        for i in range(self.period, len(close)):
            if not np.isnan(self.mfi_values[i]):
                # 超卖信号
                if self.mfi_values[i] <= 20:
                    signals[i] = 1

                # 超买信号
                elif self.mfi_values[i] >= 80:
                    signals[i] = -1

                # MFI回到中性区域时平仓
                elif 40 <= self.mfi_values[i] <= 60:
                    signals[i] = 0

        # 背离信号
        divergence_signals = self._calculate_divergence(close, self.mfi_values)

        return {
            'signals': signals,
            'divergence_signals': divergence_signals,
            'overbought_oversold': signals
        }

    def _calculate_divergence(self, price: np.ndarray, indicator: np.ndarray) -> np.ndarray:
        """
        计算背离信号

        Args:
            price: 价格序列
            indicator: 指标序列

        Returns:
            背离信号数组
        """
        divergence_signals = np.zeros(len(price))

        if len(price) < 20:
            return divergence_signals

        # 寻找价格和指标的波峰波谷
        price_highs = self._find_peaks(price, 10)
        indicator_highs = self._find_peaks(indicator, 10)
        price_lows = self._find_troughs(price, 10)
        indicator_lows = self._find_troughs(indicator, 10)

        for i in range(10, len(price) - 10):
            # 看跌背离：价格创新高，指标没有创新高
            if (price_highs[i] and not indicator_highs[i] and
                indicator[i] < 80):  # MFI不在超买区
                divergence_signals[i] = -1

            # 看涨背离：价格创新低，指标没有创新低
            if (price_lows[i] and not indicator_lows[i] and
                indicator[i] > 20):  # MFI不在超卖区
                divergence_signals[i] = 1

        return divergence_signals

    def _find_peaks(self, data: np.ndarray, window: int) -> np.ndarray:
        """寻找波峰"""
        peaks = np.zeros(len(data), dtype=bool)
        for i in range(window, len(data) - window):
            if not np.isnan(data[i]):
                is_peak = True
                for j in range(i - window, i + window + 1):
                    if j != i and not np.isnan(data[j]) and data[j] >= data[i]:
                        is_peak = False
                        break
                peaks[i] = is_peak
        return peaks

    def _find_troughs(self, data: np.ndarray, window: int) -> np.ndarray:
        """寻找波谷"""
        troughs = np.zeros(len(data), dtype=bool)
        for i in range(window, len(data) - window):
            if not np.isnan(data[i]):
                is_trough = True
                for j in range(i - window, i + window + 1):
                    if j != i and not np.isnan(data[j]) and data[j] <= data[i]:
                        is_trough = False
                        break
                troughs[i] = is_trough
        return troughs

    def get_trading_recommendations(self,
                                   high: Union[np.ndarray, pd.Series],
                                   low: Union[np.ndarray, pd.Series],
                                   close: Union[np.ndarray, pd.Series],
                                   volume: Union[np.ndarray, pd.Series]) -> Dict[str, str]:
        """
        获取交易建议

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列

        Returns:
            交易建议字典
        """
        if self.mfi_values is None:
            raise ValueError("请先调用calculate方法计算MFI")

        close = np.asarray(close)
        volume = np.asarray(volume)

        # 获取最新数据
        current_price = close[-1]
        current_mfi = self.mfi_values[-1]
        current_volume = volume[-1]
        avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else current_volume

        if np.isnan(current_mfi):
            action = "等待"
            reason = "MFI计算尚未完成"
            confidence = 0
        else:
            # 分析MFI水平
            if current_mfi <= 20:
                action = "买入"
                reason = f"MFI({current_mfi:.1f})处于超卖区(<20)，反弹机会"
                confidence = 80

                # 成交量确认
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                if volume_ratio > 1.2:  # 成交量放大
                    confidence = min(confidence + 10, 95)
                    reason += f"，成交量放大({volume_ratio:.1f}x)确认"

            elif current_mfi >= 80:
                action = "卖出"
                reason = f"MFI({current_mfi:.1f})处于超买区(>80)，回调风险"
                confidence = 80

                # 成交量确认
                volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
                if volume_ratio > 1.2:  # 成交量放大
                    confidence = min(confidence + 10, 95)
                    reason += f"，成交量放大({volume_ratio:.1f}x)确认"

            elif current_mfi <= 30:
                action = "谨慎买入"
                reason = f"MFI({current_mfi:.1f})接近超卖区，观察反弹信号"
                confidence = 60
            elif current_mfi >= 70:
                action = "谨慎卖出"
                reason = f"MFI({current_mfi:.1f})接近超买区，注意回调"
                confidence = 60
            else:
                action = "观望"
                reason = f"MFI({current_mfi:.1f})处于中性区域(30-70)，等待明确信号"
                confidence = 40

            # 检查背离
            divergence = self._calculate_divergence(close, self.mfi_values)
            if divergence[-1] == 1:
                action = "买入"
                reason = "MFI看涨背离，买入信号"
                confidence = 85
            elif divergence[-1] == -1:
                action = "卖出"
                reason = "MFI看跌背离，卖出信号"
                confidence = 85

            # MFI方向分析
            if len(self.mfi_values) >= 3:
                mfi_trend = (self.mfi_values[-1] - self.mfi_values[-3]) / 3
                if abs(mfi_trend) > 2:  # MFI快速变化
                    if mfi_trend > 0:
                        confidence = min(confidence + 5, 100)
                        reason += f"，MFI快速上升({mfi_trend:.1f})"
                    else:
                        confidence = min(confidence + 5, 100)
                        reason += f"，MFI快速下降({mfi_trend:.1f})"

        return {
            'action': action,
            'reason': reason,
            'confidence': confidence,
            'current_price': current_price,
            'current_mfi': current_mfi,
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'parameters': {
                'period': self.period
            }
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        获取MFI统计信息

        Returns:
            统计信息字典
        """
        if self.mfi_values is None:
            raise ValueError("请先调用calculate方法计算MFI")

        # 移除NaN值
        valid_mfi = self.mfi_values[~np.isnan(self.mfi_values)]
        valid_flow = self.money_flow[~np.isnan(self.money_flow)]

        if len(valid_mfi) == 0:
            return {}

        # 计算超买超卖频率
        overbought_count = np.sum(valid_mfi >= 80)
        oversold_count = np.sum(valid_mfi <= 20)
        total_count = len(valid_mfi)

        return {
            'mfi_mean': np.mean(valid_mfi),
            'mfi_std': np.std(valid_mfi),
            'mfi_min': np.min(valid_mfi),
            'mfi_max': np.max(valid_mfi),
            'money_flow_mean': np.mean(valid_flow),
            'money_flow_std': np.std(valid_flow),
            'overbought_frequency': overbought_count / total_count if total_count > 0 else 0,
            'oversold_frequency': oversold_count / total_count if total_count > 0 else 0,
            'calculation_points': len(valid_mfi)
        }