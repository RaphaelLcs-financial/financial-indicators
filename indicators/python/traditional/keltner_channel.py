"""
Keltner Channel 凯尔特纳通道实现

Keltner Channel是基于ATR的动态通道指标，比布林带更适合趋势跟踪。
它由中线、上轨和下轨组成，在加密货币交易中特别流行。

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple


class KeltnerChannel:
    """
    Keltner Channel 凯尔特纳通道

    Keltner Channel特点：
    - 基于EMA和ATR构建
    - 动态调整通道宽度
    - 适合趋势跟踪
    - 在趋势市场中表现优秀
    """

    def __init__(self,
                 ema_period: int = 20,
                 atr_period: int = 10,
                 multiplier: float = 2.0,
                 use_ema: bool = True):
        """
        初始化Keltner Channel指标

        Args:
            ema_period: EMA计算周期
            atr_period: ATR计算周期
            multiplier: ATR倍数，控制通道宽度
            use_ema: 是否使用EMA，False则使用SMA
        """
        self.ema_period = ema_period
        self.atr_period = atr_period
        self.multiplier = multiplier
        self.use_ema = use_ema

        # 计算结果存储
        self.middle_band = None
        self.upper_band = None
        self.lower_band = None
        self.atr_values = None
        self.band_width = None
        self.position_in_band = None

    def calculate_ema(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        计算指数移动平均

        Args:
            data: 数据序列
            period: 周期

        Returns:
            EMA值序列
        """
        if len(data) < period:
            return np.full(len(data), np.nan)

        ema = np.full(len(data), np.nan)
        multiplier = 2 / (period + 1)

        # 初始化第一个EMA值
        valid_data = data[:period]
        ema[period - 1] = np.mean(valid_data[~np.isnan(valid_data)])

        # 递归计算EMA
        for i in range(period, len(data)):
            if np.isnan(ema[i-1]):
                ema[i] = np.nan
            else:
                ema[i] = data[i] * multiplier + ema[i-1] * (1 - multiplier)

        return ema

    def calculate_sma(self, data: np.ndarray, period: int) -> np.ndarray:
        """
        计算简单移动平均

        Args:
            data: 数据序列
            period: 周期

        Returns:
            SMA值序列
        """
        if len(data) < period:
            return np.full(len(data), np.nan)

        sma = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            window_data = data[i-period+1:i+1]
            valid_data = window_data[~np.isnan(window_data)]
            if len(valid_data) > 0:
                sma[i] = np.mean(valid_data)

        return sma

    def calculate_atr(self,
                      high: Union[np.ndarray, pd.Series],
                      low: Union[np.ndarray, pd.Series],
                      close: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        计算平均真实范围 (ATR)

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            ATR值序列
        """
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)

        if len(high) < self.atr_period:
            return np.full(len(high), np.nan)

        # 计算真实范围
        tr = np.zeros(len(high))
        tr[0] = high[0] - low[0]

        for i in range(1, len(high)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        # 计算ATR (使用简单移动平均)
        atr = np.full(len(high), np.nan)
        for i in range(self.atr_period - 1, len(high)):
            atr[i] = np.mean(tr[i-self.atr_period+1:i+1])

        return atr

    def calculate(self,
                 high: Union[np.ndarray, pd.Series],
                 low: Union[np.ndarray, pd.Series],
                 close: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        计算Keltner Channel指标

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            包含Keltner Channel和相关指标的字典
        """
        # 转换为numpy数组
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)

        # 数据验证
        if len(high) != len(low) != len(close):
            raise ValueError("所有输入序列必须具有相同的长度")

        max_period = max(self.ema_period, self.atr_period)
        if len(high) < max_period:
            return {}

        # 计算中线
        if self.use_ema:
            middle_band = self.calculate_ema(close, self.ema_period)
        else:
            middle_band = self.calculate_sma(close, self.ema_period)

        # 计算ATR
        atr = self.calculate_atr(high, low, close)

        # 计算上下轨
        upper_band = middle_band + (self.multiplier * atr)
        lower_band = middle_band - (self.multiplier * atr)

        # 计算通道宽度
        band_width = upper_band - lower_band

        # 计算价格在通道中的位置
        position_in_band = np.full(len(close), np.nan)
        for i in range(len(close)):
            if not np.isnan(upper_band[i]) and not np.isnan(lower_band[i]):
                if upper_band[i] != lower_band[i]:
                    position_in_band[i] = (close[i] - lower_band[i]) / (upper_band[i] - lower_band[i])

        # 存储结果
        self.middle_band = middle_band
        self.upper_band = upper_band
        self.lower_band = lower_band
        self.atr_values = atr
        self.band_width = band_width
        self.position_in_band = position_in_band

        return {
            'middle_band': middle_band,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'atr': atr,
            'band_width': band_width,
            'position_in_band': position_in_band
        }

    def get_signals(self,
                   high: Union[np.ndarray, pd.Series],
                   low: Union[np.ndarray, pd.Series],
                   close: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        生成交易信号

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            交易信号字典
        """
        if self.upper_band is None or self.lower_band is None:
            raise ValueError("请先调用calculate方法计算Keltner Channel")

        close = np.asarray(close)

        signals = np.zeros(len(close))
        squeeze_signals = np.zeros(len(close))
        breakout_signals = np.zeros(len(close))

        for i in range(1, len(close)):
            if not np.isnan(self.upper_band[i]) and not np.isnan(self.lower_band[i]):
                # 基本通道信号
                if close[i] > self.upper_band[i]:
                    signals[i] = 1  # 突破上轨
                elif close[i] < self.lower_band[i]:
                    signals[i] = -1  # 突破下轨
                else:
                    signals[i] = 0  # 在通道内

                # 通道挤压信号
                if (self.band_width[i] < np.mean(self.band_width[max(0, i-20):i]) * 0.8 and
                    i > 20):  # 通道宽度显著缩小
                    squeeze_signals[i] = 1  # 通道挤压

                # 突破确认信号
                if squeeze_signals[i-1] == 1:  # 前一周期处于挤压状态
                    if close[i] > self.upper_band[i]:
                        breakout_signals[i] = 1  # 向上突破
                    elif close[i] < self.lower_band[i]:
                        breakout_signals[i] = -1  # 向下突破

        return {
            'signals': signals,
            'squeeze_signals': squeeze_signals,
            'breakout_signals': breakout_signals,
            'position_in_band': self.position_in_band
        }

    def get_trading_recommendations(self,
                                   high: Union[np.ndarray, pd.Series],
                                   low: Union[np.ndarray, pd.Series],
                                   close: Union[np.ndarray, pd.Series]) -> Dict[str, str]:
        """
        获取交易建议

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            交易建议字典
        """
        if self.upper_band is None or self.lower_band is None:
            raise ValueError("请先调用calculate方法计算Keltner Channel")

        close = np.asarray(close)

        # 获取最新数据
        current_price = close[-1]
        current_upper = self.upper_band[-1]
        current_lower = self.lower_band[-1]
        current_middle = self.middle_band[-1]
        current_atr = self.atr_values[-1]
        current_band_width = self.band_width[-1]

        if np.isnan(current_upper) or np.isnan(current_lower):
            action = "等待"
            reason = "Keltner Channel计算尚未完成"
            confidence = 0
        else:
            # 分析价格在通道中的位置
            position = self.position_in_band[-1]

            if np.isnan(position):
                action = "等待"
                reason = "位置计算尚未完成"
                confidence = 0
            else:
                # 生成建议
                if current_price > current_upper:
                    action = "谨慎买入"
                    reason = f"价格({current_price:.2f})突破上轨({current_upper:.2f})，强势但注意回调"
                    confidence = 70

                    # 检查ATR确认
                    if current_atr > np.mean(self.atr_values[-10:]) * 1.2:  # 波动率放大
                        confidence = min(confidence + 10, 90)
                        reason += f"，波动率放大(ATR:{current_atr:.2f})确认"

                elif current_price < current_lower:
                    action = "谨慎卖出"
                    reason = f"价格({current_price:.2f})跌破下轨({current_lower:.2f})，弱势但注意反弹"
                    confidence = 70

                    # 检查ATR确认
                    if current_atr > np.mean(self.atr_values[-10:]) * 1.2:  # 波动率放大
                        confidence = min(confidence + 10, 90)
                        reason += f"，波动率放大(ATR:{current_atr:.2f})确认"

                elif position > 0.8:  # 接近上轨
                    action = "谨慎卖出"
                    reason = f"价格接近上轨({current_upper:.2f})，回调风险"
                    confidence = 65
                elif position < 0.2:  # 接近下轨
                    action = "谨慎买入"
                    reason = f"价格接近下轨({current_lower:.2f})，反弹机会"
                    confidence = 65
                elif 0.4 <= position <= 0.6:  # 在中线附近
                    action = "观望"
                    reason = f"价格在中线({current_middle:.2f})附近，等待突破"
                    confidence = 50
                else:
                    action = "观望"
                    reason = f"价格在通道中部，等待明确信号"
                    confidence = 45

                # 通道挤压分析
                if len(self.band_width) >= 20:
                    avg_band_width = np.mean(self.band_width[-20:])
                    if current_band_width < avg_band_width * 0.7:  # 通道挤压
                        confidence = max(confidence, 60)
                        reason += "，通道挤压中，注意即将突破"

                # 趋势强度分析
                if len(self.middle_band) >= 5:
                    middle_slope = self.middle_band[-1] - self.middle_band[-5]
                    if middle_slope > 0:
                        if position > 0.7:
                            confidence = min(confidence + 5, 95)
                            reason += "，上升趋势强劲"
                    elif middle_slope < 0:
                        if position < 0.3:
                            confidence = min(confidence + 5, 95)
                            reason += "，下降趋势强劲"

        return {
            'action': action,
            'reason': reason,
            'confidence': confidence,
            'current_price': current_price,
            'upper_band': current_upper,
            'middle_band': current_middle,
            'lower_band': current_lower,
            'position_in_band': position if not np.isnan(position) else 0,
            'atr': current_atr,
            'band_width': current_band_width,
            'parameters': {
                'ema_period': self.ema_period,
                'atr_period': self.atr_period,
                'multiplier': self.multiplier,
                'use_ema': self.use_ema
            }
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        获取Keltner Channel统计信息

        Returns:
            统计信息字典
        """
        if self.upper_band is None or self.lower_band is None:
            raise ValueError("请先调用calculate方法计算Keltner Channel")

        # 移除NaN值
        valid_upper = self.upper_band[~np.isnan(self.upper_band)]
        valid_lower = self.lower_band[~np.isnan(self.lower_band)]
        valid_middle = self.middle_band[~np.isnan(self.middle_band)]
        valid_width = self.band_width[~np.isnan(self.band_width)]

        if len(valid_upper) == 0:
            return {}

        # 计算价格在通道中的分布
        valid_position = self.position_in_band[~np.isnan(self.position_in_band)]
        if len(valid_position) > 0:
            upper_zone_freq = np.sum(valid_position > 0.8) / len(valid_position)
            lower_zone_freq = np.sum(valid_position < 0.2) / len(valid_position)
            middle_zone_freq = np.sum((valid_position >= 0.4) & (valid_position <= 0.6)) / len(valid_position)
        else:
            upper_zone_freq = lower_zone_freq = middle_zone_freq = 0

        return {
            'upper_band_mean': np.mean(valid_upper),
            'lower_band_mean': np.mean(valid_lower),
            'middle_band_mean': np.mean(valid_middle),
            'band_width_mean': np.mean(valid_width),
            'band_width_std': np.std(valid_width),
            'upper_zone_frequency': upper_zone_freq,
            'lower_zone_frequency': lower_zone_freq,
            'middle_zone_frequency': middle_zone_freq,
            'calculation_points': len(valid_upper),
            'channel_efficiency': np.mean(valid_width) / np.mean(valid_middle) if np.mean(valid_middle) > 0 else 0
        }