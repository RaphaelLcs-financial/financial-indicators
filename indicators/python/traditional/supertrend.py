"""
Supertrend (超级趋势) 指标实现

Supertrend是2024-2025年最热门的趋势跟踪指标，基于ATR计算动态支撑和阻力位。
它能够有效识别趋势方向，并提供清晰的买卖信号，特别适合趋势市场。

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple


class Supertrend:
    """
    Supertrend (超级趋势) 指标

    Supertrend通过ATR计算动态的支撑和阻力位，当价格突破这些水平时产生趋势信号。
    特点：
    - 趋势跟踪能力强
    - 信号清晰明确
    - 参数可调整适应不同市场
    - 适合趋势交易策略
    """

    def __init__(self,
                 period: int = 10,
                 multiplier: float = 3.0,
                 use_heikin_ashi: bool = False):
        """
        初始化Supertrend指标

        Args:
            period: ATR计算周期
            multiplier: ATR倍数，影响带宽
            use_heikin_ashi: 是否使用Heikin Ashi蜡烛图
        """
        self.period = period
        self.multiplier = multiplier
        self.use_heikin_ashi = use_heikin_ashi

        # 计算结果存储
        self.supertrend = None
        self.final_upper_band = None
        self.final_lower_band = None
        self.atr_values = None
        self.basic_upper_band = None
        self.basic_lower_band = None

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

        if len(high) < self.period:
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
        for i in range(self.period - 1, len(high)):
            atr[i] = np.mean(tr[i-self.period+1:i+1])

        return atr

    def calculate_heikin_ashi(self,
                            open_: Union[np.ndarray, pd.Series],
                            high: Union[np.ndarray, pd.Series],
                            low: Union[np.ndarray, pd.Series],
                            close: Union[np.ndarray, pd.Series]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        计算Heikin Ashi蜡烛图

        Args:
            open_: 开盘价序列
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            HA开盘价、最高价、最低价、收盘价
        """
        open_ = np.asarray(open_)
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)

        ha_open = np.zeros(len(close))
        ha_high = np.zeros(len(close))
        ha_low = np.zeros(len(close))
        ha_close = np.zeros(len(close))

        ha_close[0] = (open_[0] + high[0] + low[0] + close[0]) / 4
        ha_open[0] = (open_[0] + close[0]) / 2
        ha_high[0] = high[0]
        ha_low[0] = low[0]

        for i in range(1, len(close)):
            ha_close[i] = (open_[i] + high[i] + low[i] + close[i]) / 4
            ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
            ha_high[i] = max(high[i], ha_open[i], ha_close[i])
            ha_low[i] = min(low[i], ha_open[i], ha_close[i])

        return ha_open, ha_high, ha_low, ha_close

    def calculate(self,
                 high: Union[np.ndarray, pd.Series],
                 low: Union[np.ndarray, pd.Series],
                 close: Union[np.ndarray, pd.Series],
                 open_: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, np.ndarray]:
        """
        计算Supertrend指标

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            open_: 开盘价序列（可选，Heikin Ashi时需要）

        Returns:
            包含Supertrend和相关指标的字典
        """
        # 转换为numpy数组
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)

        # 数据验证
        if len(high) != len(low) != len(close):
            raise ValueError("所有输入序列必须具有相同的长度")

        if len(high) < self.period:
            return {}

        # 使用Heikin Ashi或原始数据
        if self.use_heikin_ashi:
            if open_ is None:
                raise ValueError("使用Heikin Ashi时需要提供开盘价数据")
            ha_open, ha_high, ha_low, ha_close = self.calculate_heikin_ashi(open_, high, low, close)
            high_calc, low_calc, close_calc = ha_high, ha_low, ha_close
        else:
            high_calc, low_calc, close_calc = high, low, close

        # 计算ATR
        atr = self.calculate_atr(high_calc, low_calc, close_calc)

        # 计算基本带
        hl2 = (high_calc + low_calc) / 2.0
        basic_upper_band = hl2 + (self.multiplier * atr)
        basic_lower_band = hl2 - (self.multiplier * atr)

        # 初始化最终带
        final_upper_band = np.full(len(high), np.nan)
        final_lower_band = np.full(len(high), np.nan)
        supertrend = np.full(len(high), np.nan)

        # 计算最终带
        for i in range(self.period, len(high)):
            if i == self.period:
                final_upper_band[i] = basic_upper_band[i]
                final_lower_band[i] = basic_lower_band[i]
            else:
                # 最终上带：如果之前的上带大于基本上带或收盘价大于之前的上带，使用基本上带，否则使用之前的上带
                if basic_upper_band[i] < final_upper_band[i-1] or close_calc[i-1] > final_upper_band[i-1]:
                    final_upper_band[i] = basic_upper_band[i]
                else:
                    final_upper_band[i] = final_upper_band[i-1]

                # 最终下带：如果之前的下带小于基本下带或收盘价小于之前的下带，使用基本下带，否则使用之前的下带
                if basic_lower_band[i] > final_lower_band[i-1] or close_calc[i-1] < final_lower_band[i-1]:
                    final_lower_band[i] = basic_lower_band[i]
                else:
                    final_lower_band[i] = final_lower_band[i-1]

            # 计算Supertrend
            if i == self.period:
                supertrend[i] = final_upper_band[i]
            else:
                if supertrend[i-1] == final_upper_band[i-1]:
                    if close_calc[i] <= final_upper_band[i]:
                        supertrend[i] = final_upper_band[i]
                    else:
                        supertrend[i] = final_lower_band[i]
                else:
                    if close_calc[i] >= final_lower_band[i]:
                        supertrend[i] = final_lower_band[i]
                    else:
                        supertrend[i] = final_upper_band[i]

        # 存储结果
        self.supertrend = supertrend
        self.final_upper_band = final_upper_band
        self.final_lower_band = final_lower_band
        self.atr_values = atr
        self.basic_upper_band = basic_upper_band
        self.basic_lower_band = basic_lower_band

        return {
            'supertrend': supertrend,
            'final_upper_band': final_upper_band,
            'final_lower_band': final_lower_band,
            'atr': atr,
            'basic_upper_band': basic_upper_band,
            'basic_lower_band': basic_lower_band
        }

    def get_signals(self,
                   close: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        生成交易信号

        Args:
            close: 收盘价序列

        Returns:
            交易信号字典
        """
        if self.supertrend is None:
            raise ValueError("请先调用calculate方法计算Supertrend")

        close = np.asarray(close)
        signals = np.zeros(len(close))

        # 当收盘价上穿Supertrend时买入
        for i in range(1, len(close)):
            if not np.isnan(self.supertrend[i]) and not np.isnan(self.supertrend[i-1]):
                if close[i-1] <= self.supertrend[i-1] and close[i] > self.supertrend[i]:
                    signals[i] = 1  # 买入信号
                elif close[i-1] >= self.supertrend[i-1] and close[i] < self.supertrend[i]:
                    signals[i] = -1  # 卖出信号

        return {
            'signals': signals,
            'trend_direction': np.where(close > self.supertrend, 1, -1)
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
        if self.supertrend is None:
            raise ValueError("请先调用calculate方法计算Supertrend")

        close = np.asarray(close)
        high = np.asarray(high)
        low = np.asarray(low)

        # 获取最新数据
        current_price = close[-1]
        current_supertrend = self.supertrend[-1]
        current_atr = self.atr_values[-1] if self.atr_values is not None else 0

        # 判断趋势方向
        if np.isnan(current_supertrend):
            action = "等待"
            reason = "数据不足，等待Supertrend计算完成"
            confidence = 0
        else:
            if current_price > current_supertrend:
                action = "持有多头"
                reason = f"价格({current_price:.2f})在Supertrend({current_supertrend:.2f})之上，处于上升趋势"
                confidence = 75

                # 检查趋势强度
                distance = (current_price - current_supertrend) / current_atr if current_atr > 0 else 0
                if distance > 2:
                    confidence = min(confidence + 15, 100)
                    reason += f"，趋势强度强(距离{distance:.1f}ATR)"
                elif distance < 0.5:
                    confidence = max(confidence - 15, 25)
                    reason += f"，趋势强度弱(距离{distance:.1f}ATR)"

            else:
                action = "持有空头"
                reason = f"价格({current_price:.2f})在Supertrend({current_supertrend:.2f})之下，处于下降趋势"
                confidence = 75

                # 检查趋势强度
                distance = (current_supertrend - current_price) / current_atr if current_atr > 0 else 0
                if distance > 2:
                    confidence = min(confidence + 15, 100)
                    reason += f"，趋势强度强(距离{distance:.1f}ATR)"
                elif distance < 0.5:
                    confidence = max(confidence - 15, 25)
                    reason += f"，趋势强度弱(距离{distance:.1f}ATR)"

        # 检查是否有信号变化
        if len(close) > 1 and not np.isnan(self.supertrend[-2]):
            prev_price = close[-2]
            prev_supertrend = self.supertrend[-2]

            # 信号变化检测
            if (prev_price <= prev_supertrend and current_price > current_supertrend):
                action = "买入"
                reason = "价格上穿Supertrend，趋势反转信号"
                confidence = 90
            elif (prev_price >= prev_supertrend and current_price < current_supertrend):
                action = "卖出"
                reason = "价格下穿Supertrend，趋势反转信号"
                confidence = 90

        return {
            'action': action,
            'reason': reason,
            'confidence': confidence,
            'current_price': current_price,
            'supertrend': current_supertrend,
            'atr': current_atr,
            'parameters': {
                'period': self.period,
                'multiplier': self.multiplier,
                'heikin_ashi': self.use_heikin_ashi
            }
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        获取Supertrend统计信息

        Returns:
            统计信息字典
        """
        if self.supertrend is None or self.atr_values is None:
            raise ValueError("请先调用calculate方法计算Supertrend")

        # 移除NaN值
        valid_supertrend = self.supertrend[~np.isnan(self.supertrend)]
        valid_atr = self.atr_values[~np.isnan(self.atr_values)]

        if len(valid_supertrend) == 0:
            return {}

        return {
            'supertrend_mean': np.mean(valid_supertrend),
            'supertrend_std': np.std(valid_supertrend),
            'supertrend_min': np.min(valid_supertrend),
            'supertrend_max': np.max(valid_supertrend),
            'atr_mean': np.mean(valid_atr),
            'atr_std': np.std(valid_atr),
            'calculation_points': len(valid_supertrend),
            'efficiency_ratio': np.mean(valid_atr) / np.mean(valid_supertrend) if np.mean(valid_supertrend) > 0 else 0
        }