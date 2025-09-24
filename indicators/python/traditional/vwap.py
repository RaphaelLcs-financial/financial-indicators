"""
VWAP (成交量加权平均价格) 指标实现

VWAP是机构交易者的标准工具，计算特定时间段内的成交量加权平均价格。
它是日内交易的重要参考指标，用于判断价格的相对水平和交易时机。

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple
from datetime import datetime, time


class VWAP:
    """
    VWAP (Volume Weighted Average Price) 成交量加权平均价格指标

    VWAP是衡量平均交易成本的重要指标，帮助交易者判断：
    - 当前价格相对于日内平均成本的位置
    - 价格的相对价值和交易时机
    - 机构资金流向和市场情绪
    """

    def __init__(self,
                 reset_daily: bool = True,
                 session_start: time = time(9, 30),
                 session_end: time = time(16, 0),
                 include_bands: bool = True,
                 band_std_dev: float = 1.0):
        """
        初始化VWAP指标

        Args:
            reset_daily: 是否每日重置VWAP计算
            session_start: 交易时段开始时间
            session_end: 交易时段结束时间
            include_bands: 是否计算标准差带
            band_std_dev: 标准差倍数
        """
        self.reset_daily = reset_daily
        self.session_start = session_start
        self.session_end = session_end
        self.include_bands = include_bands
        self.band_std_dev = band_std_dev

        # 计算结果存储
        self.vwap_values = None
        self.upper_band = None
        self.lower_band = None
        self.typical_prices = None
        self.cumulative_volume = None
        self.cumulative_volume_price = None

    def calculate(self,
                 high: Union[np.ndarray, pd.Series],
                 low: Union[np.ndarray, pd.Series],
                 close: Union[np.ndarray, pd.Series],
                 volume: Union[np.ndarray, pd.Series],
                 timestamps: Optional[Union[np.ndarray, pd.Series]] = None) -> Dict[str, np.ndarray]:
        """
        计算VWAP指标

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
            timestamps: 时间戳序列（可选）

        Returns:
            包含VWAP和相关指标的字典
        """
        # 转换为numpy数组
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)
        volume = np.asarray(volume)

        # 数据验证
        if len(high) != len(low) != len(close) != len(volume):
            raise ValueError("所有输入序列必须具有相同的长度")

        if len(high) == 0:
            return {}

        # 计算典型价格
        typical_price = (high + low + close) / 3.0

        # 处理时间戳
        if timestamps is None:
            timestamps = np.arange(len(high))
        else:
            timestamps = np.asarray(timestamps)

        # 初始化数组
        vwap_values = np.full(len(high), np.nan)
        upper_band = np.full(len(high), np.nan) if self.include_bands else None
        lower_band = np.full(len(high), np.nan) if self.include_bands else None

        # 计算累积成交量
        cumulative_volume = np.zeros(len(high))
        cumulative_volume_price = np.zeros(len(high))

        if self.reset_daily:
            # 每日重置逻辑
            current_date = None
            start_idx = 0

            for i in range(len(high)):
                # 获取当前时间戳的日期
                if hasattr(timestamps[i], 'date'):
                    date = timestamps[i].date()
                else:
                    # 假设是Unix时间戳
                    date = datetime.fromtimestamp(timestamps[i]).date()

                # 如果是新的一天，重置累积值
                if current_date is None or date != current_date:
                    current_date = date
                    start_idx = i

                # 计算从开始日期到当前点的累积值
                if i == start_idx:
                    cumulative_volume[i] = volume[i]
                    cumulative_volume_price[i] = typical_price[i] * volume[i]
                else:
                    cumulative_volume[i] = cumulative_volume[i-1] + volume[i]
                    cumulative_volume_price[i] = cumulative_volume_price[i-1] + typical_price[i] * volume[i]

                # 计算VWAP
                if cumulative_volume[i] > 0:
                    vwap_values[i] = cumulative_volume_price[i] / cumulative_volume[i]

                    # 计算标准差带
                    if self.include_bands:
                        # 计算价格偏差的平方
                        price_deviation_sq = (typical_price[start_idx:i+1] - vwap_values[i]) ** 2
                        # 计算成交量加权的标准差
                        weighted_variance = np.sum(price_deviation_sq * volume[start_idx:i+1]) / cumulative_volume[i]
                        std_dev = np.sqrt(weighted_variance)

                        upper_band[i] = vwap_values[i] + self.band_std_dev * std_dev
                        lower_band[i] = vwap_values[i] - self.band_std_dev * std_dev
        else:
            # 不重置，计算全周期VWAP
            for i in range(len(high)):
                if i == 0:
                    cumulative_volume[i] = volume[i]
                    cumulative_volume_price[i] = typical_price[i] * volume[i]
                else:
                    cumulative_volume[i] = cumulative_volume[i-1] + volume[i]
                    cumulative_volume_price[i] = cumulative_volume_price[i-1] + typical_price[i] * volume[i]

                if cumulative_volume[i] > 0:
                    vwap_values[i] = cumulative_volume_price[i] / cumulative_volume[i]

                    # 计算标准差带
                    if self.include_bands:
                        price_deviation_sq = (typical_price[:i+1] - vwap_values[i]) ** 2
                        weighted_variance = np.sum(price_deviation_sq * volume[:i+1]) / cumulative_volume[i]
                        std_dev = np.sqrt(weighted_variance)

                        upper_band[i] = vwap_values[i] + self.band_std_dev * std_dev
                        lower_band[i] = vwap_values[i] - self.band_std_dev * std_dev

        # 存储结果
        self.vwap_values = vwap_values
        self.upper_band = upper_band
        self.lower_band = lower_band
        self.typical_prices = typical_price
        self.cumulative_volume = cumulative_volume
        self.cumulative_volume_price = cumulative_volume_price

        # 返回结果
        result = {
            'vwap': vwap_values,
            'typical_price': typical_price,
            'cumulative_volume': cumulative_volume,
            'cumulative_volume_price': cumulative_volume_price
        }

        if self.include_bands:
            result['upper_band'] = upper_band
            result['lower_band'] = lower_band

        return result

    def get_signals(self,
                   close: Union[np.ndarray, pd.Series],
                   threshold: float = 0.1) -> Dict[str, np.ndarray]:
        """
        生成交易信号

        Args:
            close: 收盘价序列
            threshold: 信号阈值（百分比）

        Returns:
            交易信号字典
        """
        if self.vwap_values is None:
            raise ValueError("请先调用calculate方法计算VWAP")

        close = np.asarray(close)
        signals = np.zeros(len(close))

        # 价格相对于VWAP的位置
        price_deviation = (close - self.vwap_values) / self.vwap_values

        # 生成信号
        # 1. 价格显著低于VWAP时买入（低估）
        signals[price_deviation < -threshold] = 1

        # 2. 价格显著高于VWAP时卖出（高估）
        signals[price_deviation > threshold] = -1

        # 3. 如果有标准差带，使用带信号
        if self.include_bands and self.upper_band is not None and self.lower_band is not None:
            signals = np.zeros(len(close))

            # 价格突破上轨时卖出
            signals[close > self.upper_band] = -1

            # 价格跌破下轨时买入
            signals[close < self.lower_band] = 1

            # 价格回到VWAP附近时平仓
            vwap_proximity = np.abs(close - self.vwap_values) / self.vwap_values
            signals[vwap_proximity < 0.01] = 0

        return {
            'signals': signals,
            'price_deviation': price_deviation
        }

    def get_trading_recommendations(self,
                                  close: Union[np.ndarray, pd.Series],
                                  volume: Union[np.ndarray, pd.Series]) -> Dict[str, str]:
        """
        获取交易建议

        Args:
            close: 收盘价序列
            volume: 成交量序列

        Returns:
            交易建议字典
        """
        if self.vwap_values is None:
            raise ValueError("请先调用calculate方法计算VWAP")

        close = np.asarray(close)
        volume = np.asarray(volume)

        # 获取最新数据
        current_price = close[-1]
        current_vwap = self.vwap_values[-1]
        current_volume = volume[-1]
        avg_volume = np.mean(volume[-20:])  # 20周期平均成交量

        # 计算价格偏差
        price_deviation = (current_price - current_vwap) / current_vwap

        # 生成建议
        if price_deviation < -0.02:  # 价格低于VWAP 2%以上
            action = "买入"
            reason = f"价格({current_price:.2f})显著低于VWAP({current_vwap:.2f})，存在低估机会"
            confidence = min(abs(price_deviation) * 25, 100)
        elif price_deviation > 0.02:  # 价格高于VWAP 2%以上
            action = "卖出"
            reason = f"价格({current_price:.2f})显著高于VWAP({current_vwap:.2f})，存在高估风险"
            confidence = min(price_deviation * 25, 100)
        else:
            action = "观望"
            reason = f"价格({current_price:.2f})接近VWAP({current_vwap:.2f})，等待更好的机会"
            confidence = 50

        # 成交量确认
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        if volume_ratio > 1.5:  # 成交量放大
            confidence = min(confidence + 10, 100)
            reason += f"，成交量放大({volume_ratio:.1f}x)确认信号"

        return {
            'action': action,
            'reason': reason,
            'confidence': confidence,
            'current_price': current_price,
            'vwap': current_vwap,
            'price_deviation_pct': price_deviation * 100,
            'volume_ratio': volume_ratio
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        获取VWAP统计信息

        Returns:
            统计信息字典
        """
        if self.vwap_values is None:
            raise ValueError("请先调用calculate方法计算VWAP")

        # 移除NaN值
        valid_vwap = self.vwap_values[~np.isnan(self.vwap_values)]
        valid_volume = self.cumulative_volume[~np.isnan(self.cumulative_volume)]

        if len(valid_vwap) == 0:
            return {}

        return {
            'vwap_mean': np.mean(valid_vwap),
            'vwap_std': np.std(valid_vwap),
            'vwap_min': np.min(valid_vwap),
            'vwap_max': np.max(valid_vwap),
            'total_volume': valid_volume[-1] if len(valid_volume) > 0 else 0,
            'calculation_points': len(valid_vwap)
        }