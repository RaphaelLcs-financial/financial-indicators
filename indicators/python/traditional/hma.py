"""
Hull Moving Average (HMA) 船形移动平均线实现

HMA是一种零滞后移动平均线，比传统MA更平滑，响应速度更快。
它通过加权方式减少滞后，同时保持平滑性，特别适合高频交易。

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional


class HullMovingAverage:
    """
    Hull Moving Average (HMA) 船形移动平均线

    HMA通过以下方式消除滞后：
    - 使用加权移动平均减少滞后
    - 平方根周期优化响应速度
    - 保持曲线平滑性的同时提高灵敏度
    """

    def __init__(self, period: int = 20):
        """
        初始化HMA指标

        Args:
            period: 计算周期
        """
        self.period = period
        self.hma_period = int(np.sqrt(period))

        # 计算结果存储
        self.hma_values = None
        self.wma_half = None
        self.wma_full = None
        self.raw_wma = None

    def calculate_wma(self, data: Union[np.ndarray, pd.Series], period: int) -> np.ndarray:
        """
        计算加权移动平均

        Args:
            data: 数据序列
            period: 计算周期

        Returns:
            WMA值序列
        """
        data = np.asarray(data)

        if len(data) < period:
            return np.full(len(data), np.nan)

        wma = np.full(len(data), np.nan)

        # 计算权重
        weights = np.arange(1, period + 1)
        weight_sum = np.sum(weights)

        # 计算WMA
        for i in range(period - 1, len(data)):
            window_data = data[i-period+1:i+1]
            if not np.any(np.isnan(window_data)):
                wma[i] = np.sum(window_data * weights) / weight_sum

        return wma

    def calculate(self, data: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        计算HMA指标

        Args:
            data: 价格序列

        Returns:
            包含HMA和相关指标的字典
        """
        data = np.asarray(data)

        if len(data) < self.period:
            return {}

        # 计算原始WMA
        raw_wma = self.calculate_wma(data, self.period)

        # 计算半周期WMA
        half_period = max(1, self.period // 2)
        wma_half = self.calculate_wma(data, half_period)

        # 计算HMA原始值
        hma_raw = np.full(len(data), np.nan)
        for i in range(self.period - 1, len(data)):
            if not np.isnan(raw_wma[i]) and not np.isnan(wma_half[i]):
                hma_raw[i] = 2 * wma_half[i] - raw_wma[i]

        # 计算最终HMA
        hma_values = self.calculate_wma(hma_raw, self.hma_period)

        # 存储结果
        self.hma_values = hma_values
        self.wma_half = wma_half
        self.wma_full = raw_wma
        self.raw_wma = hma_raw

        return {
            'hma': hma_values,
            'wma_half': wma_half,
            'wma_full': raw_wma,
            'raw_hma': hma_raw
        }

    def get_signals(self,
                   data: Union[np.ndarray, pd.Series],
                   fast_period: int = 12,
                   slow_period: int = 26) -> Dict[str, np.ndarray]:
        """
        生成交易信号

        Args:
            data: 价格序列
            fast_period: 快速HMA周期
            slow_period: 慢速HMA周期

        Returns:
            交易信号字典
        """
        if self.hma_values is None:
            raise ValueError("请先调用calculate方法计算HMA")

        # 计算快速和慢速HMA
        hma_fast = HullMovingAverage(period=fast_period)
        hma_slow = HullMovingAverage(period=slow_period)

        fast_result = hma_fast.calculate(data)
        slow_result = hma_slow.calculate(data)

        fast_hma = fast_result['hma']
        slow_hma = slow_result['hma']

        signals = np.zeros(len(data))

        # 交叉信号
        for i in range(1, len(data)):
            if not np.isnan(fast_hma[i]) and not np.isnan(slow_hma[i]):
                if not np.isnan(fast_hma[i-1]) and not np.isnan(slow_hma[i-1]):
                    # 金叉
                    if fast_hma[i-1] <= slow_hma[i-1] and fast_hma[i] > slow_hma[i]:
                        signals[i] = 1

                    # 死叉
                    elif fast_hma[i-1] >= slow_hma[i-1] and fast_hma[i] < slow_hma[i]:
                        signals[i] = -1

        return {
            'signals': signals,
            'fast_hma': fast_hma,
            'slow_hma': slow_hma
        }

    def get_trading_recommendations(self,
                                   data: Union[np.ndarray, pd.Series],
                                   lookback: int = 10) -> Dict[str, str]:
        """
        获取交易建议

        Args:
            data: 价格序列
            lookback: 回看周期

        Returns:
            交易建议字典
        """
        if self.hma_values is None:
            raise ValueError("请先调用calculate方法计算HMA")

        data = np.asarray(data)

        # 获取最新数据
        current_price = data[-1]
        current_hma = self.hma_values[-1]

        if np.isnan(current_hma):
            action = "等待"
            reason = "HMA计算尚未完成"
            confidence = 0
        else:
            # 计算价格与HMA的关系
            price_hma_diff = (current_price - current_hma) / current_hma

            # 计算HMA的斜率
            if len(data) >= 2:
                hma_slope = current_hma - self.hma_values[-2]
                hma_slope_pct = (hma_slope / current_hma) * 100 if current_hma != 0 else 0
            else:
                hma_slope_pct = 0

            # 生成建议
            if price_hma_diff > 0.01 and hma_slope_pct > 0:
                action = "买入"
                reason = f"价格({current_price:.2f})在HMA({current_hma:.2f})之上且HMA上升，趋势向上"
                confidence = min(75 + abs(price_hma_diff) * 1000, 95)
            elif price_hma_diff < -0.01 and hma_slope_pct < 0:
                action = "卖出"
                reason = f"价格({current_price:.2f})在HMA({current_hma:.2f})之下且HMA下降，趋势向下"
                confidence = min(75 + abs(price_hma_diff) * 1000, 95)
            elif abs(price_hma_diff) < 0.005:
                action = "观望"
                reason = f"价格({current_price:.2f})接近HMA({current_hma:.2f})，等待突破"
                confidence = 50
            else:
                action = "观望"
                reason = f"价格和HMA关系不明显，等待更明确信号"
                confidence = 40

            # 检查HMA的平滑性
            if len(data) >= lookback:
                recent_hma = self.hma_values[-lookback:]
                valid_hma = recent_hma[~np.isnan(recent_hma)]
                if len(valid_hma) > 1:
                    hma_volatility = np.std(np.diff(valid_hma))
                    if hma_volatility < 0.001:  # HMA过于平缓
                        confidence = max(confidence - 20, 20)
                        reason += "，HMA过于平缓"

        return {
            'action': action,
            'reason': reason,
            'confidence': confidence,
            'current_price': current_price,
            'current_hma': current_hma,
            'price_hma_diff_pct': price_hma_diff * 100,
            'hma_slope_pct': hma_slope_pct if 'hma_slope_pct' in locals() else 0,
            'parameters': {
                'period': self.period,
                'hma_period': self.hma_period
            }
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        获取HMA统计信息

        Returns:
            统计信息字典
        """
        if self.hma_values is None:
            raise ValueError("请先调用calculate方法计算HMA")

        # 移除NaN值
        valid_hma = self.hma_values[~np.isnan(self.hma_values)]

        if len(valid_hma) == 0:
            return {}

        # 计算滞后统计
        if len(valid_hma) > 1:
            hma_changes = np.diff(valid_hma)
            lag_indicator = np.mean(np.abs(hma_changes))  # 变化幅度反映滞后程度
        else:
            lag_indicator = 0

        return {
            'hma_mean': np.mean(valid_hma),
            'hma_std': np.std(valid_hma),
            'hma_min': np.min(valid_hma),
            'hma_max': np.max(valid_hma),
            'lag_indicator': lag_indicator,
            'calculation_points': len(valid_hma),
            'efficiency_ratio': np.std(valid_hma) / np.mean(valid_hma) if np.mean(valid_hma) > 0 else 0
        }