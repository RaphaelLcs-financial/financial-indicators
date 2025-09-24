"""
QQE (量化质量估计) 指标实现

QQE是RSI的增强版本，通过波动率调整机制提供更可靠的信号。
它减少了假信号，提高了交易准确率，在外汇和股票交易中越来越受欢迎。

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple


class QQE:
    """
    QQE (Quantitative Qualitative Estimation) 量化质量估计指标

    QQE基于RSI但通过波动率调整提供更准确的信号：
    - 使用RSI的平滑版本
    - 动态调整信号线阈值
    - 减少假信号和噪音
    - 提供更可靠的趋势确认
    """

    def __init__(self,
                 rsi_period: int = 14,
                 smoothing_period: int = 5,
                 fast_period: int = 2.618,
                 slow_period: int = 4.236,
                 wilder_period: int = 14):
        """
        初始化QQE指标

        Args:
            rsi_period: RSI计算周期
            smoothing_period: 平滑周期
            fast_period: 快速信号线周期 (2.618斐波那契数)
            slow_period: 慢速信号线周期 (4.236斐波那契数)
            wilder_period: Wilder平滑周期
        """
        self.rsi_period = rsi_period
        self.smoothing_period = smoothing_period
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.wilder_period = wilder_period

        # 计算结果存储
        self.rsi_values = None
        self.smoothed_rsi = None
        self.qqe_fast_line = None
        self.qqe_slow_line = None
        self.qqe_long_term_line = None
        self.volatility_bands = None

    def calculate_rsi(self, close: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """
        计算RSI指标

        Args:
            close: 收盘价序列

        Returns:
            RSI值序列
        """
        close = np.asarray(close)

        if len(close) < self.rsi_period + 1:
            return np.full(len(close), np.nan)

        # 计算价格变化
        price_changes = np.diff(close, prepend=close[0])

        # 分离上涨和下跌
        gains = np.where(price_changes > 0, price_changes, 0)
        losses = np.where(price_changes < 0, -price_changes, 0)

        # 计算平均收益和平均损失 (Wilder平滑)
        avg_gains = np.full(len(close), np.nan)
        avg_losses = np.full(len(close), np.nan)

        # 初始化
        avg_gains[self.rsi_period] = np.mean(gains[:self.rsi_period])
        avg_losses[self.rsi_period] = np.mean(losses[:self.rsi_period])

        # 递归计算
        for i in range(self.rsi_period + 1, len(close)):
            avg_gains[i] = (avg_gains[i-1] * (self.wilder_period - 1) + gains[i]) / self.wilder_period
            avg_losses[i] = (avg_losses[i-1] * (self.wilder_period - 1) + losses[i]) / self.wilder_period

        # 计算RSI
        rsi = np.full(len(close), np.nan)
        for i in range(self.rsi_period, len(close)):
            if avg_losses[i] == 0:
                rsi[i] = 100
            else:
                rs = avg_gains[i] / avg_losses[i]
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

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

    def calculate(self, close: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        计算QQE指标

        Args:
            close: 收盘价序列

        Returns:
            包含QQE和相关指标的字典
        """
        close = np.asarray(close)

        if len(close) < self.rsi_period + self.smoothing_period:
            return {}

        # 计算RSI
        rsi = self.calculate_rsi(close)

        # 平滑RSI
        smoothed_rsi = self.calculate_ema(rsi, self.smoothing_period)

        # 计算RSI的变化率
        rsi_changes = np.abs(np.diff(smoothed_rsi, prepend=smoothed_rsi[0]))

        # 计算波动率带
        volatility_bands = self.calculate_ema(rsi_changes, int(self.rsi_period * 0.5)) * 4.236

        # 计算QQE快速线
        qqe_fast_line = np.full(len(close), np.nan)
        for i in range(self.rsi_period + self.smoothing_period, len(close)):
            if smoothed_rsi[i-1] < smoothed_rsi[i]:
                qqe_fast_line[i] = smoothed_rsi[i] - volatility_bands[i]
            else:
                qqe_fast_line[i] = smoothed_rsi[i] + volatility_bands[i]

        # 计算QQE慢速线
        qqe_slow_line = self.calculate_ema(qqe_fast_line, int(self.slow_period))

        # 计算长期QQE线
        qqe_long_term_line = self.calculate_ema(smoothed_rsi, int(self.rsi_period * 1.5))

        # 存储结果
        self.rsi_values = rsi
        self.smoothed_rsi = smoothed_rsi
        self.qqe_fast_line = qqe_fast_line
        self.qqe_slow_line = qqe_slow_line
        self.qqe_long_term_line = qqe_long_term_line
        self.volatility_bands = volatility_bands

        return {
            'rsi': rsi,
            'smoothed_rsi': smoothed_rsi,
            'qqe_fast_line': qqe_fast_line,
            'qqe_slow_line': qqe_slow_line,
            'qqe_long_term_line': qqe_long_term_line,
            'volatility_bands': volatility_bands
        }

    def get_signals(self, close: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        生成交易信号

        Args:
            close: 收盘价序列

        Returns:
            交易信号字典
        """
        if self.qqe_fast_line is None or self.qqe_slow_line is None:
            raise ValueError("请先调用calculate方法计算QQE")

        signals = np.zeros(len(close))
        divergence_signals = np.zeros(len(close))

        # QQE交叉信号
        for i in range(1, len(close)):
            if not np.isnan(self.qqe_fast_line[i]) and not np.isnan(self.qqe_slow_line[i]):
                if not np.isnan(self.qqe_fast_line[i-1]) and not np.isnan(self.qqe_slow_line[i-1]):
                    # 快线上穿慢线 - 买入信号
                    if (self.qqe_fast_line[i-1] <= self.qqe_slow_line[i-1] and
                        self.qqe_fast_line[i] > self.qqe_slow_line[i]):
                        signals[i] = 1

                    # 快线下穿慢线 - 卖出信号
                    elif (self.qqe_fast_line[i-1] >= self.qqe_slow_line[i-1] and
                          self.qqe_fast_line[i] < self.qqe_slow_line[i]):
                        signals[i] = -1

        # 背离信号
        if len(close) > 20:
            # 寻找价格和QQE线的背离
            price_highs = self._find_peaks(close, 20)
            qqe_highs = self._find_peaks(self.qqe_slow_line, 20)
            price_lows = self._find_troughs(close, 20)
            qqe_lows = self._find_troughs(self.qqe_slow_line, 20)

            # 看跌背离：价格创新高，QQE没有创新高
            for i in range(20, len(close)):
                if (price_highs[i] and not qqe_highs[i] and
                    self.qqe_slow_line[i] < 70):  # QQE在超买区以下
                    divergence_signals[i] = -1

                # 看涨背离：价格创新低，QQE没有创新低
                if (price_lows[i] and not qqe_lows[i] and
                    self.qqe_slow_line[i] > 30):  # QQE在超卖区以上
                    divergence_signals[i] = 1

        return {
            'signals': signals,
            'divergence_signals': divergence_signals,
            'qqe_crossing': signals
        }

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

    def get_trading_recommendations(self, close: Union[np.ndarray, pd.Series]) -> Dict[str, str]:
        """
        获取交易建议

        Args:
            close: 收盘价序列

        Returns:
            交易建议字典
        """
        if self.qqe_fast_line is None or self.qqe_slow_line is None:
            raise ValueError("请先调用calculate方法计算QQE")

        close = np.asarray(close)

        # 获取最新数据
        current_price = close[-1]
        current_qqe_fast = self.qqe_fast_line[-1]
        current_qqe_slow = self.qqe_slow_line[-1]
        current_rsi = self.rsi_values[-1] if self.rsi_values is not None else 50

        # 生成建议
        if np.isnan(current_qqe_fast) or np.isnan(current_qqe_slow):
            action = "等待"
            reason = "QQE计算尚未完成"
            confidence = 0
        else:
            # 分析QQE线位置和关系
            if current_qqe_fast > current_qqe_slow:
                if current_rsi > 70:
                    action = "谨慎买入"
                    reason = f"QQE上升趋势但RSI({current_rsi:.1f})超买，注意回调风险"
                    confidence = 60
                elif current_rsi < 30:
                    action = "买入"
                    reason = f"QQE上升趋势且RSI({current_rsi:.1f})超卖，反弹机会"
                    confidence = 85
                else:
                    action = "买入"
                    reason = f"QQE快线({current_qqe_fast:.1f})在慢线({current_qqe_slow:.1f})之上，上升趋势"
                    confidence = 75

                    # 检查距离
                    distance = abs(current_qqe_fast - current_qqe_slow)
                    if distance > 5:
                        confidence = min(confidence + 10, 100)
                        reason += f"，强度大(距离{distance:.1f})"
            else:
                if current_rsi < 30:
                    action = "谨慎卖出"
                    reason = f"QQE下降趋势但RSI({current_rsi:.1f})超卖，注意反弹"
                    confidence = 60
                elif current_rsi > 70:
                    action = "卖出"
                    reason = f"QQE下降趋势且RSI({current_rsi:.1f})超买，回调风险"
                    confidence = 85
                else:
                    action = "卖出"
                    reason = f"QQE快线({current_qqe_fast:.1f})在慢线({current_qqe_slow:.1f})之下，下降趋势"
                    confidence = 75

                    # 检查距离
                    distance = abs(current_qqe_fast - current_qqe_slow)
                    if distance > 5:
                        confidence = min(confidence + 10, 100)
                        reason += f"，强度大(距离{distance:.1f})"

        # 检查是否有交叉信号
        if len(close) > 1:
            prev_qqe_fast = self.qqe_fast_line[-2]
            prev_qqe_slow = self.qqe_slow_line[-2]

            if not np.isnan(prev_qqe_fast) and not np.isnan(prev_qqe_slow):
                # 金叉
                if prev_qqe_fast <= prev_qqe_slow and current_qqe_fast > current_qqe_slow:
                    action = "买入"
                    reason = "QQE金叉，买入信号"
                    confidence = 90

                # 死叉
                elif prev_qqe_fast >= prev_qqe_slow and current_qqe_fast < current_qqe_slow:
                    action = "卖出"
                    reason = "QQE死叉，卖出信号"
                    confidence = 90

        return {
            'action': action,
            'reason': reason,
            'confidence': confidence,
            'current_price': current_price,
            'qqe_fast': current_qqe_fast,
            'qqe_slow': current_qqe_slow,
            'rsi': current_rsi,
            'parameters': {
                'rsi_period': self.rsi_period,
                'smoothing_period': self.smoothing_period,
                'fast_period': self.fast_period,
                'slow_period': self.slow_period
            }
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        获取QQE统计信息

        Returns:
            统计信息字典
        """
        if self.qqe_fast_line is None or self.qqe_slow_line is None:
            raise ValueError("请先调用calculate方法计算QQE")

        # 移除NaN值
        valid_fast = self.qqe_fast_line[~np.isnan(self.qqe_fast_line)]
        valid_slow = self.qqe_slow_line[~np.isnan(self.qqe_slow_line)]
        valid_rsi = self.rsi_values[~np.isnan(self.rsi_values)] if self.rsi_values is not None else []

        if len(valid_fast) == 0:
            return {}

        stats = {
            'qqe_fast_mean': np.mean(valid_fast),
            'qqe_fast_std': np.std(valid_fast),
            'qqe_slow_mean': np.mean(valid_slow),
            'qqe_slow_std': np.std(valid_slow),
            'calculation_points': len(valid_fast)
        }

        if len(valid_rsi) > 0:
            stats['rsi_mean'] = np.mean(valid_rsi)
            stats['rsi_std'] = np.std(valid_rsi)

        return stats