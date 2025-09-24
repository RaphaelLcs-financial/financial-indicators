"""
Ichimoku Cloud 一目均衡表实现

Ichimoku Cloud是日本开发的综合技术分析系统，提供完整的市场视图。
它通过多个指标组合识别趋势、支撑阻力、入场时机和止损位置。

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple


class IchimokuCloud:
    """
    Ichimoku Cloud 一目均衡表

    Ichimoku Cloud包含5个主要组成部分：
    - Tenkan-sen (转换线): 短期趋势指标
    - Kijun-sen (基准线): 中期趋势指标
    - Senkou Span A & B (先行带A&B): 形成云层
    - Chikou Span (滞后线): 确认趋势强度
    """

    def __init__(self,
                 tenkan_period: int = 9,
                 kijun_period: int = 26,
                 senkou_span_b_period: int = 52,
                 chikou_shift: int = 26,
                 displacement: int = 26):
        """
        初始化Ichimoku Cloud指标

        Args:
            tenkan_period: Tenkan-sen周期 (通常9)
            kijun_period: Kijun-sen周期 (通常26)
            senkou_span_b_period: Senkou Span B周期 (通常52)
            chikou_shift: Chikou Span滞后周期 (通常26)
            displacement: 先行带位移 (通常26)
        """
        self.tenkan_period = tenkan_period
        self.kijun_period = kijun_period
        self.senkou_span_b_period = senkou_span_b_period
        self.chikou_shift = chikou_shift
        self.displacement = displacement

        # 计算结果存储
        self.tenkan_sen = None
        self.kijun_sen = None
        self.senkou_span_a = None
        self.senkou_span_b = None
        self.chikou_span = None
        self.cloud_top = None
        self.cloud_bottom = None
        self.cloud_color = None

    def calculate(self,
                 high: Union[np.ndarray, pd.Series],
                 low: Union[np.ndarray, pd.Series],
                 close: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        计算Ichimoku Cloud指标

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列

        Returns:
            包含Ichimoku Cloud各个组件的字典
        """
        # 转换为numpy数组
        high = np.asarray(high)
        low = np.asarray(low)
        close = np.asarray(close)

        # 数据验证
        if len(high) != len(low) != len(close):
            raise ValueError("所有输入序列必须具有相同的长度")

        max_period = max(self.tenkan_period, self.kijun_period, self.senkou_span_b_period)
        if len(high) < max_period:
            return {}

        # 计算Tenkan-sen (转换线)
        tenkan_sen = np.full(len(high), np.nan)
        for i in range(self.tenkan_period - 1, len(high)):
            period_high = np.max(high[i-self.tenkan_period+1:i+1])
            period_low = np.min(low[i-self.tenkan_period+1:i+1])
            tenkan_sen[i] = (period_high + period_low) / 2

        # 计算Kijun-sen (基准线)
        kijun_sen = np.full(len(high), np.nan)
        for i in range(self.kijun_period - 1, len(high)):
            period_high = np.max(high[i-self.kijun_period+1:i+1])
            period_low = np.min(low[i-self.kijun_period+1:i+1])
            kijun_sen[i] = (period_high + period_low) / 2

        # 计算Senkou Span A (先行带A)
        senkou_span_a = np.full(len(high), np.nan)
        for i in range(max_period - 1, len(high)):
            if not np.isnan(tenkan_sen[i]) and not np.isnan(kijun_sen[i]):
                senkou_span_a[i] = (tenkan_sen[i] + kijun_sen[i]) / 2

        # 计算Senkou Span B (先行带B)
        senkou_span_b = np.full(len(high), np.nan)
        for i in range(self.senkou_span_b_period - 1, len(high)):
            period_high = np.max(high[i-self.senkou_span_b_period+1:i+1])
            period_low = np.min(low[i-self.senkou_span_b_period+1:i+1])
            senkou_span_b[i] = (period_high + period_low) / 2

        # 计算Chikou Span (滞后线)
        chikou_span = np.full(len(high), np.nan)
        for i in range(self.chikou_shift, len(high)):
            chikou_span[i] = close[i - self.chikou_shift]

        # 计算云层（向前位移）
        cloud_top = np.full(len(high), np.nan)
        cloud_bottom = np.full(len(high), np.nan)
        cloud_color = np.full(len(high), np.nan)  # 1为绿色云，-1为红色云

        for i in range(self.displacement, len(high)):
            senkou_a_idx = i - self.displacement
            senkou_b_idx = i - self.displacement

            if (senkou_a_idx >= 0 and senkou_b_idx >= 0 and
                not np.isnan(senkou_span_a[senkou_a_idx]) and
                not np.isnan(senkou_span_b[senkou_b_idx])):

                cloud_top[i] = max(senkou_span_a[senkou_a_idx], senkou_span_b[senkou_b_idx])
                cloud_bottom[i] = min(senkou_span_a[senkou_a_idx], senkou_span_b[senkou_b_idx])

                # 确定云层颜色
                if senkou_span_a[senkou_a_idx] >= senkou_span_b[senkou_b_idx]:
                    cloud_color[i] = 1  # 绿色云（看涨）
                else:
                    cloud_color[i] = -1  # 红色云（看跌）

        # 存储结果
        self.tenkan_sen = tenkan_sen
        self.kijun_sen = kijun_sen
        self.senkou_span_a = senkou_span_a
        self.senkou_span_b = senkou_span_b
        self.chikou_span = chikou_span
        self.cloud_top = cloud_top
        self.cloud_bottom = cloud_bottom
        self.cloud_color = cloud_color

        return {
            'tenkan_sen': tenkan_sen,
            'kijun_sen': kijun_sen,
            'senkou_span_a': senkou_span_a,
            'senkou_span_b': senkou_span_b,
            'chikou_span': chikou_span,
            'cloud_top': cloud_top,
            'cloud_bottom': cloud_bottom,
            'cloud_color': cloud_color
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
        if self.tenkan_sen is None or self.kijun_sen is None:
            raise ValueError("请先调用calculate方法计算Ichimoku Cloud")

        close = np.asarray(close)

        signals = np.zeros(len(close))
        tk_cross_signals = np.zeros(len(close))
        cloud_signals = np.zeros(len(close))
        chikou_signals = np.zeros(len(close))

        for i in range(max(self.tenkan_period, self.kijun_period), len(close)):
            # Tenkan-Kijun交叉信号
            if (i > 0 and not np.isnan(self.tenkan_sen[i-1]) and not np.isnan(self.kijun_sen[i-1]) and
                not np.isnan(self.tenkan_sen[i]) and not np.isnan(self.kijun_sen[i])):

                # 金叉（看涨信号）
                if (self.tenkan_sen[i-1] <= self.kijun_sen[i-1] and
                    self.tenkan_sen[i] > self.kijun_sen[i]):
                    tk_cross_signals[i] = 1

                # 死叉（看跌信号）
                elif (self.tenkan_sen[i-1] >= self.kijun_sen[i-1] and
                      self.tenkan_sen[i] < self.kijun_sen[i]):
                    tk_cross_signals[i] = -1

            # 价格与云层关系信号
            if not np.isnan(self.cloud_top[i]) and not np.isnan(self.cloud_bottom[i]):
                if close[i] > self.cloud_top[i]:
                    cloud_signals[i] = 1  # 价格在云层之上，强势
                elif close[i] < self.cloud_bottom[i]:
                    cloud_signals[i] = -1  # 价格在云层之下，弱势
                else:
                    cloud_signals[i] = 0  # 价格在云层之中，中性

            # Chikou Span信号
            if (i >= self.chikou_shift and not np.isnan(self.chikou_span[i]) and
                not np.isnan(close[i - self.chikou_shift])):

                chikou_price = self.chikou_span[i]
                historical_price = close[i - self.chikou_shift]

                # Chikou Span在价格之上
                if chikou_price > historical_price:
                    chikou_signals[i] = 1
                # Chikou Span在价格之下
                elif chikou_price < historical_price:
                    chikou_signals[i] = -1

            # 综合信号生成
            signal_strength = 0

            # TK交叉权重最大
            signal_strength += tk_cross_signals[i] * 3

            # 云层信号权重中等
            signal_strength += cloud_signals[i] * 2

            # Chikou信号权重较小
            signal_strength += chikou_signals[i] * 1

            # 确定最终信号
            if signal_strength >= 3:
                signals[i] = 1  # 强烈买入
            elif signal_strength >= 1:
                signals[i] = 0.5  # 温和买入
            elif signal_strength <= -3:
                signals[i] = -1  # 强烈卖出
            elif signal_strength <= -1:
                signals[i] = -0.5  # 温和卖出

        return {
            'signals': signals,
            'tk_cross_signals': tk_cross_signals,
            'cloud_signals': cloud_signals,
            'chikou_signals': chikou_signals
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
        if self.tenkan_sen is None or self.kijun_sen is None:
            raise ValueError("请先调用calculate方法计算Ichimoku Cloud")

        close = np.asarray(close)

        # 获取最新数据
        current_price = close[-1]
        current_tenkan = self.tenkan_sen[-1]
        current_kijun = self.kijun_sen[-1]
        current_cloud_top = self.cloud_top[-1]
        current_cloud_bottom = self.cloud_bottom[-1]
        current_cloud_color = self.cloud_color[-1]

        if np.isnan(current_tenkan) or np.isnan(current_kijun):
            action = "等待"
            reason = "Ichimoku Cloud计算尚未完成"
            confidence = 0
        else:
            # 分析各个组件的状态
            signals = []

            # 1. TK交叉分析
            if current_tenkan > current_kijun:
                signals.append(("Tenkan在Kijun之上", 30, "短期趋势向上"))
            else:
                signals.append(("Tenkan在Kijun之下", 30, "短期趋势向下"))

            # 2. 价格与云层关系分析
            if not np.isnan(current_cloud_top) and not np.isnan(current_cloud_bottom):
                if current_price > current_cloud_top:
                    signals.append(("价格在云层之上", 40, "强势上涨趋势"))
                elif current_price < current_cloud_bottom:
                    signals.append(("价格在云层之下", 40, "弱势下跌趋势"))
                else:
                    signals.append(("价格在云层之中", 20, "中性震荡"))

                # 3. 云层颜色分析
                if not np.isnan(current_cloud_color):
                    if current_cloud_color > 0:
                        signals.append(("云层为绿色", 20, "中期看涨"))
                    else:
                        signals.append(("云层为红色", 20, "中期看跌"))

            # 4. 计算综合信号强度
            bullish_strength = sum(score for _, score, _ in signals if "向上" in _[2] or "上涨" in _[2] or "看涨" in _[2])
            bearish_strength = sum(score for _, score, _ in signals if "向下" in _[2] or "下跌" in _[2] or "看跌" in _[2])

            # 生成最终建议
            if bullish_strength > bearish_strength + 20:
                action = "买入"
                reason = "Ichimoku多个指标显示看涨信号：" + "，".join([desc for desc, _, _ in signals if "向上" in desc or "上涨" in desc or "看涨" in desc])
                confidence = min(bullish_strength, 95)
            elif bearish_strength > bullish_strength + 20:
                action = "卖出"
                reason = "Ichimoku多个指标显示看跌信号：" + "，".join([desc for desc, _, _ in signals if "向下" in desc or "下跌" in desc or "看跌" in desc])
                confidence = min(bearish_strength, 95)
            else:
                action = "观望"
                reason = "Ichimoku信号 mixed，等待更明确的方向"
                confidence = 50

            # 检查关键位置
            if not np.isnan(current_cloud_top) and not np.isnan(current_cloud_bottom):
                cloud_thickness = current_cloud_top - current_cloud_bottom
                if cloud_thickness < (current_price * 0.01):  # 云层很薄
                    confidence = max(confidence - 10, 20)
                    reason += "，但云层较薄，趋势可能不稳定"

        return {
            'action': action,
            'reason': reason,
            'confidence': confidence,
            'current_price': current_price,
            'tenkan_sen': current_tenkan,
            'kijun_sen': current_kijun,
            'cloud_top': current_cloud_top if not np.isnan(current_cloud_top) else None,
            'cloud_bottom': current_cloud_bottom if not np.isnan(current_cloud_bottom) else None,
            'cloud_color': '绿色' if not np.isnan(current_cloud_color) and current_cloud_color > 0 else '红色',
            'signals_analysis': signals,
            'parameters': {
                'tenkan_period': self.tenkan_period,
                'kijun_period': self.kijun_period,
                'senkou_span_b_period': self.senkou_span_b_period,
                'chikou_shift': self.chikou_shift,
                'displacement': self.displacement
            }
        }

    def get_statistics(self) -> Dict[str, float]:
        """
        获取Ichimoku Cloud统计信息

        Returns:
            统计信息字典
        """
        if self.tenkan_sen is None or self.kijun_sen is None:
            raise ValueError("请先调用calculate方法计算Ichimoku Cloud")

        # 移除NaN值进行统计
        valid_tenkan = self.tenkan_sen[~np.isnan(self.tenkan_sen)]
        valid_kijun = self.kijun_sen[~np.isnan(self.kijun_sen)]
        valid_cloud_top = self.cloud_top[~np.isnan(self.cloud_top)]
        valid_cloud_bottom = self.cloud_bottom[~np.isnan(self.cloud_bottom)]

        stats = {}

        if len(valid_tenkan) > 0:
            stats['tenkan_mean'] = np.mean(valid_tenkan)
            stats['tenkan_std'] = np.std(valid_tenkan)

        if len(valid_kijun) > 0:
            stats['kijun_mean'] = np.mean(valid_kijun)
            stats['kijun_std'] = np.std(valid_kijun)

        if len(valid_cloud_top) > 0 and len(valid_cloud_bottom) > 0:
            avg_cloud_thickness = np.mean(valid_cloud_top - valid_cloud_bottom)
            stats['avg_cloud_thickness'] = avg_cloud_thickness
            stats['cloud_thickness_std'] = np.std(valid_cloud_top - valid_cloud_bottom)

            # 云层颜色统计
            valid_cloud_color = self.cloud_color[~np.isnan(self.cloud_color)]
            if len(valid_cloud_color) > 0:
                green_cloud_ratio = np.sum(valid_cloud_color > 0) / len(valid_cloud_color)
                stats['green_cloud_ratio'] = green_cloud_ratio
                stats['red_cloud_ratio'] = 1 - green_cloud_ratio

        stats['calculation_points'] = len(valid_tenkan)

        return stats