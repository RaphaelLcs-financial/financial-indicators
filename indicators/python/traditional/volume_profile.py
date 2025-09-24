"""
Volume Profile 成交量分布指标实现

Volume Profile是机构交易者的重要分析工具，通过分析特定价格水平的成交量分布，
识别关键的支撑阻力位和机构活动区域。这是2024-2025年最受欢迎的专业指标之一。

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, Dict, Optional, Tuple, List
from collections import defaultdict


class VolumeProfile:
    """
    Volume Profile 成交量分布指标

    Volume Profile通过以下方式分析市场结构：
    - 识别关键价格水平的成交量集中度
    - 发现机构活动的价值区域(VA)
    - 确定高成交量节点(VPOC)和低成交量区域
    - 提供动态支撑阻力位参考
    """

    def __init__(self,
                 bins: int = 100,
                 volume_threshold: float = 0.7,
                 min_touches: int = 2,
                 use_tick_volume: bool = False):
        """
        初始化Volume Profile指标

        Args:
            bins: 价格分箱数量
            volume_threshold: 价值区域体积阈值(0-1)
            min_touches: 最小触碰次数用于确定重要水平
            use_tick_volume: 是否使用tick volume
        """
        self.bins = bins
        self.volume_threshold = volume_threshold
        self.min_touches = min_touches
        self.use_tick_volume = use_tick_volume

        # 计算结果存储
        self.volume_profile = None
        self.price_levels = None
        self.value_area_high = None
        self.value_area_low = None
        self.vpoc = None  # Volume Point of Control
        self.support_levels = None
        self.resistance_levels = None
        self.high_volume_nodes = None
        self.low_volume_nodes = None

    def calculate(self,
                 high: Union[np.ndarray, pd.Series],
                 low: Union[np.ndarray, pd.Series],
                 close: Union[np.ndarray, pd.Series],
                 volume: Union[np.ndarray, pd.Series]) -> Dict[str, Union[np.ndarray, dict]]:
        """
        计算Volume Profile指标

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列

        Returns:
            包含Volume Profile和相关指标的字典
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

        # 确定价格范围
        price_min = np.min(low)
        price_max = np.max(high)

        # 创建价格水平
        price_levels = np.linspace(price_min, price_max, self.bins + 1)
        price_centers = (price_levels[:-1] + price_levels[1:]) / 2

        # 初始化成交量分布
        volume_distribution = np.zeros(self.bins)

        # 计算每个价格水平的成交量
        for i in range(len(high)):
            # 计算该K线覆盖的价格区间
            candle_low_idx = np.searchsorted(price_levels, low[i])
            candle_high_idx = np.searchsorted(price_levels, high[i])

            # 确保索引在有效范围内
            candle_low_idx = max(0, min(candle_low_idx, self.bins - 1))
            candle_high_idx = max(0, min(candle_high_idx, self.bins - 1))

            # 分配成交量到覆盖的价格水平
            if candle_low_idx == candle_high_idx:
                volume_distribution[candle_low_idx] += volume[i]
            else:
                # K线跨越多个价格水平，均匀分配成交量
                levels_covered = candle_high_idx - candle_low_idx + 1
                volume_per_level = volume[i] / levels_covered
                for j in range(candle_low_idx, candle_high_idx + 1):
                    if 0 <= j < self.bins:
                        volume_distribution[j] += volume_per_level

        # 找到Volume Point of Control (VPOC)
        vpoc_idx = np.argmax(volume_distribution)
        vpoc_price = price_centers[vpoc_idx]
        vpoc_volume = volume_distribution[vpoc_idx]

        # 计算价值区域(Value Area)
        total_volume = np.sum(volume_distribution)
        target_volume = total_volume * self.volume_threshold

        value_area_high = vpoc_price
        value_area_low = vpoc_price
        accumulated_volume = vpoc_volume

        # 向上扩展价值区域
        for i in range(vpoc_idx + 1, self.bins):
            if accumulated_volume >= target_volume:
                break
            accumulated_volume += volume_distribution[i]
            value_area_high = price_centers[i]

        # 向下扩展价值区域
        for i in range(vpoc_idx - 1, -1, -1):
            if accumulated_volume >= target_volume:
                break
            accumulated_volume += volume_distribution[i]
            value_area_low = price_centers[i]

        # 识别高成交量节点(重要支撑阻力)
        volume_mean = np.mean(volume_distribution)
        volume_std = np.std(volume_distribution)
        high_volume_threshold = volume_mean + volume_std

        high_volume_nodes = []
        for i in range(self.bins):
            if volume_distribution[i] >= high_volume_threshold:
                high_volume_nodes.append({
                    'price': price_centers[i],
                    'volume': volume_distribution[i],
                    'strength': volume_distribution[i] / volume_mean
                })

        # 识别低成交量区域(价格可能快速通过)
        low_volume_threshold = volume_mean * 0.3
        low_volume_nodes = []
        for i in range(self.bins):
            if volume_distribution[i] <= low_volume_threshold:
                low_volume_nodes.append({
                    'price': price_centers[i],
                    'volume': volume_distribution[i],
                    'gap_ratio': (volume_mean - volume_distribution[i]) / volume_mean
                })

        # 确定关键支撑阻力位
        current_price = close[-1]
        support_levels = []
        resistance_levels = []

        # 在当前价格下方寻找支撑
        for i in range(len(price_centers)):
            if price_centers[i] < current_price:
                if volume_distribution[i] >= volume_mean * 0.8:
                    support_levels.append({
                        'price': price_centers[i],
                        'volume': volume_distribution[i],
                        'strength': volume_distribution[i] / volume_mean,
                        'distance': (current_price - price_centers[i]) / current_price
                    })

        # 在当前价格上方寻找阻力
        for i in range(len(price_centers)):
            if price_centers[i] > current_price:
                if volume_distribution[i] >= volume_mean * 0.8:
                    resistance_levels.append({
                        'price': price_centers[i],
                        'volume': volume_distribution[i],
                        'strength': volume_distribution[i] / volume_mean,
                        'distance': (price_centers[i] - current_price) / current_price
                    })

        # 排序支撑阻力位
        support_levels.sort(key=lambda x: x['distance'])
        resistance_levels.sort(key=lambda x: x['distance'])

        # 存储结果
        self.volume_profile = volume_distribution
        self.price_levels = price_centers
        self.value_area_high = value_area_high
        self.value_area_low = value_area_low
        self.vpoc = {'price': vpoc_price, 'volume': vpoc_volume}
        self.support_levels = support_levels[:5]  # 取最近的5个支撑位
        self.resistance_levels = resistance_levels[:5]  # 取最近的5个阻力位
        self.high_volume_nodes = high_volume_nodes
        self.low_volume_nodes = low_volume_nodes

        return {
            'volume_profile': volume_distribution,
            'price_levels': price_centers,
            'value_area': {'high': value_area_high, 'low': value_area_low},
            'vpoc': {'price': vpoc_price, 'volume': vpoc_volume},
            'support_levels': support_levels[:5],
            'resistance_levels': resistance_levels[:5],
            'high_volume_nodes': high_volume_nodes,
            'low_volume_nodes': low_volume_nodes
        }

    def get_signals(self,
                   close: Union[np.ndarray, pd.Series],
                   high: Union[np.ndarray, pd.Series],
                   low: Union[np.ndarray, pd.Series]) -> Dict[str, np.ndarray]:
        """
        生成交易信号

        Args:
            close: 收盘价序列
            high: 最高价序列
            low: 最低价序列

        Returns:
            交易信号字典
        """
        if self.volume_profile is None:
            raise ValueError("请先调用calculate方法计算Volume Profile")

        close = np.asarray(close)
        high = np.asarray(high)
        low = np.asarray(low)

        signals = np.zeros(len(close))
        value_area_signals = np.zeros(len(close))
        vpoc_signals = np.zeros(len(close))

        for i in range(len(close)):
            if i < len(self.volume_profile):
                current_price = close[i]

                # 价值区域信号
                if (self.value_area_low <= current_price <= self.value_area_high):
                    value_area_signals[i] = 0  # 在价值区域内，中性
                elif current_price > self.value_area_high:
                    value_area_signals[i] = 1  # 在价值区域之上，强势
                else:
                    value_area_signals[i] = -1  # 在价值区域之下，弱势

                # VPOC信号
                vpoc_price = self.vpoc['price']
                if abs(current_price - vpoc_price) / vpoc_price < 0.01:  # 价格接近VPOC
                    vpoc_signals[i] = 1

                # 综合信号
                if value_area_signals[i] == -1 and vpoc_signals[i] == 1:
                    signals[i] = 1  # 买入信号：在价值区域下方但接近VPOC
                elif value_area_signals[i] == 1 and self.resistance_levels:
                    # 检查是否接近阻力位
                    nearest_resistance = self.resistance_levels[0]['price'] if self.resistance_levels else float('inf')
                    if abs(current_price - nearest_resistance) / nearest_resistance < 0.02:
                        signals[i] = -1  # 卖出信号：在价值区域上方且接近阻力

        return {
            'signals': signals,
            'value_area_signals': value_area_signals,
            'vpoc_signals': vpoc_signals
        }

    def get_trading_recommendations(self,
                                   close: Union[np.ndarray, pd.Series],
                                   high: Union[np.ndarray, pd.Series],
                                   low: Union[np.ndarray, pd.Series]) -> Dict[str, str]:
        """
        获取交易建议

        Args:
            close: 收盘价序列
            high: 最高价序列
            low: 最低价序列

        Returns:
            交易建议字典
        """
        if self.volume_profile is None:
            raise ValueError("请先调用calculate方法计算Volume Profile")

        close = np.asarray(close)
        current_price = close[-1]

        # 分析价格位置
        in_value_area = self.value_area_low <= current_price <= self.value_area_high
        above_value_area = current_price > self.value_area_high
        below_value_area = current_price < self.value_area_low

        # 计算与VPOC的距离
        vpoc_distance = abs(current_price - self.vpoc['price']) / self.vpoc['price']

        # 生成建议
        if below_value_area:
            action = "买入"
            reason = f"价格({current_price:.2f})在价值区域({self.value_area_low:.2f}-{self.value_area_high:.2f})下方，存在低估机会"
            confidence = 75

            # VPOC确认
            if vpoc_distance < 0.02:
                confidence = min(confidence + 15, 95)
                reason += f"，接近VPOC({self.vpoc['price']:.2f})确认支撑"

            # 支撑位确认
            if self.support_levels:
                nearest_support = self.support_levels[0]
                if vpoc_distance < 0.01:
                    confidence = min(confidence + 10, 100)
                    reason += f"，强支撑位确认"

        elif above_value_area:
            action = "谨慎卖出"
            reason = f"价格({current_price:.2f})在价值区域上方，存在回调风险"
            confidence = 70

            # 阻力位确认
            if self.resistance_levels:
                nearest_resistance = self.resistance_levels[0]
                resistance_distance = abs(current_price - nearest_resistance['price']) / nearest_resistance['price']
                if resistance_distance < 0.02:
                    confidence = min(confidence + 15, 95)
                    reason += f"，接近阻力位({nearest_resistance['price']:.2f})"

        elif in_value_area:
            action = "观望"
            reason = f"价格({current_price:.2f})在价值区域内，等待突破信号"
            confidence = 50

            # 检查在价值区域中的位置
            va_range = self.value_area_high - self.value_area_low
            position_in_va = (current_price - self.value_area_low) / va_range

            if position_in_va > 0.8:  # 接近价值区域顶部
                confidence = 65
                reason += "，接近价值区域顶部"
            elif position_in_va < 0.2:  # 接近价值区域底部
                confidence = 65
                reason += "，接近价值区域底部"
        else:
            action = "观望"
            reason = "等待更明确的市场信号"
            confidence = 40

        return {
            'action': action,
            'reason': reason,
            'confidence': confidence,
            'current_price': current_price,
            'value_area': {
                'high': self.value_area_high,
                'low': self.value_area_low,
                'range': self.value_area_high - self.value_area_low
            },
            'vpoc': self.vpoc,
            'nearest_support': self.support_levels[0] if self.support_levels else None,
            'nearest_resistance': self.resistance_levels[0] if self.resistance_levels else None,
            'parameters': {
                'bins': self.bins,
                'volume_threshold': self.volume_threshold,
                'min_touches': self.min_touches
            }
        }

    def get_statistics(self) -> Dict[str, Union[float, dict]]:
        """
        获取Volume Profile统计信息

        Returns:
            统计信息字典
        """
        if self.volume_profile is None:
            raise ValueError("请先调用calculate方法计算Volume Profile")

        total_volume = np.sum(self.volume_profile)
        avg_volume = np.mean(self.volume_profile)
        volume_std = np.std(self.volume_profile)

        # 计算价值区域统计
        va_volume = np.sum([self.volume_profile[i] for i in range(len(self.volume_profile))
                           if self.value_area_low <= self.price_levels[i] <= self.value_area_high])
        va_concentration = va_volume / total_volume if total_volume > 0 else 0

        # 计算成交量集中度
        volume_concentration = {
            'top_20_percent': np.sum(np.sort(self.volume_profile)[-int(len(self.volume_profile) * 0.2):]) / total_volume,
            'top_10_percent': np.sum(np.sort(self.volume_profile)[-int(len(self.volume_profile) * 0.1):]) / total_volume,
            'vpoc_concentration': self.vpoc['volume'] / total_volume
        }

        return {
            'total_volume': total_volume,
            'average_volume_per_level': avg_volume,
            'volume_std': volume_std,
            'value_area_concentration': va_concentration,
            'value_area_range': self.value_area_high - self.value_area_low,
            'volume_concentration': volume_concentration,
            'num_high_volume_nodes': len(self.high_volume_nodes),
            'num_low_volume_nodes': len(self.low_volume_nodes),
            'num_support_levels': len(self.support_levels),
            'num_resistance_levels': len(self.resistance_levels)
        }