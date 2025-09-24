"""
Market Facilitation Index (MFI)
================================

市场促进指数，由Bill Williams开发，用于分析价格变化和成交量
之间的关系。MFI帮助识别市场参与者的真实意图，发现潜在的
趋势变化和突破点。

核心原理：
- 价格上涨 + 成交量增加 = 绿色MFI（参与者积极参与）
- 价格上涨 + 成交量减少 = 褐色MFI（参与者兴趣减弱）
- 价格下跌 + 成交量增加 = 蓝色MFI（参与者恐慌抛售）
- 价格下跌 + 成交量减少 = 粉色MFI（参与者兴趣丧失）

特点：
- 四色MFI信号系统
- 成交量-价格关系分析
- 市场情绪识别
- 趋势强度评估
- 潜在转折点预警
- 交易时机确认

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional


class MarketFacilitationIndex:
    """
    Market Facilitation Index - 市场促进指数

    通过分析价格变化与成交量的关系来识别市场参与者的真实意图
    和潜在的趋势变化。
    """

    def __init__(self, period: int = 1, smoothing_period: int = 5):
        """
        初始化市场促进指数

        Args:
            period: MFI计算周期，默认1
            smoothing_period: 平滑周期，默认5
        """
        self.period = period
        self.smoothing_period = smoothing_period
        self.name = f"Market Facilitation Index ({period})"
        self.category = "volume"

    def calculate_mfi(self, high: Union[List[float], pd.Series],
                    low: Union[List[float], pd.Series],
                    volume: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        计算市场促进指数

        Args:
            high: 最高价序列
            low: 最低价序列
            volume: 成交量序列

        Returns:
            包含所有指标的字典
        """
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        volume_series = pd.Series(volume)

        # 计算MFI值
        mfi_value = self.calculate_mfi_value(high_series, low_series, volume_series)

        # 计算MFI变化
        mfi_change = mfi_value.diff()

        # 计算成交量变化
        volume_change = volume_series.diff()

        # 计算价格变化
        price_change = (high_series - low_series).diff()

        # 确定MFI颜色信号
        mfi_colors = self.determine_mfi_colors(mfi_value, mfi_change, volume_change)

        # 计算MFI信号强度
        signal_strength = self.calculate_signal_strength(mfi_value, volume_series)

        # 计算市场情绪
        market_sentiment = self.calculate_market_sentiment(mfi_colors)

        # 计算参与度指标
        participation_index = self.calculate_participation_index(mfi_value, volume_series)

        # 计算趋势效率
        trend_efficiency = self.calculate_trend_efficiency(mfi_value, price_change)

        # 生成交易信号
        signals = self.generate_trading_signals(mfi_colors, signal_strength, market_sentiment)

        # 识别市场阶段
        market_phase = self.identify_market_phase(mfi_colors, signal_strength)

        return {
            'mfi_value': mfi_value,
            'mfi_change': mfi_change,
            'volume_change': volume_change,
            'price_change': price_change,
            'mfi_colors': mfi_colors,
            'signal_strength': signal_strength,
            'market_sentiment': market_sentiment,
            'participation_index': participation_index,
            'trend_efficiency': trend_efficiency,
            'signals': signals,
            'market_phase': market_phase
        }

    def calculate_mfi_value(self, high: pd.Series, low: pd.Series, volume: pd.Series) -> pd.Series:
        """
        计算MFI值

        公式: MFI = (High - Low) / Volume

        Args:
            high: 最高价序列
            low: 最低价序列
            volume: 成交量序列

        Returns:
            MFI值序列
        """
        # 计算价格区间
        price_range = high - low

        # 避免除零
        volume_safe = volume.replace(0, 1)

        # 计算MFI
        mfi = price_range / volume_safe

        # 平滑处理
        mfi = mfi.rolling(window=self.smoothing_period).mean()

        return mfi.fillna(0)

    def determine_mfi_colors(self, mfi_value: pd.Series, mfi_change: pd.Series,
                            volume_change: pd.Series) -> pd.Series:
        """
        确定MFI颜色信号

        Args:
            mfi_value: MFI值序列
            mfi_change: MFI变化序列
            volume_change: 成交量变化序列

        Returns:
            MFI颜色信号序列
        """
        colors = pd.Series(0, index=mfi_value.index)

        for i in range(len(mfi_value)):
            if i == 0:
                colors.iloc[i] = 0  # 无数据
                continue

            # 绿色: MFI增加且成交量增加
            if (mfi_change.iloc[i] > 0) and (volume_change.iloc[i] > 0):
                colors.iloc[i] = 1  # 绿色 - 参与者积极参与

            # 褐色: MFI增加且成交量减少
            elif (mfi_change.iloc[i] > 0) and (volume_change.iloc[i] < 0):
                colors.iloc[i] = 2  # 褐色 - 参与者兴趣减弱

            # 蓝色: MFI减少且成交量增加
            elif (mfi_change.iloc[i] < 0) and (volume_change.iloc[i] > 0):
                colors.iloc[i] = 3  # 蓝色 - 参与者恐慌抛售

            # 粉色: MFI减少且成交量减少
            elif (mfi_change.iloc[i] < 0) and (volume_change.iloc[i] < 0):
                colors.iloc[i] = 4  # 粉色 - 参与者兴趣丧失

            else:
                colors.iloc[i] = 0  # 无变化

        return colors

    def calculate_signal_strength(self, mfi_value: pd.Series, volume: pd.Series) -> pd.Series:
        """
        计算MFI信号强度

        Args:
            mfi_value: MFI值序列
            volume: 成交量序列

        Returns:
            信号强度序列 (0-100)
        """
        # 计算MFI的相对强度
        mfi_ma = mfi_value.rolling(window=20).mean()
        mfi_std = mfi_value.rolling(window=20).std()
        mfi_zscore = (mfi_value - mfi_ma) / mfi_std

        # 计算成交量强度
        volume_ma = volume.rolling(window=20).mean()
        volume_ratio = volume / volume_ma

        # 综合信号强度
        signal_strength = (abs(mfi_zscore) * 20 + volume_ratio * 30)

        # 标准化到0-100
        signal_strength = signal_strength.clip(0, 100)

        return signal_strength.fillna(50)

    def calculate_market_sentiment(self, mfi_colors: pd.Series) -> Dict[str, pd.Series]:
        """
        计算市场情绪

        Args:
            mfi_colors: MFI颜色序列

        Returns:
            市场情绪指标
        """
        # 计算各种颜色的频率
        green_frequency = mfi_colors.rolling(window=10).apply(lambda x: (x == 1).sum() / len(x))
        brown_frequency = mfi_colors.rolling(window=10).apply(lambda x: (x == 2).sum() / len(x))
        blue_frequency = mfi_colors.rolling(window=10).apply(lambda x: (x == 3).sum() / len(x))
        pink_frequency = mfi_colors.rolling(window=10).apply(lambda x: (x == 4).sum() / len(x))

        # 计算情绪指标
        sentiment_score = (green_frequency * 1.0 + brown_frequency * 0.5 -
                         blue_frequency * 1.0 - pink_frequency * 0.5) * 100

        # 确定情绪类型
        sentiment_type = pd.Series('neutral', index=mfi_colors.index)
        sentiment_type[sentiment_score > 30] = 'bullish'
        sentiment_type[sentiment_score < -30] = 'bearish'
        sentiment_type[sentiment_score > 60] = 'very_bullish'
        sentiment_type[sentiment_score < -60] = 'very_bearish'

        return {
            'green_frequency': green_frequency.fillna(0),
            'brown_frequency': brown_frequency.fillna(0),
            'blue_frequency': blue_frequency.fillna(0),
            'pink_frequency': pink_frequency.fillna(0),
            'sentiment_score': sentiment_score.fillna(0),
            'sentiment_type': sentiment_type
        }

    def calculate_participation_index(self, mfi_value: pd.Series, volume: pd.Series) -> pd.Series:
        """
        计算参与度指标

        Args:
            mfi_value: MFI值序列
            volume: 成交量序列

        Returns:
            参与度指标序列 (0-100)
        """
        # 计算MFI和成交量的相关性
        correlation = mfi_value.rolling(window=20).corr(volume)

        # 计算参与度
        participation = (correlation + 1) * 50  # 转换为0-100

        return participation.fillna(50)

    def calculate_trend_efficiency(self, mfi_value: pd.Series, price_change: pd.Series) -> pd.Series:
        """
        计算趋势效率

        Args:
            mfi_value: MFI值序列
            price_change: 价格变化序列

        Returns:
            趋势效率序列 (0-100)
        """
        # 计算MFI效率（价格变化与MFI的关系）
        mfi_efficiency = abs(price_change) / (mfi_value + 1e-8)

        # 标准化
        efficiency_ma = mfi_efficiency.rolling(window=20).mean()
        efficiency_std = mfi_efficiency.rolling(window=20).std()
        efficiency_zscore = (mfi_efficiency - efficiency_ma) / efficiency_std

        # 转换为效率指标
        trend_efficiency = (efficiency_zscore + 3) * 16.67  # 转换为0-100
        trend_efficiency = trend_efficiency.clip(0, 100)

        return trend_efficiency.fillna(50)

    def generate_trading_signals(self, mfi_colors: pd.Series, signal_strength: pd.Series,
                               market_sentiment: Dict[str, pd.Series]) -> pd.Series:
        """
        生成交易信号

        Args:
            mfi_colors: MFI颜色序列
            signal_strength: 信号强度序列
            market_sentiment: 市场情绪

        Returns:
            交易信号序列 (2=强买入, 1=买入, 0=持有, -1=卖出, -2=强卖出)
        """
        signals = pd.Series(0, index=mfi_colors.index)

        # 获取情绪数据
        sentiment_type = market_sentiment['sentiment_type']
        green_freq = market_sentiment['green_frequency']
        blue_freq = market_sentiment['blue_frequency']

        # 强买入信号：绿色MFI主导且情绪看涨
        strong_buy = (
            (mfi_colors == 1) &  # 绿色MFI
            (sentiment_type == 'very_bullish') &
            (signal_strength > 60) &
            (green_freq > 0.6)
        )

        # 买入信号：绿色MFI增加且情绪积极
        buy = (
            (mfi_colors == 1) &
            (sentiment_type.isin(['bullish', 'very_bullish'])) &
            (signal_strength > 40)
        )

        # 强卖出信号：蓝色MFI主导且情绪看跌
        strong_sell = (
            (mfi_colors == 3) &  # 蓝色MFI
            (sentiment_type == 'very_bearish') &
            (signal_strength > 60) &
            (blue_freq > 0.6)
        )

        # 卖出信号：蓝色MFI增加且情绪消极
        sell = (
            (mfi_colors == 3) &
            (sentiment_type.isin(['bearish', 'very_bearish'])) &
            (signal_strength > 40)
        )

        # 警告信号：褐色或粉色MFI（市场犹豫）
        warning = (mfi_colors.isin([2, 4])) & (signal_strength > 50)

        signals[strong_buy] = 2
        signals[buy] = 1
        signals[strong_sell] = -2
        signals[sell] = -1
        signals[warning] = 0  # 警告信号，建议观望

        return signals

    def identify_market_phase(self, mfi_colors: pd.Series, signal_strength: pd.Series) -> pd.Series:
        """
        识别市场阶段

        Args:
            mfi_colors: MFI颜色序列
            signal_strength: 信号强度序列

        Returns:
            市场阶段序列
        """
        market_phase = pd.Series('accumulation', index=mfi_colors.index)

        # 计算最近的颜色分布
        recent_colors = mfi_colors.rolling(window=10).apply(lambda x: x.mode().iloc[0] if len(x) > 0 else 0)

        # 根据MFI颜色确定市场阶段
        market_phase[recent_colors == 1] = 'trending'      # 绿色 - 趋势阶段
        market_phase[recent_colors == 2] = 'distribution'  # 褐色 - 分配阶段
        market_phase[recent_colors == 3] = 'panic'         # 蓝色 - 恐慌阶段
        market_phase[recent_colors == 4] = 'exhaustion'     # 粉色 - 疲惫阶段

        # 根据信号强度调整
        market_phase[signal_strength > 70] = 'high_activity'
        market_phase[signal_strength < 30] = 'low_activity'

        return market_phase

    def get_comprehensive_analysis(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        获取综合分析结果

        Args:
            data: 包含OHLCV数据的字典

        Returns:
            综合分析结果
        """
        # 计算MFI
        mfi_data = self.calculate_mfi(data['high'], data['low'], data['volume'])

        # 计算多时间框架分析
        mtf_analysis = self.calculate_mtf_analysis(data)

        # 计算市场结构
        market_structure = self.analyze_market_structure(mfi_data)

        # 生成高级信号
        advanced_signals = self.generate_advanced_signals(mfi_data, mtf_analysis)

        # 计算成交量效率
        volume_efficiency = self.calculate_volume_efficiency(data, mfi_data)

        return {
            'mfi_value': mfi_data['mfi_value'],
            'mfi_colors': mfi_data['mfi_colors'],
            'signal_strength': mfi_data['signal_strength'],
            'market_sentiment': mfi_data['market_sentiment'],
            'participation_index': mfi_data['participation_index'],
            'trend_efficiency': mfi_data['trend_efficiency'],
            'signals': advanced_signals,
            'market_phase': mfi_data['market_phase'],
            'mtf_analysis': mtf_analysis,
            'market_structure': market_structure,
            'volume_efficiency': volume_efficiency
        }

    def calculate_mtf_analysis(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        计算多时间框架分析

        Args:
            data: 市场数据

        Returns:
            多时间框架分析结果
        """
        # 简化版本：使用不同周期计算MFI
        periods = [5, 10, 20]
        mtf_results = {}

        for period in periods:
            # 临时修改平滑周期
            original_smoothing = self.smoothing_period
            self.smoothing_period = period

            # 计算该周期的MFI
            temp_data = self.calculate_mfi(data['high'], data['low'], data['volume'])

            mtf_results[f'period_{period}'] = {
                'mfi_colors': temp_data['mfi_colors'],
                'signal_strength': temp_data['signal_strength'],
                'dominant_color': temp_data['mfi_colors'].iloc[-1]
            }

            # 恢复原始参数
            self.smoothing_period = original_smoothing

        # 计算多时间框架共识
        dominant_colors = [result['dominant_color'] for result in mtf_results.values()]
        color_counts = pd.Series(dominant_colors).value_counts()

        # 确定主导颜色
        if len(color_counts) > 0:
            consensus_color = color_counts.index[0]
            consensus_strength = color_counts.iloc[0] / len(dominant_colors)
        else:
            consensus_color = 0
            consensus_strength = 0

        return {
            'timeframes': mtf_results,
            'consensus_color': consensus_color,
            'consensus_strength': consensus_strength,
            'color_agreement': consensus_strength
        }

    def analyze_market_structure(self, mfi_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        分析市场结构

        Args:
            mfi_data: MFI数据

        Returns:
            市场结构分析
        """
        colors = mfi_data['mfi_colors']
        signal_strength = mfi_data['signal_strength']

        # 计算颜色模式
        color_patterns = self.identify_color_patterns(colors)

        # 计算市场强度
        market_strength = self.calculate_market_strength(colors, signal_strength)

        # 计算转换概率
        transition_probability = self.calculate_transition_probability(colors)

        return {
            'color_patterns': color_patterns,
            'market_strength': market_strength,
            'transition_probability': transition_probability
        }

    def identify_color_patterns(self, colors: pd.Series) -> Dict[str, Any]:
        """
        识别颜色模式

        Args:
            colors: MFI颜色序列

        Returns:
            颜色模式识别结果
        """
        patterns = {}

        # 连续绿色（强势趋势）
        consecutive_green = (colors == 1).rolling(window=3).sum()
        patterns['strong_trend'] = consecutive_green >= 3

        # 连续蓝色（恐慌抛售）
        consecutive_blue = (colors == 3).rolling(window=3).sum()
        patterns['panic_selling'] = consecutive_blue >= 3

        # 颜色交替（震荡）
        color_changes = colors.diff().abs().rolling(window=5).sum()
        patterns['choppy'] = color_changes >= 4

        # 颜色一致（趋势延续）
        color_consistency = colors.rolling(window=5).apply(lambda x: x.nunique() == 1)
        patterns['trend_continuation'] = color_consistency == 1

        return patterns

    def calculate_market_strength(self, colors: pd.Series, signal_strength: pd.Series) -> float:
        """
        计算市场强度

        Args:
            colors: MFI颜色序列
            signal_strength: 信号强度序列

        Returns:
            市场强度 (0-100)
        """
        # 计算绿色和蓝色的权重
        recent_colors = colors.iloc[-10:]
        green_weight = (recent_colors == 1).sum() * 1.0
        blue_weight = (recent_colors == 3).sum() * -1.0

        # 计算基础强度
        base_strength = (green_weight + blue_weight) / len(recent_colors) * 50 + 50

        # 结合信号强度
        combined_strength = (base_strength + signal_strength.iloc[-1]) / 2

        return max(0, min(100, combined_strength))

    def calculate_transition_probability(self, colors: pd.Series) -> Dict[str, float]:
        """
        计算颜色转换概率

        Args:
            colors: MFI颜色序列

        Returns:
            转换概率字典
        """
        # 简化的转换概率计算
        transitions = {}

        # 计算从绿色到其他颜色的概率
        green_indices = colors[colors == 1].index
        if len(green_indices) > 1:
            next_colors = colors.loc[green_indices[1:]]
            transition_counts = next_colors.value_counts()
            total = transition_counts.sum()
            transitions['green_to_green'] = transition_counts.get(1, 0) / total
            transitions['green_to_brown'] = transition_counts.get(2, 0) / total
            transitions['green_to_blue'] = transition_counts.get(3, 0) / total
            transitions['green_to_pink'] = transition_counts.get(4, 0) / total

        return transitions

    def generate_advanced_signals(self, mfi_data: Dict[str, pd.Series],
                                 mtf_analysis: Dict[str, Any]) -> pd.Series:
        """
        生成高级交易信号

        Args:
            mfi_data: MFI数据
            mtf_analysis: 多时间框架分析

        Returns:
            高级信号序列
        """
        signals = pd.Series(0, index=mfi_data['mfi_colors'].index)

        # 基础信号
        base_signals = mfi_data['signals']

        # 多时间框架确认
        mtf_confirmed = (
            (mtf_analysis['consensus_strength'] > 0.6) &
            (base_signals != 0)
        )

        # 市场结构过滤
        strong_trend = mfi_data['market_phase'] == 'trending'
        panic_phase = mfi_data['market_phase'] == 'panic'

        # 信号强度过滤
        high_strength = mfi_data['signal_strength'] > 60

        # 高级买入信号
        advanced_buy = (
            (base_signals > 0) &
            (mtf_confirmed | strong_trend) &
            high_strength
        )

        # 高级卖出信号
        advanced_sell = (
            (base_signals < 0) &
            (mtf_confirmed | panic_phase) &
            high_strength
        )

        signals[advanced_buy] = 2
        signals[advanced_sell] = -2

        # 保留原有的买入卖出信号
        signals[(base_signals == 1) & (signals == 0)] = 1
        signals[(base_signals == -1) & (signals == 0)] = -1

        return signals

    def calculate_volume_efficiency(self, data: Dict[str, pd.Series],
                                 mfi_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        计算成交量效率

        Args:
            data: 市场数据
            mfi_data: MFI数据

        Returns:
            成交量效率指标
        """
        # 计算价格变化
        price_change = abs(data['high'] - data['low'])

        # 计算成交量效率
        volume_efficiency = price_change / (data['volume'] + 1e-8)

        # 计算平均效率
        avg_efficiency = volume_efficiency.rolling(window=20).mean().iloc[-1]
        current_efficiency = volume_efficiency.iloc[-1]

        # 效率比率
        efficiency_ratio = current_efficiency / avg_efficiency if avg_efficiency > 0 else 1

        return {
            'current_efficiency': current_efficiency,
            'average_efficiency': avg_efficiency,
            'efficiency_ratio': efficiency_ratio,
            'efficiency_trend': 'improving' if efficiency_ratio > 1 else 'declining'
        }

    def get_trading_recommendations(self, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            analysis: 综合分析结果

        Returns:
            交易建议DataFrame
        """
        recommendations = pd.DataFrame(index=analysis['mfi_colors'].index)

        # 信号描述
        recommendations['signal'] = analysis['signals']
        recommendations['signal_description'] = analysis['signals'].map({
            2: '强烈买入 - 趋势确认',
            1: '买入 - 参与度增加',
            0: '持有 - 观望',
            -1: '卖出 - 恐慌抛售',
            -2: '强烈卖出 - 多框架确认'
        })

        # MFI颜色
        recommendations['mfi_color'] = analysis['mfi_colors'].map({
            0: '无数据',
            1: '绿色 - 积极参与',
            2: '褐色 - 兴趣减弱',
            3: '蓝色 - 恐慌抛售',
            4: '粉色 - 兴趣丧失'
        })

        # 市场阶段
        recommendations['market_phase'] = analysis['market_phase']

        # 市场情绪
        recommendations['market_sentiment'] = analysis['market_sentiment']['sentiment_type']

        # 仓位建议
        recommendations['position_size'] = analysis['signals'].map({
            2: 0.6,   # 强买入：60%仓位
            1: 0.3,   # 买入：30%仓位
            0: 0.2,   # 持有：20%仓位
            -1: 0.1,  # 卖出：10%仓位
            -2: 0.0   # 强卖出：0仓位
        })

        # 调整仓位基于参与度
        participation_adjustment = analysis['participation_index'].iloc[-1] / 100
        recommendations['position_size'] *= participation_adjustment

        # 止损止盈建议
        current_price = analysis['mfi_colors'].index.to_series().map(lambda x: analysis['mfi_value'].loc[x])
        recommendations['stop_loss'] = current_price * 0.95
        recommendations['take_profit'] = current_price * 1.06

        # 信号强度
        recommendations['signal_strength'] = analysis['signal_strength']

        # 趋势效率
        recommendations['trend_efficiency'] = analysis['trend_efficiency']

        return recommendations

    def get_color_explanation(self) -> Dict[str, str]:
        """
        获取MFI颜色解释

        Returns:
            颜色解释字典
        """
        return {
            '1': '绿色 - MFI增加且成交量增加：参与者积极参与，趋势可能继续',
            '2': '褐色 - MFI增加且成交量减少：参与者兴趣减弱，警惕趋势反转',
            '3': '蓝色 - MFI减少且成交量增加：参与者恐慌抛售，可能接近底部',
            '4': '粉色 - MFI减少且成交量减少：参与者兴趣丧失，市场可能盘整'
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    n = len(dates)

    # 生成OHLCV数据
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.random.rand(n) * 2
    low = close - np.random.rand(n) * 2
    volume = np.random.randint(1000000, 5000000, n)

    data = {
        'high': pd.Series(high, index=dates),
        'low': pd.Series(low, index=dates),
        'close': pd.Series(close, index=dates),
        'volume': pd.Series(volume, index=dates)
    }

    # 创建市场促进指数指标
    mfi = MarketFacilitationIndex()

    # 计算指标
    results = mfi.calculate_mfi(data['high'], data['low'], data['volume'])

    print("Market Facilitation Index 结果:")
    print(f"MFI值: {results['mfi_value'].iloc[-1]:.6f}")
    print(f"MFI颜色: {results['mfi_colors'].iloc[-1]}")
    print(f"信号强度: {results['signal_strength'].iloc[-1]:.2f}")
    print(f"当前信号: {results['signals'].iloc[-1]}")
    print(f"市场阶段: {results['market_phase'].iloc[-1]}")

    # 获取颜色解释
    color_explanation = mfi.get_color_explanation()
    print(f"\nMFI颜色解释:")
    print(f"当前状态: {color_explanation[str(results['mfi_colors'].iloc[-1])]}")

    # 获取综合分析
    comprehensive = mfi.get_comprehensive_analysis(data)
    print("\n综合分析结果:")
    print(f"市场情绪: {comprehensive['market_sentiment']['sentiment_type'].iloc[-1]}")
    print(f"参与度: {comprehensive['participation_index'].iloc[-1]:.2f}")
    print(f"趋势效率: {comprehensive['trend_efficiency'].iloc[-1]:.2f}")
    print(f"多时间框架共识强度: {comprehensive['mtf_analysis']['consensus_strength']:.2f}")
    print(f"高级信号: {comprehensive['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = mfi.get_trading_recommendations(comprehensive)
    print("\n最新交易建议:")
    print(recommendations.iloc[-1])