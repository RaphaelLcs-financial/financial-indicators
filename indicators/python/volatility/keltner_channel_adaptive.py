"""
Keltner Channel with Adaptive Bands
==================================

自适应凯尔特纳通道，结合了Chester Keltner的原始理念和Linda Bradford Raschke的改进。
通过ATR（平均真实波幅）来动态调整通道宽度，能够自适应市场波动性变化，
提供更精准的支撑阻力位和趋势跟踪信号。

特点：
- 动态通道宽度调整
- 多时间框架分析
- 趋势强度评估
- 突破确认机制
- 波动率自适应
- 价格行为识别
- 综合交易信号

公式：
中轨 = EMA(收盘价, period)
上轨 = 中轨 + multiplier * ATR(period)
下轨 = 中轨 - multiplier * ATR(period)

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional


class KeltnerChannelAdaptive:
    """
    自适应凯尔特纳通道

    基于ATR的动态通道指标，结合波动率自适应和趋势跟踪功能，
    提供更精确的价格边界和交易信号。
    """

    def __init__(self, period: int = 20, multiplier: float = 2.0,
                 adaptive_period: int = 10, volatility_factor: float = 1.5):
        """
        初始化自适应凯尔特纳通道

        Args:
            period: 基础周期，默认20
            multiplier: ATR乘数，默认2.0
            adaptive_period: 自适应周期，默认10
            volatility_factor: 波动率因子，默认1.5
        """
        self.period = period
        self.multiplier = multiplier
        self.adaptive_period = adaptive_period
        self.volatility_factor = volatility_factor
        self.name = f"Keltner Channel Adaptive ({period})"
        self.category = "volatility"

    def calculate_keltner_channel(self, data: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        计算自适应凯尔特纳通道

        Args:
            data: 包含OHLCV数据的字典

        Returns:
            包含所有指标的字典
        """
        high = data['high']
        low = data['low']
        close = data['close']
        volume = data.get('volume', pd.Series(1, index=close.index))

        # 计算基础指标
        middle_band = self.calculate_ema(close, self.period)
        atr = self.calculate_atr(high, low, close, self.period)

        # 计算自适应乘数
        adaptive_multiplier = self.calculate_adaptive_multiplier(close, atr, volume)

        # 计算动态通道
        upper_band = middle_band + adaptive_multiplier * atr
        lower_band = middle_band - adaptive_multiplier * atr

        # 计算通道宽度
        channel_width = upper_band - lower_band
        channel_width_percent = (channel_width / middle_band) * 100

        # 计算价格位置
        price_position = self.calculate_price_position(close, upper_band, lower_band)

        # 计算通道斜率
        channel_slope = self.calculate_channel_slope(middle_band)

        # 计算波动率
        volatility = self.calculate_volatility(close)

        # 计算趋势强度
        trend_strength = self.calculate_trend_strength(close, middle_band, atr)

        # 计算突破信号
        breakout_signals = self.calculate_breakout_signals(close, upper_band, lower_band, volume)

        # 计算回撤信号
        pullback_signals = self.calculate_pullback_signals(close, upper_band, lower_band, middle_band)

        # 计算挤压状态
        squeeze_condition = self.calculate_squeeze_condition(channel_width, atr)

        # 生成综合信号
        signals = self.generate_signals(
            close, upper_band, lower_band, middle_band,
            price_position, trend_strength, breakout_signals,
            pullback_signals, squeeze_condition, volume
        )

        # 计算动态支撑阻力
        support_resistance = self.calculate_dynamic_support_resistance(
            close, upper_band, lower_band, middle_band
        )

        return {
            'upper_band': upper_band,
            'middle_band': middle_band,
            'lower_band': lower_band,
            'atr': atr,
            'adaptive_multiplier': adaptive_multiplier,
            'channel_width': channel_width,
            'channel_width_percent': channel_width_percent,
            'price_position': price_position,
            'channel_slope': channel_slope,
            'volatility': volatility,
            'trend_strength': trend_strength,
            'breakout_signals': breakout_signals,
            'pullback_signals': pullback_signals,
            'squeeze_condition': squeeze_condition,
            'signals': signals,
            'support_levels': support_resistance['support'],
            'resistance_levels': support_resistance['resistance']
        }

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        alpha = 2 / (period + 1)
        ema = prices.copy()
        for i in range(1, len(prices)):
            ema.iloc[i] = alpha * prices.iloc[i] + (1 - alpha) * ema.iloc[i-1]
        return ema

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
        """计算平均真实波幅"""
        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算ATR
        atr = tr.rolling(window=period).mean()
        return atr.fillna(atr.iloc[:period].mean())

    def calculate_adaptive_multiplier(self, close: pd.Series, atr: pd.Series, volume: pd.Series) -> pd.Series:
        """
        计算自适应乘数

        基于波动率和成交量的动态调整
        """
        # 计算波动率比率
        volatility_ratio = atr / close.rolling(window=self.adaptive_period).mean()

        # 计算成交量变化
        volume_change = volume.pct_change().rolling(window=self.adaptive_period).mean()

        # 自适应乘数计算
        base_multiplier = self.multiplier

        # 波动率调整
        vol_adjustment = 1 + (volatility_ratio - volatility_ratio.rolling(window=50).mean()) / volatility_ratio.rolling(window=50).mean()

        # 成交量调整
        volume_adjustment = 1 + volume_change * 0.5

        # 综合调整
        adaptive_multiplier = base_multiplier * vol_adjustment * volume_adjustment

        # 限制范围
        adaptive_multiplier = adaptive_multiplier.clip(0.5, 4.0)

        return adaptive_multiplier.fillna(base_multiplier)

    def calculate_price_position(self, close: pd.Series, upper: pd.Series, lower: pd.Series) -> pd.Series:
        """计算价格在通道中的位置 (0-100)"""
        position = (close - lower) / (upper - lower) * 100
        return position.fillna(50)

    def calculate_channel_slope(self, middle_band: pd.Series) -> pd.Series:
        """计算通道斜率"""
        slope = middle_band.diff()
        # 标准化斜率
        slope_normalized = slope / middle_band.shift(1) * 100
        return slope_normalized.fillna(0)

    def calculate_volatility(self, close: pd.Series) -> pd.Series:
        """计算波动率"""
        returns = close.pct_change()
        volatility = returns.rolling(window=20).std() * np.sqrt(252)
        return volatility.fillna(0)

    def calculate_trend_strength(self, close: pd.Series, middle_band: pd.Series, atr: pd.Series) -> pd.Series:
        """计算趋势强度"""
        # 价格距离中轨的距离
        distance_from_middle = abs(close - middle_band)

        # 标准化距离
        normalized_distance = distance_from_middle / atr

        # 趋势强度
        trend_strength = normalized_distance * 100

        # 平滑处理
        trend_strength = trend_strength.rolling(window=10).mean()

        return trend_strength.fillna(0)

    def calculate_breakout_signals(self, close: pd.Series, upper: pd.Series, lower: pd.Series, volume: pd.Series) -> pd.Series:
        """计算突破信号"""
        signals = pd.Series(0, index=close.index)

        # 成交量确认
        volume_spike = volume > volume.rolling(window=20).mean() * 1.5

        # 向上突破
        breakout_up = (close > upper) & (close.shift(1) <= upper.shift(1)) & volume_spike

        # 向下突破
        breakout_down = (close < lower) & (close.shift(1) >= lower.shift(1)) & volume_spike

        signals[breakout_up] = 1
        signals[breakout_down] = -1

        return signals

    def calculate_pullback_signals(self, close: pd.Series, upper: pd.Series, lower: pd.Series, middle_band: pd.Series) -> pd.Series:
        """计算回撤信号"""
        signals = pd.Series(0, index=close.index)

        # 回撤到中轨支撑
        pullback_to_support = (
            (close > middle_band) &
            (close <= middle_band + middle_band * 0.01) &
            (close.shift(1) > middle_band.shift(1))
        )

        # 回撤到中轨阻力
        pullback_to_resistance = (
            (close < middle_band) &
            (close >= middle_band - middle_band * 0.01) &
            (close.shift(1) < middle_band.shift(1))
        )

        signals[pullback_to_support] = 1
        signals[pullback_to_resistance] = -1

        return signals

    def calculate_squeeze_condition(self, channel_width: pd.Series, atr: pd.Series) -> pd.Series:
        """计算挤压状态"""
        # 通道宽度相对ATR的比率
        width_ratio = channel_width / atr

        # 挤压条件：通道宽度相对较窄
        squeeze_threshold = width_ratio.rolling(window=50).quantile(0.2)
        squeeze_condition = width_ratio < squeeze_threshold

        return squeeze_condition.astype(int)

    def generate_signals(self, close: pd.Series, upper: pd.Series, lower: pd.Series, middle_band: pd.Series,
                       price_position: pd.Series, trend_strength: pd.Series, breakout_signals: pd.Series,
                       pullback_signals: pd.Series, squeeze_condition: pd.Series, volume: pd.Series) -> pd.Series:
        """生成综合交易信号"""
        signals = pd.Series(0, index=close.index)

        # 趋势方向
        above_middle = close > middle_band
        below_middle = close < middle_band

        # 强趋势过滤
        strong_trend = trend_strength > 50

        # 挤压突破信号（高优先级）
        squeeze_breakout_up = (
            (close > upper) &
            squeeze_condition.shift(1) &
            ~squeeze_condition &
            (volume > volume.rolling(window=20).mean())
        )

        squeeze_breakout_down = (
            (close < lower) &
            squeeze_condition.shift(1) &
            ~squeeze_condition &
            (volume > volume.rolling(window=20).mean())
        )

        # 趋势跟随信号
        trend_following_buy = (
            above_middle &
            (price_position > 70) &
            strong_trend &
            (breakout_signals == 1)
        )

        trend_following_sell = (
            below_middle &
            (price_position < 30) &
            strong_trend &
            (breakout_signals == -1)
        )

        # 回撤入场信号
        pullback_buy = (
            above_middle &
            (pullback_signals == 1) &
            (trend_strength > 30)
        )

        pullback_sell = (
            below_middle &
            (pullback_signals == -1) &
            (trend_strength > 30)
        )

        # 反转信号（在极值位置）
        reversal_buy = (
            (price_position < 10) &
            (trend_strength < 20) &
            (close.shift(1) < close)
        )

        reversal_sell = (
            (price_position > 90) &
            (trend_strength < 20) &
            (close.shift(1) > close)
        )

        # 信号强度赋值
        signals[squeeze_breakout_up] = 3     # 强买入：挤压突破
        signals[squeeze_breakout_down] = -3  # 强卖出：挤压突破
        signals[trend_following_buy] = 2     # 买入：趋势跟随
        signals[trend_following_sell] = -2   # 卖出：趋势跟随
        signals[pullback_buy] = 1            # 弱买入：回撤
        signals[pullback_sell] = -1           # 弱卖出：回撤
        signals[reversal_buy] = 1            # 买入：反转
        signals[reversal_sell] = -1          # 卖出：反转

        return signals

    def calculate_dynamic_support_resistance(self, close: pd.Series, upper: pd.Series, lower: pd.Series, middle_band: pd.Series) -> Dict[str, pd.Series]:
        """计算动态支撑阻力位"""
        support_levels = pd.Series(index=close.index, dtype=float)
        resistance_levels = pd.Series(index=close.index, dtype=float)

        for i in range(len(close)):
            if i < 20:
                support_levels.iloc[i] = lower.iloc[i]
                resistance_levels.iloc[i] = upper.iloc[i]
            else:
                # 动态支撑位
                if close.iloc[i] > middle_band.iloc[i]:
                    # 上升趋势中，支撑位在较低的价格
                    recent_lows = close.iloc[max(0, i-20):i].nsmallest(3)
                    support_levels.iloc[i] = recent_lows.mean()
                else:
                    # 下降趋势中，支撑位在下轨
                    support_levels.iloc[i] = lower.iloc[i]

                # 动态阻力位
                if close.iloc[i] < middle_band.iloc[i]:
                    # 下降趋势中，阻力位在较高的价格
                    recent_highs = close.iloc[max(0, i-20):i].nlargest(3)
                    resistance_levels.iloc[i] = recent_highs.mean()
                else:
                    # 上升趋势中，阻力位在上轨
                    resistance_levels.iloc[i] = upper.iloc[i]

        return {
            'support': support_levels,
            'resistance': resistance_levels
        }

    def get_comprehensive_analysis(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        获取综合分析结果

        Args:
            data: 包含OHLCV数据的字典

        Returns:
            综合分析结果
        """
        # 计算凯尔特纳通道
        keltner_data = self.calculate_keltner_channel(data)

        # 计算多时间框架分析
        mtf_analysis = self.calculate_mtf_analysis(data)

        # 计算市场状态
        market_state = self.analyze_market_state(keltner_data)

        # 生成高级信号
        advanced_signals = self.generate_advanced_signals(keltner_data, mtf_analysis)

        # 计算风险管理指标
        risk_metrics = self.calculate_risk_metrics(keltner_data, data)

        return {
            'upper_band': keltner_data['upper_band'],
            'middle_band': keltner_data['middle_band'],
            'lower_band': keltner_data['lower_band'],
            'atr': keltner_data['atr'],
            'adaptive_multiplier': keltner_data['adaptive_multiplier'],
            'channel_width': keltner_data['channel_width'],
            'price_position': keltner_data['price_position'],
            'trend_strength': keltner_data['trend_strength'],
            'squeeze_condition': keltner_data['squeeze_condition'],
            'signals': advanced_signals,
            'mtf_analysis': mtf_analysis,
            'market_state': market_state,
            'risk_metrics': risk_metrics,
            'support_levels': keltner_data['support_levels'],
            'resistance_levels': keltner_data['resistance_levels']
        }

    def calculate_mtf_analysis(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """计算多时间框架分析"""
        close = data['close']
        periods = [10, 20, 30]
        mtf_results = {}

        for period in periods:
            # 临时修改参数
            original_period = self.period
            self.period = period

            # 计算该周期的凯尔特纳通道
            temp_data = self.calculate_keltner_channel(data)

            mtf_results[f'period_{period}'] = {
                'upper_band': temp_data['upper_band'],
                'middle_band': temp_data['middle_band'],
                'lower_band': temp_data['lower_band'],
                'price_position': temp_data['price_position'],
                'trend_strength': temp_data['trend_strength'],
                'signal': 'bullish' if temp_data['price_position'].iloc[-1] > 50 else 'bearish'
            }

            # 恢复原始参数
            self.period = original_period

        # 计算多时间框架共识
        signals = [result['signal'] for result in mtf_results.values()]
        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')

        if bullish_count >= 2:
            consensus = 'bullish'
        elif bearish_count >= 2:
            consensus = 'bearish'
        else:
            consensus = 'mixed'

        return {
            'timeframes': mtf_results,
            'consensus': consensus,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count
        }

    def analyze_market_state(self, keltner_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """分析市场状态"""
        price_position = keltner_data['price_position']
        trend_strength = keltner_data['trend_strength']
        squeeze_condition = keltner_data['squeeze_condition']
        channel_slope = keltner_data['channel_slope']

        # 通道状态
        if squeeze_condition.iloc[-1] == 1:
            channel_state = 'squeeze'
        elif trend_strength.iloc[-1] > 60:
            channel_state = 'expansion'
        else:
            channel_state = 'normal'

        # 趋势状态
        if price_position.iloc[-1] > 70:
            trend_state = 'strong_uptrend'
        elif price_position.iloc[-1] > 55:
            trend_state = 'uptrend'
        elif price_position.iloc[-1] < 30:
            trend_state = 'strong_downtrend'
        elif price_position.iloc[-1] < 45:
            trend_state = 'downtrend'
        else:
            trend_state = 'ranging'

        # 波动率状态
        if trend_strength.iloc[-1] > 70:
            volatility_state = 'high'
        elif trend_strength.iloc[-1] > 30:
            volatility_state = 'normal'
        else:
            volatility_state = 'low'

        # 综合市场状态
        if channel_state == 'squeeze':
            market_state = 'consolidation'
        elif trend_state in ['strong_uptrend', 'strong_downtrend'] and volatility_state == 'high':
            market_state = 'trending_volatile'
        elif trend_state in ['uptrend', 'downtrend'] and volatility_state == 'normal':
            market_state = 'trending_stable'
        else:
            market_state = 'ranging_stable'

        return {
            'channel_state': channel_state,
            'trend_state': trend_state,
            'volatility_state': volatility_state,
            'market_state': market_state,
            'momentum_bias': 'bullish' if channel_slope.iloc[-1] > 0 else 'bearish'
        }

    def generate_advanced_signals(self, keltner_data: Dict[str, pd.Series], mtf_analysis: Dict[str, Any]) -> pd.Series:
        """生成高级交易信号"""
        signals = pd.Series(0, index=keltner_data['upper_band'].index)

        # 基础信号
        base_signals = keltner_data['signals']

        # 多时间框架确认
        mtf_confirmed = (
            (mtf_analysis['consensus'] in ['bullish', 'bearish']) &
            (base_signals != 0)
        )

        # 挤压突破确认
        squeeze_breakout = (
            (keltner_data['squeeze_condition'] == 0) &
            (keltner_data['squeeze_condition'].shift(1) == 1)
        )

        # 趋势强度过滤
        strong_trend = keltner_data['trend_strength'] > 40

        # 高级买入信号
        advanced_buy = (
            (base_signals > 0) &
            (mtf_confirmed | squeeze_breakout) &
            strong_trend
        )

        # 高级卖出信号
        advanced_sell = (
            (base_signals < 0) &
            (mtf_confirmed | squeeze_breakout) &
            strong_trend
        )

        signals[advanced_buy] = base_signals[advanced_buy] + 1  # 增强信号强度
        signals[advanced_sell] = base_signals[advanced_sell] - 1  # 增强信号强度

        # 确保基础信号被保留
        signals[(base_signals != 0) & ~(advanced_buy | advanced_sell)] = base_signals

        return signals

    def calculate_risk_metrics(self, keltner_data: Dict[str, pd.Series], data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """计算风险管理指标"""
        close = data['close']
        upper_band = keltner_data['upper_band']
        lower_band = keltner_data['lower_band']
        atr = keltner_data['atr']

        # 计算止损位
        if close.iloc[-1] > keltner_data['middle_band'].iloc[-1]:
            # 上升趋势，止损位在下轨或中轨下方
            stop_loss = max(lower_band.iloc[-1], keltner_data['middle_band'].iloc[-1] - 2 * atr.iloc[-1])
        else:
            # 下降趋势，止损位在上轨或中轨上方
            stop_loss = min(upper_band.iloc[-1], keltner_data['middle_band'].iloc[-1] + 2 * atr.iloc[-1])

        # 计算目标位
        channel_height = upper_band.iloc[-1] - lower_band.iloc[-1]
        if close.iloc[-1] > keltner_data['middle_band'].iloc[-1]:
            # 上升趋势，目标位在上轨上方
            take_profit = upper_band.iloc[-1] + 0.5 * channel_height
        else:
            # 下降趋势，目标位在下轨下方
            take_profit = lower_band.iloc[-1] - 0.5 * channel_height

        # 计算风险回报比
        risk = abs(close.iloc[-1] - stop_loss)
        reward = abs(take_profit - close.iloc[-1])
        risk_reward_ratio = reward / risk if risk > 0 else 0

        # 计算波动率调整后的仓位大小
        volatility_adjusted_size = 1.0 / (atr.iloc[-1] / close.iloc[-1] * 100)
        volatility_adjusted_size = min(volatility_adjusted_size, 2.0)  # 最大2倍杠杆

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk_reward_ratio': risk_reward_ratio,
            'volatility_adjusted_size': volatility_adjusted_size,
            'atr_percent': (atr.iloc[-1] / close.iloc[-1]) * 100,
            'channel_height_percent': (channel_height / close.iloc[-1]) * 100
        }

    def get_trading_recommendations(self, analysis: Dict[str, Any]) -> pd.DataFrame:
        """获取交易建议"""
        recommendations = pd.DataFrame(index=analysis['signals'].index)

        # 信号描述
        recommendations['signal'] = analysis['signals']
        recommendations['signal_description'] = analysis['signals'].map({
            4: '极强买入 - 挤压突破确认',
            3: '强买入 - 多时间框架确认',
            2: '买入 - 趋势跟随',
            1: '弱买入 - 回撤或反转',
            0: '持有 - 观望',
            -1: '弱卖出 - 回撤或反转',
            -2: '卖出 - 趋势跟随',
            -3: '强卖出 - 多时间框架确认',
            -4: '极强卖出 - 挤压突破确认'
        })

        # 市场状态
        recommendations['market_state'] = analysis['market_state']['market_state']
        recommendations['channel_state'] = analysis['market_state']['channel_state']

        # 多时间框架共识
        recommendations['mtf_consensus'] = analysis['mtf_analysis']['consensus']

        # 仓位建议
        recommendations['position_size'] = analysis['signals'].map({
            4: 0.8,   # 极强买入：80%仓位
            3: 0.6,   # 强买入：60%仓位
            2: 0.4,   # 买入：40%仓位
            1: 0.2,   # 弱买入：20%仓位
            0: 0.1,   # 持有：10%仓位
            -1: 0.1,  # 弱卖出：10%仓位
            -2: 0.2,  # 卖出：20%仓位
            -3: 0.4,  # 强卖出：40%仓位
            -4: 0.6   # 极强卖出：60%仓位
        })

        # 波动率调整
        volatility_adjustment = 1.0 / (1 + analysis['risk_metrics']['atr_percent'] / 10)
        recommendations['position_size'] *= volatility_adjustment

        # 止损止盈建议
        recommendations['stop_loss'] = analysis['risk_metrics']['stop_loss']
        recommendations['take_profit'] = analysis['risk_metrics']['take_profit']
        recommendations['risk_reward_ratio'] = analysis['risk_metrics']['risk_reward_ratio']

        # 支撑阻力位
        recommendations['support_level'] = analysis['support_levels']
        recommendations['resistance_level'] = analysis['resistance_levels']

        # 风险指标
        recommendations['atr_percent'] = analysis['risk_metrics']['atr_percent']
        recommendations['volatility_adjusted_size'] = analysis['risk_metrics']['volatility_adjusted_size']

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    n = len(dates)

    # 生成OHLCV数据
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high_prices = close_prices + np.random.rand(n) * 2
    low_prices = close_prices - np.random.rand(n) * 2
    open_prices = close_prices + np.random.randn(n) * 0.3
    volumes = np.random.randint(1000000, 5000000, n)

    data = {
        'open': pd.Series(open_prices, index=dates),
        'high': pd.Series(high_prices, index=dates),
        'low': pd.Series(low_prices, index=dates),
        'close': pd.Series(close_prices, index=dates),
        'volume': pd.Series(volumes, index=dates)
    }

    # 创建自适应凯尔特纳通道指标
    keltner = KeltnerChannelAdaptive()

    # 计算指标
    results = keltner.calculate_keltner_channel(data)

    print("自适应凯尔特纳通道结果:")
    print(f"上轨: {results['upper_band'].iloc[-1]:.2f}")
    print(f"中轨: {results['middle_band'].iloc[-1]:.2f}")
    print(f"下轨: {results['lower_band'].iloc[-1]:.2f}")
    print(f"ATR: {results['atr'].iloc[-1]:.2f}")
    print(f"自适应乘数: {results['adaptive_multiplier'].iloc[-1]:.2f}")
    print(f"通道宽度: {results['channel_width'].iloc[-1]:.2f}")
    print(f"价格位置: {results['price_position'].iloc[-1]:.2f}")
    print(f"趋势强度: {results['trend_strength'].iloc[-1]:.2f}")
    print(f"挤压状态: {results['squeeze_condition'].iloc[-1]}")
    print(f"当前信号: {results['signals'].iloc[-1]}")

    # 获取综合分析
    comprehensive = keltner.get_comprehensive_analysis(data)
    print("\n综合分析结果:")
    print(f"市场状态: {comprehensive['market_state']['market_state']}")
    print(f"通道状态: {comprehensive['market_state']['channel_state']}")
    print(f"趋势状态: {comprehensive['market_state']['trend_state']}")
    print(f"波动率状态: {comprehensive['market_state']['volatility_state']}")
    print(f"多时间框架共识: {comprehensive['mtf_analysis']['consensus']}")
    print(f"高级信号: {comprehensive['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = keltner.get_trading_recommendations(comprehensive)
    print("\n最新交易建议:")
    print(recommendations.iloc[-1])