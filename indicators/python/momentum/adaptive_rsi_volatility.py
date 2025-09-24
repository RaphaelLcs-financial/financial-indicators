"""
Adaptive RSI with Volatility Bands
====================================

自适应RSI结合波动率带，根据市场波动性动态调整RSI参数，
并基于历史波动率生成动态的超买超卖区域。这种改进版的RSI
能够更好地适应不同市场环境，提供更准确的交易信号。

特点：
- 自适应RSI周期调整
- 动态超买超卖区域
- 波动率带分析
- 多时间框架确认
- 趋势过滤机制
- 信号强度评估

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional


class AdaptiveRSIVolatility:
    """
    Adaptive RSI with Volatility Bands - 自适应RSI波动率带

    根据市场波动性动态调整RSI参数，生成动态的超买超卖区域，
    提供更准确的交易信号。
    """

    def __init__(self, base_period: int = 14, volatility_period: int = 20,
                 band_multiplier: float = 1.5, adaptation_speed: float = 0.1):
        """
        初始化自适应RSI波动率带

        Args:
            base_period: 基础RSI周期，默认14
            volatility_period: 波动率计算周期，默认20
            band_multiplier: 带宽乘数，默认1.5
            adaptation_speed: 适应速度，默认0.1
        """
        self.base_period = base_period
        self.volatility_period = volatility_period
        self.band_multiplier = band_multiplier
        self.adaptation_speed = adaptation_speed
        self.name = f"Adaptive RSI Volatility ({base_period})"
        self.category = "momentum"

    def calculate_adaptive_rsi(self, prices: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        计算自适应RSI波动率带

        Args:
            prices: 价格序列

        Returns:
            包含所有指标的字典
        """
        prices_series = pd.Series(prices)

        # 计算市场波动率
        volatility = self.calculate_volatility(prices_series)

        # 计算自适应RSI周期
        adaptive_period = self.calculate_adaptive_period(volatility)

        # 计算自适应RSI
        adaptive_rsi = self.calculate_rsi_with_period(prices_series, adaptive_period)

        # 计算动态超买超卖区域
        dynamic_levels = self.calculate_dynamic_levels(adaptive_rsi, volatility)

        # 计算波动率带
        volatility_bands = self.calculate_volatility_bands(adaptive_rsi, volatility)

        # 计算RSI动量
        rsi_momentum = self.calculate_rsi_momentum(adaptive_rsi)

        # 计算趋势过滤
        trend_filter = self.calculate_trend_filter(prices_series)

        # 生成交易信号
        signals = self.generate_signals(
            adaptive_rsi, dynamic_levels, volatility_bands,
            rsi_momentum, trend_filter
        )

        # 计算信号强度
        signal_strength = self.calculate_signal_strength(
            adaptive_rsi, dynamic_levels, volatility_bands
        )

        # 识别背离
        divergence = self.calculate_divergence(prices_series, adaptive_rsi)

        return {
            'adaptive_rsi': adaptive_rsi,
            'adaptive_period': adaptive_period,
            'volatility': volatility,
            'dynamic_overbought': dynamic_levels['overbought'],
            'dynamic_oversold': dynamic_levels['oversold'],
            'upper_band': volatility_bands['upper'],
            'lower_band': volatility_bands['lower'],
            'middle_band': volatility_bands['middle'],
            'rsi_momentum': rsi_momentum,
            'trend_filter': trend_filter,
            'signals': signals,
            'signal_strength': signal_strength,
            'divergence': divergence
        }

    def calculate_volatility(self, prices: pd.Series) -> pd.Series:
        """
        计算市场波动率

        Args:
            prices: 价格序列

        Returns:
            波动率序列
        """
        # 计算收益率
        returns = prices.pct_change().dropna()

        # 计算历史波动率
        historical_vol = returns.rolling(window=self.volatility_period).std() * np.sqrt(252)

        # 计算平均真实范围
        atr = self.calculate_atr(prices, prices, prices, self.volatility_period)
        atr_vol = atr / prices.rolling(window=self.volatility_period).mean()

        # 综合波动率
        volatility = (historical_vol + atr_vol) / 2

        return volatility.fillna(method='bfill')

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                     period: int) -> pd.Series:
        """
        计算ATR

        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            period: ATR周期

        Returns:
            ATR序列
        """
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def calculate_adaptive_period(self, volatility: pd.Series) -> pd.Series:
        """
        计算自适应RSI周期

        Args:
            volatility: 波动率序列

        Returns:
            自适应周期序列
        """
        # 根据波动率调整周期
        # 高波动率时使用较短周期，低波动率时使用较长周期

        # 标准化波动率
        vol_median = volatility.rolling(window=100).median()
        vol_ratio = volatility / vol_median

        # 计算自适应周期
        adaptive_period = self.base_period / (1 + vol_ratio * self.adaptation_speed)

        # 限制周期范围
        adaptive_period = adaptive_period.clip(5, 30)

        return adaptive_period.fillna(self.base_period)

    def calculate_rsi_with_period(self, prices: pd.Series, period: pd.Series) -> pd.Series:
        """
        使用动态周期计算RSI

        Args:
            prices: 价格序列
            period: 动态周期序列

        Returns:
            自适应RSI序列
        """
        rsi_values = pd.Series(0.0, index=prices.index)

        for i in range(self.base_period, len(prices)):
            current_period = int(period.iloc[i])

            if current_period > 0 and i >= current_period:
                # 计算价格变化
                delta = prices.iloc[i-current_period:i+1].diff()

                # 计算收益和损失
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)

                # 计算平均收益和损失
                avg_gain = gain.rolling(window=current_period).mean()
                avg_loss = loss.rolling(window=current_period).mean()

                # 计算RS和RSI
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))

                rsi_values.iloc[i] = rsi.iloc[-1] if len(rsi) > 0 else 50

        return rsi_values.fillna(method='ffill').fillna(50)

    def calculate_dynamic_levels(self, rsi: pd.Series, volatility: pd.Series) -> Dict[str, pd.Series]:
        """
        计算动态超买超卖水平

        Args:
            rsi: RSI序列
            volatility: 波动率序列

        Returns:
            动态水平字典
        """
        # 根据波动率调整超买超卖水平
        # 高波动率时扩大超买超卖区域，低波动率时收缩

        # 计算基础水平
        base_overbought = 70
        base_oversold = 30

        # 根据波动率调整
        vol_adjustment = (volatility - volatility.rolling(window=50).mean()) / volatility.rolling(window=50).std()
        vol_adjustment = vol_adjustment.clip(-2, 2)  # 限制调整范围

        # 动态调整
        dynamic_overbought = base_overbought + vol_adjustment * 5
        dynamic_oversold = base_oversold - vol_adjustment * 5

        # 限制范围
        dynamic_overbought = dynamic_overbought.clip(60, 85)
        dynamic_oversold = dynamic_oversold.clip(15, 40)

        return {
            'overbought': dynamic_overbought,
            'oversold': dynamic_oversold
        }

    def calculate_volatility_bands(self, rsi: pd.Series, volatility: pd.Series) -> Dict[str, pd.Series]:
        """
        计算RSI波动率带

        Args:
            rsi: RSI序列
            volatility: 波动率序列

        Returns:
            波动率带字典
        """
        # 计算RSI的移动平均
        middle_band = rsi.rolling(window=20).mean()

        # 计算带宽
        bandwidth = volatility * self.band_multiplier * 10

        # 计算上下带
        upper_band = middle_band + bandwidth
        lower_band = middle_band - bandwidth

        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }

    def calculate_rsi_momentum(self, rsi: pd.Series) -> pd.Series:
        """
        计算RSI动量

        Args:
            rsi: RSI序列

        Returns:
            RSI动量序列
        """
        # 计算RSI变化率
        rsi_change = rsi.diff()

        # 计算动量
        momentum = rsi_change.rolling(window=5).mean()

        return momentum.fillna(0)

    def calculate_trend_filter(self, prices: pd.Series) -> pd.Series:
        """
        计算趋势过滤

        Args:
            prices: 价格序列

        Returns:
            趋势过滤序列 (1=上升趋势, -1=下降趋势, 0=横盘)
        """
        # 计算短期和长期移动平均
        short_ma = prices.rolling(window=20).mean()
        long_ma = prices.rolling(window=50).mean()

        # 计算趋势方向
        trend_filter = pd.Series(0, index=prices.index)

        trend_filter[short_ma > long_ma] = 1    # 上升趋势
        trend_filter[short_ma < long_ma] = -1   # 下降趋势

        return trend_filter

    def generate_signals(self, rsi: pd.Series, dynamic_levels: Dict[str, pd.Series],
                        volatility_bands: Dict[str, pd.Series], rsi_momentum: pd.Series,
                        trend_filter: pd.Series) -> pd.Series:
        """
        生成交易信号

        Args:
            rsi: RSI序列
            dynamic_levels: 动态水平
            volatility_bands: 波动率带
            rsi_momentum: RSI动量
            trend_filter: 趋势过滤

        Returns:
            信号序列 (2=强买入, 1=买入, 0=持有, -1=卖出, -2=强卖出)
        """
        signals = pd.Series(0, index=rsi.index)

        # 获取动态水平
        overbought = dynamic_levels['overbought']
        oversold = dynamic_levels['oversold']

        # 获取波动率带
        upper_band = volatility_bands['upper']
        lower_band = volatility_bands['lower']

        # 强买入信号：RSI从超卖区域反弹且动量为正
        strong_buy = (
            (rsi > oversold) &
            (rsi.shift(1) <= oversold.shift(1)) &
            (rsi_momentum > 0) &
            (trend_filter >= 0) &
            (rsi < 50)
        )

        # 买入信号：RSI上穿下带或动量转正
        buy = (
            ((rsi > lower_band) & (rsi.shift(1) <= lower_band.shift(1))) |
            ((rsi_momentum > 0) & (rsi_momentum.shift(1) <= 0))
        ) & (trend_filter >= 0) & (rsi < 60)

        # 强卖出信号：RSI从超买区域回落且动量为负
        strong_sell = (
            (rsi < overbought) &
            (rsi.shift(1) >= overbought.shift(1)) &
            (rsi_momentum < 0) &
            (trend_filter <= 0) &
            (rsi > 50)
        )

        # 卖出信号：RSI下穿上带或动量转负
        sell = (
            ((rsi < upper_band) & (rsi.shift(1) >= upper_band.shift(1))) |
            ((rsi_momentum < 0) & (rsi_momentum.shift(1) >= 0))
        ) & (trend_filter <= 0) & (rsi > 40)

        signals[strong_buy] = 2
        signals[buy] = 1
        signals[strong_sell] = -2
        signals[sell] = -1

        return signals

    def calculate_signal_strength(self, rsi: pd.Series, dynamic_levels: Dict[str, pd.Series],
                                 volatility_bands: Dict[str, pd.Series]) -> pd.Series:
        """
        计算信号强度

        Args:
            rsi: RSI序列
            dynamic_levels: 动态水平
            volatility_bands: 波动率带

        Returns:
            信号强度序列 (0-100)
        """
        # 计算RSI在动态区域中的位置
        overbought = dynamic_levels['overbought']
        oversold = dynamic_levels['oversold']

        # 计算极值强度
        extreme_strength = 0
        if rsi.iloc[-1] > overbought.iloc[-1]:
            extreme_strength = (rsi.iloc[-1] - overbought.iloc[-1]) / (100 - overbought.iloc[-1]) * 100
        elif rsi.iloc[-1] < oversold.iloc[-1]:
            extreme_strength = (oversold.iloc[-1] - rsi.iloc[-1]) / oversold.iloc[-1] * 100

        # 计算带突破强度
        upper_band = volatility_bands['upper']
        lower_band = volatility_bands['lower']

        if rsi.iloc[-1] > upper_band.iloc[-1]:
            band_strength = (rsi.iloc[-1] - upper_band.iloc[-1]) / (100 - upper_band.iloc[-1]) * 100
        elif rsi.iloc[-1] < lower_band.iloc[-1]:
            band_strength = (lower_band.iloc[-1] - rsi.iloc[-1]) / lower_band.iloc[-1] * 100
        else:
            band_strength = 0

        # 综合信号强度
        signal_strength = max(extreme_strength, band_strength)

        return pd.Series(min(100, signal_strength), index=rsi.index).fillna(0)

    def calculate_divergence(self, prices: pd.Series, rsi: pd.Series) -> pd.Series:
        """
        计算背离

        Args:
            prices: 价格序列
            rsi: RSI序列

        Returns:
            背离标记序列 (1=多头背离, -1=空头背离, 0=无背离)
        """
        divergence = pd.Series(0, index=prices.index)

        # 识别价格和RSI的极值点
        for i in range(20, len(prices)):
            # 多头背离检查
            if (prices.iloc[i] < prices.iloc[i-5] and
                rsi.iloc[i] > rsi.iloc[i-5] and
                rsi.iloc[i] < 40):
                divergence.iloc[i] = 1

            # 空头背离检查
            elif (prices.iloc[i] > prices.iloc[i-5] and
                  rsi.iloc[i] < rsi.iloc[i-5] and
                  rsi.iloc[i] > 60):
                divergence.iloc[i] = -1

        return divergence

    def get_comprehensive_analysis(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        获取综合分析结果

        Args:
            data: 包含OHLCV数据的字典

        Returns:
            综合分析结果
        """
        prices = data['close']

        # 计算自适应RSI
        adaptive_rsi_data = self.calculate_adaptive_rsi(prices)

        # 计算多时间框架分析
        mtf_analysis = self.calculate_mtf_analysis(prices)

        # 计算市场状态
        market_state = self.analyze_market_state(prices, adaptive_rsi_data)

        # 生成高级信号
        advanced_signals = self.generate_advanced_signals(
            adaptive_rsi_data, mtf_analysis, market_state
        )

        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics(prices, adaptive_rsi_data)

        return {
            'adaptive_rsi': adaptive_rsi_data['adaptive_rsi'],
            'adaptive_period': adaptive_rsi_data['adaptive_period'],
            'volatility': adaptive_rsi_data['volatility'],
            'dynamic_levels': adaptive_rsi_data['dynamic_overbought'],
            'volatility_bands': adaptive_rsi_data['upper_band'],
            'signals': advanced_signals,
            'signal_strength': adaptive_rsi_data['signal_strength'],
            'divergence': adaptive_rsi_data['divergence'],
            'mtf_analysis': mtf_analysis,
            'market_state': market_state,
            'risk_metrics': risk_metrics
        }

    def calculate_mtf_analysis(self, prices: pd.Series) -> Dict[str, Any]:
        """
        计算多时间框架分析

        Args:
            prices: 价格序列

        Returns:
            多时间框架分析结果
        """
        # 不同周期的分析
        periods = [10, 20, 30, 50]
        mtf_results = {}

        for period in periods:
            # 计算该周期的自适应RSI
            volatility = self.calculate_volatility(prices)
            adaptive_period = self.calculate_adaptive_period(volatility)
            rsi = self.calculate_rsi_with_period(prices, adaptive_period)

            mtf_results[f'period_{period}'] = {
                'rsi': rsi,
                'period': adaptive_period,
                'signal': 'bullish' if rsi.iloc[-1] > 50 else 'bearish'
            }

        # 计算多时间框架共识
        signals = [result['signal'] for result in mtf_results.values()]
        bullish_count = signals.count('bullish')
        bearish_count = signals.count('bearish')

        if bullish_count >= 3:
            consensus = 'strong_bullish'
        elif bullish_count >= 2:
            consensus = 'bullish'
        elif bearish_count >= 3:
            consensus = 'strong_bearish'
        elif bearish_count >= 2:
            consensus = 'bearish'
        else:
            consensus = 'neutral'

        return {
            'period_analysis': mtf_results,
            'consensus': consensus,
            'bullish_count': bullish_count,
            'bearish_count': bearish_count
        }

    def analyze_market_state(self, prices: pd.Series, rsi_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        分析市场状态

        Args:
            prices: 价格序列
            rsi_data: RSI数据

        Returns:
            市场状态分析
        """
        rsi = rsi_data['adaptive_rsi']
        volatility = rsi_data['volatility']

        # 计算趋势强度
        trend_strength = self.calculate_trend_strength(prices)

        # 计算波动率状态
        vol_median = volatility.rolling(window=50).median()
        vol_state = 'high' if volatility.iloc[-1] > vol_median.iloc[-1] * 1.2 else 'low'

        # 计算RSI状态
        rsi_state = 'overbought' if rsi.iloc[-1] > 70 else 'oversold' if rsi.iloc[-1] < 30 else 'normal'

        # 综合市场状态
        if trend_strength > 0.7 and vol_state == 'low':
            market_state = 'strong_trend'
        elif trend_strength > 0.5:
            market_state = 'trending'
        elif vol_state == 'high':
            market_state = 'volatile'
        else:
            market_state = 'ranging'

        return {
            'trend_strength': trend_strength,
            'volatility_state': vol_state,
            'rsi_state': rsi_state,
            'market_state': market_state
        }

    def calculate_trend_strength(self, prices: pd.Series) -> float:
        """
        计算趋势强度

        Args:
            prices: 价格序列

        Returns:
            趋势强度 (0-1)
        """
        # 计算线性回归斜率
        recent_prices = prices.iloc[-20:]
        x = np.arange(len(recent_prices))
        slope = np.polyfit(x, recent_prices, 1)[0]

        # 计算R平方
        y_pred = np.polyval(np.polyfit(x, recent_prices, 1), x)
        ss_res = np.sum((recent_prices - y_pred) ** 2)
        ss_tot = np.sum((recent_prices - np.mean(recent_prices)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # 趋势强度
        trend_strength = abs(slope) * r_squared * 100

        return min(1.0, trend_strength)

    def generate_advanced_signals(self, rsi_data: Dict[str, pd.Series],
                                 mtf_analysis: Dict[str, Any],
                                 market_state: Dict[str, Any]) -> pd.Series:
        """
        生成高级交易信号

        Args:
            rsi_data: RSI数据
            mtf_analysis: 多时间框架分析
            market_state: 市场状态

        Returns:
            高级信号序列
        """
        signals = pd.Series(0, index=rsi_data['adaptive_rsi'].index)

        # 基础信号
        base_signals = rsi_data['signals']

        # 多时间框架确认
        mtf_confirmed = (
            (mtf_analysis['consensus'] in ['strong_bullish', 'strong_bearish']) &
            (base_signals != 0)
        )

        # 市场状态过滤
        state_filtered = (
            (market_state['market_state'] in ['strong_trend', 'trending']) &
            (base_signals != 0)
        )

        # 信号强度过滤
        strength_filtered = rsi_data['signal_strength'] > 50

        # 背离确认
        divergence_confirmed = rsi_data['divergence'] != 0

        # 高级买入信号
        advanced_buy = (
            (base_signals > 0) &
            (mtf_confirmed | state_filtered) &
            strength_filtered
        )

        # 高级卖出信号
        advanced_sell = (
            (base_signals < 0) &
            (mtf_confirmed | state_filtered) &
            strength_filtered
        )

        # 背离信号
        divergence_buy = (rsi_data['divergence'] == 1) & (rsi_data['adaptive_rsi'] < 40)
        divergence_sell = (rsi_data['divergence'] == -1) & (rsi_data['adaptive_rsi'] > 60)

        signals[advanced_buy] = 2
        signals[advanced_sell] = -2
        signals[divergence_buy] = 1
        signals[divergence_sell] = -1

        return signals

    def calculate_risk_metrics(self, prices: pd.Series, rsi_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        计算风险指标

        Args:
            prices: 价格序列
            rsi_data: RSI数据

        Returns:
            风险指标字典
        """
        rsi = rsi_data['adaptive_rsi']
        volatility = rsi_data['volatility']

        # 计算波动率风险
        vol_risk = volatility.iloc[-1] / volatility.rolling(window=50).mean().iloc[-1]

        # 计算RSI极值风险
        if rsi.iloc[-1] > 80:
            rsi_risk = 0.8
        elif rsi.iloc[-1] < 20:
            rsi_risk = 0.2
        else:
            rsi_risk = 0.5

        # 计算综合风险评分
        risk_score = (vol_risk * 0.6 + rsi_risk * 0.4)

        return {
            'volatility_risk': vol_risk,
            'rsi_risk': rsi_risk,
            'risk_score': min(1.0, risk_score)
        }

    def get_trading_recommendations(self, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            analysis: 综合分析结果

        Returns:
            交易建议DataFrame
        """
        recommendations = pd.DataFrame(index=analysis['adaptive_rsi'].index)

        # 信号描述
        recommendations['signal'] = analysis['signals']
        recommendations['signal_description'] = analysis['signals'].map({
            2: '强烈买入 - 多时间框架确认',
            1: '买入 - 背离或动量转正',
            0: '持有 - 观望',
            -1: '卖出 - 背离或动量转负',
            -2: '强烈卖出 - 多时间框架确认'
        })

        # 市场状态
        recommendations['market_state'] = analysis['market_state']['market_state']
        recommendations['trend_strength'] = analysis['market_state']['trend_strength']

        # 多时间框架共识
        recommendations['mtf_consensus'] = analysis['mtf_analysis']['consensus']

        # 仓位建议
        recommendations['position_size'] = analysis['signals'].map({
            2: 0.6,   # 强买入：60%仓位
            1: 0.3,   # 买入：30%仓位
            0: 0.2,   # 持有：20%仓位
            -1: 0.1,  # 卖出：10%仓位
            -2: 0.0   # 强卖出：0仓位
        })

        # 风险调整
        risk_adjustment = 1 - analysis['risk_metrics']['risk_score']
        recommendations['position_size'] *= risk_adjustment

        # 止损止盈建议
        current_price = analysis['adaptive_rsi'].index.to_series().map(lambda x: analysis['adaptive_rsi'].loc[x])
        recommendations['stop_loss'] = current_price * 0.96
        recommendations['take_profit'] = current_price * 1.06

        # 信号强度
        recommendations['signal_strength'] = analysis['signal_strength']

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)

    data = {'close': pd.Series(prices, index=dates)}

    # 创建自适应RSI波动率带指标
    adaptive_rsi = AdaptiveRSIVolatility()

    # 计算指标
    results = adaptive_rsi.calculate_adaptive_rsi(data['close'])

    print("Adaptive RSI with Volatility Bands 结果:")
    print(f"自适应RSI: {results['adaptive_rsi'].iloc[-1]:.2f}")
    print(f"自适应周期: {results['adaptive_period'].iloc[-1]:.2f}")
    print(f"波动率: {results['volatility'].iloc[-1]:.4f}")
    print(f"动态超买: {results['dynamic_overbought'].iloc[-1]:.2f}")
    print(f"动态超卖: {results['dynamic_oversold'].iloc[-1]:.2f}")
    print(f"上轨: {results['upper_band'].iloc[-1]:.2f}")
    print(f"下轨: {results['lower_band'].iloc[-1]:.2f}")
    print(f"信号强度: {results['signal_strength'].iloc[-1]:.2f}")
    print(f"当前信号: {results['signals'].iloc[-1]}")

    # 获取综合分析
    comprehensive = adaptive_rsi.get_comprehensive_analysis(data)
    print("\n综合分析结果:")
    print(f"市场状态: {comprehensive['market_state']['market_state']}")
    print(f"趋势强度: {comprehensive['market_state']['trend_strength']:.2f}")
    print(f"多时间框架共识: {comprehensive['mtf_analysis']['consensus']}")
    print(f"风险评分: {comprehensive['risk_metrics']['risk_score']:.2f}")
    print(f"高级信号: {comprehensive['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = adaptive_rsi.get_trading_recommendations(comprehensive)
    print("\n最新交易建议:")
    print(recommendations.iloc[-1])