"""
Chande Momentum Oscillator (CMO)
================================

由Tushar Chande开发的动量振荡器，是RSI的改进版本。CMO通过
计算上涨日和下跌日的价格变化差异来衡量动量，具有更好的
趋势识别能力和更准确的超买超卖信号。

特点：
- 纯价格动量计算
- 更敏感的趋势识别
- 自适应超买超卖区域
- 背离检测能力
- 趋势强度评估
- 多时间框架分析

公式：
CMO = 100 * (Su - Sd) / (Su + Sd)
其中：
- Su = n日内上涨日的价格变化总和
- Sd = n日内下跌日的价格变化总和

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional


class ChandeMomentumOscillator:
    """
    Chande Momentum Oscillator - 钱德动量振荡器

    纯价格动量指标，通过计算上涨日和下跌日的价格变化差异
    来衡量市场动量和趋势强度。
    """

    def __init__(self, period: int = 20, smooth_period: int = 5,
                 overbought: float = 50, oversold: float = -50):
        """
        初始化钱德动量振荡器

        Args:
            period: CMO周期，默认20
            smooth_period: 平滑周期，默认5
            overbought: 超买水平，默认50
            oversold: 超卖水平，默认-50
        """
        self.period = period
        self.smooth_period = smooth_period
        self.overbought = overbought
        self.oversold = oversold
        self.name = f"Chande Momentum Oscillator ({period})"
        self.category = "momentum"

    def calculate_cmo(self, prices: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        计算钱德动量振荡器

        Args:
            prices: 价格序列

        Returns:
            包含所有指标的字典
        """
        prices_series = pd.Series(prices)

        # 计算价格变化
        price_changes = prices_series.diff()

        # 计算上涨日和下跌日的变化总和
        su, sd = self.calculate_su_sd(price_changes)

        # 计算CMO
        cmo_value = self.calculate_cmo_value(su, sd)

        # 平滑处理
        cmo_smooth = cmo_value.rolling(window=self.smooth_period).mean()

        # 计算CMO动量
        cmo_momentum = cmo_value.diff()

        # 计算CMO趋势强度
        trend_strength = self.calculate_trend_strength(cmo_value)

        # 计算动态超买超卖水平
        dynamic_levels = self.calculate_dynamic_levels(cmo_value, prices_series)

        # 生成交易信号
        signals = self.generate_signals(cmo_value, cmo_momentum, dynamic_levels)

        # 计算背离
        divergence = self.calculate_divergence(prices_series, cmo_value)

        # 计算CMO效率比率
        efficiency_ratio = self.calculate_efficiency_ratio(price_changes)

        return {
            'cmo_value': cmo_value,
            'cmo_smooth': cmo_smooth,
            'cmo_momentum': cmo_momentum,
            'trend_strength': trend_strength,
            'dynamic_overbought': dynamic_levels['overbought'],
            'dynamic_oversold': dynamic_levels['oversold'],
            'signals': signals,
            'divergence': divergence,
            'efficiency_ratio': efficiency_ratio,
            'su': su,
            'sd': sd
        }

    def calculate_su_sd(self, price_changes: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """
        计算上涨日和下跌日的价格变化总和

        Args:
            price_changes: 价格变化序列

        Returns:
            su, sd 元组
        """
        # 分离上涨和下跌
        gains = price_changes.where(price_changes > 0, 0)
        losses = -price_changes.where(price_changes < 0, 0)

        # 计算滚动总和
        su = gains.rolling(window=self.period).sum()
        sd = losses.rolling(window=self.period).sum()

        return su, sd

    def calculate_cmo_value(self, su: pd.Series, sd: pd.Series) -> pd.Series:
        """
        计算CMO值

        Args:
            su: 上涨日变化总和
            sd: 下跌日变化总和

        Returns:
            CMO值序列
        """
        # 计算CMO
        cmo = 100 * (su - sd) / (su + sd)

        # 处理除零情况
        cmo = cmo.replace([np.inf, -np.inf], 0)
        cmo = cmo.fillna(0)

        return cmo

    def calculate_trend_strength(self, cmo_value: pd.Series) -> pd.Series:
        """
        计算趋势强度

        Args:
            cmo_value: CMO值序列

        Returns:
            趋势强度序列 (0-100)
        """
        # 计算CMO绝对值
        cmo_abs = cmo_value.abs()

        # 计算趋势强度
        trend_strength = cmo_abs

        # 平滑处理
        trend_strength = trend_strength.rolling(window=10).mean()

        return trend_strength.fillna(0)

    def calculate_dynamic_levels(self, cmo_value: pd.Series, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        计算动态超买超卖水平

        Args:
            cmo_value: CMO值序列
            prices: 价格序列

        Returns:
            动态水平字典
        """
        # 计算价格波动率
        volatility = prices.pct_change().rolling(window=20).std() * np.sqrt(252)

        # 根据波动率调整超买超卖水平
        vol_multiplier = 1 + volatility / volatility.rolling(window=50).mean()
        vol_multiplier = vol_multiplier.clip(0.5, 2.0)

        # 动态水平
        dynamic_overbought = self.overbought * vol_multiplier
        dynamic_oversold = self.oversold * vol_multiplier

        # 限制范围
        dynamic_overbought = dynamic_overbought.clip(30, 80)
        dynamic_oversold = dynamic_oversold.clip(-80, -30)

        return {
            'overbought': dynamic_overbought,
            'oversold': dynamic_oversold
        }

    def generate_signals(self, cmo_value: pd.Series, cmo_momentum: pd.Series,
                        dynamic_levels: Dict[str, pd.Series]) -> pd.Series:
        """
        生成交易信号

        Args:
            cmo_value: CMO值序列
            cmo_momentum: CMO动量序列
            dynamic_levels: 动态水平

        Returns:
            信号序列 (2=强买入, 1=买入, 0=持有, -1=卖出, -2=强卖出)
        """
        signals = pd.Series(0, index=cmo_value.index)

        # 获取动态水平
        overbought = dynamic_levels['overbought']
        oversold = dynamic_levels['oversold']

        # 零线穿越
        cmo_above_zero = cmo_value > 0
        cmo_cross_above = cmo_above_zero & ~cmo_above_zero.shift(1)
        cmo_cross_below = ~cmo_above_zero & cmo_above_zero.shift(1)

        # 动量转正/负
        momentum_positive = cmo_momentum > 0
        momentum_negative = cmo_momentum < 0

        # 超买超卖区域
        overbought_zone = cmo_value > overbought
        oversold_zone = cmo_value < oversold

        # 强买入信号：从超卖区域反弹且动量转正
        strong_buy = (
            (cmo_value > oversold) &
            (cmo_value.shift(1) <= oversold.shift(1)) &
            momentum_positive &
            cmo_above_zero
        )

        # 买入信号：上穿零线或动量转正
        buy = (
            cmo_cross_above |
            (momentum_positive & ~momentum_positive.shift(1) & cmo_above_zero)
        )

        # 强卖出信号：从超买区域回落且动量转负
        strong_sell = (
            (cmo_value < overbought) &
            (cmo_value.shift(1) >= overbought.shift(1)) &
            momentum_negative &
            ~cmo_above_zero
        )

        # 卖出信号：下穿零线或动量转负
        sell = (
            cmo_cross_below |
            (momentum_negative & ~momentum_negative.shift(1) & ~cmo_above_zero)
        )

        signals[strong_buy] = 2
        signals[buy] = 1
        signals[strong_sell] = -2
        signals[sell] = -1

        return signals

    def calculate_divergence(self, prices: pd.Series, cmo_value: pd.Series) -> pd.Series:
        """
        计算背离

        Args:
            prices: 价格序列
            cmo_value: CMO值序列

        Returns:
            背离标记序列 (1=多头背离, -1=空头背离, 0=无背离)
        """
        divergence = pd.Series(0, index=prices.index)

        # 识别价格和CMO的极值点
        for i in range(30, len(prices)):
            # 多头背离检查
            if (self.is_lower_low(prices, i) and
                not self.is_lower_low(cmo_value, i) and
                cmo_value.iloc[i] < -20):
                divergence.iloc[i] = 1

            # 空头背离检查
            elif (self.is_higher_high(prices, i) and
                  not self.is_higher_high(cmo_value, i) and
                  cmo_value.iloc[i] > 20):
                divergence.iloc[i] = -1

        return divergence

    def calculate_efficiency_ratio(self, price_changes: pd.Series) -> pd.Series:
        """
        计算效率比率

        Args:
            price_changes: 价格变化序列

        Returns:
            效率比率序列
        """
        # 计算净价格变化
        net_change = price_changes.rolling(window=self.period).sum()

        # 计算总价格变化
        total_change = price_changes.abs().rolling(window=self.period).sum()

        # 计算效率比率
        efficiency_ratio = abs(net_change) / (total_change + 1e-8)

        return efficiency_ratio.fillna(0)

    def is_lower_low(self, series: pd.Series, index: int, lookback: int = 10) -> bool:
        """检查是否为更低低点"""
        if index < lookback:
            return False
        recent_values = series.iloc[index-lookback:index+1]
        return series.iloc[index] == recent_values.min()

    def is_higher_high(self, series: pd.Series, index: int, lookback: int = 10) -> bool:
        """检查是否为更高高点"""
        if index < lookback:
            return False
        recent_values = series.iloc[index-lookback:index+1]
        return series.iloc[index] == recent_values.max()

    def get_comprehensive_analysis(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        获取综合分析结果

        Args:
            data: 包含OHLCV数据的字典

        Returns:
            综合分析结果
        """
        prices = data['close']

        # 计算CMO
        cmo_data = self.calculate_cmo(prices)

        # 计算多时间框架分析
        mtf_analysis = self.calculate_mtf_analysis(prices)

        # 计算市场状态
        market_state = self.analyze_market_state(cmo_data)

        # 生成高级信号
        advanced_signals = self.generate_advanced_signals(cmo_data, mtf_analysis)

        # 计算动量指标
        momentum_metrics = self.calculate_momentum_metrics(cmo_data)

        return {
            'cmo_value': cmo_data['cmo_value'],
            'cmo_smooth': cmo_data['cmo_smooth'],
            'cmo_momentum': cmo_data['cmo_momentum'],
            'trend_strength': cmo_data['trend_strength'],
            'dynamic_levels': cmo_data['dynamic_overbought'],
            'signals': advanced_signals,
            'divergence': cmo_data['divergence'],
            'efficiency_ratio': cmo_data['efficiency_ratio'],
            'mtf_analysis': mtf_analysis,
            'market_state': market_state,
            'momentum_metrics': momentum_metrics
        }

    def calculate_mtf_analysis(self, prices: pd.Series) -> Dict[str, Any]:
        """
        计算多时间框架分析

        Args:
            prices: 价格序列

        Returns:
            多时间框架分析结果
        """
        # 不同周期的CMO
        periods = [10, 20, 30]
        mtf_results = {}

        for period in periods:
            # 临时修改参数
            original_period = self.period
            self.period = period

            # 计算该周期的CMO
            temp_data = self.calculate_cmo(prices)

            mtf_results[f'period_{period}'] = {
                'cmo': temp_data['cmo_value'],
                'trend_strength': temp_data['trend_strength'],
                'signal': 'bullish' if temp_data['cmo_value'].iloc[-1] > 0 else 'bearish'
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

    def analyze_market_state(self, cmo_data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        分析市场状态

        Args:
            cmo_data: CMO数据

        Returns:
            市场状态分析
        """
        cmo_value = cmo_data['cmo_value']
        trend_strength = cmo_data['trend_strength']

        # 计算CMO状态
        if cmo_value.iloc[-1] > 50:
            cmo_state = 'strong_bullish'
        elif cmo_value.iloc[-1] > 0:
            cmo_state = 'bullish'
        elif cmo_value.iloc[-1] > -50:
            cmo_state = 'bearish'
        else:
            cmo_state = 'strong_bearish'

        # 计算趋势状态
        if trend_strength.iloc[-1] > 70:
            trend_state = 'strong_trend'
        elif trend_strength.iloc[-1] > 40:
            trend_state = 'moderate_trend'
        else:
            trend_state = 'weak_trend'

        # 综合市场状态
        if cmo_state in ['strong_bullish', 'strong_bearish'] and trend_state == 'strong_trend':
            market_state = 'trending_strongly'
        elif cmo_state in ['bullish', 'bearish'] and trend_state in ['moderate_trend', 'strong_trend']:
            market_state = 'trending'
        else:
            market_state = 'ranging'

        return {
            'cmo_state': cmo_state,
            'trend_state': trend_state,
            'market_state': market_state,
            'momentum_bias': 'bullish' if cmo_value.iloc[-1] > 0 else 'bearish'
        }

    def generate_advanced_signals(self, cmo_data: Dict[str, pd.Series],
                                 mtf_analysis: Dict[str, Any]) -> pd.Series:
        """
        生成高级交易信号

        Args:
            cmo_data: CMO数据
            mtf_analysis: 多时间框架分析

        Returns:
            高级信号序列
        """
        signals = pd.Series(0, index=cmo_data['cmo_value'].index)

        # 基础信号
        base_signals = cmo_data['signals']

        # 多时间框架确认
        mtf_confirmed = (
            (mtf_analysis['consensus'] in ['bullish', 'bearish']) &
            (base_signals != 0)
        )

        # 背离确认
        has_divergence = cmo_data['divergence'] != 0

        # 趋势强度过滤
        strong_trend = cmo_data['trend_strength'] > 50

        # 高级买入信号
        advanced_buy = (
            (base_signals > 0) &
            (mtf_confirmed | has_divergence) &
            strong_trend
        )

        # 高级卖出信号
        advanced_sell = (
            (base_signals < 0) &
            (mtf_confirmed | has_divergence) &
            strong_trend
        )

        # 背离信号
        divergence_buy = (cmo_data['divergence'] == 1)
        divergence_sell = (cmo_data['divergence'] == -1)

        signals[advanced_buy] = 2
        signals[advanced_sell] = -2
        signals[divergence_buy] = 1
        signals[divergence_sell] = -1

        return signals

    def calculate_momentum_metrics(self, cmo_data: Dict[str, pd.Series]) -> Dict[str, float]:
        """
        计算动量指标

        Args:
            cmo_data: CMO数据

        Returns:
            动量指标字典
        """
        cmo_value = cmo_data['cmo_value']
        cmo_momentum = cmo_data['cmo_momentum']
        efficiency_ratio = cmo_data['efficiency_ratio']

        # 计算动量强度
        momentum_strength = np.sqrt(cmo_momentum.iloc[-1]**2 + cmo_value.iloc[-1]**2)

        # 计算动量方向
        momentum_direction = 'bullish' if (cmo_momentum.iloc[-1] + cmo_value.iloc[-1]) > 0 else 'bearish'

        # 计算效率评分
        efficiency_score = efficiency_ratio.iloc[-1] * 100

        return {
            'momentum_strength': momentum_strength,
            'momentum_direction': momentum_direction,
            'efficiency_score': efficiency_score,
            'momentum_quality': 'high' if efficiency_score > 70 else 'medium' if efficiency_score > 40 else 'low'
        }

    def get_trading_recommendations(self, analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            analysis: 综合分析结果

        Returns:
            交易建议DataFrame
        """
        recommendations = pd.DataFrame(index=analysis['cmo_value'].index)

        # 信号描述
        recommendations['signal'] = analysis['signals']
        recommendations['signal_description'] = analysis['signals'].map({
            2: '强烈买入 - 趋势确认',
            1: '买入 - 背离或动量转强',
            0: '持有 - 观望',
            -1: '卖出 - 背离或动量转弱',
            -2: '强烈卖出 - 趋势确认'
        })

        # 市场状态
        recommendations['market_state'] = analysis['market_state']['market_state']
        recommendations['cmo_state'] = analysis['market_state']['cmo_state']

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

        # 效率调整
        efficiency_adjustment = analysis['momentum_metrics']['efficiency_score'] / 100
        recommendations['position_size'] *= efficiency_adjustment

        # 止损止盈建议
        current_price = analysis['cmo_value'].index.to_series().map(lambda x: analysis['cmo_value'].loc[x])
        recommendations['stop_loss'] = current_price * 0.95
        recommendations['take_profit'] = current_price * 1.08

        # 动量指标
        recommendations['momentum_direction'] = analysis['momentum_metrics']['momentum_direction']
        recommendations['momentum_strength'] = analysis['momentum_metrics']['momentum_strength']
        recommendations['efficiency_score'] = analysis['momentum_metrics']['efficiency_score']

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)

    data = {'close': pd.Series(prices, index=dates)}

    # 创建钱德动量振荡器指标
    cmo = ChandeMomentumOscillator()

    # 计算指标
    results = cmo.calculate_cmo(data['close'])

    print("Chande Momentum Oscillator 结果:")
    print(f"CMO值: {results['cmo_value'].iloc[-1]:.2f}")
    print(f"平滑CMO: {results['cmo_smooth'].iloc[-1]:.2f}")
    print(f"CMO动量: {results['cmo_momentum'].iloc[-1]:.4f}")
    print(f"趋势强度: {results['trend_strength'].iloc[-1]:.2f}")
    print(f"动态超买: {results['dynamic_overbought'].iloc[-1]:.2f}")
    print(f"动态超卖: {results['dynamic_oversold'].iloc[-1]:.2f}")
    print(f"效率比率: {results['efficiency_ratio'].iloc[-1]:.4f}")
    print(f"当前信号: {results['signals'].iloc[-1]}")

    # 获取综合分析
    comprehensive = cmo.get_comprehensive_analysis(data)
    print("\n综合分析结果:")
    print(f"市场状态: {comprehensive['market_state']['market_state']}")
    print(f"CMO状态: {comprehensive['market_state']['cmo_state']}")
    print(f"动量方向: {comprehensive['momentum_metrics']['momentum_direction']}")
    print(f"动量强度: {comprehensive['momentum_metrics']['momentum_strength']:.2f}")
    print(f"效率评分: {comprehensive['momentum_metrics']['efficiency_score']:.2f}")
    print(f"多时间框架共识: {comprehensive['mtf_analysis']['consensus']}")
    print(f"高级信号: {comprehensive['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = cmo.get_trading_recommendations(comprehensive)
    print("\n最新交易建议:")
    print(recommendations.iloc[-1])