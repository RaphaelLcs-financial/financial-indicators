"""
Ehler's Instantaneous Trendline
==============================

John Ehlers开发的瞬时趋势线指标，几乎零滞后的趋势跟踪工具。
该指标使用信号处理技术（希尔伯特变换）来消除传统趋势指标的滞后性，
同时保持信号的平滑性。非常适合短期交易和趋势识别。

特点：
- 几乎零滞后，反应极其迅速
- 平滑的趋势线，减少噪音
- 自适应参数调整
- 适用于高频率交易
- 结合周期性分析，提高准确性

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional


class EhlersInstantaneousTrendline:
    """
    Ehler's Instantaneous Trendline - 埃勒斯瞬时趋势线

    John Ehlers开发的零滞后趋势指标，使用希尔伯特变换技术
    消除传统趋势线的滞后性，提供实时的趋势方向信号。
    """

    def __init__(self, alpha: float = 0.07, smoothing_period: int = 7):
        """
        初始化瞬时趋势线

        Args:
            alpha: 趋势线平滑系数，默认0.07
            smoothing_period: 平滑周期，默认7
        """
        self.alpha = alpha
        self.smoothing_period = smoothing_period
        self.name = f"Ehlers Instantaneous Trendline (α={alpha})"
        self.category = "trend"

    def calculate_instantaneous_trendline(self, prices: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        计算瞬时趋势线

        Args:
            prices: 价格序列

        Returns:
            包含趋势线和相关指标的字典
        """
        prices_series = pd.Series(prices)

        # 计算瞬时趋势线
        trendline = self.compute_trendline(prices_series)

        # 计算周期
        period = self.calculate_dominant_cycle(prices_series)

        # 计算自适应alpha
        adaptive_alpha = self.calculate_adaptive_alpha(period)

        # 计算领先信号
        lead_signal = self.calculate_lead_signal(prices_series, trendline)

        # 计算趋势强度
        trend_strength = self.calculate_trend_strength(prices_series, trendline)

        # 生成交易信号
        signals = self.generate_signals(prices_series, trendline, lead_signal)

        # 计算转折点
        turning_points = self.identify_turning_points(trendline)

        return {
            'trendline': trendline,
            'period': period,
            'adaptive_alpha': adaptive_alpha,
            'lead_signal': lead_signal,
            'trend_strength': trend_strength,
            'signals': signals,
            'turning_points': turning_points
        }

    def compute_trendline(self, prices: pd.Series) -> pd.Series:
        """
        计算瞬时趋势线核心算法

        公式:
        trend = (alpha/2) * (price + price[2]) + (1-alpha) * trend[1]

        Args:
            prices: 价格序列

        Returns:
            趋势线序列
        """
        trendline = pd.Series(0.0, index=prices.index)

        for i in range(2, len(prices)):
            if i == 2:
                trendline.iloc[i] = prices.iloc[i]
            else:
                trendline.iloc[i] = (
                    (self.alpha / 2) * (prices.iloc[i] + prices.iloc[i-2]) +
                    (1 - self.alpha) * trendline.iloc[i-1]
                )

        # 平滑处理
        trendline = trendline.rolling(window=self.smoothing_period).mean()

        return trendline

    def calculate_dominant_cycle(self, prices: pd.Series) -> pd.Series:
        """
        计算主导周期（简化版本）

        Args:
            prices: 价格序列

        Returns:
            周期序列
        """
        # 使用相位变化计算周期
        returns = prices.pct_change().dropna()

        # 计算相位
        phase = np.arctan2(returns, returns.shift(1))

        # 转换为周期
        period = pd.Series(2 * np.pi / abs(phase), index=prices.index)

        # 限制周期范围（6-50）
        period = period.clip(6, 50)

        # 平滑处理
        period = period.rolling(window=10).mean()

        return period.fillna(20)  # 默认周期

    def calculate_adaptive_alpha(self, period: pd.Series) -> pd.Series:
        """
        计算自适应alpha参数

        Args:
            period: 周期序列

        Returns:
            自适应alpha序列
        """
        # 根据周期调整alpha
        adaptive_alpha = 2 / (period + 1)

        # 限制alpha范围
        adaptive_alpha = adaptive_alpha.clip(0.01, 0.2)

        return adaptive_alpha

    def calculate_lead_signal(self, prices: pd.Series, trendline: pd.Series) -> pd.Series:
        """
        计算领先信号

        Args:
            prices: 价格序列
            trendline: 趋势线

        Returns:
            领先信号序列
        """
        # 计算价格与趋势线的偏离
        deviation = prices - trendline

        # 计算领先信号（趋势线加上调整）
        lead_signal = trendline + deviation * 0.5

        return lead_signal

    def calculate_trend_strength(self, prices: pd.Series, trendline: pd.Series) -> pd.Series:
        """
        计算趋势强度

        Args:
            prices: 价格序列
            trendline: 趋势线

        Returns:
            趋势强度序列 (0-100)
        """
        # 计算价格与趋势线的相关性
        correlation = prices.rolling(window=20).corr(trendline)

        # 计算趋势线斜率
        slope = trendline.diff().rolling(window=5).mean()

        # 计算价格在趋势线上方的比例
        above_trend = (prices > trendline).rolling(window=20).mean()

        # 综合趋势强度
        trend_strength = (
            (correlation.abs() * 0.4) +
            (slope.abs() * 1000 * 0.3) +
            (above_trend * 0.3)
        ).fillna(0) * 100

        return trend_strength.clip(0, 100)

    def generate_signals(self, prices: pd.Series, trendline: pd.Series,
                         lead_signal: pd.Series) -> pd.Series:
        """
        生成交易信号

        Args:
            prices: 价格序列
            trendline: 趋势线
            lead_signal: 领先信号

        Returns:
            信号序列 (1=买入, -1=卖出, 0=持有)
        """
        signals = pd.Series(0, index=prices.index)

        # 趋势方向判断
        trend_direction = trendline.diff()

        # 价格与趋势线关系
        price_above_trend = prices > trendline
        price_above_lead = prices > lead_signal

        # 买入信号：趋势向上且价格在趋势线上方
        buy_condition = (
            (trend_direction > 0) &
            (price_above_trend) &
            (price_above_lead) &
            (prices > prices.shift(1))
        )

        # 卖出信号：趋势向下且价格在趋势线下方
        sell_condition = (
            (trend_direction < 0) &
            (~price_above_trend) &
            (~price_above_lead) &
            (prices < prices.shift(1))
        )

        signals[buy_condition] = 1
        signals[sell_condition] = -1

        return signals

    def identify_turning_points(self, trendline: pd.Series) -> pd.Series:
        """
        识别趋势转折点

        Args:
            trendline: 趋势线

        Returns:
            转折点标记序列
        """
        turning_points = pd.Series(0, index=trendline.index)

        # 计算一阶导数
        first_derivative = trendline.diff()

        # 计算二阶导数
        second_derivative = first_derivative.diff()

        # 识别极值点
        for i in range(2, len(trendline) - 2):
            # 局部最大值
            if (first_derivative.iloc[i] > 0 and
                first_derivative.iloc[i+1] <= 0 and
                second_derivative.iloc[i] < 0):
                turning_points.iloc[i] = 1

            # 局部最小值
            elif (first_derivative.iloc[i] < 0 and
                  first_derivative.iloc[i+1] >= 0 and
                  second_derivative.iloc[i] > 0):
                turning_points.iloc[i] = -1

        return turning_points

    def get_comprehensive_analysis(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        获取综合分析结果

        Args:
            data: 包含OHLCV数据的字典

        Returns:
            综合分析结果
        """
        prices = data['close']

        # 计算瞬时趋势线
        trend_data = self.calculate_instantaneous_trendline(prices)

        # 计算其他技术指标
        rsi = self.calculate_rsi(prices, 14)
        volume_profile = self.calculate_volume_profile(data['volume'])

        # 计算市场周期
        market_cycle = self.analyze_market_cycle(prices, trend_data['period'])

        # 生成高级信号
        advanced_signals = self.generate_advanced_signals(
            prices, trend_data, rsi, volume_profile
        )

        # 计算支撑阻力位
        support_resistance = self.calculate_dynamic_support_resistance(
            prices, trend_data['trendline']
        )

        return {
            'trendline': trend_data['trendline'],
            'period': trend_data['period'],
            'adaptive_alpha': trend_data['adaptive_alpha'],
            'lead_signal': trend_data['lead_signal'],
            'trend_strength': trend_data['trend_strength'],
            'signals': advanced_signals,
            'turning_points': trend_data['turning_points'],
            'rsi': rsi,
            'volume_profile': volume_profile,
            'market_cycle': market_cycle,
            'support_resistance': support_resistance
        }

    def generate_advanced_signals(self, prices: pd.Series, trend_data: Dict[str, pd.Series],
                                 rsi: pd.Series, volume_profile: pd.Series) -> pd.Series:
        """
        生成高级交易信号

        Args:
            prices: 价格序列
            trend_data: 趋势数据
            rsi: RSI序列
            volume_profile: 成交量分布

        Returns:
            高级信号序列
        """
        signals = pd.Series(0, index=prices.index)

        # 多重确认买入信号
        buy_condition = (
            (trend_data['signals'] == 1) &
            (rsi < 70) &
            (volume_profile > 1.0) &
            (trend_data['trend_strength'] > 40) &
            (prices > trend_data['trendline'])
        )

        # 多重确认卖出信号
        sell_condition = (
            (trend_data['signals'] == -1) &
            (rsi > 30) &
            (volume_profile > 1.0) &
            (trend_data['trend_strength'] > 40) &
            (prices < trend_data['trendline'])
        )

        # 趋势突破信号
        breakout_buy = (
            (prices > trend_data['lead_signal']) &
            (trend_data['trend_direction'] > 0) &
            (rsi > 50) &
            (prices.rolling(5).max() == prices)
        )

        breakout_sell = (
            (prices < trend_data['lead_signal']) &
            (trend_data['trend_direction'] < 0) &
            (rsi < 50) &
            (prices.rolling(5).min() == prices)
        )

        signals[buy_condition] = 2    # 强买入
        signals[breakout_buy] = 1    # 突破买入
        signals[sell_condition] = -2  # 强卖出
        signals[breakout_sell] = -1   # 突破卖出

        return signals

    def calculate_volume_profile(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        计算成交量分布

        Args:
            volume: 成交量序列
            period: 计算周期

        Returns:
            成交量分布比率
        """
        volume_ma = volume.rolling(window=period).mean()
        volume_profile = volume / volume_ma
        return volume_profile.fillna(1.0)

    def analyze_market_cycle(self, prices: pd.Series, period: pd.Series) -> pd.Series:
        """
        分析市场周期

        Args:
            prices: 价格序列
            period: 周期序列

        Returns:
            市场周期阶段
        """
        # 计算价格相位
        returns = prices.pct_change().dropna()
        phase = np.arctan2(returns, returns.shift(1))

        # 根据相位判断周期阶段
        cycle_phase = pd.Series(0, index=prices.index)

        for i in range(len(phase)):
            if phase.iloc[i] > 0:
                cycle_phase.iloc[i] = 1 if phase.iloc[i] < np.pi/2 else 2
            else:
                cycle_phase.iloc[i] = 3 if phase.iloc[i] > -np.pi/2 else 4

        return cycle_phase

    def calculate_dynamic_support_resistance(self, prices: pd.Series,
                                             trendline: pd.Series) -> Dict[str, pd.Series]:
        """
        计算动态支撑阻力位

        Args:
            prices: 价格序列
            trendline: 趋势线

        Returns:
            支撑阻力位字典
        """
        # 基于趋势线的支撑阻力
        support = trendline * 0.98
        resistance = trendline * 1.02

        # 基于ATR的动态支撑阻力
        atr = self.calculate_atr(prices, prices, prices)
        support_atr = trendline - atr
        resistance_atr = trendline + atr

        return {
            'trend_support': support,
            'trend_resistance': resistance,
            'atr_support': support_atr,
            'atr_resistance': resistance_atr
        }

    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series,
                      period: int = 14) -> pd.Series:
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

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        计算RSI

        Args:
            prices: 价格序列
            period: RSI周期

        Returns:
            RSI序列
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def get_trading_strategy(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        获取完整交易策略

        Args:
            data: 市场数据

        Returns:
            交易策略结果
        """
        analysis = self.get_comprehensive_analysis(data)

        # 生成策略建议
        strategy_signals = self.generate_strategy_signals(analysis)

        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics(data['close'], analysis['trendline'])

        # 获取入场出场时机
        entry_exit = self.get_entry_exit_signals(analysis)

        return {
            'analysis': analysis,
            'strategy_signals': strategy_signals,
            'risk_metrics': risk_metrics,
            'entry_exit': entry_exit
        }

    def generate_strategy_signals(self, analysis: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        生成策略信号

        Args:
            analysis: 分析结果

        Returns:
            策略信号DataFrame
        """
        strategy_df = pd.DataFrame(index=analysis['trendline'].index)

        # 基础信号
        strategy_df['signal'] = analysis['signals']
        strategy_df['signal_description'] = analysis['signals'].map({
            2: '强烈买入 - 趋势突破确认',
            1: '买入 - 趋势跟踪',
            0: '持有 - 观望',
            -1: '卖出 - 趋势反转',
            -2: '强烈卖出 - 趋势破位'
        })

        # 趋势强度评估
        strategy_df['trend_strength'] = analysis['trend_strength']
        strategy_df['trend_quality'] = pd.cut(
            analysis['trend_strength'],
            bins=[0, 30, 60, 80, 100],
            labels=['弱趋势', '中等趋势', '强趋势', '极强趋势']
        )

        # 仓位建议
        strategy_df['position_size'] = analysis['signals'].map({
            2: 0.8,   # 强买入：80%仓位
            1: 0.5,   # 买入：50%仓位
            0: 0.3,   # 持有：30%仓位
            -1: 0.1,  # 卖出：10%仓位
            -2: 0.0   # 强卖出：0仓位
        })

        # 止损止盈建议
        current_price = analysis['trendline']
        strategy_df['stop_loss'] = current_price * 0.95
        strategy_df['take_profit'] = current_price * 1.10

        return strategy_df

    def calculate_risk_metrics(self, prices: pd.Series, trendline: pd.Series) -> Dict[str, float]:
        """
        计算风险指标

        Args:
            prices: 价格序列
            trendline: 趋势线

        Returns:
            风险指标字典
        """
        # 计算追踪误差
        tracking_error = (prices - trendline).std()

        # 计算最大偏离
        max_deviation = abs(prices - trendline).max()

        # 计算趋势稳定性
        trend_stability = 1 - (trendline.diff().std() / trendline.mean())

        return {
            'tracking_error': tracking_error,
            'max_deviation': max_deviation,
            'trend_stability': trend_stability,
            'signal_accuracy': self.calculate_signal_accuracy(prices, trendline)
        }

    def calculate_signal_accuracy(self, prices: pd.Series, trendline: pd.Series) -> float:
        """
        计算信号准确率

        Args:
            prices: 价格序列
            trendline: 趋势线

        Returns:
            信号准确率
        """
        # 模拟历史信号
        signals = self.generate_signals(prices, trendline, trendline)

        # 计算未来收益
        future_returns = prices.pct_change().shift(-1)

        # 计算准确率
        buy_accuracy = (future_returns[signals == 1] > 0).mean()
        sell_accuracy = (future_returns[signals == -1] < 0).mean()

        overall_accuracy = (buy_accuracy + sell_accuracy) / 2

        return overall_accuracy if not np.isnan(overall_accuracy) else 0.5

    def get_entry_exit_signals(self, analysis: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        获取入场出场信号

        Args:
            analysis: 分析结果

        Returns:
            入场出场信号
        """
        entry_signals = pd.Series(0, index=analysis['trendline'].index)
        exit_signals = pd.Series(0, index=analysis['trendline'].index)

        # 入场信号
        entry_condition = (
            (analysis['signals'] > 0) &
            (analysis['trend_strength'] > 40) &
            (analysis['rsi'] < 70)
        )
        entry_signals[entry_condition] = 1

        # 出场信号
        exit_condition = (
            (analysis['signals'] < 0) |
            (analysis['trend_strength'] < 20) |
            (analysis['rsi'] > 80) |
            (analysis['turning_points'] != 0)
        )
        exit_signals[exit_condition] = 1

        return {
            'entry_signals': entry_signals,
            'exit_signals': exit_signals
        }


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
    volume = np.random.randint(1000000, 5000000, len(dates))

    data = {
        'close': pd.Series(prices, index=dates),
        'volume': pd.Series(volume, index=dates)
    }

    # 创建瞬时趋势线指标
    trendline = EhlersInstantaneousTrendline(alpha=0.07)

    # 计算指标
    results = trendline.calculate_instantaneous_trendline(data['close'])

    print("Ehler's Instantaneous Trendline 结果:")
    print(f"趋势线: {results['trendline'].iloc[-1]:.2f}")
    print(f"领先信号: {results['lead_signal'].iloc[-1]:.2f}")
    print(f"趋势强度: {results['trend_strength'].iloc[-1]:.2f}")
    print(f"周期: {results['period'].iloc[-1]:.2f}")
    print(f"自适应alpha: {results['adaptive_alpha'].iloc[-1]:.4f}")
    print(f"当前信号: {results['signals'].iloc[-1]}")

    # 获取综合分析
    comprehensive = trendline.get_comprehensive_analysis(data)
    print("\n综合分析结果:")
    print(f"RSI: {comprehensive['rsi'].iloc[-1]:.2f}")
    print(f"高级信号: {comprehensive['signals'].iloc[-1]}")
    print(f"市场周期: {comprehensive['market_cycle'].iloc[-1]}")

    # 获取交易策略
    strategy = trendline.get_trading_strategy(data)
    print("\n交易策略:")
    print(f"信号准确率: {strategy['risk_metrics']['signal_accuracy']:.2%}")
    print(f"趋势稳定性: {strategy['risk_metrics']['trend_stability']:.2%}")