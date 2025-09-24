"""
Hull Moving Average Envelope (HMA Envelope)
==========================================

A popular trend-following indicator that combines Hull Moving Average with
envelope bands to provide dynamic support and resistance levels. The HMA
eliminates lag while maintaining smoothness, and the envelope bands help
identify overbought/oversold conditions within the trend.

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional


class HullMovingAverageEnvelope:
    """
    Hull Moving Average Envelope - 赫尔移动平均线包络带

    HMA消除了传统移动平均线的滞后性，同时保持了平滑性。
    包络带提供了动态的支撑和阻力水平，帮助识别趋势内的超买超卖条件。

    特点：
    - 几乎零滞后，反应迅速
    - 包络带提供动态支撑阻力
    - 适用于趋势交易和突破策略
    - 减少假信号，提高交易质量
    """

    def __init__(self, period: int = 20, envelope_percent: float = 2.0):
        """
        初始化HMA包络带

        Args:
            period: HMA周期，默认20
            envelope_percent: 包络带百分比，默认2.0%
        """
        self.period = period
        self.envelope_percent = envelope_percent
        self.name = f"HMA Envelope ({period}, {envelope_percent}%)"
        self.category = "trend"

    def calculate_hma(self, prices: Union[List[float], pd.Series], period: int) -> pd.Series:
        """
        计算赫尔移动平均线 (HMA)

        公式: HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))

        Args:
            prices: 价格序列
            period: 计算周期

        Returns:
            HMA序列
        """
        prices_series = pd.Series(prices)

        # 计算整数周期
        half_period = max(1, int(period / 2))
        sqrt_period = max(1, int(np.sqrt(period)))

        # 计算WMA
        wma_half = self.calculate_wma(prices_series, half_period)
        wma_full = self.calculate_wma(prices_series, period)

        # 计算2*WMA(n/2) - WMA(n)
        raw_hma = 2 * wma_half - wma_full

        # 对结果计算WMA(sqrt(n))
        hma = self.calculate_wma(raw_hma, sqrt_period)

        return hma

    def calculate_wma(self, prices: pd.Series, period: int) -> pd.Series:
        """
        计算加权移动平均线 (WMA)

        Args:
            prices: 价格序列
            period: 计算周期

        Returns:
            WMA序列
        """
        weights = np.arange(1, period + 1)
        wma = prices.rolling(window=period).apply(
            lambda x: np.sum(x * weights) / weights.sum(), raw=True
        )
        return wma

    def calculate_envelope(self, prices: Union[List[float], pd.Series]) -> Dict[str, pd.Series]:
        """
        计算HMA包络带

        Args:
            prices: 价格序列

        Returns:
            包含HMA和包络带的字典
        """
        prices_series = pd.Series(prices)

        # 计算HMA
        hma = self.calculate_hma(prices_series, self.period)

        # 计算包络带
        envelope_value = hma * (self.envelope_percent / 100)
        upper_band = hma + envelope_value
        lower_band = hma - envelope_value

        # 计算价格在包络带中的位置
        price_position = (prices_series - lower_band) / (upper_band - lower_band)

        # 计算趋势强度
        trend_strength = self.calculate_trend_strength(prices_series, hma)

        # 生成交易信号
        signals = self.generate_signals(prices_series, hma, upper_band, lower_band)

        return {
            'hma': hma,
            'upper_band': upper_band,
            'lower_band': lower_band,
            'price_position': price_position,
            'trend_strength': trend_strength,
            'signals': signals
        }

    def calculate_trend_strength(self, prices: pd.Series, hma: pd.Series) -> pd.Series:
        """
        计算趋势强度

        Args:
            prices: 价格序列
            hma: HMA序列

        Returns:
            趋势强度序列 (0-100)
        """
        # 计算价格与HMA的距离
        distance = abs(prices - hma) / hma * 100

        # 计算价格在HMA上方的比例
        above_hma = (prices > hma).rolling(window=self.period).mean() * 100

        # 计算HMA斜率
        hma_slope = hma.diff().rolling(window=5).mean()

        # 综合趋势强度
        trend_strength = (
            (distance * 0.3) +
            (above_hma * 0.4) +
            (hma_slope.abs() * 1000 * 0.3)
        ).clip(0, 100)

        return trend_strength

    def generate_signals(self, prices: pd.Series, hma: pd.Series,
                        upper_band: pd.Series, lower_band: pd.Series) -> pd.Series:
        """
        生成交易信号

        Args:
            prices: 价格序列
            hma: HMA序列
            upper_band: 上轨
            lower_band: 下轨

        Returns:
            信号序列 (1=买入, -1=卖出, 0=持有)
        """
        signals = pd.Series(0, index=prices.index)

        # 买入信号：价格从下轨反弹且在HMA上方
        buy_condition = (
            (prices > lower_band) &
            (prices.shift(1) <= lower_band.shift(1)) &
            (prices > hma) &
            (hma.diff() > 0)
        )

        # 卖出信号：价格从上轨回落且在HMA下方
        sell_condition = (
            (prices < upper_band) &
            (prices.shift(1) >= upper_band.shift(1)) &
            (prices < hma) &
            (hma.diff() < 0)
        )

        signals[buy_condition] = 1
        signals[sell_condition] = -1

        return signals

    def get_trading_signals(self, data: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        获取完整的交易信号信息

        Args:
            data: 包含OHLCV数据的字典

        Returns:
            完整的交易信号信息
        """
        prices = data['close']

        # 计算HMA包络带
        envelope_data = self.calculate_envelope(prices)

        # 计算其他指标
        rsi = self.calculate_rsi(prices, 14)
        volume_surge = self.calculate_volume_surge(data['volume'])

        # 生成增强信号
        enhanced_signals = self.generate_enhanced_signals(
            prices, envelope_data, rsi, volume_surge
        )

        # 计算支撑阻力位
        support_resistance = self.calculate_support_resistance(
            prices, envelope_data['hma']
        )

        return {
            'hma': envelope_data['hma'],
            'upper_band': envelope_data['upper_band'],
            'lower_band': envelope_data['lower_band'],
            'price_position': envelope_data['price_position'],
            'trend_strength': envelope_data['trend_strength'],
            'signals': enhanced_signals,
            'support_resistance': support_resistance,
            'rsi': rsi,
            'volume_surge': volume_surge
        }

    def generate_enhanced_signals(self, prices: pd.Series, envelope_data: Dict[str, pd.Series],
                                  rsi: pd.Series, volume_surge: pd.Series) -> pd.Series:
        """
        生成增强的交易信号

        Args:
            prices: 价格序列
            envelope_data: HMA包络带数据
            rsi: RSI序列
            volume_surge: 成交量激增序列

        Returns:
            增强的信号序列
        """
        signals = pd.Series(0, index=prices.index)

        # 多重确认买入信号
        buy_condition = (
            (envelope_data['signals'] == 1) &
            (rsi < 70) &
            (volume_surge > 1.0) &
            (envelope_data['trend_strength'] > 30)
        )

        # 多重确认卖出信号
        sell_condition = (
            (envelope_data['signals'] == -1) &
            (rsi > 30) &
            (volume_surge > 1.0) &
            (envelope_data['trend_strength'] > 30)
        )

        # 趋势跟踪信号
        trend_buy = (
            (prices > envelope_data['hma']) &
            (envelope_data['hma'].diff() > 0) &
            (envelope_data['price_position'] > 0.3) &
            (envelope_data['price_position'] < 0.8)
        )

        trend_sell = (
            (prices < envelope_data['hma']) &
            (envelope_data['hma'].diff() < 0) &
            (envelope_data['price_position'] < 0.7) &
            (envelope_data['price_position'] > 0.2)
        )

        signals[buy_condition] = 2  # 强买入
        signals[trend_buy] = 1     # 趋势买入
        signals[sell_condition] = -2  # 强卖出
        signals[trend_sell] = -1     # 趋势卖出

        return signals

    def calculate_support_resistance(self, prices: pd.Series, hma: pd.Series) -> Dict[str, pd.Series]:
        """
        计算动态支撑阻力位

        Args:
            prices: 价格序列
            hma: HMA序列

        Returns:
            支撑阻力位字典
        """
        # 基于HMA的动态支撑阻力
        support = hma * 0.98  # HMA下方2%
        resistance = hma * 1.02  # HMA上方2%

        # 基于近期高低点的支撑阻力
        recent_high = prices.rolling(window=20).max()
        recent_low = prices.rolling(window=20).min()

        # 动态调整
        volatility = prices.rolling(window=20).std() / prices.rolling(window=20).mean()
        dynamic_support = recent_low * (1 - volatility)
        dynamic_resistance = recent_high * (1 + volatility)

        return {
            'hma_support': support,
            'hma_resistance': resistance,
            'dynamic_support': dynamic_support,
            'dynamic_resistance': dynamic_resistance
        }

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

    def calculate_volume_surge(self, volume: pd.Series, period: int = 20) -> pd.Series:
        """
        计算成交量激增

        Args:
            volume: 成交量序列
            period: 计算周期

        Returns:
            成交量激增比率
        """
        volume_ma = volume.rolling(window=period).mean()
        volume_surge = volume / volume_ma
        return volume_surge

    def get_trading_recommendations(self, signals: pd.Series, prices: pd.Series,
                                   trend_strength: pd.Series) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            signals: 信号序列
            prices: 价格序列
            trend_strength: 趋势强度

        Returns:
            交易建议DataFrame
        """
        recommendations = pd.DataFrame(index=prices.index)

        # 信号解释
        recommendations['signal'] = signals
        recommendations['signal_description'] = signals.map({
            2: '强买入 - 趋势突破确认',
            1: '买入 - 趋势跟踪',
            0: '持有 - 观望',
            -1: '卖出 - 趋势反转',
            -2: '强卖出 - 趋势破位'
        })

        # 风险等级
        recommendations['risk_level'] = pd.cut(
            trend_strength,
            bins=[0, 20, 40, 60, 80, 100],
            labels=['极低', '低', '中', '高', '极高']
        )

        # 建议仓位
        recommendations['position_size'] = signals.map({
            2: 0.8,   # 强买入：80%仓位
            1: 0.5,   # 买入：50%仓位
            0: 0.2,   # 持有：20%仓位
            -1: 0.1,  # 卖出：10%仓位
            -2: 0.0   # 强卖出：0仓位
        })

        # 止损止盈建议
        recommendations['stop_loss'] = prices * 0.95  # 5%止损
        recommendations['take_profit'] = prices * 1.10  # 10%止盈

        return recommendations

    def analyze_market_condition(self, prices: pd.Series, hma: pd.Series,
                                upper_band: pd.Series, lower_band: pd.Series) -> Dict[str, Any]:
        """
        分析市场状况

        Args:
            prices: 价格序列
            hma: HMA序列
            upper_band: 上轨
            lower_band: 下轨

        Returns:
            市场状况分析结果
        """
        latest_price = prices.iloc[-1]
        latest_hma = hma.iloc[-1]
        latest_upper = upper_band.iloc[-1]
        latest_lower = lower_band.iloc[-1]

        # 计算市场状态
        price_position = (latest_price - latest_lower) / (latest_upper - latest_lower)

        # 趋势方向
        if latest_price > latest_hma and hma.diff().iloc[-1] > 0:
            trend_direction = "上升趋势"
        elif latest_price < latest_hma and hma.diff().iloc[-1] < 0:
            trend_direction = "下降趋势"
        else:
            trend_direction = "横盘整理"

        # 波动性状态
        bandwidth = (latest_upper - latest_lower) / latest_hma * 100
        if bandwidth > 5:
            volatility_state = "高波动"
        elif bandwidth > 3:
            volatility_state = "中波动"
        else:
            volatility_state = "低波动"

        # 超买超卖状态
        if price_position > 0.8:
            overbought_oversold = "超买"
        elif price_position < 0.2:
            overbought_oversold = "超卖"
        else:
            overbought_oversold = "正常"

        return {
            'trend_direction': trend_direction,
            'volatility_state': volatility_state,
            'overbought_oversold': overbought_oversold,
            'price_position': price_position,
            'bandwidth': bandwidth,
            'market_regime': self.determine_market_regime(
                trend_direction, volatility_state, overbought_oversold
            )
        }

    def determine_market_regime(self, trend: str, volatility: str, state: str) -> str:
        """
        确定市场制度

        Args:
            trend: 趋势方向
            volatility: 波动性状态
            state: 超买超卖状态

        Returns:
            市场制度
        """
        if trend == "上升趋势" and volatility == "低波动":
            return "稳定上涨"
        elif trend == "上升趋势" and volatility == "高波动":
            return "强势上涨"
        elif trend == "下降趋势" and volatility == "低波动":
            return "稳定下跌"
        elif trend == "下降趋势" and volatility == "高波动":
            return "恐慌下跌"
        elif trend == "横盘整理" and volatility == "低波动":
            return "盘整蓄势"
        else:
            return "震荡市"


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

    # 创建HMA包络带指标
    hma_envelope = HullMovingAverageEnvelope(period=20, envelope_percent=2.0)

    # 计算指标
    results = hma_envelope.get_trading_signals(data)

    print("Hull Moving Average Envelope 结果:")
    print(f"HMA: {results['hma'].iloc[-1]:.2f}")
    print(f"上轨: {results['upper_band'].iloc[-1]:.2f}")
    print(f"下轨: {results['lower_band'].iloc[-1]:.2f}")
    print(f"价格位置: {results['price_position'].iloc[-1]:.2f}")
    print(f"趋势强度: {results['trend_strength'].iloc[-1]:.2f}")
    print(f"当前信号: {results['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = hma_envelope.get_trading_recommendations(
        results['signals'], data['close'], results['trend_strength']
    )
    print("\n最新交易建议:")
    print(recommendations.iloc[-1])

    # 分析市场状况
    market_analysis = hma_envelope.analyze_market_condition(
        data['close'], results['hma'], results['upper_band'], results['lower_band']
    )
    print("\n市场状况分析:")
    for key, value in market_analysis.items():
        print(f"{key}: {value}")