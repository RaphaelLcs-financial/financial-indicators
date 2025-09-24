"""
On-Chain MVRV Indicator
=======================

基于链上数据的MVRV（Market Value to Realized Value）指标，
用于评估加密货币的估值水平。MVRV比率是市场价值
与实现价值的比值，是识别市场周期顶部和底部的
重要链上指标。

特点：
- 链上数据集成
- 估值水平评估
- 市场周期识别
- 持仓者盈利分析
- 风险评估量化
- 巨鲸行为追踪
- 网络健康度监测

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional


class OnChainMVRVIndicator:
    """
    链上MVRV指标

    通过分析链上数据计算MVRV比率，
    评估加密货币的估值水平和市场状态。
    """

    def __init__(self, mvrv_windows: List[int] = [1, 7, 30, 90, 365],
                 threshold_levels: Dict[str, float] = None,
                 sensitivity: float = 1.5):
        """
        初始化链上MVRV指标

        Args:
            mvrv_windows: MVRV计算窗口，默认[1,7,30,90,365]天
            threshold_levels: 阈值水平
            sensitivity: 敏感度参数，默认1.5
        """
        self.mvrv_windows = mvrv_windows
        self.sensitivity = sensitivity

        # 默认阈值水平
        if threshold_levels is None:
            self.threshold_levels = {
                'extreme_fear': 0.8,
                'fear': 1.0,
                'neutral': 1.5,
                'greed': 2.5,
                'extreme_greed': 3.5
            }
        else:
            self.threshold_levels = threshold_levels

        self.name = f"On-Chain MVRV Indicator"
        self.category = "crypto_onchain"

    def calculate_market_value(self, price_data: pd.Series, supply_data: pd.Series) -> pd.Series:
        """
        计算市场价值

        Args:
            price_data: 价格数据
            supply_data: 供应量数据

        Returns:
            市场价值序列
        """
        return price_data * supply_data

    def calculate_realized_value(self, price_data: pd.Series, transaction_data: pd.DataFrame) -> pd.Series:
        """
        计算实现价值

        Args:
            price_data: 价格数据
            transaction_data: 交易数据，包含区块时间和交易价格

        Returns:
            实现价值序列
        """
        realized_value = pd.Series(0.0, index=price_data.index)

        # 模拟计算实现价值（实际需要真实的链上UTXO数据）
        for i in range(len(price_data)):
            if i < 30:  # 需要足够的历史数据
                continue

            current_time = price_data.index[i]
            lookback_period = 365  # 1年

            # 计算移动平均价格作为实现价值的代理
            start_idx = max(0, i - lookback_period)
            realized_prices = price_data.iloc[start_idx:i+1]
            realized_value.iloc[i] = realized_prices.mean()

        return realized_value

    def calculate_mvrv_ratios(self, market_value: pd.Series, realized_value: pd.Series) -> Dict[int, pd.Series]:
        """
        计算不同时间窗口的MVRV比率

        Args:
            market_value: 市场价值
            realized_value: 实现价值

        Returns:
            不同窗口的MVRV比率
        """
        mvrv_ratios = {}

        for window in self.mvrv_windows:
            # 计算滚动市场价值
            rolling_market_value = market_value.rolling(window=window).mean()

            # 计算MVRV比率
            mvrv_ratio = rolling_market_value / realized_value
            mvrv_ratio = mvrv_ratio.replace([np.inf, -np.inf], 0).fillna(1.0)

            mvrv_ratios[window] = mvrv_ratio

        return mvrv_ratios

    def analyze_investor_behavior(self, mvrv_ratios: Dict[int, pd.Series], price_data: pd.Series) -> Dict[str, pd.Series]:
        """
        分析投资者行为

        Args:
            mvrv_ratios: MVRV比率
            price_data: 价格数据

        Returns:
            投资者行为指标
        """
        behavior_metrics = {}

        # 计算短期投机者行为（1天MVRV）
        short_term_mvrv = mvrv_ratios.get(1, pd.Series(1.0, index=price_data.index))
        behavior_metrics['short_term_sentiment'] = self.normalize_mvrv(short_term_mvrv)

        # 计算中期投资者行为（30天MVRV）
        mid_term_mvrv = mvrv_ratios.get(30, pd.Series(1.0, index=price_data.index))
        behavior_metrics['mid_term_sentiment'] = self.normalize_mvrv(mid_term_mvrv)

        # 计算长期持有者行为（365天MVRV）
        long_term_mvrv = mvrv_ratios.get(365, pd.Series(1.0, index=price_data.index))
        behavior_metrics['long_term_sentiment'] = self.normalize_mvrv(long_term_mvrv)

        # 计算综合市场情绪
        behavior_metrics['composite_sentiment'] = (
            behavior_metrics['short_term_sentiment'] * 0.2 +
            behavior_metrics['mid_term_sentiment'] * 0.3 +
            behavior_metrics['long_term_sentiment'] * 0.5
        )

        return behavior_metrics

    def normalize_mvrv(self, mvrv_series: pd.Series) -> pd.Series:
        """
        标准化MVRV比率

        Args:
            mvrv_series: MVRV比率序列

        Returns:
            标准化后的序列
        """
        # 使用对数变换来标准化
        log_mvrv = np.log(mvrv_series.clip(0.1, 10))

        # 归一化到0-1范围
        min_val = log_mvrv.rolling(window=365).min()
        max_val = log_mvrv.rolling(window=365).max()

        normalized = (log_mvrv - min_val) / (max_val - min_val + 1e-10)
        return normalized.fillna(0.5)

    def identify_market_cycles(self, mvrv_ratios: Dict[int, pd.Series]) -> Dict[str, pd.Series]:
        """
        识别市场周期

        Args:
            mvrv_ratios: MVRV比率

        Returns:
            市场周期指标
        """
        cycle_indicators = {}

        # 使用30天MVRV作为主要周期指标
        main_mvrv = mvrv_ratios.get(30, pd.Series(1.0, index=mvrv_ratios[1].index))

        # 识别市场顶部
        market_top = main_mvrv > self.threshold_levels['extreme_greed']
        cycle_indicators['market_top'] = market_top.astype(int)

        # 识别市场底部
        market_bottom = main_mvrv < self.threshold_levels['extreme_fear']
        cycle_indicators['market_bottom'] = market_bottom.astype(int)

        # 计算周期强度
        cycle_indicators['cycle_strength'] = abs(main_mvrv - 1.0) * self.sensitivity

        # 计算周期相位
        cycle_indicators['cycle_phase'] = np.arctan2(main_mvrv - 1, 1) / np.pi

        return cycle_indicators

    def calculate_risk_metrics(self, mvrv_ratios: Dict[int, pd.Series], price_data: pd.Series) -> Dict[str, pd.Series]:
        """
        计算风险指标

        Args:
            mvrv_ratios: MVRV比率
            price_data: 价格数据

        Returns:
            风险指标
        """
        risk_metrics = {}

        # 计算估值风险
        main_mvrv = mvrv_ratios.get(30, pd.Series(1.0, index=price_data.index))
        valuation_risk = np.maximum(0, main_mvrv - self.threshold_levels['neutral'])
        risk_metrics['valuation_risk'] = valuation_risk

        # 计算泡沫风险
        bubble_risk = np.maximum(0, main_mvrv - self.threshold_levels['greed'])
        risk_metrics['bubble_risk'] = bubble_risk

        # 计算恐慌风险
        panic_risk = np.maximum(0, self.threshold_levels['fear'] - main_mvrv)
        risk_metrics['panic_risk'] = panic_risk

        # 计算综合风险
        risk_metrics['composite_risk'] = (
            valuation_risk * 0.4 + bubble_risk * 0.4 + panic_risk * 0.2
        )

        return risk_metrics

    def generate_trading_signals(self, mvrv_ratios: Dict[int, pd.Series],
                               behavior_metrics: Dict[str, pd.Series],
                               cycle_indicators: Dict[str, pd.Series],
                               risk_metrics: Dict[str, pd.Series]) -> pd.Series:
        """
        生成交易信号

        Args:
            mvrv_ratios: MVRV比率
            behavior_metrics: 行为指标
            cycle_indicators: 周期指标
            risk_metrics: 风险指标

        Returns:
            交易信号
        """
        signals = pd.Series(0, index=mvrv_ratios[1].index)

        # 获取主要MVRV比率
        main_mvrv = mvrv_ratios.get(30, pd.Series(1.0, index=signals.index))
        long_term_mvrv = mvrv_ratios.get(365, pd.Series(1.0, index=signals.index))

        # 强烈买入信号：市场底部 + 长期价值低估
        strong_buy = (
            (main_mvrv < self.threshold_levels['extreme_fear']) &
            (long_term_mvrv < 1.2) &
            (cycle_indicators['market_bottom'] == 1)
        )

        # 买入信号：价值区域 + 风险较低
        buy = (
            (main_mvrv < self.threshold_levels['fear']) &
            (behavior_metrics['composite_sentiment'] < 0.3) &
            (risk_metrics['composite_risk'] < 0.5)
        )

        # 卖出信号：高估区域 + 风险较高
        sell = (
            (main_mvrv > self.threshold_levels['greed']) &
            (behavior_metrics['composite_sentiment'] > 0.7) &
            (risk_metrics['composite_risk'] > 1.0)
        )

        # 强烈卖出信号：市场顶部 + 泡沫风险
        strong_sell = (
            (main_mvrv > self.threshold_levels['extreme_greed']) &
            (cycle_indicators['market_top'] == 1) &
            (risk_metrics['bubble_risk'] > 1.0)
        )

        # 分配信号强度
        signals[strong_buy] = 3
        signals[buy] = 2
        signals[sell] = -2
        signals[strong_sell] = -3

        return signals

    def analyze_on_chain_data(self, price_data: pd.Series, supply_data: pd.Series,
                            transaction_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        分析链上数据

        Args:
            price_data: 价格数据
            supply_data: 供应量数据
            transaction_data: 交易数据（可选）

        Returns:
            综合分析结果
        """
        # 计算市场价值和实现价值
        market_value = self.calculate_market_value(price_data, supply_data)

        if transaction_data is None:
            # 如果没有交易数据，使用简化方法计算实现价值
            realized_value = price_data.rolling(window=365).mean()
        else:
            realized_value = self.calculate_realized_value(price_data, transaction_data)

        # 计算MVRV比率
        mvrv_ratios = self.calculate_mvrv_ratios(market_value, realized_value)

        # 分析投资者行为
        behavior_metrics = self.analyze_investor_behavior(mvrv_ratios, price_data)

        # 识别市场周期
        cycle_indicators = self.identify_market_cycles(mvrv_ratios)

        # 计算风险指标
        risk_metrics = self.calculate_risk_metrics(mvrv_ratios, price_data)

        # 生成交易信号
        signals = self.generate_trading_signals(
            mvrv_ratios, behavior_metrics, cycle_indicators, risk_metrics
        )

        return {
            'market_value': market_value,
            'realized_value': realized_value,
            'mvrv_ratios': mvrv_ratios,
            'behavior_metrics': behavior_metrics,
            'cycle_indicators': cycle_indicators,
            'risk_metrics': risk_metrics,
            'signals': signals
        }

    def get_whale_activity_signals(self, large_transactions: pd.DataFrame, price_data: pd.Series) -> Dict[str, pd.Series]:
        """
        获取巨鲸活动信号

        Args:
            large_transactions: 大额交易数据
            price_data: 价格数据

        Returns:
            巨鲸活动信号
        """
        whale_signals = {}

        # 计算大额交易频率
        transaction_frequency = large_transactions.resample('1D').size()
        whale_signals['whale_activity_frequency'] = transaction_frequency.rolling(7).mean()

        # 计算大额交易规模
        transaction_volume = large_transactions['value'].resample('1D').sum()
        whale_signals['whale_volume'] = transaction_volume.rolling(7).mean()

        # 计算巨鲸积累/分布信号
        whale_net_flow = large_transactions.groupby(large_transactions.index.date)['value'].sum()
        whale_net_flow = whale_net_flow.rolling(7).mean()

        # 标准化巨鲸净流量
        whale_signals['whale_accumulation'] = (whale_net_flow - whale_net_flow.rolling(30).mean()) / whale_net_flow.rolling(30).std()
        whale_signals['whale_accumulation'] = whale_signals['whale_accumulation'].fillna(0)

        return whale_signals

    def get_trading_recommendations(self, analysis_results: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            analysis_results: 分析结果

        Returns:
            交易建议DataFrame
        """
        # 使用30天MVRV作为主要指标
        main_mvrv = analysis_results['mvrv_ratios'].get(30, pd.Series(1.0))
        recommendations = pd.DataFrame(index=main_mvrv.index)

        # MVRV数据
        for window in self.mvrv_windows:
            recommendations[f'mvrv_{window}d'] = analysis_results['mvrv_ratios'][window]

        recommendations['market_value'] = analysis_results['market_value']
        recommendations['realized_value'] = analysis_results['realized_value']

        # 行为指标
        for metric_name, metric_data in analysis_results['behavior_metrics'].items():
            recommendations[metric_name] = metric_data

        # 周期指标
        for indicator_name, indicator_data in analysis_results['cycle_indicators'].items():
            recommendations[indicator_name] = indicator_data

        # 风险指标
        for risk_name, risk_data in analysis_results['risk_metrics'].items():
            recommendations[risk_name] = risk_data

        # 交易信号
        recommendations['signals'] = analysis_results['signals']

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 极度低估，市场底部',
            2: '买入 - 价值区域，风险较低',
            0: '持有 - 估值合理',
            -2: '卖出 - 高估区域，风险较高',
            -3: '强烈卖出 - 极度高估，市场顶部'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 市场情绪
        def get_market_sentiment(mvrv_value):
            if mvrv_value < self.threshold_levels['extreme_fear']:
                return 'extreme_fear'
            elif mvrv_value < self.threshold_levels['fear']:
                return 'fear'
            elif mvrv_value < self.threshold_levels['neutral']:
                return 'neutral'
            elif mvrv_value < self.threshold_levels['greed']:
                return 'greed'
            else:
                return 'extreme_greed'

        recommendations['market_sentiment'] = recommendations['mvrv_30d'].apply(get_market_sentiment)

        # 仓位建议
        position_map = {
            3: 0.8, 2: 0.5, 0: 0.2, -2: 0.1, -3: 0.0
        }
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 风险调整
        risk_adjustment = 1 / (1 + recommendations['composite_risk'])
        recommendations['position_size'] *= risk_adjustment

        # 投资期限建议
        def get_investment_horizon(mvrv_value, cycle_phase):
            if mvrv_value < 1.0:
                return 'long_term'
            elif mvrv_value < 1.5:
                return 'medium_term'
            else:
                return 'short_term'

        recommendations['investment_horizon'] = recommendations.apply(
            lambda x: get_investment_horizon(x['mvrv_30d'], x['cycle_phase']),
            axis=1
        )

        # 止损建议
        if 'realized_value' in recommendations.columns:
            recommendations['stop_loss'] = recommendations['realized_value'] * 0.9

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    n = len(dates)

    # 生成模拟价格数据
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    supply = pd.Series(19000000, index=dates)  # 固定供应量

    data = pd.DataFrame({
        'close': close_prices,
        'supply': supply
    }, index=dates)

    # 创建链上MVRV指标
    mvrv_indicator = OnChainMVRVIndicator()

    # 分析链上数据
    print("分析链上MVRV数据...")
    analysis_results = mvrv_indicator.analyze_on_chain_data(
        data['close'], data['supply']
    )

    # 显示结果
    print(f"最新30天MVRV: {analysis_results['mvrv_ratios'][30].iloc[-1]:.2f}")
    print(f"市场价值: {analysis_results['market_value'].iloc[-1]:.2e}")
    print(f"实现价值: {analysis_results['realized_value'].iloc[-1]:.2e}")
    print(f"市场情绪: {analysis_results['behavior_metrics']['composite_sentiment'].iloc[-1]:.2f}")
    print(f"周期强度: {analysis_results['cycle_indicators']['cycle_strength'].iloc[-1]:.2f}")
    print(f"综合风险: {analysis_results['risk_metrics']['composite_risk'].iloc[-1]:.2f}")
    print(f"当前信号: {analysis_results['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = mvrv_indicator.get_trading_recommendations(analysis_results)
    print("\n最新交易建议:")
    print(recommendations.iloc[-1][['mvrv_1d', 'mvrv_30d', 'mvrv_365d', 'market_sentiment',
                                      'signal_description', 'position_size', 'investment_horizon']])