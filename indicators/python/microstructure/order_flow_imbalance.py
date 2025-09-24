"""
Order Flow Imbalance Indicator
==============================

基于订单流不平衡的微观结构指标，分析买卖压力的
实时失衡状态，识别短期价格变动和流动性冲击。
该指标通过分析订单簿动态、交易方向性和大额
交易流来预测短期价格走势。

特点：
- 订单簿深度分析
- 买卖压力差分
- 流动性冲击检测
- 大额交易识别
- 价格冲击建模
- 微观结构噪声
- 短期反转预测

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.stats import norm, zscore
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler


class OrderFlowImbalance:
    """
    订单流不平衡指标

    分析市场微观结构中的订单流动态，
    识别买卖压力失衡和交易机会。
    """

    def __init__(self, window_sizes: List[int] = [5, 10, 20],
                 imbalance_threshold: float = 0.6,
                 large_trade_threshold: float = 2.0):
        """
        初始化订单流不平衡指标

        Args:
            window_sizes: 分析窗口大小列表，默认[5,10,20]
            imbalance_threshold: 失衡阈值，默认0.6
            large_trade_threshold: 大额交易阈值，默认2.0
        """
        self.window_sizes = window_sizes
        self.imbalance_threshold = imbalance_threshold
        self.large_trade_threshold = large_trade_threshold
        self.name = f"Order Flow Imbalance ({window_sizes[0]})"
        self.category = "microstructure"

    def calculate_tick_imbalance(self, trades: pd.DataFrame) -> pd.Series:
        """
        计算Tick级别的订单流不平衡

        Args:
            trades: 交易数据，包含价格、数量、买卖方向

        Returns:
            Tick不平衡序列
        """
        # 确保交易数据有买卖方向
        if 'direction' not in trades.columns:
            trades = self.infer_trade_direction(trades)

        # 计算买卖交易量
        buy_volume = trades[trades['direction'] == 1]['volume'].resample('1T').sum()
        sell_volume = trades[trades['direction'] == -1]['volume'].resample('1T').sum()

        # 计算不平衡
        total_volume = buy_volume + sell_volume
        imbalance = (buy_volume - sell_volume) / (total_volume + 1e-10)

        return imbalance.fillna(0)

    def infer_trade_direction(self, trades: pd.DataFrame) -> pd.DataFrame:
        """
        推断交易方向（如果没有明确方向）

        Args:
            trades: 交易数据

        Returns:
            带有买卖方向的交易数据
        """
        trades = trades.copy()

        # 使用Lee-Ready算法推断交易方向
        if 'price' in trades.columns:
            # 计算中报价（如果没有提供，使用前一笔交易价格）
            trades['mid_price'] = trades['price'].shift(1)

            # 推断方向
            trades['direction'] = np.where(
                trades['price'] > trades['mid_price'], 1,  # 买入
                np.where(trades['price'] < trades['mid_price'], -1, 0)  # 卖出
            )

        return trades

    def calculate_price_imbalance(self, trades: pd.DataFrame) -> pd.Series:
        """
        计算价格驱动的不平衡

        Args:
            trades: 交易数据

        Returns:
            价格不平衡序列
        """
        if 'price' not in trades.columns:
            return pd.Series()

        # 计算价格变化
        price_changes = trades['price'].diff()

        # 计算价格冲击
        price_impact = price_changes * trades.get('volume', 1)

        # 按时间汇总
        price_imbalance = price_impact.resample('1T').sum()

        # 标准化
        price_imbalance = zscore(price_imbalance.fillna(0))

        return price_imbalance

    def calculate_large_trade_imbalance(self, trades: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算大额交易不平衡

        Args:
            trades: 交易数据

        Returns:
            大额交易分析结果
        """
        if 'volume' not in trades.columns:
            return {}

        # 识别大额交易
        volume_zscore = zscore(trades['volume'].fillna(0))
        large_trades_mask = volume_zscore > self.large_trade_threshold

        # 计算大额交易的不平衡
        if 'direction' in trades.columns:
            large_buy_volume = trades[large_trades_mask & (trades['direction'] == 1)]['volume'].resample('1T').sum()
            large_sell_volume = trades[large_trades_mask & (trades['direction'] == -1)]['volume'].resample('1T').sum()

            large_total = large_buy_volume + large_sell_volume
            large_imbalance = (large_buy_volume - large_sell_volume) / (large_total + 1e-10)
        else:
            large_imbalance = pd.Series()

        # 计算大额交易频率
        large_trade_frequency = large_trades_mask.astype(int).resample('1T').sum()

        # 计算大额交易强度
        large_trade_intensity = trades.loc[large_trades_mask, 'volume'].resample('1T').sum()

        return {
            'large_trade_imbalance': large_imbalance.fillna(0),
            'large_trade_frequency': large_trade_frequency.fillna(0),
            'large_trade_intensity': large_trade_intensity.fillna(0)
        }

    def calculate_liquidity_measures(self, trades: pd.DataFrame, order_book: pd.DataFrame = None) -> Dict[str, pd.Series]:
        """
        计算流动性指标

        Args:
            trades: 交易数据
            order_book: 订单簿数据（可选）

        Returns:
            流动性指标
        """
        liquidity_metrics = {}

        # 基于交易的流动性指标
        if 'volume' in trades.columns and 'price' in trades.columns:
            # Amihud非流动性指标
            price_changes = trades['price'].pct_change().fillna(0)
            volume_dollar = trades['volume'] * trades['price']

            # 按时间汇总
            abs_return = abs(price_changes).resample('1T').sum()
            dollar_volume = volume_dollar.resample('1T').sum()

            amihud_illiquidity = abs_return / (dollar_volume + 1e-10)
            liquidity_metrics['amihud_illiquidity'] = amihud_illiquidity.fillna(0)

            # Kyle's Lambda（价格冲击）
            net_order_flow = self.calculate_tick_imbalance(trades)
            price_changes_aggregated = trades['price'].resample('1T').last().pct_change().fillna(0)

            kyle_lambda = abs(price_changes_aggregated) / (abs(net_order_flow) + 1e-10)
            liquidity_metrics['kyle_lambda'] = kyle_lambda.fillna(0)

        # 基于订单簿的流动性指标
        if order_book is not None:
            # 订单簿深度
            if 'bid_depth' in order_book.columns and 'ask_depth' in order_book.columns:
                total_depth = order_book['bid_depth'] + order_book['ask_depth']
                liquidity_metrics['order_book_depth'] = total_depth.resample('1T').mean().fillna(0)

                # 买卖价差
                if 'bid_price' in order_book.columns and 'ask_price' in order_book.columns:
                    spread = order_book['ask_price'] - order_book['bid_price']
                    mid_price = (order_book['ask_price'] + order_book['bid_price']) / 2
                    relative_spread = spread / mid_price
                    liquidity_metrics['relative_spread'] = relative_spread.resample('1T').mean().fillna(0)

        return liquidity_metrics

    def calculate_microstructure_noise(self, trades: pd.DataFrame) -> pd.Series:
        """
        计算微观结构噪声

        Args:
            trades: 交易数据

        Returns:
            噪声指标
        """
        if 'price' not in trades.columns:
            return pd.Series()

        # 计算价格收益率
        returns = trades['price'].pct_change().fillna(0)

        # 计算已实现方差
        realized_variance = returns.rolling(window=5).var().fillna(0)

        # 计算买卖价差跳跃（如果没有订单簿数据，使用价格变化）
        price_jumps = abs(returns).rolling(window=5).quantile(0.9)

        # 微观结构噪声指标
        noise = realized_variance * price_jumps

        return noise.resample('1T').mean().fillna(0)

    def analyze_order_flow_dynamics(self, trades: pd.DataFrame, order_book: pd.DataFrame = None) -> Dict[str, Any]:
        """
        分析订单流动态

        Args:
            trades: 交易数据
            order_book: 订单簿数据（可选）

        Returns:
            订单流分析结果
        """
        results = {}

        # 基础不平衡指标
        tick_imbalance = self.calculate_tick_imbalance(trades)
        results['tick_imbalance'] = tick_imbalance

        # 价格不平衡
        price_imbalance = self.calculate_price_imbalance(trades)
        if not price_imbalance.empty:
            results['price_imbalance'] = price_imbalance

        # 大额交易不平衡
        large_trade_results = self.calculate_large_trade_imbalance(trades)
        results.update(large_trade_results)

        # 流动性指标
        liquidity_metrics = self.calculate_liquidity_measures(trades, order_book)
        results.update(liquidity_metrics)

        # 微观结构噪声
        microstructure_noise = self.calculate_microstructure_noise(trades)
        if not microstructure_noise.empty:
            results['microstructure_noise'] = microstructure_noise

        # 计算多时间窗口的指标
        for window in self.window_sizes:
            # 滚动平均不平衡
            rolling_imbalance = tick_imbalance.rolling(window=window).mean()
            results[f'rolling_imbalance_{window}'] = rolling_imbalance

            # 不平衡的波动率
            imbalance_volatility = tick_imbalance.rolling(window=window).std()
            results[f'imbalance_volatility_{window}'] = imbalance_volatility

        # 计算综合订单流压力
        results['composite_flow_pressure'] = self.calculate_composite_flow_pressure(results)

        # 生成交易信号
        results['signals'] = self.generate_order_flow_signals(results)

        return results

    def calculate_composite_flow_pressure(self, metrics: Dict[str, pd.Series]) -> pd.Series:
        """
        计算综合订单流压力

        Args:
            metrics: 各类指标

        Returns:
            综合压力指标
        """
        pressure_components = []

        # 基础不平衡权重
        if 'tick_imbalance' in metrics:
            pressure_components.append(metrics['tick_imbalance'] * 0.4)

        # 大额交易影响
        if 'large_trade_imbalance' in metrics:
            pressure_components.append(metrics['large_trade_imbalance'] * 0.3)

        # 价格影响
        if 'price_imbalance' in metrics:
            pressure_components.append(metrics['price_imbalance'] * 0.2)

        # 流动性影响（反向）
        if 'amihud_illiquidity' in metrics:
            # 流动性越差，压力越大
            illiquidity_normalized = zscore(metrics['amihud_illiquidity'].fillna(0))
            pressure_components.append(illiquidity_normalized * 0.1)

        if pressure_components:
            composite_pressure = pd.Series(0.0, index=pressure_components[0].index)
            for component in pressure_components:
                composite_pressure += component

            # 标准化到[-1,1]范围
            composite_pressure = composite_pressure.clip(-3, 3) / 3
            return composite_pressure.fillna(0)
        else:
            return pd.Series()

    def generate_order_flow_signals(self, metrics: Dict[str, pd.Series]) -> pd.Series:
        """
        基于订单流生成交易信号

        Args:
            metrics: 订单流指标

        Returns:
            交易信号
        """
        if 'composite_flow_pressure' not in metrics:
            return pd.Series()

        flow_pressure = metrics['composite_flow_pressure']
        signals = pd.Series(0, index=flow_pressure.index)

        # 基础阈值信号
        extreme_buy = flow_pressure > self.imbalance_threshold
        extreme_sell = flow_pressure < -self.imbalance_threshold
        moderate_buy = (flow_pressure > self.imbalance_threshold * 0.5) & (flow_pressure <= self.imbalance_threshold)
        moderate_sell = (flow_pressure < -self.imbalance_threshold * 0.5) & (flow_pressure >= -self.imbalance_threshold)

        # 大额交易确认
        if 'large_trade_frequency' in metrics:
            large_trade_confirm = metrics['large_trade_frequency'] > metrics['large_trade_frequency'].quantile(0.8)
            extreme_buy &= large_trade_confirm
            extreme_sell &= large_trade_confirm

        # 流动性确认
        if 'amihud_illiquidity' in metrics:
            liquidity_crisis = metrics['amihud_illiquidity'] > metrics['amihud_illiquidity'].quantile(0.8)
            # 在流动性危机时，信号减弱
            signals[liquidity_crisis & extreme_buy] = 1  # 降低强度
            signals[liquidity_crisis & extreme_sell] = -1

        # 多窗口确认
        confirmation_signals = []
        for window in self.window_sizes:
            if f'rolling_imbalance_{window}' in metrics:
                window_imbalance = metrics[f'rolling_imbalance_{window}']
                buy_confirmation = window_imbalance > self.imbalance_threshold * 0.7
                sell_confirmation = window_imbalance < -self.imbalance_threshold * 0.7
                confirmation_signals.append(buy_confirmation | sell_confirmation)

        # 综合确认
        if confirmation_signals:
            multi_window_confirmation = pd.Series(False, index=flow_pressure.index)
            for signal in confirmation_signals:
                multi_window_confirmation |= signal

            extreme_buy &= multi_window_confirmation
            extreme_sell &= multi_window_confirmation

        # 分配信号值
        signals[extreme_buy] = 3
        signals[extreme_sell] = -3
        signals[moderate_buy] = 2
        signals[moderate_sell] = -2

        return signals

    def get_trading_recommendations(self, order_flow_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            order_flow_analysis: 订单流分析结果

        Returns:
            交易建议DataFrame
        """
        # 找到一个指标序列作为索引
        index_keys = [k for k, v in order_flow_analysis.items() if isinstance(v, pd.Series) and not v.empty]
        if not index_keys:
            return pd.DataFrame()

        index = order_flow_analysis[index_keys[0]].index
        recommendations = pd.DataFrame(index=index)

        # 添加订单流指标
        for key, value in order_flow_analysis.items():
            if isinstance(value, pd.Series):
                recommendations[key] = value

        # 交易信号
        recommendations['signals'] = order_flow_analysis.get('signals', pd.Series(0, index=index))

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 极度买入压力',
            2: '买入 - 买入压力确认',
            0: '持有 - 压力平衡',
            -2: '卖出 - 卖出压力确认',
            -3: '强烈卖出 - 极度卖出压力'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 市场微观结构状态
        def classify_microstructure_state(row):
            flow_pressure = row.get('composite_flow_pressure', 0)
            liquidity = row.get('amihud_illiquidity', 0)
            noise = row.get('microstructure_noise', 0)

            if abs(flow_pressure) > 0.7 and liquidity > liquidity.quantile(0.8):
                return 'extreme_pressure_illiquid'
            elif abs(flow_pressure) > 0.5:
                return 'significant_pressure'
            elif liquidity > liquidity.quantile(0.7):
                return 'illiquid_market'
            elif noise > noise.quantile(0.8):
                return 'high_noise_market'
            else:
                return 'normal_microstructure'

        # 获取参考分位数（这里简化处理）
        if 'amihud_illiquidity' in recommendations.columns:
            liquidity_q80 = recommendations['amihud_illiquidity'].quantile(0.8)
        else:
            liquidity_q80 = 1.0

        if 'microstructure_noise' in recommendations.columns:
            noise_q80 = recommendations['microstructure_noise'].quantile(0.8)
        else:
            noise_q80 = 1.0

        recommendations['market_state'] = recommendations.apply(
            lambda x: classify_microstructure_state(x) if liquidity_q80 > 0 else 'normal_microstructure',
            axis=1
        )

        # 仓位建议
        position_map = {3: 0.7, 2: 0.4, 0: 0.1, -2: 0.1, -3: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 流动性调整
        if 'amihud_illiquidity' in recommendations.columns:
            liquidity_adjustment = 1 / (1 + recommendations['amihud_illiquidity'])
            recommendations['position_size'] *= liquidity_adjustment

        # 交易成本考虑
        if 'relative_spread' in recommendations.columns:
            spread_penalty = 1 - recommendations['relative_spread']
            recommendations['position_size'] *= spread_penalty

        # 策略建议
        def get_trading_strategy(market_state, signal_strength):
            if market_state == 'extreme_pressure_illiquid':
                return 'defensive'  # 避免在极度压力且流动性差时交易
            elif market_state == 'high_noise_market':
                return 'mean_reversion'  # 高噪声市场适合均值回归
            elif abs(signal_strength) >= 2:
                return 'momentum'  # 强信号时趋势跟踪
            else:
                return 'neutral'

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_trading_strategy(x['market_state'], abs(x['signals'])),
            axis=1
        )

        # 风险评估
        def assess_microstructure_risk(row):
            risk_score = 0

            # 流动性风险
            if 'amihud_illiquidity' in row.index:
                if row['amihud_illiquidity'] > row['amihud_illiquidity'].quantile(0.8):
                    risk_score += 0.4

            # 噪声风险
            if 'microstructure_noise' in row.index:
                if row['microstructure_noise'] > row['microstructure_noise'].quantile(0.8):
                    risk_score += 0.3

            # 压力风险
            if abs(row.get('composite_flow_pressure', 0)) > 0.8:
                risk_score += 0.3

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['risk_level'] = recommendations.apply(assess_microstructure_risk, axis=1)

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成模拟交易数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=1000, freq='1T')

    # 生成价格和交易量
    base_price = 100
    price_changes = np.random.randn(1000) * 0.01
    prices = [base_price]
    for change in price_changes[1:]:
        prices.append(prices[-1] * (1 + change))

    # 生成交易数据
    n_trades = len(dates)
    trade_directions = np.random.choice([-1, 1], n_trades, p=[0.45, 0.55])  # 略微偏向买入
    trade_volumes = np.random.lognormal(10, 1, n_trades).astype(int)

    trades = pd.DataFrame({
        'timestamp': dates,
        'price': prices,
        'volume': trade_volumes,
        'direction': trade_directions
    }).set_index('timestamp')

    # 创建订单流不平衡指标
    ofi = OrderFlowImbalance(window_sizes=[5, 10, 20])

    # 分析订单流动态
    print("分析订单流动态...")
    order_flow_results = ofi.analyze_order_flow_dynamics(trades)

    # 显示主要指标
    print(f"最新订单流不平衡: {order_flow_results['tick_imbalance'].iloc[-1]:.3f}")
    print(f"综合流压力: {order_flow_results['composite_flow_pressure'].iloc[-1]:.3f}")

    if 'large_trade_imbalance' in order_flow_results:
        print(f"大额交易不平衡: {order_flow_results['large_trade_imbalance'].iloc[-1]:.3f}")
        print(f"大额交易频率: {order_flow_results['large_trade_frequency'].iloc[-1]:.0f}")

    if 'amihud_illiquidity' in order_flow_results:
        print(f"Amihud非流动性: {order_flow_results['amihud_illiquidity'].iloc[-1]:.6f}")

    if 'microstructure_noise' in order_flow_results:
        print(f"微观结构噪声: {order_flow_results['microstructure_noise'].iloc[-1]:.6f}")

    print(f"当前信号: {order_flow_results['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = ofi.get_trading_recommendations(order_flow_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['tick_imbalance', 'composite_flow_pressure', 'signal_description',
                                          'market_state', 'strategy', 'position_size', 'risk_level']])