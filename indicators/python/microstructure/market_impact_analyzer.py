"""
Market Impact Analyzer
======================

基于市场冲击的微观结构分析器，量化交易对价格
的冲击效应和流动性成本。该指标通过分析大额
交易的价格影响、订单簿动态和永久冲击来
评估交易的执行成本和市场效率。

特点：
- 价格冲击建模
- 永久临时冲击分解
- 流动性成本量化
- 订单簿弹性分析
- 交易成本优化
- 市场深度评估
- 执行策略建议

Author: Claude AI Assistant
Version: 1.0
Date: 2025-25-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.optimize import curve_fit
from scipy.stats import linregress
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MarketImpactAnalyzer:
    """
    市场冲击分析器

    分析交易对市场的冲击效应，
    量化执行成本和流动性影响。
    """

    def __init__(self, impact_window: int = 30, decay_factor: float = 0.95,
                 cost_threshold: float = 0.001, volume_profile_window: int = 20):
        """
        初始化市场冲击分析器

        Args:
            impact_window: 冲击分析窗口，默认30
            decay_factor: 衰减因子，默认0.95
            cost_threshold: 成本阈值，默认0.001
            volume_profile_window: 成交量分布窗口，默认20
        """
        self.impact_window = impact_window
        self.decay_factor = decay_factor
        self.cost_threshold = cost_threshold
        self.volume_profile_window = volume_profile_window
        self.name = f"Market Impact Analyzer ({impact_window})"
        self.category = "microstructure"

    def calculate_temporary_impact(self, trades: pd.DataFrame, prices: pd.Series) -> pd.Series:
        """
        计算临时价格冲击

        Args:
            trades: 交易数据
            prices: 价格序列

        Returns:
            临时冲击序列
        """
        # 对齐时间索引
        aligned_trades = trades.copy()
        if not aligned_trades.index.equals(prices.index):
            aligned_trades = aligned_trades.reindex(prices.index, method='ffill').fillna(0)

        # 计算交易量标准化
        if 'volume' in aligned_trades.columns:
            avg_volume = aligned_trades['volume'].rolling(self.volume_profile_window).mean()
            normalized_volume = aligned_trades['volume'] / (avg_volume + 1e-10)
        else:
            normalized_volume = pd.Series(1.0, index=prices.index)

        # 计算短期价格反应
        price_changes = prices.pct_change().fillna(0)

        # 计算临时冲击（短期价格反应）
        temporary_impact = price_changes * normalized_volume

        # 应用指数衰减
        temporary_impact = temporary_impact.ewm(alpha=1-self.decay_factor).mean()

        return temporary_impact.fillna(0)

    def calculate_permanent_impact(self, trades: pd.DataFrame, prices: pd.Series) -> pd.Series:
        """
        计算永久价格冲击

        Args:
            trades: 交易数据
            prices: 价格序列

        Returns:
            永久冲击序列
        """
        # 计算累计净交易量
        if 'volume' in trades.columns and 'direction' in trades.columns:
            # 确保有方向信息
            if 'direction' not in trades.columns:
                trades = self.infer_trade_direction(trades, prices)

            signed_volume = trades['volume'] * trades['direction']
            cumulative_flow = signed_volume.cumsum()
        else:
            cumulative_flow = pd.Series(0, index=prices.index)

        # 对齐时间索引
        cumulative_flow = cumulative_flow.reindex(prices.index, method='ffill').fillna(0)

        # 计算长期价格趋势
        long_term_trend = prices.rolling(self.impact_window).mean().pct_change().fillna(0)

        # 计算永久冲击（长期价格效应）
        permanent_impact = long_term_trend * np.sign(cumulative_flow)

        return permanent_impact.fillna(0)

    def infer_trade_direction(self, trades: pd.DataFrame, prices: pd.Series) -> pd.DataFrame:
        """
        推断交易方向

        Args:
            trades: 交易数据
            prices: 价格序列

        Returns:
            带有方向的交易数据
        """
        trades = trades.copy()

        # 对齐价格
        aligned_prices = prices.reindex(trades.index, method='ffill')

        # 使用价格变化推断方向
        if 'price' in trades.columns:
            # 计算相对于前一价格的变动
            trades['prev_price'] = aligned_prices.shift(1)
            trades['price_change'] = trades['price'] - trades['prev_price']

            # 推断方向
            trades['direction'] = np.where(
                trades['price_change'] > 0, 1,
                np.where(trades['price_change'] < 0, -1, 0)
            )

        return trades

    def decompose_market_impact(self, trades: pd.DataFrame, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        分解市场冲击成分

        Args:
            trades: 交易数据
            prices: 价格序列

        Returns:
            冲击分解结果
        """
        results = {}

        # 计算临时和永久冲击
        temporary_impact = self.calculate_temporary_impact(trades, prices)
        permanent_impact = self.calculate_permanent_impact(trades, prices)

        results['temporary_impact'] = temporary_impact
        results['permanent_impact'] = permanent_impact

        # 计算总冲击
        total_impact = temporary_impact + permanent_impact
        results['total_impact'] = total_impact

        # 计算冲击比率
        total_abs_impact = abs(total_impact)
        temp_abs_impact = abs(temporary_impact)
        perm_abs_impact = abs(permanent_impact)

        impact_ratio = total_abs_impact / (total_abs_impact + 1e-10)
        temp_ratio = temp_abs_impact / (total_abs_impact + 1e-10)
        perm_ratio = perm_abs_impact / (total_abs_impact + 1e-10)

        results['impact_ratio'] = impact_ratio
        results['temporary_ratio'] = temp_ratio
        results['permanent_ratio'] = perm_ratio

        # 计算冲击持续性
        impact_persistence = self.calculate_impact_persistence(total_impact)
        results['impact_persistence'] = impact_persistence

        return results

    def calculate_impact_persistence(self, impact_series: pd.Series) -> pd.Series:
        """
        计算冲击持续性

        Args:
            impact_series: 冲击序列

        Returns:
            持续性指标
        """
        # 计算冲击的自相关性
        autocorr_lag1 = impact_series.autocorr(lag=1)
        autocorr_lag5 = impact_series.autocorr(lag=5)

        # 计算冲击的半衰期
        decay_rate = -np.log(abs(autocorr_lag1) + 1e-10)
        half_life = np.log(2) / (decay_rate + 1e-10)

        # 综合持续性指标
        persistence = (abs(autocorr_lag1) + abs(autocorr_lag5)) / 2
        persistence = np.clip(persistence, 0, 1)

        return pd.Series(persistence, index=impact_series.index).fillna(0.5)

    def estimate_impact_curve(self, trades: pd.DataFrame, prices: pd.Series) -> Dict[str, Any]:
        """
        估计冲击曲线参数

        Args:
            trades: 交易数据
            prices: 价格序列

        Returns:
            冲击曲线参数
        """
        # 准备数据
        if 'volume' in trades.columns:
            volumes = trades['volume'].values
        else:
            volumes = np.ones(len(trades))

        # 计算价格冲击
        price_changes = prices.pct_change().fillna(0)
        aligned_changes = price_changes.reindex(trades.index, method='ffill').fillna(0)

        # 标准化交易量
        normalized_volumes = volumes / (np.median(volumes) + 1e-10)
        abs_changes = np.abs(aligned_changes)

        # 定义冲击函数: impact = a * volume^b
        def impact_function(volume, a, b):
            return a * (volume ** b)

        try:
            # 拟合冲击曲线
            valid_indices = (normalized_volumes > 0) & (abs_changes > 0)
            if np.sum(valid_indices) > 10:
                popt, _ = curve_fit(impact_function, normalized_volumes[valid_indices],
                                 abs_changes[valid_indices], maxfev=1000)

                impact_params = {
                    'temporary_alpha': popt[0],  # 临时冲击系数
                    'temporary_beta': popt[1],   # 临时冲击弹性
                    'impact_function': 'power_law'
                }
            else:
                impact_params = {
                    'temporary_alpha': 0.001,
                    'temporary_beta': 0.5,
                    'impact_function': 'linear'
                }

        except:
            impact_params = {
                'temporary_alpha': 0.001,
                'temporary_beta': 0.5,
                'impact_function': 'linear'
            }

        # 估计永久冲击参数
        permanent_impact = self.calculate_permanent_impact(trades, prices)
        if 'volume' in trades.columns:
            signed_volumes = trades['volume'] * trades.get('direction', 1)
            correlation = np.corrcoef(signed_volumes.fillna(0), permanent_impact.fillna(0))[0, 1]
        else:
            correlation = 0

        impact_params['permanent_correlation'] = correlation
        impact_params['permanent_alpha'] = abs(correlation) * impact_params['temporary_alpha']

        return impact_params

    def calculate_execution_costs(self, trades: pd.DataFrame, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        计算执行成本

        Args:
            trades: 交易数据
            prices: 价格序列

        Returns:
            执行成本指标
        """
        costs = {}

        # 实现落差（如果交易价格与基准价格有差异）
        if 'price' in trades.columns:
            # 使用VWAP作为基准
            vwap = (trades['price'] * trades['volume']).cumsum() / trades['volume'].cumsum()

            # 计算实现落差
            implementation_shortfall = (trades['price'] - vwap) / vwap
            costs['implementation_shortfall'] = implementation_shortfall.fillna(0)

            # 计算机会成本
            if 'direction' in trades.columns:
                # 买入交易的机会成本
                buy_mask = trades['direction'] == 1
                if buy_mask.any():
                    future_prices = prices.shift(-5)  # 5期后价格
                    buy_opportunity_cost = (future_prices - trades['price']) / trades['price']
                    costs['buy_opportunity_cost'] = buy_opportunity_cost.fillna(0)

                # 卖出交易的机会成本
                sell_mask = trades['direction'] == -1
                if sell_mask.any():
                    future_prices = prices.shift(-5)
                    sell_opportunity_cost = (trades['price'] - future_prices) / trades['price']
                    costs['sell_opportunity_cost'] = sell_opportunity_cost.fillna(0)

        # 市场冲击成本
        impact_decomposition = self.decompose_market_impact(trades, prices)
        costs['market_impact_cost'] = impact_decomposition['total_impact']

        # 流动性成本
        if 'volume' in trades.columns:
            # 基于交易量的流动性成本
            volume_percentile = trades['volume'].rolling(self.volume_profile_window).rank(pct=True)
            liquidity_cost = volume_percentile * 0.001  # 假设最大成本为0.1%
            costs['liquidity_cost'] = liquidity_cost.fillna(0)

        # 时变成本
        if 'volume' in trades.columns:
            trading_intensity = trades['volume'].rolling(10).mean()
            avg_intensity = trading_intensity.mean()
            intensity_ratio = trading_intensity / (avg_intensity + 1e-10)
            timing_cost = intensity_ratio * 0.0005  # 时变成本
            costs['timing_cost'] = timing_cost.fillna(0)

        # 总执行成本
        total_cost = pd.Series(0.0, index=prices.index)
        for cost_component in costs.values():
            total_cost += abs(cost_component)

        costs['total_execution_cost'] = total_cost

        return costs

    def analyze_market_liquidity(self, trades: pd.DataFrame, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        分析市场流动性

        Args:
            trades: 交易数据
            prices: 价格序列

        Returns:
            流动性分析结果
        """
        liquidity = {}

        # 基于价格的流动性指标
        price_volatility = prices.pct_change().rolling(self.impact_window).std().fillna(0)
        liquidity['price_volatility'] = price_volatility

        # 基于交易量的流动性指标
        if 'volume' in trades.columns:
            # 交易量频率
            trade_frequency = trades['volume'].resample('1T').count()
            liquidity['trade_frequency'] = trade_frequency.reindex(prices.index, method='ffill').fillna(0)

            # 交易量深度
            volume_depth = trades['volume'].rolling(self.volume_profile_window).mean()
            liquidity['volume_depth'] = volume_depth.reindex(prices.index, method='ffill').fillna(0)

            # 交易量集中度
            volume_concentration = trades['volume'].rolling(self.volume_profile_window).std() / (trades['volume'].rolling(self.volume_profile_window).mean() + 1e-10)
            liquidity['volume_concentration'] = volume_concentration.reindex(prices.index, method='ffill').fillna(0)

        # 价格影响弹性
        if 'volume' in trades.columns:
            volume_normalized = trades['volume'] / (trades['volume'].rolling(self.volume_profile_window).mean() + 1e-10)
            price_response = prices.pct_change().fillna(0)
            impact_elasticity = abs(price_response) / (volume_normalized + 1e-10)
            liquidity['impact_elasticity'] = impact_elasticity.rolling(10).mean().fillna(1.0)

        # 综合流动性指标
        liquidity_components = []
        if 'price_volatility' in liquidity:
            liquidity_components.append(1 / (1 + liquidity['price_volatility']))

        if 'trade_frequency' in liquidity:
            freq_normalized = liquidity['trade_frequency'] / (liquidity['trade_frequency'].max() + 1e-10)
            liquidity_components.append(freq_normalized)

        if 'volume_depth' in liquidity:
            depth_normalized = liquidity['volume_depth'] / (liquidity['volume_depth'].max() + 1e-10)
            liquidity_components.append(depth_normalized)

        if liquidity_components:
            composite_liquidity = pd.Series(0.0, index=prices.index)
            for component in liquidity_components:
                composite_liquidity += component
            composite_liquidity /= len(liquidity_components)
            liquidity['composite_liquidity'] = composite_liquidity
        else:
            liquidity['composite_liquidity'] = pd.Series(0.5, index=prices.index)

        return liquidity

    def generate_optimal_execution_signals(self, impact_analysis: Dict[str, pd.Series],
                                         cost_analysis: Dict[str, pd.Series]) -> pd.Series:
        """
        生成最优执行信号

        Args:
            impact_analysis: 冲击分析结果
            cost_analysis: 成本分析结果

        Returns:
            执行信号
        """
        # 综合考虑冲击和成本
        if 'total_impact' in impact_analysis and 'total_execution_cost' in cost_analysis:
            impact = impact_analysis['total_impact']
            cost = cost_analysis['total_execution_cost']

            signals = pd.Series(0, index=impact.index)

            # 基于冲击和成本的交易时机
            low_impact_threshold = impact.quantile(0.3)
            high_impact_threshold = impact.quantile(0.7)
            low_cost_threshold = cost.quantile(0.3)
            high_cost_threshold = cost.quantile(0.7)

            # 优化执行时机
            optimal_timing = (impact < low_impact_threshold) & (cost < low_cost_threshold)
            avoid_timing = (impact > high_impact_threshold) & (cost > high_cost_threshold)

            # 基于冲击比率的策略
            if 'temporary_ratio' in impact_analysis:
                temp_ratio = impact_analysis['temporary_ratio']
                # 临时冲击比率高时适合反向操作
                reversal_opportunity = temp_ratio > 0.7
                # 永久冲击比率高时趋势跟随
                momentum_opportunity = impact_analysis['permanent_ratio'] > 0.7

                signals[reversal_opportunity] = 1  # 反向操作
                signals[momentum_opportunity] = 2  # 趋势跟随

            signals[optimal_timing] = 3  # 最佳执行时机
            signals[avoid_timing] = -3  # 避免执行

            return signals
        else:
            return pd.Series()

    def get_comprehensive_analysis(self, trades: pd.DataFrame, prices: pd.Series) -> Dict[str, Any]:
        """
        获取综合市场冲击分析

        Args:
            trades: 交易数据
            prices: 价格序列

        Returns:
            综合分析结果
        """
        results = {}

        # 冲击分解
        impact_decomposition = self.decompose_market_impact(trades, prices)
        results.update(impact_decomposition)

        # 冲击曲线估计
        impact_curve = self.estimate_impact_curve(trades, prices)
        results['impact_curve_parameters'] = impact_curve

        # 执行成本分析
        execution_costs = self.calculate_execution_costs(trades, prices)
        results.update(execution_costs)

        # 流动性分析
        liquidity_analysis = self.analyze_market_liquidity(trades, prices)
        results.update(liquidity_analysis)

        # 生成执行信号
        execution_signals = self.generate_optimal_execution_signals(impact_decomposition, execution_costs)
        results['execution_signals'] = execution_signals

        # 计算执行优化指标
        optimization_metrics = self.calculate_execution_optimization_metrics(results)
        results.update(optimization_metrics)

        return results

    def calculate_execution_optimization_metrics(self, analysis_results: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """
        计算执行优化指标

        Args:
            analysis_results: 分析结果

        Returns:
            优化指标
        """
        optimization = {}

        # 执行效率评分
        if 'total_execution_cost' in analysis_results:
            cost_efficiency = 1 / (1 + analysis_results['total_execution_cost'] / self.cost_threshold)
            optimization['cost_efficiency'] = cost_efficiency.clip(0, 1)

        # 冲击效率评分
        if 'total_impact' in analysis_results:
            impact_efficiency = 1 / (1 + abs(analysis_results['total_impact']))
            optimization['impact_efficiency'] = impact_efficiency.clip(0, 1)

        # 流动性评分
        if 'composite_liquidity' in analysis_results:
            optimization['liquidity_score'] = analysis_results['composite_liquidity']

        # 综合执行评分
        efficiency_components = []
        for component in ['cost_efficiency', 'impact_efficiency', 'liquidity_score']:
            if component in optimization:
                efficiency_components.append(optimization[component])

        if efficiency_components:
            composite_efficiency = pd.Series(0.0, index=efficiency_components[0].index)
            for component in efficiency_components:
                composite_efficiency += component
            composite_efficiency /= len(efficiency_components)
            optimization['composite_execution_score'] = composite_efficiency
        else:
            optimization['composite_execution_score'] = pd.Series(0.5, index=list(analysis_results.values())[0].index)

        return optimization

    def get_trading_recommendations(self, impact_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            impact_analysis: 冲击分析结果

        Returns:
            交易建议DataFrame
        """
        # 找到一个指标序列作为索引
        index_keys = [k for k, v in impact_analysis.items() if isinstance(v, pd.Series) and not v.empty]
        if not index_keys:
            return pd.DataFrame()

        index = impact_analysis[index_keys[0]].index
        recommendations = pd.DataFrame(index=index)

        # 添加冲击分析指标
        for key, value in impact_analysis.items():
            if isinstance(value, pd.Series):
                recommendations[key] = value

        # 执行信号
        recommendations['execution_signals'] = impact_analysis.get('execution_signals', pd.Series(0, index=index))

        # 信号描述
        signal_descriptions = {
            3: '最佳执行时机 - 低冲击低成本',
            2: '趋势跟随 - 永久冲击主导',
            1: '反向操作 - 临时冲击主导',
            0: '正常执行',
            -3: '避免执行 - 高冲击高成本'
        }
        recommendations['signal_description'] = recommendations['execution_signals'].map(signal_descriptions)

        # 执行策略分类
        def classify_execution_strategy(row):
            impact = abs(row.get('total_impact', 0))
            cost = row.get('total_execution_cost', 0)
            liquidity = row.get('composite_liquidity', 0.5)
            temp_ratio = row.get('temporary_ratio', 0.5)
            perm_ratio = row.get('permanent_ratio', 0.5)

            if impact < 0.001 and cost < 0.001:
                return 'immediate_execution'
            elif liquidity > 0.7:
                return 'large_participation'
            elif temp_ratio > 0.7:
                return ' opportunistic_execution'
            elif perm_ratio > 0.7:
                return 'momentum_execution'
            else:
                return 'standard_execution'

        recommendations['execution_strategy'] = recommendations.apply(classify_execution_strategy, axis=1)

        # 执行规模建议
        def calculate_execution_size(signal, liquidity, cost_efficiency):
            base_size = {
                3: 1.0,   # 最佳时机：全额执行
                2: 0.8,   # 趋势跟随：较大规模
                1: 0.6,   # 反向操作：中等规模
                0: 0.4,   # 正常执行：小规模
                -3: 0.0   # 避免执行：不执行
            }

            size = base_size.get(signal, 0.4)
            # 流动性调整
            size *= liquidity
            # 成本效率调整
            if 'cost_efficiency' in recommendations.columns:
                cost_adjustment = recommendations['cost_efficiency']
                size *= cost_adjustment

            return min(size, 1.0)

        recommendations['execution_size'] = recommendations.apply(
            lambda x: calculate_execution_size(x['execution_signals'], x.get('composite_liquidity', 0.5), x.get('cost_efficiency', 0.5)),
            axis=1
        )

        # 执行速度建议
        def recommend_execution_speed(impact_persistence, liquidity):
            if impact_persistence > 0.7:
                return 'fast'  # 高持续性：快速执行
            elif liquidity > 0.7:
                return 'medium'  # 高流动性：中等速度
            else:
                return 'slow'  # 低流动性：慢速执行

        if 'impact_persistence' in recommendations.columns:
            recommendations['execution_speed'] = recommendations.apply(
                lambda x: recommend_execution_speed(x.get('impact_persistence', 0.5), x.get('composite_liquidity', 0.5)),
                axis=1
            )

        # 风险评估
        def assess_execution_risk(impact, cost, liquidity):
            risk_score = 0

            # 冲击风险
            if impact > 0.005:
                risk_score += 0.4
            elif impact > 0.002:
                risk_score += 0.2

            # 成本风险
            if cost > 0.005:
                risk_score += 0.4
            elif cost > 0.002:
                risk_score += 0.2

            # 流动性风险
            if liquidity < 0.3:
                risk_score += 0.2

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['execution_risk'] = recommendations.apply(
            lambda x: assess_execution_risk(abs(x.get('total_impact', 0)), x.get('total_execution_cost', 0), x.get('composite_liquidity', 0.5)),
            axis=1
        )

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成模拟交易和价格数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='5T')

    # 生成价格序列
    base_price = 100
    price_trend = np.cumsum(np.random.randn(500) * 0.002)
    prices = pd.Series(base_price + price_trend, index=dates)

    # 生成交易数据
    n_trades = len(dates)
    trade_prices = prices + np.random.randn(n_trades) * 0.1
    trade_volumes = np.random.lognormal(8, 1, n_trades).astype(int)
    trade_directions = np.random.choice([-1, 0, 1], n_trades, p=[0.3, 0.4, 0.3])

    trades = pd.DataFrame({
        'price': trade_prices,
        'volume': trade_volumes,
        'direction': trade_directions
    }, index=dates)

    # 创建市场冲击分析器
    impact_analyzer = MarketImpactAnalyzer(impact_window=20)

    # 进行综合冲击分析
    print("进行市场冲击分析...")
    analysis_results = impact_analyzer.get_comprehensive_analysis(trades, prices)

    # 显示主要结果
    print(f"临时冲击: {analysis_results['temporary_impact'].iloc[-1]:.6f}")
    print(f"永久冲击: {analysis_results['permanent_impact'].iloc[-1]:.6f}")
    print(f"总冲击: {analysis_results['total_impact'].iloc[-1]:.6f}")
    print(f"冲击持续性: {analysis_results['impact_persistence'].iloc[-1]:.3f}")
    print(f"总执行成本: {analysis_results['total_execution_cost'].iloc[-1]:.6f}")
    print(f"综合执行评分: {analysis_results['composite_execution_score'].iloc[-1]:.3f}")

    # 显示冲击曲线参数
    curve_params = analysis_results.get('impact_curve_parameters', {})
    print(f"\n冲击曲线参数:")
    print(f"临时冲击系数: {curve_params.get('temporary_alpha', 0):.6f}")
    print(f"临时冲击弹性: {curve_params.get('temporary_beta', 0):.3f}")
    print(f"永久相关性: {curve_params.get('permanent_correlation', 0):.3f}")

    print(f"当前执行信号: {analysis_results['execution_signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = impact_analyzer.get_trading_recommendations(analysis_results)
    if not recommendations.empty:
        print("\n最新执行建议:")
        print(recommendations.iloc[-1][['total_impact', 'total_execution_cost', 'signal_description',
                                          'execution_strategy', 'execution_size', 'execution_speed', 'execution_risk']])