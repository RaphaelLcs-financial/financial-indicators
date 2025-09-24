"""
Market Entropy Analyzer
=======================

基于信息论的市场熵分析器，使用香农熵、Rényi熵、
传递熵等信息论度量来分析市场的随机性、可预测性
和复杂性。该指标通过分析价格序列的信息含量，
识别市场状态转换和交易机会。

特点：
- 多种熵度量方法
- 信息含量分析
- 市场效率评估
- 复杂性量化
- 状态转换识别
- 信息流分析
- 预测性评估

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.stats import entropy
from scipy.signal import hilbert
from sklearn.decomposition import PCA
from sklearn.preprocessing import KBinsDiscretizer


class MarketEntropyAnalyzer:
    """
    市场熵分析器

    使用信息论方法分析市场数据，
    量化市场的不确定性和复杂性。
    """

    def __init__(self, window_size: int = 50, n_bins: int = 10,
                 alpha_values: List[float] = [0.5, 1.0, 2.0],
                 entropy_methods: List[str] = ['shannon', 'renyi', 'tsallis']):
        """
        初始化市场熵分析器

        Args:
            window_size: 分析窗口大小，默认50
            n_bins: 离散化分箱数，默认10
            alpha_values: Rényi熵参数，默认[0.5,1.0,2.0]
            entropy_methods: 熵计算方法，默认['shannon','renyi','tsallis']
        """
        self.window_size = window_size
        self.n_bins = n_bins
        self.alpha_values = alpha_values
        self.entropy_methods = entropy_methods
        self.name = f"Market Entropy Analyzer ({window_size})"
        self.category = "information_theory"

    def discretize_time_series(self, time_series: pd.Series, method: str = 'uniform') -> np.ndarray:
        """
        离散化时间序列

        Args:
            time_series: 时间序列数据
            method: 离散化方法 ('uniform', 'quantile', 'kmeans')

        Returns:
            离散化后的序列
        """
        values = time_series.values.reshape(-1, 1)

        if method == 'uniform':
            # 均匀分箱
            bins = np.linspace(values.min(), values.max(), self.n_bins + 1)
            discretized = np.digitize(values.flatten(), bins[1:-1])
        elif method == 'quantile':
            # 分位数分箱
            discretized = pd.qcut(time_series, q=self.n_bins, labels=False, duplicates='drop')
        else:  # kmeans
            # K-means分箱
            kbd = KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='kmeans')
            discretized = kbd.fit_transform(values).flatten().astype(int)

        return discretized

    def calculate_shannon_entropy(self, discretized_series: np.ndarray, base: float = 2) -> float:
        """
        计算香农熵

        Args:
            discretized_series: 离散化序列
            base: 对数底数，默认2

        Returns:
            香农熵值
        """
        # 计算概率分布
        unique, counts = np.unique(discretized_series, return_counts=True)
        probabilities = counts / len(discretized_series)

        # 计算香农熵
        shannon_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10)) / np.log(base)

        return shannon_entropy

    def calculate_renyi_entropy(self, discretized_series: np.ndarray, alpha: float, base: float = 2) -> float:
        """
        计算Rényi熵

        Args:
            discretized_series: 离散化序列
            alpha: Rényi参数
            base: 对数底数

        Returns:
            Rényi熵值
        """
        unique, counts = np.unique(discretized_series, return_counts=True)
        probabilities = counts / len(discretized_series)

        if abs(alpha - 1.0) < 1e-10:
            # alpha=1时退化为香农熵
            return self.calculate_shannon_entropy(discretized_series, base)
        else:
            renyi_entropy = (1 / (1 - alpha)) * np.log(np.sum(probabilities ** alpha) + 1e-10) / np.log(base)
            return renyi_entropy

    def calculate_tsallis_entropy(self, discretized_series: np.ndarray, q: float) -> float:
        """
        计算Tsallis熵

        Args:
            discretized_series: 离散化序列
            q: Tsallis参数

        Returns:
            Tsallis熵值
        """
        unique, counts = np.unique(discretized_series, return_counts=True)
        probabilities = counts / len(discretized_series)

        if abs(q - 1.0) < 1e-10:
            # q=1时退化为香农熵
            return self.calculate_shannon_entropy(discretized_series)
        else:
            tsallis_entropy = (1 / (q - 1)) * (1 - np.sum(probabilities ** q))
            return tsallis_entropy

    def calculate_approximate_entropy(self, time_series: pd.Series, m: int = 2, r: float = 0.2) -> float:
        """
        计算近似熵

        Args:
            time_series: 时间序列
            m: 模式长度，默认2
            r: 容忍度，默认0.2

        Returns:
            近似熵值
        """
        N = len(time_series)
        if N < m + 1:
            return 0.0

        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            patterns = np.array([time_series[i:i+m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)

            for i in range(N - m + 1):
                template_i = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template_i, patterns[j]) <= r:
                        C[i] += 1

            C = C / (N - m + 1)
            phi = np.mean(np.log(C))
            return phi

        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0

    def calculate_sample_entropy(self, time_series: pd.Series, m: int = 2, r: float = 0.2) -> float:
        """
        计算样本熵

        Args:
            time_series: 时间序列
            m: 模式长度，默认2
            r: 容忍度，默认0.2

        Returns:
            样本熵值
        """
        N = len(time_series)
        if N < m + 1:
            return 0.0

        patterns_m = np.array([time_series[i:i+m] for i in range(N - m)])
        patterns_m1 = np.array([time_series[i:i+m+1] for i in range(N - m)])

        def _count_matches(patterns, r):
            B = 0
            for i in range(len(patterns)):
                for j in range(len(patterns)):
                    if i != j:
                        if max(abs(patterns[i] - patterns[j])) <= r:
                            B += 1
            return B

        try:
            B = _count_matches(patterns_m, r)
            A = _count_matches(patterns_m1, r)

            if B == 0 or A == 0:
                return 0.0

            return -np.log(A / B)
        except:
            return 0.0

    def calculate_transfer_entropy(self, source: pd.Series, target: pd.Series, delay: int = 1) -> float:
        """
        计算传递熵

        Args:
            source: 源序列
            target: 目标序列
            delay: 时间延迟，默认1

        Returns:
            传递熵值
        """
        if len(source) != len(target) or len(source) < delay + 2:
            return 0.0

        # 离散化序列
        source_disc = self.discretize_time_series(source)
        target_disc = self.discretize_time_series(target)

        # 计算联合概率
        def calculate_joint_entropy(x, y):
            joint_states = np.column_stack((x, y))
            unique_states, counts = np.unique(joint_states, axis=0, return_counts=True)
            probabilities = counts / len(joint_states)
            return -np.sum(probabilities * np.log(probabilities + 1e-10))

        # 计算不同的熵项
        try:
            # H(target_future | target_past, source_past)
            n = len(target_disc) - delay - 1
            target_future = target_disc[delay+1:]
            target_past = target_disc[1:n+1]
            source_past = source_disc[1:n+1]

            joint_past = np.column_stack((target_past, source_past))
            unique_joint, counts_joint = np.unique(joint_past, axis=0, return_counts=True)
            prob_joint = counts_joint / len(joint_past)

            conditional_entropy = 0
            for i, state in enumerate(unique_joint):
                mask = (joint_past == state).all(axis=1)
                if np.sum(mask) > 0:
                    target_future_subset = target_future[mask]
                    unique_future, counts_future = np.unique(target_future_subset, return_counts=True)
                    prob_future = counts_future / len(target_future_subset)
                    conditional_entropy -= prob_joint[i] * np.sum(prob_future * np.log(prob_future + 1e-10))

            # H(target_future | target_past)
            target_past_only = target_disc[1:n+1]
            target_future_only = target_disc[delay+1:]

            unique_target, counts_target = np.unique(target_past_only, return_counts=True)
            prob_target = counts_target / len(target_past_only)

            marginal_entropy = 0
            for i, state in enumerate(unique_target):
                mask = (target_past_only == state)
                if np.sum(mask) > 0:
                    target_future_subset = target_future_only[mask]
                    unique_future, counts_future = np.unique(target_future_subset, return_counts=True)
                    prob_future = counts_future / len(target_future_subset)
                    marginal_entropy -= prob_target[i] * np.sum(prob_future * np.log(prob_future + 1e-10))

            transfer_entropy = marginal_entropy - conditional_entropy
            return max(0, transfer_entropy)

        except:
            return 0.0

    def analyze_market_entropy(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        分析市场熵

        Args:
            data: OHLCV数据

        Returns:
            熵分析结果
        """
        close_prices = data['close']
        returns = close_prices.pct_change().fillna(0)

        results = {}

        # 计算滚动香农熵
        shannon_entropy = pd.Series(0.0, index=close_prices.index)
        for i in range(self.window_size, len(close_prices)):
            window_returns = returns.iloc[i-self.window_size:i]
            discretized = self.discretize_time_series(window_returns)
            shannon_entropy.iloc[i] = self.calculate_shannon_entropy(discretized)
        results['shannon_entropy'] = shannon_entropy

        # 计算Rényi熵
        for alpha in self.alpha_values:
            renyi_entropy = pd.Series(0.0, index=close_prices.index)
            for i in range(self.window_size, len(close_prices)):
                window_returns = returns.iloc[i-self.window_size:i]
                discretized = self.discretize_time_series(window_returns)
                renyi_entropy.iloc[i] = self.calculate_renyi_entropy(discretized, alpha)
            results[f'renyi_entropy_alpha_{alpha}'] = renyi_entropy

        # 计算Tsallis熵
        for q in self.alpha_values:
            tsallis_entropy = pd.Series(0.0, index=close_prices.index)
            for i in range(self.window_size, len(close_prices)):
                window_returns = returns.iloc[i-self.window_size:i]
                discretized = self.discretize_time_series(window_returns)
                tsallis_entropy.iloc[i] = self.calculate_tsallis_entropy(discretized, q)
            results[f'tsallis_entropy_q_{q}'] = tsallis_entropy

        # 计算近似熵和样本熵
        approximate_entropy = pd.Series(0.0, index=close_prices.index)
        sample_entropy = pd.Series(0.0, index=close_prices.index)

        for i in range(self.window_size, len(close_prices)):
            window_prices = close_prices.iloc[i-self.window_size:i]
            approximate_entropy.iloc[i] = self.calculate_approximate_entropy(window_prices)
            sample_entropy.iloc[i] = self.calculate_sample_entropy(window_prices)

        results['approximate_entropy'] = approximate_entropy
        results['sample_entropy'] = sample_entropy

        # 计算传递熵（价格到成交量）
        if 'volume' in data.columns:
            transfer_entropy = pd.Series(0.0, index=close_prices.index)
            for i in range(self.window_size, len(close_prices)):
                window_prices = close_prices.iloc[i-self.window_size:i]
                window_volume = data['volume'].iloc[i-self.window_size:i]
                transfer_entropy.iloc[i] = self.calculate_transfer_entropy(window_prices, window_volume)
            results['transfer_entropy'] = transfer_entropy

        # 计算信息效率
        results['information_efficiency'] = self.calculate_information_efficiency(results)

        # 计算市场复杂性
        results['market_complexity'] = self.calculate_market_complexity(results)

        # 生成交易信号
        results['signals'] = self.generate_entropy_signals(results)

        return results

    def calculate_information_efficiency(self, entropy_results: Dict[str, pd.Series]) -> pd.Series:
        """
        计算信息效率

        Args:
            entropy_results: 熵计算结果

        Returns:
            信息效率序列
        """
        shannon_entropy = entropy_results['shannon_entropy']

        # 最大熵（完全随机）
        max_entropy = np.log2(self.n_bins)

        # 信息效率 = 1 - (实际熵 / 最大熵)
        efficiency = 1 - (shannon_entropy / max_entropy)
        efficiency = efficiency.clip(0, 1)

        return efficiency

    def calculate_market_complexity(self, entropy_results: Dict[str, pd.Series]) -> pd.Series:
        """
        计算市场复杂性

        Args:
            entropy_results: 熵计算结果

        Returns:
            市场复杂性序列
        """
        # 综合多种熵度量
        complexity_components = []

        if 'shannon_entropy' in entropy_results:
            complexity_components.append(entropy_results['shannon_entropy'])

        if 'approximate_entropy' in entropy_results:
            complexity_components.append(entropy_results['approximate_entropy'])

        if 'sample_entropy' in entropy_results:
            complexity_components.append(entropy_results['sample_entropy'])

        if complexity_components:
            # 标准化并加权平均
            complexity = pd.Series(0.0, index=complexity_components[0].index)
            for component in complexity_components:
                normalized_component = (component - component.mean()) / (component.std() + 1e-10)
                complexity += normalized_component

            complexity = complexity / len(complexity_components)
            # 转换为0-1范围
            complexity = (complexity - complexity.min()) / (complexity.max() - complexity.min() + 1e-10)
        else:
            complexity = pd.Series(0.5, index=entropy_results['shannon_entropy'].index)

        return complexity

    def generate_entropy_signals(self, entropy_results: Dict[str, pd.Series]) -> pd.Series:
        """
        生成基于熵的交易信号

        Args:
            entropy_results: 熵分析结果

        Returns:
            交易信号
        """
        signals = pd.Series(0, index=entropy_results['shannon_entropy'].index)

        shannon_entropy = entropy_results['shannon_entropy']
        information_efficiency = entropy_results.get('information_efficiency', pd.Series(0.5, index=shannon_entropy.index))
        market_complexity = entropy_results.get('market_complexity', pd.Series(0.5, index=shannon_entropy.index))

        # 计算熵的变化率
        entropy_change = shannon_entropy.diff()

        # 识别市场状态转换
        low_entropy_threshold = shannon_entropy.quantile(0.3)
        high_entropy_threshold = shannon_entropy.quantile(0.7)

        # 低熵信号（有序状态，趋势可能）
        low_entropy_state = shannon_entropy < low_entropy_threshold
        # 高熵信号（混沌状态，均值回归可能）
        high_entropy_state = shannon_entropy > high_entropy_threshold

        # 信息效率信号
        high_efficiency = information_efficiency > 0.7
        low_efficiency = information_efficiency < 0.3

        # 熵变化信号
        decreasing_entropy = entropy_change < -entropy_change.std()
        increasing_entropy = entropy_change > entropy_change.std()

        # 生成交易信号
        # 低熵 + 效率上升 = 趋势形成
        trend_formation = low_entropy_state & high_efficiency
        # 高熵 + 低效率 = 混沌状态，避免交易
        chaos_state = high_entropy_state & low_efficiency
        # 熵下降 = 秩序增加
        order_increasing = decreasing_entropy & low_entropy_state
        # 熵上升 = 混沌增加
        chaos_increasing = increasing_entropy & high_entropy_state

        signals[trend_formation] = 2
        signals[order_increasing] = 1
        signals[chaos_increasing] = -1
        signals[chaos_state] = -2

        return signals

    def get_trading_recommendations(self, entropy_analysis: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            entropy_analysis: 熵分析结果

        Returns:
            交易建议DataFrame
        """
        recommendations = pd.DataFrame(index=entropy_analysis['shannon_entropy'].index)

        # 熵指标
        recommendations['shannon_entropy'] = entropy_analysis['shannon_entropy']
        recommendations['information_efficiency'] = entropy_analysis.get('information_efficiency', pd.Series(0.5))
        recommendations['market_complexity'] = entropy_analysis.get('market_complexity', pd.Series(0.5))

        # 添加其他熵指标
        for key, value in entropy_analysis.items():
            if 'entropy' in key and key not in ['shannon_entropy', 'information_efficiency', 'market_complexity']:
                recommendations[key] = value

        # 交易信号
        recommendations['signals'] = entropy_analysis['signals']

        # 信号描述
        signal_descriptions = {
            2: '买入 - 趋势形成期',
            1: '弱买入 - 秩序增加',
            0: '持有 - 熵稳定',
            -1: '弱卖出 - 混沌增加',
            -2: '卖出 - 混沌状态'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 市场状态分类
        def classify_market_state(shannon_entropy, efficiency, complexity):
            if efficiency > 0.7 and shannon_entropy < shannon_entropy.quantile(0.3):
                return 'efficient_trend'
            elif efficiency < 0.3 and shannon_entropy > shannon_entropy.quantile(0.7):
                return 'inefficient_chaos'
            elif complexity > 0.7:
                return 'high_complexity'
            else:
                return 'normal'

        recommendations['market_state'] = recommendations.apply(
            lambda x: classify_market_state(x['shannon_entropy'], x['information_efficiency'], x['market_complexity']),
            axis=1
        )

        # 仓位建议
        position_map = {2: 0.6, 1: 0.3, 0: 0.2, -1: 0.1, -2: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 效率调整
        efficiency_adjustment = recommendations['information_efficiency']
        recommendations['position_size'] *= efficiency_adjustment

        # 策略建议
        strategy_map = {
            'efficient_trend': 'trend_following',
            'inefficient_chaos': 'avoid_trading',
            'high_complexity': 'mean_reversion',
            'normal': 'balanced'
        }
        recommendations['strategy'] = recommendations['market_state'].map(strategy_map)

        # 风险评估
        recommendations['risk_level'] = recommendations.apply(
            lambda x: 'high' if x['market_state'] == 'inefficient_chaos' else
                     'medium' if x['market_state'] == 'high_complexity' else 'low',
            axis=1
        )

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    n = len(dates)

    # 生成价格数据
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volumes = np.random.randint(1000000, 5000000, n)

    data = pd.DataFrame({
        'close': close_prices,
        'volume': volumes
    }, index=dates)

    # 创建市场熵分析器
    entropy_analyzer = MarketEntropyAnalyzer(
        window_size=30,
        n_bins=8,
        alpha_values=[0.5, 1.0, 2.0]
    )

    # 分析市场熵
    print("分析市场熵...")
    entropy_results = entropy_analyzer.analyze_market_entropy(data)

    print(f"最新香农熵: {entropy_results['shannon_entropy'].iloc[-1]:.3f}")
    print(f"信息效率: {entropy_results['information_efficiency'].iloc[-1]:.3f}")
    print(f"市场复杂性: {entropy_results['market_complexity'].iloc[-1]:.3f}")
    print(f"近似熵: {entropy_results['approximate_entropy'].iloc[-1]:.3f}")
    print(f"样本熵: {entropy_results['sample_entropy'].iloc[-1]:.3f}")

    if 'transfer_entropy' in entropy_results:
        print(f"传递熵: {entropy_results['transfer_entropy'].iloc[-1]:.3f}")

    for alpha in [0.5, 2.0]:
        key = f'renyi_entropy_alpha_{alpha}'
        if key in entropy_results:
            print(f"Rényi熵(α={alpha}): {entropy_results[key].iloc[-1]:.3f}")

    print(f"当前信号: {entropy_results['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = entropy_analyzer.get_trading_recommendations(entropy_results)
    print("\n最新交易建议:")
    print(recommendations.iloc[-1][['shannon_entropy', 'information_efficiency', 'market_complexity',
                                      'signal_description', 'market_state', 'strategy', 'position_size']])