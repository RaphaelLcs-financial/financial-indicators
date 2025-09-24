"""
Mutual Information Analyzer
============================

基于互信息的市场关联分析器，使用信息论中的
互信息、条件互信息、多变量互信息等度量来分析
市场间的信息流动、因果关系和协同运动模式。

特点：
- 信息流分析
- 因果关系识别
- 市场协同性评估
- 信息传递效率
- 多变量信息分解
- 网络信息流
- 领先滞后关系

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-2024
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.stats import gaussian_kde
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')


class MutualInformationAnalyzer:
    """
    互信息分析器

    使用信息论方法分析市场间的信息流动
    和相关性结构。
    """

    def __init__(self, window_size: int = 50, n_bins: int = 10,
                 k_neighbors: int = 5, time_delays: List[int] = [0, 1, 5, 10]):
        """
        初始化互信息分析器

        Args:
            window_size: 分析窗口大小，默认50
            n_bins: 离散化分箱数，默认10
            k_neighbors: k近邻方法的邻居数，默认5
            time_delays: 时间延迟列表，默认[0,1,5,10]
        """
        self.window_size = window_size
        self.n_bins = n_bins
        self.k_neighbors = k_neighbors
        self.time_delays = time_delays
        self.name = f"Mutual Information Analyzer ({window_size})"
        self.category = "information_theory"

    def discretize_series(self, series: pd.Series) -> np.ndarray:
        """
        离散化时间序列

        Args:
            series: 时间序列

        Returns:
            离散化后的序列
        """
        # 使用分位数分箱
        try:
            discretized = pd.qcut(series, q=self.n_bins, labels=False, duplicates='drop')
        except:
            # 如果分位数分箱失败，使用均匀分箱
            bins = np.linspace(series.min(), series.max(), self.n_bins + 1)
            discretized = np.digitize(series, bins[1:-1])

        return discretized.values

    def calculate_mutual_information_discrete(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算离散变量的互信息

        Args:
            x: 离散序列x
            y: 离散序列y

        Returns:
            互信息值
        """
        # 计算联合概率
        xy = np.column_stack((x, y))
        unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
        p_xy = counts_xy / len(xy)

        # 计算边缘概率
        unique_x, counts_x = np.unique(x, return_counts=True)
        p_x = counts_x / len(x)

        unique_y, counts_y = np.unique(y, return_counts=True)
        p_y = counts_y / len(y)

        # 计算互信息
        mi = 0.0
        for i, xy_val in enumerate(unique_xy):
            x_val, y_val = xy_val
            x_idx = np.where(unique_x == x_val)[0][0]
            y_idx = np.where(unique_y == y_val)[0][0]

            if p_xy[i] > 0 and p_x[x_idx] > 0 and p_y[y_idx] > 0:
                mi += p_xy[i] * np.log2(p_xy[i] / (p_x[x_idx] * p_y[y_idx]))

        return max(0, mi)

    def calculate_mutual_information_knn(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        使用k近邻方法计算连续变量的互信息

        Args:
            x: 连续序列x
            y: 连续序列y

        Returns:
            互信息值
        """
        n = len(x)
        if n < 2:
            return 0.0

        # 合并数据
        xy = np.column_stack((x, y))

        # 计算k近邻距离
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1).fit(xy)
        distances, _ = nbrs.kneighbors(xy)

        # epsilon = k+1近邻的距离
        epsilon = distances[:, -1]

        # 计算在x和y空间中的近邻数
        x_space = x.reshape(-1, 1)
        y_space = y.reshape(-1, 1)

        nbrs_x = NearestNeighbors(radius=epsilon[0]).fit(x_space)
        nbrs_y = NearestNeighbors(radius=epsilon[0]).fit(y_space)

        # 计算互信息
        mi = 0.0
        for i in range(n):
            n_x = nbrs_x.radius_neighbors([x_space[i]], count_only=True)[0]
            n_y = nbrs_y.radius_neighbors([y_space[i]], count_only=True)[0]

            if n_x > 0 and n_y > 0:
                mi += np.log(n) + np.log(epsilon[i]) - np.log(n_x) - np.log(n_y)

        mi = max(0, mi / n - np.log(self.k_neighbors))
        return mi

    def calculate_transfer_entropy(self, source: pd.Series, target: pd.Series, delay: int = 1) -> float:
        """
        计算传递熵

        Args:
            source: 源序列
            target: 目标序列
            delay: 时间延迟

        Returns:
            传递熵值
        """
        if len(source) != len(target) or len(source) < delay + 2:
            return 0.0

        # 离散化
        source_disc = self.discretize_series(source)
        target_disc = self.discretize_series(target)

        n = len(target_disc) - delay - 1

        # 准备序列
        target_future = target_disc[delay+1:]
        target_past = target_disc[1:n+1]
        source_past = source_disc[1:n+1]

        # 计算条件熵
        def conditional_entropy(x, y):
            """计算H(X|Y)"""
            joint = np.column_stack((x, y))
            unique_joint, counts_joint = np.unique(joint, axis=0, return_counts=True)
            unique_y, counts_y = np.unique(y, return_counts=True)

            cond_entropy = 0.0
            for i, joint_state in enumerate(unique_joint):
                mask = (joint == joint_state).all(axis=1)
                if np.sum(mask) > 0:
                    prob_joint = counts_joint[i] / len(joint)
                    y_state = joint_state[1]
                    y_idx = np.where(unique_y == y_state)[0][0]
                    prob_y = counts_y[y_idx] / len(y)
                    cond_entropy -= prob_joint * np.log(prob_joint / prob_y)

            return cond_entropy

        try:
            # H(target_future | target_past)
            h_target_given_target = conditional_entropy(target_future, target_past)

            # H(target_future | target_past, source_past)
            joint_past = np.column_stack((target_past, source_past))
            h_target_given_both = conditional_entropy(target_future, joint_past)

            transfer_entropy = h_target_given_target - h_target_given_both
            return max(0, transfer_entropy)

        except:
            return 0.0

    def calculate_conditional_mutual_information(self, x: pd.Series, y: pd.Series, z: pd.Series) -> float:
        """
        计算条件互信息 I(X;Y|Z)

        Args:
            x: 序列X
            y: 序列Y
            z: 条件序列Z

        Returns:
            条件互信息值
        """
        if len(x) != len(y) or len(x) != len(z):
            return 0.0

        # 离散化
        x_disc = self.discretize_series(x)
        y_disc = self.discretize_series(y)
        z_disc = self.discretize_series(z)

        # 计算各种概率分布
        xyz = np.column_stack((x_disc, y_disc, z_disc))
        unique_xyz, counts_xyz = np.unique(xyz, axis=0, return_counts=True)
        p_xyz = counts_xyz / len(xyz)

        xy = np.column_stack((x_disc, y_disc))
        unique_xy, counts_xy = np.unique(xy, axis=0, return_counts=True)
        p_xy = counts_xy / len(xy)

        xz = np.column_stack((x_disc, z_disc))
        unique_xz, counts_xz = np.unique(xz, axis=0, return_counts=True)
        p_xz = counts_xz / len(xz)

        yz = np.column_stack((y_disc, z_disc))
        unique_yz, counts_yz = np.unique(yz, axis=0, return_counts=True)
        p_yz = counts_yz / len(yz)

        unique_z, counts_z = np.unique(z_disc, return_counts=True)
        p_z = counts_z / len(z_disc)

        # 计算条件互信息
        cmi = 0.0
        for i, xyz_state in enumerate(unique_xyz):
            x_val, y_val, z_val = xyz_state

            # 查找对应的联合概率
            xy_idx = np.where((unique_xy == [x_val, y_val]).all(axis=1))[0]
            xz_idx = np.where((unique_xz == [x_val, z_val]).all(axis=1))[0]
            yz_idx = np.where((unique_yz == [y_val, z_val]).all(axis=1))[0]
            z_idx = np.where(unique_z == z_val)[0]

            if (p_xyz[i] > 0 and p_xy[xy_idx] > 0 and
                p_xz[xz_idx] > 0 and p_yz[yz_idx] > 0 and p_z[z_idx] > 0):

                cmi += p_xyz[i] * np.log2(
                    (p_xyz[i] * p_z[z_idx]) / (p_xz[xz_idx] * p_yz[yz_idx])
                )

        return max(0, cmi)

    def analyze_multivariate_information(self, data_dict: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        分析多变量信息结构

        Args:
            data_dict: 多变量数据字典

        Returns:
            多变量信息分析结果
        """
        variables = list(data_dict.keys())
        n_vars = len(variables)

        if n_vars < 2:
            return {'error': 'Insufficient variables'}

        results = {}

        # 计算所有变量对的互信息
        pairwise_mi = pd.DataFrame(index=variables, columns=variables, dtype=float)

        for i, j in combinations(range(n_vars), 2):
            var1, var2 = variables[i], variables[j]

            # 使用滑动窗口计算互信息
            mi_series = pd.Series(0.0, index=data_dict[var1].index)

            for t in range(self.window_size, len(data_dict[var1])):
                window_var1 = data_dict[var1].iloc[t-self.window_size:t]
                window_var2 = data_dict[var2].iloc[t-self.window_size:t]

                # 离散化并计算互信息
                disc1 = self.discretize_series(window_var1)
                disc2 = self.discretize_series(window_var2)

                mi = self.calculate_mutual_information_discrete(disc1, disc2)
                mi_series.iloc[t] = mi

            pairwise_mi.loc[var1, var2] = mi_series.iloc[-1]
            pairwise_mi.loc[var2, var1] = mi_series.iloc[-1]

            results[f'mi_{var1}_{var2}'] = mi_series

        # 计算传递熵矩阵
        transfer_entropy_matrix = pd.DataFrame(index=variables, columns=variables, dtype=float)

        for i, j in combinations(range(n_vars), 2):
            var1, var2 = variables[i], variables[j]

            # 计算双向传递熵
            te_12 = self.calculate_transfer_entropy(data_dict[var1], data_dict[var2])
            te_21 = self.calculate_transfer_entropy(data_dict[var2], data_dict[var1])

            transfer_entropy_matrix.loc[var1, var2] = te_12
            transfer_entropy_matrix.loc[var2, var1] = te_21

        results['transfer_entropy_matrix'] = transfer_entropy_matrix

        # 计算多变量协同信息
        if n_vars >= 3:
            multivariate_info = self.calculate_multivariate_synergy(data_dict)
            results.update(multivariate_info)

        # 计算信息流网络指标
        network_metrics = self.calculate_network_metrics(pairwise_mi, transfer_entropy_matrix)
        results.update(network_metrics)

        # 生成交易信号
        trading_signals = self.generate_trading_signals(results, variables)
        results.update(trading_signals)

        return results

    def calculate_multivariate_synergy(self, data_dict: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        计算多变量协同信息

        Args:
            data_dict: 多变量数据

        Returns:
            协同信息分析结果
        """
        variables = list(data_dict.keys())
        synergy_results = {}

        # 计算总相关 (Total Correlation)
        def calculate_total_correlation(series_list):
            """计算总相关"""
            # 离散化所有序列
            disc_series = [self.discretize_series(s) for s in series_list]

            # 计算联合熵
            joint_array = np.column_stack(disc_series)
            unique_joint, counts_joint = np.unique(joint_array, axis=0, return_counts=True)
            p_joint = counts_joint / len(joint_array)
            joint_entropy = -np.sum(p_joint * np.log2(p_joint + 1e-10))

            # 计算边缘熵之和
            marginal_entropies = []
            for disc in disc_series:
                unique, counts = np.unique(disc, return_counts=True)
                p = counts / len(disc)
                marginal_entropies.append(-np.sum(p * np.log2(p + 1e-10)))

            total_correlation = sum(marginal_entropies) - joint_entropy
            return max(0, total_correlation)

        # 计算交互信息 (Interaction Information)
        def calculate_interaction_information(series_list):
            """计算交互信息"""
            if len(series_list) < 3:
                return 0.0

            disc_series = [self.discretize_series(s) for s in series_list]

            # 计算各种熵项
            def entropy(series):
                unique, counts = np.unique(series, return_counts=True)
                p = counts / len(series)
                return -np.sum(p * np.log2(p + 1e-10))

            # 单变量熵
            h_individual = [entropy(disc) for disc in disc_series]

            # 两两联合熵
            h_pairs = []
            for i, j in combinations(range(len(disc_series)), 2):
                joint = np.column_stack((disc_series[i], disc_series[j]))
                unique_joint, counts_joint = np.unique(joint, axis=0, return_counts=True)
                p_joint = counts_joint / len(joint)
                h_pairs.append(-np.sum(p_joint * np.log2(p_joint + 1e-10)))

            # 三变量联合熵
            if len(disc_series) >= 3:
                joint_all = np.column_stack(disc_series[:3])
                unique_joint, counts_joint = np.unique(joint_all, axis=0, return_counts=True)
                p_joint = counts_joint / len(joint_all)
                h_triple = -np.sum(p_joint * np.log2(p_joint + 1e-10))

                # 交互信息
                interaction = (h_individual[0] + h_individual[1] + h_individual[2] -
                             h_pairs[0] - h_pairs[1] - h_pairs[2] + h_triple)
                return interaction

            return 0.0

        # 滑动窗口计算
        tc_series = pd.Series(0.0, index=data_dict[variables[0]].index)
        ii_series = pd.Series(0.0, index=data_dict[variables[0]].index)

        for t in range(self.window_size, len(data_dict[variables[0]])):
            window_data = [data_dict[var].iloc[t-self.window_size:t] for var in variables]

            if len(window_data) >= 2:
                tc = calculate_total_correlation(window_data)
                tc_series.iloc[t] = tc

            if len(window_data) >= 3:
                ii = calculate_interaction_information(window_data)
                ii_series.iloc[t] = ii

        synergy_results['total_correlation'] = tc_series
        synergy_results['interaction_information'] = ii_series

        return synergy_results

    def calculate_network_metrics(self, mi_matrix: pd.DataFrame, te_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        计算网络指标

        Args:
            mi_matrix: 互信息矩阵
            te_matrix: 传递熵矩阵

        Returns:
            网络指标
        """
        network_results = {}

        # 信息流强度
        information_flow_strength = te_matrix.sum(axis=1) / (te_matrix.sum().sum() + 1e-10)
        network_results['information_flow_strength'] = pd.Series(information_flow_strength)

        # 网络中心性
        centrality = mi_matrix.sum(axis=1) / (mi_matrix.sum().sum() + 1e-10)
        network_results['network_centrality'] = pd.Series(centrality)

        # 信息传递效率
        def calculate_efficiency(matrix):
            """计算网络效率"""
            n = len(matrix)
            if n <= 1:
                return 1.0

            # 计算最短路径（这里简化处理）
            efficiency = 0.0
            for i in range(n):
                for j in range(n):
                    if i != j:
                        if matrix.iloc[i, j] > 0:
                            efficiency += 1 / matrix.iloc[i, j]

            return efficiency / (n * (n - 1))

        mi_efficiency = calculate_efficiency(mi_matrix)
        te_efficiency = calculate_efficiency(te_matrix)

        network_results['mi_network_efficiency'] = mi_efficiency
        network_results['te_network_efficiency'] = te_efficiency

        # 网络连通性
        mi_density = (mi_matrix > 0).sum().sum() / (len(mi_matrix) * (len(mi_matrix) - 1))
        te_density = (te_matrix > 0).sum().sum() / (len(te_matrix) * (len(te_matrix) - 1))

        network_results['mi_network_density'] = mi_density
        network_results['te_network_density'] = te_density

        return network_results

    def generate_trading_signals(self, results: Dict[str, Any], variables: List[str]) -> Dict[str, pd.Series]:
        """
        生成交易信号

        Args:
            results: 分析结果
            variables: 变量列表

        Returns:
            交易信号
        """
        signals = {}

        # 为每个变量生成信号
        for var in variables:
            signal_series = pd.Series(0, index=results[f'mi_{variables[0]}_{variables[1]}'].index)

            # 基于信息流强度生成信号
            if 'information_flow_strength' in results:
                flow_strength = results['information_flow_strength']
                high_flow = flow_strength > flow_strength.quantile(0.7)
                low_flow = flow_strength < flow_strength.quantile(0.3)

                signal_series[high_flow] = 1
                signal_series[low_flow] = -1

            # 基于网络中心性生成信号
            if 'network_centrality' in results:
                centrality = results['network_centrality']
                high_centrality = centrality > centrality.quantile(0.7)

                signal_series[high_centrality] = signal_series.clip(0, 2)

            # 基于协同信息生成信号
            if 'total_correlation' in results:
                tc = results['total_correlation']
                high_synergy = tc > tc.quantile(0.7)
                low_synergy = tc < tc.quantile(0.3)

                signal_series[high_synergy] = signal_series.where(signal_series == 0, 2).fillna(2)
                signal_series[low_synergy] = signal_series.where(signal_series == 0, -2).fillna(-2)

            signals[f'signal_{var}'] = signal_series

        return signals

    def get_trading_recommendations(self, info_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            info_analysis: 信息分析结果

        Returns:
            交易建议DataFrame
        """
        # 找到一个信号序列作为索引
        signal_keys = [k for k in info_analysis.keys() if k.startswith('signal_')]
        if not signal_keys:
            return pd.DataFrame()

        index = info_analysis[signal_keys[0]].index
        recommendations = pd.DataFrame(index=index)

        # 添加互信息指标
        mi_keys = [k for k in info_analysis.keys() if k.startswith('mi_')]
        for key in mi_keys:
            recommendations[key] = info_analysis[key]

        # 添加网络指标
        network_keys = ['information_flow_strength', 'network_centrality',
                       'mi_network_efficiency', 'te_network_efficiency']
        for key in network_keys:
            if key in info_analysis:
                if isinstance(info_analysis[key], pd.Series):
                    recommendations[key] = info_analysis[key]
                else:
                    recommendations[key] = info_analysis[key]

        # 添加协同信息指标
        synergy_keys = ['total_correlation', 'interaction_information']
        for key in synergy_keys:
            if key in info_analysis:
                recommendations[key] = info_analysis[key]

        # 添加信号
        for key in signal_keys:
            recommendations[key] = info_analysis[key]

        # 计算综合信号
        if len(signal_keys) > 0:
            signal_cols = [col for col in recommendations.columns if col.startswith('signal_')]
            if signal_cols:
                recommendations['composite_signal'] = recommendations[signal_cols].mean(axis=1)

        # 信号描述
        def describe_signal(signal_value):
            if signal_value > 1.5:
                return '强烈买入 - 高信息流'
            elif signal_value > 0.5:
                return '买入 - 正向信息流'
            elif signal_value > -0.5:
                return '持有 - 信息平衡'
            elif signal_value > -1.5:
                return '卖出 - 负向信息流'
            else:
                return '强烈卖出 - 低信息流'

        if 'composite_signal' in recommendations.columns:
            recommendations['signal_description'] = recommendations['composite_signal'].apply(describe_signal)

        # 仓位建议
        if 'composite_signal' in recommendations.columns:
            def calculate_position_size(signal, flow_strength=None):
                base_size = min(abs(signal) / 2, 1.0) * 0.5

                if flow_strength is not None:
                    # 信息流强度调整
                    flow_adjustment = 1 + flow_strength
                    base_size *= flow_adjustment

                return min(base_size, 1.0)

            if 'information_flow_strength' in recommendations.columns:
                recommendations['position_size'] = recommendations.apply(
                    lambda x: calculate_position_size(x['composite_signal'], x.get('information_flow_strength')),
                    axis=1
                )
            else:
                recommendations['position_size'] = recommendations['composite_signal'].apply(
                    lambda x: calculate_position_size(x)
                )

        # 信息效率评估
        def assess_information_efficiency(row):
            efficiency_score = 0

            # 网络效率
            if 'mi_network_efficiency' in row.index:
                efficiency_score += row['mi_network_efficiency']

            # 信息流强度
            if 'information_flow_strength' in row.index:
                efficiency_score += row['information_flow_strength']

            # 协同信息
            if 'total_correlation' in row.index:
                efficiency_score += row['total_correlation'] / row['total_correlation'].max()

            return min(efficiency_score / 3, 1.0) if efficiency_score > 0 else 0.5

        recommendations['information_efficiency'] = recommendations.apply(assess_information_efficiency, axis=1)

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    n = len(dates)

    # 生成多个相关的时间序列
    base_trend = np.cumsum(np.random.randn(n) * 0.1)

    price1 = 100 + base_trend + np.random.randn(n) * 0.5
    price2 = 105 + base_trend * 0.8 + np.random.randn(n) * 0.3
    price3 = 98 + base_trend * 1.2 + np.random.randn(n) * 0.4

    volume = np.random.randint(1000000, 5000000, n)

    data_dict = {
        'price1': pd.Series(price1, index=dates),
        'price2': pd.Series(price2, index=dates),
        'price3': pd.Series(price3, index=dates),
        'volume': pd.Series(volume, index=dates)
    }

    # 创建互信息分析器
    mi_analyzer = MutualInformationAnalyzer(window_size=30)

    # 分析多变量信息
    print("分析多变量信息结构...")
    info_results = mi_analyzer.analyze_multivariate_information(data_dict)

    # 显示互信息矩阵
    print("\n互信息矩阵:")
    mi_matrix = info_results.get('mi_matrix', pd.DataFrame())
    if not mi_matrix.empty:
        print(mi_matrix)

    # 显示传递熵矩阵
    print("\n传递熵矩阵:")
    te_matrix = info_results.get('transfer_entropy_matrix', pd.DataFrame())
    if not te_matrix.empty:
        print(te_matrix)

    # 显示网络指标
    if 'information_flow_strength' in info_results:
        print(f"\n信息流强度:")
        print(info_results['information_flow_strength'])

    if 'network_centrality' in info_results:
        print(f"\n网络中心性:")
        print(info_results['network_centrality'])

    if 'total_correlation' in info_results:
        print(f"\n总相关 (最新): {info_results['total_correlation'].iloc[-1]:.3f}")

    # 显示信号
    signal_keys = [k for k in info_results.keys() if k.startswith('signal_')]
    if signal_keys:
        print(f"\n交易信号:")
        for key in signal_keys:
            print(f"{key}: {info_results[key].iloc[-1]:.2f}")

    # 获取交易建议
    recommendations = mi_analyzer.get_trading_recommendations(info_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        key_cols = [col for col in recommendations.columns if not col.startswith('mi_')]
        print(recommendations.iloc[-1][key_cols])