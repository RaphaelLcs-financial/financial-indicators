"""
Market Network Analyzer
=======================

基于复杂网络理论的市场网络分析器，将金融市场
建模为复杂网络，分析市场拓扑结构、社区发现、
传染效应和系统性风险。该指标通过相关性网络、
信息流网络等多种网络类型来理解市场结构。

特点：
- 网络拓扑分析
- 社区结构发现
- 系统性风险评估
- 传染效应建模
- 中心性分析
- 网络鲁棒性
- 动态网络演化

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class MarketNetworkAnalyzer:
    """
    市场网络分析器

    使用复杂网络理论分析市场结构，
    识别系统性风险和传染路径。
    """

    def __init__(self, window_size: int = 50, correlation_threshold: float = 0.5,
                 network_types: List[str] = ['correlation', 'partial_correlation', 'mutual_information']):
        """
        初始化市场网络分析器

        Args:
            window_size: 分析窗口大小，默认50
            correlation_threshold: 相关性阈值，默认0.5
            network_types: 网络类型列表
        """
        self.window_size = window_size
        self.correlation_threshold = correlation_threshold
        self.network_types = network_types
        self.name = f"Market Network Analyzer ({window_size})"
        self.category = "complex_networks"

    def calculate_correlation_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """
        计算相关性矩阵

        Args:
            data: 多变量数据

        Returns:
            相关性矩阵
        """
        returns = data.pct_change().fillna(0)
        correlation_matrix = returns.corr().fillna(0).values
        return correlation_matrix

    def calculate_partial_correlation_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """
        计算偏相关矩阵

        Args:
            data: 多变量数据

        Returns:
            偏相关矩阵
        """
        returns = data.pct_change().fillna(0)
        correlation_matrix = returns.corr().fillna(0).values

        # 计算偏相关矩阵
        try:
            precision_matrix = np.linalg.inv(correlation_matrix)
            d = np.sqrt(np.diag(precision_matrix))
            partial_correlation = -precision_matrix / np.outer(d, d)
            np.fill_diagonal(partial_correlation, 1)
            return partial_correlation
        except:
            return correlation_matrix

    def build_correlation_network(self, correlation_matrix: np.ndarray, assets: List[str]) -> nx.Graph:
        """
        构建相关性网络

        Args:
            correlation_matrix: 相关性矩阵
            assets: 资产列表

        Returns:
            相关性网络图
        """
        G = nx.Graph()

        # 添加节点
        for i, asset in enumerate(assets):
            G.add_node(asset, index=i)

        # 添加边（基于相关性阈值）
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                correlation = correlation_matrix[i, j]
                if abs(correlation) > self.correlation_threshold:
                    G.add_edge(assets[i], assets[j], weight=abs(correlation), correlation=correlation)

        return G

    def build_minimum_spanning_tree(self, correlation_matrix: np.ndarray, assets: List[str]) -> nx.Graph:
        """
        构建最小生成树

        Args:
            correlation_matrix: 相关性矩阵
            assets: 资产列表

        Returns:
            最小生成树
        """
        # 创建距离矩阵（将相关性转换为距离）
        distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

        # 构建完全图
        G = nx.complete_graph(len(assets))

        # 设置边权重
        for i in range(len(assets)):
            for j in range(len(assets)):
                if i != j:
                    G[i][j]['weight'] = distance_matrix[i, j]

        # 计算最小生成树
        mst = nx.minimum_spanning_tree(G)

        # 重新标记节点
        mapping = {i: asset for i, asset in enumerate(assets)}
        mst = nx.relabel_nodes(mst, mapping)

        return mst

    def build_planar_maximally_filtered_graph(self, correlation_matrix: np.ndarray, assets: List[str]) -> nx.Graph:
        """
        构建平面最大过滤图（PMFG）

        Args:
            correlation_matrix: 相关性矩阵
            assets: 资产列表

        Returns:
            PMFG网络
        """
        n = len(assets)

        # 创建距离矩阵
        distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

        # 构建完全图
        G = nx.complete_graph(n)

        # 设置边权重
        for i in range(n):
            for j in range(n):
                if i != j:
                    G[i][j]['weight'] = distance_matrix[i, j]
                    G[i][j]['correlation'] = correlation_matrix[i, j]

        # PMFG构建算法
        edges_sorted = sorted(G.edges(data=True), key=lambda x: x[2]['weight'])

        pmfg = nx.Graph()
        pmfg.add_nodes_from(G.nodes())

        for edge in edges_sorted:
            pmfg.add_edge(edge[0], edge[1], **edge[2])

            # 检查是否保持平面性（简化处理）
            if not nx.is_planar(pmfg):
                pmfg.remove_edge(edge[0], edge[1])

        # 重新标记节点
        mapping = {i: asset for i, asset in enumerate(assets)}
        pmfg = nx.relabel_nodes(pmfg, mapping)

        return pmfg

    def calculate_network_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """
        计算网络拓扑指标

        Args:
            G: 网络图

        Returns:
            网络指标字典
        """
        metrics = {}

        if len(G.nodes()) == 0:
            return metrics

        # 基本拓扑指标
        metrics['number_of_nodes'] = len(G.nodes())
        metrics['number_of_edges'] = len(G.edges())
        metrics['density'] = nx.density(G)

        # 连通性指标
        if nx.is_connected(G):
            metrics['average_path_length'] = nx.average_shortest_path_length(G)
            metrics['diameter'] = nx.diameter(G)
            metrics['clustering_coefficient'] = nx.average_clustering(G)
        else:
            metrics['average_path_length'] = float('inf')
            metrics['diameter'] = float('inf')
            metrics['clustering_coefficient'] = 0

        # 中心性指标
        try:
            degree_centrality = nx.degree_centrality(G)
            betweenness_centrality = nx.betweenness_centrality(G)
            closeness_centrality = nx.closeness_centrality(G)
            eigenvector_centrality = nx.eigenvector_centrality(G)

            metrics['degree_centrality'] = degree_centrality
            metrics['betweenness_centrality'] = betweenness_centrality
            metrics['closeness_centrality'] = closeness_centrality
            metrics['eigenvector_centrality'] = eigenvector_centrality

            # 中心性统计
            metrics['max_degree_centrality'] = max(degree_centrality.values())
            metrics['max_betweenness_centrality'] = max(betweenness_centrality.values())
            metrics['max_closeness_centrality'] = max(closeness_centrality.values())
            metrics['max_eigenvector_centrality'] = max(eigenvector_centrality.values())

        except:
            metrics['degree_centrality'] = {}
            metrics['betweenness_centrality'] = {}
            metrics['closeness_centrality'] = {}
            metrics['eigenvector_centrality'] = {}

        # 社区结构
        try:
            communities = nx.community.greedy_modularity_communities(G)
            metrics['number_of_communities'] = len(communities)
            metrics['modularity'] = nx.community.modularity(G, communities)
            metrics['communities'] = [list(comm) for comm in communities]
        except:
            metrics['number_of_communities'] = 0
            metrics['modularity'] = 0
            metrics['communities'] = []

        # 网络鲁棒性
        metrics['assortativity'] = nx.degree_assortativity_coefficient(G)

        return metrics

    def calculate_systemic_risk(self, G: nx.Graph, correlation_matrix: np.ndarray) -> Dict[str, float]:
        """
        计算系统性风险指标

        Args:
            G: 网络图
            correlation_matrix: 相关性矩阵

        Returns:
            系统性风险指标
        """
        risk_metrics = {}

        if len(G.nodes()) == 0:
            return risk_metrics

        # 基于网络的系统性风险
        try:
            # 节点度分布
            degrees = dict(G.degree())
            degree_variance = np.var(list(degrees.values()))
            risk_metrics['degree_variance'] = degree_variance

            # 连接密度
            risk_metrics['connection_density'] = nx.density(G)

            # 平均聚类系数
            risk_metrics['average_clustering'] = nx.average_clustering(G)

            # 网络中心化程度
            degree_centrality = nx.degree_centrality(G)
            centrality_concentration = np.var(list(degree_centrality.values()))
            risk_metrics['centrality_concentration'] = centrality_concentration

            # 基于相关性的风险
            correlation_risk = np.mean(np.abs(correlation_matrix))
            risk_metrics['correlation_risk'] = correlation_risk

            # 系统性风险综合指标
            systemic_risk = (
                0.3 * risk_metrics['degree_variance'] +
                0.2 * risk_metrics['connection_density'] +
                0.2 * risk_metrics['centrality_concentration'] +
                0.3 * risk_metrics['correlation_risk']
            )
            risk_metrics['systemic_risk_index'] = systemic_risk

        except Exception as e:
            risk_metrics['systemic_risk_index'] = 0.5

        return risk_metrics

    def detect_communities(self, G: nx.Graph) -> Dict[str, Any]:
        """
        检测网络社区结构

        Args:
            G: 网络图

        Returns:
            社区分析结果
        """
        community_results = {}

        if len(G.nodes()) == 0:
            return community_results

        try:
            # 使用多种社区检测算法
            algorithms = {
                'greedy_modularity': nx.community.greedy_modularity_communities,
                'label_propagation': nx.community.label_propagation_communities,
                'asyn_lpa': nx.community.asyn_lpa_communities
            }

            for name, algorithm in algorithms.items():
                try:
                    communities = algorithm(G)
                    n_communities = len(communities)
                    modularity = nx.community.modularity(G, communities)

                    community_results[f'{name}_communities'] = [list(comm) for comm in communities]
                    community_results[f'{name}_n_communities'] = n_communities
                    community_results[f'{name}_modularity'] = modularity

                except:
                    continue

            # 计算社区稳定性
            if 'greedy_modularity_communities' in community_results:
                communities = community_results['greedy_modularity_communities']
                community_sizes = [len(comm) for comm in communities]
                community_results['community_size_variance'] = np.var(community_sizes)
                community_results['max_community_size'] = max(community_sizes)
                community_results['min_community_size'] = min(community_sizes)

        except Exception as e:
            community_results['error'] = str(e)

        return community_results

    def analyze_contagion_effects(self, G: nx.Graph, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析传染效应

        Args:
            G: 网络图
            data: 价格数据

        Returns:
            传染效应分析结果
        """
        contagion_results = {}

        if len(G.nodes()) == 0:
            return contagion_results

        try:
            # 计算波动率
            returns = data.pct_change().fillna(0)
            volatility = returns.std()

            # 基于网络连接的传染风险
            node_degrees = dict(G.degree())
            max_degree_node = max(node_degrees, key=node_degrees.get)

            # 计算中心节点的传染影响
            central_nodes = sorted(
                nx.degree_centrality(G).items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]

            contagion_results['central_nodes'] = [node[0] for node in central_nodes]
            contagion_results['central_node_influence'] = [node[1] for node in central_nodes]

            # 计算网络连通性对传染的影响
            if nx.is_connected(G):
                avg_path_length = nx.average_shortest_path_length(G)
                contagion_results['contagion_speed'] = 1 / avg_path_length if avg_path_length > 0 else 0
            else:
                contagion_results['contagion_speed'] = 0

            # 计算聚类系数对传染的影响
            clustering = nx.average_clustering(G)
            contagion_results['contagion_clustering'] = clustering

            # 综合传染指数
            contagion_index = (
                0.4 * contagion_results['contagion_speed'] +
                0.3 * contagion_results['contagion_clustering'] +
                0.3 * max(contagion_results['central_node_influence'], default=0)
            )
            contagion_results['contagion_index'] = contagion_index

        except Exception as e:
            contagion_results['contagion_index'] = 0.5

        return contagion_results

    def analyze_dynamic_networks(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析动态网络演化

        Args:
            data: 多变量时间序列数据

        Returns:
            动态网络分析结果
        """
        dynamic_results = {}

        assets = data.columns.tolist()
        n_assets = len(assets)

        if n_assets < 2:
            return dynamic_results

        # 滑动窗口分析
        metrics_history = {
            'density': [],
            'clustering': [],
            'path_length': [],
            'modularity': [],
            'systemic_risk': []
        }

        networks_history = []

        for t in range(self.window_size, len(data)):
            window_data = data.iloc[t-self.window_size:t]

            try:
                # 计算相关性矩阵
                correlation_matrix = self.calculate_correlation_matrix(window_data)

                # 构建网络
                G = self.build_correlation_network(correlation_matrix, assets)

                # 计算网络指标
                metrics = self.calculate_network_metrics(G)

                metrics_history['density'].append(metrics.get('density', 0))
                metrics_history['clustering'].append(metrics.get('clustering_coefficient', 0))
                metrics_history['path_length'].append(metrics.get('average_path_length', 0))
                metrics_history['modularity'].append(metrics.get('modularity', 0))

                # 计算系统性风险
                risk_metrics = self.calculate_systemic_risk(G, correlation_matrix)
                metrics_history['systemic_risk'].append(risk_metrics.get('systemic_risk_index', 0.5))

                networks_history.append(G)

            except:
                # 如果计算失败，使用默认值
                for key in metrics_history:
                    metrics_history[key].append(0.5)

        # 转换为时间序列
        for key, values in metrics_history.items():
            dynamic_results[key] = pd.Series(values, index=data.index[self.window_size:])

        # 计算网络稳定性
        for key in ['density', 'clustering', 'systemic_risk']:
            if key in dynamic_results:
                series = dynamic_results[key]
                stability = 1 / (1 + series.std())
                dynamic_results[f'{key}_stability'] = stability

        # 生成动态信号
        dynamic_results['signals'] = self.generate_dynamic_network_signals(dynamic_results)

        return dynamic_results

    def generate_dynamic_network_signals(self, dynamic_results: Dict[str, pd.Series]) -> pd.Series:
        """
        基于动态网络生成交易信号

        Args:
            dynamic_results: 动态网络分析结果

        Returns:
            交易信号
        """
        # 使用系统性风险指标作为主要信号
        if 'systemic_risk' not in dynamic_results:
            return pd.Series(0, index=list(dynamic_results.values())[0].index)

        systemic_risk = dynamic_results['systemic_risk']
        signals = pd.Series(0, index=systemic_risk.index)

        # 风险阈值
        high_risk_threshold = systemic_risk.quantile(0.8)
        low_risk_threshold = systemic_risk.quantile(0.2)

        # 基于系统性风险生成信号
        high_risk = systemic_risk > high_risk_threshold
        low_risk = systemic_risk < low_risk_threshold

        # 网络密度信号
        if 'density' in dynamic_results:
            density = dynamic_results['density']
            high_density = density > density.quantile(0.8)
            low_density = density < density.quantile(0.2)

            signals[high_density & high_risk] = -2  # 高密度高风险：卖出
            signals[low_density & low_risk] = 2     # 低密度低风险：买入
            signals[high_risk] = -1                  # 高风险：卖出
            signals[low_risk] = 1                   # 低风险：买入

        return signals

    def get_comprehensive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        获取综合网络分析

        Args:
            data: 多变量数据

        Returns:
            综合分析结果
        """
        assets = data.columns.tolist()
        n_assets = len(assets)

        if n_assets < 2:
            return {'error': 'Insufficient assets for network analysis'}

        results = {}

        # 计算相关性矩阵
        correlation_matrix = self.calculate_correlation_matrix(data)
        results['correlation_matrix'] = correlation_matrix

        # 构建不同类型的网络
        networks = {}

        # 相关性网络
        corr_network = self.build_correlation_network(correlation_matrix, assets)
        networks['correlation_network'] = corr_network
        results['correlation_metrics'] = self.calculate_network_metrics(corr_network)

        # 最小生成树
        mst_network = self.build_minimum_spanning_tree(correlation_matrix, assets)
        networks['minimum_spanning_tree'] = mst_network
        results['mst_metrics'] = self.calculate_network_metrics(mst_network)

        # 平面最大过滤图
        try:
            pmfg_network = self.build_planar_maximally_filtered_graph(correlation_matrix, assets)
            networks['pmfg_network'] = pmfg_network
            results['pmfg_metrics'] = self.calculate_network_metrics(pmfg_network)
        except:
            results['pmfg_metrics'] = {}

        # 系统性风险分析
        results['systemic_risk'] = self.calculate_systemic_risk(corr_network, correlation_matrix)

        # 社区检测
        results['community_analysis'] = self.detect_communities(corr_network)

        # 传染效应分析
        results['contagion_analysis'] = self.analyze_contagion_effects(corr_network, data)

        # 动态网络分析
        results['dynamic_analysis'] = self.analyze_dynamic_networks(data)

        # 网络可视化数据
        results['network_data'] = self.prepare_network_visualization_data(networks)

        return results

    def prepare_network_visualization_data(self, networks: Dict[str, nx.Graph]) -> Dict[str, Any]:
        """
        准备网络可视化数据

        Args:
            networks: 网络字典

        Returns:
            可视化数据
        """
        viz_data = {}

        for network_name, G in networks.items():
            if len(G.nodes()) == 0:
                continue

            # 节点位置（使用spring layout）
            try:
                pos = nx.spring_layout(G)
            except:
                pos = nx.random_layout(G)

            # 边数据
            edges = []
            for edge in G.edges(data=True):
                edge_data = {
                    'source': edge[0],
                    'target': edge[1],
                    'weight': edge[2].get('weight', 1.0),
                    'correlation': edge[2].get('correlation', 0.0)
                }
                edges.append(edge_data)

            # 节点数据
            nodes = []
            for node in G.nodes():
                node_data = {
                    'id': node,
                    'degree': G.degree(node),
                    'centrality': nx.degree_centrality(G).get(node, 0),
                    'x': pos[node][0],
                    'y': pos[node][1]
                }
                nodes.append(node_data)

            viz_data[network_name] = {
                'nodes': nodes,
                'edges': edges
            }

        return viz_data

    def get_trading_recommendations(self, network_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            network_analysis: 网络分析结果

        Returns:
            交易建议DataFrame
        """
        # 使用动态分析结果
        if 'dynamic_analysis' not in network_analysis:
            return pd.DataFrame()

        dynamic_data = network_analysis['dynamic_analysis']
        if not dynamic_data:
            return pd.DataFrame()

        recommendations = pd.DataFrame(index=dynamic_data['signals'].index)

        # 动态网络指标
        for key, value in dynamic_data.items():
            if isinstance(value, pd.Series):
                recommendations[key] = value

        # 交易信号
        recommendations['signals'] = dynamic_data['signals']

        # 信号描述
        signal_descriptions = {
            2: '买入 - 低风险网络状态',
            1: '弱买入 - 风险较低',
            0: '持有 - 网络稳定',
            -1: '卖出 - 风险较高',
            -2: '强烈卖出 - 高风险网络状态'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 网络状态分类
        def classify_network_state(row):
            systemic_risk = row.get('systemic_risk', 0.5)
            density = row.get('density', 0.5)
            clustering = row.get('clustering', 0.5)

            if systemic_risk > 0.7 and density > 0.6:
                return 'high_risk_connected'
            elif systemic_risk < 0.3 and density < 0.4:
                return 'low_risk_disconnected'
            elif clustering > 0.6:
                return 'clustered_stable'
            else:
                return 'normal'

        recommendations['network_state'] = recommendations.apply(classify_network_state, axis=1)

        # 仓位建议
        position_map = {2: 0.6, 1: 0.3, 0: 0.2, -1: 0.1, -2: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 风险调整
        if 'systemic_risk' in recommendations.columns:
            risk_adjustment = 1 / (1 + recommendations['systemic_risk'])
            recommendations['position_size'] *= risk_adjustment

        # 策略建议
        strategy_map = {
            'high_risk_connected': 'defensive',
            'low_risk_disconnected': 'opportunistic',
            'clustered_stable': 'balanced',
            'normal': 'neutral'
        }
        recommendations['strategy'] = recommendations['network_state'].map(strategy_map)

        # 风险评估
        recommendations['risk_level'] = recommendations['systemic_risk'].apply(
            lambda x: 'high' if x > 0.7 else 'medium' if x > 0.4 else 'low'
        )

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    n = len(dates)

    # 生成多个相关的资产价格
    base_factor = np.cumsum(np.random.randn(n) * 0.1)

    assets_data = {}
    assets = ['Tech', 'Finance', 'Energy', 'Healthcare', 'Consumer']

    for i, asset in enumerate(assets):
        # 每个资产有不同的因子暴露
        factor_loading = 0.5 + i * 0.2
        idiosyncratic = np.random.randn(n) * 0.3
        price = 100 + factor_loading * base_factor + idiosyncratic
        assets_data[asset] = price

    data = pd.DataFrame(assets_data, index=dates)

    # 创建市场网络分析器
    network_analyzer = MarketNetworkAnalyzer(window_size=30)

    # 进行综合网络分析
    print("进行市场网络分析...")
    analysis_results = network_analyzer.get_comprehensive_analysis(data)

    # 显示相关性网络指标
    print("\n相关性网络指标:")
    corr_metrics = analysis_results.get('correlation_metrics', {})
    for key, value in corr_metrics.items():
        if isinstance(value, (int, float)):
            print(f"{key}: {value:.3f}")

    # 显示系统性风险
    systemic_risk = analysis_results.get('systemic_risk', {})
    print(f"\n系统性风险指标:")
    for key, value in systemic_risk.items():
        print(f"{key}: {value:.3f}")

    # 显示社区分析
    community_analysis = analysis_results.get('community_analysis', {})
    print(f"\n社区数量: {community_analysis.get('greedy_modularity_n_communities', 0)}")
    print(f"模块度: {community_analysis.get('greedy_modularity_modularity', 0):.3f}")

    # 显示动态分析
    dynamic_analysis = analysis_results.get('dynamic_analysis', {})
    if dynamic_analysis:
        print(f"\n最新网络状态:")
        print(f"密度: {dynamic_analysis['density'].iloc[-1]:.3f}")
        print(f"聚类系数: {dynamic_analysis['clustering'].iloc[-1]:.3f}")
        print(f"系统性风险: {dynamic_analysis['systemic_risk'].iloc[-1]:.3f}")
        print(f"当前信号: {dynamic_analysis['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = network_analyzer.get_trading_recommendations(analysis_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['density', 'systemic_risk', 'signal_description',
                                          'network_state', 'strategy', 'position_size']])