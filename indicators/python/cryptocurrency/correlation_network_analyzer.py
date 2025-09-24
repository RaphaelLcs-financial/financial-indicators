"""
相关性网络分析器金融指标

本模块实现了基于复杂网络理论的加密货币相关性分析指标，包括资产间相关性、网络中心性、系统性风险等。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import networkx as nx
import warnings
warnings.filterwarnings('ignore')


class CorrelationNetworkAnalyzer:
    """
    相关性网络分析器

    分析加密货币市场中的相关性结构和网络拓扑特征
    """

    def __init__(self,
                 correlation_window: int = 60,
                 correlation_threshold: float = 0.7,
                 network_window: int = 30):
        """
        初始化相关性网络分析器

        Args:
            correlation_window: 相关性计算窗口
            correlation_threshold: 相关性阈值
            network_window: 网络分析窗口
        """
        self.correlation_window = correlation_window
        self.correlation_threshold = correlation_threshold
        self.network_window = network_window

        # 网络历史
        self.network_history = {}
        self.centrality_history = {}

    def calculate_correlation_matrix(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算加密货币相关性矩阵

        Args:
            price_data: 多资产价格数据

        Returns:
            相关性矩阵
        """
        if len(price_data) < self.correlation_window:
            return pd.DataFrame()

        # 计算收益率
        returns = price_data.pct_change().dropna()

        # 计算滚动相关性
        correlation_matrix = returns.rolling(self.correlation_window).corr()

        return correlation_matrix

    def calculate_dynamic_correlations(self, price_data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        计算动态相关性指标

        Args:
            price_data: 多资产价格数据

        Returns:
            动态相关性指标字典
        """
        if len(price_data) < self.correlation_window:
            return {}

        metrics = {}

        # 计算收益率
        returns = price_data.pct_change().dropna()

        # 滚动平均相关性
        assets = price_data.columns
        n_assets = len(assets)

        # 平均相关性
        avg_correlations = []
        for i in range(self.correlation_window, len(returns)):
            window_returns = returns.iloc[i-self.correlation_window:i]

            # 计算相关性矩阵
            corr_matrix = window_returns.corr()

            # 平均相关性（不包括对角线）
            mask = ~np.eye(n_assets, dtype=bool)
            avg_corr = corr_matrix.values[mask].mean()
            avg_correlations.append(avg_corr)

        metrics['average_correlation'] = pd.Series(avg_correlations,
                                                    index=returns.index[self.correlation_window:])

        # 相关性集中度
        correlation_concentration = []
        for i in range(self.correlation_window, len(returns)):
            window_returns = returns.iloc[i-self.correlation_window:i]
            corr_matrix = window_returns.corr()

            # 计算相关性集中度（使用方差）
            mask = ~np.eye(n_assets, dtype=bool)
            correlations = corr_matrix.values[mask]
            concentration = np.var(correlations)
            correlation_concentration.append(concentration)

        metrics['correlation_concentration'] = pd.Series(correlation_concentration,
                                                         index=returns.index[self.correlation_window:])

        # 最大相关性
        max_correlations = []
        for i in range(self.correlation_window, len(returns)):
            window_returns = returns.iloc[i-self.correlation_window:i]
            corr_matrix = window_returns.corr()

            # 最大相关性（不包括对角线）
            mask = ~np.eye(n_assets, dtype=bool)
            max_corr = corr_matrix.values[mask].max()
            max_correlations.append(max_corr)

        metrics['maximum_correlation'] = pd.Series(max_correlations,
                                                  index=returns.index[self.correlation_window:])

        # 最小相关性
        min_correlations = []
        for i in range(self.correlation_window, len(returns)):
            window_returns = returns.iloc[i-self.correlation_window:i]
            corr_matrix = window_returns.corr()

            # 最小相关性
            mask = ~np.eye(n_assets, dtype=bool)
            min_corr = corr_matrix.values[mask].min()
            min_correlations.append(min_corr)

        metrics['minimum_correlation'] = pd.Series(min_correlations,
                                                  index=returns.index[self.correlation_window:])

        return metrics

    def build_correlation_network(self, correlation_matrix: pd.DataFrame) -> nx.Graph:
        """
        构建相关性网络

        Args:
            correlation_matrix: 相关性矩阵

        Returns:
            NetworkX图对象
        """
        # 创建无向图
        G = nx.Graph()

        # 添加节点
        assets = correlation_matrix.columns
        for asset in assets:
            G.add_node(asset)

        # 添加边（基于相关性阈值）
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:  # 避免重复边
                    correlation = correlation_matrix.iloc[i, j]
                    if abs(correlation) >= self.correlation_threshold:
                        # 边权重为相关性强度
                        G.add_edge(asset1, asset2, weight=abs(correlation),
                                 correlation=correlation, type='positive' if correlation > 0 else 'negative')

        return G

    def calculate_network_metrics(self, network: nx.Graph) -> Dict[str, float]:
        """
        计算网络拓扑指标

        Args:
            network: 相关性网络

        Returns:
            网络指标字典
        """
        if len(network.nodes) == 0:
            return {}

        metrics = {}

        try:
            # 基础网络指标
            metrics['number_of_nodes'] = len(network.nodes)
            metrics['number_of_edges'] = len(network.edges)
            metrics['network_density'] = nx.density(network)

            # 连通性指标
            if nx.is_connected(network):
                metrics['average_path_length'] = nx.average_shortest_path_length(network)
                metrics['diameter'] = nx.diameter(network)
                metrics['clustering_coefficient'] = nx.average_clustering(network)
            else:
                # 处理不连通网络
                largest_component = max(nx.connected_components(network), key=len)
                subgraph = network.subgraph(largest_component)
                metrics['average_path_length'] = nx.average_shortest_path_length(subgraph)
                metrics['diameter'] = nx.diameter(subgraph)
                metrics['clustering_coefficient'] = nx.average_clustering(network)

            # 中心性指标
            degree_centrality = nx.degree_centrality(network)
            betweenness_centrality = nx.betweenness_centrality(network)
            closeness_centrality = nx.closeness_centrality(network)
            eigenvector_centrality = nx.eigenvector_centrality(network, max_iter=1000)

            metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
            metrics['avg_betweenness_centrality'] = np.mean(list(betweenness_centrality.values()))
            metrics['avg_closeness_centrality'] = np.mean(list(closeness_centrality.values()))
            metrics['avg_eigenvector_centrality'] = np.mean(list(eigenvector_centrality.values()))

            # 度分布指标
            degrees = [d for n, d in network.degree()]
            metrics['max_degree'] = max(degrees) if degrees else 0
            metrics['degree_entropy'] = self._calculate_entropy(degrees)

            # 模块性（社区检测）
            try:
                communities = nx.community.greedy_modularity_communities(network)
                metrics['modularity'] = nx.community.modularity(network, communities)
                metrics['number_of_communities'] = len(communities)
            except:
                metrics['modularity'] = 0
                metrics['number_of_communities'] = 1

        except Exception as e:
            print(f"网络计算错误: {e}")
            return {}

        return metrics

    def _calculate_entropy(self, degree_sequence: List[int]) -> float:
        """计算度分布熵"""
        if not degree_sequence:
            return 0

        # 计算度分布
        degree_counts = {}
        for degree in degree_sequence:
            degree_counts[degree] = degree_counts.get(degree, 0) + 1

        total = sum(degree_counts.values())
        entropy = 0
        for count in degree_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * np.log(p)

        return entropy

    def calculate_asset_centrality(self, network: nx.Graph) -> Dict[str, Dict[str, float]]:
        """
        计算各资产的中心性

        Args:
            network: 相关性网络

        Returns:
            各资产的中心性指标
        """
        if len(network.nodes) == 0:
            return {}

        centrality_measures = {}

        for node in network.nodes():
            try:
                # 各种中心性度量
                degree_cent = nx.degree_centrality(network)[node]
                betweenness_cent = nx.betweenness_centrality(network)[node]
                closeness_cent = nx.closeness_centrality(network)[node]

                # PageRank
                pagerank = nx.pagerank(network, alpha=0.85)[node]

                centrality_measures[node] = {
                    'degree_centrality': degree_cent,
                    'betweenness_centrality': betweenness_cent,
                    'closeness_centrality': closeness_cent,
                    'pagerank': pagerank
                }

            except Exception as e:
                centrality_measures[node] = {
                    'degree_centrality': 0,
                    'betweenness_centrality': 0,
                    'closeness_centrality': 0,
                    'pagerank': 0
                }

        return centrality_measures

    def detect_correlation_clusters(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """
        检测相关性聚类

        Args:
            correlation_matrix: 相关性矩阵

        Returns:
            聚类分析结果
        """
        if len(correlation_matrix) == 0:
            return {}

        # 转换为距离矩阵
        distance_matrix = np.sqrt(2 * (1 - correlation_matrix))

        # 层次聚类
        try:
            linkage_matrix = linkage(squareform(distance_matrix), method='ward')

            # 构建聚类字典
            clusters = {}
            n_assets = len(correlation_matrix.columns)

            for n_clusters in range(2, min(n_assets, 8)):
                # 从层次聚类中提取指定数量的簇
                from scipy.cluster.hierarchy import fcluster
                cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

                cluster_dict = {}
                for i, asset in enumerate(correlation_matrix.columns):
                    cluster_dict[asset] = int(cluster_labels[i])

                clusters[f'clusters_{n_clusters}'] = cluster_dict

            return {
                'linkage_matrix': linkage_matrix,
                'clusters': clusters,
                'assets': correlation_matrix.columns.tolist()
            }

        except Exception as e:
            return {'error': str(e)}

    def calculate_systemic_risk_metrics(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算系统性风险指标

        Args:
            price_data: 多资产价格数据

        Returns:
            系统性风险指标字典
        """
        metrics = {}

        returns = price_data.pct_change().dropna()

        # 计算各资产的波动率
        volatilities = returns.rolling(self.network_window).std()

        # 计算市场组合波动率（等权重）
        market_returns = returns.mean(axis=1)
        market_volatility = market_returns.rolling(self.network_window).std()
        metrics['market_volatility'] = market_volatility

        # 计算Beta系数
        betas = pd.DataFrame(index=returns.columns, columns=returns.index[self.network_window:])

        for asset in returns.columns:
            asset_returns = returns[asset]
            cov_with_market = returns.covwith(asset_returns).rolling(self.network_window).sum()
            market_variance = market_returns.rolling(self.network_window).var()

            beta_series = cov_with_market / market_variance
            betas.loc[asset] = beta_series

        metrics['average_beta'] = betas.mean()
        metrics['beta_dispersion'] = betas.std()

        # 计算系统性风险指标
        # MES (Marginal Expected Shortfall)
        mes_values = []
        for i in range(self.network_window, len(returns)):
            window_returns = returns.iloc[i-self.network_window:i]
            market_returns_window = market_returns.iloc[i-self.network_window:i]

            # 找到市场最差的5%天数
            threshold = market_returns_window.quantile(0.05)
            worst_days = market_returns_window[market_returns_window <= threshold]

            if len(worst_days) > 0:
                # 计算这些天数内的平均资产收益
                worst_asset_returns = window_returns.loc[worst_days.index]
                mes = worst_asset_returns.mean()
            else:
                mes = pd.Series(0, index=returns.columns)

            mes_values.append(mes)

        mes_df = pd.concat(mes_values, axis=1).T
        metrics['MES'] = mes_df.mean(axis=1)
        metrics['systemic_risk_score'] = metrics['MES'].rolling(self.network_window).mean()

        # 计算CoVaR
        covar_values = []
        for i in range(self.network_window, len(returns)):
            window_returns = returns.iloc[i-self.network_window:i]
            market_returns_window = market_returns.iloc[i-self.network_window:i]

            # 市场处于压力状态下的条件VaR
            market_stress = market_returns_window.quantile(0.05)
            stress_periods = market_returns_window[market_returns_window <= market_stress]

            if len(stress_periods) > 0:
                stress_returns = window_returns.loc[stress_periods.index]
                covar = stress_returns.quantile(0.05, axis=0)
            else:
                covar = pd.Series(0, index=returns.columns)

            covar_values.append(covar)

        covar_df = pd.concat(covar_values, axis=1).T
        metrics['CoVaR'] = covar_df.mean(axis=1)
        metrics['conditional_risk'] = metrics['CoVaR'].rolling(self.network_window).mean()

        return metrics

    def calculate_network_dynamics(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算网络动态指标

        Args:
            price_data: 多资产价格数据

        Returns:
            网络动态指标字典
        """
        dynamics = {}

        returns = price_data.pct_change().dropna()

        # 计算滚动网络指标
        network_evolution = []
        for i in range(self.correlation_window, len(returns)):
            window_returns = returns.iloc[i-self.correlation_window:i]

            # 计算相关性矩阵
            corr_matrix = window_returns.corr()

            # 构建网络
            network = self.build_correlation_network(corr_matrix)

            # 计算网络指标
            network_metrics = self.calculate_network_metrics(network)
            network_evolution.append(network_metrics)

        # 转换为时间序列
        if network_evolution:
            metrics_keys = network_evolution[0].keys()
            for key in metrics_keys:
                values = [metrics.get(key, 0) for metrics in network_evolution]
                dynamics[key] = pd.Series(values, index=returns.index[self.correlation_window:])

        return dynamics

    def calculate_correlation_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有相关性网络指标

        Args:
            price_data: 多资产价格数据

        Returns:
            相关性网络指标DataFrame
        """
        if len(price_data) < self.correlation_window:
            return pd.DataFrame()

        # 计算相关性矩阵
        correlation_matrix = self.calculate_correlation_matrix(price_data)
        latest_correlation = correlation_matrix.iloc[-len(price_data.columns):, -len(price_data.columns):]

        # 计算动态相关性指标
        dynamic_correlations = self.calculate_dynamic_correlations(price_data)

        # 构建网络
        network = self.build_correlation_network(latest_correlation)

        # 计算网络指标
        network_metrics = self.calculate_network_metrics(network)

        # 计算资产中心性
        asset_centrality = self.calculate_asset_centrality(network)

        # 检测聚类
        clustering_result = self.detect_correlation_clusters(latest_correlation)

        # 计算系统性风险
        systemic_risk = self.calculate_systemic_risk_metrics(price_data)

        # 计算网络动态
        network_dynamics = self.calculate_network_dynamics(price_data)

        # 创建指标DataFrame
        indicators = pd.DataFrame(index=price_data.index[-1:])

        # 添加动态相关性指标
        for name, series in dynamic_correlations.items():
            indicators[f'dynamic_{name}'] = series.iloc[-1]

        # 添加网络指标
        for key, value in network_metrics.items():
            indicators[f'network_{key}'] = value

        # 添加系统性风险指标
        for name, series in systemic_risk.items():
            if isinstance(series, pd.Series):
                indicators[f'systemic_{name}'] = series.iloc[-1]

        # 添加网络动态指标
        for name, series in network_dynamics.items():
            if isinstance(series, pd.Series):
                indicators[f'dynamics_{name}'] = series.iloc[-1]

        # 计算市场状态指标
        if 'average_correlation' in dynamic_correlations:
            avg_corr = dynamic_correlations['average_correlation'].iloc[-1]
            if avg_corr > 0.6:
                market_regime = 'high_correlation'
            elif avg_corr < 0.3:
                market_regime = 'low_correlation'
            else:
                market_regime = 'normal_correlation'

            indicators['market_regime'] = {'high_correlation': 3, 'normal_correlation': 2, 'low_correlation': 1}[market_regime]

        # 计算风险分散指标
        if 'correlation_concentration' in dynamic_correlations:
            concentration = dynamic_correlations['correlation_concentration'].iloc[-1]
            diversification_score = 1 - concentration  # 集中度越高，分散化程度越低
            indicators['diversification_score'] = diversification_score

        return indicators

    def generate_portfolio_insights(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        生成投资组合洞察

        Args:
            price_data: 多资产价格数据

        Returns:
            投资组合洞察结果
        """
        if len(price_data) < self.correlation_window:
            return {'error': 'Insufficient data'}

        # 计算最新相关性矩阵
        correlation_matrix = self.calculate_correlation_matrix(price_data)
        latest_correlation = correlation_matrix.iloc[-len(price_data.columns):, -len(price_data.columns):]

        # 构建网络
        network = self.build_correlation_network(latest_correlation)

        # 计算资产中心性
        asset_centrality = self.calculate_asset_centrality(network)

        # 检测聚类
        clustering_result = self.detect_correlation_clusters(latest_correlation)

        # 生成洞察
        insights = {}

        # 中心资产识别
        central_assets = sorted(asset_centrality.items(),
                              key=lambda x: x[1]['pagerank'], reverse=True)[:5]
        insights['most_central_assets'] = [(asset, centrality['pagerank'])
                                         for asset, centrality in central_assets]

        # 孤立资产识别
        isolated_assets = [asset for asset, degree in dict(network.degree()).items() if degree == 0]
        insights['isolated_assets'] = isolated_assets

        # 相关性聚类
        if 'clusters' in clustering_result:
            optimal_clusters = clustering_result['clusters'].get('clusters_3', {})
            insights['correlation_clusters'] = self._format_clusters(optimal_clusters)

        # 风险集中度
        if len(network.nodes) > 0:
            degree_centralities = nx.degree_centrality(network)
            concentration_score = np.std(list(degree_centralities.values()))
            insights['risk_concentration'] = concentration_score

        # 网络稳定性
        if 'network_density' in network:
            density = network['network_density']
            if density > 0.7:
                stability = 'high_risk'  # 高度相关，风险集中
            elif density < 0.3:
                stability = 'low_risk'   # 低相关，分散化
            else:
                stability = 'moderate_risk'

            insights['network_stability'] = stability

        return insights

    def _format_clusters(self, cluster_dict: Dict[str, int]) -> Dict[int, List[str]]:
        """格式化聚类结果"""
        formatted_clusters = {}
        for asset, cluster_id in cluster_dict.items():
            if cluster_id not in formatted_clusters:
                formatted_clusters[cluster_id] = []
            formatted_clusters[cluster_id].append(asset)
        return formatted_clusters


class CryptoPortfolioOptimizer:
    """
    加密货币投资组合优化器

    基于相关性网络的投资组合优化
    """

    def __init__(self,
                 risk_aversion: float = 1.0,
                 correlation_weight: float = 0.3):
        """
        初始化投资组合优化器

        Args:
            risk_aversion: 风险厌恶系数
            correlation_weight: 相关性权重
        """
        self.risk_aversion = risk_aversion
        self.correlation_weight = correlation_weight

    def optimize_portfolio_weights(self, price_data: pd.DataFrame,
                                 network_analyzer: CorrelationNetworkAnalyzer) -> Dict[str, Any]:
        """
        优化投资组合权重

        Args:
            price_data: 价格数据
            network_analyzer: 相关性网络分析器

        Returns:
            投资组合优化结果
        """
        if len(price_data) < 30:
            return {'error': 'Insufficient data'}

        returns = price_data.pct_change().dropna()
        expected_returns = returns.mean() * 252  # 年化期望收益
        covariance_matrix = returns.cov() * 252   # 年化协方差矩阵

        # 获取相关性网络洞察
        portfolio_insights = network_analyzer.generate_portfolio_insights(price_data)

        # 构建相关性惩罚项
        correlation_penalty = self._build_correlation_penalty(
            price_data.columns, portfolio_insights
        )

        # 优化投资组合
        optimization_result = self._mean_variance_optimization(
            expected_returns, covariance_matrix, correlation_penalty
        )

        return {
            'optimal_weights': optimization_result['weights'],
            'expected_return': optimization_result['expected_return'],
            'expected_risk': optimization_result['expected_risk'],
            'sharpe_ratio': optimization_result['sharpe_ratio'],
            'correlation_adjustment': correlation_penalty,
            'portfolio_insights': portfolio_insights
        }

    def _build_correlation_penalty(self, assets: List[str],
                                insights: Dict[str, Any]) -> pd.DataFrame:
        """构建相关性惩罚矩阵"""
        n_assets = len(assets)
        penalty_matrix = pd.DataFrame(0, index=assets, columns=assets)

        # 基于中心资产的惩罚
        if 'most_central_assets' in insights:
            central_assets = [asset for asset, _ in insights['most_central_assets']]
            for i, asset1 in enumerate(assets):
                for j, asset2 in enumerate(assets):
                    if i != j and asset1 in central_assets and asset2 in central_assets:
                        penalty_matrix.iloc[i, j] = self.correlation_weight

        return penalty_matrix

    def _mean_variance_optimization(self, expected_returns: pd.Series,
                                 covariance_matrix: pd.DataFrame,
                                 correlation_penalty: pd.DataFrame) -> Dict[str, Any]:
        """均值-方差优化"""
        n_assets = len(expected_returns)

        def objective_function(weights):
            # 计算投资组合收益
            portfolio_return = np.dot(weights, expected_returns)

            # 计算投资组合风险
            portfolio_variance = np.dot(weights.T, np.dot(covariance_matrix, weights))

            # 相关性惩罚
            correlation_cost = np.dot(weights.T, np.dot(correlation_penalty, weights))

            # 目标函数：最大化夏普比率（考虑相关性惩罚）
            utility = portfolio_return - 0.5 * self.risk_aversion * portfolio_variance - correlation_cost

            return -utility  # 最小化负效用

        # 约束条件
        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # 权重和为1
            {'type': 'ineq', 'fun': lambda w: w}  # 权重非负
        )

        # 初始权重（等权重）
        initial_weights = np.array([1.0 / n_assets] * n_assets)

        # 优化
        try:
            from scipy.optimize import minimize
            result = minimize(objective_function, initial_weights,
                            method='SLSQP', constraints=constraints)

            if result.success:
                optimal_weights = result.x

                # 计算投资组合指标
                expected_return = np.dot(optimal_weights, expected_returns)
                expected_risk = np.sqrt(np.dot(optimal_weights.T,
                                             np.dot(covariance_matrix, optimal_weights)))
                sharpe_ratio = expected_return / expected_risk if expected_risk > 0 else 0

                return {
                    'weights': dict(zip(expected_returns.index, optimal_weights)),
                    'expected_return': expected_return,
                    'expected_risk': expected_risk,
                    'sharpe_ratio': sharpe_ratio,
                    'success': True
                }
            else:
                return {'success': False, 'error': 'Optimization failed'}

        except Exception as e:
            return {'success': False, 'error': str(e)}


def create_correlation_network_features(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    创建相关性网络特征

    Args:
        price_data: 多资产价格数据

    Returns:
        相关性网络特征DataFrame
    """
    analyzer = CorrelationNetworkAnalyzer()
    indicators = analyzer.calculate_correlation_indicators(price_data)
    return indicators


# 主要功能函数
def calculate_correlation_network_indicators(price_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有相关性网络指标

    Args:
        price_data: 多资产价格DataFrame

    Returns:
        包含所有指标值的DataFrame
    """
    if len(price_data) < 60:
        raise ValueError("数据长度不足，至少需要60个数据点")

    if len(price_data.columns) < 3:
        raise ValueError("至少需要3个资产来计算相关性网络指标")

    return create_correlation_network_features(price_data)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

    # 模拟多个加密资产价格
    assets = ['BTC', 'ETH', 'BNB', 'ADA', 'DOT', 'SOL', 'MATIC', 'AVAX']
    n_assets = len(assets)

    # 生成相关但独立的收益率
    base_returns = np.random.multivariate_normal(
        mean=[0.001] * n_assets,
        cov=np.eye(n_assets) * 0.0004,  # 基础波动率
        size=200
    )

    # 添加市场因子
    market_factor = np.random.normal(0.001, 0.002, 200)
    market_loadings = np.random.uniform(0.3, 0.8, n_assets)

    correlated_returns = base_returns + np.outer(market_factor, market_loadings)

    # 转换为价格
    initial_prices = np.random.uniform(100, 50000, n_assets)
    prices = []

    for i in range(n_assets):
        asset_prices = [initial_prices[i]]
        for ret in correlated_returns[:, i]:
            asset_prices.append(asset_prices[-1] * (1 + ret))
        prices.append(asset_prices)

    # 创建DataFrame
    price_data = pd.DataFrame(
        np.array(prices).T,
        columns=assets,
        index=dates
    )

    # 计算指标
    try:
        indicators = calculate_correlation_network_indicators(price_data)
        print("相关性网络指标计算成功!")
        print(f"指标数量: {indicators.shape[1]}")
        print("最新指标值:")
        print(indicators.iloc[-1])

        # 生成投资组合洞察
        analyzer = CorrelationNetworkAnalyzer()
        insights = analyzer.generate_portfolio_insights(price_data)
        print("\n投资组合洞察:")
        print(f"最中心资产: {insights['most_central_assets'][:3]}")
        print(f"孤立资产: {insights['isolated_assets']}")
        print(f"网络稳定性: {insights['network_stability']}")

    except Exception as e:
        print(f"计算错误: {e}")