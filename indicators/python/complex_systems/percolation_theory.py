"""
金融渗流理论指标
Financial Percolation Theory Indicator

基于渗流理论的市场相变和系统性风险分析
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

class FinancialPercolationAnalyzer:
    """
    金融渗流分析器

    基于渗流理论分析市场相变、临界现象和系统性风险
    检测市场状态的突然转变和泡沫形成
    """

    def __init__(self, window_size: int = 50, correlation_threshold: float = 0.7):
        """
        初始化渗流分析器

        Args:
            window_size: 分析窗口大小
            correlation_threshold: 相关性阈值（渗流阈值）
        """
        self.window_size = window_size
        self.correlation_threshold = correlation_threshold

        # 渗流参数
        self.critical_exponents = {
            'beta': 0.4,     # 序参量临界指数
            'gamma': 1.2,   # 磁化率临界指数
            'nu': 0.8,      # 关联长度临界指数
            'alpha': 0.1    # 比热临界指数
        }

        # 分析结果存储
        self.percolation_history = []
        self.phase_transitions = []
        self.critical_points = []

    def _calculate_correlation_matrix(self, data: pd.DataFrame) -> np.ndarray:
        """计算资产相关性矩阵"""
        returns = data.pct_change().dropna()

        # 如果是单资产时间序列，创建滚动窗口相关性
        if returns.shape[1] == 1:
            # 使用不同滞后期的相关性
            correlation_matrix = np.zeros((self.window_size, self.window_size))

            for i in range(self.window_size):
                for j in range(self.window_size):
                    if i != j:
                        # 计算不同时间点的相关性
                        series1 = returns.iloc[-(self.window_size + i):-(i) if i > 0 else len(returns)]
                        series2 = returns.iloc[-(self.window_size + j):-(j) if j > 0 else len(returns)]
                        min_len = min(len(series1), len(series2))
                        if min_len > 10:
                            correlation_matrix[i, j] = np.corrcoef(series1[-min_len:], series2[-min_len:])[0, 1]
        else:
            # 多资产相关性矩阵
            correlation_matrix = returns.corr().fillna(0).values

        return np.abs(correlation_matrix)  # 使用绝对值相关性

    def _create_correlation_network(self, correlation_matrix: np.ndarray) -> nx.Graph:
        """创建相关性网络"""
        G = nx.Graph()

        n_assets = correlation_matrix.shape[0]

        # 添加节点
        for i in range(n_assets):
            G.add_node(i)

        # 添加边（基于相关性阈值）
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if correlation_matrix[i, j] > self.correlation_threshold:
                    G.add_edge(i, j, weight=correlation_matrix[i, j])

        return G

    def _calculate_percolation_parameters(self, G: nx.Graph) -> Dict[str, float]:
        """计算渗流参数"""
        if len(G.nodes()) == 0:
            return {
                'largest_cluster_size': 0,
                'number_of_clusters': 0,
                'average_cluster_size': 0,
                'percolation_probability': 0,
                'correlation_length': 0
            }

        # 计算连通分量
        connected_components = list(nx.connected_components(G))

        if not connected_components:
            return {
                'largest_cluster_size': 0,
                'number_of_clusters': 0,
                'average_cluster_size': 0,
                'percolation_probability': 0,
                'correlation_length': 0
            }

        # 最大簇大小
        largest_cluster = max(connected_components, key=len)
        largest_cluster_size = len(largest_cluster) / len(G.nodes())

        # 簇数量
        number_of_clusters = len(connected_components)

        # 平均簇大小
        average_cluster_size = np.mean([len(cluster) for cluster in connected_components])

        # 渗流概率（最大簇的相对大小）
        percolation_probability = largest_cluster_size

        # 关联长度（使用平均最短路径）
        if len(G.nodes()) > 1 and len(G.edges()) > 0:
            try:
                correlation_length = nx.average_shortest_path_length(G)
            except:
                correlation_length = 0
        else:
            correlation_length = 0

        return {
            'largest_cluster_size': largest_cluster_size,
            'number_of_clusters': number_of_clusters,
            'average_cluster_size': average_cluster_size,
            'percolation_probability': percolation_probability,
            'correlation_length': correlation_length
        }

    def _detect_critical_point(self, percolation_params: Dict[str, float],
                            history: List[Dict[str, float]]) -> bool:
        """检测临界点"""
        if len(history) < 5:
            return False

        # 计算最近渗流概率的变化率
        recent_percolation = [p['percolation_probability'] for p in history[-5:]]
        percolation_change = np.diff(recent_percolation)

        # 检测突然变化（相变特征）
        sudden_change_threshold = 0.1
        if np.any(np.abs(percolation_change) > sudden_change_threshold):
            return True

        # 检测临界点附近的幂律行为
        current_percolation = percolation_params['percolation_probability']
        if 0.3 < current_percolation < 0.7:  # 临界区域
            # 检查簇大小分布是否符合幂律
            if self._check_power_law_behavior(percolation_params):
                return True

        return False

    def _check_power_law_behavior(self, percolation_params: Dict[str, float]) -> bool:
        """检查幂律行为"""
        # 简化的幂律检测
        # 在实际应用中，这里应该进行更严格的统计检验

        # 检查标度关系
        expected_scaling = 0.5  # 期望的标度指数

        # 基于渗流理论的标度关系
        largest_cluster = percolation_params['largest_cluster_size']
        correlation_length = percolation_params['correlation_length']

        if correlation_length > 0:
            scaling_ratio = largest_cluster / (correlation_length ** expected_scaling)
            # 如果在合理范围内，认为符合幂律
            return 0.1 < scaling_ratio < 10

        return False

    def _calculate_order_parameter(self, percolation_params: Dict[str, float]) -> float:
        """计算序参量"""
        # 在渗流理论中，序参量通常是最大簇的大小
        return percolation_params['largest_cluster_size']

    def _calculate_susceptibility(self, percolation_params: Dict[str, float],
                                 history: List[Dict[str, float]]) -> float:
        """计算磁化率（ susceptibility）"""
        if len(history) < 2:
            return 0

        # 磁化率定义为序参量的涨落
        recent_order_parameters = [self._calculate_order_parameter(p) for p in history[-10:]]

        if len(recent_order_parameters) > 1:
            susceptibility = np.var(recent_order_parameters)
        else:
            susceptibility = 0

        return susceptibility

    def analyze_market_phase(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析市场相态"""
        print("🌀 开始渗流理论市场分析...")

        # 计算相关性矩阵
        correlation_matrix = self._calculate_correlation_matrix(data)

        # 创建相关性网络
        correlation_network = self._create_correlation_network(correlation_matrix)

        # 计算渗流参数
        percolation_params = self._calculate_percolation_parameters(correlation_network)

        # 检测临界点
        is_critical = self._detect_critical_point(percolation_params, self.percolation_history)

        # 计算序参量和磁化率
        order_parameter = self._calculate_order_parameter(percolation_params)
        susceptibility = self._calculate_susceptibility(percolation_params, self.percolation_history)

        # 保存历史
        self.percolation_history.append(percolation_params.copy())

        # 检测相变
        if is_critical:
            self.critical_points.append(len(self.percolation_history) - 1)

        # 确定市场相态
        market_phase = self._determine_market_phase(order_parameter, susceptibility, is_critical)

        # 计算系统性风险指标
        systemic_risk = self._calculate_systemic_risk(percolation_params, correlation_network)

        # 生成交易信号
        trading_signal = self._generate_trading_signal(market_phase, systemic_risk, order_parameter)

        results = {
            'timestamp': data.index[-1] if hasattr(data.index, '__getitem__') else len(data),
            'market_phase': market_phase,
            'order_parameter': order_parameter,
            'susceptibility': susceptibility,
            'is_critical': is_critical,
            'systemic_risk': systemic_risk,
            'trading_signal': trading_signal,
            'percolation_parameters': percolation_params,
            'network_metrics': self._calculate_network_metrics(correlation_network),
            'phase_transition_probability': self._calculate_phase_transition_probability()
        }

        return results

    def _determine_market_phase(self, order_parameter: float, susceptibility: float,
                               is_critical: bool) -> str:
        """确定市场相态"""
        if is_critical:
            return "CRITICAL"

        # 基于序参量和磁化率确定相态
        if order_parameter < 0.3:
            if susceptibility > 0.1:
                return "DISORDERED_HIGH_VOLATILITY"
            else:
                return "DISORDERED_LOW_VOLATILITY"
        elif order_parameter > 0.7:
            if susceptibility > 0.1:
                return "ORDERED_HIGH_CORRELATION"
            else:
                return "ORDERED_STABLE"
        else:
            return "TRANSITIONAL"

    def _calculate_systemic_risk(self, percolation_params: Dict[str, float],
                               network: nx.Graph) -> float:
        """计算系统性风险"""
        # 基于渗流参数的系统性风险指标
        risk_components = []

        # 1. 渗流风险（大簇形成）
        percolation_risk = percolation_params['percolation_probability']
        risk_components.append(percolation_risk)

        # 2. 网络密度风险
        if len(network.nodes()) > 1:
            density = nx.density(network)
            risk_components.append(density)

        # 3. 连通性风险
        if len(network.nodes()) > 0:
            connectivity = len(network.edges()) / (len(network.nodes()) * (len(network.nodes()) - 1) / 2)
            risk_components.append(connectivity)

        # 4. 集中风险
        if len(network.nodes()) > 0:
            centrality = nx.degree_centrality(network)
            centrality_concentration = np.std(list(centrality.values()))
            risk_components.append(centrality_concentration)

        # 综合风险指标
        if risk_components:
            systemic_risk = np.mean(risk_components)
        else:
            systemic_risk = 0

        return systemic_risk

    def _calculate_network_metrics(self, network: nx.Graph) -> Dict[str, float]:
        """计算网络指标"""
        if len(network.nodes()) == 0:
            return {
                'density': 0,
                'clustering_coefficient': 0,
                'average_path_length': 0,
                'assortativity': 0,
                'modularity': 0
            }

        metrics = {}

        # 网络密度
        metrics['density'] = nx.density(network)

        # 聚类系数
        try:
            metrics['clustering_coefficient'] = nx.average_clustering(network)
        except:
            metrics['clustering_coefficient'] = 0

        # 平均路径长度
        if len(network.edges()) > 0:
            try:
                metrics['average_path_length'] = nx.average_shortest_path_length(network)
            except:
                metrics['average_path_length'] = 0
        else:
            metrics['average_path_length'] = 0

        # 同配性
        try:
            metrics['assortativity'] = nx.degree_assortativity_coefficient(network)
        except:
            metrics['assortativity'] = 0

        # 模块度
        try:
            communities = nx.community.greedy_modularity_communities(network)
            metrics['modularity'] = nx.community.modularity(network, communities)
        except:
            metrics['modularity'] = 0

        return metrics

    def _calculate_phase_transition_probability(self) -> float:
        """计算相变概率"""
        if len(self.percolation_history) < 10:
            return 0

        # 基于历史数据计算相变概率
        recent_order_params = [p['percolation_probability'] for p in self.percolation_history[-10:]]

        # 计算变化率
        changes = np.diff(recent_order_params)
        volatility = np.std(changes)

        # 相变概率与波动性正相关
        transition_prob = min(volatility * 10, 1.0)

        return transition_prob

    def _generate_trading_signal(self, market_phase: str, systemic_risk: float,
                               order_parameter: float) -> int:
        """生成交易信号"""
        # 基于市场相态和系统性风险的交易逻辑

        if market_phase == "CRITICAL":
            # 临界点：观望或减仓
            return -1 if systemic_risk > 0.7 else 0

        elif market_phase == "DISORDERED_HIGH_VOLATILITY":
            # 高波动无序状态：谨慎或反向操作
            return 1 if systemic_risk < 0.5 else -1

        elif market_phase == "ORDERED_HIGH_CORRELATION":
            # 高相关性有序状态：风险较高，考虑减仓
            return -1 if systemic_risk > 0.6 else 0

        elif market_phase == "ORDERED_STABLE":
            # 稳定有序状态：适合趋势跟踪
            return 1

        elif market_phase == "TRANSITIONAL":
            # 过渡状态：观望
            return 0

        else:
            return 0

    def analyze_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """时间序列分析"""
        print(f"📈 开始时间序列渗流分析（{len(data)}个数据点）...")

        results = []

        # 滚动窗口分析
        for i in range(self.window_size, len(data)):
            window_data = data.iloc[i - self.window_size:i]

            try:
                analysis_result = self.analyze_market_phase(window_data)
                results.append(analysis_result)
            except Exception as e:
                print(f"分析窗口 {i} 时出错: {e}")
                continue

        # 转换为DataFrame
        results_df = pd.DataFrame(results)

        if not results_df.empty:
            # 计算额外指标
            results_df['phase_change'] = results_df['market_phase'].ne(
                results_df['market_phase'].shift()).fillna(False).astype(int)

            # 计算风险等级
            results_df['risk_level'] = self._calculate_risk_level(results_df)

        return results_df

    def _calculate_risk_level(self, results_df: pd.DataFrame) -> pd.Series:
        """计算风险等级"""
        risk_scores = []

        for idx, row in results_df.iterrows():
            risk_score = 0

            # 基于市场相态
            phase_risk = {
                'CRITICAL': 5,
                'DISORDERED_HIGH_VOLATILITY': 4,
                'ORDERED_HIGH_CORRELATION': 4,
                'TRANSITIONAL': 3,
                'DISORDERED_LOW_VOLATILITY': 2,
                'ORDERED_STABLE': 1
            }
            risk_score += phase_risk.get(row['market_phase'], 0)

            # 基于系统性风险
            risk_score += row['systemic_risk'] * 5

            # 基于临界点状态
            if row['is_critical']:
                risk_score += 2

            risk_scores.append(min(risk_score, 10))  # 最大风险等级为10

        return pd.Series(risk_scores, index=results_df.index)

    def get_critical_points_analysis(self) -> Dict[str, Any]:
        """获取临界点分析"""
        if not self.critical_points:
            return {'critical_points': [], 'analysis': 'No critical points detected'}

        analysis = {
            'critical_points': self.critical_points,
            'number_of_critical_points': len(self.critical_points),
            'critical_point_density': len(self.critical_points) / max(len(self.percolation_history), 1),
            'average_distance_between_critical_points': np.mean(np.diff(self.critical_points)) if len(self.critical_points) > 1 else 0
        }

        return analysis

    def reset_analysis(self):
        """重置分析状态"""
        self.percolation_history = []
        self.phase_transitions = []
        self.critical_points = []

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'window_size': self.window_size,
            'correlation_threshold': self.correlation_threshold,
            'critical_exponents': self.critical_exponents,
            'analysis_history_length': len(self.percolation_history),
            'number_of_critical_points': len(self.critical_points),
            'model_type': 'Financial Percolation Theory Analyzer'
        }

# 便捷函数
def create_percolation_analyzer(window_size: int = 50,
                              correlation_threshold: float = 0.7) -> FinancialPercolationAnalyzer:
    """创建渗流分析器实例"""
    return FinancialPercolationAnalyzer(window_size, correlation_threshold)

def quick_percolation_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """快速渗流分析"""
    analyzer = FinancialPercolationAnalyzer()

    # 执行分析
    result = analyzer.analyze_market_phase(data)

    # 获取模型信息
    model_info = analyzer.get_model_info()

    return {
        'analysis_result': result,
        'model_info': model_info,
        'critical_points_analysis': analyzer.get_critical_points_analysis()
    }