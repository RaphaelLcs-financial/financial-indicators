"""
市场知识图谱图神经网络指标
Market Knowledge Graph with Graph Neural Networks

基于图神经网络和知识图谱的市场关系分析和预测
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch_geometric.data import Data, DataLoader
    from torch_geometric.nn import GCNConv, GATConv, GraphSAGE
    TORCH_GEOMETRIC_AVAILABLE = True
    print("🔗 PyTorch Geometric 已启用")
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False
    print("⚠️ PyTorch Geometric 不可用")

class MarketKnowledgeGraph:
    """
    市场知识图谱分析器

    基于图神经网络分析资产间的复杂关系和依赖性
    构建金融知识图谱进行市场预测和风险评估
    """

    def __init__(self, graph_type: str = 'correlation', gnn_type: str = 'gcn',
                 hidden_dim: int = 64, num_layers: int = 3):
        """
        初始化市场知识图谱

        Args:
            graph_type: 图类型 ('correlation', 'causal', 'semantic')
            gnn_type: GNN类型 ('gcn', 'gat', 'graphsage')
            hidden_dim: 隐藏层维度
            num_layers: 网络层数
        """
        self.graph_type = graph_type
        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 图数据
        self.graph = nx.Graph()
        self.node_features = {}
        self.edge_features = {}

        # GNN模型
        self.gnn_model = None
        self.is_trained = False

        # 知识图谱
        self.knowledge_graph = {}
        self.entity_types = {}
        self.relation_types = {}

        # 分析结果
        self.analysis_history = []

    def build_correlation_graph(self, data: pd.DataFrame, window: int = 20) -> nx.Graph:
        """构建相关性图"""
        print("📊 构建市场相关性图...")

        # 计算滚动相关性
        returns = data.pct_change().fillna(0)
        correlation_matrix = returns.rolling(window=window).corr().dropna()

        # 创建图
        G = nx.Graph()

        # 添加节点（资产）
        assets = correlation_matrix.columns
        for asset in assets:
            G.add_node(asset)

        # 添加边（相关性）
        threshold = 0.5  # 相关性阈值
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets[i+1:], i+1):
                # 使用最新相关性
                latest_corr = correlation_matrix.iloc[-1, correlation_matrix.columns.get_loc(asset2)]
                if abs(latest_corr) > threshold:
                    G.add_edge(asset1, asset2, weight=latest_corr, correlation_type='positive' if latest_corr > 0 else 'negative')

        return G

    def build_semantic_graph(self, data: pd.DataFrame, entity_info: Dict[str, str]) -> nx.Graph:
        """构建语义图"""
        print("🧠 构建市场语义图...")

        G = nx.Graph()

        # 添加节点
        for entity, entity_type in entity_info.items():
            G.add_node(entity, type=entity_type)
            self.entity_types[entity] = entity_type

        # 基于语义关系添加边
        # 这里可以基于行业、板块、地域等语义关系
        for entity1, type1 in entity_info.items():
            for entity2, type2 in entity_info.items():
                if entity1 != entity2:
                    # 简单的语义关系：同类型实体有更强连接
                    if type1 == type2:
                        G.add_edge(entity1, entity2, weight=0.8, relation_type='same_type')
                    else:
                        G.add_edge(entity1, entity2, weight=0.3, relation_type='different_type')

        return G

    def create_gnn_model(self, num_node_features: int, num_classes: int = 3):
        """创建GNN模型"""
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GNN models")

        class GNNModel(nn.Module):
            def __init__(self, num_features, hidden_dim, num_classes, gnn_type, num_layers):
                super(GNNModel, self).__init__()

                self.gnn_type = gnn_type
                self.num_layers = num_layers

                # 输入层
                if gnn_type == 'gcn':
                    self.conv1 = GCNConv(num_features, hidden_dim)
                elif gnn_type == 'gat':
                    self.conv1 = GATConv(num_features, hidden_dim, heads=4, concat=False)
                elif gnn_type == 'graphsage':
                    self.conv1 = GraphSAGE(num_features, hidden_dim)

                # 隐藏层
                self.convs = nn.ModuleList()
                for _ in range(num_layers - 2):
                    if gnn_type == 'gcn':
                        self.convs.append(GCNConv(hidden_dim, hidden_dim))
                    elif gnn_type == 'gat':
                        self.convs.append(GATConv(hidden_dim, hidden_dim, heads=4, concat=False))
                    elif gnn_type == 'graphsage':
                        self.convs.append(GraphSAGE(hidden_dim, hidden_dim))

                # 输出层
                if gnn_type == 'gcn':
                    self.conv_out = GCNConv(hidden_dim, num_classes)
                elif gnn_type == 'gat':
                    self.conv_out = GATConv(hidden_dim, num_classes, heads=1, concat=False)
                elif gnn_type == 'graphsage':
                    self.conv_out = GraphSAGE(hidden_dim, num_classes)

            def forward(self, x, edge_index):
                # 输入层
                x = F.relu(self.conv1(x, edge_index))
                x = F.dropout(x, p=0.2, training=self.training)

                # 隐藏层
                for conv in self.convs:
                    x = F.relu(conv(x, edge_index))
                    x = F.dropout(x, p=0.2, training=self.training)

                # 输出层
                x = self.conv_out(x, edge_index)
                return F.log_softmax(x, dim=1)

        self.gnn_model = GNNModel(
            num_node_features=num_node_features,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            gnn_type=self.gnn_type,
            num_layers=self.num_layers
        )

    def extract_node_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """提取节点特征"""
        features = {}

        for asset in data.columns:
            asset_data = data[asset]

            # 基础统计特征
            returns = asset_data.pct_change().fillna(0)
            features[asset] = np.array([
                np.mean(returns),           # 平均收益
                np.std(returns),            # 波动率
                np.min(returns),            # 最小收益
                np.max(returns),            # 最大收益
                np.median(returns),         # 中位数收益
                np.skew(returns),           # 偏度
                np.kurtosis(returns),       # 峰度
                np.percentile(returns, 5),  # 5%分位数
                np.percentile(returns, 95), # 95%分位数
                len(returns[returns > 0]) / len(returns),  # 正收益比例
            ])

        return features

    def prepare_graph_data(self, graph: nx.Graph, node_features: Dict[str, np.ndarray]) -> Data:
        """准备图数据"""
        # 节点映射
        node_mapping = {node: i for i, node in enumerate(graph.nodes())}

        # 节点特征
        feature_matrix = np.array([node_features[node] for node in graph.nodes()])

        # 边索引
        edge_index = []
        for edge in graph.edges():
            edge_index.append([node_mapping[edge[0]], node_mapping[edge[1]]])
            edge_index.append([node_mapping[edge[1]], node_mapping[edge[0]]])  # 无向图

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

        # 边权重
        edge_weights = []
        for edge in graph.edges():
            weight = graph.edges[edge].get('weight', 1.0)
            edge_weights.extend([weight, weight])

        edge_weights = torch.tensor(edge_weights, dtype=torch.float)

        return Data(
            x=torch.tensor(feature_matrix, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_weights
        )

    def train_gnn_model(self, data: Data, labels: np.ndarray, epochs: int = 100) -> Dict[str, Any]:
        """训练GNN模型"""
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError("PyTorch Geometric is required for GNN training")

        print("🧠 开始训练GNN模型...")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data = data.to(device)

        # 创建模型
        self.create_gnn_model(
            num_node_features=data.x.shape[1],
            num_classes=len(np.unique(labels))
        )

        model = self.gnn_model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.NLLLoss()

        # 训练循环
        model.train()
        training_loss = []

        for epoch in range(epochs):
            optimizer.zero_grad()
            out = model(data.x, data.edge_index)

            # 只使用有标签的节点进行训练
            labeled_mask = torch.tensor(labels != -1, dtype=torch.bool)
            if labeled_mask.sum() > 0:
                loss = criterion(out[labeled_mask], torch.tensor(labels[labeled_mask], dtype=torch.long).to(device))
                loss.backward()
                optimizer.step()
                training_loss.append(loss.item())

            if epoch % 20 == 0:
                print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}")

        self.is_trained = True

        return {
            'final_loss': training_loss[-1] if training_loss else 0,
            'training_loss': training_loss,
            'model_parameters': sum(p.numel() for p in model.parameters())
        }

    def analyze_market_structure(self, graph: nx.Graph) -> Dict[str, Any]:
        """分析市场结构"""
        print("🏗️ 分析市场拓扑结构...")

        # 基础图指标
        analysis = {
            'num_nodes': graph.number_of_nodes(),
            'num_edges': graph.number_of_edges(),
            'density': nx.density(graph),
            'average_degree': np.mean([d for n, d in graph.degree()]),
            'clustering_coefficient': nx.average_clustering(graph),
            'diameter': nx.diameter(graph) if nx.is_connected(graph) else float('inf'),
            'average_path_length': nx.average_shortest_path_length(graph) if nx.is_connected(graph) else 0
        }

        # 中心性分析
        centrality_measures = {
            'degree_centrality': nx.degree_centrality(graph),
            'betweenness_centrality': nx.betweenness_centrality(graph),
            'closeness_centrality': nx.closeness_centrality(graph),
            'eigenvector_centrality': nx.eigenvector_centrality_numpy(graph)
        }

        analysis['centrality'] = centrality_measures

        # 社区检测
        try:
            communities = nx.community.greedy_modularity_communities(graph)
            analysis['communities'] = {
                'num_communities': len(communities),
                'modularity': nx.community.modularity(graph, communities),
                'community_sizes': [len(c) for c in communities]
            }
        except:
            analysis['communities'] = {'error': 'Community detection failed'}

        # 系统性风险分析
        analysis['systemic_risk'] = self._calculate_systemic_risk(graph)

        return analysis

    def _calculate_systemic_risk(self, graph: nx.Graph) -> Dict[str, float]:
        """计算系统性风险"""
        risk_metrics = {}

        # 连通性风险
        if nx.is_connected(graph):
            # 移除关键节点的影响
            centrality = nx.degree_centrality(graph)
            critical_nodes = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]

            fragmentation_scores = []
            for node, _ in critical_nodes:
                temp_graph = graph.copy()
                temp_graph.remove_node(node)
                if nx.is_connected(temp_graph):
                    fragmentation_scores.append(1.0)
                else:
                    # 计算连通分量大小
                    components = list(nx.connected_components(temp_graph))
                    largest_component_size = len(max(components, key=len))
                    fragmentation = largest_component_size / len(temp_graph.nodes())
                    fragmentation_scores.append(fragmentation)

            risk_metrics['node_removal_impact'] = np.mean(fragmentation_scores)
        else:
            risk_metrics['node_removal_impact'] = 0

        # 边密度风险
        risk_metrics['edge_density_risk'] = 1 - nx.density(graph)

        # 集中度风险
        degrees = [d for n, d in graph.degree()]
        risk_metrics['concentration_risk'] = np.std(degrees) / np.mean(degrees) if np.mean(degrees) > 0 else 0

        return risk_metrics

    def predict_market_movements(self, data: pd.DataFrame, horizon: int = 5) -> Dict[str, Any]:
        """预测市场走势"""
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train_gnn_model方法")

        print("🔮 预测市场走势...")

        # 构建图
        if self.graph_type == 'correlation':
            graph = self.build_correlation_graph(data)
        else:
            graph = self.build_semantic_graph(data, {})

        # 提取特征
        node_features = self.extract_node_features(data)
        graph_data = self.prepare_graph_data(graph, node_features)

        # 使用GNN预测
        if TORCH_GEOMETRIC_AVAILABLE and self.gnn_model is not None:
            self.gnn_model.eval()
            with torch.no_grad():
                predictions = self.gnn_model(graph_data.x, graph_data.edge_index)
                predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
                probabilities = torch.exp(predictions).cpu().numpy()
        else:
            # 简化的预测方法
            predicted_labels = np.zeros(len(graph.nodes()))
            probabilities = np.ones((len(graph.nodes()), 3)) / 3

        # 分析预测结果
        prediction_analysis = {
            'predicted_labels': dict(zip(graph.nodes(), predicted_labels)),
            'probabilities': dict(zip(graph.nodes(), probabilities)),
            'market_sentiment': self._analyze_market_sentiment(predicted_labels),
            'risk_assessment': self._assess_prediction_risk(probabilities)
        }

        return prediction_analysis

    def _analyze_market_sentiment(self, labels: np.ndarray) -> Dict[str, float]:
        """分析市场情绪"""
        unique_labels, counts = np.unique(labels, return_counts=True)
        total = len(labels)

        sentiment = {}
        for label, count in zip(unique_labels, counts):
            if label == 0:
                sentiment['neutral'] = count / total
            elif label == 1:
                sentiment['bullish'] = count / total
            elif label == 2:
                sentiment['bearish'] = count / total

        return sentiment

    def _assess_prediction_risk(self, probabilities: np.ndarray) -> Dict[str, float]:
        """评估预测风险"""
        # 计算预测不确定性
        uncertainty = 1 - np.max(probabilities, axis=1)
        avg_uncertainty = np.mean(uncertainty)

        # 计算置信度分布
        confidence_scores = np.max(probabilities, axis=1)
        low_confidence_ratio = (confidence_scores < 0.6).mean()

        return {
            'average_uncertainty': avg_uncertainty,
            'low_confidence_ratio': low_confidence_ratio,
            'prediction_reliability': 1 - avg_uncertainty
        }

    def detect_anomalies(self, data: pd.DataFrame, threshold: float = 2.0) -> Dict[str, Any]:
        """检测市场异常"""
        print("🚨 检测市场异常...")

        # 构建图
        graph = self.build_correlation_graph(data)

        # 计算节点异常分数
        anomaly_scores = {}

        for node in graph.nodes():
            # 基于中心性的异常检测
            degree_centrality = nx.degree_centrality(graph)[node]
            betweenness_centrality = nx.betweenness_centrality(graph)[node]

            # 基于局部结构的异常检测
            neighbors = list(graph.neighbors(node))
            if neighbors:
                neighbor_degrees = [graph.degree(n) for n in neighbors]
                degree_anomaly = abs(graph.degree(node) - np.mean(neighbor_degrees)) / (np.std(neighbor_degrees) + 1e-6)
            else:
                degree_anomaly = 0

            # 综合异常分数
            anomaly_score = (degree_anomaly + betweenness_centrality * 2) / 3
            anomaly_scores[node] = anomaly_score

        # 识别异常节点
        threshold_score = np.mean(list(anomaly_scores.values())) + threshold * np.std(list(anomaly_scores.values()))
        anomalies = {node: score for node, score in anomaly_scores.items() if score > threshold_score}

        return {
            'anomaly_scores': anomaly_scores,
            'anomalies': anomalies,
            'num_anomalies': len(anomalies),
            'anomaly_threshold': threshold_score,
            'anomaly_severity': self._classify_anomaly_severity(anomalies)
        }

    def _classify_anomaly_severity(self, anomalies: Dict[str, float]) -> Dict[str, int]:
        """分类异常严重程度"""
        if not anomalies:
            return {'low': 0, 'medium': 0, 'high': 0}

        scores = list(anomalies.values())
        mean_score = np.mean(scores)

        severity = {
            'low': len([s for s in scores if s < mean_score]),
            'medium': len([s for s in scores if mean_score <= s < mean_score * 1.5]),
            'high': len([s for s in scores if s >= mean_score * 1.5])
        }

        return severity

    def analyze_market_regimes(self, data: pd.DataFrame, window: int = 20) -> Dict[str, Any]:
        """分析市场机制"""
        print("📈 分析市场机制...")

        # 滚动窗口分析
        regime_history = []

        for i in range(window, len(data)):
            window_data = data.iloc[i-window:i]

            # 构建窗口图
            graph = self.build_correlation_graph(window_data)

            # 计算图特征
            graph_features = {
                'density': nx.density(graph),
                'clustering': nx.average_clustering(graph),
                'average_path_length': nx.average_shortest_path_length(graph) if nx.is_connected(graph) else 0,
                'assortativity': nx.degree_assortativity_coefficient(graph) if len(graph.edges()) > 0 else 0
            }

            # 识别机制
            regime = self._identify_regime(graph_features)

            regime_history.append({
                'date': data.index[i],
                'regime': regime,
                'features': graph_features
            })

        # 分析机制转换
        regime_transitions = self._analyze_regime_transitions(regime_history)

        return {
            'regime_history': regime_history,
            'current_regime': regime_history[-1]['regime'] if regime_history else 'unknown',
            'regime_transitions': regime_transitions,
            'regime_stability': self._calculate_regime_stability(regime_history)
        }

    def _identify_regime(self, features: Dict[str, float]) -> str:
        """识别市场机制"""
        density = features.get('density', 0)
        clustering = features.get('clustering', 0)
        path_length = features.get('average_path_length', 0)

        if density > 0.7 and clustering > 0.6:
            return 'high_correlation'
        elif density < 0.3 and path_length > 2.0:
            return 'low_correlation'
        elif clustering > 0.5:
            return 'clustered'
        else:
            return 'normal'

    def _analyze_regime_transitions(self, history: List[Dict]) -> Dict[str, Any]:
        """分析机制转换"""
        if len(history) < 2:
            return {'transitions': [], 'stability_score': 0}

        transitions = []
        for i in range(1, len(history)):
            if history[i]['regime'] != history[i-1]['regime']:
                transitions.append({
                    'from_regime': history[i-1]['regime'],
                    'to_regime': history[i]['regime'],
                    'date': history[i]['date']
                })

        # 计算稳定性
        regime_changes = len(transitions)
        total_periods = len(history)
        stability_score = 1 - (regime_changes / total_periods)

        return {
            'transitions': transitions,
            'num_transitions': len(transitions),
            'stability_score': stability_score
        }

    def _calculate_regime_stability(self, history: List[Dict]) -> Dict[str, float]:
        """计算机制稳定性"""
        if not history:
            return {'overall_stability': 0, 'regime_durations': {}}

        # 计算各机制持续时间
        regime_durations = {}
        current_regime = None
        current_duration = 0

        for period in history:
            regime = period['regime']
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime is not None:
                    if current_regime not in regime_durations:
                        regime_durations[current_regime] = []
                    regime_durations[current_regime].append(current_duration)
                current_regime = regime
                current_duration = 1

        # 添加最后一个机制
        if current_regime is not None:
            if current_regime not in regime_durations:
                regime_durations[current_regime] = []
            regime_durations[current_regime].append(current_duration)

        # 计算平均持续时间
        avg_durations = {}
        for regime, durations in regime_durations.items():
            avg_durations[regime] = np.mean(durations)

        overall_stability = np.mean(list(avg_durations.values())) if avg_durations else 0

        return {
            'overall_stability': overall_stability,
            'average_regime_durations': avg_durations,
            'regime_distributions': {k: len(v) for k, v in regime_durations.items()}
        }

    def get_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用相关训练方法")

        # 市场结构分析
        graph = self.build_correlation_graph(data)
        structure_analysis = self.analyze_market_structure(graph)

        # 异常检测
        anomaly_analysis = self.detect_anomalies(data)

        # 机制分析
        regime_analysis = self.analyze_market_regimes(data)

        # 生成综合信号
        signals = []

        for i, date in enumerate(data.index):
            signal = self._generate_composite_signal(
                structure_analysis,
                anomaly_analysis,
                regime_analysis,
                i,
                len(data)
            )
            signals.append(signal)

        signals_df = pd.DataFrame({
            'signal': signals,
            'market_regime': regime_analysis['current_regime'],
            'anomaly_count': anomaly_analysis['num_anomalies'],
            'systemic_risk': structure_analysis['systemic_risk'].get('concentration_risk', 0)
        }, index=data.index)

        return signals_df

    def _generate_composite_signal(self, structure_analysis: Dict, anomaly_analysis: Dict,
                                 regime_analysis: Dict, current_idx: int, total_length: int) -> int:
        """生成综合交易信号"""
        signal = 0

        # 基于市场机制的信号
        current_regime = regime_analysis['current_regime']
        regime_signals = {
            'high_correlation': -1,  # 高相关性减仓
            'low_correlation': 0,    # 低相关性观望
            'clustered': 0,         # 集群状态观望
            'normal': 1             # 正常状态适量参与
        }
        signal += regime_signals.get(current_regime, 0)

        # 基于异常的信号
        anomaly_count = anomaly_analysis['num_anomalies']
        if anomaly_count > len(structure_analysis.get('centrality', {}).get('degree_centrality', {})) * 0.2:
            signal -= 1  # 异常较多时减仓

        # 基于系统性风险的信号
        systemic_risk = structure_analysis['systemic_risk'].get('concentration_risk', 0)
        if systemic_risk > 1.0:
            signal -= 1
        elif systemic_risk < 0.5:
            signal += 1

        # 限制信号范围
        return max(-1, min(1, signal))

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'graph_type': self.graph_type,
            'gnn_type': self.gnn_type,
            'hidden_dim': self.hidden_dim,
            'num_layers': self.num_layers,
            'is_trained': self.is_trained,
            'torch_geometric_available': TORCH_GEOMETRIC_AVAILABLE,
            'graph_stats': {
                'num_nodes': self.graph.number_of_nodes(),
                'num_edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph) if len(self.graph.nodes()) > 0 else 0
            },
            'model_type': 'Market Knowledge Graph with GNN'
        }

# 便捷函数
def create_market_knowledge_graph(graph_type: str = 'correlation', gnn_type: str = 'gcn') -> MarketKnowledgeGraph:
    """创建市场知识图谱实例"""
    return MarketKnowledgeGraph(graph_type, gnn_type)

def quick_graph_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """快速图分析"""
    graph_analyzer = MarketKnowledgeGraph()

    # 构建图
    graph = graph_analyzer.build_correlation_graph(data)

    # 分析结构
    structure_analysis = graph_analyzer.analyze_market_structure(graph)

    # 检测异常
    anomaly_analysis = graph_analyzer.detect_anomalies(data)

    # 分析机制
    regime_analysis = graph_analyzer.analyze_market_regimes(data)

    return {
        'structure_analysis': structure_analysis,
        'anomaly_analysis': anomaly_analysis,
        'regime_analysis': regime_analysis,
        'model_info': graph_analyzer.get_model_info()
    }