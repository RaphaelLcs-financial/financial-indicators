"""
神经符号金融推理系统
Neuro-Symbolic Financial Reasoning System

结合神经网络与符号推理的混合智能系统，实现可解释的金融决策
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    print("🧠 PyTorch 已启用")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 不可用")

try:
    import sympy as sp
    from sympy import symbols, Eq, solve, sympify
    SYMPY_AVAILABLE = True
    print("🔢 SymPy 已启用")
except ImportError:
    SYMPY_AVAILABLE = False
    print("⚠️ SymPy 不可用")

class NeuroSymbolicFinancialReasoner:
    """
    神经符号金融推理器

    结合深度学习的模式识别能力与符号推理的逻辑解释能力
    """

    def __init__(self,
                 neural_hidden_dims: List[int] = [128, 64, 32],
                 symbolic_rules: List[str] = None,
                 reasoning_depth: int = 3,
                 confidence_threshold: float = 0.7):
        """
        初始化神经符号推理器

        Args:
            neural_hidden_dims: 神经网络隐藏层维度
            symbolic_rules: 符号规则列表
            reasoning_depth: 推理深度
            confidence_threshold: 置信度阈值
        """
        self.neural_hidden_dims = neural_hidden_dims
        self.symbolic_rules = symbolic_rules or self._default_financial_rules()
        self.reasoning_depth = reasoning_depth
        self.confidence_threshold = confidence_threshold

        # 神经网络组件
        self.neural_network = None
        self.symbolic_reasoner = SymbolicReasoner(self.symbolic_rules)

        # 推理结果
        self.reasoning_trace = []
        self.explanation_chain = []
        self.confidence_scores = {}

        # 知识图谱
        self.knowledge_graph = FinancialKnowledgeGraph()

        if TORCH_AVAILABLE:
            self._build_neural_network()

    def _default_financial_rules(self) -> List[str]:
        """默认金融规则"""
        return [
            "IF price > sma_20 AND rsi < 30 THEN buy_signal",
            "IF price < sma_20 AND rsi > 70 THEN sell_signal",
            "IF volume > avg_volume * 1.5 AND price_change > 0 THEN strong_buy",
            "IF volatility > avg_volatility * 2 THEN risk_warning",
            "IF correlation > 0.8 THEN market_sync",
            "IF dividend_yield > 0.05 THEN value_investment"
        ]

    def _build_neural_network(self):
        """构建神经网络"""
        if not TORCH_AVAILABLE:
            return

        class NeuralFeatureExtractor(nn.Module):
            def __init__(self, input_dim: int, hidden_dims: List[int]):
                super().__init__()
                layers = []
                prev_dim = input_dim

                for hidden_dim in hidden_dims:
                    layers.extend([
                        nn.Linear(prev_dim, hidden_dim),
                        nn.ReLU(),
                        nn.BatchNorm1d(hidden_dim),
                        nn.Dropout(0.2)
                    ])
                    prev_dim = hidden_dim

                layers.append(nn.Linear(prev_dim, hidden_dims[-1]))
                self.network = nn.Sequential(*layers)

            def forward(self, x):
                return self.network(x)

        self.neural_network = NeuralFeatureExtractor(10, self.neural_hidden_dims)

    def extract_neural_features(self, market_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """提取神经网络特征"""
        print("🧠 提取神经网络特征...")

        features = {}

        # 基础市场特征
        if 'close' in market_data.columns:
            prices = market_data['close'].dropna()
            if len(prices) > 0:
                features['price_momentum'] = self._calculate_momentum(prices)
                features['volatility_regime'] = self._classify_volatility_regime(prices)
                features['trend_strength'] = self._calculate_trend_strength(prices)

        # 多变量特征
        numeric_columns = market_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            correlation_matrix = market_data[numeric_columns].corr()
            features['market_structure'] = self._analyze_market_structure(correlation_matrix)
            features['dependency_patterns'] = self._extract_dependency_patterns(correlation_matrix)

        # 时序模式特征
        if len(market_data) > 20:
            features['temporal_patterns'] = self._extract_temporal_patterns(market_data)
            features['cycle_components'] = self._decompose_cycles(market_data)

        return features

    def symbolic_reasoning(self,
                          neural_features: Dict[str, np.ndarray],
                          market_data: pd.DataFrame) -> Dict[str, Any]:
        """符号推理"""
        print("🔍 执行符号推理...")

        reasoning_results = {}

        # 将神经网络特征转换为符号表示
        symbolic_facts = self._neural_to_symbolic(neural_features)

        # 应用规则推理
        for rule in self.symbolic_rules:
            rule_result = self.symbolic_reasoner.apply_rule(rule, symbolic_facts, market_data)
            reasoning_results[rule] = rule_result

        # 逻辑推理链
        logical_chain = self._build_reasoning_chain(symbolic_facts, reasoning_results)

        # 冲突检测与解决
        conflicts = self._detect_conflicts(reasoning_results)
        resolved_conflicts = self._resolve_conflicts(conflicts, logical_chain)

        reasoning_results['logical_chain'] = logical_chain
        reasoning_results['conflicts'] = conflicts
        reasoning_results['resolved_conflicts'] = resolved_conflicts

        return reasoning_results

    def hybrid_reasoning(self,
                        market_data: pd.DataFrame,
                        query: str = None) -> Dict[str, Any]:
        """混合推理主函数"""
        print("🤖 执行神经符号混合推理...")

        # 第一步：神经网络特征提取
        neural_features = self.extract_neural_features(market_data)

        # 第二步：符号推理
        symbolic_results = self.symbolic_reasoning(neural_features, market_data)

        # 第三步：知识图谱增强
        knowledge_enhancement = self.knowledge_graph.enhance_reasoning(
            neural_features, symbolic_results
        )

        # 第四步：决策融合
        final_decision = self._fuse_decisions(neural_features, symbolic_results, knowledge_enhancement)

        # 第五步：生成解释
        explanation = self._generate_explanation(
            neural_features, symbolic_results, final_decision, query
        )

        # 第六步：置信度评估
        confidence_assessment = self._assess_confidence(
            neural_features, symbolic_results, final_decision
        )

        results = {
            'neural_features': neural_features,
            'symbolic_reasoning': symbolic_results,
            'knowledge_enhancement': knowledge_enhancement,
            'final_decision': final_decision,
            'explanation': explanation,
            'confidence': confidence_assessment,
            'reasoning_trace': self.reasoning_trace,
            'timestamp': pd.Timestamp.now()
        }

        return results

    def _calculate_momentum(self, prices: pd.Series) -> float:
        """计算动量特征"""
        if len(prices) < 5:
            return 0.0

        returns = prices.pct_change().dropna()
        if len(returns) == 0:
            return 0.0

        # 多时间尺度动量
        momentum_5 = returns.iloc[-5:].mean() if len(returns) >= 5 else 0
        momentum_10 = returns.iloc[-10:].mean() if len(returns) >= 10 else 0
        momentum_20 = returns.iloc[-20:].mean() if len(returns) >= 20 else 0

        # 加权合成
        total_weight = 0
        weighted_momentum = 0

        if len(returns) >= 5:
            weighted_momentum += momentum_5 * 0.5
            total_weight += 0.5
        if len(returns) >= 10:
            weighted_momentum += momentum_10 * 0.3
            total_weight += 0.3
        if len(returns) >= 20:
            weighted_momentum += momentum_20 * 0.2
            total_weight += 0.2

        return weighted_momentum / total_weight if total_weight > 0 else 0

    def _classify_volatility_regime(self, prices: pd.Series) -> str:
        """分类波动率制度"""
        if len(prices) < 10:
            return "unknown"

        returns = prices.pct_change().dropna()
        volatility = returns.std()

        # 基于历史分位数分类
        if volatility < np.percentile(returns.std(), 25):
            return "low_volatility"
        elif volatility < np.percentile(returns.std(), 75):
            return "normal_volatility"
        else:
            return "high_volatility"

    def _calculate_trend_strength(self, prices: pd.Series) -> float:
        """计算趋势强度"""
        if len(prices) < 10:
            return 0.0

        # 线性回归趋势
        x = np.arange(len(prices))
        y = prices.values

        slope, intercept = np.polyfit(x, y, 1)

        # 趋势强度标准化
        trend_strength = slope / np.mean(y) if np.mean(y) != 0 else 0

        return np.clip(trend_strength * 100, -1, 1)

    def _analyze_market_structure(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """分析市场结构"""
        structure = {}

        # 相关性聚类
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.7:
                    high_corr_pairs.append((
                        correlation_matrix.columns[i],
                        correlation_matrix.columns[j],
                        correlation_matrix.iloc[i, j]
                    ))

        structure['high_correlation_pairs'] = high_corr_pairs
        structure['market_clustering'] = self._cluster_market_structure(correlation_matrix)
        structure['centrality_measures'] = self._calculate_centrality(correlation_matrix)

        return structure

    def _extract_dependency_patterns(self, correlation_matrix: pd.DataFrame) -> Dict[str, Any]:
        """提取依赖模式"""
        patterns = {}

        # 主成分分析
        try:
            eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix.fillna(0))
            patterns['principal_components'] = {
                'eigenvalues': eigenvalues.real,
                'explained_variance_ratio': eigenvalues.real / np.sum(eigenvalues.real)
            }
        except:
            patterns['principal_components'] = {'eigenvalues': [], 'explained_variance_ratio': []}

        # 依赖网络属性
        patterns['network_density'] = self._calculate_network_density(correlation_matrix)
        patterns['dependency_strength'] = np.mean(np.abs(correlation_matrix.values))

        return patterns

    def _extract_temporal_patterns(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """提取时序模式"""
        patterns = {}

        if 'close' in market_data.columns:
            prices = market_data['close'].dropna()
            if len(prices) > 20:
                # 自相关分析
                autocorr = [prices.autocorr(lag=i) for i in range(1, min(11, len(prices)//2))]
                patterns['autocorrelation'] = autocorr

                # 季节性检测
                patterns['seasonality'] = self._detect_seasonality(prices)

        return patterns

    def _decompose_cycles(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """分解周期成分"""
        cycles = {}

        if 'close' in market_data.columns and len(market_data) > 50:
            prices = market_data['close'].dropna()

            # 简化的周期分解
            try:
                # FFT频谱分析
                fft_values = np.fft.fft(prices.values)
                frequencies = np.fft.fftfreq(len(prices))

                # 主频率识别
                power_spectrum = np.abs(fft_values) ** 2
                dominant_freq_idx = np.argmax(power_spectrum[1:len(power_spectrum)//2]) + 1

                cycles['dominant_frequency'] = frequencies[dominant_freq_idx]
                cycles['dominant_period'] = 1 / frequencies[dominant_freq_idx] if frequencies[dominant_freq_idx] != 0 else 0
                cycles['power_spectrum'] = power_spectrum[:len(power_spectrum)//2]

            except Exception as e:
                cycles['error'] = str(e)

        return cycles

    def _neural_to_symbolic(self, neural_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """神经网络特征转符号表示"""
        symbolic_facts = {}

        for feature_name, feature_value in neural_features.items():
            if isinstance(feature_value, (int, float)):
                # 数值特征符号化
                symbolic_facts[feature_name] = self._quantize_feature(feature_value, feature_name)
            elif isinstance(feature_value, dict):
                # 字典特征递归处理
                for sub_key, sub_value in feature_value.items():
                    symbolic_facts[f"{feature_name}_{sub_key}"] = self._quantize_feature(sub_value, f"{feature_name}_{sub_key}")

        return symbolic_facts

    def _quantize_feature(self, value: float, feature_name: str) -> str:
        """特征量化为符号"""
        if abs(value) < 1e-6:
            return "zero"
        elif value > 0:
            if value < 0.3:
                return "low_positive"
            elif value < 0.7:
                return "medium_positive"
            else:
                return "high_positive"
        else:
            if value > -0.3:
                return "low_negative"
            elif value > -0.7:
                return "medium_negative"
            else:
                return "high_negative"

    def _cluster_market_structure(self, correlation_matrix: pd.DataFrame) -> List[List[str]]:
        """市场结构聚类"""
        # 简化的层次聚类
        clusters = []
        assets = correlation_matrix.columns.tolist()

        # 基于高相关性分组
        used_assets = set()
        for asset in assets:
            if asset in used_assets:
                continue

            cluster = [asset]
            used_assets.add(asset)

            # 找高相关资产
            for other_asset in assets:
                if other_asset not in used_assets:
                    correlation = correlation_matrix.loc[asset, other_asset]
                    if not np.isnan(correlation) and abs(correlation) > 0.7:
                        cluster.append(other_asset)
                        used_assets.add(other_asset)

            if len(cluster) > 1:
                clusters.append(cluster)

        return clusters

    def _calculate_centrality(self, correlation_matrix: pd.DataFrame) -> Dict[str, float]:
        """计算中心性"""
        centralities = {}

        # 度中心性（高相关性连接数）
        for asset in correlation_matrix.columns:
            high_corr_count = np.sum(np.abs(correlation_matrix[asset]) > 0.7) - 1  # 减去自身
            centralities[asset] = high_corr_count

        return centralities

    def _calculate_network_density(self, correlation_matrix: pd.DataFrame) -> float:
        """计算网络密度"""
        n = len(correlation_matrix)
        if n <= 1:
            return 0.0

        # 强连接数（相关系数绝对值 > 0.5）
        strong_connections = 0
        for i in range(n):
            for j in range(i+1, n):
                if abs(correlation_matrix.iloc[i, j]) > 0.5:
                    strong_connections += 1

        max_possible_connections = n * (n - 1) / 2
        return strong_connections / max_possible_connections if max_possible_connections > 0 else 0

    def _detect_seasonality(self, prices: pd.Series) -> Dict[str, Any]:
        """检测季节性"""
        seasonality = {}

        if len(prices) < 20:
            return seasonality

        # 简单的季节性检测
        try:
            # 计算不同滞后期的自相关
            seasonal_lags = [5, 10, 20]  # 假设的周期
            seasonal_strengths = {}

            for lag in seasonal_lags:
                if len(prices) > lag:
                    autocorr = prices.autocorr(lag=lag)
                    seasonal_strengths[f"lag_{lag}"] = autocorr if not np.isnan(autocorr) else 0

            seasonality['seasonal_strengths'] = seasonal_strengths
            seasonality['dominant_seasonality'] = max(seasonal_strengths.items(), key=lambda x: abs(x[1]))[0] if seasonal_strengths else None

        except Exception as e:
            seasonality['error'] = str(e)

        return seasonality

    def _build_reasoning_chain(self, facts: Dict[str, Any], rule_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """构建推理链"""
        chain = []

        # 事实到推理的链
        for fact_name, fact_value in facts.items():
            chain.append({
                'type': 'fact',
                'content': f"{fact_name} is {fact_value}",
                'confidence': 0.9
            })

        # 规则应用
        for rule, result in rule_results.items():
            if isinstance(result, dict) and 'triggered' in result and result['triggered']:
                chain.append({
                    'type': 'rule_application',
                    'content': f"Rule '{rule}' triggered with confidence {result.get('confidence', 0)}",
                    'confidence': result.get('confidence', 0)
                })

        return chain

    def _detect_conflicts(self, reasoning_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测推理冲突"""
        conflicts = []

        # 检测相反的信号
        buy_signals = []
        sell_signals = []

        for rule, result in reasoning_results.items():
            if isinstance(result, dict) and result.get('triggered', False):
                if 'buy' in rule.lower():
                    buy_signals.append((rule, result.get('confidence', 0)))
                elif 'sell' in rule.lower():
                    sell_signals.append((rule, result.get('confidence', 0)))

        # 如果同时存在买卖信号，标记为冲突
        if buy_signals and sell_signals:
            conflicts.append({
                'type': 'buy_sell_conflict',
                'buy_signals': buy_signals,
                'sell_signals': sell_signals,
                'severity': 'high'
            })

        return conflicts

    def _resolve_conflicts(self, conflicts: List[Dict[str, Any]], reasoning_chain: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解决冲突"""
        resolved = {}

        for conflict in conflicts:
            if conflict['type'] == 'buy_sell_conflict':
                # 基于置信度解决冲突
                buy_confidence = max([conf for _, conf in conflict['buy_signals']], default=0)
                sell_confidence = max([conf for _, conf in conflict['sell_signals']], default=0)

                if buy_confidence > sell_confidence:
                    resolved['resolution'] = 'prefer_buy'
                    resolved['confidence'] = buy_confidence
                elif sell_confidence > buy_confidence:
                    resolved['resolution'] = 'prefer_sell'
                    resolved['confidence'] = sell_confidence
                else:
                    resolved['resolution'] = 'no_action'
                    resolved['confidence'] = 0

        return resolved

    def _fuse_decisions(self,
                       neural_features: Dict[str, np.ndarray],
                       symbolic_results: Dict[str, Any],
                       knowledge_enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """融合决策"""

        # 神经网络决策
        neural_decision = self._neural_decision(neural_features)

        # 符号推理决策
        symbolic_decision = self._symbolic_decision(symbolic_results)

        # 知识图谱增强决策
        knowledge_decision = self._knowledge_decision(knowledge_enhancement)

        # 加权融合
        final_decision = {
            'signal': self._weighted_signal_fusion(neural_decision, symbolic_decision, knowledge_decision),
            'confidence': self._calculate_fusion_confidence(neural_decision, symbolic_decision, knowledge_decision),
            'neural_contribution': neural_decision,
            'symbolic_contribution': symbolic_decision,
            'knowledge_contribution': knowledge_decision,
            'fusion_weights': {
                'neural': 0.4,
                'symbolic': 0.4,
                'knowledge': 0.2
            }
        }

        return final_decision

    def _neural_decision(self, neural_features: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """神经网络决策"""
        decision = {'signal': 0, 'confidence': 0}

        # 基于神经网络特征的简单决策逻辑
        if 'price_momentum' in neural_features:
            momentum = neural_features['price_momentum']
            if isinstance(momentum, (int, float)):
                decision['signal'] += np.sign(momentum) * 0.5
                decision['confidence'] += abs(momentum) * 0.3

        if 'trend_strength' in neural_features:
            trend = neural_features['trend_strength']
            if isinstance(trend, (int, float)):
                decision['signal'] += trend * 0.3
                decision['confidence'] += abs(trend) * 0.2

        decision['signal'] = np.clip(decision['signal'], -1, 1)
        decision['confidence'] = np.clip(decision['confidence'], 0, 1)

        return decision

    def _symbolic_decision(self, symbolic_results: Dict[str, Any]) -> Dict[str, Any]:
        """符号推理决策"""
        decision = {'signal': 0, 'confidence': 0}

        # 统计触发的规则
        triggered_rules = []
        for rule, result in symbolic_results.items():
            if isinstance(result, dict) and result.get('triggered', False):
                triggered_rules.append((rule, result.get('confidence', 0)))

        if triggered_rules:
            # 基于置信度加权
            total_confidence = sum(conf for _, conf in triggered_rules)
            if total_confidence > 0:
                for rule, conf in triggered_rules:
                    if 'buy' in rule.lower():
                        decision['signal'] += (conf / total_confidence) * 1.0
                    elif 'sell' in rule.lower():
                        decision['signal'] -= (conf / total_confidence) * 1.0

                decision['confidence'] = total_confidence / len(triggered_rules)

        decision['signal'] = np.clip(decision['signal'], -1, 1)
        decision['confidence'] = np.clip(decision['confidence'], 0, 1)

        return decision

    def _knowledge_decision(self, knowledge_enhancement: Dict[str, Any]) -> Dict[str, Any]:
        """知识图谱决策"""
        decision = {'signal': 0, 'confidence': 0}

        # 简化的知识图谱决策
        if 'similar_patterns' in knowledge_enhancement:
            similar_patterns = knowledge_enhancement['similar_patterns']
            if similar_patterns:
                avg_outcome = np.mean([p.get('outcome', 0) for p in similar_patterns])
                decision['signal'] = avg_outcome * 0.3
                decision['confidence'] = 0.5

        return decision

    def _weighted_signal_fusion(self, neural: Dict, symbolic: Dict, knowledge: Dict) -> float:
        """加权信号融合"""
        weights = {'neural': 0.4, 'symbolic': 0.4, 'knowledge': 0.2}

        fused_signal = (
            weights['neural'] * neural['signal'] +
            weights['symbolic'] * symbolic['signal'] +
            weights['knowledge'] * knowledge['signal']
        )

        return np.clip(fused_signal, -1, 1)

    def _calculate_fusion_confidence(self, neural: Dict, symbolic: Dict, knowledge: Dict) -> float:
        """计算融合置信度"""
        weights = {'neural': 0.4, 'symbolic': 0.4, 'knowledge': 0.2}

        fused_confidence = (
            weights['neural'] * neural['confidence'] +
            weights['symbolic'] * symbolic['confidence'] +
            weights['knowledge'] * knowledge['confidence']
        )

        return np.clip(fused_confidence, 0, 1)

    def _generate_explanation(self,
                            neural_features: Dict[str, np.ndarray],
                            symbolic_results: Dict[str, Any],
                            final_decision: Dict[str, Any],
                            query: str = None) -> str:
        """生成解释"""
        explanation_parts = []

        # 神经网络部分解释
        explanation_parts.append("🧠 神经网络分析:")
        if 'price_momentum' in neural_features:
            momentum = neural_features['price_momentum']
            if isinstance(momentum, (int, float)):
                direction = "上涨" if momentum > 0 else "下跌"
                explanation_parts.append(f"  - 价格动量显示{direction}趋势，强度: {abs(momentum):.3f}")

        # 符号推理部分解释
        explanation_parts.append("🔍 符号推理结果:")
        triggered_rules = []
        for rule, result in symbolic_results.items():
            if isinstance(result, dict) and result.get('triggered', False):
                triggered_rules.append(rule)

        if triggered_rules:
            for rule in triggered_rules[:3]:  # 显示前3个规则
                explanation_parts.append(f"  - 触发规则: {rule}")
        else:
            explanation_parts.append("  - 未触发明显规则")

        # 最终决策解释
        explanation_parts.append("🎯 最终决策:")
        signal_direction = "买入" if final_decision['signal'] > 0.1 else "卖出" if final_decision['signal'] < -0.1 else "持有"
        explanation_parts.append(f"  - 建议{signal_direction}，信号强度: {abs(final_decision['signal']):.3f}")
        explanation_parts.append(f"  - 置信度: {final_decision['confidence']:.3f}")

        return "\n".join(explanation_parts)

    def _assess_confidence(self,
                           neural_features: Dict[str, np.ndarray],
                           symbolic_results: Dict[str, Any],
                           final_decision: Dict[str, Any]) -> Dict[str, Any]:
        """评估置信度"""
        assessment = {
            'overall_confidence': final_decision['confidence'],
            'confidence_components': {
                'neural_confidence': final_decision['neural_contribution']['confidence'],
                'symbolic_confidence': final_decision['symbolic_contribution']['confidence'],
                'knowledge_confidence': final_decision['knowledge_contribution']['confidence']
            },
            'confidence_factors': []
        }

        # 置信度因子分析
        if final_decision['confidence'] > 0.8:
            assessment['confidence_factors'].append("高置信度：多系统一致确认")
        elif final_decision['confidence'] > 0.5:
            assessment['confidence_factors'].append("中等置信度：部分系统确认")
        else:
            assessment['confidence_factors'].append("低置信度：系统意见不一致")

        # 神经网络特征质量
        neural_quality = len(neural_features) / 10.0  # 简化的质量评估
        assessment['confidence_factors'].append(f"神经网络特征质量: {neural_quality:.2f}")

        # 符号规则一致性
        triggered_count = sum(1 for result in symbolic_results.values()
                           if isinstance(result, dict) and result.get('triggered', False))
        rule_consistency = triggered_count / len(self.symbolic_rules) if self.symbolic_rules else 0
        assessment['confidence_factors'].append(f"符号规则一致性: {rule_consistency:.2f}")

        return assessment

    def get_reasoning_summary(self) -> Dict[str, Any]:
        """获取推理总结"""
        return {
            'total_reasoning_steps': len(self.reasoning_trace),
            'explanation_depth': len(self.explanation_chain),
            'symbolic_rules_count': len(self.symbolic_rules),
            'confidence_threshold': self.confidence_threshold,
            'model_info': self.get_model_info()
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'Neuro-Symbolic Financial Reasoner',
            'neural_network_available': TORCH_AVAILABLE,
            'symbolic_reasoning_available': SYMPY_AVAILABLE,
            'hidden_dimensions': self.neural_hidden_dims,
            'reasoning_depth': self.reasoning_depth,
            'rules_count': len(self.symbolic_rules),
            'knowledge_graph_nodes': len(getattr(self.knowledge_graph, 'nodes', {}))
        }


class SymbolicReasoner:
    """符号推理器"""

    def __init__(self, rules: List[str]):
        self.rules = rules
        self.facts = {}
        self.inference_engine = InferenceEngine()

    def apply_rule(self, rule: str, facts: Dict[str, Any], market_data: pd.DataFrame) -> Dict[str, Any]:
        """应用规则"""
        try:
            # 解析规则
            conditions, conclusion = self._parse_rule(rule)

            # 检查条件
            conditions_met = self._check_conditions(conditions, facts, market_data)

            # 计算置信度
            confidence = self._calculate_rule_confidence(conditions_met, conditions)

            return {
                'triggered': conditions_met,
                'confidence': confidence,
                'conditions_checked': conditions,
                'conclusion': conclusion
            }

        except Exception as e:
            return {
                'triggered': False,
                'confidence': 0,
                'error': str(e)
            }

    def _parse_rule(self, rule: str) -> Tuple[List[str], str]:
        """解析规则"""
        if 'IF' in rule and 'THEN' in rule:
            parts = rule.split('THEN')
            condition_part = parts[0].replace('IF', '').strip()
            conclusion = parts[1].strip()

            # 简单的条件解析
            conditions = [cond.strip() for cond in condition_part.split('AND')]
            return conditions, conclusion
        else:
            return [], rule

    def _check_conditions(self, conditions: List[str], facts: Dict[str, Any], market_data: pd.DataFrame) -> bool:
        """检查条件"""
        for condition in conditions:
            if not self._evaluate_condition(condition, facts, market_data):
                return False
        return True

    def _evaluate_condition(self, condition: str, facts: Dict[str, Any], market_data: pd.DataFrame) -> bool:
        """评估单个条件"""
        try:
            # 简化的条件评估
            if '>' in condition:
                var, value = condition.split('>')
                var = var.strip()
                value = float(value.strip())

                # 从facts或market_data获取值
                if var in facts:
                    return float(str(facts[var]).replace('positive_', '1').replace('negative_', '-1')) > value
                elif var in market_data.columns:
                    return float(market_data[var].iloc[-1]) > value

            elif '<' in condition:
                var, value = condition.split('<')
                var = var.strip()
                value = float(value.strip())

                if var in facts:
                    return float(str(facts[var]).replace('positive_', '1').replace('negative_', '-1')) < value
                elif var in market_data.columns:
                    return float(market_data[var].iloc[-1]) < value

            return False

        except:
            return False

    def _calculate_rule_confidence(self, conditions_met: bool, conditions: List[str]) -> float:
        """计算规则置信度"""
        if not conditions_met:
            return 0.0

        # 基于条件数量和匹配度的简单置信度计算
        base_confidence = 0.7
        condition_bonus = min(len(conditions) * 0.1, 0.3)

        return min(base_confidence + condition_bonus, 1.0)


class InferenceEngine:
    """推理引擎"""
    def forward_chaining(self, facts: Dict[str, Any], rules: List[str]) -> Dict[str, Any]:
        """前向链推理"""
        new_facts = facts.copy()
        derived_facts = {}

        for rule in rules:
            # 这里可以实现更复杂的前向链推理
            pass

        return derived_facts

    def backward_chaining(self, goal: str, facts: Dict[str, Any], rules: List[str]) -> bool:
        """后向链推理"""
        # 这里可以实现后向链推理
        return False


class FinancialKnowledgeGraph:
    """金融知识图谱"""

    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.concepts = self._initialize_financial_concepts()

    def _initialize_financial_concepts(self) -> Dict[str, Any]:
        """初始化金融概念"""
        return {
            'market_regime': ['bull_market', 'bear_market', 'sideways_market'],
            'technical_patterns': ['head_and_shoulders', 'double_top', 'triangle'],
            'risk_factors': ['volatility', 'liquidity', 'correlation'],
            'market_anomalies': ['momentum_anomaly', 'value_anomaly', 'size_anomaly']
        }

    def enhance_reasoning(self,
                          neural_features: Dict[str, np.ndarray],
                          symbolic_results: Dict[str, Any]) -> Dict[str, Any]:
        """知识图谱增强推理"""
        enhancement = {}

        # 模式匹配
        enhancement['similar_patterns'] = self._find_similar_patterns(neural_features)

        # 概念关联
        enhancement['concept_relations'] = self._find_concept_relations(symbolic_results)

        # 历史案例
        enhancement['historical_cases'] = self._find_historical_cases(neural_features)

        return enhancement

    def _find_similar_patterns(self, features: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """查找相似模式"""
        # 简化的模式匹配
        similar_patterns = []

        if 'trend_strength' in features:
            trend = features['trend_strength']
            if isinstance(trend, (int, float)) and abs(trend) > 0.5:
                similar_patterns.append({
                    'pattern_type': 'strong_trend',
                    'similarity': abs(trend),
                    'outcome': 1.0 if trend > 0 else -1.0
                })

        return similar_patterns

    def _find_concept_relations(self, symbolic_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """查找概念关联"""
        relations = []

        # 简化的概念关联
        for rule, result in symbolic_results.items():
            if isinstance(result, dict) and result.get('triggered', False):
                if 'trend' in rule.lower():
                    relations.append({
                        'concept': 'trend_analysis',
                        'related_concepts': ['momentum', 'volatility', 'market_regime']
                    })

        return relations

    def _find_historical_cases(self, features: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """查找历史案例"""
        # 这里应该连接到历史数据库
        return []


# 便捷函数
def create_neuro_symbolic_reasoner(rules: List[str] = None) -> NeuroSymbolicFinancialReasoner:
    """创建神经符号推理器实例"""
    return NeuroSymbolicFinancialReasoner(symbolic_rules=rules)

def quick_financial_reasoning(market_data: pd.DataFrame, query: str = None) -> Dict[str, Any]:
    """快速金融推理"""
    reasoner = create_neuro_symbolic_reasoner()
    return reasoner.hybrid_reasoning(market_data, query)

def analyze_with_explanation(market_data: pd.DataFrame, query: str = None) -> str:
    """带解释的分析"""
    results = quick_financial_reasoning(market_data, query)
    return results.get('explanation', '分析失败')