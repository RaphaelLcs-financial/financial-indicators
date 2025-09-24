"""
因果发现和推断指标
Causal Discovery and Inference Indicator

基于因果推断的市场驱动因素分析和因果效应评估
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    from causalnex.structure import DAGRegressor
    from causalnex.structure.notears import from_pandas
    from causalnex.plots import plot_structure, NODE_STYLE, EDGE_STYLE
    from pgmpy.models import BayesianNetwork
    from pgmpy.estimators import BicScore, K2Score
    import networkx as nx
    CAUSAL_LIBRARIES_AVAILABLE = True
    print("🔗 因果推断库已启用")
except ImportError:
    CAUSAL_LIBRARIES_AVAILABLE = False
    print("⚠️ 因果推断库不可用")

class CausalDiscoveryIndicator:
    """
    因果发现交易指标

    基于因果推断理论识别市场驱动因素
    评估变量间的因果关系和因果效应
    """

    def __init__(self, method: str = 'notears', significance_level: float = 0.05,
                 max_lag: int = 5, bootstrap_samples: int = 1000):
        """
        初始化因果发现指标

        Args:
            method: 因果发现方法 ('notears', 'pc', 'ges', 'lingam')
            significance_level: 显著性水平
            max_lag: 最大滞后阶数
            bootstrap_samples: 自助法样本数
        """
        self.method = method
        self.significance_level = significance_level
        self.max_lag = max_lag
        self.bootstrap_samples = bootstrap_samples

        # 因果图
        self.causal_graph = nx.DiGraph()
        self.causal_strengths = {}
        self.causal_directions = {}

        # 分析结果
        self.causal_relationships = {}
        self.causal_effects = {}
        self.confounding_factors = {}

        # 稳健性检验
        self.robustness_results = {}

    def preprocess_time_series(self, data: pd.DataFrame) -> pd.DataFrame:
        """预处理时间序列数据"""
        print("📊 预处理时间序列数据...")

        processed_data = data.copy()

        # 计算收益率
        for col in processed_data.columns:
            if processed_data[col].dtype in ['float64', 'int64']:
                processed_data[f'{col}_returns'] = processed_data[col].pct_change().fillna(0)

        # 计算技术指标
        for col in processed_data.columns:
            if processed_data[col].dtype in ['float64', 'int64'] and not col.endswith('_returns'):
                # 移动平均
                processed_data[f'{col}_sma_5'] = processed_data[col].rolling(5).mean().fillna(0)
                processed_data[f'{col}_sma_20'] = processed_data[col].rolling(20).mean().fillna(0)

                # 波动率
                returns = processed_data[f'{col}_returns'] if f'{col}_returns' in processed_data.columns else processed_data[col].pct_change().fillna(0)
                processed_data[f'{col}_volatility'] = returns.rolling(10).std().fillna(0)

        return processed_data.fillna(0)

    def discover_causal_structure(self, data: pd.DataFrame) -> nx.DiGraph:
        """发现因果结构"""
        print(f"🔍 使用 {self.method} 方法发现因果结构...")

        if self.method == 'notears' and CAUSAL_LIBRARIES_AVAILABLE:
            return self._discover_notears(data)
        else:
            return self._discover_granger_causality(data)

    def _discover_notears(self, data: pd.DataFrame) -> nx.DiGraph:
        """使用NOTEARS方法发现因果结构"""
        try:
            # NOTEARS需要连续数据
            continuous_data = data.select_dtypes(include=[np.number])

            # 结构学习
            structural_model = from_pandas(continuous_data, tabu_edges=[], tabu_parent_nodes=[])

            # 创建有向图
            graph = nx.DiGraph()
            for i, col1 in enumerate(continuous_data.columns):
                for j, col2 in enumerate(continuous_data.columns):
                    if i != j and structural_model[i, j] != 0:
                        weight = abs(structural_model[i, j])
                        graph.add_edge(col1, col2, weight=weight)

            return graph
        except Exception as e:
            print(f"NOTEARS方法失败: {e}")
            return self._discover_granger_causality(data)

    def _discover_granger_causality(self, data: pd.DataFrame) -> nx.DiGraph:
        """使用Granger因果发现"""
        from statsmodels.tsa.stattools import grangercausalitytests

        graph = nx.DiGraph()
        variables = data.select_dtypes(include=[np.number]).columns

        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    try:
                        # Granger因果检验
                        test_result = grangercausalitytests(
                            data[[var1, var2]].dropna(),
                            maxlag=min(self.max_lag, len(data) // 10)
                        )

                        # 获取最小p值
                        min_p_value = min([result[0]['ssr_ftest'][1] for result in test_result.values()])

                        # 如果显著，添加边
                        if min_p_value < self.significance_level:
                            strength = -np.log(min_p_value)  # 负对数强度
                            graph.add_edge(var2, var1, weight=strength, p_value=min_p_value)

                    except Exception as e:
                        print(f"Granger因果检验失败 {var2} -> {var1}: {e}")

        return graph

    def estimate_causal_effects(self, data: pd.DataFrame, treatment: str, outcome: str) -> Dict[str, float]:
        """估计因果效应"""
        print(f"📈 估计因果效应: {treatment} -> {outcome}")

        # 方法1: 回归调整
        regression_effect = self._regression_adjustment(data, treatment, outcome)

        # 方法2: 倾向得分匹配
        propensity_effect = self._propensity_score_matching(data, treatment, outcome)

        # 方法3: 双重差分
        diff_effect = self._difference_in_differences(data, treatment, outcome)

        # 综合因果效应
        causal_effects = {
            'regression_adjustment': regression_effect,
            'propensity_score_matching': propensity_effect,
            'difference_in_differences': diff_effect,
            'average_effect': np.mean([regression_effect, propensity_effect, diff_effect])
        }

        self.causal_effects[(treatment, outcome)] = causal_effects

        return causal_effects

    def _regression_adjustment(self, data: pd.DataFrame, treatment: str, outcome: str) -> float:
        """回归调整估计因果效应"""
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        # 选择协变量
        covariates = [col for col in data.columns if col not in [treatment, outcome]]
        covariates = covariates[:10]  # 限制协变量数量

        # 准备数据
        X = data[[treatment] + covariates].fillna(0)
        y = data[outcome].fillna(0)

        # 标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 线性回归
        model = LinearRegression()
        model.fit(X_scaled, y)

        # 治理变量系数为因果效应
        treatment_idx = 0  # 治理变量是第一个特征
        causal_effect = model.coef_[treatment_idx]

        return causal_effect

    def _propensity_score_matching(self, data: pd.DataFrame, treatment: str, outcome: str) -> float:
        """倾向得分匹配"""
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors

        # 二值化处理变量
        treatment_median = data[treatment].median()
        treatment_binary = (data[treatment] > treatment_median).astype(int)

        # 选择协变量
        covariates = [col for col in data.columns if col not in [treatment, outcome]]
        covariates = covariates[:5]  # 限制数量

        # 计算倾向得分
        X = data[covariates].fillna(0)
        propensity_model = LogisticRegression()
        propensity_model.fit(X, treatment_binary)
        propensity_scores = propensity_model.predict_proba(X)[:, 1]

        # 匹配
        treated_mask = treatment_binary == 1
        control_mask = treatment_binary == 0

        if treated_mask.sum() > 0 and control_mask.sum() > 0:
            # 找到最近邻匹配
            treated_scores = propensity_scores[treated_mask]
            control_scores = propensity_scores[control_mask]

            treated_outcomes = data[outcome][treated_mask]
            control_outcomes = data[outcome][control_mask]

            # 简单匹配：比较平均值
            ate = treated_outcomes.mean() - control_outcomes.mean()
        else:
            ate = 0

        return ate

    def _difference_in_differences(self, data: pd.DataFrame, treatment: str, outcome: str) -> float:
        """双重差分法"""
        # 简化的DID实现
        mid_point = len(data) // 2

        # 前、后期的处理组和对照组
        treatment_binary = (data[treatment] > data[treatment].median()).astype(int)

        # 前、后期结果
        pre_treatment = data[outcome][:mid_point]
        post_treatment = data[outcome][mid_point:]

        pre_treatment_binary = treatment_binary[:mid_point]
        post_treatment_binary = treatment_binary[mid_point:]

        # 计算DID估计量
        pre_diff = pre_treatment[pre_treatment_binary == 1].mean() - pre_treatment[pre_treatment_binary == 0].mean()
        post_diff = post_treatment[post_treatment_binary == 1].mean() - post_treatment[post_treatment_binary == 0].mean()

        did_estimate = post_diff - pre_diff

        return did_estimate

    def identify_confounding_factors(self, causal_graph: nx.DiGraph, data: pd.DataFrame) -> Dict[str, List[str]]:
        """识别混杂因素"""
        print("🔍 识别混杂因素...")

        confounding_factors = {}

        for edge in causal_graph.edges():
            cause, effect = edge

            # 查找共同原因（混杂因素）
            common_causes = set()
            for node in causal_graph.nodes():
                if node != cause and node != effect:
                    # 检查是否同时指向cause和effect
                    has_path_to_cause = nx.has_path(causal_graph, node, cause) if node in causal_graph and cause in causal_graph else False
                    has_path_to_effect = nx.has_path(causal_graph, node, effect) if node in causal_graph and effect in causal_graph else False

                    if has_path_to_cause and has_path_to_effect:
                        common_causes.add(node)

            confounding_factors[(cause, effect)] = list(common_causes)

        self.confounding_factors = confounding_factors
        return confounding_factors

    def perform_counterfactual_analysis(self, data: pd.DataFrame, intervention: Dict[str, float]) -> Dict[str, Any]:
        """执行反事实分析"""
        print("🔮 执行反事实分析...")

        results = {}

        for variable, intervention_value in intervention.items():
            print(f"分析干预: {variable} = {intervention_value}")

            # 创建反事实数据
            counterfactual_data = data.copy()
            counterfactual_data[variable] = intervention_value

            # 预测反事实结果
            counterfactual_predictions = self._predict_counterfactual_outcomes(counterfactual_data, variable)

            # 计算因果效应
            baseline_predictions = self._predict_counterfactual_outcomes(data, variable)
            causal_effects = {outcome: counterfactual_predictions[outcome] - baseline_predictions[outcome]
                              for outcome in counterfactual_predictions}

            results[variable] = {
                'intervention_value': intervention_value,
                'counterfactual_predictions': counterfactual_predictions,
                'causal_effects': causal_effects,
                'effect_magnitude': np.mean(list(causal_effects.values()))
            }

        return results

    def _predict_counterfactual_outcomes(self, data: pd.DataFrame, intervention_variable: str) -> Dict[str, float]:
        """预测反事实结果"""
        # 简化的预测方法
        outcomes = {}

        # 选择受影响的变量
        affected_variables = [col for col in data.columns if col != intervention_variable]

        for outcome in affected_variables[:5]:  # 限制数量
            # 简单线性关系
            correlation = data[[intervention_variable, outcome]].corr().iloc[0, 1]
            if not np.isnan(correlation):
                # 基于相关性预测
                intervention_value = data[intervention_variable].iloc[-1]
                baseline_value = data[outcome].iloc[-1]
                predicted_change = correlation * (intervention_value - data[intervention_variable].mean())
                outcomes[outcome] = baseline_value + predicted_change
            else:
                outcomes[outcome] = data[outcome].iloc[-1]

        return outcomes

    def assess_causal_robustness(self, data: pd.DataFrame) -> Dict[str, Any]:
        """评估因果关系的稳健性"""
        print("🔍 评估因果关系的稳健性...")

        robustness_results = {}

        # 自助法检验
        bootstrap_results = self._bootstrap_causal_discovery(data)

        # 敏感性分析
        sensitivity_results = self._sensitivity_analysis(data)

        # 稳健性评分
        robustness_scores = self._calculate_robustness_scores(bootstrap_results, sensitivity_results)

        self.robustness_results = {
            'bootstrap_results': bootstrap_results,
            'sensitivity_results': sensitivity_results,
            'robustness_scores': robustness_scores
        }

        return self.robustness_results

    def _bootstrap_causal_discovery(self, data: pd.DataFrame) -> Dict[str, float]:
        """自助法因果发现"""
        edge_frequencies = {}

        for _ in range(self.bootstrap_samples):
            # 重采样
            sample_data = data.sample(n=len(data), replace=True)

            # 因果发现
            bootstrap_graph = self.discover_causal_structure(sample_data)

            # 记录边频率
            for edge in bootstrap_graph.edges():
                edge_key = (edge[0], edge[1])
                edge_frequencies[edge_key] = edge_frequencies.get(edge_key, 0) + 1

        # 计算频率
        for edge_key in edge_frequencies:
            edge_frequencies[edge_key] /= self.bootstrap_samples

        return edge_frequencies

    def _sensitivity_analysis(self, data: pd.DataFrame) -> Dict[str, float]:
        """敏感性分析"""
        sensitivity_results = {}

        # 添加噪声
        noise_levels = [0.01, 0.05, 0.1]

        for noise_level in noise_levels:
            noisy_data = data.copy()
            for col in noisy_data.select_dtypes(include=[np.number]).columns:
                noise = np.random.normal(0, noise_level * noisy_data[col].std(), len(noisy_data))
                noisy_data[col] += noise

            # 因果发现
            noisy_graph = self.discover_causal_structure(noisy_data)

            # 比较与原图的相似性
            original_graph = self.discover_causal_structure(data)
            similarity = self._calculate_graph_similarity(original_graph, noisy_graph)

            sensitivity_results[noise_level] = similarity

        return sensitivity_results

    def _calculate_graph_similarity(self, graph1: nx.DiGraph, graph2: nx.DiGraph) -> float:
        """计算图相似度"""
        edges1 = set(graph1.edges())
        edges2 = set(graph2.edges())

        if len(edges1) == 0 and len(edges2) == 0:
            return 1.0

        intersection = edges1.intersection(edges2)
        union = edges1.union(edges2)

        if len(union) == 0:
            return 0.0

        jaccard_similarity = len(intersection) / len(union)
        return jaccard_similarity

    def _calculate_robustness_scores(self, bootstrap_results: Dict, sensitivity_results: Dict) -> Dict[str, float]:
        """计算稳健性评分"""
        robustness_scores = {}

        # 边频率评分
        if bootstrap_results:
            avg_edge_frequency = np.mean(list(bootstrap_results.values()))
            robustness_scores['edge_frequency_score'] = avg_edge_frequency

        # 敏感性评分
        if sensitivity_results:
            avg_sensitivity = np.mean(list(sensitivity_results.values()))
            robustness_scores['sensitivity_score'] = avg_sensitivity

        # 综合稳健性评分
        if robustness_scores:
            robustness_scores['overall_robustness'] = np.mean(list(robustness_scores.values()))
        else:
            robustness_scores['overall_robustness'] = 0.5

        return robustness_scores

    def generate_causal_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成因果交易信号"""
        print("📊 生成因果交易信号...")

        # 因果发现
        causal_graph = self.discover_causal_structure(data)

        # 识别关键驱动因素
        key_drivers = self._identify_key_drivers(causal_graph)

        # 估计因果效应
        causal_effects = {}
        for driver in key_drivers[:3]:  # 分析前3个驱动因素
            for outcome in data.select_dtypes(include=[np.number]).columns[:3]:
                if driver != outcome:
                    effects = self.estimate_causal_effects(data, driver, outcome)
                    causal_effects[(driver, outcome)] = effects['average_effect']

        # 生成信号
        signals = []

        for i in range(len(data)):
            window_data = data.iloc[max(0, i-20):i+1]

            # 基于因果关系的信号
            causal_signal = self._calculate_causal_signal(window_data, causal_effects)

            # 基于稳健性的信号
            robustness_signal = self._calculate_robustness_signal(causal_effects)

            # 综合信号
            combined_signal = 0.6 * causal_signal + 0.4 * robustness_signal

            signals.append(np.clip(combined_signal, -1, 1))

        signals_df = pd.DataFrame({
            'signal': signals,
            'causal_confidence': self._calculate_causal_confidence(causal_effects),
            'robustness_score': self.robustness_results.get('robustness_scores', {}).get('overall_robustness', 0.5)
        }, index=data.index)

        return signals_df

    def _identify_key_drivers(self, causal_graph: nx.DiGraph) -> List[str]:
        """识别关键驱动因素"""
        # 基于出度识别驱动因素
        out_degrees = dict(causal_graph.out_degree())
        key_drivers = sorted(out_degrees.items(), key=lambda x: x[1], reverse=True)

        return [driver[0] for driver in key_drivers[:5]]

    def _calculate_causal_signal(self, data: pd.DataFrame, causal_effects: Dict) -> float:
        """计算因果信号"""
        signal = 0

        for (driver, outcome), effect in causal_effects.items():
            if driver in data.columns and outcome in data.columns:
                # 获取最新数据
                driver_value = data[driver].iloc[-1]
                outcome_value = data[outcome].iloc[-1]

                # 基于因果效应和当前值生成信号
                if effect > 0:  # 正向因果效应
                    if driver_value > data[driver].median():
                        signal += 0.5
                    else:
                        signal -= 0.3
                else:  # 负向因果效应
                    if driver_value > data[driver].median():
                        signal -= 0.5
                    else:
                        signal += 0.3

        return signal / max(len(causal_effects), 1)

    def _calculate_robustness_signal(self, causal_effects: Dict) -> float:
        """计算稳健性信号"""
        # 基于因果效应的稳健性生成信号
        if not causal_effects:
            return 0

        effect_magnitudes = [abs(effect) for effect in causal_effects.values()]
        avg_magnitude = np.mean(effect_magnitudes)

        # 效应越强，信号越强
        robustness_signal = min(avg_magnitude / 0.1, 1.0)  # 标准化

        return robustness_signal

    def _calculate_causal_confidence(self, causal_effects: Dict) -> float:
        """计算因果置信度"""
        if not causal_effects:
            return 0.5

        # 基于效应大小和一致性计算置信度
        effect_magnitudes = [abs(effect) for effect in causal_effects.values()]
        effect_consistency = 1 - (np.std(effect_magnitudes) / (np.mean(effect_magnitudes) + 1e-6))

        return effect_consistency

    def get_causal_insights(self) -> Dict[str, Any]:
        """获取因果洞察"""
        return {
            'causal_relationships': self.causal_relationships,
            'causal_effects': self.causal_effects,
            'confounding_factors': self.confounding_factors,
            'robustness_results': self.robustness_results,
            'method': self.method,
            'significance_level': self.significance_level
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'method': self.method,
            'significance_level': self.significance_level,
            'max_lag': self.max_lag,
            'bootstrap_samples': self.bootstrap_samples,
            'causal_libraries_available': CAUSAL_LIBRARIES_AVAILABLE,
            'graph_stats': {
                'num_nodes': self.causal_graph.number_of_nodes(),
                'num_edges': self.causal_graph.number_of_edges(),
                'density': nx.density(self.causal_graph) if len(self.causal_graph.nodes()) > 0 else 0
            },
            'model_type': 'Causal Discovery and Inference Indicator'
        }

# 便捷函数
def create_causal_discovery_indicator(method: str = 'notears') -> CausalDiscoveryIndicator:
    """创建因果发现指标实例"""
    return CausalDiscoveryIndicator(method)

def quick_causal_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """快速因果分析"""
    causal_analyzer = CausalDiscoveryIndicator()

    # 因果发现
    causal_graph = causal_analyzer.discover_causal_structure(data)

    # 稳健性评估
    robustness_results = causal_analyzer.assess_causal_robustness(data)

    # 生成交易信号
    signals = causal_analyzer.generate_causal_trading_signals(data)

    return {
        'causal_graph_stats': {
            'num_nodes': causal_graph.number_of_nodes(),
            'num_edges': causal_graph.number_of_edges()
        },
        'robustness_results': robustness_results,
        'latest_signal': signals['signal'].iloc[-1] if len(signals) > 0 else 0,
        'causal_confidence': signals['causal_confidence'].iloc[-1] if len(signals) > 0 else 0,
        'model_info': causal_analyzer.get_model_info()
    }