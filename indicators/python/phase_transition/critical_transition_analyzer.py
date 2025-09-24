"""
临界相变市场预警系统
Critical Phase Transition Market Warning System

基于统计物理和临界理论的金融市场相变检测与预警系统
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
    from torch.distributions import MultivariateNormal
    TORCH_AVAILABLE = True
    print("🧠 PyTorch 已启用")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 不可用")

try:
    from scipy import stats
    from scipy.optimize import minimize_scalar
    from scipy.signal import find_peaks
    from scipy.fft import fft, fftfreq
    SCIPY_AVAILABLE = True
    print("🔬 SciPy 已启用")
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy 不可用")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
    print("🕸️ NetworkX 已启用")
except ImportError:
    NETWORKX_AVAILABLE = False
    print("⚠️ NetworkX 不可用")

class CriticalTransitionAnalyzer:
    """
    临界相变分析器

    检测金融市场的临界相变现象，提供早期预警信号
    """

    def __init__(self,
                 window_size: int = 100,
                 detection_threshold: float = 2.0,
                 early_warning_threshold: float = 1.5,
                 sensitivity: float = 0.8):
        """
        初始化临界相变分析器

        Args:
            window_size: 分析窗口大小
            detection_threshold: 检测阈值
            early_warning_threshold: 早期预警阈值
            sensitivity: 系统灵敏度
        """
        self.window_size = window_size
        self.detection_threshold = detection_threshold
        self.early_warning_threshold = early_warning_threshold
        self.sensitivity = sensitivity

        # 临界指标
        self.critical_indicators = {}
        self.early_warning_signals = []
        self.phase_transition_history = []

        # 临界参数
        self.critical_exponents = {}
        self.correlation_length = {}
        self.order_parameters = {}

        # 预警状态
        self.warning_level = 0  # 0: 正常, 1: 警告, 2: 危险, 3: 临界
        self.warning_history = []

        # 系统状态
        self.system_state = 'normal'
        self.stability_metrics = {}
        self.resilience_indicators = {}

        # 相变检测模型
        self.phase_detection_model = None
        self.early_warning_model = None

        if TORCH_AVAILABLE:
            self._build_detection_models()

    def _build_detection_models(self):
        """构建检测模型"""
        # 临界相变检测模型
        self.phase_detection_model = CriticalPhaseDetector(
            input_dim=10,  # 多种临界指标
            hidden_dims=[64, 32, 16],
            sensitivity=self.sensitivity
        )

        # 早期预警模型
        self.early_warning_model = EarlyWarningPredictor(
            window_size=self.window_size,
            prediction_horizon=20
        )

    def analyze_market_phase_transition(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """分析市场相变"""
        print("🌀 分析市场临界相变...")

        # 第一步：计算临界指标
        critical_metrics = self._calculate_critical_metrics(market_data)

        # 第二步：检测早期预警信号
        early_warnings = self._detect_early_warning_signals(critical_metrics)

        # 第三步：识别临界相变
        phase_transitions = self._identify_phase_transitions(critical_metrics)

        # 第四步：计算临界指数
        critical_exponents = self._calculate_critical_exponents(market_data, critical_metrics)

        # 第五步：评估系统稳定性
        stability_assessment = self._assess_system_stability(critical_metrics)

        # 第六步：预测相变风险
        risk_prediction = self._predict_transition_risk(critical_metrics, early_warnings)

        # 第七步：生成预警报告
        warning_report = self._generate_warning_report(
            critical_metrics, early_warnings, phase_transitions, risk_prediction
        )

        results = {
            'critical_metrics': critical_metrics,
            'early_warnings': early_warnings,
            'phase_transitions': phase_transitions,
            'critical_exponents': critical_exponents,
            'stability_assessment': stability_assessment,
            'risk_prediction': risk_prediction,
            'warning_report': warning_report,
            'system_state': self.system_state,
            'warning_level': self.warning_level,
            'analysis_timestamp': pd.Timestamp.now()
        }

        # 更新历史记录
        self.phase_transition_history.append(results)
        self.warning_history.append(self.warning_level)

        return results

    def _calculate_critical_metrics(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """计算临界指标"""
        print("📊 计算临界指标...")

        metrics = {}

        # 价格数据提取
        if 'close' in market_data.columns:
            prices = market_data['close'].dropna()
            if len(prices) > self.window_size:
                # 波动率指标
                metrics['volatility_metrics'] = self._calculate_volatility_metrics(prices)

                # 相关性指标
                metrics['correlation_metrics'] = self._calculate_correlation_metrics(market_data)

                # 分形指标
                metrics['fractal_metrics'] = self._calculate_fractal_metrics(prices)

                # 信息论指标
                metrics['information_metrics'] = self._calculate_information_metrics(prices)

                # 临界接近度指标
                metrics['critical_proximity'] = self._calculate_critical_proximity(prices)

                # 序统计量
                metrics['order_statistics'] = self._calculate_order_statistics(prices)

                # 功率谱指标
                metrics['power_spectrum_metrics'] = self._calculate_power_spectrum_metrics(prices)

                # 网络指标
                metrics['network_metrics'] = self._calculate_network_metrics(market_data)

                # 时序指标
                metrics['temporal_metrics'] = self._calculate_temporal_metrics(prices)

        return metrics

    def _calculate_volatility_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """计算波动率指标"""
        metrics = {}

        returns = prices.pct_change().dropna()
        if len(returns) < 10:
            return metrics

        # 基础波动率
        metrics['volatility'] = returns.std()

        # 条件波动率
        metrics['conditional_volatility'] = self._calculate_conditional_volatility(returns)

        # 波动率聚集性
        metrics['volatility_clustering'] = self._calculate_volatility_clustering(returns)

        # 波动率偏度
        metrics['volatility_skewness'] = returns.skew()

        # 波动率峰度
        metrics['volatility_kurtosis'] = returns.kurtosis()

        # 波动率长记忆性
        metrics['volatility_long_memory'] = self._calculate_long_memory(returns)

        return metrics

    def _calculate_conditional_volatility(self, returns: pd.Series) -> float:
        """计算条件波动率"""
        if len(returns) < 20:
            return 0.0

        # GARCH(1,1) 简化版本
        omega = 0.1 * returns.var()
        alpha = 0.1
        beta = 0.85

        conditional_var = [omega / (1 - alpha - beta)]  # 长期方差

        for i in range(1, min(len(returns), 50)):
            new_var = omega + alpha * returns.iloc[-i]**2 + beta * conditional_var[-1]
            conditional_var.append(new_var)

        return np.sqrt(conditional_var[-1])

    def _calculate_volatility_clustering(self, returns: pd.Series) -> float:
        """计算波动率聚集性"""
        if len(returns) < 20:
            return 0.0

        # 计算绝对收益的自相关
        abs_returns = np.abs(returns)
        autocorr = [abs_returns.autocorr(lag=i) for i in range(1, min(11, len(returns)//2))]

        return np.mean(np.abs(autocorr))

    def _calculate_long_memory(self, returns: pd.Series) -> float:
        """计算长记忆性"""
        if len(returns) < 50:
            return 0.0

        # 简化的长记忆性检测
        abs_returns = np.abs(returns)

        # 计算不同滞后期的自相关
        lags = range(1, min(21, len(returns)//4))
        autocorrs = [abs_returns.autocorr(lag=lag) for lag in lags]

        # 拟合幂律衰减
        if len(autocorrs) > 5:
            log_lags = np.log(lags)
            log_autocorrs = np.log(np.abs(autocorrs))

            # 线性回归拟合
            slope, _ = np.polyfit(log_lags, log_autocorrs, 1)

            return -slope  # 赫斯特指数的近似

        return 0.5

    def _calculate_correlation_metrics(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """计算相关性指标"""
        metrics = {}

        numeric_columns = market_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < 2:
            return metrics

        # 相关性矩阵
        correlation_matrix = market_data[numeric_columns].corr()

        # 平均相关性
        avg_correlation = correlation_matrix.abs().mean()
        metrics['average_correlation'] = avg_correlation

        # 相关性波动率
        metrics['correlation_volatility'] = correlation_matrix.values.std()

        # 最大相关性
        metrics['maximum_correlation'] = correlation_matrix.abs().max().max()

        # 相关性偏度
        correlations_flat = correlation_matrix.values.flatten()
        correlations_flat = correlations_flat[~np.isnan(correlations_flat)]
        if len(correlations_flat) > 0:
            metrics['correlation_skewness'] = stats.skew(correlations_flat)

        # 相关性系统风险
        metrics['systemic_risk_measure'] = self._calculate_systemic_risk(correlation_matrix)

        return metrics

    def _calculate_systemic_risk(self, correlation_matrix: pd.DataFrame) -> float:
        """计算系统性风险"""
        try:
            # 最大特征值（系统性风险指标）
            eigenvalues = np.linalg.eigvals(correlation_matrix.fillna(0))
            max_eigenvalue = np.max(eigenvalues.real)

            # 归一化
            n = len(correlation_matrix)
            normalized_max_eigenvalue = max_eigenvalue / n

            return normalized_max_eigenvalue
        except:
            return 0.0

    def _calculate_fractal_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """计算分形指标"""
        metrics = {}

        if len(prices) < 50:
            return metrics

        # 赫斯特指数
        metrics['hurst_exponent'] = self._calculate_hurst_exponent(prices)

        # 分形维数
        metrics['fractal_dimension'] = self._calculate_fractal_dimension(prices)

        # 重标极差分析
        metrics['rescaled_range'] = self._calculate_rescaled_range(prices)

        return metrics

    def _calculate_hurst_exponent(self, prices: pd.Series) -> float:
        """计算赫斯特指数"""
        if len(prices) < 100:
            return 0.5

        try:
            # R/S分析
            N = len(prices)
            max_k = min(N // 4, 50)

            R_S_values = []
            k_values = []

            for k in range(10, max_k + 1):
                # 分割时间序列
                segments = N // k
                if segments < 2:
                    continue

                R_S_ratios = []
                for i in range(segments):
                    segment = prices.iloc[i*k:(i+1)*k]
                    if len(segment) < 2:
                        continue

                    # 计算累积偏差
                    mean = segment.mean()
                    cumulative_deviation = (segment - mean).cumsum()
                    R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
                    S = segment.std()

                    if S > 0:
                        R_S_ratios.append(R / S)

                if R_S_ratios:
                    R_S_values.append(np.mean(R_S_ratios))
                    k_values.append(k)

            if len(R_S_values) > 5:
                # 对数线性回归
                log_k = np.log(k_values)
                log_R_S = np.log(R_S_values)

                slope, _ = np.polyfit(log_k, log_R_S, 1)
                return slope

        except Exception as e:
            print(f"⚠️ 赫斯特指数计算失败: {e}")

        return 0.5

    def _calculate_fractal_dimension(self, prices: pd.Series) -> float:
        """计算分形维数"""
        if len(prices) < 20:
            return 1.0

        try:
            # 简化的盒维数计算
            N = len(prices)
            price_range = prices.max() - prices.min()

            if price_range == 0:
                return 1.0

            # 不同尺度的盒子
            scales = np.logspace(0, np.log10(N//10), 10)
            counts = []

            for scale in scales:
                box_size = price_range / scale
                if box_size <= 0:
                    continue

                # 计算需要的盒子数
                normalized_prices = (prices - prices.min()) / price_range
                boxes = set()
                for i, price in enumerate(normalized_prices):
                    box_index = int(price * scale)
                    boxes.add(box_index)

                counts.append(len(boxes))

            if len(counts) > 5:
                # 对数线性回归
                log_scales = np.log(scales[:len(counts)])
                log_counts = np.log(counts)

                slope, _ = np.polyfit(log_scales, log_counts, 1)
                return -slope

        except Exception as e:
            print(f"⚠️ 分形维数计算失败: {e}")

        return 1.0

    def _calculate_rescaled_range(self, prices: pd.Series) -> float:
        """计算重标极差"""
        if len(prices) < 20:
            return 0.0

        try:
            # 简化的R/S计算
            N = len(prices)
            mean_price = prices.mean()
            cumulative_deviation = (prices - mean_price).cumsum()

            R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
            S = prices.std()

            return R / S if S > 0 else 0.0

        except:
            return 0.0

    def _calculate_information_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """计算信息论指标"""
        metrics = {}

        if len(prices) < 20:
            return metrics

        returns = prices.pct_change().dropna()
        if len(returns) < 10:
            return metrics

        # 近似熵
        metrics['approximate_entropy'] = self._calculate_approximate_entropy(returns)

        # 样本熵
        metrics['sample_entropy'] = self._calculate_sample_entropy(returns)

        # 排列熵
        metrics['permutation_entropy'] = self._calculate_permutation_entropy(returns)

        # 李亚普诺夫指数
        metrics['lyapunov_exponent'] = self._calculate_lyapunov_exponent(returns)

        return metrics

    def _calculate_approximate_entropy(self, data: pd.Series, m: int = 2, r: float = 0.2) -> float:
        """计算近似熵"""
        if len(data) < m + 1:
            return 0.0

        try:
            N = len(data)
            data_std = data.std()

            if data_std == 0:
                return 0.0

            r_value = r * data_std

            # 计算模板匹配
            def _count_matches(data, m, r):
                matches = 0
                total = 0

                for i in range(N - m + 1):
                    template = data.iloc[i:i+m]
                    for j in range(N - m + 1):
                        if i != j:
                            comparison = data.iloc[j:j+m]
                            if np.max(np.abs(template - comparison)) <= r_value:
                                matches += 1
                    total += 1

                return matches / total if total > 0 else 0

            # 计算m和m+1维的匹配
            phi_m = np.log(_count_matches(data, m, r) + 1e-10)
            phi_m1 = np.log(_count_matches(data, m+1, r) + 1e-10)

            return phi_m - phi_m1

        except:
            return 0.0

    def _calculate_sample_entropy(self, data: pd.Series, m: int = 2, r: float = 0.2) -> float:
        """计算样本熵"""
        if len(data) < m + 1:
            return 0.0

        try:
            N = len(data)
            data_std = data.std()

            if data_std == 0:
                return 0.0

            r_value = r * data_std

            # 计算模板匹配（不包括自匹配）
            def _count_matches_sample(data, m, r):
                matches_A = 0
                matches_B = 0

                for i in range(N - m):
                    template_m = data.iloc[i:i+m]
                    template_m1 = data.iloc[i:i+m+1]

                    for j in range(N - m):
                        if i != j:
                            comparison_m = data.iloc[j:j+m]
                            comparison_m1 = data.iloc[j:j+m+1]

                            if np.max(np.abs(template_m - comparison_m)) <= r_value:
                                matches_A += 1
                                if np.max(np.abs(template_m1 - comparison_m1)) <= r_value:
                                    matches_B += 1

                return matches_A, matches_B

            matches_A, matches_B = _count_matches_sample(data, m, r)

            if matches_A > 0:
                return -np.log(matches_B / matches_A)
            else:
                return 0.0

        except:
            return 0.0

    def _calculate_permutation_entropy(self, data: pd.Series, m: int = 3) -> float:
        """计算排列熵"""
        if len(data) < m + 1:
            return 0.0

        try:
            N = len(data)
            patterns = []

            for i in range(N - m + 1):
                # 提取模式
                pattern = data.iloc[i:i+m]
                # 排序获取模式类型
                sorted_indices = np.argsort(pattern.values)
                pattern_type = tuple(sorted_indices)

                patterns.append(pattern_type)

            # 计算模式频率
            unique_patterns, counts = np.unique(patterns, return_counts=True)
            probabilities = counts / len(patterns)

            # 计算熵
            entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

            # 归一化
            max_entropy = np.log(np.math.factorial(m))

            return entropy / max_entropy if max_entropy > 0 else 0.0

        except:
            return 0.0

    def _calculate_lyapunov_exponent(self, data: pd.Series) -> float:
        """计算李亚普诺夫指数"""
        if len(data) < 50:
            return 0.0

        try:
            # 简化的李亚普诺夫指数计算
            N = len(data)
            m = min(10, N // 5)  # 嵌入维度
            tau = 1  # 时间延迟

            # 相空间重构
            trajectories = []
            for i in range(N - (m-1)*tau):
                trajectory = data.iloc[i:i+(m-1)*tau+1:tau].values
                trajectories.append(trajectory)

            if len(trajectories) < 10:
                return 0.0

            # 计算最近邻发散
            divergences = []
            for i, traj in enumerate(trajectories[:-1]):
                # 找到最近邻
                distances = [np.linalg.norm(traj - other) for j, other in enumerate(trajectories) if i != j]
                if distances:
                    min_distance = min(distances)
                    # 计算一步后的发散
                    if i+1 < len(trajectories):
                        next_distance = np.linalg.norm(trajectories[i+1] - trajectories[distances.index(min_distance)+1])
                        if min_distance > 0:
                            divergence = np.log(next_distance / min_distance)
                            divergences.append(divergence)

            if divergences:
                return np.mean(divergences)

        except:
            return 0.0

        return 0.0

    def _calculate_critical_proximity(self, prices: pd.Series) -> Dict[str, float]:
        """计算临界接近度"""
        proximity = {}

        if len(prices) < 50:
            return proximity

        returns = prices.pct_change().dropna()
        if len(returns) < 20:
            return proximity

        # 价格序列临界接近度
        proximity['price_criticality'] = self._assess_price_criticality(prices)

        # 波动率临界接近度
        proximity['volatility_criticality'] = self._assess_volatility_criticality(returns)

        # 相关性临界接近度
        proximity['correlation_criticality'] = self._assess_correlation_criticality(returns)

        # 综合临界接近度
        criticality_scores = [
            proximity.get('price_criticality', 0),
            proximity.get('volatility_criticality', 0),
            proximity.get('correlation_criticality', 0)
        ]
        proximity['overall_criticality'] = np.mean([score for score in criticality_scores if score > 0])

        return proximity

    def _assess_price_criticality(self, prices: pd.Series) -> float:
        """评估价格临界性"""
        try:
            # 价格序列的统计特性
            returns = prices.pct_change().dropna()

            # 价格变动的方差
            variance = returns.var()

            # 价格变动的偏度
            skewness = returns.skew()

            # 价格变动的峰度
            kurtosis = returns.kurtosis()

            # 临界性评分（综合指标）
            criticality_score = 0

            # 高波动率贡献
            if variance > 0.01:  # 日波动率超过1%
                criticality_score += 0.3

            # 极端偏度贡献
            if abs(skewness) > 1:
                criticality_score += 0.3

            # 高峰度贡献
            if kurtosis > 3:
                criticality_score += 0.4

            return min(criticality_score, 1.0)

        except:
            return 0.0

    def _assess_volatility_criticality(self, returns: pd.Series) -> float:
        """评估波动率临界性"""
        try:
            # 波动率聚集性
            abs_returns = np.abs(returns)
            clustering = abs_returns.autocorr()

            # 波动率长记忆性
            long_memory = self._calculate_long_memory(returns)

            criticality_score = 0

            # 强聚集性
            if clustering > 0.2:
                criticality_score += 0.5

            # 长记忆性
            if long_memory > 0.6:
                criticality_score += 0.5

            return min(criticality_score, 1.0)

        except:
            return 0.0

    def _assess_correlation_criticality(self, returns: pd.Series) -> float:
        """评估相关性临界性"""
        try:
            # 自相关分析
            autocorrs = [returns.autocorr(lag=i) for i in range(1, min(11, len(returns)//2))]

            if not autocorrs:
                return 0.0

            # 自相关的持续性
            persistence = np.mean(np.abs(autocorrs))

            criticality_score = 0

            # 强自相关
            if persistence > 0.3:
                criticality_score += 1.0

            return min(criticality_score, 1.0)

        except:
            return 0.0

    def _calculate_order_statistics(self, prices: pd.Series) -> Dict[str, float]:
        """计算序统计量"""
        order_stats = {}

        if len(prices) < 20:
            return order_stats

        returns = prices.pct_change().dropna()
        if len(returns) < 10:
            return order_stats

        # 分位数分析
        quantiles = [0.05, 0.25, 0.5, 0.75, 0.95]
        for q in quantiles:
            order_stats[f'quantile_{int(q*100)}'] = returns.quantile(q)

        # 极值统计
        order_stats['max_return'] = returns.max()
        order_stats['min_return'] = returns.min()
        order_stats['range'] = returns.max() - returns.min()

        # 极值间隔
        order_stats['extreme_interval'] = self._calculate_extreme_interval(returns)

        return order_stats

    def _calculate_extreme_interval(self, returns: pd.Series, threshold: float = 0.95) -> float:
        """计算极值间隔"""
        try:
            # 确定极值阈值
            high_threshold = returns.quantile(threshold)
            low_threshold = returns.quantile(1 - threshold)

            # 找到极值点
            extreme_points = returns[(returns > high_threshold) | (returns < low_threshold)]

            if len(extreme_points) < 2:
                return 0.0

            # 计算平均间隔
            intervals = []
            for i in range(1, len(extreme_points)):
                interval = extreme_points.index[i] - extreme_points.index[i-1]
                intervals.append(interval)

            return np.mean(intervals) if intervals else 0.0

        except:
            return 0.0

    def _calculate_power_spectrum_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """计算功率谱指标"""
        spectrum_metrics = {}

        if len(prices) < 50:
            return spectrum_metrics

        try:
            # FFT变换
            returns = prices.pct_change().dropna()
            if len(returns) < 20:
                return spectrum_metrics

            # 去除趋势
            detrended = returns - returns.mean()

            # FFT
            fft_values = fft(detrended.values)
            power_spectrum = np.abs(fft_values) ** 2
            frequencies = fftfreq(len(detrended))

            # 只考虑正频率
            positive_freq_idx = frequencies > 0
            positive_frequencies = frequencies[positive_freq_idx]
            positive_power = power_spectrum[positive_freq_idx]

            if len(positive_power) > 0:
                # 功率谱斜率
                log_freq = np.log(positive_frequencies[1:])  # 去除直流分量
                log_power = np.log(positive_power[1:])

                if len(log_freq) > 5:
                    slope, _ = np.polyfit(log_freq, log_power, 1)
                    spectrum_metrics['power_slope'] = slope

                # 光谱熵
                power_prob = positive_power / np.sum(positive_power)
                spectral_entropy = -np.sum(power_prob * np.log(power_prob + 1e-10))
                spectrum_metrics['spectral_entropy'] = spectral_entropy

                # 主频率
                dominant_freq_idx = np.argmax(positive_power)
                spectrum_metrics['dominant_frequency'] = positive_frequencies[dominant_freq_idx]

        except:
            pass

        return spectrum_metrics

    def _calculate_network_metrics(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """计算网络指标"""
        network_metrics = {}

        if not NETWORKX_AVAILABLE:
            return network_metrics

        numeric_columns = market_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) < 3:
            return network_metrics

        try:
            # 构建相关性网络
            correlation_matrix = market_data[numeric_columns].corr()
            adjacency_matrix = correlation_matrix.abs()

            # 创建网络
            G = nx.from_numpy_array(adjacency_matrix.values)

            # 网络密度
            network_metrics['network_density'] = nx.density(G)

            # 聚类系数
            network_metrics['clustering_coefficient'] = nx.average_clustering(G)

            # 平均路径长度
            if nx.is_connected(G):
                network_metrics['average_path_length'] = nx.average_shortest_path_length(G)
            else:
                network_metrics['average_path_length'] = 0.0

            # 网络中心性
            centrality = nx.degree_centrality(G)
            network_metrics['centrality_entropy'] = self._calculate_entropy(list(centrality.values()))

        except:
            pass

        return network_metrics

    def _calculate_entropy(self, values: List[float]) -> float:
        """计算熵"""
        if not values:
            return 0.0

        # 归一化
        values = np.array(values)
        values = values / np.sum(values)

        # 计算熵
        entropy = -np.sum(values * np.log(values + 1e-10))

        return entropy

    def _calculate_temporal_metrics(self, prices: pd.Series) -> Dict[str, float]:
        """计算时序指标"""
        temporal_metrics = {}

        if len(prices) < 30:
            return temporal_metrics

        returns = prices.pct_change().dropna()
        if len(returns) < 20:
            return temporal_metrics

        # 时序复杂度
        temporal_metrics['temporal_complexity'] = self._calculate_temporal_complexity(returns)

        # 不可预测性
        temporal_metrics['unpredictability'] = self._calculate_unpredictability(returns)

        # 记忆性
        temporal_metrics['memory_strength'] = self._calculate_memory_strength(returns)

        return temporal_metrics

    def _calculate_temporal_complexity(self, returns: pd.Series) -> float:
        """计算时序复杂度"""
        try:
            # 基于排列熵的复杂度
            perm_entropy = self._calculate_permutation_entropy(returns, m=4)

            # 基于样本熵的复杂度
            sample_entropy = self._calculate_sample_entropy(returns, m=2, r=0.2)

            # 综合复杂度
            complexity = (perm_entropy + sample_entropy) / 2

            return min(complexity, 1.0)

        except:
            return 0.0

    def _calculate_unpredictability(self, returns: pd.Series) -> float:
        """计算不可预测性"""
        try:
            # 基于自相关的不可预测性
            autocorrs = [returns.autocorr(lag=i) for i in range(1, min(6, len(returns)//3))]

            if autocorrs:
                # 自相关的平均强度
                autocorr_strength = np.mean(np.abs(autocorrs))
                unpredictability = 1.0 - autocorr_strength
                return max(0.0, unpredictability)

        except:
            pass

        return 0.5

    def _calculate_memory_strength(self, returns: pd.Series) -> float:
        """计算记忆强度"""
        try:
            # 长记忆性
            hurst = self._calculate_hurst_exponent(returns)

            # 记忆强度
            if hurst > 0.5:
                memory_strength = (hurst - 0.5) * 2  # 0到1之间
            elif hurst < 0.5:
                memory_strength = (0.5 - hurst) * 2  # 反记忆
            else:
                memory_strength = 0.0

            return min(memory_strength, 1.0)

        except:
            return 0.0

    def _detect_early_warning_signals(self, critical_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """检测早期预警信号"""
        print("⚠️ 检测早期预警信号...")

        warnings = []

        # 临界接近度预警
        if 'critical_proximity' in critical_metrics:
            proximity = critical_metrics['critical_proximity']
            overall_criticality = proximity.get('overall_criticality', 0)

            if overall_criticality > self.early_warning_threshold:
                warnings.append({
                    'type': 'critical_proximity',
                    'severity': 'high' if overall_criticality > self.detection_threshold else 'medium',
                    'value': overall_criticality,
                    'threshold': self.early_warning_threshold,
                    'description': '市场接近临界状态'
                })

        # 波动率预警
        if 'volatility_metrics' in critical_metrics:
            vol_metrics = critical_metrics['volatility_metrics']
            volatility = vol_metrics.get('volatility', 0)
            clustering = vol_metrics.get('volatility_clustering', 0)

            if volatility > 0.05:  # 日波动率超过5%
                warnings.append({
                    'type': 'high_volatility',
                    'severity': 'high',
                    'value': volatility,
                    'threshold': 0.05,
                    'description': '市场波动率异常升高'
                })

            if clustering > 0.3:
                warnings.append({
                    'type': 'volatility_clustering',
                    'severity': 'medium',
                    'value': clustering,
                    'threshold': 0.3,
                    'description': '检测到波动率聚集现象'
                })

        # 相关性预警
        if 'correlation_metrics' in critical_metrics:
            corr_metrics = critical_metrics['correlation_metrics']
            systemic_risk = corr_metrics.get('systemic_risk_measure', 0)

            if systemic_risk > 0.8:
                warnings.append({
                    'type': 'systemic_risk',
                    'severity': 'high',
                    'value': systemic_risk,
                    'threshold': 0.8,
                    'description': '系统性风险显著增加'
                })

        # 分形预警
        if 'fractal_metrics' in critical_metrics:
            fractal_metrics = critical_metrics['fractal_metrics']
            hurst = fractal_metrics.get('hurst_exponent', 0.5)

            if hurst > 0.8:
                warnings.append({
                    'type': 'persistent_trend',
                    'severity': 'medium',
                    'value': hurst,
                    'threshold': 0.8,
                    'description': '检测到强持续性趋势'
                })

        # 信息论预警
        if 'information_metrics' in critical_metrics:
            info_metrics = critical_metrics['information_metrics']
            entropy = info_metrics.get('approximate_entropy', 0)

            if entropy < 0.5:
                warnings.append({
                    'type': 'low_complexity',
                    'severity': 'medium',
                    'value': entropy,
                    'threshold': 0.5,
                    'description': '市场复杂度降低，可能预示相变'
                })

        return warnings

    def _identify_phase_transitions(self, critical_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """识别临界相变"""
        print("🔄 识别临界相变...")

        transitions = []

        # 综合临界性评分
        criticality_score = self._calculate_composite_criticality(critical_metrics)

        # 判断相变状态
        if criticality_score > self.detection_threshold:
            transitions.append({
                'type': 'critical_transition',
                'severity': 'critical',
                'criticality_score': criticality_score,
                'threshold': self.detection_threshold,
                'description': '检测到临界相变，系统可能发生重大转变',
                'timestamp': pd.Timestamp.now()
            })

            # 更新系统状态
            self.system_state = 'critical'
            self.warning_level = 3
        elif criticality_score > self.early_warning_threshold:
            transitions.append({
                'type': 'approaching_transition',
                'severity': 'warning',
                'criticality_score': criticality_score,
                'threshold': self.early_warning_threshold,
                'description': '系统接近临界状态，需要密切关注',
                'timestamp': pd.Timestamp.now()
            })

            # 更新系统状态
            self.system_state = 'warning'
            self.warning_level = 2
        else:
            self.system_state = 'normal'
            self.warning_level = 1

        return transitions

    def _calculate_composite_criticality(self, critical_metrics: Dict[str, Any]) -> float:
        """计算综合临界性评分"""
        criticality_components = []

        # 从各个指标组提取临界性
        if 'critical_proximity' in critical_metrics:
            proximity = critical_metrics['critical_proximity']
            overall_criticality = proximity.get('overall_criticality', 0)
            criticality_components.append(overall_criticality)

        if 'volatility_metrics' in critical_metrics:
            vol_metrics = critical_metrics['volatility_metrics']
            volatility = vol_metrics.get('volatility', 0)
            clustering = vol_metrics.get('volatility_clustering', 0)
            criticality_components.extend([min(volatility * 10, 1.0), clustering])

        if 'correlation_metrics' in critical_metrics:
            corr_metrics = critical_metrics['correlation_metrics']
            systemic_risk = corr_metrics.get('systemic_risk_measure', 0)
            criticality_components.append(systemic_risk)

        if 'information_metrics' in critical_metrics:
            info_metrics = critical_metrics['information_metrics']
            entropy = info_metrics.get('approximate_entropy', 1.0)
            # 低熵预示临界性
            criticality_components.append(1.0 - entropy)

        # 加权综合
        if criticality_components:
            weights = [0.3, 0.2, 0.2, 0.3]  # 可以调整权重
            weighted_criticality = np.average(
                criticality_components[:len(weights)],
                weights=weights[:len(criticality_components)]
            )
            return weighted_criticality

        return 0.0

    def _calculate_critical_exponents(self, market_data: pd.DataFrame, critical_metrics: Dict[str, Any]) -> Dict[str, float]:
        """计算临界指数"""
        exponents = {}

        if 'close' in market_data.columns:
            prices = market_data['close'].dropna()
            if len(prices) > self.window_size:
                # 相关长度指数
                exponents['correlation_length'] = self._estimate_correlation_length_exponent(prices)

                # 序参数指数
                exponents['order_parameter'] = self._estimate_order_parameter_exponent(prices)

                # 动力学指数
                exponents['dynamical'] = self._estimate_dynamical_exponent(prices)

        return exponents

    def _estimate_correlation_length_exponent(self, prices: pd.Series) -> float:
        """估计相关长度指数"""
        try:
            returns = prices.pct_change().dropna()
            if len(returns) < 50:
                return 0.0

            # 计算不同时间尺度的自相关
            lags = range(1, min(31, len(returns)//3))
            autocorrs = [returns.autocorr(lag=lag) for lag in lags]

            # 拟合幂律衰减
            if len(autocorrs) > 5:
                log_lags = np.log(lags)
                log_autocorrs = np.log(np.abs(autocorrs))

                # 线性回归
                slope, _ = np.polyfit(log_lags, log_autocorrs, 1)

                # 相关长度指数
                nu = -1.0 / slope if slope != 0 else 0.0

                return max(0.0, min(nu, 5.0))  # 合理范围限制

        except:
            pass

        return 0.0

    def _estimate_order_parameter_exponent(self, prices: pd.Series) -> float:
        """估计序参数指数"""
        try:
            # 使用价格变动的幅度作为序参数
            returns = prices.pct_change().dropna()
            if len(returns) < 50:
                return 0.0

            # 计算不同窗口大小的序参数
            window_sizes = [10, 20, 30, 40, 50]
            order_parameters = []

            for window_size in window_sizes:
                if len(returns) >= window_size:
                    # 窗口内的平均绝对收益
                    order_param = np.abs(returns.iloc[-window_size:]).mean()
                    order_parameters.append(order_param)

            if len(order_parameters) > 3:
                # 拟合幂律关系
                log_windows = np.log(window_sizes[:len(order_parameters)])
                log_order_params = np.log(order_parameters)

                slope, _ = np.polyfit(log_windows, log_order_params, 1)

                # 序参数指数
                beta = slope

                return max(-2.0, min(beta, 2.0))  # 合理范围限制

        except:
            pass

        return 0.0

    def _estimate_dynamical_exponent(self, prices: pd.Series) -> float:
        """估计动力学指数"""
        try:
            returns = prices.pct_change().dropna()
            if len(returns) < 100:
                return 0.0

            # 使用多重分形分析估计动力学指数
            q_values = [-2, -1, 0, 1, 2]
            tau_q = []

            for q in q_values:
                # 简化的多重分形谱计算
                if q == 0:
                    continue

                # 计算q阶矩
                absolute_returns = np.abs(returns)
                if len(absolute_returns) < 10:
                    continue

                # 不同时间尺度的q阶矩
                scales = [5, 10, 15, 20, 25]
                q_moments = []

                for scale in scales:
                    if len(absolute_returns) >= scale:
                        # 分割序列并计算q阶矩
                        segments = len(absolute_returns) // scale
                        if segments > 0:
                            moment_segments = []
                            for i in range(segments):
                                segment = absolute_returns.iloc[i*scale:(i+1)*scale]
                                if len(segment) > 0:
                                    segment_sum = np.sum(segment ** q)
                                    moment_segments.append(segment_sum)

                            if moment_segments:
                                q_moment = np.mean(moment_segments) ** (1.0/q)
                                q_moments.append(q_moment)

                if len(q_moments) > 3:
                    # 拟合标度关系
                    log_scales = np.log(scales[:len(q_moments)])
                    log_q_moments = np.log(q_moments)

                    slope, _ = np.polyfit(log_scales, log_q_moments, 1)
                    tau_q.append(slope)

            if len(tau_q) > 2:
                # 计算多重分形谱
                q_values_used = [q for q in q_values if q != 0][:len(tau_q)]
                if len(q_values_used) > 2:
                    # 估计动力学指数
                    tau_q = np.array(tau_q)
                    q_array = np.array(q_values_used)

                    # 线性回归估计
                    slope, _ = np.polyfit(q_array, tau_q, 1)
                    z = slope

                    # 动力学指数
                    if z != 0:
                        dynamical_exponent = 1.0 / z
                        return max(0.1, min(dynamical_exponent, 10.0))

        except:
            pass

        return 1.0

    def _assess_system_stability(self, critical_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """评估系统稳定性"""
        stability = {}

        # 综合稳定性评分
        stability_score = self._calculate_stability_score(critical_metrics)
        stability['overall_stability'] = stability_score

        # 稳定性分类
        if stability_score > 0.8:
            stability['stability_level'] = 'high_stability'
            stability['risk_level'] = 'low'
        elif stability_score > 0.6:
            stability['stability_level'] = 'moderate_stability'
            stability['risk_level'] = 'medium'
        elif stability_score > 0.4:
            stability['stability_level'] = 'low_stability'
            stability['risk_level'] = 'high'
        else:
            stability['stability_level'] = 'critical_instability'
            stability['risk_level'] = 'critical'

        # 稳定性因素分析
        stability['stability_factors'] = self._analyze_stability_factors(critical_metrics)

        # 恢复力评估
        stability['resilience_metrics'] = self._assess_resilience(critical_metrics)

        return stability

    def _calculate_stability_score(self, critical_metrics: Dict[str, Any]) -> float:
        """计算稳定性评分"""
        stability_components = []

        # 波动率稳定性
        if 'volatility_metrics' in critical_metrics:
            vol_metrics = critical_metrics['volatility_metrics']
            volatility = vol_metrics.get('volatility', 0)
            # 低波动率贡献稳定性
            stability_components.append(max(0, 1.0 - volatility * 10))

        # 相关性稳定性
        if 'correlation_metrics' in critical_metrics:
            corr_metrics = critical_metrics['correlation_metrics']
            systemic_risk = corr_metrics.get('systemic_risk_measure', 0)
            # 低系统性风险贡献稳定性
            stability_components.append(max(0, 1.0 - systemic_risk))

        # 复杂度稳定性
        if 'information_metrics' in critical_metrics:
            info_metrics = critical_metrics['information_metrics']
            entropy = info_metrics.get('approximate_entropy', 0.5)
            # 适度复杂度贡献稳定性
            stability_components.append(1.0 - abs(entropy - 0.7))

        # 临界接近度稳定性
        if 'critical_proximity' in critical_metrics:
            proximity = critical_metrics['critical_proximity']
            overall_criticality = proximity.get('overall_criticality', 0)
            # 低临界性贡献稳定性
            stability_components.append(max(0, 1.0 - overall_criticality))

        if stability_components:
            return np.mean(stability_components)

        return 0.5

    def _analyze_stability_factors(self, critical_metrics: Dict[str, Any]) -> Dict[str, float]:
        """分析稳定性因素"""
        factors = {}

        # 波动率贡献
        if 'volatility_metrics' in critical_metrics:
            vol_metrics = critical_metrics['volatility_metrics']
            factors['volatility_contribution'] = min(vol_metrics.get('volatility', 0) * 10, 1.0)

        # 相关性贡献
        if 'correlation_metrics' in critical_metrics:
            corr_metrics = critical_metrics['correlation_metrics']
            factors['correlation_contribution'] = min(corr_metrics.get('systemic_risk_measure', 0), 1.0)

        # 复杂度贡献
        if 'information_metrics' in critical_metrics:
            info_metrics = critical_metrics['information_metrics']
            entropy = info_metrics.get('approximate_entropy', 0.5)
            factors['complexity_contribution'] = abs(entropy - 0.7)

        # 临界性贡献
        if 'critical_proximity' in critical_metrics:
            proximity = critical_metrics['critical_proximity']
            factors['criticality_contribution'] = proximity.get('overall_criticality', 0)

        return factors

    def _assess_resilience(self, critical_metrics: Dict[str, Any]) -> Dict[str, float]:
        """评估恢复力"""
        resilience = {}

        # 基于分形维数的恢复力
        if 'fractal_metrics' in critical_metrics:
            fractal_metrics = critical_metrics['fractal_metrics']
            fractal_dim = fractal_metrics.get('fractal_dimension', 1.0)
            # 适中的分形维数表示较好的恢复力
            resilience['fractal_resilience'] = 1.0 - abs(fractal_dim - 1.5) / 1.5

        # 基于赫斯特指数的恢复力
        if 'fractal_metrics' in critical_metrics:
            fractal_metrics = critical_metrics['fractal_metrics']
            hurst = fractal_metrics.get('hurst_exponent', 0.5)
            # 接近0.5的赫斯特指数表示较好的恢复力
            resilience['hurst_resilience'] = 1.0 - abs(hurst - 0.5) * 2

        # 基于网络指标的恢复力
        if 'network_metrics' in critical_metrics:
            network_metrics = critical_metrics['network_metrics']
            clustering = network_metrics.get('clustering_coefficient', 0)
            # 高聚类系数表示较好的恢复力
            resilience['network_resilience'] = clustering

        # 综合恢复力
        resilience_values = [v for v in resilience.values() if isinstance(v, (int, float))]
        resilience['overall_resilience'] = np.mean(resilience_values) if resilience_values else 0.5

        return resilience

    def _predict_transition_risk(self, critical_metrics: Dict[str, Any], early_warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """预测相变风险"""
        risk_prediction = {}

        # 基于临界指标的风险评分
        criticality_score = self._calculate_composite_criticality(critical_metrics)

        # 基于预警信号的风险评分
        warning_score = 0
        for warning in early_warnings:
            if warning['severity'] == 'high':
                warning_score += 0.4
            elif warning['severity'] == 'medium':
                warning_score += 0.2

        warning_score = min(warning_score, 1.0)

        # 综合风险评分
        risk_prediction['overall_risk_score'] = (criticality_score + warning_score) / 2

        # 风险等级
        if risk_prediction['overall_risk_score'] > 0.8:
            risk_prediction['risk_level'] = 'critical'
            risk_prediction['probability'] = 0.9
        elif risk_prediction['overall_risk_score'] > 0.6:
            risk_prediction['risk_level'] = 'high'
            risk_prediction['probability'] = 0.7
        elif risk_prediction['overall_risk_score'] > 0.4:
            risk_prediction['risk_level'] = 'medium'
            risk_prediction['probability'] = 0.4
        else:
            risk_prediction['risk_level'] = 'low'
            risk_prediction['probability'] = 0.1

        # 时间窗口预测
        risk_prediction['expected_timeframe'] = self._predict_transition_timeframe(criticality_score, warning_score)

        # 风险因素分析
        risk_prediction['risk_factors'] = self._analyze_risk_factors(critical_metrics, early_warnings)

        return risk_prediction

    def _predict_transition_timeframe(self, criticality_score: float, warning_score: float) -> str:
        """预测相变时间框架"""
        combined_score = (criticality_score + warning_score) / 2

        if combined_score > 0.8:
            return "immediate (0-5 days)"
        elif combined_score > 0.6:
            return "short_term (5-15 days)"
        elif combined_score > 0.4:
            return "medium_term (15-30 days)"
        else:
            return "long_term (30+ days)"

    def _analyze_risk_factors(self, critical_metrics: Dict[str, Any], early_warnings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """分析风险因素"""
        risk_factors = []

        # 临界接近度风险
        if 'critical_proximity' in critical_metrics:
            proximity = critical_metrics['critical_proximity']
            for factor, value in proximity.items():
                if isinstance(value, (int, float)) and value > 0.5:
                    risk_factors.append({
                        'factor': f'critical_proximity_{factor}',
                        'contribution': value,
                        'type': 'criticality'
                    })

        # 预警信号风险
        for warning in early_warnings:
            risk_factors.append({
                'factor': warning['type'],
                'contribution': warning['value'] / warning['threshold'],
                'type': 'early_warning'
            })

        # 按贡献度排序
        risk_factors.sort(key=lambda x: x['contribution'], reverse=True)

        return risk_factors[:5]  # 返回前5个主要风险因素

    def _generate_warning_report(self,
                              critical_metrics: Dict[str, Any],
                              early_warnings: List[Dict[str, Any]],
                              phase_transitions: List[Dict[str, Any]],
                              risk_prediction: Dict[str, Any]) -> Dict[str, Any]:
        """生成预警报告"""
        report = {
            'warning_level': self.warning_level,
            'system_state': self.system_state,
            'critical_summary': self._generate_critical_summary(critical_metrics),
            'warning_summary': self._generate_warning_summary(early_warnings),
            'transition_summary': self._generate_transition_summary(phase_transitions),
            'risk_summary': self._generate_risk_summary(risk_prediction),
            'recommendations': self._generate_recommendations(),
            'confidence_assessment': self._assess_prediction_confidence(critical_metrics, early_warnings)
        }

        return report

    def _generate_critical_summary(self, critical_metrics: Dict[str, Any]) -> str:
        """生成临界状态摘要"""
        summary_parts = []

        # 临界接近度
        if 'critical_proximity' in critical_metrics:
            proximity = critical_metrics['critical_proximity']
            overall_criticality = proximity.get('overall_criticality', 0)

            if overall_criticality > 0.8:
                summary_parts.append("系统处于高度临界状态")
            elif overall_criticality > 0.6:
                summary_parts.append("系统临界性显著增加")
            elif overall_criticality > 0.4:
                summary_parts.append("系统临界性中等")
            else:
                summary_parts.append("系统临界性较低")

        # 波动率状态
        if 'volatility_metrics' in critical_metrics:
            vol_metrics = critical_metrics['volatility_metrics']
            volatility = vol_metrics.get('volatility', 0)

            if volatility > 0.05:
                summary_parts.append("市场波动率异常高")
            elif volatility > 0.02:
                summary_parts.append("市场波动率较高")
            else:
                summary_parts.append("市场波动率正常")

        return "；".join(summary_parts)

    def _generate_warning_summary(self, early_warnings: List[Dict[str, Any]]) -> str:
        """生成预警摘要"""
        if not early_warnings:
            return "未检测到明显预警信号"

        warning_types = [w['type'] for w in early_warnings]
        severe_warnings = [w for w in early_warnings if w['severity'] == 'high']

        if severe_warnings:
            return f"检测到{len(severe_warnings)}个高级别预警信号：{', '.join([w['type'] for w in severe_warnings])}"
        else:
            return f"检测到{len(early_warnings)}个预警信号：{', '.join(warning_types)}"

    def _generate_transition_summary(self, phase_transitions: List[Dict[str, Any]]) -> str:
        """生成相变摘要"""
        if not phase_transitions:
            return "当前未检测到明显相变"

        transition = phase_transitions[0]  # 取最主要的相变
        return f"检测到{transition['type']}，严重程度：{transition['severity']}"

    def _generate_risk_summary(self, risk_prediction: Dict[str, Any]) -> str:
        """生成风险摘要"""
        risk_level = risk_prediction.get('risk_level', 'unknown')
        probability = risk_prediction.get('probability', 0)
        timeframe = risk_prediction.get('expected_timeframe', 'unknown')

        return f"相变风险等级：{risk_level}，发生概率：{probability:.1%}，预期时间框架：{timeframe}"

    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []

        if self.warning_level >= 3:  # 临界状态
            recommendations.extend([
                "立即采取保护性措施",
                "考虑减少高风险资产敞口",
                "增加现金和防御性资产配置",
                "设置严格的止损点",
                "密切监控市场动态"
            ])
        elif self.warning_level >= 2:  # 警告状态
            recommendations.extend([
                "适度降低风险敞口",
                "增加投资组合多样性",
                "设置预警和止损机制",
                "关注市场情绪变化"
            ])
        elif self.warning_level >= 1:  # 正常状态
            recommendations.extend([
                "维持正常投资策略",
                "保持适度风险敞口",
                "继续常规监控"
            ])

        return recommendations

    def _assess_prediction_confidence(self, critical_metrics: Dict[str, Any], early_warnings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """评估预测置信度"""
        confidence_assessment = {}

        # 数据质量评估
        data_quality = self._assess_data_quality(critical_metrics)
        confidence_assessment['data_quality'] = data_quality

        # 信号强度评估
        signal_strength = self._assess_signal_strength(critical_metrics, early_warnings)
        confidence_assessment['signal_strength'] = signal_strength

        # 模型一致性评估
        model_consistency = self._assess_model_consistency(critical_metrics)
        confidence_assessment['model_consistency'] = model_consistency

        # 综合置信度
        confidence_factors = [data_quality, signal_strength, model_consistency]
        confidence_assessment['overall_confidence'] = np.mean(confidence_factors)

        # 置信度等级
        if confidence_assessment['overall_confidence'] > 0.8:
            confidence_assessment['confidence_level'] = 'high'
        elif confidence_assessment['overall_confidence'] > 0.6:
            confidence_assessment['confidence_level'] = 'medium'
        else:
            confidence_assessment['confidence_level'] = 'low'

        return confidence_assessment

    def _assess_data_quality(self, critical_metrics: Dict[str, Any]) -> float:
        """评估数据质量"""
        # 基于指标完整性评估数据质量
        total_metrics = 0
        available_metrics = 0

        metric_categories = [
            'volatility_metrics', 'correlation_metrics', 'fractal_metrics',
            'information_metrics', 'critical_proximity', 'order_statistics'
        ]

        for category in metric_categories:
            total_metrics += 1
            if category in critical_metrics and critical_metrics[category]:
                available_metrics += 1

        return available_metrics / total_metrics if total_metrics > 0 else 0.0

    def _assess_signal_strength(self, critical_metrics: Dict[str, Any], early_warnings: List[Dict[str, Any]]) -> float:
        """评估信号强度"""
        # 基于预警信号强度评估
        if not early_warnings:
            return 0.0

        signal_strength = 0
        for warning in early_warnings:
            if warning['severity'] == 'high':
                signal_strength += 0.5
            elif warning['severity'] == 'medium':
                signal_strength += 0.3

        return min(signal_strength, 1.0)

    def _assess_model_consistency(self, critical_metrics: Dict[str, Any]) -> float:
        """评估模型一致性"""
        # 基于不同指标之间的一致性评估
        consistency_score = 0.0

        # 检查不同指标组之间的逻辑一致性
        volatility_high = False
        correlation_high = False
        criticality_high = False

        if 'volatility_metrics' in critical_metrics:
            vol_metrics = critical_metrics['volatility_metrics']
            if vol_metrics.get('volatility', 0) > 0.03:
                volatility_high = True

        if 'correlation_metrics' in critical_metrics:
            corr_metrics = critical_metrics['correlation_metrics']
            if corr_metrics.get('systemic_risk_measure', 0) > 0.7:
                correlation_high = True

        if 'critical_proximity' in critical_metrics:
            proximity = critical_metrics['critical_proximity']
            if proximity.get('overall_criticality', 0) > 0.6:
                criticality_high = True

        # 评估一致性
        aligned_signals = sum([volatility_high, correlation_high, criticality_high])
        total_signals = 3

        if aligned_signals == total_signals or aligned_signals == 0:
            consistency_score = 1.0  # 完全一致
        elif aligned_signals == 2:
            consistency_score = 0.7  # 大部分一致
        elif aligned_signals == 1:
            consistency_score = 0.3  # 部分一致
        else:
            consistency_score = 0.0  # 不一致

        return consistency_score

    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析总结"""
        return {
            'window_size': self.window_size,
            'detection_threshold': self.detection_threshold,
            'early_warning_threshold': self.early_warning_threshold,
            'sensitivity': self.sensitivity,
            'current_warning_level': self.warning_level,
            'current_system_state': self.system_state,
            'total_analysis_sessions': len(self.phase_transition_history),
            'phase_transition_detected': any(pt['type'] == 'critical_transition' for pt in self.phase_transition_history),
            'model_info': self.get_model_info()
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'Critical Phase Transition Analyzer',
            'torch_available': TORCH_AVAILABLE,
            'scipy_available': SCIPY_AVAILABLE,
            'networkx_available': NETWORKX_AVAILABLE,
            'window_size': self.window_size,
            'detection_threshold': self.detection_threshold,
            'early_warning_threshold': self.early_warning_threshold,
            'sensitivity': self.sensitivity,
            'phase_detection_model_available': self.phase_detection_model is not None,
            'early_warning_model_available': self.early_warning_model is not None
        }


class CriticalPhaseDetector(nn.Module):
    """临界相变检测模型"""
    def __init__(self, input_dim: int, hidden_dims: List[int], sensitivity: float = 0.8):
        super().__init__()
        self.input_dim = input_dim
        self.sensitivity = sensitivity

        # 神经网络层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # 输出层：临界概率
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class EarlyWarningPredictor(nn.Module):
    """早期预警预测模型"""
    def __init__(self, window_size: int, prediction_horizon: int):
        super().__init__()
        self.window_size = window_size
        self.prediction_horizon = prediction_horizon

        # LSTM层用于时序预测
        self.lstm = nn.LSTM(input_size=10, hidden_size=32, num_layers=2, batch_first=True)
        self.fc = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, window_size, features)
        lstm_out, _ = self.lstm(x)
        prediction = self.sigmoid(self.fc(lstm_out[:, -1, :]))
        return prediction


# 便捷函数
def create_critical_transition_analyzer(window_size: int = 100) -> CriticalTransitionAnalyzer:
    """创建临界相变分析器实例"""
    return CriticalTransitionAnalyzer(window_size=window_size)

def quick_phase_analysis(market_data: pd.DataFrame) -> Dict[str, Any]:
    """快速相变分析"""
    analyzer = create_critical_transition_analyzer()
    return analyzer.analyze_market_phase_transition(market_data)

def get_critical_warning_level(market_data: pd.DataFrame) -> str:
    """获取临界预警等级"""
    results = quick_phase_analysis(market_data)
    warning_level = results.get('warning_level', 0)

    level_descriptions = {
        0: 'normal',
        1: 'normal',
        2: 'warning',
        3: 'critical'
    }

    return level_descriptions.get(warning_level, 'unknown')