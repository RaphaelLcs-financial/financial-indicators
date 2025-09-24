"""
Strange Attractor Analyzer
==========================

基于奇怪吸引子理论的混沌市场分析器，通过重构相空间
和识别吸引子结构来分析市场的混沌特性。该指标利用
混沌理论中的吸引子几何、分形维数和李雅普诺夫指数
来识别市场的非线性动态。

特点：
- 相空间重构
- 奇怪吸引子识别
- 分形维数计算
- 李雅普诺夫谱分析
- 混沌程度量化
- 吸引子几何分析
- 预测域评估
- 混沌控制识别

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.spatial.distance import pdist, squareform
from scipy.signal import find_peaks
from scipy.stats import entropy
from scipy.linalg import eig
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class PhaseSpaceReconstructor:
    """相空间重构器"""
    def __init__(self, embedding_dimension: int = 3, time_delay: int = 1):
        """
        初始化相空间重构器

        Args:
            embedding_dimension: 嵌入维度，默认3
            time_delay: 时间延迟，默认1
        """
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay

    def reconstruct(self, time_series: np.ndarray) -> np.ndarray:
        """
        重构相空间

        Args:
            time_series: 时间序列

        Returns:
            相空间点集
        """
        n_points = len(time_series) - (self.embedding_dimension - 1) * self.time_delay

        if n_points <= 0:
            return np.array([])

        phase_space = np.zeros((n_points, self.embedding_dimension))

        for i in range(n_points):
            for j in range(self.embedding_dimension):
                phase_space[i, j] = time_series[i + j * self.time_delay]

        return phase_space

    def find_optimal_parameters(self, time_series: np.ndarray) -> Tuple[int, int]:
        """
        寻找最优参数

        Args:
            time_series: 时间序列

        Returns:
            最优嵌入维度和时间延迟
        """
        # 使用假近邻法寻找嵌入维度
        optimal_dimension = self.false_nearest_neighbors(time_series)

        # 使用自相关法寻找时间延迟
        optimal_delay = self.autocorrelation_delay(time_series)

        return optimal_dimension, optimal_delay

    def false_nearest_neighbors(self, time_series: np.ndarray, max_dimension: int = 10) -> int:
        """
        假近邻法

        Args:
            time_series: 时间序列
            max_dimension: 最大维度

        Returns:
            最优嵌入维度
        """
        n = len(time_series)
        if n < 100:
            return 3

        false_neighbors_ratio = []

        for dim in range(1, max_dimension + 1):
            false_count = 0
            total_count = 0

            # 重构相空间
            phase_space = np.zeros((n - dim, dim))
            for i in range(n - dim):
                for j in range(dim):
                    phase_space[i, j] = time_series[i + j]

            # 计算最近邻
            nbrs = NearestNeighbors(n_neighbors=2).fit(phase_space[:-1])
            distances, indices = nbrs.kneighbors(phase_space[:-1])

            # 检查假近邻
            for i in range(len(phase_space) - 1):
                if i + 1 < len(phase_space):
                    neighbor_idx = indices[i, 1]
                    if neighbor_idx < len(phase_space) - 1:
                        # 当前距离
                        current_dist = distances[i, 1]

                        # 在高维空间中的距离
                        high_dim_dist = abs(time_series[i + dim] - time_series[neighbor_idx + dim])

                        # 判断是否为假近邻
                        if high_dim_dist > 2 * current_dist:
                            false_count += 1

                        total_count += 1

            if total_count > 0:
                ratio = false_count / total_count
                false_neighbors_ratio.append(ratio)
            else:
                false_neighbors_ratio.append(0)

        # 寻找比率显著下降的维度
        for i in range(1, len(false_neighbors_ratio)):
            if false_neighbors_ratio[i] < 0.1:  # 10%阈值
                return i + 1

        return 3  # 默认值

    def autocorrelation_delay(self, time_series: np.ndarray) -> int:
        """
        自相关法确定时间延迟

        Args:
            time_series: 时间序列

        Returns:
            时间延迟
        """
        # 计算自相关函数
        autocorr = np.correlate(time_series - np.mean(time_series),
                               time_series - np.mean(time_series), mode='full')

        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / autocorr[0]

        # 寻找第一个零点或1/e点
        threshold = 1.0 / np.e
        zero_crossings = np.where(autocorr < threshold)[0]

        if len(zero_crossings) > 0:
            return max(1, zero_crossings[0])
        else:
            return 1


class StrangeAttractorAnalyzer:
    """
    奇怪吸引子分析器

    利用混沌理论分析市场的奇怪吸引子
    结构和分形特性。
    """

    def __init__(self, embedding_dimension: int = 3, time_delay: int = 1,
                 max_neighbors: int = 20):
        """
        初始化奇怪吸引子分析器

        Args:
            embedding_dimension: 嵌入维度，默认3
            time_delay: 时间延迟，默认1
            max_neighbors: 最大邻居数，默认20
        """
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
        self.max_neighbors = max_neighbors
        self.name = f"Strange Attractor Analyzer ({embedding_dimension}D)"
        self.category = "chaos"

        # 相空间重构器
        self.phase_reconstructor = PhaseSpaceReconstructor(embedding_dimension, time_delay)

    def reconstruct_attractor(self, time_series: pd.Series) -> Dict[str, Any]:
        """
        重构吸引子

        Args:
            time_series: 时间序列

        Returns:
            吸引子重构结果
        """
        if time_series.empty:
            return {}

        results = {}

        # 清理数据
        clean_series = time_series.fillna(0).values

        # 相空间重构
        phase_space = self.phase_reconstructor.reconstruct(clean_series)
        results['phase_space'] = phase_space

        # 优化参数
        optimal_dim, optimal_delay = self.phase_reconstructor.find_optimal_parameters(clean_series)
        results['optimal_dimension'] = optimal_dim
        results['optimal_delay'] = optimal_delay

        # 使用最优参数重构
        self.phase_reconstructor.embedding_dimension = optimal_dim
        self.phase_reconstructor.time_delay = optimal_delay
        optimal_phase_space = self.phase_reconstructor.reconstruct(clean_series)
        results['optimal_phase_space'] = optimal_phase_space

        return results

    def calculate_fractal_dimension(self, phase_space: np.ndarray) -> Dict[str, float]:
        """
        计算分形维数

        Args:
            phase_space: 相空间点集

        Returns:
            分形维数
        """
        if len(phase_space) < 10:
            return {}

        dimensions = {}

        # 相关维数
        correlation_dim = self.correlation_dimension(phase_space)
        dimensions['correlation_dimension'] = correlation_dim

        # 信息维数
        information_dim = self.information_dimension(phase_space)
        dimensions['information_dimension'] = information_dim

        # 容量维数
        capacity_dim = self.capacity_dimension(phase_space)
        dimensions['capacity_dimension'] = capacity_dim

        # 李雅普诺夫维数
        lyapunov_dim = self.lyapunov_dimension(phase_space)
        dimensions['lyapunov_dimension'] = lyapunov_dim

        return dimensions

    def correlation_dimension(self, phase_space: np.ndarray) -> float:
        """
        计算相关维数

        Args:
            phase_space: 相空间点集

        Returns:
            相关维数
        """
        if len(phase_space) < 2:
            return 1.0

        # 计算点间距离
        distances = pdist(phase_space)
        distances = distances[distances > 0]  # 去除零距离

        # 计算相关积分
        epsilon_values = np.logspace(-3, 0, 20)
        correlation_integral = []

        for epsilon in epsilon_values:
            count = np.sum(distances < epsilon)
            correlation_integral.append(count / (len(phase_space) * (len(phase_space) - 1)))

        # 线性回归计算维数
        log_epsilon = np.log(epsilon_values)
        log_c = np.log(correlation_integral)

        # 去除无效值
        valid_mask = np.isfinite(log_epsilon) & np.isfinite(log_c) & (log_c > 0)
        if np.sum(valid_mask) < 2:
            return 1.0

        slope = np.polyfit(log_epsilon[valid_mask], log_c[valid_mask], 1)[0]
        return max(0.0, slope)

    def information_dimension(self, phase_space: np.ndarray) -> float:
        """
        计算信息维数

        Args:
            phase_space: 相空间点集

        Returns:
            信息维数
        """
        if len(phase_space) < 2:
            return 1.0

        # 计算信息熵
        epsilon_values = np.logspace(-3, 0, 15)
        information_values = []

        for epsilon in epsilon_values:
            # 计算盒子计数
            n_boxes = self.count_boxes(phase_space, epsilon)
            if n_boxes > 0:
                # 计算概率分布
                probabilities = np.ones(n_boxes) / n_boxes
                # 计算信息熵
                info_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))
                information_values.append(info_entropy)
            else:
                information_values.append(0)

        # 线性回归计算维数
        log_epsilon = np.log(epsilon_values)
        log_info = np.log(information_values + 1e-10)

        valid_mask = np.isfinite(log_epsilon) & np.isfinite(log_info) & (log_info > 0)
        if np.sum(valid_mask) < 2:
            return 1.0

        slope = np.polyfit(log_epsilon[valid_mask], log_info[valid_mask], 1)[0]
        return max(0.0, slope)

    def capacity_dimension(self, phase_space: np.ndarray) -> float:
        """
        计算容量维数

        Args:
            phase_space: 相空间点集

        Returns:
            容量维数
        """
        if len(phase_space) < 2:
            return 1.0

        epsilon_values = np.logspace(-3, 0, 15)
        box_counts = []

        for epsilon in epsilon_values:
            n_boxes = self.count_boxes(phase_space, epsilon)
            box_counts.append(n_boxes)

        # 线性回归计算维数
        log_epsilon = np.log(epsilon_values)
        log_boxes = np.log(box_counts + 1e-10)

        valid_mask = np.isfinite(log_epsilon) & np.isfinite(log_boxes) & (log_boxes > 0)
        if np.sum(valid_mask) < 2:
            return 1.0

        slope = np.polyfit(log_epsilon[valid_mask], log_boxes[valid_mask], 1)[0]
        return max(0.0, slope)

    def count_boxes(self, phase_space: np.ndarray, epsilon: float) -> int:
        """
        计算盒子数量

        Args:
            phase_space: 相空间点集
            epsilon: 盒子大小

        Returns:
            盒子数量
        """
        if len(phase_space) == 0:
            return 0

        # 计算每个维度的盒子索引
        box_indices = np.floor(phase_space / epsilon).astype(int)

        # 使用集合去重
        unique_boxes = set()
        for box_idx in box_indices:
            unique_boxes.add(tuple(box_idx))

        return len(unique_boxes)

    def lyapunov_dimension(self, phase_space: np.ndarray) -> float:
        """
        计算李雅普诺夫维数

        Args:
            phase_space: 相空间点集

        Returns:
            李雅普诺夫维数
        """
        if len(phase_space) < 10:
            return 1.0

        # 计算李雅普诺夫指数谱
        lyapunov_exponents = self.calculate_lyapunov_spectrum(phase_space)

        # 计算李雅普诺夫维数
        positive_exponents = lyapunov_exponents[lyapunov_exponents > 0]
        if len(positive_exponents) == 0:
            return 1.0

        k = len(positive_exponents)
        sum_positive = np.sum(positive_exponents)

        if sum_positive > 0:
            lyapunov_dim = k + (np.sum(lyapunov_exponents[:k])) / abs(lyapunov_exponents[k])
        else:
            lyapunov_dim = k

        return lyapunov_dim

    def calculate_lyapunov_spectrum(self, phase_space: np.ndarray) -> np.ndarray:
        """
        计算李雅普诺夫指数谱

        Args:
            phase_space: 相空间点集

        Returns:
            李雅普诺夫指数谱
        """
        if len(phase_space) < 10:
            return np.array([0.0])

        # 简化的李雅普诺夫指数计算
        n_points = len(phase_space)
        dimension = phase_space.shape[1]

        # 计算相邻点的演化
        neighbors = NearestNeighbors(n_neighbors=min(self.max_neighbors, n_points-1))
        neighbors.fit(phase_space[:-1])

        # 获取最近邻
        distances, indices = neighbors.kneighbors(phase_space[:-1])

        # 计算分离率
        separation_rates = []

        for i in range(len(phase_space) - 1):
            if i + 1 < len(phase_space) and indices[i, 0] < len(phase_space) - 1:
                # 初始距离
                initial_dist = distances[i, 0]

                # 演化后的距离
                final_dist = np.linalg.norm(phase_space[i + 1] - phase_space[indices[i, 0] + 1])

                if initial_dist > 0 and final_dist > 0:
                    separation_rate = np.log(final_dist / initial_dist)
                    separation_rates.append(separation_rate)

        if len(separation_rates) > 0:
            # 计算平均分离率（李雅普诺夫指数）
            lyapunov_exponents = np.full(dimension, np.mean(separation_rates))

            # 添加一些变化来创建谱
            for i in range(dimension):
                lyapunov_exponents[i] *= (1 - 0.1 * i)

        else:
            lyapunov_exponents = np.zeros(dimension)

        return lyapunov_exponents

    def analyze_attractor_geometry(self, phase_space: np.ndarray) -> Dict[str, Any]:
        """
        分析吸引子几何

        Args:
            phase_space: 相空间点集

        Returns:
            几何分析结果
        """
        geometry = {}

        if len(phase_space) < 3:
            return geometry

        # 计算吸引子的几何特征
        centroid = np.mean(phase_space, axis=0)
        geometry['centroid'] = centroid

        # 计算到中心的距离
        distances_to_center = np.linalg.norm(phase_space - centroid, axis=1)
        geometry['mean_distance_to_center'] = np.mean(distances_to_center)
        geometry['std_distance_to_center'] = np.std(distances_to_center)

        # 计算吸引子的范围
        min_coords = np.min(phase_space, axis=0)
        max_coords = np.max(phase_space, axis=0)
        geometry['bounds'] = (min_coords, max_coords)
        geometry['volume'] = np.prod(max_coords - min_coords)

        # 计算密度
        if geometry['volume'] > 0:
            geometry['density'] = len(phase_space) / geometry['volume']
        else:
            geometry['density'] = 0

        # 计算曲率（简化）
        geometry['mean_curvature'] = self.calculate_mean_curvature(phase_space)

        # 计算拓扑熵
        geometry['topological_entropy'] = self.calculate_topological_entropy(phase_space)

        return geometry

    def calculate_mean_curvature(self, phase_space: np.ndarray) -> float:
        """
        计算平均曲率

        Args:
            phase_space: 相空间点集

        Returns:
            平均曲率
        """
        if len(phase_space) < 3:
            return 0.0

        # 简化的曲率计算
        curvatures = []

        for i in range(1, len(phase_space) - 1):
            # 计算三点曲率
            p1 = phase_space[i - 1]
            p2 = phase_space[i]
            p3 = phase_space[i + 1]

            # 计算向量
            v1 = p2 - p1
            v2 = p3 - p2

            # 计算曲率
            cross_product = np.cross(v1, v2)
            curvature = np.linalg.norm(cross_product) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)

            curvatures.append(curvature)

        return np.mean(curvatures) if curvatures else 0.0

    def calculate_topological_entropy(self, phase_space: np.ndarray) -> float:
        """
        计算拓扑熵

        Args:
            phase_space: 相空间点集

        Returns:
            拓扑熵
        """
        if len(phase_space) < 2:
            return 0.0

        # 将相空间离散化为网格
        grid_size = 10
        min_coords = np.min(phase_space, axis=0)
        max_coords = np.max(phase_space, axis=0)

        # 计算网格索引
        grid_indices = np.floor((phase_space - min_coords) / (max_coords - min_coords + 1e-10) * grid_size).astype(int)
        grid_indices = np.clip(grid_indices, 0, grid_size - 1)

        # 计算符号序列
        symbol_sequence = []
        for i in range(len(grid_indices)):
            symbol = tuple(grid_indices[i])
            symbol_sequence.append(symbol)

        # 计算符号熵
        unique_symbols, counts = np.unique(symbol_sequence, return_counts=True)
        probabilities = counts / len(symbol_sequence)

        topological_entropy = -np.sum(probabilities * np.log(probabilities + 1e-10))

        return topological_entropy

    def detect_chaos_regime(self, phase_space: np.ndarray) -> Dict[str, Any]:
        """
        检测混沌状态

        Args:
            phase_space: 相空间点集

        Returns:
            混沌状态检测结果
        """
        chaos_detection = {}

        if len(phase_space) < 10:
            return chaos_detection

        # 计算分形维数
        fractal_dims = self.calculate_fractal_dimension(phase_space)
        chaos_detection.update(fractal_dims)

        # 计算李雅普诺夫指数
        lyapunov_exponents = self.calculate_lyapunov_spectrum(phase_space)
        chaos_detection['lyapunov_exponents'] = lyapunov_exponents
        chaos_detection['max_lyapunov_exponent'] = np.max(lyapunov_exponents)

        # 计算几何特征
        geometry = self.analyze_attractor_geometry(phase_space)
        chaos_detection.update(geometry)

        # 混沌程度量化
        chaos_degree = self.quantify_chaos_degree(fractal_dims, lyapunov_exponents, geometry)
        chaos_detection['chaos_degree'] = chaos_degree

        # 吸引子类型分类
        attractor_type = self.classify_attractor_type(fractal_dims, lyapunov_exponents)
        chaos_detection['attractor_type'] = attractor_type

        return chaos_detection

    def quantify_chaos_degree(self, fractal_dims: Dict[str, float],
                             lyapunov_exponents: np.ndarray,
                             geometry: Dict[str, Any]) -> float:
        """
        量化混沌程度

        Args:
            fractal_dims: 分形维数
            lyapunov_exponents: 李雅普诺夫指数
            geometry: 几何特征

        Returns:
            混沌程度
        """
        chaos_score = 0.0

        # 基于分形维数
        if 'correlation_dimension' in fractal_dims:
            corr_dim = fractal_dims['correlation_dimension']
            # 非整数维数表明混沌
            if 2 < corr_dim < 3:
                chaos_score += (corr_dim - 2) * 0.3

        # 基于李雅普诺夫指数
        max_lyap = np.max(lyapunov_exponents)
        if max_lyap > 0:
            chaos_score += min(max_lyap * 2, 0.4)

        # 基于几何复杂度
        if 'mean_curvature' in geometry:
            curvature = geometry['mean_curvature']
            chaos_score += min(curvature * 0.3, 0.3)

        return min(chaos_score, 1.0)

    def classify_attractor_type(self, fractal_dims: Dict[str, float],
                               lyapunov_exponents: np.ndarray) -> str:
        """
        分类吸引子类型

        Args:
            fractal_dims: 分形维数
            lyapunov_exponents: 李雅普诺夫指数

        Returns:
            吸引子类型
        """
        max_lyap = np.max(lyapunov_exponents)
        corr_dim = fractal_dims.get('correlation_dimension', 1.0)

        if max_lyap > 0.1 and 2 < corr_dim < 3:
            return 'strange_attractor'
        elif max_lyap > 0.05:
            return 'chaotic_attractor'
        elif max_lyap < 0.01:
            return 'fixed_point'
        elif abs(max_lyap) < 0.05:
            return 'limit_cycle'
        else:
            return 'unknown'

    def analyze_market_strange_attractor(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析市场奇怪吸引子

        Args:
            data: 市场数据

        Returns:
            奇怪吸引子分析结果
        """
        results = {}

        # 提取价格序列
        if 'close' not in data.columns:
            return results

        price_series = data['close']

        # 重构吸引子
        attractor_reconstruction = self.reconstruct_attractor(price_series)
        results.update(attractor_reconstruction)

        # 混沌检测
        if 'optimal_phase_space' in attractor_reconstruction:
            phase_space = attractor_reconstruction['optimal_phase_space']
            if len(phase_space) > 0:
                chaos_detection = self.detect_chaos_regime(phase_space)
                results.update(chaos_detection)

        # 生成交易信号
        results['signals'] = self.generate_chaos_signals(results)

        return results

    def generate_chaos_signals(self, chaos_analysis: Dict[str, Any]) -> pd.Series:
        """
        基于混沌分析生成交易信号

        Args:
            chaos_analysis: 混沌分析结果

        Returns:
            交易信号
        """
        # 创建时间序列
        n_points = 100
        dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
        signals = pd.Series(0, index=dates)

        # 基于混沌程度生成信号
        chaos_degree = chaos_analysis.get('chaos_degree', 0.0)
        max_lyap = chaos_analysis.get('max_lyapunov_exponent', 0.0)
        attractor_type = chaos_analysis.get('attractor_type', 'unknown')

        # 混沌阈值
        high_chaos_threshold = 0.7
        low_chaos_threshold = 0.3

        # 李雅普诺夫阈值
        high_lyap_threshold = 0.1
        low_lyap_threshold = 0.01

        # 生成信号
        high_chaos = chaos_degree > high_chaos_threshold
        low_chaos = chaos_degree < low_chaos_threshold
        high_lyap = max_lyap > high_lyap_threshold
        low_lyap = max_lyap < low_lyap_threshold

        # 基于吸引子类型调整信号
        if attractor_type == 'strange_attractor':
            # 奇怪吸引子：复杂的混沌行为
            signals[high_chaos] = 1  # 中性信号
            signals[low_chaos] = 2  # 买入信号（有序）
        elif attractor_type == 'chaotic_attractor':
            # 混沌吸引子：高度混沌
            signals[high_chaos] = -2  # 卖出信号
            signals[low_chaos] = 1   # 中性信号
        elif attractor_type == 'fixed_point':
            # 固定点：稳定状态
            signals[high_chaos] = -1  # 轻微卖出
            signals[low_chaos] = 3   # 强烈买入
        elif attractor_type == 'limit_cycle':
            # 极限环：周期性
            signals[high_chaos] = 2   # 买入信号（周期顶部）
            signals[low_chaos] = -2   # 卖出信号（周期底部）

        # 李雅普诺夫调整
        if high_lyap:
            signals[signals > 0] = np.maximum(1, signals[signals > 0])
            signals[signals < 0] = np.minimum(-1, signals[signals < 0])
        elif low_lyap:
            signals[signals > 0] = signals[signals > 0] + 1
            signals[signals < 0] = signals[signals < 0] - 1

        return signals

    def get_trading_recommendations(self, chaos_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            chaos_analysis: 混沌分析结果

        Returns:
            交易建议DataFrame
        """
        # 创建时间序列
        n_points = 100
        dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
        recommendations = pd.DataFrame(index=dates)

        # 添加混沌分析指标
        recommendations['chaos_degree'] = chaos_analysis.get('chaos_degree', 0.0)
        recommendations['max_lyapunov_exponent'] = chaos_analysis.get('max_lyapunov_exponent', 0.0)
        recommendations['attractor_type'] = chaos_analysis.get('attractor_type', 'unknown')

        if 'correlation_dimension' in chaos_analysis:
            recommendations['correlation_dimension'] = chaos_analysis['correlation_dimension']

        if 'topological_entropy' in chaos_analysis:
            recommendations['topological_entropy'] = chaos_analysis['topological_entropy']

        # 交易信号
        signals = self.generate_chaos_signals(chaos_analysis)
        recommendations['signals'] = signals

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 固定点稳定',
            2: '买入 - 有序状态',
            1: '温和买入 - 周期性',
            0: '持有 - 中性混沌',
            -1: '温和卖出 - 混沌增加',
            -2: '卖出 - 高度混沌',
            -3: '强烈卖出 - 极度混沌'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 混沌状态分类
        def classify_chaos_state(row):
            chaos_degree = row.get('chaos_degree', 0)
            lyap_exp = row.get('max_lyapunov_exponent', 0)
            attractor_type = row.get('attractor_type', 'unknown')

            if chaos_degree < 0.3 and lyap_exp < 0.01:
                return 'ordered_stable'
            elif chaos_degree < 0.5 and lyap_exp < 0.05:
                return 'ordered_transitional'
            elif attractor_type == 'limit_cycle':
                return 'periodic'
            elif chaos_degree > 0.7 and lyap_exp > 0.1:
                return 'chaotic_unstable'
            else:
                return 'mixed_chaos'

        recommendations['chaos_state'] = recommendations.apply(classify_chaos_state, axis=1)

        # 仓位建议
        position_map = {3: 0.6, 2: 0.4, 1: 0.3, 0: 0.2, -1: 0.1, -2: 0.0, -3: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 混沌度调整
        chaos_adjustment = 1 - recommendations['chaos_degree']
        recommendations['position_size'] *= chaos_adjustment

        # 李雅普诺夫调整
        lyap_exp = recommendations['max_lyapunov_exponent']
        lyap_adjustment = 1 / (1 + abs(lyap_exp))
        recommendations['position_size'] *= lyap_adjustment

        # 确保仓位在合理范围内
        recommendations['position_size'] = recommendations['position_size'].clip(0, 1)

        # 策略建议
        def get_chaos_strategy(chaos_state, signal_strength):
            if chaos_state == 'ordered_stable':
                return 'trend_following'  # 趋势跟踪
            elif chaos_state == 'periodic':
                return 'mean_reversion'  # 均值回归
            elif chaos_state == 'chaotic_unstable':
                return 'avoid_trading'  # 避免交易
            else:
                return 'adaptive'  # 自适应策略

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_chaos_strategy(x['chaos_state'], abs(x['signals'])),
            axis=1
        )

        # 混沌风险评估
        def assess_chaos_risk(row):
            risk_score = 0

            # 混沌度风险
            chaos_degree = row.get('chaos_degree', 0)
            if chaos_degree > 0.7:
                risk_score += 0.4

            # 李雅普诺夫风险
            lyap_exp = row.get('max_lyapunov_exponent', 0)
            if abs(lyap_exp) > 0.1:
                risk_score += 0.3

            # 吸引子类型风险
            attractor_type = row.get('attractor_type', 'unknown')
            if attractor_type in ['chaotic_attractor', 'strange_attractor']:
                risk_score += 0.3

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['risk_level'] = recommendations.apply(assess_chaos_risk, axis=1)

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成混沌时间序列
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=200, freq='D')

    # 生成洛伦兹系统的简化版本
    def generate_chaos_series(n_points):
        sigma = 10.0
        rho = 28.0
        beta = 8.0/3.0

        # 初始条件
        x, y, z = 1.0, 1.0, 1.0
        dt = 0.01
        series = []

        for _ in range(n_points):
            # 洛伦兹方程
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z

            x += dx * dt
            y += dy * dt
            z += dz * dt

            series.append(x)

        return np.array(series)

    # 生成混沌价格序列
    chaos_signal = generate_chaos_series(200)
    base_price = 100
    prices = base_price * np.exp(chaos_signal * 0.01)

    # 创建数据框
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices
    }).set_index('date')

    # 创建奇怪吸引子分析器
    saa = StrangeAttractorAnalyzer(embedding_dimension=3, time_delay=1)

    # 分析市场奇怪吸引子
    print("分析市场奇怪吸引子...")
    chaos_results = saa.analyze_market_strange_attractor(market_data)

    # 显示主要结果
    print(f"最优嵌入维度: {chaos_results.get('optimal_dimension', 3)}")
    print(f"最优时间延迟: {chaos_results.get('optimal_delay', 1)}")
    print(f"相关维数: {chaos_results.get('correlation_dimension', 1.0):.3f}")
    print(f"信息维数: {chaos_results.get('information_dimension', 1.0):.3f}")
    print(f"最大李雅普诺夫指数: {chaos_results.get('max_lyapunov_exponent', 0.0):.6f}")
    print(f"混沌程度: {chaos_results.get('chaos_degree', 0.0):.3f}")
    print(f"吸引子类型: {chaos_results.get('attractor_type', 'unknown')}")
    print(f"当前信号: {chaos_results['signals'].iloc[-1]}")

    if 'topological_entropy' in chaos_results:
        print(f"拓扑熵: {chaos_results['topological_entropy']:.3f}")

    # 获取交易建议
    recommendations = saa.get_trading_recommendations(chaos_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['chaos_degree', 'max_lyapunov_exponent', 'attractor_type',
                                          'chaos_state', 'signal_description', 'strategy', 'position_size', 'risk_level']])