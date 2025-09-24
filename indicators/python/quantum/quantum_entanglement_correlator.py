"""
Quantum Entanglement Correlator
=================================

基于量子纠缠原理的市场相关性分析器，模拟量子纠缠现象
来分析不同资产间的复杂相关性关系。该指标利用量子力学
中的纠缠态概念来识别市场中的隐藏相关性和同步性。

特点：
- 量子纠缠模拟
- 多资产相关性分析
- 纠缠度量计算
- 贝尔不等式检验
- 量子态纯度分析
- 纠缠熵计算
- 同步性检测
- 隐藏相关性发现

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.linalg import svd, eigvalsh, det
from scipy.stats import pearsonr, spearmanr, kendalltau
from scipy.fft import fft, ifft
import warnings
warnings.filterwarnings('ignore')


class QuantumState:
    """量子态表示"""
    def __init__(self, amplitudes: np.ndarray):
        """
        初始化量子态

        Args:
            amplitudes: 振幅数组
        """
        self.amplitudes = amplitudes.astype(complex)
        self.normalize()

    def normalize(self):
        """归一化量子态"""
        norm = np.sqrt(np.sum(np.abs(self.amplitudes) ** 2))
        if norm > 0:
            self.amplitudes /= norm

    def density_matrix(self) -> np.ndarray:
        """计算密度矩阵"""
        return np.outer(self.amplitudes, np.conj(self.amplitudes))

    def partial_trace(self, subsystem_dims: Tuple[int, int], keep: int = 0) -> np.ndarray:
        """
        计算偏迹

        Args:
            subsystem_dims: 子系统维度
            keep: 保留的子系统

        Returns:
            约化密度矩阵
        """
        dim1, dim2 = subsystem_dims
        density = self.density_matrix()

        if keep == 0:
            # 迹出第二个系统
            reduced = np.zeros((dim1, dim1), dtype=complex)
            for i in range(dim1):
                for j in range(dim1):
                    for k in range(dim2):
                        reduced[i, j] += density[i * dim2 + k, j * dim2 + k]
        else:
            # 迹出第一个系统
            reduced = np.zeros((dim2, dim2), dtype=complex)
            for i in range(dim2):
                for j in range(dim2):
                    for k in range(dim1):
                        reduced[i, j] += density[k * dim2 + i, k * dim2 + j]

        return reduced

    def entropy(self) -> float:
        """计算冯诺依曼熵"""
        rho = self.density_matrix()
        eigenvalues = eigvalsh(rho)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # 去除数值误差

        entropy = -np.sum(eigenvalues * np.log2(eigenvalues + 1e-10))
        return entropy


class EntangledState:
    """纠缠态"""
    def __init__(self, n_qubits: int):
        """
        初始化纠缠态

        Args:
            n_qubits: 量子比特数
        """
        self.n_qubits = n_qubits
        self.amplitudes = np.zeros(2**n_qubits, dtype=complex)
        self.create_bell_state()

    def create_bell_state(self):
        """创建贝尔态"""
        if self.n_qubits >= 2:
            # 创建|Φ⁺⟩ = (|00⟩ + |11⟩)/√2
            self.amplitudes[0] = 1.0 / np.sqrt(2)  # |00⟩
            self.amplitudes[3] = 1.0 / np.sqrt(2)  # |11⟩

    def create_ghz_state(self):
        """创建GHZ态"""
        if self.n_qubits >= 3:
            # 创建|GHZ⟩ = (|000⟩ + |111⟩)/√2
            self.amplitudes[0] = 1.0 / np.sqrt(2)  # |000⟩
            self.amplitudes[7] = 1.0 / np.sqrt(2)  # |111⟩

    def apply_correlation(self, correlation_matrix: np.ndarray):
        """
        应用相关性矩阵

        Args:
            correlation_matrix: 相关性矩阵
        """
        # 将相关性转换为量子态
        for i in range(min(len(correlation_matrix), 2**self.n_qubits)):
            for j in range(min(len(correlation_matrix), 2**self.n_qubits)):
                if correlation_matrix[i, j] > 0.5:  # 强相关
                    phase = np.pi * correlation_matrix[i, j]
                    self.amplitudes[i] *= np.exp(1j * phase / 2)
                    self.amplitudes[j] *= np.exp(1j * phase / 2)

        self.normalize()


class QuantumEntanglementCorrelator:
    """
    量子纠缠相关器

    利用量子纠缠原理分析市场中的
    复杂相关性和同步性模式。
    """

    def __init__(self, max_assets: int = 8, entanglement_threshold: float = 0.7):
        """
        初始化量子纠缠相关器

        Args:
            max_assets: 最大资产数量，默认8
            entanglement_threshold: 纠缠阈值，默认0.7
        """
        self.max_assets = max_assets
        self.entanglement_threshold = entanglement_threshold
        self.name = f"Quantum Entanglement Correlator ({max_assets})"
        self.category = "quantum"

    def encode_market_data_to_quantum(self, data: pd.DataFrame) -> Dict[str, QuantumState]:
        """
        编码市场数据到量子态

        Args:
            data: 市场数据（多资产）

        Returns:
            量子态字典
        """
        quantum_states = {}

        # 提取资产数据
        asset_columns = [col for col in data.columns if col not in ['date', 'index']]
        asset_columns = asset_columns[:self.max_assets]  # 限制资产数量

        for asset in asset_columns:
            if asset in data.columns:
                series = data[asset].fillna(0)

                # 计算特征
                features = self.extract_market_features(series)

                # 编码为量子态振幅
                amplitudes = self.features_to_amplitudes(features)

                # 创建量子态
                quantum_state = QuantumState(amplitudes)
                quantum_states[asset] = quantum_state

        return quantum_states

    def extract_market_features(self, series: pd.Series) -> np.ndarray:
        """
        提取市场特征

        Args:
            series: 价格序列

        Returns:
            特征数组
        """
        features = []

        # 价格特征
        if len(series) > 1:
            returns = series.pct_change().fillna(0)

            # 动量特征
            for window in [5, 10, 20]:
                if len(series) >= window:
                    momentum = series.pct_change(window).fillna(0)
                    features.append(momentum.iloc[-1])

            # 波动率特征
            if len(returns) >= 20:
                volatility = returns.rolling(window=20).std().fillna(0)
                features.append(volatility.iloc[-1])

            # 趋势特征
            if len(series) >= 10:
                trend = np.polyfit(range(len(series[-10:])), series[-10:].values, 1)[0]
                features.append(trend)

            # 技术指标
            if len(series) >= 14:
                # RSI
                delta = series.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / (loss + 1e-10)
                rsi = 100 - (100 / (1 + rs))
                features.append(rsi.iloc[-1] if len(rsi) > 0 else 50)

        # 填充特征到标准长度
        target_length = 8  # 3个量子比特
        while len(features) < target_length:
            features.append(0.0)
        features = features[:target_length]

        return np.array(features)

    def features_to_amplitudes(self, features: np.ndarray) -> np.ndarray:
        """
        将特征转换为振幅

        Args:
            features: 特征数组

        Returns:
            量子态振幅
        """
        # 归一化特征
        if len(features) > 0 and np.std(features) > 0:
            features = (features - np.mean(features)) / np.std(features)

        # 创建振幅
        n_states = 2 ** int(np.log2(len(features)) + 1) if len(features) > 0 else 2
        amplitudes = np.zeros(n_states, dtype=complex)

        # 线性映射特征到振幅
        for i, feature in enumerate(features):
            state_idx = min(i, n_states - 1)
            # 振幅包含实部和虚部
            amplitudes[state_idx] = complex(feature * 0.1, feature * 0.05)

        # 添加叠加态
        for i in range(len(features), n_states):
            amplitudes[i] = complex(0.01, 0.01)

        return amplitudes

    def calculate_entanglement_measures(self, quantum_states: Dict[str, QuantumState]) -> Dict[str, Any]:
        """
        计算纠缠度量

        Args:
            quantum_states: 量子态字典

        Returns:
            纠缠度量结果
        """
        results = {}

        assets = list(quantum_states.keys())
        if len(assets) < 2:
            return results

        # 计算两两纠缠度
        entanglement_matrix = np.zeros((len(assets), len(assets)))
        concurrence_matrix = np.zeros((len(assets), len(assets)))

        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    # 计算纠缠度
                    entanglement = self.calculate_pairwise_entanglement(
                        quantum_states[asset1], quantum_states[asset2]
                    )
                    entanglement_matrix[i, j] = entanglement
                    entanglement_matrix[j, i] = entanglement

                    # 计算concurrence
                    concurrence = self.calculate_concurrence(
                        quantum_states[asset1], quantum_states[asset2]
                    )
                    concurrence_matrix[i, j] = concurrence
                    concurrence_matrix[j, i] = concurrence

        results['entanglement_matrix'] = entanglement_matrix
        results['concurrence_matrix'] = concurrence_matrix

        # 计算全局纠缠度量
        global_entanglement = self.calculate_global_entanglement(quantum_states)
        results['global_entanglement'] = global_entanglement

        # 计算纠缠熵
        entanglement_entropy = self.calculate_entanglement_entropy(quantum_states)
        results['entanglement_entropy'] = entanglement_entropy

        return results

    def calculate_pairwise_entanglement(self, state1: QuantumState, state2: QuantumState) -> float:
        """
        计算两两纠缠度

        Args:
            state1: 量子态1
            state2: 量子态2

        Returns:
            纠缠度
        """
        # 创建复合系统
        composite_amplitudes = np.outer(state1.amplitudes, state2.amplitudes).flatten()
        composite_state = QuantumState(composite_amplitudes)

        # 计算约化密度矩阵
        dim1 = len(state1.amplitudes)
        dim2 = len(state2.amplitudes)
        reduced_density = composite_state.partial_trace((dim1, dim2), keep=0)

        # 计算纠缠度（使用线性熵）
        purity = np.real(np.trace(reduced_density @ reduced_density))
        entanglement = 1 - purity

        return entanglement

    def calculate_concurrence(self, state1: QuantumState, state2: QuantumState) -> float:
        """
        计算concurrence

        Args:
            state1: 量子态1
            state2: 量子态2

        Returns:
            Concurrence
        """
        # 简化的concurrence计算
        entanglement = self.calculate_pairwise_entanglement(state1, state2)
        concurrence = np.sqrt(max(0, entanglement))

        return concurrence

    def calculate_global_entanglement(self, quantum_states: Dict[str, QuantumState]) -> float:
        """
        计算全局纠缠度量

        Args:
            quantum_states: 量子态字典

        Returns:
            全局纠缠度
        """
        if len(quantum_states) < 2:
            return 0.0

        # 计算所有两两纠缠度的平均值
        total_entanglement = 0
        count = 0

        assets = list(quantum_states.keys())
        for i, asset1 in enumerate(assets):
            for j, asset2 in enumerate(assets):
                if i < j:
                    entanglement = self.calculate_pairwise_entanglement(
                        quantum_states[asset1], quantum_states[asset2]
                    )
                    total_entanglement += entanglement
                    count += 1

        return total_entanglement / count if count > 0 else 0.0

    def calculate_entanglement_entropy(self, quantum_states: Dict[str, QuantumState]) -> Dict[str, float]:
        """
        计算纠缠熵

        Args:
            quantum_states: 量子态字典

        Returns:
            纠缠熵字典
        """
        entropy_results = {}

        for asset, state in quantum_states.items():
            # 计算单个系统的熵
            entropy = state.entropy()
            entropy_results[f'{asset}_entropy'] = entropy

        return entropy_results

    def bell_inequality_test(self, quantum_states: Dict[str, QuantumState]) -> Dict[str, Any]:
        """
        贝尔不等式检验

        Args:
            quantum_states: 量子态字典

        Returns:
            贝尔检验结果
        """
        results = {}

        assets = list(quantum_states.keys())
        if len(assets) < 2:
            return results

        # 选择两个资产进行检验
        asset1, asset2 = assets[0], assets[1]
        state1 = quantum_states[asset1]
        state2 = quantum_states[asset2]

        # 计算贝尔参数
        bell_parameter = self.calculate_bell_parameter(state1, state2)

        # 经典极限
        classical_limit = 2.0
        quantum_limit = 2.0 * np.sqrt(2)

        results['bell_parameter'] = bell_parameter
        results['classical_limit'] = classical_limit
        results['quantum_limit'] = quantum_limit
        results['violates_classical'] = bell_parameter > classical_limit
        results['quantum_correlation_strength'] = (bell_parameter - classical_limit) / (quantum_limit - classical_limit)

        return results

    def calculate_bell_parameter(self, state1: QuantumState, state2: QuantumState) -> float:
        """
        计算贝尔参数

        Args:
            state1: 量子态1
            state2: 量子态2

        Returns:
            贝尔参数
        """
        # 简化的贝尔参数计算
        # 基于量子态的相关性

        # 计算不同测量基的相关性
        correlations = []

        # 模拟不同角度的测量
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

        for angle1 in angles:
            for angle2 in angles:
                # 计算相关性
                corr = self.calculate_quantum_correlation(state1, state2, angle1, angle2)
                correlations.append(corr)

        # 计算贝尔参数
        if len(correlations) >= 4:
            bell_parameter = abs(correlations[0] - correlations[1]) + abs(correlations[2] + correlations[3])
        else:
            bell_parameter = 2.0  # 经典值

        return bell_parameter

    def calculate_quantum_correlation(self, state1: QuantumState, state2: QuantumState,
                                   angle1: float, angle2: float) -> float:
        """
        计算量子相关性

        Args:
            state1: 量子态1
            state2: 量子态2
            angle1: 测量角度1
            angle2: 测量角度2

        Returns:
            量子相关性
        """
        # 简化的量子相关性计算
        # 基于振幅的相位关系

        # 提取相位信息
        phases1 = np.angle(state1.amplitudes)
        phases2 = np.angle(state2.amplitudes)

        # 计算相位差
        phase_diff = np.mean(phases1) - np.mean(phases2)

        # 考虑测量角度
        correlation = np.cos(phase_diff + angle1 - angle2)

        return correlation

    def detect_quantum_synchronization(self, quantum_states: Dict[str, QuantumState]) -> Dict[str, Any]:
        """
        检测量子同步性

        Args:
            quantum_states: 量子态字典

        Returns:
            同步性检测结果
        """
        results = {}

        assets = list(quantum_states.keys())
        if len(assets) < 2:
            return results

        # 计算相位同步性
        phase_sync = self.calculate_phase_synchronization(quantum_states)
        results['phase_synchronization'] = phase_sync

        # 计算振幅同步性
        amplitude_sync = self.calculate_amplitude_synchronization(quantum_states)
        results['amplitude_synchronization'] = amplitude_sync

        # 计算频率同步性
        frequency_sync = self.calculate_frequency_synchronization(quantum_states)
        results['frequency_synchronization'] = frequency_sync

        # 综合同步性
        overall_sync = (phase_sync + amplitude_sync + frequency_sync) / 3
        results['overall_synchronization'] = overall_sync

        return results

    def calculate_phase_synchronization(self, quantum_states: Dict[str, QuantumState]) -> float:
        """
        计算相位同步性

        Args:
            quantum_states: 量子态字典

        Returns:
            相位同步性
        """
        phases = []
        for state in quantum_states.values():
            phase = np.angle(np.mean(state.amplitudes))
            phases.append(phase)

        if len(phases) < 2:
            return 0.0

        # 计算相位一致性
        phase_coherence = np.abs(np.mean(np.exp(1j * np.array(phases))))
        return phase_coherence

    def calculate_amplitude_synchronization(self, quantum_states: Dict[str, QuantumState]) -> float:
        """
        计算振幅同步性

        Args:
            quantum_states: 量子态字典

        Returns:
            振幅同步性
        """
        amplitudes = []
        for state in quantum_states.values():
            amplitude = np.mean(np.abs(state.amplitudes))
            amplitudes.append(amplitude)

        if len(amplitudes) < 2:
            return 0.0

        # 计算振幅相关性
        amp_array = np.array(amplitudes)
        correlation = np.corrcoef(amp_array.reshape(1, -1), amp_array.reshape(1, -1))[0, 1]

        return correlation if np.isfinite(correlation) else 0.0

    def calculate_frequency_synchronization(self, quantum_states: Dict[str, QuantumState]) -> float:
        """
        计算频率同步性

        Args:
            quantum_states: 量子态字典

        Returns:
            频率同步性
        """
        frequencies = []
        for state in quantum_states.values():
            # 简化的频率计算（基于振幅变化）
            amplitude_series = np.abs(state.amplitudes)
            if len(amplitude_series) > 1:
                # 计算主导频率
                fft_result = fft(amplitude_series)
                dominant_freq = np.argmax(np.abs(fft_result[1:len(fft_result)//2]))
                frequencies.append(dominant_freq)

        if len(frequencies) < 2:
            return 0.0

        # 计算频率相似性
        freq_array = np.array(frequencies)
        similarity = 1 / (1 + np.std(freq_array))

        return similarity

    def analyze_market_entanglement(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析市场纠缠

        Args:
            data: 市场数据

        Returns:
            量子纠缠分析结果
        """
        results = {}

        # 编码市场数据到量子态
        quantum_states = self.encode_market_data_to_quantum(data)
        results['quantum_states'] = quantum_states

        # 计算纠缠度量
        entanglement_measures = self.calculate_entanglement_measures(quantum_states)
        results.update(entanglement_measures)

        # 贝尔不等式检验
        bell_test = self.bell_inequality_test(quantum_states)
        results.update(bell_test)

        # 检测量子同步性
        sync_detection = self.detect_quantum_synchronization(quantum_states)
        results.update(sync_detection)

        # 生成交易信号
        results['signals'] = self.generate_entanglement_signals(results)

        return results

    def generate_entanglement_signals(self, entanglement_analysis: Dict[str, Any]) -> pd.Series:
        """
        基于纠缠分析生成交易信号

        Args:
            entanglement_analysis: 纠缠分析结果

        Returns:
            交易信号
        """
        # 创建时间序列（简化处理）
        n_points = 50
        dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
        signals = pd.Series(0, index=dates)

        # 基于全局纠缠度生成信号
        global_entanglement = entanglement_analysis.get('global_entanglement', 0.0)

        # 基于同步性生成信号
        overall_sync = entanglement_analysis.get('overall_synchronization', 0.0)

        # 基于贝尔参数生成信号
        bell_parameter = entanglement_analysis.get('bell_parameter', 2.0)
        violates_classical = entanglement_analysis.get('violates_classical', False)

        # 信号阈值
        high_entanglement_threshold = self.entanglement_threshold
        high_sync_threshold = 0.7

        # 生成信号
        high_entanglement = global_entanglement > high_entanglement_threshold
        high_sync = overall_sync > high_sync_threshold

        # 买入信号：高纠缠 + 高同步 + 违反经典物理
        buy_signal = high_entanglement & high_sync & violates_classical

        # 卖出信号：低纠缠 + 低同步
        low_entanglement = global_entanglement < 0.3
        low_sync = overall_sync < 0.3
        sell_signal = low_entanglement & low_sync

        # 分配信号值
        signals[buy_signal] = 3
        signals[sell_signal] = -3

        # 中等信号
        moderate_entanglement = (global_entanglement >= 0.3) & (global_entanglement <= high_entanglement_threshold)
        moderate_sync = (overall_sync >= 0.3) & (overall_sync <= high_sync_threshold)
        signals[moderate_entanglement & moderate_sync] = 1

        return signals

    def get_trading_recommendations(self, entanglement_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            entanglement_analysis: 纠缠分析结果

        Returns:
            交易建议DataFrame
        """
        # 创建时间序列
        n_points = 50
        dates = pd.date_range('2023-01-01', periods=n_points, freq='D')
        recommendations = pd.DataFrame(index=dates)

        # 添加纠缠分析指标
        recommendations['global_entanglement'] = entanglement_analysis.get('global_entanglement', 0.0)
        recommendations['overall_synchronization'] = entanglement_analysis.get('overall_synchronization', 0.0)
        recommendations['bell_parameter'] = entanglement_analysis.get('bell_parameter', 2.0)
        recommendations['quantum_correlation_strength'] = entanglement_analysis.get('quantum_correlation_strength', 0.0)

        # 交易信号
        signals = self.generate_entanglement_signals(entanglement_analysis)
        recommendations['signals'] = signals

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 量子纠缠确认',
            1: '温和买入 - 量子相关性',
            0: '持有 - 经典相关性',
            -3: '强烈卖出 - 量子退纠缠'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 量子状态分类
        def classify_quantum_correlation_state(row):
            entanglement = row.get('global_entanglement', 0)
            sync = row.get('overall_synchronization', 0)
            bell_param = row.get('bell_parameter', 2.0)

            if entanglement > self.entanglement_threshold and sync > 0.7:
                return 'quantum_entangled'
            elif bell_param > 2.0:
                return 'quantum_correlated'
            elif entanglement > 0.3:
                return 'semi_classical'
            else:
                return 'classical'

        recommendations['correlation_state'] = recommendations.apply(classify_quantum_correlation_state, axis=1)

        # 仓位建议
        position_map = {3: 0.5, 1: 0.3, 0: 0.2, -3: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 纠缠度调整
        entanglement_adjustment = recommendations['global_entanglement']
        recommendations['position_size'] *= entanglement_adjustment

        # 同步性调整
        sync_adjustment = recommendations['overall_synchronization']
        recommendations['position_size'] *= sync_adjustment

        # 确保仓位在合理范围内
        recommendations['position_size'] = recommendations['position_size'].clip(0, 1)

        # 策略建议
        def get_quantum_strategy(correlation_state, signal_strength):
            if correlation_state == 'quantum_entangled':
                return 'quantum_entanglement'  # 量子纠缠策略
            elif correlation_state == 'quantum_correlated':
                return 'quantum_correlation'  # 量子相关策略
            elif correlation_state == 'semi_classical':
                return 'semi_classical'  # 半经典策略
            else:
                return 'classical'  # 经典策略

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_quantum_strategy(x['correlation_state'], abs(x['signals'])),
            axis=1
        )

        # 量子风险评估
        def assess_quantum_risk(row):
            risk_score = 0

            # 贝尔参数风险
            bell_param = row.get('bell_parameter', 2.0)
            if bell_param > 2.5:
                risk_score += 0.4

            # 低纠缠风险
            entanglement = row.get('global_entanglement', 0)
            if entanglement < 0.2:
                risk_score += 0.3

            # 低同步风险
            sync = row.get('overall_synchronization', 0)
            if sync < 0.2:
                risk_score += 0.3

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['risk_level'] = recommendations.apply(assess_quantum_risk, axis=1)

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成模拟多资产市场数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=100, freq='D')

    # 生成4个资产的价格序列
    assets = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
    base_prices = [150, 2500, 300, 150]

    market_data = pd.DataFrame(index=dates)

    for i, (asset, base_price) in enumerate(zip(assets, base_prices)):
        # 生成相关但不同的价格序列
        common_factor = np.random.randn(100) * 0.02
        specific_factor = np.random.randn(100) * 0.01

        # 添加一些相关性
        if i > 0:
            correlation_factor = market_data[assets[0]].pct_change().fillna(0) * 0.3
        else:
            correlation_factor = 0

        returns = common_factor + specific_factor + correlation_factor
        prices = [base_price]
        for ret in returns:
            prices.append(prices[-1] * (1 + ret))

        market_data[asset] = prices[1:]

    # 创建量子纠缠相关器
    qec = QuantumEntanglementCorrelator(max_assets=4)

    # 分析市场纠缠
    print("分析市场量子纠缠...")
    entanglement_results = qec.analyze_market_entanglement(market_data)

    # 显示主要结果
    print(f"全局纠缠度: {entanglement_results['global_entanglement']:.3f}")
    print(f"整体同步性: {entanglement_results['overall_synchronization']:.3f}")
    print(f"贝尔参数: {entanglement_results['bell_parameter']:.3f}")
    print(f"违反经典物理: {entanglement_results['violates_classical']}")
    print(f"量子相关强度: {entanglement_results['quantum_correlation_strength']:.3f}")
    print(f"当前信号: {entanglement_results['signals'].iloc[-1]}")

    # 显示纠缠矩阵
    if 'entanglement_matrix' in entanglement_results:
        print("\n纠缠矩阵:")
        print(entanglement_results['entanglement_matrix'])

    # 获取交易建议
    recommendations = qec.get_trading_recommendations(entanglement_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['global_entanglement', 'overall_synchronization', 'correlation_state',
                                          'signal_description', 'strategy', 'position_size', 'risk_level']])