"""
Hilbert-Huang Transform Analyzer
==================================

基于希尔伯特-黄变换的信号处理指标，通过经验模态分解(EMD)
将复杂的市场信号分解为多个本征模态函数(IMF)，然后进行
希尔伯特变换以获得瞬时频率和幅值特征。

特点：
- 经验模态分解(EMD)
- 本征模态函数(IMF)提取
- 瞬时频率分析
- 瞬时幅值分析
- 时频分析
- 非线性信号处理
- 自适应信号分解
- 市场周期识别

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.signal import hilbert, find_peaks
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')


class EMD:
    """经验模态分解(EMD)实现"""
    def __init__(self, max_imf: int = 10, max_sift: int = 1000):
        """
        初始化EMD

        Args:
            max_imf: 最大IMF数量，默认10
            max_sift: 最大筛选次数，默认1000
        """
        self.max_imf = max_imf
        self.max_sift = max_sift

    def extract_imf(self, signal: np.ndarray) -> List[np.ndarray]:
        """
        提取本征模态函数

        Args:
            signal: 输入信号

        Returns:
            IMF列表
        """
        imfs = []
        residue = signal.copy()

        for i in range(self.max_imf):
            if len(residue) < 3:
                break

            imf = self.sift(residue)
            if len(imf) == 0 or np.allclose(imf, 0):
                break

            imfs.append(imf)
            residue = residue - imf

            # 检查残差是否为单调函数
            if self.is_monotonic(residue):
                break

        if not np.allclose(residue, 0):
            imfs.append(residue)

        return imfs

    def sift(self, signal: np.ndarray) -> np.ndarray:
        """
        筛选过程

        Args:
            signal: 输入信号

        Returns:
            IMF
        """
        h = signal.copy()

        for _ in range(self.max_sift):
            # 找到极值点
            max_peaks, _ = find_peaks(h)
            min_peaks, _ = find_peaks(-h)

            if len(max_peaks) < 2 or len(min_peaks) < 2:
                break

            # 插值包络
            max_envelope = self.interpolate_envelope(h, max_peaks, kind='cubic')
            min_envelope = self.interpolate_envelope(h, min_peaks, kind='cubic')

            # 计算均值
            mean_envelope = (max_envelope + min_envelope) / 2

            # 更新h
            h_new = h - mean_envelope

            # 检查收敛条件
            if np.mean((h - h_new) ** 2) < 1e-6:
                break

            h = h_new

        return h

    def interpolate_envelope(self, signal: np.ndarray, peaks: np.ndarray, kind: str = 'cubic') -> np.ndarray:
        """
        插值包络

        Args:
            signal: 信号
            peaks: 峰值位置
            kind: 插值方法

        Returns:
            包络
        """
        if len(peaks) < 2:
            return np.zeros_like(signal)

        # 创建插值函数
        f = interp1d(peaks, signal[peaks], kind=kind, bounds_error=False, fill_value='extrapolate')

        # 计算包络
        envelope = f(np.arange(len(signal)))

        # 平滑处理
        envelope = gaussian_filter1d(envelope, sigma=1)

        return envelope

    def is_monotonic(self, signal: np.ndarray) -> bool:
        """
        检查信号是否单调

        Args:
            signal: 信号

        Returns:
            是否单调
        """
        diff = np.diff(signal)
        return np.all(diff >= 0) or np.all(diff <= 0)


class HilbertHuangAnalyzer:
    """
    希尔伯特-黄变换分析器

    利用经验模态分解和希尔伯特变换
    进行市场信号的时频分析和特征提取。
    """

    def __init__(self, max_imf: int = 8, freq_bands: List[Tuple[float, float]] = None):
        """
        初始化希尔伯特-黄变换分析器

        Args:
            max_imf: 最大IMF数量，默认8
            freq_bands: 频率波段，默认None
        """
        self.max_imf = max_imf
        self.freq_bands = freq_bands or [
            (0, 0.001),    # 超低频（长期趋势）
            (0.001, 0.01), # 低频（中期趋势）
            (0.01, 0.05),  # 中频（短期波动）
            (0.05, 0.2),   # 高频（日内波动）
            (0.2, 0.5)     # 超高频（噪声）
        ]
        self.name = f"Hilbert-Huang Analyzer ({max_imf})"
        self.category = "signal_processing"

        # EMD处理器
        self.emd = EMD(max_imf=max_imf)

    def empirical_mode_decomposition(self, signal: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        经验模态分解

        Args:
            signal: 输入信号

        Returns:
            EMD分解结果
        """
        if signal.empty:
            return {}

        # 移除NaN值
        clean_signal = signal.fillna(0).values

        # 执行EMD
        imfs = self.emd.extract_imf(clean_signal)

        # 创建结果DataFrame
        results = {}
        imf_df = pd.DataFrame(index=signal.index)

        for i, imf in enumerate(imfs):
            imf_df[f'IMF_{i+1}'] = imf

        results['IMFs'] = imf_df

        # 残差
        if len(imfs) > 1:
            residue = clean_signal - np.sum(imfs[:-1], axis=0)
            results['residue'] = pd.Series(residue, index=signal.index)

        return results

    def hilbert_spectrum_analysis(self, imfs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        希尔伯特谱分析

        Args:
            imfs: IMF分解结果

        Returns:
            希尔伯特谱分析结果
        """
        if 'IMFs' not in imfs:
            return {}

        results = {}
        imf_df = imfs['IMFs']

        # 对每个IMF进行希尔伯特变换
        instantaneous_freqs = {}
        instantaneous_amps = {}
        instantaneous_phases = {}

        for col in imf_df.columns:
            imf_signal = imf_df[col].fillna(0).values

            # 希尔伯特变换
            analytic_signal = hilbert(imf_signal)

            # 瞬时幅值
            instantaneous_amps[col] = np.abs(analytic_signal)

            # 瞬时频率
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_freqs[col] = np.diff(instantaneous_phase) / (2.0 * np.pi)

            # 瞬时相位
            instantaneous_phases[col] = instantaneous_phase

        # 创建DataFrame
        freq_df = pd.DataFrame(index=imf_df.index[1:])  # 频率差分，长度减1
        amp_df = pd.DataFrame(index=imf_df.index)
        phase_df = pd.DataFrame(index=imf_df.index)

        for col in imf_df.columns:
            freq_df[col] = np.pad(instantaneous_freqs[col], (0, 1), 'edge')  # 填充到原长度
            amp_df[col] = instantaneous_amps[col]
            phase_df[col] = instantaneous_phases[col]

        results['instantaneous_frequencies'] = freq_df
        results['instantaneous_amplitudes'] = amp_df
        results['instantaneous_phases'] = phase_df

        # 希尔伯特谱
        hilbert_spectrum = self.calculate_hilbert_spectrum(freq_df, amp_df)
        results['hilbert_spectrum'] = hilbert_spectrum

        return results

    def calculate_hilbert_spectrum(self, freq_df: pd.DataFrame, amp_df: pd.DataFrame) -> pd.DataFrame:
        """
        计算希尔伯特谱

        Args:
            freq_df: 瞬时频率
            amp_df: 瞬时幅值

        Returns:
            希尔伯特谱
        """
        spectrum = pd.DataFrame(index=freq_df.index)

        # 计算能量谱
        for col in freq_df.columns:
            energy = amp_df[col] ** 2
            spectrum[col] = energy

        # 频率带能量
        for i, (low_freq, high_freq) in enumerate(self.freq_bands):
            band_energy = pd.Series(0.0, index=freq_df.index)

            for col in freq_df.columns:
                mask = (freq_df[col] >= low_freq) & (freq_df[col] <= high_freq)
                band_energy += amp_df[col] ** 2 * mask

            spectrum[f'band_{i}_energy'] = band_energy

        return spectrum

    def analyze_intrinsic_characteristics(self, signal: pd.Series) -> Dict[str, pd.Series]:
        """
        分析内在特征

        Args:
            signal: 输入信号

        Returns:
            内在特征分析
        """
        characteristics = {}

        if signal.empty:
            return characteristics

        # EMD分解
        emd_results = self.empirical_mode_decomposition(signal)
        if not emd_results:
            return characteristics

        # 希尔伯特谱分析
        spectrum_results = self.hilbert_spectrum_analysis(emd_results)

        # 特征提取
        if 'instantaneous_frequencies' in spectrum_results:
            freq_df = spectrum_results['instantaneous_frequencies']

            # 平均瞬时频率
            mean_freq = freq_df.mean(axis=1)
            characteristics['mean_instantaneous_frequency'] = mean_freq

            # 频率波动率
            freq_volatility = freq_df.std(axis=1)
            characteristics['frequency_volatility'] = freq_volatility

        if 'instantaneous_amplitudes' in spectrum_results:
            amp_df = spectrum_results['instantaneous_amplitudes']

            # 平均瞬时幅值
            mean_amp = amp_df.mean(axis=1)
            characteristics['mean_instantaneous_amplitude'] = mean_amp

            # 幅值波动率
            amp_volatility = amp_df.std(axis=1)
            characteristics['amplitude_volatility'] = amp_volatility

        # 能量特征
        if 'hilbert_spectrum' in spectrum_results:
            spectrum = spectrum_results['hilbert_spectrum']

            # 总能量
            total_energy = spectrum.sum(axis=1)
            characteristics['total_energy'] = total_energy

            # 能量集中度
            energy_concentration = spectrum.max(axis=1) / (spectrum.sum(axis=1) + 1e-10)
            characteristics['energy_concentration'] = energy_concentration

            # 主频带
            dominant_band = spectrum.idxmax(axis=1)
            characteristics['dominant_frequency_band'] = dominant_band

        # 非线性特征
        characteristics.update(self.calculate_nonlinear_features(signal))

        return characteristics

    def calculate_nonlinear_features(self, signal: pd.Series) -> Dict[str, pd.Series]:
        """
        计算非线性特征

        Args:
            signal: 输入信号

        Returns:
            非线性特征
        """
        nonlinear_features = {}

        if signal.empty:
            return nonlinear_features

        clean_signal = signal.fillna(0).values

        # 1. 近似熵
        approximate_entropy = self.calculate_approximate_entropy(clean_signal)
        nonlinear_features['approximate_entropy'] = pd.Series(approximate_entropy, index=signal.index)

        # 2. 样本熵
        sample_entropy = self.calculate_sample_entropy(clean_signal)
        nonlinear_features['sample_entropy'] = pd.Series(sample_entropy, index=signal.index)

        # 3. 模糊熵
        fuzzy_entropy = self.calculate_fuzzy_entropy(clean_signal)
        nonlinear_features['fuzzy_entropy'] = pd.Series(fuzzy_entropy, index=signal.index)

        # 4. Hurst指数
        hurst_exponent = self.calculate_hurst_exponent(clean_signal)
        nonlinear_features['hurst_exponent'] = pd.Series(hurst_exponent, index=signal.index)

        # 5. 分形维数
        fractal_dimension = self.calculate_fractal_dimension(clean_signal)
        nonlinear_features['fractal_dimension'] = pd.Series(fractal_dimension, index=signal.index)

        return nonlinear_features

    def calculate_approximate_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        计算近似熵

        Args:
            signal: 信号
            m: 模式长度
            r: 容忍度

        Returns:
            近似熵
        """
        N = len(signal)
        if N < m + 1:
            return 0.0

        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        def _phi(m):
            patterns = np.array([signal[i:i+m] for i in range(N - m + 1)])
            C = np.zeros(N - m + 1)

            for i in range(N - m + 1):
                template_i = patterns[i]
                for j in range(N - m + 1):
                    if _maxdist(template_i, patterns[j]) <= r:
                        C[i] += 1

            C = C / (N - m + 1)
            phi = np.mean(np.log(C + 1e-10))
            return phi

        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0

    def calculate_sample_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2) -> float:
        """
        计算样本熵

        Args:
            signal: 信号
            m: 模式长度
            r: 容忍度

        Returns:
            样本熵
        """
        N = len(signal)
        if N < m + 1:
            return 0.0

        def _maxdist(xi, xj):
            return max([abs(ua - va) for ua, va in zip(xi, xj)])

        patterns = np.array([signal[i:i+m] for i in range(N - m)])
        patterns_next = np.array([signal[i:i+m+1] for i in range(N - m)])

        A = np.zeros(N - m)
        B = np.zeros(N - m)

        for i in range(N - m):
            template_i = patterns[i]
            for j in range(N - m):
                if _maxdist(template_i, patterns[j]) <= r:
                    B[i] += 1
                    if j < N - m - 1 and _maxdist(template_i, patterns_next[j]) <= r:
                        A[i] += 1

        A = A / (N - m)
        B = B / (N - m)

        try:
            return -np.mean(np.log(A / (B + 1e-10)))
        except:
            return 0.0

    def calculate_fuzzy_entropy(self, signal: np.ndarray, m: int = 2, r: float = 0.2, n: int = 2) -> float:
        """
        计算模糊熵

        Args:
            signal: 信号
            m: 模式长度
            r: 容忍度
            n: 模糊函数参数

        Returns:
            模糊熵
        """
        N = len(signal)
        if N < m + 1:
            return 0.0

        def _fuzzy_similarity(xi, xj):
            distance = np.sqrt(np.sum((xi - xj) ** 2))
            return np.exp(-(distance / r) ** n)

        def _phi(m):
            patterns = np.array([signal[i:i+m] for i in range(N - m + 1)])
            phi = np.zeros(N - m + 1)

            for i in range(N - m + 1):
                template_i = patterns[i]
                similarity_sum = 0
                for j in range(N - m + 1):
                    similarity_sum += _fuzzy_similarity(template_i, patterns[j])

                phi[i] = similarity_sum / (N - m + 1 - 1)

            return np.mean(np.log(phi + 1e-10))

        try:
            return _phi(m) - _phi(m + 1)
        except:
            return 0.0

    def calculate_hurst_exponent(self, signal: np.ndarray) -> float:
        """
        计算Hurst指数

        Args:
            signal: 信号

        Returns:
            Hurst指数
        """
        N = len(signal)
        if N < 100:
            return 0.5

        max_k = N // 4
        k_values = np.arange(10, max_k + 1)

        R_S = []

        for k in k_values:
            # 分割信号
            segments = [signal[i:i+k] for i in range(0, N - k + 1, k)]

            R_S_values = []
            for segment in segments:
                if len(segment) < 2:
                    continue

                # 计算累积偏差
                mean_segment = np.mean(segment)
                cumulative_deviation = np.cumsum(segment - mean_segment)

                # 计算极差
                R = np.max(cumulative_deviation) - np.min(cumulative_deviation)

                # 计算标准差
                S = np.std(segment)

                if S > 0:
                    R_S_values.append(R / S)

            if R_S_values:
                R_S.append(np.mean(R_S_values))

        if len(R_S) < 2:
            return 0.5

        # 线性回归
        log_k = np.log(k_values)
        log_RS = np.log(R_S)

        # 去除无效值
        valid_mask = np.isfinite(log_k) & np.isfinite(log_RS)
        if np.sum(valid_mask) < 2:
            return 0.5

        slope = np.polyfit(log_k[valid_mask], log_RS[valid_mask], 1)[0]

        return slope

    def calculate_fractal_dimension(self, signal: np.ndarray) -> float:
        """
        计算分形维数

        Args:
            signal: 信号

        Returns:
            分形维数
        """
        N = len(signal)
        if N < 10:
            return 1.0

        # 计算盒计数
        scales = np.logspace(0, np.log10(N//2), 20)
        counts = []

        for scale in scales:
            scale_int = int(scale)
            if scale_int < 1:
                continue

            # 分割信号
            boxes = [signal[i:i+scale_int] for i in range(0, N, scale_int)]

            # 计算非空盒子数量
            non_empty_boxes = sum(1 for box in boxes if len(box) > 0 and not np.allclose(box, 0))

            if non_empty_boxes > 0:
                counts.append(non_empty_boxes)

        if len(counts) < 2:
            return 1.0

        # 线性回归
        log_scales = np.log([int(scale) for scale in scales[:len(counts)]])
        log_counts = np.log(counts)

        slope = -np.polyfit(log_scales, log_counts, 1)[0]

        return slope

    def detect_market_cycles(self, signal: pd.Series) -> Dict[str, pd.Series]:
        """
        检测市场周期

        Args:
            signal: 输入信号

        Returns:
            市场周期检测
        """
        cycle_detection = {}

        # EMD分解
        emd_results = self.empirical_mode_decomposition(signal)
        if not emd_results:
            return cycle_detection

        # 希尔伯特谱分析
        spectrum_results = self.hilbert_spectrum_analysis(emd_results)

        if 'instantaneous_frequencies' in spectrum_results:
            freq_df = spectrum_results['instantaneous_frequencies']

            # 主频率分析
            dominant_freq = freq_df.mean(axis=1)
            cycle_detection['dominant_frequency'] = dominant_freq

            # 周期长度
            cycle_length = 1 / (dominant_freq + 1e-10)
            cycle_detection['cycle_length'] = cycle_length

            # 频率稳定性
            freq_stability = 1 / (freq_df.std(axis=1) + 1e-10)
            cycle_detection['frequency_stability'] = freq_stability

        # 能量周期分析
        if 'hilbert_spectrum' in spectrum_results:
            spectrum = spectrum_results['hilbert_spectrum']

            # 能量周期
            for i, (low_freq, high_freq) in enumerate(self.freq_bands):
                band_energy = spectrum[f'band_{i}_energy']

                # 周期强度
                cycle_strength = band_energy / band_energy.rolling(window=20).mean()
                cycle_detection[f'band_{i}_cycle_strength'] = cycle_strength.fillna(0)

        return cycle_detection

    def analyze_market_signal(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析市场信号

        Args:
            data: 市场数据

        Returns:
            希尔伯特-黄变换分析结果
        """
        results = {}

        # 提取价格信号
        if 'close' in data.columns:
            price_signal = data['close']

            # EMD分解
            emd_results = self.empirical_mode_decomposition(price_signal)
            results.update(emd_results)

            # 希尔伯特谱分析
            spectrum_results = self.hilbert_spectrum_analysis(emd_results)
            results.update(spectrum_results)

            # 内在特征分析
            characteristics = self.analyze_intrinsic_characteristics(price_signal)
            results.update(characteristics)

            # 市场周期检测
            cycle_detection = self.detect_market_cycles(price_signal)
            results.update(cycle_detection)

        # 成交量信号分析（如果可用）
        if 'volume' in data.columns:
            volume_signal = data['volume']
            volume_characteristics = self.analyze_intrinsic_characteristics(volume_signal)

            # 重命名以区分价格和成交量特征
            for key, value in volume_characteristics.items():
                results[f'volume_{key}'] = value

        # 生成交易信号
        results['signals'] = self.generate_hht_signals(results)

        return results

    def generate_hht_signals(self, hht_analysis: Dict[str, Any]) -> pd.Series:
        """
        基于HHT分析生成交易信号

        Args:
            hht_analysis: HHT分析结果

        Returns:
            交易信号
        """
        # 找到一个指标序列作为索引
        index_keys = [k for k, v in hht_analysis.items() if isinstance(v, pd.Series) and not v.empty]
        if not index_keys:
            return pd.Series()

        index = hht_analysis[index_keys[0]].index
        signals = pd.Series(0, index=index)

        # 频率特征信号
        dominant_freq = hht_analysis.get('dominant_frequency', pd.Series(0, index=index))
        freq_stability = hht_analysis.get('frequency_stability', pd.Series(0, index=index))

        # 幅值特征信号
        mean_amp = hht_analysis.get('mean_instantaneous_amplitude', pd.Series(0, index=index))
        amp_volatility = hht_analysis.get('amplitude_volatility', pd.Series(0, index=index))

        # 能量特征信号
        total_energy = hht_analysis.get('total_energy', pd.Series(0, index=index))
        energy_concentration = hht_analysis.get('energy_concentration', pd.Series(0, index=index))

        # 非线性特征信号
        hurst_exponent = hht_analysis.get('hurst_exponent', pd.Series(0.5, index=index))
        fractal_dimension = hht_analysis.get('fractal_dimension', pd.Series(1.0, index=index))

        # 信号阈值
        freq_threshold = dominant_freq.quantile(0.8)
        energy_threshold = total_energy.quantile(0.8)
        stability_threshold = freq_stability.quantile(0.2)

        # 频率信号
        low_freq_signal = (dominant_freq < freq_threshold * 0.5) & (freq_stability > stability_threshold)
        high_freq_signal = (dominant_freq > freq_threshold * 1.5) & (freq_stability < stability_threshold)

        # 能量信号
        high_energy_signal = total_energy > energy_threshold
        low_energy_signal = total_energy < total_energy.quantile(0.2)

        # 非线性信号
        persistent_signal = hurst_exponent > 0.6
        anti_persistent_signal = hurst_exponent < 0.4

        # 综合信号逻辑
        # 买入信号：低频 + 高能量 + 持续性
        buy_signal = low_freq_signal & high_energy_signal & persistent_signal

        # 卖出信号：高频 + 低能量 + 反持续性
        sell_signal = high_freq_signal & low_energy_signal & anti_persistent_signal

        # 强信号确认
        strong_buy = buy_signal & (energy_concentration > energy_concentration.quantile(0.8))
        strong_sell = sell_signal & (amp_volatility > amp_volatility.quantile(0.8))

        # 分配信号值
        signals[strong_buy] = 3
        signals[buy_signal] = 2
        signals[strong_sell] = -3
        signals[sell_signal] = -2

        return signals

    def get_trading_recommendations(self, hht_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            hht_analysis: HHT分析结果

        Returns:
            交易建议DataFrame
        """
        # 找到一个指标序列作为索引
        index_keys = [k for k, v in hht_analysis.items() if isinstance(v, pd.Series) and not v.empty]
        if not index_keys:
            return pd.DataFrame()

        index = hht_analysis[index_keys[0]].index
        recommendations = pd.DataFrame(index=index)

        # 添加HHT分析指标
        for key, value in hht_analysis.items():
            if isinstance(value, pd.Series):
                recommendations[key] = value

        # 交易信号
        recommendations['signals'] = hht_analysis.get('signals', pd.Series(0, index=index))

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 低频高能量持续信号',
            2: '买入 - 低频稳定信号',
            0: '持有 - 中性信号',
            -2: '卖出 - 高频低能量信号',
            -3: '强烈卖出 - 高频不稳定信号'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 市场状态分类
        def classify_market_state(row):
            dominant_freq = row.get('dominant_frequency', 0)
            freq_stability = row.get('frequency_stability', 0)
            total_energy = row.get('total_energy', 0)
            hurst_exponent = row.get('hurst_exponent', 0.5)

            if freq_stability > freq_stability.quantile(0.8) and total_energy > total_energy.quantile(0.7):
                return 'stable_high_energy'
            elif dominant_freq > dominant_freq.quantile(0.8) and freq_stability < freq_stability.quantile(0.2):
                return 'high_freq_unstable'
            elif hurst_exponent > 0.6:
                return 'trending_market'
            elif hurst_exponent < 0.4:
                return 'mean_reverting'
            else:
                return 'normal_market'

        recommendations['market_state'] = recommendations.apply(classify_market_state, axis=1)

        # 仓位建议
        position_map = {3: 0.5, 2: 0.3, 0: 0.2, -2: 0.1, -3: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 频率稳定性调整
        freq_stability = recommendations.get('frequency_stability', 1.0)
        stability_adjustment = np.minimum(freq_stability / freq_stability.quantile(0.5), 2.0)
        recommendations['position_size'] *= stability_adjustment

        # 能量调整
        total_energy = recommendations.get('total_energy', 1.0)
        energy_adjustment = total_energy / (total_energy.quantile(0.5) + 1e-10)
        recommendations['position_size'] *= energy_adjustment

        # 确保仓位在合理范围内
        recommendations['position_size'] = recommendations['position_size'].clip(0, 1)

        # 策略建议
        def get_hht_strategy(market_state, signal_strength):
            if market_state == 'stable_high_energy':
                return 'trend_following'  # 趋势跟踪
            elif market_state == 'high_freq_unstable':
                return 'avoid_trading'  # 避免交易
            elif market_state == 'trending_market':
                return 'momentum'  # 动量策略
            elif market_state == 'mean_reverting':
                return 'mean_reversion'  # 均值回归
            else:
                return 'balanced'

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_hht_strategy(x['market_state'], abs(x['signals'])),
            axis=1
        )

        # HHT风险评估
        def assess_hht_risk(row):
            risk_score = 0

            # 频率风险
            if row.get('dominant_frequency', 0) > row.get('dominant_frequency', 0).quantile(0.8):
                risk_score += 0.3

            # 稳定性风险
            if row.get('frequency_stability', 0) < row.get('frequency_stability', 0).quantile(0.2):
                risk_score += 0.3

            # 非线性风险
            hurst_exp = row.get('hurst_exponent', 0.5)
            if hurst_exp < 0.3 or hurst_exp > 0.7:
                risk_score += 0.2

            # 能量风险
            if row.get('total_energy', 0) < row.get('total_energy', 0).quantile(0.2):
                risk_score += 0.2

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['risk_level'] = recommendations.apply(assess_hht_risk, axis=1)

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成模拟市场数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')

    # 生成复杂信号（包含多个频率分量）
    t = np.arange(300)
    signal = (
        np.sin(2 * np.pi * t / 100) +  # 低频趋势
        0.5 * np.sin(2 * np.pi * t / 20) +  # 中频周期
        0.2 * np.sin(2 * np.pi * t / 5) +   # 高频波动
        0.1 * np.random.randn(300)  # 噪声
    )

    # 转换为价格序列
    base_price = 100
    prices = base_price * np.exp(np.cumsum(signal * 0.01))

    # 生成成交量
    volume = 1000000 + 500000 * np.sin(2 * np.pi * t / 50) + 200000 * np.random.randn(300)

    # 创建数据框
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volume
    }).set_index('date')

    # 创建希尔伯特-黄变换分析器
    hha = HilbertHuangAnalyzer(max_imf=6)

    # 分析市场信号
    print("分析市场信号...")
    hht_results = hha.analyze_market_signal(market_data)

    # 显示主要结果
    print(f"主频率: {hht_results['dominant_frequency'].iloc[-1]:.6f}")
    print(f"周期长度: {hht_results['cycle_length'].iloc[-1]:.2f}")
    print(f"频率稳定性: {hht_results['frequency_stability'].iloc[-1]:.3f}")
    print(f"总能量: {hht_results['total_energy'].iloc[-1]:.6f}")
    print(f"Hurst指数: {hht_results['hurst_exponent'].iloc[-1]:.3f}")
    print(f"分形维数: {hht_results['fractal_dimension'].iloc[-1]:.3f}")
    print(f"当前信号: {hht_results['signals'].iloc[-1]}")

    # 显示IMF信息
    if 'IMFs' in hht_results:
        print(f"\nIMF数量: {hht_results['IMFs'].shape[1]}")

    # 获取交易建议
    recommendations = hha.get_trading_recommendations(hht_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['dominant_frequency', 'cycle_length', 'market_state',
                                          'signal_description', 'strategy', 'position_size', 'risk_level']])