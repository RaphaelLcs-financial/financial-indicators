"""
Wavelet Multiresolution Analyzer
================================

基于小波多分辨率分析的信号处理指标，利用小波变换
在不同尺度上分析市场信号，提取多时间尺度特征
和识别周期模式。该指标特别适合分析非平稳信号
和局部特征。

特点：
- 多分辨率分析
- 小波分解与重构
- 时频局部化分析
- 奇异点检测
- 去噪处理
- 特征提取
- 周期识别
- 趋势分离

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.signal import convolve, morlet, ricker
from scipy.fft import fft, ifft, fftfreq
import pywt
import warnings
warnings.filterwarnings('ignore')


class WaveletProcessor:
    """小波处理器"""
    def __init__(self, wavelet_name: str = 'db4', max_level: int = 6):
        """
        初始化小波处理器

        Args:
            wavelet_name: 小波名称，默认'db4'
            max_level: 最大分解级别，默认6
        """
        self.wavelet_name = wavelet_name
        self.max_level = max_level
        self.wavelet = pywt.Wavelet(wavelet_name)

    def decompose_signal(self, signal: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        小波分解信号

        Args:
            signal: 输入信号

        Returns:
            近似系数和细节系数列表
        """
        # 确保信号长度为2的幂次方
        n = len(signal)
        new_n = 2 ** int(np.log2(n)) if n > 0 else n
        if new_n != n:
            signal_padded = np.interp(np.linspace(0, 1, new_n), np.linspace(0, 1, n), signal)
        else:
            signal_padded = signal

        # 执行小波分解
        coeffs = pywt.wavedec(signal_padded, self.wavelet, level=min(self.max_level, pywt.dwt_max_level(new_n, self.wavelet)))

        # 分离近似系数和细节系数
        approximation = coeffs[0]
        details = coeffs[1:]

        return details, approximation

    def reconstruct_signal(self, details: List[np.ndarray], approximation: np.ndarray) -> np.ndarray:
        """
        重构信号

        Args:
            details: 细节系数列表
            approximation: 近似系数

        Returns:
            重构信号
        """
        # 组合系数
        coeffs = [approximation] + details

        # 重构信号
        reconstructed = pywt.waverec(coeffs, self.wavelet)

        return reconstructed

    def continuous_wavelet_transform(self, signal: np.ndarray, scales: np.ndarray) -> np.ndarray:
        """
        连续小波变换

        Args:
            signal: 输入信号
            scales: 尺度数组

        Returns:
            小波系数
        """
        # 使用Morlet小波
        coef, _ = pywt.cwt(signal, scales, self.wavelet_name)
        return coef

    def calculate_energy_distribution(self, details: List[np.ndarray], approximation: np.ndarray) -> Dict[str, float]:
        """
        计算能量分布

        Args:
            details: 细节系数
            approximation: 近似系数

        Returns:
            能量分布字典
        """
        energy_dist = {}

        # 计算各层能量
        total_energy = np.sum(approximation ** 2)
        for i, detail in enumerate(details):
            detail_energy = np.sum(detail ** 2)
            total_energy += detail_energy
            energy_dist[f'detail_{i+1}'] = detail_energy

        energy_dist['approximation'] = np.sum(approximation ** 2)

        # 归一化
        if total_energy > 0:
            for key in energy_dist:
                energy_dist[key] /= total_energy

        return energy_dist


class WaveletMultiresolutionAnalyzer:
    """
    小波多分辨率分析器

    利用小波变换进行多时间尺度分析
    和特征提取，识别市场信号中的隐藏模式。
    """

    def __init__(self, wavelet_name: str = 'db4', max_level: int = 5,
                 scales: Optional[np.ndarray] = None):
        """
        初始化小波多分辨率分析器

        Args:
            wavelet_name: 小波名称，默认'db4'
            max_level: 最大分解级别，默认5
            scales: 尺度数组，默认None
        """
        self.wavelet_name = wavelet_name
        self.max_level = max_level
        self.scales = scales or np.logspace(0, 2, 20)
        self.name = f"Wavelet Multiresolution Analyzer ({wavelet_name})"
        self.category = "signal_processing"

        # 小波处理器
        self.wavelet_processor = WaveletProcessor(wavelet_name, max_level)

    def multiresolution_decomposition(self, signal: pd.Series) -> Dict[str, pd.DataFrame]:
        """
        多分辨率分解

        Args:
            signal: 输入信号

        Returns:
            分解结果
        """
        if signal.empty:
            return {}

        # 准备信号
        clean_signal = signal.fillna(0).values

        # 小波分解
        details, approximation = self.wavelet_processor.decompose_signal(clean_signal)

        results = {}

        # 创建细节系数DataFrame
        detail_df = pd.DataFrame(index=signal.index)
        for i, detail in enumerate(details):
            # 调整长度匹配
            if len(detail) < len(signal):
                detail_padded = np.pad(detail, (0, len(signal) - len(detail)), 'edge')
            else:
                detail_padded = detail[:len(signal)]

            detail_df[f'Detail_{i+1}'] = detail_padded

        results['details'] = detail_df

        # 创建近似系数DataFrame
        if len(approximation) < len(signal):
            approx_padded = np.pad(approximation, (0, len(signal) - len(approximation)), 'edge')
        else:
            approx_padded = approximation[:len(signal)]

        results['approximation'] = pd.Series(approx_padded, index=signal.index)

        # 重构信号
        reconstructed = self.wavelet_processor.reconstruct_signal(details, approximation)
        if len(reconstructed) < len(signal):
            reconstructed_padded = np.pad(reconstructed, (0, len(signal) - len(reconstructed)), 'edge')
        else:
            reconstructed_padded = reconstructed[:len(signal)]

        results['reconstructed'] = pd.Series(reconstructed_padded, index=signal.index)

        return results

    def continuous_wavelet_analysis(self, signal: pd.Series) -> Dict[str, Any]:
        """
        连续小波分析

        Args:
            signal: 输入信号

        Returns:
            连续小波分析结果
        """
        if signal.empty:
            return {}

        results = {}
        clean_signal = signal.fillna(0).values

        # 连续小波变换
        cwt_coeffs = self.wavelet_processor.continuous_wavelet_transform(clean_signal, self.scales)

        # 创建小波系数DataFrame
        cwt_df = pd.DataFrame(cwt_coeffs.T, index=signal.index,
                             columns=[f'Scale_{i}' for i in range(len(self.scales))])
        results['cwt_coefficients'] = cwt_df

        # 计算小波能量谱
        energy_spectrum = np.abs(cwt_coeffs) ** 2
        energy_df = pd.DataFrame(energy_spectrum.T, index=signal.index,
                               columns=[f'Energy_Scale_{i}' for i in range(len(self.scales))])
        results['energy_spectrum'] = energy_df

        # 计算小波相位谱
        phase_spectrum = np.angle(cwt_coeffs)
        phase_df = pd.DataFrame(phase_spectrum.T, index=signal.index,
                              columns=[f'Phase_Scale_{i}' for i in range(len(self.scales))])
        results['phase_spectrum'] = phase_df

        # 计算尺度平均能量
        scale_avg_energy = energy_spectrum.mean(axis=1)
        results['scale_average_energy'] = pd.Series(scale_avg_energy, index=signal.index)

        # 计算主尺度
        dominant_scale_idx = np.argmax(energy_spectrum, axis=0)
        dominant_scale_values = self.scales[dominant_scale_idx]
        results['dominant_scale'] = pd.Series(dominant_scale_values, index=signal.index)

        return results

    def calculate_multiscale_features(self, decomposition: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        计算多尺度特征

        Args:
            decomposition: 分解结果

        Returns:
            多尺度特征
        """
        features = {}

        if 'details' not in decomposition:
            return features

        detail_df = decomposition['details']

        # 1. 各尺度能量
        for col in detail_df.columns:
            energy = detail_df[col] ** 2
            features[f'{col}_energy'] = energy

        # 2. 能量比率
        total_energy = sum(features[f'{col}_energy'] for col in detail_df.columns)
        if total_energy.sum() > 0:
            for col in detail_df.columns:
                energy_ratio = features[f'{col}_energy'] / (total_energy + 1e-10)
                features[f'{col}_energy_ratio'] = energy_ratio

        # 3. 尺度熵
        for col in detail_df.columns:
            entropy = self.calculate_wavelet_entropy(detail_df[col])
            features[f'{col}_entropy'] = entropy

        # 4. 尺度相关性
        if len(detail_df.columns) >= 2:
            for i in range(len(detail_df.columns) - 1):
                col1 = detail_df.columns[i]
                col2 = detail_df.columns[i + 1]
                correlation = detail_df[col1].rolling(window=20).corr(detail_df[col2])
                features[f'scale_correlation_{i+1}_{i+2}'] = correlation.fillna(0)

        # 5. 奇异性检测
        for col in detail_df.columns:
            singularity = self.detect_singularity(detail_df[col])
            features[f'{col}_singularity'] = singularity

        # 6. 尺度方差
        for col in detail_df.columns:
            variance = detail_df[col].rolling(window=10).var()
            features[f'{col}_variance'] = variance.fillna(0)

        return features

    def calculate_wavelet_entropy(self, signal: pd.Series) -> pd.Series:
        """
        计算小波熵

        Args:
            signal: 信号

        Returns:
            小波熵序列
        """
        entropy_values = []

        for i in range(len(signal)):
            # 使用滑动窗口
            start_idx = max(0, i - 20)
            end_idx = i + 1

            window_signal = signal.iloc[start_idx:end_idx]

            if len(window_signal) < 2:
                entropy_values.append(0)
                continue

            # 计算能量分布
            energy = window_signal ** 2
            total_energy = energy.sum()

            if total_energy > 0:
                probability = energy / total_energy
                # 计算熵
                entropy = -np.sum(probability * np.log(probability + 1e-10))
            else:
                entropy = 0

            entropy_values.append(entropy)

        return pd.Series(entropy_values, index=signal.index)

    def detect_singularity(self, signal: pd.Series) -> pd.Series:
        """
        检测奇异点

        Args:
            signal: 信号

        Returns:
            奇异点强度
        """
        # 计算梯度
        gradient = np.abs(np.diff(signal, prepend=signal.iloc[0]))

        # 计算曲率
        second_derivative = np.abs(np.diff(gradient, prepend=gradient[0]))

        # 奇异性指标
        singularity = gradient * second_derivative

        return pd.Series(singularity, index=signal.index).fillna(0)

    def analyze_market_regimes(self, signal: pd.Series) -> Dict[str, pd.Series]:
        """
        分析市场状态

        Args:
            signal: 输入信号

        Returns:
            市场状态分析
        """
        regime_analysis = {}

        # 多分辨率分解
        decomposition = self.multiresolution_decomposition(signal)
        if not decomposition:
            return regime_analysis

        # 计算多尺度特征
        multiscale_features = self.calculate_multiscale_features(decomposition)

        # 趋势强度（基于近似系数）
        if 'approximation' in decomposition:
            approximation = decomposition['approximation']
            trend_strength = approximation.rolling(window=20).std()
            regime_analysis['trend_strength'] = trend_strength.fillna(0)

        # 波动性（基于细节系数）
        if 'details' in decomposition:
            detail_df = decomposition['details']

            # 高频波动
            if len(detail_df.columns) > 0:
                high_freq_volatility = detail_df[detail_df.columns[0]].rolling(window=10).std()
                regime_analysis['high_frequency_volatility'] = high_freq_volatility.fillna(0)

            # 低频波动
            if len(detail_df.columns) > 2:
                low_freq_volatility = detail_df[detail_df.columns[-1]].rolling(window=20).std()
                regime_analysis['low_frequency_volatility'] = low_freq_volatility.fillna(0)

        # 噪声水平
        noise_level = self.estimate_noise_level(signal)
        regime_analysis['noise_level'] = noise_level

        # 信号复杂度
        signal_complexity = self.calculate_signal_complexity(multiscale_features)
        regime_analysis['signal_complexity'] = signal_complexity

        # 市场状态分类
        market_state = self.classify_market_regime(regime_analysis)
        regime_analysis['market_regime'] = market_state

        return regime_analysis

    def estimate_noise_level(self, signal: pd.Series) -> pd.Series:
        """
        估计噪声水平

        Args:
            signal: 信号

        Returns:
            噪声水平
        """
        # 使用最高频细节系数估计噪声
        decomposition = self.multiresolution_decomposition(signal)

        if 'details' in decomposition and len(decomposition['details'].columns) > 0:
            highest_freq_detail = decomposition['details'].iloc[:, 0]
            noise_level = highest_freq_detail.rolling(window=10).std()
        else:
            noise_level = signal.rolling(window=10).std().fillna(0)

        return noise_level

    def calculate_signal_complexity(self, multiscale_features: Dict[str, pd.Series]) -> pd.Series:
        """
        计算信号复杂度

        Args:
            multiscale_features: 多尺度特征

        Returns:
            信号复杂度
        """
        complexity = pd.Series(0.0, index=next(iter(multiscale_features.values())).index)

        # 基于能量分布计算复杂度
        energy_ratios = []
        for key, value in multiscale_features.items():
            if 'energy_ratio' in key:
                energy_ratios.append(value.fillna(0))

        if energy_ratios:
            # 计算能量分布的熵
            energy_matrix = np.column_stack(energy_ratios)
            energy_entropy = np.zeros(len(energy_matrix))

            for i in range(len(energy_matrix)):
                row = energy_matrix[i]
                if np.sum(row) > 0:
                    row = row / np.sum(row)
                    energy_entropy[i] = -np.sum(row * np.log(row + 1e-10))

            complexity = pd.Series(energy_entropy, index=complexity.index)

        return complexity

    def classify_market_regime(self, regime_analysis: Dict[str, pd.Series]) -> pd.Series:
        """
        分类市场状态

        Args:
            regime_analysis: 状态分析

        Returns:
            市场状态分类
        """
        # 获取第一个特征的索引
        if not regime_analysis:
            return pd.Series()

        index = next(iter(regime_analysis.values())).index
        regime_class = pd.Series('normal', index=index)

        # 趋势强度
        trend_strength = regime_analysis.get('trend_strength', pd.Series(0, index=index))
        trend_threshold = trend_strength.quantile(0.8)

        # 高频波动
        high_freq_vol = regime_analysis.get('high_frequency_volatility', pd.Series(0, index=index))
        high_vol_threshold = high_freq_vol.quantile(0.8)

        # 低频波动
        low_freq_vol = regime_analysis.get('low_frequency_volatility', pd.Series(0, index=index))
        low_vol_threshold = low_freq_vol.quantile(0.8)

        # 噪声水平
        noise_level = regime_analysis.get('noise_level', pd.Series(0, index=index))
        noise_threshold = noise_level.quantile(0.8)

        # 信号复杂度
        complexity = regime_analysis.get('signal_complexity', pd.Series(0, index=index))
        complexity_threshold = complexity.quantile(0.8)

        # 分类逻辑
        strong_trending = trend_strength > trend_threshold
        high_volatility = high_freq_vol > high_vol_threshold
        low_volatility = low_freq_vol < low_freq_vol.quantile(0.2)
        high_noise = noise_level > noise_threshold
        high_complexity = complexity > complexity_threshold

        regime_class[strong_trending & low_volatility] = 'trending'
        regime_class[strong_trending & high_volatility] = 'volatile_trend'
        regime_class[high_volatility & high_noise] = 'chaotic'
        regime_class[low_volatility & ~high_noise] = 'stable'
        regime_class[high_complexity] = 'complex'

        return regime_class

    def denoise_signal(self, signal: pd.Series, threshold_method: str = 'universal') -> pd.Series:
        """
        信号去噪

        Args:
            signal: 输入信号
            threshold_method: 阈值方法，默认'universal'

        Returns:
            去噪后信号
        """
        # 小波分解
        decomposition = self.multiresolution_decomposition(signal)
        if not decomposition or 'details' not in decomposition:
            return signal

        detail_df = decomposition['details'].copy()

        # 计算阈值
        if threshold_method == 'universal':
            # 通用阈值
            sigma = np.median(np.abs(detail_df.iloc[:, 0])) / 0.6745  # 基于最高频细节
            threshold = sigma * np.sqrt(2 * np.log(len(signal)))
        elif threshold_method == 'adaptive':
            # 自适应阈值
            threshold = np.std(detail_df.iloc[:, 0]) * 2
        else:
            threshold = np.std(detail_df.iloc[:, 0])

        # 软阈值去噪
        for col in detail_df.columns:
            detail = detail_df[col]
            # 软阈值
            thresholded = np.sign(detail) * np.maximum(np.abs(detail) - threshold, 0)
            detail_df[col] = thresholded

        # 重构信号
        denoised = self.wavelet_processor.reconstruct_signal(
            [detail_df[col].values for col in detail_df.columns],
            decomposition['approximation'].values
        )

        # 调整长度
        if len(denoised) < len(signal):
            denoised_padded = np.pad(denoised, (0, len(signal) - len(denoised)), 'edge')
        else:
            denoised_padded = denoised[:len(signal)]

        return pd.Series(denoised_padded, index=signal.index)

    def detect_market_cycles(self, signal: pd.Series) -> Dict[str, pd.Series]:
        """
        检测市场周期

        Args:
            signal: 输入信号

        Returns:
            市场周期检测
        """
        cycle_detection = {}

        # 连续小波分析
        cwt_results = self.continuous_wavelet_analysis(signal)
        if not cwt_results:
            return cycle_detection

        # 主周期分析
        if 'dominant_scale' in cwt_results:
            dominant_scale = cwt_results['dominant_scale']
            cycle_detection['dominant_scale'] = dominant_scale

            # 转换为周期长度
            dominant_period = dominant_scale * 2  # 近似转换
            cycle_detection['dominant_period'] = dominant_period

        # 尺度能量分析
        if 'scale_average_energy' in cwt_results:
            scale_energy = cwt_results['scale_average_energy']
            cycle_detection['scale_energy'] = scale_energy

            # 周期强度
            energy_strength = scale_energy / scale_energy.rolling(window=20).mean()
            cycle_detection['cycle_strength'] = energy_strength.fillna(0)

        # 周期稳定性
        if 'dominant_scale' in cwt_results:
            scale_stability = 1 / (cwt_results['dominant_scale'].rolling(window=20).std() + 1e-10)
            cycle_detection['cycle_stability'] = scale_stability.fillna(0)

        return cycle_detection

    def analyze_market_with_wavelets(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        使用小波分析市场

        Args:
            data: 市场数据

        Returns:
            小波分析结果
        """
        results = {}

        # 价格信号分析
        if 'close' in data.columns:
            price_signal = data['close']

            # 多分辨率分解
            decomposition = self.multiresolution_decomposition(price_signal)
            results.update(decomposition)

            # 连续小波分析
            cwt_results = self.continuous_wavelet_analysis(price_signal)
            results.update(cwt_results)

            # 多尺度特征
            multiscale_features = self.calculate_multiscale_features(decomposition)
            results.update(multiscale_features)

            # 市场状态分析
            regime_analysis = self.analyze_market_regimes(price_signal)
            results.update(regime_analysis)

            # 信号去噪
            denoised_signal = self.denoise_signal(price_signal)
            results['denoised_signal'] = denoised_signal

            # 市场周期检测
            cycle_detection = self.detect_market_cycles(price_signal)
            results.update(cycle_detection)

        # 成交量信号分析（如果可用）
        if 'volume' in data.columns:
            volume_signal = data['volume']
            volume_decomposition = self.multiresolution_decomposition(volume_signal)
            volume_features = self.calculate_multiscale_features(volume_decomposition)

            # 重命名以区分价格和成交量特征
            for key, value in volume_features.items():
                results[f'volume_{key}'] = value

        # 生成交易信号
        results['signals'] = self.generate_wavelet_signals(results)

        return results

    def generate_wavelet_signals(self, wavelet_analysis: Dict[str, Any]) -> pd.Series:
        """
        基于小波分析生成交易信号

        Args:
            wavelet_analysis: 小波分析结果

        Returns:
            交易信号
        """
        # 找到一个指标序列作为索引
        index_keys = [k for k, v in wavelet_analysis.items() if isinstance(v, pd.Series) and not v.empty]
        if not index_keys:
            return pd.Series()

        index = wavelet_analysis[index_keys[0]].index
        signals = pd.Series(0, index=index)

        # 趋势信号
        trend_strength = wavelet_analysis.get('trend_strength', pd.Series(0, index=index))
        strong_trend = trend_strength > trend_strength.quantile(0.8)

        # 波动性信号
        high_freq_vol = wavelet_analysis.get('high_frequency_volatility', pd.Series(0, index=index))
        low_freq_vol = wavelet_analysis.get('low_frequency_volatility', pd.Series(0, index=index))
        high_volatility = high_freq_vol > high_freq_vol.quantile(0.8)

        # 周期信号
        cycle_strength = wavelet_analysis.get('cycle_strength', pd.Series(0, index=index))
        strong_cycle = cycle_strength > cycle_strength.quantile(0.8)

        # 噪声信号
        noise_level = wavelet_analysis.get('noise_level', pd.Series(0, index=index))
        low_noise = noise_level < noise_level.quantile(0.2)

        # 市场状态信号
        market_regime = wavelet_analysis.get('market_regime', pd.Series('normal', index=index))

        # 信号逻辑
        # 买入信号：强趋势 + 低波动 + 强周期
        buy_signal = strong_trend & ~high_volatility & strong_cycle & low_noise

        # 卖出信号：高波动 + 高噪声
        sell_signal = high_volatility & (noise_level > noise_level.quantile(0.8))

        # 趋势跟踪信号
        trend_buy = (market_regime == 'trending') & low_noise
        trend_sell = (market_regime == 'volatile_trend') & high_volatility

        # 均值回归信号
        mean_reversion_buy = (market_regime == 'stable') & ~strong_trend
        mean_reversion_sell = (market_regime == 'chaotic')

        # 分配信号值
        signals[buy_signal] = 3
        signals[mean_reversion_buy] = 2
        signals[trend_buy] = 2
        signals[sell_signal] = -3
        signals[mean_reversion_sell] = -2
        signals[trend_sell] = -2

        return signals

    def get_trading_recommendations(self, wavelet_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            wavelet_analysis: 小波分析结果

        Returns:
            交易建议DataFrame
        """
        # 找到一个指标序列作为索引
        index_keys = [k for k, v in wavelet_analysis.items() if isinstance(v, pd.Series) and not v.empty]
        if not index_keys:
            return pd.DataFrame()

        index = wavelet_analysis[index_keys[0]].index
        recommendations = pd.DataFrame(index=index)

        # 添加小波分析指标
        for key, value in wavelet_analysis.items():
            if isinstance(value, pd.Series):
                recommendations[key] = value

        # 交易信号
        recommendations['signals'] = wavelet_analysis.get('signals', pd.Series(0, index=index))

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 强趋势低噪声信号',
            2: '买入 - 趋势或稳定信号',
            0: '持有 - 中性信号',
            -2: '卖出 - 高波动或混沌信号',
            -3: '强烈卖出 - 高噪声高风险信号'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 市场状态
        market_regime = recommendations.get('market_regime', pd.Series('normal', index=index))
        recommendations['market_state'] = market_regime

        # 周期特征
        if 'dominant_period' in recommendations.columns:
            period_strength = recommendations.get('cycle_strength', pd.Series(0, index=index))
            recommendations['cycle_confidence'] = period_strength.fillna(0)

        # 仓位建议
        position_map = {3: 0.5, 2: 0.3, 0: 0.2, -2: 0.1, -3: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 噪声调整
        noise_level = recommendations.get('noise_level', pd.Series(0, index=index))
        noise_adjustment = 1 - noise_level / (noise_level.max() + 1e-10)
        recommendations['position_size'] *= noise_adjustment

        # 趋势强度调整
        trend_strength = recommendations.get('trend_strength', pd.Series(0, index=index))
        trend_adjustment = trend_strength / (trend_strength.max() + 1e-10)
        recommendations['position_size'] *= trend_adjustment

        # 确保仓位在合理范围内
        recommendations['position_size'] = recommendations['position_size'].clip(0, 1)

        # 策略建议
        def get_wavelet_strategy(market_state, signal_strength):
            if market_state == 'trending':
                return 'trend_following'  # 趋势跟踪
            elif market_state == 'stable':
                return 'mean_reversion'  # 均值回归
            elif market_state == 'volatile_trend':
                return 'momentum'  # 动量策略
            elif market_state == 'chaotic':
                return 'avoid_trading'  # 避免交易
            else:
                return 'balanced'

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_wavelet_strategy(x['market_state'], abs(x['signals'])),
            axis=1
        )

        # 小波风险评估
        def assess_wavelet_risk(row):
            risk_score = 0

            # 噪声风险
            noise_level = row.get('noise_level', 0)
            if noise_level > noise_level.quantile(0.8):
                risk_score += 0.3

            # 波动性风险
            high_freq_vol = row.get('high_frequency_volatility', 0)
            if high_freq_vol > high_freq_vol.quantile(0.8):
                risk_score += 0.3

            # 复杂性风险
            complexity = row.get('signal_complexity', 0)
            if complexity > complexity.quantile(0.8):
                risk_score += 0.2

            # 趋势风险
            trend_strength = row.get('trend_strength', 0)
            if trend_strength > trend_strength.quantile(0.9):
                risk_score += 0.2

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['risk_level'] = recommendations.apply(assess_wavelet_risk, axis=1)

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成模拟市场数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')

    # 生成多尺度信号
    t = np.arange(300)
    signal = (
        2 * np.sin(2 * np.pi * t / 100) +    # 长期趋势
        0.5 * np.sin(2 * np.pi * t / 20) +  # 中期周期
        0.2 * np.sin(2 * np.pi * t / 5) +   # 短期波动
        0.1 * np.random.randn(300)          # 噪声
    )

    # 转换为价格序列
    base_price = 100
    prices = base_price * np.exp(np.cumsum(signal * 0.005))

    # 生成成交量
    volume = 1000000 + 300000 * np.sin(2 * np.pi * t / 30) + 100000 * np.random.randn(300)

    # 创建数据框
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices,
        'volume': volume
    }).set_index('date')

    # 创建小波多分辨率分析器
    wma = WaveletMultiresolutionAnalyzer(wavelet_name='db4', max_level=4)

    # 分析市场
    print("使用小波分析市场...")
    wavelet_results = wma.analyze_market_with_wavelets(market_data)

    # 显示主要结果
    print(f"趋势强度: {wavelet_results['trend_strength'].iloc[-1]:.3f}")
    print(f"高频波动: {wavelet_results['high_frequency_volatility'].iloc[-1]:.3f}")
    print(f"低频波动: {wavelet_results['low_frequency_volatility'].iloc[-1]:.3f}")
    print(f"噪声水平: {wavelet_results['noise_level'].iloc[-1]:.3f}")
    print(f"主周期: {wavelet_results.get('dominant_period', pd.Series(0)).iloc[-1]:.2f}")
    print(f"周期强度: {wavelet_results.get('cycle_strength', pd.Series(0)).iloc[-1]:.3f}")
    print(f"市场状态: {wavelet_results['market_regime'].iloc[-1]}")
    print(f"当前信号: {wavelet_results['signals'].iloc[-1]}")

    # 显示分解信息
    if 'details' in wavelet_results:
        print(f"\n细节系数层数: {wavelet_results['details'].shape[1]}")

    # 获取交易建议
    recommendations = wma.get_trading_recommendations(wavelet_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['trend_strength', 'market_state', 'signal_description',
                                          'strategy', 'position_size', 'risk_level']])