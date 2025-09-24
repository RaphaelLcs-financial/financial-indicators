"""
Lyapunov Market Predictor
========================

基于李雅普诺夫指数的混沌理论市场预测器，用于分析
市场的混沌特性和可预测性。李雅普诺夫指数可以量化
系统对初始条件的敏感依赖性，帮助识别市场的混沌状态
和预测能力边界。

特点：
- 混沌理论分析
- 李雅普诺夫指数计算
- 市场可预测性评估
- 相空间重构
- 奇异吸引子分析
- 混沌控制策略
- 非线性动力学特征

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.spatial.distance import cdist
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


class LyapunovMarketPredictor:
    """
    李雅普诺夫市场预测器

    使用混沌理论中的李雅普诺夫指数来分析市场的
    混沌特性，并提供基于非线性动力学的市场预测。
    """

    def __init__(self, embedding_dimension: int = 3, time_delay: int = 2,
                 prediction_horizon: int = 5, n_neighbors: int = 10):
        """
        初始化李雅普诺夫市场预测器

        Args:
            embedding_dimension: 嵌入维度，默认3
            time_delay: 时间延迟，默认2
            prediction_horizon: 预测时间范围，默认5
            n_neighbors: 最近邻数量，默认10
        """
        self.embedding_dimension = embedding_dimension
        self.time_delay = time_delay
        self.prediction_horizon = prediction_horizon
        self.n_neighbors = n_neighbors
        self.name = f"Lyapunov Market Predictor ({embedding_dimension})"
        self.category = "chaos"

        # 计算参数
        self.lyapunov_exponent = None
        self.is_chaotic = None
        self.predictability = None

    def reconstruct_phase_space(self, time_series: pd.Series) -> np.ndarray:
        """
        重构相空间

        Args:
            time_series: 时间序列数据

        Returns:
            相空间矩阵
        """
        n = len(time_series)
        m = self.embedding_dimension
        tau = self.time_delay

        # 计算重构后的相空间点数
        n_points = n - (m - 1) * tau

        if n_points <= 0:
            return np.array([])

        # 重构相空间
        phase_space = np.zeros((n_points, m))
        for i in range(m):
            phase_space[:, i] = time_series.iloc[i*tau : i*tau + n_points].values

        return phase_space

    def calculate_lyapunov_exponent(self, time_series: pd.Series) -> float:
        """
        计算最大李雅普诺夫指数

        Args:
            time_series: 时间序列数据

        Returns:
            最大李雅普诺夫指数
        """
        # 重构相空间
        phase_space = self.reconstruct_phase_space(time_series)
        if len(phase_space) == 0:
            return 0.0

        n_points = len(phase_space)
        if n_points < 2 * self.n_neighbors:
            return 0.0

        # 找到每个点的最近邻
        nbrs = NearestNeighbors(n_neighbors=self.n_neighbors+1).fit(phase_space)
        distances, indices = nbrs.kneighbors(phase_space)

        # 计算邻居距离演化
        lyapunov_rates = []

        for i in range(min(n_points - self.prediction_horizon, 100)):
            current_point = phase_space[i]
            neighbors = phase_space[indices[i, 1:]]  # 排除自己

            # 计算初始距离
            initial_distances = np.linalg.norm(neighbors - current_point, axis=1)

            # 跳过距离为0的邻居
            valid_mask = initial_distances > 1e-10
            if not np.any(valid_mask):
                continue

            valid_neighbors = neighbors[valid_mask]
            valid_distances = initial_distances[valid_mask]

            # 预测邻居的未来位置
            future_distances = []
            for j in range(len(valid_neighbors)):
                if i + self.prediction_horizon < n_points:
                    future_point = phase_space[i + self.prediction_horizon]
                    neighbor_future = valid_neighbors[j]  # 简化假设

                    future_dist = np.linalg.norm(future_point - neighbor_future)
                    if future_dist > 1e-10:
                        future_distances.append(future_dist / valid_distances[j])

            if future_distances:
                # 计算平均发散率
                avg_divergence = np.mean(np.log(future_distances))
                lyapunov_rates.append(avg_divergence / self.prediction_horizon)

        if lyapunov_rates:
            lyapunov_exponent = np.mean(lyapunov_rates)
        else:
            lyapunov_exponent = 0.0

        return lyapunov_exponent

    def analyze_chaos_characteristics(self, time_series: pd.Series) -> Dict[str, Any]:
        """
        分析混沌特征

        Args:
            time_series: 时间序列数据

        Returns:
            混沌特征分析结果
        """
        # 计算李雅普诺夫指数
        lyapunov_exp = self.calculate_lyapunov_exponent(time_series)

        # 判断混沌状态
        self.is_chaotic = lyapunov_exp > 0.01  # 正李雅普诺夫指数表示混沌

        # 计算可预测性
        self.predictability = max(0, 1 - abs(lyapunov_exp) * 10)

        # 计算分形维度
        correlation_dimension = self.calculate_correlation_dimension(time_series)

        # 计算熵
        time_series_entropy = self.calculate_time_series_entropy(time_series)

        # 计算吸引子特征
        attractor_features = self.analyze_attractor(time_series)

        return {
            'lyapunov_exponent': lyapunov_exp,
            'is_chaotic': self.is_chaotic,
            'predictability': self.predictability,
            'correlation_dimension': correlation_dimension,
            'entropy': time_series_entropy,
            'attractor_features': attractor_features
        }

    def calculate_correlation_dimension(self, time_series: pd.Series) -> float:
        """
        计算关联维度

        Args:
            time_series: 时间序列数据

        Returns:
            关联维度
        """
        phase_space = self.reconstruct_phase_space(time_series)
        if len(phase_space) < 10:
            return 1.0

        # 计算点间距离
        distances = cdist(phase_space, phase_space)

        # 计算关联积分
        epsilon_values = np.logspace(-3, 0, 20)
        correlation_integrals = []

        for epsilon in epsilon_values:
            correlation_integral = np.sum(distances < epsilon) / (len(phase_space) ** 2)
            correlation_integrals.append(correlation_integral)

        # 计算关联维度（斜率）
        if len(correlation_integrals) > 1:
            log_epsilon = np.log(epsilon_values)
            log_c = np.log(correlation_integrals)

            # 移除无效值
            valid_mask = (log_c > -np.inf) & (log_c < np.inf)
            if np.sum(valid_mask) > 1:
                correlation_dim = np.polyfit(log_epsilon[valid_mask], log_c[valid_mask], 1)[0]
                return max(0.5, min(5.0, correlation_dim))

        return 2.0

    def calculate_time_series_entropy(self, time_series: pd.Series) -> float:
        """
        计算时间序列熵

        Args:
            time_series: 时间序列数据

        Returns:
            熵值
        """
        # 离散化时间序列
        n_bins = min(20, len(time_series) // 10)
        if n_bins < 2:
            return 0.0

        hist, _ = np.histogram(time_series, bins=n_bins)
        prob = hist / np.sum(hist)

        # 计算香农熵
        entropy_val = -np.sum(prob * np.log(prob + 1e-10))
        return entropy_val

    def analyze_attractor(self, time_series: pd.Series) -> Dict[str, Any]:
        """
        分析奇异吸引子特征

        Args:
            time_series: 时间序列数据

        Returns:
            吸引子特征
        """
        phase_space = self.reconstruct_phase_space(time_series)
        if len(phase_space) < 10:
            return {'type': 'unknown', 'stability': 0.5}

        # 使用PCA降维分析
        if phase_space.shape[1] > 2:
            pca = PCA(n_components=2)
            reduced_space = pca.fit_transform(phase_space)
        else:
            reduced_space = phase_space[:, :2]

        # 计算吸引子特征
        center = np.mean(reduced_space, axis=0)
        radius = np.mean(np.linalg.norm(reduced_space - center, axis=1))

        # 判断吸引子类型
        if radius < np.std(time_series) * 0.1:
            attractor_type = 'fixed_point'
        elif radius > np.std(time_series) * 2:
            attractor_type = 'strange'
        else:
            attractor_type = 'limit_cycle'

        # 计算稳定性
        stability = 1 / (1 + radius / np.std(time_series))

        return {
            'type': attractor_type,
            'center': center,
            'radius': radius,
            'stability': stability
        }

    def predict_chaos_based_trend(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        基于混沌理论预测趋势

        Args:
            data: OHLCV数据

        Returns:
            混沌预测结果
        """
        close_prices = data['close']

        # 分析混沌特征
        chaos_analysis = self.analyze_chaos_characteristics(close_prices)

        # 生成混沌预测
        predictions = self.generate_chaos_predictions(close_prices, chaos_analysis)

        # 计算混沌强度
        chaos_intensity = self.calculate_chaos_intensity(close_prices)

        # 生成交易信号
        signals = self.generate_chaos_signals(predictions, chaos_analysis, chaos_intensity)

        # 计算预测置信度
        confidence = self.calculate_chaos_confidence(chaos_analysis, predictions)

        return {
            'lyapunov_exponent': pd.Series([chaos_analysis['lyapunov_exponent']] * len(data), index=data.index),
            'is_chaotic': pd.Series([chaos_analysis['is_chaotic']] * len(data), index=data.index),
            'predictability': pd.Series([chaos_analysis['predictability']] * len(data), index=data.index),
            'chaos_intensity': chaos_intensity,
            'predictions': predictions,
            'signals': signals,
            'confidence': confidence
        }

    def generate_chaos_predictions(self, time_series: pd.Series, chaos_analysis: Dict[str, Any]) -> pd.Series:
        """
        生成混沌预测

        Args:
            time_series: 时间序列
            chaos_analysis: 混沌分析结果

        Returns:
            预测序列
        """
        n = len(time_series)
        predictions = pd.Series(0.0, index=time_series.index)

        if n < self.embedding_dimension * self.time_delay:
            return predictions

        # 重构相空间
        phase_space = self.reconstruct_phase_space(time_series)

        if len(phase_space) < 2:
            return predictions

        # 使用局部线性预测
        for i in range(len(phase_space) - self.prediction_horizon):
            current_point = phase_space[i]

            # 找到最近邻
            distances = np.linalg.norm(phase_space[:i] - current_point, axis=1)
            if len(distances) > 0:
                nearest_idx = np.argmin(distances)

                # 使用最近邻的未来值作为预测
                if nearest_idx + self.prediction_horizon < len(phase_space):
                    future_change = phase_space[nearest_idx + self.prediction_horizon] - phase_space[nearest_idx]
                    predicted_change = future_change * chaos_analysis['predictability']

                    # 转换为价格预测
                    pred_index = i + self.embedding_dimension * self.time_delay
                    if pred_index < n:
                        predictions.iloc[pred_index] = predicted_change[0]  # 使用第一个维度

        return predictions

    def calculate_chaos_intensity(self, time_series: pd.Series) -> pd.Series:
        """
        计算混沌强度

        Args:
            time_series: 时间序列

        Returns:
            混沌强度序列
        """
        # 计算价格变化的非线性程度
        returns = time_series.pct_change()
        volatility = returns.rolling(window=20).std()

        # 计算趋势变化频率
        trend_changes = (np.sign(returns).diff().abs() == 2).astype(int)
        trend_change_frequency = trend_changes.rolling(window=20).mean()

        # 计算混沌强度
        chaos_intensity = volatility * trend_change_frequency * 100

        return chaos_intensity.fillna(0)

    def generate_chaos_signals(self, predictions: pd.Series, chaos_analysis: Dict[str, Any], chaos_intensity: pd.Series) -> pd.Series:
        """
        生成混沌交易信号

        Args:
            predictions: 预测序列
            chaos_analysis: 混沌分析
            chaos_intensity: 混沌强度

        Returns:
            交易信号
        """
        signals = pd.Series(0, index=predictions.index)

        # 根据混沌状态调整策略
        if chaos_analysis['is_chaotic']:
            # 混沌市场：短期交易，快进快出
            threshold = 0.001
            position_size_factor = 0.3
        else:
            # 非混沌市场：趋势跟踪
            threshold = 0.002
            position_size_factor = 0.6

        # 生成信号
        strong_buy = (predictions > threshold * 2) & (chaos_intensity < 50)
        buy = (predictions > threshold) & (chaos_intensity < 70)
        strong_sell = (predictions < -threshold * 2) & (chaos_intensity < 50)
        sell = (predictions < -threshold) & (chaos_intensity < 70)

        signals[strong_buy] = 2 * position_size_factor
        signals[buy] = 1 * position_size_factor
        signals[strong_sell] = -2 * position_size_factor
        signals[sell] = -1 * position_size_factor

        return signals

    def calculate_chaos_confidence(self, chaos_analysis: Dict[str, Any], predictions: pd.Series) -> pd.Series:
        """
        计算混沌预测置信度

        Args:
            chaos_analysis: 混沌分析
            predictions: 预测序列

        Returns:
            置信度序列
        """
        base_confidence = chaos_analysis['predictability']

        # 根据混沌强度调整置信度
        n = len(predictions)
        confidence = pd.Series(base_confidence, index=predictions.index)

        # 混沌强度越高，置信度越低
        if 'chaos_intensity' in locals():
            intensity_adjustment = 1 / (1 + chaos_intensity / 100)
            confidence *= intensity_adjustment

        return confidence.fillna(0.1)

    def get_trading_recommendations(self, chaos_predictions: Dict[str, pd.Series]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            chaos_predictions: 混沌预测结果

        Returns:
            交易建议DataFrame
        """
        recommendations = pd.DataFrame(index=chaos_predictions['predictions'].index)

        # 混沌特征
        recommendations['lyapunov_exponent'] = chaos_predictions['lyapunov_exponent']
        recommendations['is_chaotic'] = chaos_predictions['is_chaotic']
        recommendations['predictability'] = chaos_predictions['predictability']
        recommendations['chaos_intensity'] = chaos_predictions['chaos_intensity']

        # 预测信息
        recommendations['predictions'] = chaos_predictions['predictions']
        recommendations['signals'] = chaos_predictions['signals']
        recommendations['confidence'] = chaos_predictions['confidence']

        # 信号描述
        signal_descriptions = {
            0.6: '强烈买入 - 混沌预测确认',
            0.3: '买入 - 混沌预测正向',
            0.0: '持有 - 预测不明确',
            -0.3: '卖出 - 混沌预测负向',
            -0.6: '强烈卖出 - 混沌预测确认'
        }

        def map_signal_description(signal):
            if signal >= 0.5:
                return signal_descriptions[0.6]
            elif signal > 0.1:
                return signal_descriptions[0.3]
            elif signal > -0.1:
                return signal_descriptions[0.0]
            elif signal > -0.5:
                return signal_descriptions[-0.3]
            else:
                return signal_descriptions[-0.6]

        recommendations['signal_description'] = recommendations['signals'].apply(map_signal_description)

        # 仓位建议
        recommendations['position_size'] = abs(recommendations['signals']) * 0.5

        # 置信度调整
        confidence_adjustment = recommendations['confidence']
        recommendations['position_size'] *= confidence_adjustment

        # 策略建议
        def get_strategy(is_chaotic, chaos_intensity):
            if is_chaotic:
                if chaos_intensity > 70:
                    return 'avoid_trading'
                elif chaos_intensity > 40:
                    return 'short_term_scalping'
                else:
                    return 'mean_reversion'
            else:
                if chaos_intensity < 30:
                    return 'trend_following'
                else:
                    return 'balanced'

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_strategy(x['is_chaotic'], x['chaos_intensity']),
            axis=1
        )

        # 风险等级
        recommendations['risk_level'] = recommendations.apply(
            lambda x: 'high' if x['is_chaotic'] and x['chaos_intensity'] > 60 else
                     'medium' if x['chaos_intensity'] > 40 else 'low',
            axis=1
        )

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2023-12-31')
    n = len(dates)

    # 生成具有混沌特征的价格序列
    close_prices = 100 + np.cumsum(np.random.randn(n) * 0.5)

    # 添加一些非线性成分
    for i in range(2, n):
        close_prices[i] += 0.1 * np.sin(close_prices[i-1] / 10) * close_prices[i-2]

    data = pd.DataFrame({
        'close': close_prices
    }, index=dates)

    # 创建李雅普诺夫市场预测器
    lyapunov_predictor = LyapunovMarketPredictor(
        embedding_dimension=3,
        time_delay=2,
        prediction_horizon=5
    )

    # 分析混沌特征
    print("分析市场混沌特征...")
    chaos_analysis = lyapunov_predictor.analyze_chaos_characteristics(data['close'])
    print(f"李雅普诺夫指数: {chaos_analysis['lyapunov_exponent']:.4f}")
    print(f"是否混沌: {chaos_analysis['is_chaotic']}")
    print(f"可预测性: {chaos_analysis['predictability']:.2f}")
    print(f"关联维度: {chaos_analysis['correlation_dimension']:.2f}")
    print(f"熵: {chaos_analysis['entropy']:.2f}")
    print(f"吸引子类型: {chaos_analysis['attractor_features']['type']}")

    # 进行混沌预测
    print("\n进行混沌预测...")
    predictions = lyapunov_predictor.predict_chaos_based_trend(data)

    print(f"最新李雅普诺夫指数: {predictions['lyapunov_exponent'].iloc[-1]:.4f}")
    print(f"混沌状态: {'混沌' if predictions['is_chaotic'].iloc[-1] else '非混沌'}")
    print(f"可预测性: {predictions['predictability'].iloc[-1]:.2f}")
    print(f"混沌强度: {predictions['chaos_intensity'].iloc[-1]:.2f}")
    print(f"预测值: {predictions['predictions'].iloc[-1]:.4f}")
    print(f"当前信号: {predictions['signals'].iloc[-1]:.2f}")
    print(f"预测置信度: {predictions['confidence'].iloc[-1]:.2f}")

    # 获取交易建议
    recommendations = lyapunov_predictor.get_trading_recommendations(predictions)
    print("\n最新交易建议:")
    print(recommendations.iloc[-1])