"""
Behavioral Pattern Recognizer
=============================

基于行为金融学的模式识别指标，识别市场中的非理性
行为模式和投资者心理偏差。该指标通过分析价格行为、
成交模式和市场情绪来识别经典的行为金融学模式。

特点：
- 行为模式识别
- 羊群效应检测
- 过度反应识别
- 锚定效应分析
- 损失厌恶量化
- 禀赋效应测量
- 框架效应识别
- 确认偏误检测

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Union, List, Dict, Any, Tuple, Optional
from scipy.stats import norm, zscore
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


class BehavioralPatternRecognizer:
    """
    行为模式识别器

    识别市场中的非理性行为模式和
    投资者心理偏差模式。
    """

    def __init__(self, pattern_windows: List[int] = [5, 10, 20, 50],
                 pattern_threshold: float = 2.0,
                 cluster_patterns: bool = True):
        """
        初始化行为模式识别器

        Args:
            pattern_windows: 模式识别窗口，默认[5,10,20,50]
            pattern_threshold: 模式阈值，默认2.0
            cluster_patterns: 是否聚类模式，默认True
        """
        self.pattern_windows = pattern_windows
        self.pattern_threshold = pattern_threshold
        self.cluster_patterns = cluster_patterns
        self.name = f"Behavioral Pattern Recognizer ({pattern_windows[0]})"
        self.category = "behavioral"

    def detect_herding_behavior(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测羊群效应

        Args:
            data: 市场数据

        Returns:
            羊群效应指标
        """
        herding_indicators = {}

        if 'close' not in data.columns:
            return herding_indicators

        prices = data['close']
        returns = prices.pct_change().fillna(0)

        # 1. 价格同步性（个股与大盘的同步性）
        if 'market_index' in data.columns:
            market_returns = data['market_index'].pct_change().fillna(0)

            for window in self.pattern_windows:
                # 计算滚动相关性
                correlation = returns.rolling(window=window).corr(market_returns)
                herding_indicators[f'price_synchronicity_{window}'] = correlation.fillna(0)

                # 高同步性可能表明羊群效应
                high_synchronicity = correlation > correlation.quantile(0.8)
                herding_indicators[f'high_synchronicity_{window}'] = high_synchronicity.astype(int)

        # 2. 成交量羊群效应
        if 'volume' in data.columns:
            volume = data['volume']
            volume_returns = volume.pct_change().fillna(0)

            # 成交量异常一致
            volume_zscore = zscore(volume_returns.fillna(0))
            volume_herding = abs(volume_zscore) > self.pattern_threshold
            herding_indicators['volume_herding'] = volume_herding.astype(int)

            # 成交量集中度
            volume_concentration = volume.rolling(window=20).std() / volume.rolling(window=20).mean()
            herding_indicators['volume_concentration'] = volume_concentration.fillna(0)

        # 3. 价格变化集中度
        for window in self.pattern_windows:
            # 价格变化的标准差
            returns_std = returns.rolling(window=window).std()
            returns_mean = returns.rolling(window=window).mean()

            # 变异系数
            coefficient_variation = returns_std / (abs(returns_mean) + 1e-10)
            herding_indicators[f'returns_concentration_{window}'] = coefficient_variation.fillna(0)

            # 低变异系数可能表明羊群效应
            low_variation = coefficient_variation < coefficient_variation.quantile(0.2)
            herding_indicators[f'low_variation_herding_{window}'] = low_variation.astype(int)

        return herding_indicators

    def detect_overreaction_patterns(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测过度反应模式

        Args:
            data: 市场数据

        Returns:
            过度反应指标
        """
        overreaction_indicators = {}

        if 'close' not in data.columns:
            return overreaction_indicators

        prices = data['close']
        returns = prices.pct_change().fillna(0)

        # 1. 极端价格反应
        returns_zscore = zscore(returns.fillna(0))
        extreme_moves = abs(returns_zscore) > self.pattern_threshold
        overreaction_indicators['extreme_moves'] = extreme_moves.astype(int)

        # 2. 过度反应后的反转
        for window in [3, 5, 10]:
            # 极端移动后的反转
            extreme_moves_lagged = extreme_moves.shift(window)
            subsequent_returns = returns.rolling(window=window).mean().shift(-window)

            # 反转程度
            reversal_strength = -extreme_moves_lagged.astype(float) * subsequent_returns
            overreaction_indicators[f'reversal_strength_{window}'] = reversal_strength.fillna(0)

        # 3. 连续过度反应
        consecutive_extreme = extreme_moves.rolling(window=3).sum()
        overreaction_indicators['consecutive_overreaction'] = consecutive_extreme.fillna(0)

        # 4. 成交量确认的过度反应
        if 'volume' in data.columns:
            volume = data['volume']
            volume_zscore = zscore(volume.fillna(volume.mean()))

            # 价格和成交量同时极端
            price_volume_extreme = extreme_moves & (abs(volume_zscore) > self.pattern_threshold)
            overreaction_indicators['price_volume_overreaction'] = price_volume_extreme.astype(int)

        # 5. 情绪过度反应
        sentiment_momentum = returns.rolling(window=5).mean()
        sentiment_acceleration = sentiment_momentum.diff().fillna(0)
        extreme_sentiment = abs(sentiment_acceleration) > sentiment_acceleration.quantile(0.9)
        overreaction_indicators['sentiment_overreaction'] = extreme_sentiment.astype(int)

        return overreaction_indicators

    def detect_anchoring_bias(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测锚定效应

        Args:
            data: 市场数据

        Returns:
            锚定效应指标
        """
        anchoring_indicators = {}

        if 'close' not in data.columns:
            return anchoring_indicators

        prices = data['close']

        # 1. 历史高点锚定
        for window in [20, 50, 200]:
            high_price = prices.rolling(window=window).max()
            low_price = prices.rolling(window=window).min()

            # 价格相对于高点的位置
            high_anchor_ratio = prices / high_price
            anchoring_indicators[f'high_anchor_ratio_{window}'] = high_anchor_ratio.fillna(0)

            # 高点锚定强度（接近高点时的行为）
            high_anchor_strength = (high_anchor_ratio > 0.9) & (high_anchor_ratio <= 1.0)
            anchoring_indicators[f'high_anchor_strength_{window}'] = high_anchor_strength.astype(int)

        # 2. 历史低点锚定
        for window in [20, 50, 200]:
            low_anchor_ratio = prices / low_price
            anchoring_indicators[f'low_anchor_ratio_{window}'] = low_anchor_ratio.fillna(0)

            # 低点锚定强度
            low_anchor_strength = (low_anchor_ratio > 1.0) & (low_anchor_ratio < 1.1)
            anchoring_indicators[f'low_anchor_strength_{window}'] = low_anchor_strength.astype(int)

        # 3. IPO价格锚定
        if 'ipo_price' in data.columns:
            ipo_anchor_ratio = prices / data['ipo_price']
            anchoring_indicators['ipo_anchor_ratio'] = ipo_anchor_ratio.fillna(0)

            # IPO价格锚定偏差
            ipo_anchor_deviation = abs(ipo_anchor_ratio - 1.0)
            anchoring_indicators['ipo_anchor_deviation'] = ipo_anchor_deviation.fillna(0)

        # 4. 移动平均锚定
        for window in [20, 50, 200]:
            moving_avg = prices.rolling(window=window).mean()
            ma_anchor_ratio = prices / moving_avg
            anchoring_indicators[f'ma_anchor_ratio_{window}'] = ma_anchor_ratio.fillna(0)

            # 移动平均锚定强度
            ma_anchor_strength = abs(ma_anchor_ratio - 1.0)
            anchoring_indicators[f'ma_anchor_strength_{window}'] = ma_anchor_strength.fillna(0)

        return anchoring_indicators

    def detect_loss_aversion(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测损失厌恶

        Args:
            data: 市场数据

        Returns:
            损失厌恶指标
        """
        loss_aversion_indicators = {}

        if 'close' not in data.columns:
            return loss_aversion_indicators

        prices = data['close']
        returns = prices.pct_change().fillna(0)

        # 1. 损失厌恶比率
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]

        # 滚动计算损失厌恶比率
        rolling_loss_aversion = pd.Series(0.0, index=prices.index)
        for i in range(20, len(returns)):
            window_returns = returns.iloc[i-20:i]
            pos_ret = window_returns[window_returns > 0]
            neg_ret = window_returns[window_returns < 0]

            if len(pos_ret) > 0 and len(neg_ret) > 0:
                # 损失厌恶比率 = 平均损失痛苦 / 平均收益快乐
                loss_ratio = abs(neg_ret.mean()) / pos_ret.mean()
                rolling_loss_aversion.iloc[i] = loss_ratio

        loss_aversion_indicators['loss_aversion_ratio'] = rolling_loss_aversion.fillna(0)

        # 2. 处置效应（倾向卖出盈利股票，持有亏损股票）
        if 'volume' in data.columns:
            volume = data['volume']

            # 盈利时的成交量 vs 亏损时的成交量
            profit_periods = returns > 0
            loss_periods = returns < 0

            profit_volume = volume[profit_periods]
            loss_volume = volume[loss_periods]

            # 计算处置效应强度
            if len(profit_volume) > 0 and len(loss_volume) > 0:
                disposal_effect = profit_volume.mean() / (loss_volume.mean() + 1e-10)
                loss_aversion_indicators['disposal_effect_strength'] = pd.Series(disposal_effect, index=prices.index).fillna(0)

        # 3. 损失厌恶持续性
        consecutive_losses = (returns < 0).rolling(window=5).sum()
        loss_aversion_indicators['consecutive_losses'] = consecutive_losses.fillna(0)

        # 4. 损失厌恶情绪强度
        loss_sentiment = -returns.copy()
        loss_sentiment[loss_sentiment < 0] *= 2.0  # 损失的痛苦是收益的2倍
        loss_aversion_indicators['loss_sentiment_intensity'] = loss_sentiment.fillna(0)

        return loss_aversion_indicators

    def detect_overconfidence_bias(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测过度自信偏差

        Args:
            data: 市场数据

        Returns:
            过度自信指标
        """
        overconfidence_indicators = {}

        if 'close' not in data.columns:
            return overconfidence_indicators

        prices = data['close']
        returns = prices.pct_change().fillna(0)

        # 1. 交易频率过度自信
        if 'volume' in data.columns:
            volume = data['volume']

            # 成交量异常增加
            volume_zscore = zscore(volume.fillna(volume.mean()))
            high_volume = volume_zscore > 1.5
            overconfidence_indicators['high_volume_confidence'] = high_volume.astype(int)

            # 交易频率（成交量变化）
            volume_frequency = volume.pct_change().rolling(window=5).std()
            overconfidence_indicators['trading_frequency'] = volume_frequency.fillna(0)

        # 2. 预测过度自信
        # 价格趋势持续性（过度自信认为趋势会持续）
        trend_persistence = returns.rolling(window=5).apply(lambda x: np.sum(x > 0) / len(x))
        overconfidence_indicators['trend_persistence'] = trend_persistence.fillna(0)

        # 3. 波动率低估（过度自信时低估风险）
        realized_volatility = returns.rolling(window=20).std()
        implied_volatility = realized_volatility.rolling(window=5).mean()  # 简化的隐含波动率

        volatility_mispricing = realized_volatility / (implied_volatility + 1e-10)
        overconfidence_indicators['volatility_mispricing'] = volatility_mispricing.fillna(0)

        # 4. 过度自信的价格表现
        price_momentum = returns.rolling(window=10).mean()
        overconfidence_price = price_momentum * (1 + abs(returns))  # 过度自信放大价格影响
        overconfidence_indicators['overconfidence_price_impact'] = overconfidence_price.fillna(0)

        return overconfidence_indicators

    def detect_frame_dependence(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测框架依赖（框架效应）

        Args:
            data: 市场数据

        Returns:
            框架依赖指标
        """
        frame_indicators = {}

        if 'close' not in data.columns:
            return frame_indicators

        prices = data['close']
        returns = returns = prices.pct_change().fillna(0)

        # 1. 信息框架效应
        # 好消息vs坏消息的框架差异
        positive_news = returns > 0
        negative_news = returns < 0

        # 正面信息的反应强度
        positive_reaction = returns[positive_news].rolling(window=10).mean()
        # 负面信息的反应强度
        negative_reaction = returns[negative_news].rolling(window=10).mean()

        # 框架效应强度
        frame_effect = abs(positive_reaction.fillna(0) - negative_reaction.fillna(0))
        frame_indicators['frame_effect_strength'] = frame_effect.fillna(0)

        # 2. 参考点依赖
        # 不同参考点下的决策差异
        reference_points = [
            prices.rolling(window=20).mean(),  # 短期均线
            prices.rolling(window=50).mean(),  # 中期均线
            prices.rolling(window=200).mean()  # 长期均线
        ]

        for i, ref_point in enumerate(reference_points):
            relative_price = prices / ref_point
            frame_indicators[f'relative_price_frame_{i}'] = relative_price.fillna(0)

        # 3. 时间框架效应
        # 不同时间框架下的决策差异
        for window in [5, 10, 20]:
            short_term_trend = returns.rolling(window=window).mean()
            long_term_trend = returns.rolling(window=window*4).mean()

            # 时间框架不一致性
            frame_inconsistency = abs(short_term_trend - long_term_trend)
            frame_indicators[f'time_frame_inconsistency_{window}'] = frame_inconsistency.fillna(0)

        return frame_indicators

    def detect_mental_accounting(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        检测心理账户

        Args:
            data: 市场数据

        Returns:
            心理账户指标
        """
        mental_accounting_indicators = {}

        if 'close' not in data.columns:
            return mental_accounting_indicators

        prices = data['close']
        returns = prices.pct_change().fillna(0)

        # 1. 投资组合分离效应
        # 不同风险资产的分离管理
        if 'volume' in data.columns:
            volume = data['volume']

            # 成交量的分离管理（不同规模交易）
            volume_clusters = self.cluster_volume_patterns(volume)
            mental_accounting_indicators['volume_separation'] = volume_clusters.fillna(0)

        # 2. 赢利/亏损账户分离
        paper_gains = prices > prices.rolling(window=50).max() * 0.9  # 浮盈
        paper_losses = prices < prices.rolling(window=50).min() * 1.1  # 浮亏

        mental_accounting_indicators['paper_gains_account'] = paper_gains.astype(int)
        mental_accounting_indicators['paper_losses_account'] = paper_losses.astype(int)

        # 3. 收入/本金账户分离
        # 价格相对于成本价的处理
        if 'cost_basis' in data.columns:
            capital_account = prices > data['cost_basis']  # 本金账户
            income_account = prices <= data['cost_basis']  # 收入账户

            mental_accounting_indicators['capital_account'] = capital_account.astype(int)
            mental_accounting_indicators['income_account'] = income_account.astype(int)

        # 4. 心理账户风险偏好差异
        # 在盈利账户和亏损账户的风险偏好差异
        returns_gains = returns[paper_gains]
        returns_losses = returns[paper_losses]

        if len(returns_gains) > 0 and len(returns_losses) > 0:
            risk_preference_diff = returns_gains.std() - returns_losses.std()
            mental_accounting_indicators['risk_preference_difference'] = pd.Series(risk_preference_diff, index=prices.index).fillna(0)

        return mental_accounting_indicators

    def cluster_volume_patterns(self, volume: pd.Series) -> pd.Series:
        """
        聚类成交量模式

        Args:
            volume: 成交量序列

        Returns:
            成交量聚类标签
        """
        if len(volume) < 10:
            return pd.Series(0, index=volume.index)

        # 准备特征
        features = []
        for i in range(10, len(volume)):
            window_volume = volume.iloc[i-10:i]
            features.append([
                window_volume.mean(),
                window_volume.std(),
                window_volume.max(),
                window_volume.min(),
                window_volume.pct_change().std()
            ])

        features = np.array(features)

        # K-means聚类
        n_clusters = min(3, len(features))
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(features)

            # 创建完整序列
            full_labels = np.zeros(len(volume))
            full_labels[10:] = cluster_labels

            return pd.Series(full_labels, index=volume.index)
        else:
            return pd.Series(0, index=volume.index)

    def calculate_behavioral_pattern_score(self, behavioral_indicators: Dict[str, pd.Series]) -> pd.Series:
        """
        计算行为模式得分

        Args:
            behavioral_indicators: 行为指标

        Returns:
            行为模式得分
        """
        if not behavioral_indicators:
            return pd.Series()

        # 找到共同的索引
        common_index = None
        for indicator in behavioral_indicators.values():
            if not indicator.empty:
                if common_index is None:
                    common_index = indicator.index
                else:
                    common_index = common_index.intersection(indicator.index)

        if common_index is None or len(common_index) == 0:
            return pd.Series()

        # 计算综合行为得分
        behavioral_score = pd.Series(0.0, index=common_index)
        total_weight = 0

        for name, indicator in behavioral_indicators.items():
            if not indicator.empty and len(indicator) > 0:
                # 根据行为偏差类型分配权重
                if 'herding' in name:
                    weight = 0.2
                elif 'overreaction' in name:
                    weight = 0.2
                elif 'anchoring' in name:
                    weight = 0.15
                elif 'loss_aversion' in name:
                    weight = 0.15
                elif 'overconfidence' in name:
                    weight = 0.15
                elif 'frame' in name:
                    weight = 0.1
                elif 'mental' in name:
                    weight = 0.05
                else:
                    weight = 0.05

                aligned_indicator = indicator.reindex(common_index).fillna(0)
                behavioral_score += aligned_indicator * weight
                total_weight += weight

        if total_weight > 0:
            behavioral_score /= total_weight

        # 标准化到[0,1]
        behavioral_score = (behavioral_score - behavioral_score.min()) / (behavioral_score.max() - behavioral_score.min() + 1e-10)

        return behavioral_score.fillna(0)

    def recognize_behavioral_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        识别行为模式

        Args:
            data: 市场数据

        Returns:
            行为模式识别结果
        """
        results = {}

        # 羊群效应
        herding_patterns = self.detect_herding_behavior(data)
        results.update(herding_patterns)

        # 过度反应
        overreaction_patterns = self.detect_overreaction_patterns(data)
        results.update(overreaction_patterns)

        # 锚定效应
        anchoring_patterns = self.detect_anchoring_bias(data)
        results.update(anchoring_patterns)

        # 损失厌恶
        loss_aversion_patterns = self.detect_loss_aversion(data)
        results.update(loss_aversion_patterns)

        # 过度自信
        overconfidence_patterns = self.detect_overconfidence_bias(data)
        results.update(overconfidence_patterns)

        # 框架依赖
        frame_patterns = self.detect_frame_dependence(data)
        results.update(frame_patterns)

        # 心理账户
        mental_accounting_patterns = self.detect_mental_accounting(data)
        results.update(mental_accounting_patterns)

        # 行为模式得分
        behavioral_score = self.calculate_behavioral_pattern_score(results)
        results['behavioral_pattern_score'] = behavioral_score

        # 生成交易信号
        results['signals'] = self.generate_behavioral_signals(results)

        return results

    def generate_behavioral_signals(self, behavioral_analysis: Dict[str, Any]) -> pd.Series:
        """
        基于行为分析生成交易信号

        Args:
            behavioral_analysis: 行为分析结果

        Returns:
            交易信号
        """
        if 'behavioral_pattern_score' not in behavioral_analysis:
            return pd.Series()

        behavioral_score = behavioral_analysis['behavioral_pattern_score']
        signals = pd.Series(0, index=behavioral_score.index)

        # 极端行为模式信号
        extreme_behavior = behavioral_score > 0.8
        normal_behavior = behavioral_score < 0.2

        # 羊群效应信号
        herding_signals = []
        for key, value in behavioral_analysis.items():
            if 'herding' in key and isinstance(value, pd.Series):
                herding_signals.append(value > 0)

        if herding_signals:
            strong_herding = pd.Series(False, index=behavioral_score.index)
            for signal in herding_signals:
                strong_herding |= signal

            # 羊群效应通常预示趋势反转
            herd_reversal_buy = strong_herding & (behavioral_score > 0.7)
            herd_reversal_sell = strong_herding & (behavioral_score < 0.3)
        else:
            herd_reversal_buy = pd.Series(False, index=behavioral_score.index)
            herd_reversal_sell = pd.Series(False, index=behavioral_score.index)

        # 过度反应信号
        overreaction_signals = []
        for key, value in behavioral_analysis.items():
            if 'overreaction' in key and isinstance(value, pd.Series):
                if 'reversal' in key:
                    overreaction_signals.append(value > 0)

        if overreaction_signals:
            strong_overreaction = pd.Series(False, index=behavioral_score.index)
            for signal in overreaction_signals:
                strong_overreaction |= signal
        else:
            strong_overreaction = pd.Series(False, index=behavioral_score.index)

        # 损失厌恶信号
        loss_aversion_ratio = behavioral_analysis.get('loss_aversion_ratio', pd.Series(1.0, index=behavioral_score.index))
        high_loss_aversion = loss_aversion_ratio > 2.0

        # 综合信号逻辑
        # 强烈买入：过度反应反转 + 羊群效应反转 + 低损失厌恶
        strong_buy = strong_overreaction & herd_reversal_buy & ~high_loss_aversion

        # 买入：行为模式极端
        buy_signal = extreme_behavior & (behavioral_score < 0.5)

        # 强烈卖出：过度自信 + 高损失厌恶
        overconfidence = behavioral_analysis.get('high_volume_confidence', pd.Series(0, index=behavioral_score.index))
        strong_sell = (overconfidence > 0) & high_loss_aversion

        # 卖出：行为模式正常但可能错过机会
        sell_signal = normal_behavior & (behavioral_score > 0.5)

        # 分配信号值
        signals[strong_buy] = 3
        signals[buy_signal] = 2
        signals[strong_sell] = -3
        signals[sell_signal] = -2

        return signals

    def get_trading_recommendations(self, behavioral_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            behavioral_analysis: 行为分析结果

        Returns:
            交易建议DataFrame
        """
        # 找到一个指标序列作为索引
        index_keys = [k for k, v in behavioral_analysis.items() if isinstance(v, pd.Series) and not v.empty]
        if not index_keys:
            return pd.DataFrame()

        index = behavioral_analysis[index_keys[0]].index
        recommendations = pd.DataFrame(index=index)

        # 添加行为分析指标
        for key, value in behavioral_analysis.items():
            if isinstance(value, pd.Series):
                recommendations[key] = value

        # 交易信号
        recommendations['signals'] = behavioral_analysis.get('signals', pd.Series(0, index=index))

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 过度反应反转机会',
            2: '买入 - 行为模式极端的逆向机会',
            0: '持有 - 行为模式正常',
            -2: '卖出 - 行为回归正常',
            -3: '强烈卖出 - 过度自信风险'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 行为模式分类
        def classify_behavioral_pattern(row):
            score = row.get('behavioral_pattern_score', 0)
            herding = row.get('high_synchronicity_20', 0)
            overreaction = row.get('extreme_moves', 0)
            loss_aversion = row.get('loss_aversion_ratio', 1.0)

            if score > 0.8 and herding > 0:
                return 'extreme_herding'
            elif score > 0.7 and overreaction > 0:
                return 'overreaction_pattern'
            elif loss_aversion > 2.0:
                return 'loss_aversion_dominant'
            elif score < 0.2:
                return 'rational_behavior'
            else:
                return 'normal_behavioral'

        recommendations['behavioral_pattern_type'] = recommendations.apply(classify_behavioral_pattern, axis=1)

        # 仓位建议
        position_map = {3: 0.5, 2: 0.3, 0: 0.2, -2: 0.1, -3: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 行为偏差调整
        loss_aversion = recommendations.get('loss_aversion_ratio', 1.0)
        overconfidence = recommendations.get('high_volume_confidence', 0)

        # 损失厌恶调整
        loss_aversion_adjustment = 1 / (1 + (loss_aversion - 1) * 0.3)
        recommendations['position_size'] *= loss_aversion_adjustment

        # 过度自信调整
        overconfidence_adjustment = 1 - overconfidence * 0.2
        recommendations['position_size'] *= overconfidence_adjustment

        # 确保仓位在合理范围内
        recommendations['position_size'] = recommendations['position_size'].clip(0, 1)

        # 策略建议
        def get_behavioral_strategy(pattern_type, signal_strength):
            if pattern_type == 'extreme_herding':
                return 'contrarian'  # 逆向策略
            elif pattern_type == 'overreaction_pattern':
                return 'mean_reversion'  # 均值回归
            elif pattern_type == 'loss_aversion_dominant':
                return 'defensive'  # 防御性策略
            elif pattern_type == 'rational_behavior':
                return 'fundamental'  # 基本面策略
            else:
                return 'balanced'

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_behavioral_strategy(x['behavioral_pattern_type'], abs(x['signals'])),
            axis=1
        )

        # 行为风险评估
        def assess_behavioral_risk(row):
            risk_score = 0

            # 行为模式风险
            if row.get('behavioral_pattern_score', 0) > 0.8:
                risk_score += 0.3

            # 损失厌恶风险
            if row.get('loss_aversion_ratio', 1.0) > 2.0:
                risk_score += 0.3

            # 过度自信风险
            if row.get('high_volume_confidence', 0) > 0:
                risk_score += 0.2

            # 过度反应风险
            if row.get('extreme_moves', 0) > 0:
                risk_score += 0.2

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['risk_level'] = recommendations.apply(assess_behavioral_risk, axis=1)

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成模拟市场数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=500, freq='D')

    # 生成价格序列（包含行为偏差）
    base_price = 100
    price_changes = np.random.randn(500) * 0.02

    # 添加羊群效应（周期性同步）
    herding_effect = np.sin(np.linspace(0, 6*np.pi, 500)) * 0.01
    price_changes += herding_effect

    # 添加过度反应
    overreaction_spikes = np.random.choice([0, 1], 500, p=[0.95, 0.05])
    price_changes += overreaction_spikes * np.random.randn(500) * 0.05

    # 构建价格
    prices = [base_price]
    for change in price_changes:
        prices.append(prices[-1] * (1 + change))

    # 生成成交量
    base_volume = 1000000
    volume_changes = np.random.lognormal(0, 0.3, 500)
    volumes = base_volume * volume_changes

    # 创建数据框
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices[1:],
        'volume': volumes
    }).set_index('date')

    # 创建行为模式识别器
    bpr = BehavioralPatternRecognizer(pattern_windows=[5, 10, 20, 50])

    # 识别行为模式
    print("识别行为模式...")
    behavioral_results = bpr.recognize_behavioral_patterns(market_data)

    # 显示主要指标
    print(f"行为模式得分: {behavioral_results['behavioral_pattern_score'].iloc[-1]:.3f}")

    if 'high_synchronicity_20' in behavioral_results:
        print(f"高同步性: {behavioral_results['high_synchronicity_20'].iloc[-1]}")
        print(f"过度反应: {behavioral_results['extreme_moves'].iloc[-1]}")

    if 'loss_aversion_ratio' in behavioral_results:
        print(f"损失厌恶比率: {behavioral_results['loss_aversion_ratio'].iloc[-1]:.3f}")

    if 'high_volume_confidence' in behavioral_results:
        print(f"过度自信: {behavioral_results['high_volume_confidence'].iloc[-1]}")

    print(f"当前信号: {behavioral_results['signals'].iloc[-1]}")

    # 获取交易建议
    recommendations = bpr.get_trading_recommendations(behavioral_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['behavioral_pattern_score', 'behavioral_pattern_type',
                                          'signal_description', 'strategy', 'position_size', 'risk_level']])