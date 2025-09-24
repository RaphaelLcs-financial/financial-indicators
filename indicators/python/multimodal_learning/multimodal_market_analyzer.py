"""
多模态市场分析器
Multimodal Market Analyzer

融合文本、图像、时序数据的跨模态市场分析框架
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
    print("🧠 PyTorch 已启用")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 不可用")

class MultimodalMarketAnalyzer:
    """
    多模态市场分析器

    整合多种数据模态进行综合市场分析：
    - 时序数据：价格、成交量等传统金融数据
    - 文本数据：新闻、报告、社交媒体情绪
    - 图像数据：图表模式、技术形态识别
    """

    def __init__(self, modalities: List[str] = None, fusion_method: str = 'attention'):
        """
        初始化多模态分析器

        Args:
            modalities: 使用的模态列表 ['time_series', 'text', 'image']
            fusion_method: 融合方法 ['attention', 'concat', 'transformer']
        """
        self.modalities = modalities or ['time_series', 'text', 'image']
        self.fusion_method = fusion_method

        # 模态处理器
        self.time_series_processor = TimeSeriesProcessor()
        self.text_processor = TextProcessor()
        self.image_processor = ImageProcessor()

        # 融合模型
        self.fusion_model = None
        self.is_trained = False

        # 分析结果
        self.multimodal_features = {}
        self.cross_modal_correlations = {}
        self.fusion_weights = {}

    def process_time_series_data(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """处理时序数据"""
        print("📈 处理时序数据...")

        features = {}

        # 基础统计特征
        for col in data.select_dtypes(include=[np.number]).columns:
            col_data = data[col].dropna()
            if len(col_data) > 0:
                features[f'{col}_mean'] = [np.mean(col_data)]
                features[f'{col}_std'] = [np.std(col_data)]
                features[f'{col}_trend'] = [np.polyfit(range(len(col_data)), col_data.values, 1)[0]]
                features[f'{col}_volatility'] = [np.std(np.diff(col_data.values))]

        # 技术指标特征
        if 'close' in data.columns:
            close_prices = data['close'].dropna()
            if len(close_prices) > 20:
                # 移动平均
                features['sma_20'] = [close_prices.rolling(20).mean().iloc[-1]]
                features['ema_12'] = [close_prices.ewm(span=12).mean().iloc[-1]]

                # RSI
                returns = close_prices.pct_change().dropna()
                if len(returns) > 14:
                    gains = returns[returns > 0]
                    losses = -returns[returns < 0]
                    avg_gain = np.mean(gains[-14:]) if len(gains) > 0 else 0
                    avg_loss = np.mean(losses[-14:]) if len(losses) > 0 else 0
                    if avg_loss > 0:
                        rsi = 100 - (100 / (1 + avg_gain / avg_loss))
                        features['rsi'] = [rsi]

        return features

    def process_text_data(self, text_data: List[str]) -> Dict[str, np.ndarray]:
        """处理文本数据"""
        print("📝 处理文本数据...")

        features = {}

        # 简单的文本特征
        all_text = ' '.join(text_data)

        # 基础统计
        features['text_length'] = [len(all_text)]
        features['num_articles'] = [len(text_data)]

        # 情绪分析（简化版）
        positive_words = ['bull', 'growth', 'increase', 'rise', 'positive', 'up', 'gain', 'profit']
        negative_words = ['bear', 'decline', 'decrease', 'fall', 'negative', 'down', 'loss', 'risk']

        positive_count = sum(1 for word in positive_words if word in all_text.lower())
        negative_count = sum(1 for word in negative_words if word in all_text.lower())

        total_sentiment_words = positive_count + negative_count
        if total_sentiment_words > 0:
            sentiment_score = (positive_count - negative_count) / total_sentiment_words
        else:
            sentiment_score = 0

        features['sentiment_score'] = [sentiment_score]
        features['positive_word_count'] = [positive_count]
        features['negative_word_count'] = [negative_count]

        return features

    def process_image_data(self, image_data: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """处理图像数据"""
        print("🖼️ 处理图像数据...")

        features = {}

        for i, image in enumerate(image_data):
            # 简单的图像特征
            if len(image.shape) == 3:
                # 颜色特征
                mean_colors = np.mean(image, axis=(0, 1))
                features[f'image_{i}_mean_r'] = [mean_colors[0]]
                features[f'image_{i}_mean_g'] = [mean_colors[1]]
                features[f'image_{i}_mean_b'] = [mean_colors[2]]

                # 纹理特征（标准差）
                std_colors = np.std(image, axis=(0, 1))
                features[f'image_{i}_std_r'] = [std_colors[0]]
                features[f'image_{i}_std_g'] = [std_colors[1]]
                features[f'image_{i}_std_b'] = [std_colors[2]]

        return features

    def fuse_modalities(self, modality_features: Dict[str, Dict[str, np.ndarray]]) -> np.ndarray:
        """融合多模态特征"""
        print("🔗 融合多模态特征...")

        # 收集所有特征
        all_features = []
        feature_names = []

        for modality, features in modality_features.items():
            for feature_name, feature_value in features.items():
                all_features.append(feature_value)
                feature_names.append(f"{modality}_{feature_name}")

        if not all_features:
            return np.array([])

        # 特征对齐
        max_length = max(len(feat) for feat in all_features)
        aligned_features = []

        for feature in all_features:
            if len(feature) < max_length:
                # 填充到相同长度
                padded_feature = np.pad(feature, (0, max_length - len(feature)), 'constant')
            else:
                padded_feature = feature[:max_length]
            aligned_features.append(padded_feature)

        # 融合方法
        if self.fusion_method == 'concat':
            fused_features = np.concatenate(aligned_features, axis=0)
        elif self.fusion_method == 'attention':
            fused_features = self._attention_fusion(aligned_features)
        else:
            # 默认简单平均
            fused_features = np.mean(aligned_features, axis=0)

        return fused_features

    def _attention_fusion(self, features: List[np.ndarray]) -> np.ndarray:
        """注意力机制融合"""
        if not features:
            return np.array([])

        # 简化的注意力机制
        feature_matrix = np.array(features)

        # 计算注意力权重
        attention_weights = np.exp(feature_matrix.sum(axis=1))
        attention_weights = attention_weights / attention_weights.sum()

        # 加权融合
        fused_features = np.average(feature_matrix, axis=0, weights=attention_weights)

        return fused_features

    def analyze_cross_modal_correlations(self, modality_features: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, float]:
        """分析跨模态相关性"""
        print("🔗 分析跨模态相关性...")

        correlations = {}

        # 提取各模态的代表特征
        modality_representations = {}
        for modality, features in modality_features.items():
            if features:
                # 使用第一个特征作为代表
                first_feature = list(features.values())[0]
                if len(first_feature) > 0:
                    modality_representations[modality] = first_feature[0]

        # 计算模态间相关性
        modalities = list(modality_representations.keys())
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities[i+1:], i+1):
                if mod1 in modality_representations and mod2 in modality_representations:
                    # 简化的相关性计算
                    corr = np.corrcoef(
                        [modality_representations[mod1]],
                        [modality_representations[mod2]]
                    )[0, 1]
                    correlations[f"{mod1}_{mod2}"] = corr

        self.cross_modal_correlations = correlations
        return correlations

    def generate_multimodal_signals(self, time_series_data: pd.DataFrame,
                                   text_data: List[str] = None,
                                   image_data: List[np.ndarray] = None) -> pd.DataFrame:
        """生成多模态交易信号"""
        print("🎯 生成多模态交易信号...")

        # 处理各模态数据
        modality_features = {}

        if 'time_series' in self.modalities:
            modality_features['time_series'] = self.process_time_series_data(time_series_data)

        if 'text' in self.modalities and text_data:
            modality_features['text'] = self.process_text_data(text_data)

        if 'image' in self.modalities and image_data:
            modality_features['image'] = self.process_image_data(image_data)

        # 融合特征
        fused_features = self.fuse_modalities(modality_features)

        # 分析跨模态相关性
        cross_modal_corr = self.analyze_cross_modal_correlations(modality_features)

        # 生成信号
        signals = []

        for i in range(len(time_series_data)):
            # 基于各模态的信号
            time_series_signal = self._generate_time_series_signal(modality_features.get('time_series', {}), i)
            text_signal = self._generate_text_signal(modality_features.get('text', {}))
            image_signal = self._generate_image_signal(modality_features.get('image', {}))

            # 融合信号
            weights = self._calculate_modality_weights(cross_modal_corr)
            combined_signal = (
                weights.get('time_series', 0.5) * time_series_signal +
                weights.get('text', 0.25) * text_signal +
                weights.get('image', 0.25) * image_signal
            )

            signals.append(np.clip(combined_signal, -1, 1))

        signals_df = pd.DataFrame({
            'signal': signals,
            'time_series_confidence': [abs(time_series_signal)] * len(signals),
            'text_confidence': [abs(text_signal)] * len(signals),
            'image_confidence': [abs(image_signal)] * len(signals),
            'fusion_confidence': [np.mean([abs(time_series_signal), abs(text_signal), abs(image_signal)])] * len(signals)
        }, index=time_series_data.index)

        return signals_df

    def _generate_time_series_signal(self, time_series_features: Dict, index: int) -> float:
        """生成时序信号"""
        signal = 0

        # 基于技术指标的信号
        if 'rsi' in time_series_features:
            rsi_value = time_series_features['rsi'][0] if time_series_features['rsi'] else 50
            if rsi_value < 30:
                signal += 0.5
            elif rsi_value > 70:
                signal -= 0.5

        # 基于趋势的信号
        if 'close_trend' in time_series_features:
            trend = time_series_features['close_trend'][0] if time_series_features['close_trend'] else 0
            signal += np.sign(trend) * 0.3

        return np.clip(signal, -1, 1)

    def _generate_text_signal(self, text_features: Dict) -> float:
        """生成文本信号"""
        signal = 0

        # 基于情绪的信号
        if 'sentiment_score' in text_features:
            sentiment = text_features['sentiment_score'][0] if text_features['sentiment_score'] else 0
            signal += sentiment * 0.8

        # 基于词频的信号
        if 'positive_word_count' in text_features and 'negative_word_count' in text_features:
            pos_count = text_features['positive_word_count'][0] if text_features['positive_word_count'] else 0
            neg_count = text_features['negative_word_count'][0] if text_features['negative_word_count'] else 0

            if pos_count + neg_count > 0:
                word_sentiment = (pos_count - neg_count) / (pos_count + neg_count)
                signal += word_sentiment * 0.2

        return np.clip(signal, -1, 1)

    def _generate_image_signal(self, image_features: Dict) -> float:
        """生成图像信号"""
        signal = 0

        # 简化的图像信号生成
        # 实际应用中应该使用更复杂的图像分析
        if image_features:
            # 基于颜色特征的简单信号
            red_features = [v for k, v in image_features.items() if 'mean_r' in k]
            if red_features:
                # 红色特征可能表示警示
                avg_red = np.mean(red_features)
                if avg_red > 0.6:
                    signal -= 0.3
                elif avg_red < 0.4:
                    signal += 0.3

        return np.clip(signal, -1, 1)

    def _calculate_modality_weights(self, cross_modal_corr: Dict) -> Dict[str, float]:
        """计算模态权重"""
        weights = {
            'time_series': 0.6,
            'text': 0.25,
            'image': 0.15
        }

        # 基于相关性调整权重
        if cross_modal_corr:
            # 如果相关性高，降低依赖性强的模态权重
            high_corr_count = sum(1 for corr in cross_modal_corr.values() if abs(corr) > 0.7)
            if high_corr_count > 0:
                # 重新分配权重
                weights['time_series'] = 0.7
                weights['text'] = 0.2
                weights['image'] = 0.1

        return weights

    def get_multimodal_insights(self) -> Dict[str, Any]:
        """获取多模态洞察"""
        return {
            'modalities': self.modalities,
            'fusion_method': self.fusion_method,
            'cross_modal_correlations': self.cross_modal_correlations,
            'modality_weights': self.fusion_weights,
            'feature_count': len(self.multimodal_features)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'modalities': self.modalities,
            'fusion_method': self.fusion_method,
            'is_trained': self.is_trained,
            'torch_available': TORCH_AVAILABLE,
            'model_type': 'Multimodal Market Analyzer'
        }

class TimeSeriesProcessor:
    """时序数据处理器"""
    def extract_features(self, data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """提取时序特征"""
        features = {}
        # 这里可以添加更复杂的时序特征提取
        return features

class TextProcessor:
    """文本数据处理器"""
    def extract_features(self, text: str) -> Dict[str, np.ndarray]:
        """提取文本特征"""
        features = {}
        # 这里可以添加更复杂的文本特征提取
        return features

class ImageProcessor:
    """图像数据处理器"""
    def extract_features(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """提取图像特征"""
        features = {}
        # 这里可以添加更复杂的图像特征提取
        return features

# 便捷函数
def create_multimodal_analyzer(modalities: List[str] = None, fusion_method: str = 'attention') -> MultimodalMarketAnalyzer:
    """创建多模态分析器实例"""
    return MultimodalMarketAnalyzer(modalities, fusion_method)

def quick_multimodal_analysis(time_series_data: pd.DataFrame, text_data: List[str] = None) -> Dict[str, Any]:
    """快速多模态分析"""
    analyzer = MultimodalMarketAnalyzer()

    # 生成信号
    signals = analyzer.generate_multimodal_signals(time_series_data, text_data)

    return {
        'latest_signal': signals['signal'].iloc[-1] if len(signals) > 0 else 0,
        'fusion_confidence': signals['fusion_confidence'].iloc[-1] if len(signals) > 0 else 0,
        'cross_modal_correlations': analyzer.cross_modal_correlations,
        'model_info': analyzer.get_model_info()
    }