"""
高级自然语言处理情感分析指标
Advanced NLP Sentiment Analysis Indicators

基于深度学习的金融文本情感分析系统
支持多模态文本分析和市场情绪预测
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
    from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
    from sentence_transformers import SentenceTransformer
    TORCH_AVAILABLE = True
    print("🧠 PyTorch 已启用")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 不可用")

# 传统NLP支持
try:
    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    NLTK_AVAILABLE = True
    print("📝 NLTK 已启用")
except ImportError:
    NLTK_AVAILABLE = False
    print("⚠️ NLTK 不可用")

class AdvancedSentimentAnalyzer:
    """
    高级情感分析器

    整合多种NLP技术进行金融文本情感分析
    支持预训练模型、主题建模、情感演化分析
    """

    def __init__(self, model_type: str = 'financial_bert',
                 embedding_dim: int = 768, num_topics: int = 10):
        """
        初始化高级情感分析器

        Args:
            model_type: 模型类型 ['financial_bert', 'finbert', 'general_bert']
            embedding_dim: 嵌入维度
            num_topics: 主题数量
        """
        self.model_type = model_type
        self.embedding_dim = embedding_dim
        self.num_topics = num_topics

        # 情感分析模型
        self.sentiment_model = None
        self.tokenizer = None
        self.embedding_model = None

        # 主题模型
        self.topic_model = None
        self.vectorizer = None

        # 情感字典
        self.financial_sentiment_lexicon = self._create_financial_sentiment_lexicon()

        # 分析历史
        self.sentiment_history = []
        self.topic_evolution = []
        self.emotion_patterns = {}

        # 性能指标
        self.analysis_stats = {
            'text_count': 0,
            'sentiment_distribution': {},
            'topic_coherence': 0.0,
            'emotion_accuracy': 0.0
        }

    def _create_financial_sentiment_lexicon(self) -> Dict[str, float]:
        """创建金融情感词典"""
        return {
            # 牛市词汇
            'bullish': 0.8, 'rally': 0.7, 'surge': 0.9, 'boom': 0.8, 'growth': 0.6,
            'expansion': 0.7, 'positive': 0.5, 'optimistic': 0.6, 'confidence': 0.5,
            'breakthrough': 0.8, 'outperform': 0.7, 'upgrade': 0.6, 'strong': 0.5,

            # 熊市词汇
            'bearish': -0.8, 'crash': -0.9, 'plunge': -0.8, 'slump': -0.7, 'recession': -0.8,
            'contraction': -0.7, 'negative': -0.5, 'pessimistic': -0.6, 'risk': -0.4,
            'downturn': -0.7, 'underperform': -0.6, 'downgrade': -0.7, 'weak': -0.5,

            # 波动性词汇
            'volatile': -0.3, 'uncertainty': -0.4, 'fluctuation': -0.2, 'chaos': -0.6,
            'turbulent': -0.5, 'unstable': -0.4, 'erratic': -0.3,

            # 技术词汇
            'resistance': 0.2, 'support': -0.2, 'breakout': 0.6, 'breakdown': -0.6,
            'overbought': -0.3, 'oversold': 0.3, 'momentum': 0.4, 'trend': 0.1,

            # 财务词汇
            'earnings': 0.3, 'revenue': 0.3, 'profit': 0.4, 'loss': -0.4, 'dividend': 0.2,
            'margin': 0.2, 'leverage': -0.1, 'liquidity': 0.1, 'solvency': 0.2
        }

    def initialize_models(self):
        """初始化NLP模型"""
        if TORCH_AVAILABLE:
            try:
                # 初始化情感分析模型
                if self.model_type == 'financial_bert':
                    model_name = 'yiyanghkust/finbert-tone'
                elif self.model_type == 'finbert':
                    model_name = 'ProsusAI/finbert'
                else:
                    model_name = 'bert-base-uncased'

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name)

                # 初始化嵌入模型
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

                print(f"✅ {model_name} 模型加载成功")

            except Exception as e:
                print(f"⚠️ 模型加载失败: {e}")
                self.sentiment_model = None
                self.embedding_model = None

        if NLTK_AVAILABLE:
            try:
                # 初始化NLTK组件
                self.sia = SentimentIntensityAnalyzer()
                nltk.download('punkt', quiet=True)
                nltk.download('stopwords', quiet=True)
                print("✅ NLTK 组件初始化成功")
            except Exception as e:
                print(f"⚠️ NLTK 初始化失败: {e}")

    def analyze_financial_text(self, texts: List[str]) -> Dict[str, Any]:
        """分析金融文本情感"""
        print(f"📝 分析 {len(texts)} 条金融文本...")

        results = {
            'texts': texts,
            'sentiment_scores': [],
            'emotion_profiles': [],
            'topic_distributions': [],
            'market_impact_predictions': [],
            'semantic_embeddings': []
        }

        for i, text in enumerate(texts):
            text_result = self._analyze_single_text(text, i)

            results['sentiment_scores'].append(text_result['sentiment_score'])
            results['emotion_profiles'].append(text_result['emotion_profile'])
            results['topic_distributions'].append(text_result['topic_distribution'])
            results['market_impact_predictions'].append(text_result['market_impact'])
            results['semantic_embeddings'].append(text_result['embedding'])

        # 更新统计信息
        self.analysis_stats['text_count'] += len(texts)
        self._update_sentiment_distribution(results['sentiment_scores'])

        return results

    def _analyze_single_text(self, text: str, text_id: int) -> Dict[str, Any]:
        """分析单个文本"""
        # 情感分析
        sentiment_score = self._calculate_sentiment_score(text)

        # 情绪分析
        emotion_profile = self._analyze_emotions(text)

        # 主题分析
        topic_distribution = self._analyze_topics(text)

        # 市场影响预测
        market_impact = self._predict_market_impact(text, sentiment_score)

        # 语义嵌入
        embedding = self._get_text_embedding(text)

        return {
            'text_id': text_id,
            'sentiment_score': sentiment_score,
            'emotion_profile': emotion_profile,
            'topic_distribution': topic_distribution,
            'market_impact': market_impact,
            'embedding': embedding
        }

    def _calculate_sentiment_score(self, text: str) -> Dict[str, float]:
        """计算情感分数"""
        sentiment_scores = {}

        # 方法1: 基于词典的情感分析
        lexicon_score = self._lexicon_based_sentiment(text)
        sentiment_scores['lexicon'] = lexicon_score

        # 方法2: 基于Transformer的情感分析
        if TORCH_AVAILABLE and self.sentiment_model is not None:
            transformer_score = self._transformer_sentiment(text)
            sentiment_scores['transformer'] = transformer_score
        else:
            sentiment_scores['transformer'] = 0.0

        # 方法3: 基于NLTK的情感分析
        if NLTK_AVAILABLE:
            nltk_score = self._nltk_sentiment(text)
            sentiment_scores['nltk'] = nltk_score
        else:
            sentiment_scores['nltk'] = 0.0

        # 综合情感分数
        weights = {'lexicon': 0.3, 'transformer': 0.5, 'nltk': 0.2}
        available_scores = {k: v for k, v in sentiment_scores.items() if v != 0.0}

        if available_scores:
            total_weight = sum(weights.get(k, 0) for k in available_scores.keys())
            normalized_weights = {k: weights.get(k, 0) / total_weight for k in available_scores.keys()}

            composite_score = sum(normalized_weights[k] * v for k, v in available_scores.items())
            sentiment_scores['composite'] = composite_score
        else:
            sentiment_scores['composite'] = 0.0

        return sentiment_scores

    def _lexicon_based_sentiment(self, text: str) -> float:
        """基于词典的情感分析"""
        words = text.lower().split()
        sentiment_sum = 0
        word_count = 0

        for word in words:
            # 去除标点符号
            clean_word = ''.join(c for c in word if c.isalpha())
            if clean_word in self.financial_sentiment_lexicon:
                sentiment_sum += self.financial_sentiment_lexicon[clean_word]
                word_count += 1

        return sentiment_sum / max(word_count, 1)

    def _transformer_sentiment(self, text: str) -> float:
        """基于Transformer的情感分析"""
        if not TORCH_AVAILABLE or self.sentiment_model is None:
            return 0.0

        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)

            with torch.no_grad():
                outputs = self.sentiment_model(**inputs)
                predictions = F.softmax(outputs.logits, dim=-1)

            # 假设输出是 [negative, neutral, positive]
            sentiment_score = predictions[0][2].item() - predictions[0][0].item()  # positive - negative
            return sentiment_score

        except Exception as e:
            print(f"Transformer情感分析失败: {e}")
            return 0.0

    def _nltk_sentiment(self, text: str) -> float:
        """基于NLTK的情感分析"""
        if not NLTK_AVAILABLE:
            return 0.0

        try:
            scores = self.sia.polarity_scores(text)
            return scores['compound']
        except Exception as e:
            print(f"NLTK情感分析失败: {e}")
            return 0.0

    def _analyze_emotions(self, text: str) -> Dict[str, float]:
        """分析情绪特征"""
        emotion_keywords = {
            'fear': ['fear', 'scared', 'panic', 'worried', 'anxious', 'concerned'],
            'greed': ['greed', 'profit', 'gain', 'rich', 'money', 'wealth'],
            'hope': ['hope', 'optimistic', 'positive', 'confidence', 'belief'],
            'regret': ['regret', 'missed', 'opportunity', 'should', 'wrong'],
            'excitement': ['excited', 'amazing', 'incredible', 'fantastic', 'great'],
            'anger': ['angry', 'frustrated', 'disappointed', 'upset', 'annoyed']
        }

        emotion_scores = {}
        words = text.lower().split()

        for emotion, keywords in emotion_keywords.items():
            count = sum(1 for word in words if any(keyword in word for keyword in keywords))
            emotion_scores[emotion] = count / len(words) if words else 0.0

        return emotion_scores

    def _analyze_topics(self, text: str) -> Dict[str, float]:
        """分析主题分布"""
        if not NLTK_AVAILABLE:
            return {'general': 1.0}

        try:
            # 简化的主题分析
            topic_keywords = {
                'earnings': ['earnings', 'revenue', 'profit', 'income', 'financial'],
                'market': ['market', 'stock', 'trading', 'investment', 'portfolio'],
                'economy': ['economy', 'economic', 'gdp', 'inflation', 'interest'],
                'policy': ['policy', 'government', 'regulation', 'fed', 'central'],
                'technology': ['tech', 'technology', 'innovation', 'digital', 'ai'],
                'energy': ['energy', 'oil', 'gas', 'renewable', 'solar']
            }

            topic_scores = {}
            words = text.lower().split()

            for topic, keywords in topic_keywords.items():
                count = sum(1 for word in words if any(keyword in word for keyword in keywords))
                topic_scores[topic] = count / len(words) if words else 0.0

            # 归一化
            total_score = sum(topic_scores.values())
            if total_score > 0:
                topic_scores = {k: v / total_score for k, v in topic_scores.items()}
            else:
                topic_scores['general'] = 1.0

            return topic_scores

        except Exception as e:
            print(f"主题分析失败: {e}")
            return {'general': 1.0}

    def _predict_market_impact(self, text: str, sentiment_score: Dict[str, float]) -> Dict[str, float]:
        """预测市场影响"""
        composite_sentiment = sentiment_score.get('composite', 0.0)

        # 基于情感强度预测市场影响
        impact_strength = abs(composite_sentiment)

        # 市场影响预测
        market_impact = {
            'price_impact': composite_sentiment * 0.02,  # 2%最大影响
            'volatility_impact': impact_strength * 0.015,  # 波动率影响
            'volume_impact': impact_strength * 0.1,  # 成交量影响
            'confidence': min(impact_strength, 1.0)  # 预测置信度
        }

        return market_impact

    def _get_text_embedding(self, text: str) -> List[float]:
        """获取文本嵌入向量"""
        if TORCH_AVAILABLE and self.embedding_model is not None:
            try:
                embedding = self.embedding_model.encode(text)
                return embedding.tolist()
            except Exception as e:
                print(f"嵌入生成失败: {e}")

        # 返回零向量作为fallback
        return [0.0] * self.embedding_dim

    def track_sentiment_evolution(self, texts_over_time: List[Tuple[str, pd.Timestamp]]) -> Dict[str, Any]:
        """跟踪情感演化"""
        print("📈 跟踪情感演化...")

        evolution_data = []
        sentiment_series = []

        for text, timestamp in texts_over_time:
            result = self.analyze_financial_text([text])
            sentiment_score = result['sentiment_scores'][0]['composite']

            evolution_data.append({
                'timestamp': timestamp,
                'text': text,
                'sentiment_score': sentiment_score,
                'emotion_profile': result['emotion_profiles'][0],
                'topic_distribution': result['topic_distributions'][0]
            })

            sentiment_series.append(sentiment_score)

        # 计算演化趋势
        evolution_analysis = self._analyze_sentiment_trends(evolution_data, sentiment_series)

        return {
            'evolution_data': evolution_data,
            'trend_analysis': evolution_analysis,
            'volatility_analysis': self._calculate_sentiment_volatility(sentiment_series),
            'turning_points': self._detect_sentiment_turning_points(sentiment_series)
        }

    def _analyze_sentiment_trends(self, evolution_data: List[Dict], sentiment_series: List[float]) -> Dict[str, Any]:
        """分析情感趋势"""
        if len(sentiment_series) < 3:
            return {'insufficient_data': True}

        # 计算趋势
        sentiment_array = np.array(sentiment_series)

        # 线性趋势
        time_points = np.arange(len(sentiment_array))
        trend_coef = np.polyfit(time_points, sentiment_array, 1)[0]

        # 移动平均
        ma_5 = pd.Series(sentiment_array).rolling(5).mean().values
        ma_10 = pd.Series(sentiment_array).rolling(10).mean().values

        # 趋势强度
        trend_strength = abs(trend_coef)
        trend_direction = 'positive' if trend_coef > 0 else 'negative'

        return {
            'trend_coefficient': trend_coef,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength,
            'moving_averages': {
                'ma_5': ma_5.tolist() if not np.isnan(ma_5).all() else [],
                'ma_10': ma_10.tolist() if not np.isnan(ma_10).all() else []
            }
        }

    def _calculate_sentiment_volatility(self, sentiment_series: List[float]) -> Dict[str, float]:
        """计算情感波动率"""
        if len(sentiment_series) < 2:
            return {'volatility': 0.0}

        sentiment_array = np.array(sentiment_series)

        # 计算波动率指标
        volatility = np.std(sentiment_array)
        mean_reversion_speed = -np.corrcoef(sentiment_array[:-1], sentiment_array[1:])[0, 1]

        return {
            'volatility': volatility,
            'mean_reversion_speed': mean_reversion_speed,
            'range': np.max(sentiment_array) - np.min(sentiment_array)
        }

    def _detect_sentiment_turning_points(self, sentiment_series: List[float]) -> List[Dict]:
        """检测情感转折点"""
        if len(sentiment_series) < 5:
            return []

        turning_points = []
        sentiment_array = np.array(sentiment_series)

        # 简单的转折点检测
        for i in range(2, len(sentiment_array) - 2):
            # 检查是否为局部极值
            if (sentiment_array[i] > sentiment_array[i-1] and
                sentiment_array[i] > sentiment_array[i+1] and
                sentiment_array[i] > sentiment_array[i-2] and
                sentiment_array[i] > sentiment_array[i+2]):

                turning_points.append({
                    'index': i,
                    'type': 'peak',
                    'value': sentiment_array[i],
                    'strength': (sentiment_array[i] - np.mean(sentiment_array[i-2:i+3])) / np.std(sentiment_array[i-2:i+3])
                })

            elif (sentiment_array[i] < sentiment_array[i-1] and
                  sentiment_array[i] < sentiment_array[i+1] and
                  sentiment_array[i] < sentiment_array[i-2] and
                  sentiment_array[i] < sentiment_array[i+2]):

                turning_points.append({
                    'index': i,
                    'type': 'trough',
                    'value': sentiment_array[i],
                    'strength': (np.mean(sentiment_array[i-2:i+3]) - sentiment_array[i]) / np.std(sentiment_array[i-2:i+3])
                })

        return turning_points

    def generate_sentiment_signals(self, market_data: pd.DataFrame,
                                news_texts: List[str] = None) -> pd.DataFrame:
        """生成情感交易信号"""
        print("🎯 生成情感交易信号...")

        # 如果没有提供新闻文本，使用模拟数据
        if news_texts is None:
            news_texts = self._generate_simulated_news(len(market_data))

        # 分析情感
        sentiment_results = self.analyze_financial_text(news_texts)

        # 生成信号
        signals = []
        confidences = []
        sentiment_scores = []

        for i, result in enumerate(sentiment_results['sentiment_scores']):
            composite_sentiment = result.get('composite', 0.0)

            # 基于情感生成信号
            if composite_sentiment > 0.5:
                signal = 1.0  # 买入信号
            elif composite_sentiment < -0.5:
                signal = -1.0  # 卖出信号
            else:
                signal = 0.0  # 持有信号

            # 置信度基于情感强度
            confidence = min(abs(composite_sentiment), 1.0)

            signals.append(signal)
            confidences.append(confidence)
            sentiment_scores.append(composite_sentiment)

        signals_df = pd.DataFrame({
            'sentiment_signal': signals,
            'confidence': confidences,
            'sentiment_score': sentiment_scores,
            'emotion_intensity': [max(emotion.values()) for emotion in sentiment_results['emotion_profiles']],
            'topic_concentration': [max(topic.values()) for topic in sentiment_results['topic_distributions']]
        }, index=market_data.index)

        return signals_df

    def _generate_simulated_news(self, n_samples: int) -> List[str]:
        """生成模拟新闻文本"""
        templates = [
            "Company reported strong earnings growth of {}%",
            "Market analysts predict {} for the next quarter",
            "Economic indicators show {} in the market",
            "Federal Reserve signals {} on interest rates",
            "Tech sector experiences {} in innovation",
            "Energy prices {} due to supply concerns",
            "Consumer confidence {} according to latest survey",
            "Housing market {} with changing interest rates"
        ]

        positive_adjectives = ["significant growth", "positive trends", "strong performance", "optimistic outlook"]
        negative_adjectives = ["major decline", "negative trends", "weak performance", "pessimistic outlook"]
        neutral_adjectives = ["stable conditions", "mixed signals", "moderate changes", "uncertain outlook"]

        news_texts = []
        for i in range(n_samples):
            template = np.random.choice(templates)

            # 随机选择情感倾向
            sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], p=[0.3, 0.3, 0.4])

            if sentiment_type == 'positive':
                adjective = np.random.choice(positive_adjectives)
            elif sentiment_type == 'negative':
                adjective = np.random.choice(negative_adjectives)
            else:
                adjective = np.random.choice(neutral_adjectives)

            news_text = template.format(adjective)
            news_texts.append(news_text)

        return news_texts

    def _update_sentiment_distribution(self, sentiment_scores: List[Dict[str, float]]):
        """更新情感分布统计"""
        composite_scores = [score.get('composite', 0.0) for score in sentiment_scores]

        if composite_scores:
            # 分类统计
            positive_count = sum(1 for score in composite_scores if score > 0.2)
            negative_count = sum(1 for score in composite_scores if score < -0.2)
            neutral_count = len(composite_scores) - positive_count - negative_count

            self.analysis_stats['sentiment_distribution'] = {
                'positive': positive_count / len(composite_scores),
                'negative': negative_count / len(composite_scores),
                'neutral': neutral_count / len(composite_scores)
            }

    def get_sentiment_insights(self) -> Dict[str, Any]:
        """获取情感分析洞察"""
        return {
            'model_type': self.model_type,
            'embedding_dim': self.embedding_dim,
            'num_topics': self.num_topics,
            'analysis_stats': self.analysis_stats,
            'sentiment_history': self.sentiment_history[-10:],  # 最近10条历史
            'emotion_patterns': self.emotion_patterns,
            'torch_available': TORCH_AVAILABLE,
            'nltk_available': NLTK_AVAILABLE
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': self.model_type,
            'embedding_dim': self.embedding_dim,
            'num_topics': self.num_topics,
            'analysis_stats': self.analysis_stats,
            'models_initialized': self.sentiment_model is not None,
            'torch_available': TORCH_AVAILABLE,
            'nltk_available': NLTK_AVAILABLE,
            'model_type': 'Advanced NLP Sentiment Analyzer'
        }

# 便捷函数
def create_advanced_sentiment_analyzer(model_type: str = 'financial_bert',
                                     embedding_dim: int = 768,
                                     num_topics: int = 10) -> AdvancedSentimentAnalyzer:
    """创建高级情感分析器实例"""
    analyzer = AdvancedSentimentAnalyzer(model_type, embedding_dim, num_topics)
    analyzer.initialize_models()
    return analyzer

def quick_sentiment_analysis(texts: List[str]) -> Dict[str, Any]:
    """快速情感分析"""
    analyzer = AdvancedSentimentAnalyzer()
    analyzer.initialize_models()

    results = analyzer.analyze_financial_text(texts)

    return {
        'average_sentiment': np.mean([score['composite'] for score in results['sentiment_scores']]),
        'sentiment_distribution': analyzer.analysis_stats['sentiment_distribution'],
        'dominant_emotion': max(results['emotion_profiles'][0].items(), key=lambda x: x[1])[0] if results['emotion_profiles'] else 'neutral',
        'model_info': analyzer.get_model_info()
    }