"""
情绪分析器金融指标

本模块实现了基于另类数据的情绪分析指标，包括社交媒体情绪、新闻情绪、搜索趋势等非传统数据源。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalyzer:
    """
    情绪分析器

    分析各种另类数据源中的情绪信号，为投资决策提供情绪维度指标
    """

    def __init__(self,
                 sentiment_window: int = 30,
                 decay_factor: float = 0.95,
                 sentiment_threshold: float = 0.6):
        """
        初始化情绪分析器

        Args:
            sentiment_window: 情绪分析窗口
            decay_factor: 衰减因子
            sentiment_threshold: 情绪阈值
        """
        self.sentiment_window = sentiment_window
        self.decay_factor = decay_factor
        self.sentiment_threshold = sentiment_threshold

        # 情绪权重
        self.sentiment_weights = {
            'social_media': 0.3,
            'news': 0.25,
            'search_trends': 0.2,
            'options_flow': 0.15,
            'institutional': 0.1
        }

    def calculate_social_media_sentiment(self, social_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算社交媒体情绪指标

        Args:
            social_data: 社交媒体数据

        Returns:
            社交媒体情绪指标字典
        """
        metrics = {}

        # Twitter/X情绪
        if 'twitter_sentiment' in social_data.columns:
            twitter_sentiment = social_data['twitter_sentiment']
            metrics['twitter_sentiment_ma'] = twitter_sentiment.rolling(self.sentiment_window).mean()
            metrics['twitter_sentiment_volatility'] = twitter_sentiment.rolling(self.sentiment_window).std()
            metrics['twitter_bull_ratio'] = (twitter_sentiment > 0.6).rolling(self.sentiment_window).mean()
            metrics['twitter_bear_ratio'] = (twitter_sentiment < 0.4).rolling(self.sentiment_window).mean()

        # Reddit情绪
        if 'reddit_sentiment' in social_data.columns:
            reddit_sentiment = social_data['reddit_sentiment']
            metrics['reddit_sentiment_ma'] = reddit_sentiment.rolling(self.sentiment_window).mean()
            metrics['reddit_sentiment_momentum'] = reddit_sentiment.pct_change().rolling(self.sentiment_window).mean()

        # 提及频率
        if 'mention_count' in social_data.columns:
            mentions = social_data['mention_count']
            metrics['mention_frequency'] = mentions.rolling(self.sentiment_window).mean()
            metrics['mention_growth'] = mentions.pct_change().rolling(self.sentiment_window).mean()

            # 异常提及检测
            mention_zscore = (mentions - mentions.rolling(self.sentiment_window).mean()) / \
                           (mentions.rolling(self.sentiment_window).std() + 1e-8)
            metrics['anomaly_mention_score'] = mention_zscore.abs()

        # 情绪分歧度
        if 'sentiment_std' in social_data.columns:
            sentiment_std = social_data['sentiment_std']
            metrics['sentiment_disagreement'] = sentiment_std.rolling(self.sentiment_window).mean()
            metrics['extreme_sentiment_ratio'] = (sentiment_std > 0.3).rolling(self.sentiment_window).mean()

        # 情绪极化指标
        if 'positive_ratio' in social_data.columns and 'negative_ratio' in social_data.columns:
            positive_ratio = social_data['positive_ratio']
            negative_ratio = social_data['negative_ratio']
            metrics['sentiment_polarization'] = abs(positive_ratio - negative_ratio).rolling(self.sentiment_window).mean()

        return metrics

    def calculate_news_sentiment_metrics(self, news_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算新闻情绪指标

        Args:
            news_data: 新闻数据

        Returns:
            新闻情绪指标字典
        """
        metrics = {}

        # 新闻情绪得分
        if 'news_sentiment_score' in news_data.columns:
            news_sentiment = news_data['news_sentiment_score']
            metrics['news_sentiment_ma'] = news_sentiment.rolling(self.sentiment_window).mean()
            metrics['news_sentiment_trend'] = news_sentiment.diff().rolling(self.sentiment_window).mean()

        # 新闻数量
        if 'news_count' in news_data.columns:
            news_count = news_data['news_count']
            metrics['news_volume_ma'] = news_count.rolling(self.sentiment_window).mean()
            metrics['news_volume_spike'] = (news_count > news_count.rolling(self.sentiment_window).quantile(0.8)).astype(int)

        # 新闻情绪分歧
        if 'news_sentiment_variance' in news_data.columns:
            sentiment_variance = news_data['news_sentiment_variance']
            metrics['news_disagreement'] = sentiment_variance.rolling(self.sentiment_window).mean()

        # 标题情绪
        if 'headline_sentiment' in news_data.columns:
            headline_sentiment = news_data['headline_sentiment']
            metrics['headline_sentiment_ma'] = headline_sentiment.rolling(self.sentiment_window).mean()
            metrics['headline_sentiment_skew'] = headline_sentiment.rolling(self.sentiment_window).skew()

        # 情绪极值检测
        if 'extreme_positive_news' in news_data.columns:
            extreme_positive = news_data['extreme_positive_news']
            metrics['extreme_positive_ratio'] = extreme_positive.rolling(self.sentiment_window).mean()

        if 'extreme_negative_news' in news_data.columns:
            extreme_negative = news_data['extreme_negative_news']
            metrics['extreme_negative_ratio'] = extreme_negative.rolling(self.sentiment_window).mean()

        # 新闻情绪领先指标
        if 'sentiment_lead_indicator' in news_data.columns:
            lead_indicator = news_data['sentiment_lead_indicator']
            metrics['news_sentiment_lead'] = lead_indicator.rolling(self.sentiment_window).mean()

        return metrics

    def calculate_search_trend_metrics(self, search_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算搜索趋势指标

        Args:
            search_data: 搜索数据

        Returns:
            搜索趋势指标字典
        """
        metrics = {}

        # 搜索量
        if 'search_volume' in search_data.columns:
            search_volume = search_data['search_volume']
            metrics['search_volume_ma'] = search_volume.rolling(self.sentiment_window).mean()
            metrics['search_volume_growth'] = search_volume.pct_change().rolling(self.sentiment_window).mean()

            # 搜索量标准化
            search_normalized = (search_volume - search_volume.rolling(252).min()) / \
                              (search_volume.rolling(252).max() - search_volume.rolling(252).min() + 1e-8)
            metrics['search_volume_normalized'] = search_normalized

        # 相关搜索词
        if 'related_searches' in search_data.columns:
            related_searches = search_data['related_searches']
            metrics['related_search_diversity'] = related_searches.rolling(self.sentiment_window).apply(
                lambda x: len(set(x)) if isinstance(x, (list, pd.Series)) else 1
            )

        # 搜索意图分析
        if 'buy_search_ratio' in search_data.columns:
            buy_ratio = search_data['buy_search_ratio']
            metrics['buy_intent_ma'] = buy_ratio.rolling(self.sentiment_window).mean()
            metrics['buy_intent_momentum'] = buy_ratio.pct_change().rolling(self.sentiment_window).mean()

        if 'sell_search_ratio' in search_data.columns:
            sell_ratio = search_data['sell_search_ratio']
            metrics['sell_intent_ma'] = sell_ratio.rolling(self.sentiment_window).mean()
            metrics['sell_buy_ratio'] = sell_ratio / (buy_ratio + 1e-8)

        # 预测性搜索
        if 'prediction_searches' in search_data.columns:
            prediction_searches = search_data['prediction_searches']
            metrics['prediction_search_ratio'] = prediction_searches.rolling(self.sentiment_window).mean()
            metrics['prediction_search_growth'] = prediction_searches.pct_change().rolling(self.sentiment_window).mean()

        # 信息搜索
        if 'information_searches' in search_data.columns:
            info_searches = search_data['information_searches']
            metrics['information_seek_ratio'] = info_searches.rolling(self.sentiment_window).mean()

        return metrics

    def calculate_options_flow_sentiment(self, options_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算期权流情绪指标

        Args:
            options_data: 期权数据

        Returns:
            期权流情绪指标字典
        """
        metrics = {}

        # 看跌看涨比率
        if 'put_call_ratio' in options_data.columns:
            pcr = options_data['put_call_ratio']
            metrics['put_call_ratio_ma'] = pcr.rolling(self.sentiment_window).mean()
            metrics['put_call_ratio_zscore'] = (pcr - pcr.rolling(self.sentiment_window).mean()) / \
                                           (pcr.rolling(self.sentiment_window).std() + 1e-8)

        # 期权成交量
        if 'options_volume' in options_data.columns:
            options_volume = options_data['options_volume']
            metrics['options_volume_ma'] = options_volume.rolling(self.sentiment_window).mean()
            metrics['options_volume_growth'] = options_volume.pct_change().rolling(self.sentiment_window).mean()

        # 未平仓合约
        if 'open_interest' in options_data.columns:
            open_interest = options_data['open_interest']
            metrics['open_interest_ma'] = open_interest.rolling(self.sentiment_window).mean()
            metrics['open_interest_change'] = open_interest.diff().rolling(self.sentiment_window).mean()

        # 隐含波动率
        if 'implied_volatility' in options_data.columns:
            iv = options_data['implied_volatility']
            metrics['implied_volatility_ma'] = iv.rolling(self.sentiment_window).mean()
            metrics['iv_premium'] = (iv - iv.rolling(60).mean()) / iv.rolling(60).mean()

        # 大额期权交易
        if 'large_options_trades' in options_data.columns:
            large_trades = options_data['large_options_trades']
            metrics['large_options_frequency'] = (large_trades > 0).rolling(self.sentiment_window).sum()
            metrics['large_options_notional'] = large_trades.rolling(self.sentiment_window).sum()

        # 期权情绪综合指标
        if 'options_sentiment_index' in options_data.columns:
            options_sentiment = options_data['options_sentiment_index']
            metrics['options_sentiment_ma'] = options_sentiment.rolling(self.sentiment_window).mean()
            metrics['options_sentiment_momentum'] = options_sentiment.diff().rolling(self.sentiment_window).mean()

        return metrics

    def calculate_institutional_sentiment(self, institutional_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算机构情绪指标

        Args:
            institutional_data: 机构数据

        Returns:
            机构情绪指标字典
        """
        metrics = {}

        # 机构持仓
        if 'institutional_holdings' in institutional_data.columns:
            holdings = institutional_data['institutional_holdings']
            metrics['institutional_holdings_ma'] = holdings.rolling(self.sentiment_window).mean()
            metrics['institutional_holdings_change'] = holdings.pct_change().rolling(self.sentiment_window).mean()

        # 做空比率
        if 'short_interest_ratio' in institutional_data.columns:
            short_ratio = institutional_data['short_interest_ratio']
            metrics['short_interest_ma'] = short_ratio.rolling(self.sentiment_window).mean()
            metrics['short_interest_zscore'] = (short_ratio - short_ratio.rolling(self.sentiment_window).mean()) / \
                                            (short_ratio.rolling(self.sentiment_window).std() + 1e-8)

        # 内部人交易
        if 'insider_trading_ratio' in institutional_data.columns:
            insider_ratio = institutional_data['insider_trading_ratio']
            metrics['insider_trading_ma'] = insider_ratio.rolling(self.sentiment_window).mean()
            metrics['insider_buy_sell_ratio'] = insider_ratio

        # 分析师评级
        if 'analyst_ratings' in institutional_data.columns:
            analyst_ratings = institutional_data['analyst_ratings']
            metrics['analyst_sentiment_ma'] = analyst_ratings.rolling(self.sentiment_window).mean()
            metrics['analyst_rating_changes'] = analyst_ratings.diff().rolling(self.sentiment_window).mean()

        # 机构资金流向
        if 'institutional_flows' in institutional_data.columns:
            flows = institutional_data['institutional_flows']
            metrics['institutional_net_flow'] = flows.rolling(self.sentiment_window).mean()
            metrics['flow_momentum'] = flows.pct_change().rolling(self.sentiment_window).mean()

        return metrics

    def calculate_composite_sentiment_index(self, social_metrics: Dict[str, pd.Series],
                                         news_metrics: Dict[str, pd.Series],
                                         search_metrics: Dict[str, pd.Series],
                                         options_metrics: Dict[str, pd.Series],
                                         institutional_metrics: Dict[str, pd.Series]) -> pd.Series:
        """
        计算综合情绪指数

        Args:
            social_metrics: 社交媒体情绪指标
            news_metrics: 新闻情绪指标
            search_metrics: 搜索趋势指标
            options_metrics: 期权流情绪指标
            institutional_metrics: 机构情绪指标

        Returns:
            综合情绪指数
        """
        # 初始化各维度得分
        dimension_scores = {}

        # 社交媒体得分
        if social_metrics:
            social_score = 0
            weight_sum = 0
            for metric in ['twitter_sentiment_ma', 'reddit_sentiment_ma', 'mention_growth']:
                if metric in social_metrics:
                    weight = self.sentiment_weights['social_media'] / 3
                    social_score += social_metrics[metric].iloc[-1] * weight
                    weight_sum += weight
            dimension_scores['social'] = social_score / weight_sum if weight_sum > 0 else 0.5

        # 新闻得分
        if news_metrics:
            news_score = 0
            weight_sum = 0
            for metric in ['news_sentiment_ma', 'headline_sentiment_ma']:
                if metric in news_metrics:
                    weight = self.sentiment_weights['news'] / 2
                    news_score += news_metrics[metric].iloc[-1] * weight
                    weight_sum += weight
            dimension_scores['news'] = news_score / weight_sum if weight_sum > 0 else 0.5

        # 搜索趋势得分
        if search_metrics:
            search_score = 0
            weight_sum = 0
            for metric in ['search_volume_growth', 'buy_intent_ma']:
                if metric in search_metrics:
                    weight = self.sentiment_weights['search_trends'] / 2
                    search_score += search_metrics[metric].iloc[-1] * weight
                    weight_sum += weight
            dimension_scores['search'] = search_score / weight_sum if weight_sum > 0 else 0.5

        # 期权流得分
        if options_metrics:
            options_score = 0
            weight_sum = 0
            for metric in ['options_sentiment_ma', 'put_call_ratio_ma']:
                if metric in options_metrics:
                    weight = self.sentiment_weights['options_flow'] / 2
                    if metric == 'put_call_ratio_ma':
                        # PCR是反向指标
                        options_score += (1 - options_metrics[metric].iloc[-1]) * weight
                    else:
                        options_score += options_metrics[metric].iloc[-1] * weight
                    weight_sum += weight
            dimension_scores['options'] = options_score / weight_sum if weight_sum > 0 else 0.5

        # 机构得分
        if institutional_metrics:
            institutional_score = 0
            weight_sum = 0
            for metric in ['institutional_holdings_change', 'analyst_sentiment_ma']:
                if metric in institutional_metrics:
                    weight = self.sentiment_weights['institutional'] / 2
                    institutional_score += institutional_metrics[metric].iloc[-1] * weight
                    weight_sum += weight
            dimension_scores['institutional'] = institutional_score / weight_sum if weight_sum > 0 else 0.5

        # 计算加权综合得分
        composite_score = 0
        total_weight = 0

        for dimension, score in dimension_scores.items():
            weight = self.sentiment_weights.get(dimension, 0.2)
            composite_score += score * weight
            total_weight += weight

        final_score = composite_score / total_weight if total_weight > 0 else 0.5

        # 归一化到[0,1]范围
        final_score = np.clip(final_score, 0, 1)

        return pd.Series([final_score])

    def detect_sentiment_signals(self, social_metrics: Dict[str, pd.Series],
                              news_metrics: Dict[str, pd.Series],
                              search_metrics: Dict[str, pd.Series],
                              options_metrics: Dict[str, pd.Series],
                              institutional_metrics: Dict[str, pd.Series]) -> Dict[str, Any]:
        """
        检测情绪交易信号

        Args:
            social_metrics: 社交媒体情绪指标
            news_metrics: 新闻情绪指标
            search_metrics: 搜索趋势指标
            options_metrics: 期权流情绪指标
            institutional_metrics: 机构情绪指标

        Returns:
            情绪信号字典
        """
        signals = {}

        # 社交媒体信号
        if social_metrics:
            if 'twitter_sentiment_ma' in social_metrics:
                twitter_sentiment = social_metrics['twitter_sentiment_ma'].iloc[-1]
                if twitter_sentiment > 0.7:
                    signals['twitter_extreme_bullish'] = {'signal': 'sell', 'strength': (twitter_sentiment - 0.7) * 3}
                elif twitter_sentiment < 0.3:
                    signals['twitter_extreme_bearish'] = {'signal': 'buy', 'strength': (0.3 - twitter_sentiment) * 3}

            if 'mention_growth' in social_metrics:
                mention_growth = social_metrics['mention_growth'].iloc[-1]
                if mention_growth > 2.0:  # 提及量增长200%+
                    signals['mention_spike'] = {'signal': 'buy', 'strength': min(2, mention_growth / 2)}

        # 新闻情绪信号
        if news_metrics:
            if 'news_sentiment_ma' in news_metrics:
                news_sentiment = news_metrics['news_sentiment_ma'].iloc[-1]
                if news_sentiment > 0.8:
                    signals['news_extreme_positive'] = {'signal': 'sell', 'strength': (news_sentiment - 0.8) * 5}
                elif news_sentiment < 0.2:
                    signals['news_extreme_negative'] = {'signal': 'buy', 'strength': (0.2 - news_sentiment) * 5}

        # 搜索趋势信号
        if search_metrics:
            if 'buy_intent_ma' in search_metrics and 'sell_intent_ma' in search_metrics:
                buy_intent = search_metrics['buy_intent_ma'].iloc[-1]
                sell_intent = search_metrics['sell_intent_ma'].iloc[-1]
                buy_sell_ratio = buy_intent / (sell_intent + 1e-8)

                if buy_sell_ratio > 2.0:
                    signals['search_buy_dominance'] = {'signal': 'sell', 'strength': min(2, buy_sell_ratio / 2)}
                elif buy_sell_ratio < 0.5:
                    signals['search_sell_dominance'] = {'signal': 'buy', 'strength': min(2, 2 / buy_sell_ratio)}

        # 期权流信号
        if options_metrics:
            if 'put_call_ratio_ma' in options_metrics:
                pcr = options_metrics['put_call_ratio_ma'].iloc[-1]
                if pcr > 1.5:  # 高PCR，看跌情绪浓
                    signals['high_put_call_ratio'] = {'signal': 'buy', 'strength': min(2, pcr / 1.5)}
                elif pcr < 0.5:  # 低PCR，看涨情绪浓
                    signals['low_put_call_ratio'] = {'signal': 'sell', 'strength': min(2, 1.5 / pcr)}

        # 机构信号
        if institutional_metrics:
            if 'short_interest_ma' in institutional_metrics:
                short_interest = institutional_metrics['short_interest_ma'].iloc[-1]
                if short_interest > 0.1:  # 做空比率超过10%
                    signals['high_short_interest'] = {'signal': 'buy', 'strength': min(2, short_interest * 10)}

            if 'insider_buy_sell_ratio' in institutional_metrics:
                insider_ratio = institutional_metrics['insider_buy_sell_ratio'].iloc[-1]
                if insider_ratio > 2.0:  # 内部人大量买入
                    signals['insider_buying_spike'] = {'signal': 'buy', 'strength': min(2, insider_ratio / 2)}

        # 情绪分歧信号
        if social_metrics and 'sentiment_disagreement' in social_metrics:
            disagreement = social_metrics['sentiment_disagreement'].iloc[-1]
            if disagreement > 0.4:  # 高分歧
                signals['high_sentiment_disagreement'] = {'signal': 'neutral', 'strength': disagreement * 2}

        return signals

    def calculate_sentiment_indicators(self, social_data: pd.DataFrame,
                                    news_data: pd.DataFrame,
                                    search_data: pd.DataFrame,
                                    options_data: pd.DataFrame,
                                    institutional_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有情绪指标

        Args:
            social_data: 社交媒体数据
            news_data: 新闻数据
            search_data: 搜索数据
            options_data: 期权数据
            institutional_data: 机构数据

        Returns:
            情绪指标DataFrame
        """
        # 计算各类情绪指标
        social_metrics = self.calculate_social_media_sentiment(social_data) if social_data is not None else {}
        news_metrics = self.calculate_news_sentiment_metrics(news_data) if news_data is not None else {}
        search_metrics = self.calculate_search_trend_metrics(search_data) if search_data is not None else {}
        options_metrics = self.calculate_options_flow_sentiment(options_data) if options_data is not None else {}
        institutional_metrics = self.calculate_institutional_sentiment(institutional_data) if institutional_data is not None else {}

        # 创建指标DataFrame
        indicators = pd.DataFrame()

        # 添加社交媒体指标
        for name, series in social_metrics.items():
            indicators[f'social_{name}'] = series.iloc[-1] if len(series) > 0 else 0

        # 添加新闻指标
        for name, series in news_metrics.items():
            indicators[f'news_{name}'] = series.iloc[-1] if len(series) > 0 else 0

        # 添加搜索指标
        for name, series in search_metrics.items():
            indicators[f'search_{name}'] = series.iloc[-1] if len(series) > 0 else 0

        # 添加期权指标
        for name, series in options_metrics.items():
            indicators[f'options_{name}'] = series.iloc[-1] if len(series) > 0 else 0

        # 添加机构指标
        for name, series in institutional_metrics.items():
            indicators[f'institutional_{name}'] = series.iloc[-1] if len(series) > 0 else 0

        # 计算综合情绪指数
        if any([social_metrics, news_metrics, search_metrics, options_metrics, institutional_metrics]):
            composite_sentiment = self.calculate_composite_sentiment_index(
                social_metrics, news_metrics, search_metrics, options_metrics, institutional_metrics
            )
            indicators['composite_sentiment_index'] = composite_sentiment.iloc[0] if len(composite_sentiment) > 0 else 0.5

        # 检测情绪信号
        sentiment_signals = self.detect_sentiment_signals(
            social_metrics, news_metrics, search_metrics, options_metrics, institutional_metrics
        )

        # 计算信号强度
        buy_strength = sum(signal['strength'] for signal in sentiment_signals.values() if signal['signal'] == 'buy')
        sell_strength = sum(signal['strength'] for signal in sentiment_signals.values() if signal['signal'] == 'sell')

        indicators['buy_sentiment_signal'] = buy_strength
        indicators['sell_sentiment_signal'] = sell_strength
        indicators['net_sentiment_signal'] = buy_strength - sell_strength

        # 情绪极值检测
        if 'composite_sentiment_index' in indicators.columns:
            sentiment_index = indicators['composite_sentiment_index'].iloc[0]
            if sentiment_index > 0.8:
                indicators['extreme_bullish_sentiment'] = 3
            elif sentiment_index < 0.2:
                indicators['extreme_bearish_sentiment'] = -3
            elif sentiment_index > 0.6:
                indicators['bullish_sentiment'] = 2
            elif sentiment_index < 0.4:
                indicators['bearish_sentiment'] = -2
            else:
                indicators['neutral_sentiment'] = 1

        # 情绪变化率
        if 'social_twitter_sentiment_ma' in indicators.columns:
            twitter_sentiment = indicators['social_twitter_sentiment_ma'].iloc[0]
            indicators['sentiment_momentum'] = twitter_sentiment - 0.5  # 相对于中性的偏差

        return indicators.T


class SentimentDivergenceAnalyzer:
    """
    情绪分歧分析器

    分析不同情绪源之间的分歧和冲突
    """

    def __init__(self):
        """初始化情绪分歧分析器"""
        self.divergence_threshold = 0.3

    def calculate_sentiment_divergence(self, sentiment_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        计算情绪分歧指标

        Args:
            sentiment_metrics: 各维度情绪指标

        Returns:
            情绪分歧分析结果
        """
        if not sentiment_metrics:
            return {}

        # 计算情绪维度间的标准差
        sentiment_values = list(sentiment_metrics.values())
        sentiment_std = np.std(sentiment_values)

        # 计算最大最小差异
        max_sentiment = max(sentiment_values)
        min_sentiment = min(sentiment_values)
        sentiment_range = max_sentiment - min_sentiment

        # 识别分歧源
        divergent_sources = []
        for source, sentiment in sentiment_metrics.items():
            if abs(sentiment - np.mean(sentiment_values)) > self.divergence_threshold:
                divergent_sources.append(source)

        # 计算分歧强度
        divergence_strength = sentiment_std * 2  # 放大标准差

        return {
            'sentiment_std': sentiment_std,
            'sentiment_range': sentiment_range,
            'divergent_sources': divergent_sources,
            'divergence_strength': divergence_strength,
            'convergence_score': 1 - min(1, divergence_strength),
            'max_divergence': sentiment_range
        }

    def detect_sentiment_conflicts(self, sentiment_signals: Dict[str, Any]) -> Dict[str, List[str]]:
        """
        检测情绪信号冲突

        Args:
            sentiment_signals: 情绪信号

        Returns:
            信号冲突分析
        """
        conflicts = {'buy_conflicts': [], 'sell_conflicts': [], 'mixed_signals': []}

        # 按信号类型分组
        buy_signals = [name for name, signal in sentiment_signals.items() if signal.get('signal') == 'buy']
        sell_signals = [name for name, signal in sentiment_signals.items() if signal.get('signal') == 'sell']

        # 检测冲突
        if buy_signals and sell_signals:
            conflicts['mixed_signals'] = buy_signals + sell_signals

        # 检测强度冲突
        strong_buy = [name for name, signal in sentiment_signals.items()
                     if signal.get('signal') == 'buy' and signal.get('strength', 0) > 1.5]
        weak_buy = [name for name, signal in sentiment_signals.items()
                   if signal.get('signal') == 'buy' and signal.get('strength', 0) < 0.5]

        if strong_buy and weak_buy:
            conflicts['buy_conflicts'] = strong_buy + weak_buy

        strong_sell = [name for name, signal in sentiment_signals.items()
                       if signal.get('signal') == 'sell' and signal.get('strength', 0) > 1.5]
        weak_sell = [name for name, signal in sentiment_signals.items()
                    if signal.get('signal') == 'sell' and signal.get('strength', 0) < 0.5]

        if strong_sell and weak_sell:
            conflicts['sell_conflicts'] = strong_sell + weak_sell

        return conflicts


def create_sentiment_features(social_data: pd.DataFrame, news_data: pd.DataFrame,
                            search_data: pd.DataFrame, options_data: pd.DataFrame,
                            institutional_data: pd.DataFrame) -> pd.DataFrame:
    """
    创建情绪分析特征

    Args:
        social_data: 社交媒体数据
        news_data: 新闻数据
        search_data: 搜索数据
        options_data: 期权数据
        institutional_data: 机构数据

    Returns:
        情绪分析特征DataFrame
    """
    analyzer = SentimentAnalyzer()
    indicators = analyzer.calculate_sentiment_indicators(
        social_data, news_data, search_data, options_data, institutional_data
    )
    return indicators


# 主要功能函数
def calculate_sentiment_indicators(social_data: pd.DataFrame = None,
                                news_data: pd.DataFrame = None,
                                search_data: pd.DataFrame = None,
                                options_data: pd.DataFrame = None,
                                institutional_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算所有情绪分析指标

    Args:
        social_data: 社交媒体数据DataFrame（可选）
        news_data: 新闻数据DataFrame（可选）
        search_data: 搜索数据DataFrame（可选）
        options_data: 期权数据DataFrame（可选）
        institutional_data: 机构数据DataFrame（可选）

    Returns:
        包含所有指标值的DataFrame
    """
    return create_sentiment_features(social_data, news_data, search_data, options_data, institutional_data)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

    # 模拟社交媒体数据
    social_data = pd.DataFrame({
        'twitter_sentiment': np.random.uniform(0.3, 0.7, 100),
        'reddit_sentiment': np.random.uniform(0.2, 0.8, 100),
        'mention_count': np.random.randint(1000, 10000, 100),
        'sentiment_std': np.random.uniform(0.1, 0.4, 100),
        'positive_ratio': np.random.uniform(0.4, 0.6, 100),
        'negative_ratio': np.random.uniform(0.2, 0.4, 100)
    }, index=dates)

    # 模拟新闻数据
    news_data = pd.DataFrame({
        'news_sentiment_score': np.random.uniform(0.35, 0.65, 100),
        'news_count': np.random.randint(10, 100, 100),
        'headline_sentiment': np.random.uniform(0.3, 0.7, 100),
        'extreme_positive_news': np.random.randint(0, 5, 100),
        'extreme_negative_news': np.random.randint(0, 5, 100)
    }, index=dates)

    # 模拟搜索数据
    search_data = pd.DataFrame({
        'search_volume': np.random.randint(100, 1000, 100),
        'buy_search_ratio': np.random.uniform(0.4, 0.7, 100),
        'sell_search_ratio': np.random.uniform(0.2, 0.5, 100),
        'prediction_searches': np.random.randint(10, 100, 100),
        'information_searches': np.random.randint(20, 200, 100)
    }, index=dates)

    # 模拟期权数据
    options_data = pd.DataFrame({
        'put_call_ratio': np.random.uniform(0.5, 1.5, 100),
        'options_volume': np.random.randint(1000, 10000, 100),
        'options_sentiment_index': np.random.uniform(0.3, 0.7, 100),
        'large_options_trades': np.random.randint(0, 10, 100)
    }, index=dates)

    # 模拟机构数据
    institutional_data = pd.DataFrame({
        'institutional_holdings': np.random.uniform(0.5, 0.8, 100),
        'short_interest_ratio': np.random.uniform(0.02, 0.08, 100),
        'insider_trading_ratio': np.random.uniform(0.8, 1.5, 100),
        'analyst_ratings': np.random.uniform(3.5, 4.5, 100),
        'institutional_flows': np.random.uniform(-100, 100, 100)
    }, index=dates)

    # 计算指标
    try:
        indicators = calculate_sentiment_indicators(
            social_data, news_data, search_data, options_data, institutional_data
        )
        print("情绪分析指标计算成功!")
        print(f"指标数量: {indicators.shape[0]}")
        print("最新指标值:")
        print(indicators)

    except Exception as e:
        print(f"计算错误: {e}")