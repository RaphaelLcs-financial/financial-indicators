"""
波动率模式分析器金融指标

本模块实现了专门针对加密货币波动率模式的分析指标，包括历史波动率、隐含波动率、波动率曲面等。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class VolatilityPatternAnalyzer:
    """
    加密货币波动率模式分析器

    分析加密货币特有的波动率模式和特征
    """

    def __init__(self,
                 volatility_window: int = 30,
                 garch_window: int = 100,
                 regime_threshold: float = 1.5):
        """
        初始化波动率模式分析器

        Args:
            volatility_window: 波动率计算窗口
            garch_window: GARCH模型窗口
            regime_threshold: 波动率状态切换阈值
        """
        self.volatility_window = volatility_window
        self.garch_window = garch_window
        self.regime_threshold = regime_threshold

        # 波动率状态
        self.volatility_regimes = []
        self.volatility_clusters = []

    def calculate_crypto_volatility_metrics(self, price_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算加密货币特有波动率指标

        Args:
            price_data: 价格数据

        Returns:
            波动率指标字典
        """
        metrics = {}

        returns = price_data['close'].pct_change().dropna()

        # 历史波动率（多时间框架）
        for period in [7, 14, 30, 60, 90]:
            volatility = returns.rolling(period).std() * np.sqrt(252)
            metrics[f'hv_{period}d'] = volatility

        # 加权历史波动率（指数加权）
        metrics['ewma_volatility'] = returns.ewm(span=self.volatility_window).std() * np.sqrt(252)

        # Parkinson波动率（基于高低价）
        parkinson_volatility = self._calculate_parkinson_volatility(price_data)
        metrics['parkinson_volatility'] = parkinson_volatility.rolling(self.volatility_window).mean() * np.sqrt(252)

        # Garman-Klass波动率
        gk_volatility = self._calculate_garman_klass_volatility(price_data)
        metrics['garman_klass_volatility'] = gk_volatility.rolling(self.volatility_window).mean() * np.sqrt(252)

        # Yang-Zhang波动率
        yz_volatility = self._calculate_yang_zhang_volatility(price_data)
        metrics['yang_zhang_volatility'] = yz_volatility.rolling(self.volatility_window).mean() * np.sqrt(252)

        # 实际波动率（高频）
        if 'high_frequency_data' in price_data.columns:
            realized_vol = self._calculate_realized_volatility(price_data['high_frequency_data'])
            metrics['realized_volatility'] = realized_vol

        # 波动率偏度和峰度
        metrics['volatility_skewness'] = returns.rolling(self.volatility_window).skew()
        metrics['volatility_kurtosis'] = returns.rolling(self.volatility_window).kurtosis()

        # 波动率集中度
        metrics['volatility_concentration'] = self._calculate_volatility_concentration(returns)

        return metrics

    def _calculate_parkinson_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """计算Parkinson波动率"""
        high_low_ratio = np.log(price_data['high'] / price_data['low'])
        parkinson = high_low_ratio / (2 * np.sqrt(np.log(2)))
        return parkinson

    def _calculate_garman_klass_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """计算Garman-Klass波动率"""
        log_hl = np.log(price_data['high'] / price_data['low'])
        log_co = np.log(price_data['close'] / price_data['open'])

        gk = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
        return np.sqrt(gk)

    def _calculate_yang_zhang_volatility(self, price_data: pd.DataFrame) -> pd.Series:
        """计算Yang-Zhang波动率"""
        # 开盘到收盘的波动率
        log_co = np.log(price_data['close'] / price_data['open'])
        volatility_oc = log_co**2

        # 隔夜波动率
        log_open_close = np.log(price_data['open'] / price_data['close'].shift(1))
        volatility_overnight = log_open_close**2

        # Parkinson波动率
        log_hl = np.log(price_data['high'] / price_data['low'])
        volatility_parkinson = log_hl**2 / (4 * np.log(2))

        # Yang-Zhang公式
        k = 0.34 / (1.34 + (self.volatility_window + 1) / (self.volatility_window - 1))
        yz_variance = volatility_overnight.rolling(self.volatility_window).mean() + \
                     k * volatility_oc.rolling(self.volatility_window).mean() + \
                     (1 - k) * volatility_parkinson.rolling(self.volatility_window).mean()

        return np.sqrt(yz_variance)

    def _calculate_realized_volatility(self, high_freq_data: pd.DataFrame) -> pd.Series:
        """计算实际波动率（高频数据）"""
        if high_freq_data is None or len(high_freq_data) == 0:
            return pd.Series(dtype=float)

        # 基于高频收益率计算
        hf_returns = high_freq_data.pct_change().dropna()
        realized_variance = hf_returns**2

        return np.sqrt(realized_variance.sum())

    def _calculate_volatility_concentration(self, returns: pd.Series) -> pd.Series:
        """计算波动率集中度"""
        concentration = []

        for i in range(self.volatility_window, len(returns)):
            window_returns = returns.iloc[i-self.volatility_window:i]

            # 计算绝对收益率
            abs_returns = abs(window_returns)

            # 计算集中度（大波动日的占比）
            threshold = abs_returns.quantile(0.8)
            large_moves = (abs_returns > threshold).sum()
            concentration_ratio = large_moves / len(window_returns)

            concentration.append(concentration_ratio)

        return pd.Series(concentration, index=returns.index[self.volatility_window:])

    def calculate_garch_volatility(self, returns: pd.Series) -> Dict[str, Any]:
        """
        计算GARCH波动率

        Args:
            returns: 收益率序列

        Returns:
            GARCH模型结果
        """
        if len(returns) < self.garch_window:
            return {'error': 'Insufficient data for GARCH model'}

        # 简化的GARCH(1,1)模型
        try:
            # 初始化参数
            omega = 0.0001  # 长期方差
            alpha = 0.1     # ARCH系数
            beta = 0.85     # GARCH系数

            # 计算条件方差
            conditional_var = []
            variances = [returns.var()]  # 初始化方差

            for i in range(1, len(returns)):
                # GARCH(1,1)公式
                var = omega + alpha * returns.iloc[i-1]**2 + beta * variances[-1]
                variances.append(var)
                conditional_var.append(var)

            # 计算波动率
            garch_volatility = np.sqrt(variances[1:])  # 去掉初始值
            garch_series = pd.Series(garch_volatility, index=returns.index[1:])

            # 预测下一期波动率
            last_return = returns.iloc[-1]**2
            predicted_variance = omega + alpha * last_return + beta * variances[-1]
            predicted_volatility = np.sqrt(predicted_variance)

            return {
                'garch_volatility': garch_series,
                'current_volatility': garch_series.iloc[-1],
                'predicted_volatility': predicted_volatility,
                'model_parameters': {'omega': omega, 'alpha': alpha, 'beta': beta},
                'persistence': alpha + beta  # 持续性参数
            }

        except Exception as e:
            return {'error': str(e)}

    def detect_volatility_regimes(self, returns: pd.Series) -> Dict[str, Any]:
        """
        检测波动率状态

        Args:
            returns: 收益率序列

        Returns:
            波动率状态分析结果
        """
        # 计算滚动波动率
        rolling_vol = returns.rolling(self.volatility_window).std()

        # 计算波动率分位数
        vol_quantiles = rolling_vol.quantile([0.25, 0.5, 0.75])

        # 定义状态
        low_vol_threshold = vol_quantiles[0.25]
        high_vol_threshold = vol_quantiles[0.75]

        # 分类波动率状态
        regimes = []
        for vol in rolling_vol.dropna():
            if vol < low_vol_threshold:
                regime = 'low'
            elif vol > high_vol_threshold:
                regime = 'high'
            else:
                regime = 'normal'

            regimes.append(regime)

        regime_series = pd.Series(regimes, index=rolling_vol.dropna().index)

        # 计算状态转移概率
        transition_matrix = self._calculate_transition_matrix(regime_series)

        # 计算状态持续时间
        regime_durations = self._calculate_regime_durations(regime_series)

        return {
            'regime_series': regime_series,
            'current_regime': regime_series.iloc[-1],
            'transition_matrix': transition_matrix,
            'regime_durations': regime_durations,
            'low_vol_threshold': low_vol_threshold,
            'high_vol_threshold': high_vol_threshold
        }

    def _calculate_transition_matrix(self, regime_series: pd.Series) -> Dict[str, Dict[str, float]]:
        """计算状态转移矩阵"""
        unique_regimes = regime_series.unique()
        transition_counts = {from_regime: {to_regime: 0 for to_regime in unique_regimes}
                           for from_regime in unique_regimes}

        # 统计转移
        for i in range(1, len(regime_series)):
            from_regime = regime_series.iloc[i-1]
            to_regime = regime_series.iloc[i]
            transition_counts[from_regime][to_regime] += 1

        # 计算转移概率
        transition_matrix = {}
        for from_regime in unique_regimes:
            total_transitions = sum(transition_counts[from_regime].values())
            if total_transitions > 0:
                transition_matrix[from_regime] = {
                    to_regime: count / total_transitions
                    for to_regime, count in transition_counts[from_regime].items()
                }
            else:
                transition_matrix[from_regime] = {to_regime: 0 for to_regime in unique_regimes}

        return transition_matrix

    def _calculate_regime_durations(self, regime_series: pd.Series) -> Dict[str, float]:
        """计算状态持续时间"""
        durations = {'low': [], 'normal': [], 'high': []}

        current_regime = None
        current_duration = 0

        for regime in regime_series:
            if regime == current_regime:
                current_duration += 1
            else:
                if current_regime is not None:
                    durations[current_regime].append(current_duration)
                current_regime = regime
                current_duration = 1

        # 添加最后一个状态
        if current_regime is not None:
            durations[current_regime].append(current_duration)

        # 计算平均持续时间
        avg_durations = {}
        for regime, dur_list in durations.items():
            if dur_list:
                avg_durations[regime] = np.mean(dur_list)
            else:
                avg_durations[regime] = 0

        return avg_durations

    def calculate_volatility_surface_metrics(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算波动率曲面指标

        Args:
            option_data: 期权数据

        Returns:
            波动率曲面分析结果
        """
        if option_data is None or len(option_data) == 0:
            return {'error': 'No option data provided'}

        metrics = {}

        try:
            # 计算隐含波动率偏度
            if 'implied_volatility' in option_data.columns and 'moneyness' in option_data.columns:
                iv_skew = self._calculate_iv_skew(option_data)
                metrics['iv_skew'] = iv_skew

            # 计算波动率曲面斜率
            if 'term' in option_data.columns and 'implied_volatility' in option_data.columns:
                term_structure = self._calculate_term_structure(option_data)
                metrics['term_structure'] = term_structure

            # 计算波动率锥
            volatility_cone = self._calculate_volatility_cone(option_data)
            metrics['volatility_cone'] = volatility_cone

        except Exception as e:
            metrics['error'] = str(e)

        return metrics

    def _calculate_iv_skew(self, option_data: pd.DataFrame) -> Dict[str, float]:
        """计算隐含波动率偏度"""
        # 按moneyness分组
        otm_options = option_data[option_data['moneyness'] < 0.95]  # 虚值期权
        itm_options = option_data[option_data['moneyness'] > 1.05]  # 实值期权
        atm_options = option_data[(option_data['moneyness'] >= 0.95) & (option_data['moneyness'] <= 1.05)]  # 平值期权

        # 计算各组平均IV
        otm_iv = otm_options['implied_volatility'].mean() if len(otm_options) > 0 else 0
        itm_iv = itm_options['implied_volatility'].mean() if len(itm_options) > 0 else 0
        atm_iv = atm_options['implied_volatility'].mean() if len(atm_options) > 0 else 0

        # 计算偏度
        skew_25_delta = otm_iv - atm_iv if atm_iv > 0 else 0
        risk_reversal = otm_iv - itm_iv if itm_iv > 0 else 0

        return {
            'skew_25_delta': skew_25_delta,
            'risk_reversal': risk_reversal,
            'atm_iv': atm_iv,
            'otm_iv': otm_iv,
            'itm_iv': itm_iv
        }

    def _calculate_term_structure(self, option_data: pd.DataFrame) -> Dict[str, float]:
        """计算期限结构"""
        # 按到期时间分组
        short_term = option_data[option_data['term'] <= 30]  # 30天以内
        medium_term = option_data[(option_data['term'] > 30) & (option_data['term'] <= 90)]  # 30-90天
        long_term = option_data[option_data['term'] > 90]  # 90天以上

        # 计算各期限平均IV
        short_iv = short_term['implied_volatility'].mean() if len(short_term) > 0 else 0
        medium_iv = medium_term['implied_volatility'].mean() if len(medium_term) > 0 else 0
        long_iv = long_term['implied_volatility'].mean() if len(long_term) > 0 else 0

        # 计算期限结构斜率
        short_medium_slope = medium_iv - short_iv
        medium_long_slope = long_iv - medium_iv

        return {
            'short_term_iv': short_iv,
            'medium_term_iv': medium_iv,
            'long_term_iv': long_iv,
            'short_medium_slope': short_medium_slope,
            'medium_long_slope': medium_long_slope,
            'contango_backwardation': 'contango' if medium_long_slope > 0 else 'backwardation'
        }

    def _calculate_volatility_cone(self, option_data: pd.DataFrame) -> Dict[str, float]:
        """计算波动率锥"""
        # 计算不同期限的波动率范围
        volatility_ranges = {}

        for term in [7, 14, 30, 60, 90]:
            term_options = option_data[option_data['term'] == term]
            if len(term_options) > 0:
                ivs = term_options['implied_volatility']
                volatility_ranges[f'{term}d_min'] = ivs.min()
                volatility_ranges[f'{term}d_max'] = ivs.max()
                volatility_ranges[f'{term}d_median'] = ivs.median()

        return volatility_ranges

    def calculate_volatility_trading_signals(self, price_data: pd.DataFrame,
                                          option_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        计算波动率交易信号

        Args:
            price_data: 价格数据
            option_data: 期权数据（可选）

        Returns:
            波动率交易信号
        """
        signals = {}

        returns = price_data['close'].pct_change().dropna()

        # 计算波动率指标
        vol_metrics = self.calculate_crypto_volatility_metrics(price_data)
        garch_result = self.calculate_garch_volatility(returns)
        regime_result = self.detect_volatility_regimes(returns)

        # 历史波动率信号
        if 'hv_30d' in vol_metrics:
            current_hv = vol_metrics['hv_30d'].iloc[-1]
            hv_percentile = (vol_metrics['hv_30d'].rank(pct=True).iloc[-1])

            if hv_percentile > 0.8:  # 历史高位
                signals['high_historical_vol'] = {'signal': 'sell_vol', 'strength': hv_percentile}
            elif hv_percentile < 0.2:  # 历史低位
                signals['low_historical_vol'] = {'signal': 'buy_vol', 'strength': 1 - hv_percentile}

        # GARCH预测信号
        if 'predicted_volatility' in garch_result:
            predicted_vol = garch_result['predicted_volatility']
            current_garch = garch_result['current_volatility']

            if predicted_vol > current_garch * 1.2:  # 预测波动率上升
                signals['garch_vol_increase'] = {'signal': 'buy_vol', 'strength': min(1, predicted_vol / current_garch - 1)}
            elif predicted_vol < current_garch * 0.8:  # 预测波动率下降
                signals['garch_vol_decrease'] = {'signal': 'sell_vol', 'strength': min(1, 1 - predicted_vol / current_garch)}

        # 波动率状态信号
        if 'current_regime' in regime_result:
            current_regime = regime_result['current_regime']

            if current_regime == 'low':  # 低波动率状态
                signals['low_vol_regime'] = {'signal': 'buy_vol', 'strength': 0.7}
            elif current_regime == 'high':  # 高波动率状态
                signals['high_vol_regime'] = {'signal': 'sell_vol', 'strength': 0.7}

        # 波动率集中度信号
        if 'volatility_concentration' in vol_metrics:
            concentration = vol_metrics['volatility_concentration'].iloc[-1]
            if concentration > 0.6:  # 波动率集中度高
                signals['vol_concentration_high'] = {'signal': 'buy_vol', 'strength': concentration}

        # 期权相关信号（如果有期权数据）
        if option_data is not None and len(option_data) > 0:
            iv_surface = self.calculate_volatility_surface_metrics(option_data)

            if 'iv_skew' in iv_surface:
                skew = iv_surface['iv_skew']['skew_25_delta']
                if skew > 0.1:  # 高偏度
                    signals['high_iv_skew'] = {'signal': 'sell_vol_skew', 'strength': min(1, skew * 5)}

            if 'term_structure' in iv_surface:
                term_slope = iv_surface['term_structure']['medium_long_slope']
                if term_slope < -0.05:  # backwardation
                    signals['term_backwardation'] = {'signal': 'buy_vol_term', 'strength': min(1, abs(term_slope) * 10)}

        # 波动率均值回归信号
        if 'hv_30d' in vol_metrics:
            hv_ma = vol_metrics['hv_30d'].rolling(60).mean().iloc[-1]
            current_hv = vol_metrics['hv_30d'].iloc[-1]

            if current_hv > hv_ma * 1.3:  # 显著高于均值
                signals['vol_mean_reversion_sell'] = {'signal': 'sell_vol', 'strength': min(1, current_hv / hv_ma - 1)}
            elif current_hv < hv_ma * 0.7:  # 显著低于均值
                signals['vol_mean_reversion_buy'] = {'signal': 'buy_vol', 'strength': min(1, 1 - current_hv / hv_ma)}

        return signals

    def calculate_volatility_indicators(self, price_data: pd.DataFrame,
                                     option_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        计算所有波动率指标

        Args:
            price_data: 价格数据
            option_data: 期权数据（可选）

        Returns:
            波动率指标DataFrame
        """
        if len(price_data) < self.volatility_window:
            return pd.DataFrame()

        # 计算波动率指标
        vol_metrics = self.calculate_crypto_volatility_metrics(price_data)

        returns = price_data['close'].pct_change().dropna()
        garch_result = self.calculate_garch_volatility(returns)
        regime_result = self.detect_volatility_regimes(returns)

        # 创建指标DataFrame
        indicators = pd.DataFrame(index=price_data.index[-1:])

        # 添加波动率指标
        for name, series in vol_metrics.items():
            indicators[name] = series.iloc[-1]

        # 添加GARCH结果
        if 'current_volatility' in garch_result:
            indicators['garch_volatility'] = garch_result['current_volatility']
            indicators['predicted_garch_vol'] = garch_result['predicted_volatility']
            indicators['garch_persistence'] = garch_result['persistence']

        # 添加状态信息
        if 'current_regime' in regime_result:
            indicators['volatility_regime'] = {'low': 1, 'normal': 2, 'high': 3}[regime_result['current_regime']]

        # 计算波动率交易信号
        trading_signals = self.calculate_volatility_trading_signals(price_data, option_data)

        # 计算信号强度
        buy_vol_strength = sum(signal['strength'] for signal in trading_signals.values()
                             if signal['signal'] in ['buy_vol', 'buy_vol_skew', 'buy_vol_term'])
        sell_vol_strength = sum(signal['strength'] for signal in trading_signals.values()
                              if signal['signal'] in ['sell_vol', 'sell_vol_skew'])

        indicators['buy_vol_signal'] = buy_vol_strength
        indicators['sell_vol_signal'] = sell_vol_strength
        indicators['net_vol_signal'] = buy_vol_strength - sell_vol_strength

        # 波动率风险指标
        if 'hv_30d' in vol_metrics and 'volatility_kurtosis' in vol_metrics:
            current_vol = vol_metrics['hv_30d'].iloc[-1]
            vol_kurtosis = vol_metrics['volatility_kurtosis'].iloc[-1]

            # 波动率风险评分
            vol_risk = current_vol * (1 + abs(vol_kurtosis) / 10)
            indicators['volatility_risk_score'] = min(5, vol_risk * 100)

        return indicators


class CryptoVolatilitySurface:
    """
    加密货币波动率曲面

    构建和分析加密货币期权的波动率曲面
    """

    def __init__(self):
        """初始化波动率曲面"""
        self.surface_points = {}
        self.interpolated_surface = None

    def build_volatility_surface(self, option_data: pd.DataFrame) -> Dict[str, Any]:
        """
        构建波动率曲面

        Args:
            option_data: 期权数据

        Returns:
            波动率曲面分析结果
        """
        if option_data is None or len(option_data) == 0:
            return {'error': 'No option data provided'}

        # 数据预处理
        clean_data = self._preprocess_option_data(option_data)

        # 构建曲面
        surface = self._interpolate_surface(clean_data)

        # 计算曲面特征
        surface_features = self._calculate_surface_features(surface)

        return {
            'surface': surface,
            'features': surface_features,
            'data_quality': self._assess_data_quality(clean_data)
        }

    def _preprocess_option_data(self, option_data: pd.DataFrame) -> pd.DataFrame:
        """预处理期权数据"""
        clean_data = option_data.copy()

        # 移除异常值
        if 'implied_volatility' in clean_data.columns:
            iv_mean = clean_data['implied_volatility'].mean()
            iv_std = clean_data['implied_volatility'].std()

            # 保留3个标准差内的数据
            clean_data = clean_data[
                (clean_data['implied_volatility'] > iv_mean - 3 * iv_std) &
                (clean_data['implied_volatility'] < iv_mean + 3 * iv_std)
            ]

        return clean_data

    def _interpolate_surface(self, option_data: pd.DataFrame) -> Dict[str, np.ndarray]:
        """插值构建波动率曲面"""
        if 'moneyness' not in option_data.columns or 'term' not in option_data.columns:
            return {'error': 'Missing required columns for surface construction'}

        # 创建网格
        moneyness_grid = np.linspace(0.7, 1.3, 50)  # Moneyness从0.7到1.3
        term_grid = np.linspace(7, 365, 50)          # Term从7天到365天

        # 插值
        from scipy.interpolate import griddata

        points = option_data[['moneyness', 'term']].values
        values = option_data['implied_volatility'].values

        # 网格插值
        M, T = np.meshgrid(moneyness_grid, term_grid)
        IV = griddata(points, values, (M, T), method='linear')

        return {
            'moneyness_grid': moneyness_grid,
            'term_grid': term_grid,
            'iv_surface': IV,
            'mesh_m': M,
            'mesh_t': T
        }

    def _calculate_surface_features(self, surface: Dict[str, np.ndarray]) -> Dict[str, float]:
        """计算波动率曲面特征"""
        if 'iv_surface' not in surface:
            return {}

        iv_surface = surface['iv_surface']

        # 计算曲面斜率
        moneyness_slope = np.gradient(iv_surface, axis=1)  # Moneyness方向
        term_slope = np.gradient(iv_surface, axis=0)        # Term方向

        # 计算曲面曲率
        moneyness_curvature = np.gradient(moneyness_slope, axis=1)
        term_curvature = np.gradient(term_slope, axis=0)

        # 计算统计特征
        surface_features = {
            'avg_iv': np.nanmean(iv_surface),
            'iv_std': np.nanstd(iv_surface),
            'max_skew': np.nanmax(moneyness_slope),
            'min_skew': np.nanmin(moneyness_slope),
            'avg_term_slope': np.nanmean(term_slope),
            'surface_curvature': np.nanmean(abs(moneyness_curvature) + abs(term_curvature))
        }

        return surface_features

    def _assess_data_quality(self, option_data: pd.DataFrame) -> Dict[str, float]:
        """评估数据质量"""
        quality_metrics = {}

        if len(option_data) == 0:
            quality_metrics['data_coverage'] = 0
            return quality_metrics

        # 数据覆盖率
        moneyness_range = option_data['moneyness'].max() - option_data['moneyness'].min()
        term_range = option_data['term'].max() - option_data['term'].min()

        quality_metrics['data_coverage'] = min(1.0, moneyness_range / 0.6) * min(1.0, term_range / 358)

        # 数据密度
        quality_metrics['data_density'] = len(option_data) / 100  # 每100个理想数据点的比例

        # 数据一致性
        if 'implied_volatility' in option_data.columns:
            iv_std = option_data['implied_volatility'].std()
            iv_mean = option_data['implied_volatility'].mean()
            quality_metrics['data_consistency'] = 1 / (1 + iv_std / iv_mean) if iv_mean > 0 else 0

        return quality_metrics


def create_volatility_pattern_features(price_data: pd.DataFrame, option_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    创建波动率模式特征

    Args:
        price_data: 价格数据
        option_data: 期权数据（可选）

    Returns:
    """
    analyzer = VolatilityPatternAnalyzer()
    indicators = analyzer.calculate_volatility_indicators(price_data, option_data)
    return indicators


# 主要功能函数
def calculate_volatility_pattern_indicators(price_data: pd.DataFrame, option_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    计算所有波动率模式指标

    Args:
        price_data: 包含OHLC数据的DataFrame
        option_data: 期权数据DataFrame（可选）

    Returns:
        包含所有指标值的DataFrame
    """
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in price_data.columns for col in required_columns):
        raise ValueError("价格数据必须包含 'open', 'high', 'low', 'close' 列")

    if len(price_data) < 30:
        raise ValueError("价格数据长度不足，至少需要30个数据点")

    return create_volatility_pattern_features(price_data, option_data)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

    # 模拟加密货币价格数据（高波动率）
    base_returns = np.random.normal(0.002, 0.05, 200)  # 更高的波动率
    prices = 50000 * np.exp(np.cumsum(base_returns))

    sample_price_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.02))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.02))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000000, 50000000000, 200)
    }, index=dates)

    # 模拟期权数据
    np.random.seed(42)
    n_options = 100
    option_sample = pd.DataFrame({
        'moneyness': np.random.uniform(0.7, 1.3, n_options),
        'term': np.random.uniform(7, 365, n_options),
        'implied_volatility': np.random.uniform(0.5, 2.0, n_options)
    })

    # 计算指标
    try:
        indicators = calculate_volatility_pattern_indicators(sample_price_data, option_sample)
        print("波动率模式指标计算成功!")
        print(f"指标数量: {indicators.shape[1]}")
        print("最新指标值:")
        print(indicators.iloc[-1])

        # 检测交易信号
        analyzer = VolatilityPatternAnalyzer()
        signals = analyzer.calculate_volatility_trading_signals(sample_price_data, option_sample)
        print(f"\n检测到 {len(signals)} 个波动率交易信号:")
        for signal_name, signal_info in signals.items():
            print(f"{signal_name}: {signal_info['signal']} (强度: {signal_info['strength']:.2f})")

    except Exception as e:
        print(f"计算错误: {e}")