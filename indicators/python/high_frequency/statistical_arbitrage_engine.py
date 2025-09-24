"""
统计套利引擎金融指标

本模块实现了基于统计套利理论的高频交易指标系统，包括配对交易、均值回归、协整分析等策略。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class StatisticalArbitrageEngine:
    """
    统计套利引擎

    实现多种统计套利策略，包括协整分析、配对交易、均值回归等
    """

    def __init__(self,
                 lookback_window: int = 100,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 half_life_threshold: float = 20):
        """
        初始化统计套利引擎

        Args:
            lookback_window: 回看窗口
            entry_threshold: 入场阈值（标准差倍数）
            exit_threshold: 出场阈值（标准差倍数）
            half_life_threshold: 半衰期阈值
        """
        self.lookback_window = lookback_window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.half_life_threshold = half_life_threshold

        # 协整关系
        self.cointegration_pairs = {}
        self.hedge_ratios = {}

        # 统计量
        self.spread_statistics = {}
        self.mean_reversion_metrics = {}

    def test_cointegration(self, series1: pd.Series, series2: pd.Series) -> Dict[str, Any]:
        """
        检验协整关系

        Args:
            series1: 第一个价格序列
            series2: 第二个价格序列

        Returns:
            协整检验结果
        """
        # 对齐数据
        aligned_data = pd.concat([series1, series2], axis=1).dropna()
        if len(aligned_data) < 30:
            return {'cointegrated': False, 'error': 'Insufficient data'}

        y = aligned_data.iloc[:, 0]
        x = aligned_data.iloc[:, 1]

        # 计算对数价格
        log_y = np.log(y)
        log_x = np.log(x)

        # 线性回归计算对冲比率
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(log_x, log_y)

            # 计算价差
            spread = log_y - (slope * log_x + intercept)

            # ADF检验（简化版，使用自相关检验）
            spread_autocorr = spread.autocorr()
            spread_volatility = spread.std()

            # 简化的协整检验标准
            cointegrated = (abs(spread_autocorr) < 0.3 and spread_volatility < 0.1 and r_value**2 > 0.7)

            # 计算半衰期
            half_life = self._calculate_half_life(spread)

            return {
                'cointegrated': cointegrated,
                'hedge_ratio': slope,
                'intercept': intercept,
                'r_squared': r_value**2,
                'p_value': p_value,
                'spread_autocorr': spread_autocorr,
                'spread_volatility': spread_volatility,
                'half_life': half_life,
                'spread_series': spread
            }

        except Exception as e:
            return {'cointegrated': False, 'error': str(e)}

    def _calculate_half_life(self, spread: pd.Series) -> float:
        """计算均值回归半衰期"""
        try:
            # 计算滞后项
            spread_lag = spread.shift(1)
            spread_diff = spread - spread_lag

            # 移除NaN
            valid_data = pd.concat([spread_lag, spread_diff], axis=1).dropna()
            if len(valid_data) < 10:
                return float('inf')

            # 线性回归
            X = valid_data.iloc[:, 0].values.reshape(-1, 1)
            y = valid_data.iloc[:, 1].values

            slope, _, _, _, _ = stats.linregress(X.flatten(), y)

            if slope >= 0:
                return float('inf')  # 不存在均值回归

            half_life = -np.log(2) / slope
            return half_life

        except:
            return float('inf')

    def find_cointegrated_pairs(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        寻找协整对

        Args:
            price_data: 多个资产的价格数据

        Returns:
            协整对分析结果
        """
        assets = price_data.columns
        cointegrated_pairs = []

        # 检查所有可能的配对
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                asset1 = assets[i]
                asset2 = assets[j]

                result = self.test_cointegration(price_data[asset1], price_data[asset2])

                if result['cointegrated']:
                    pair_key = f"{asset1}_{asset2}"
                    self.cointegration_pairs[pair_key] = result
                    self.hedge_ratios[pair_key] = result['hedge_ratio']

                    cointegrated_pairs.append({
                        'pair': pair_key,
                        'asset1': asset1,
                        'asset2': asset2,
                        'hedge_ratio': result['hedge_ratio'],
                        'r_squared': result['r_squared'],
                        'half_life': result['half_life'],
                        'spread_volatility': result['spread_volatility']
                    })

        # 按R²排序
        cointegrated_pairs.sort(key=lambda x: x['r_squared'], reverse=True)

        return {
            'cointegrated_pairs': cointegrated_pairs,
            'total_pairs': len(cointegrated_pairs)
        }

    def calculate_pairs_trading_signals(self, price_data: pd.DataFrame,
                                       pair_key: str) -> Dict[str, Any]:
        """
        计算配对交易信号

        Args:
            price_data: 价格数据
            pair_key: 配对键

        Returns:
            交易信号
        """
        if pair_key not in self.cointegration_pairs:
            return {'error': f'Pair {pair_key} not found or not cointegrated'}

        pair_info = self.cointegration_pairs[pair_key]
        asset1, asset2 = pair_key.split('_')

        # 获取价格序列
        prices1 = price_data[asset1]
        prices2 = price_data[asset2]

        # 计算当前价差
        log_prices1 = np.log(prices1)
        log_prices2 = np.log(prices2)
        hedge_ratio = pair_info['hedge_ratio']
        intercept = pair_info['intercept']

        current_spread = log_prices1.iloc[-1] - (hedge_ratio * log_prices2.iloc[-1] + intercept)

        # 计算价差统计量
        spread_series = log_prices1 - (hedge_ratio * log_prices2 + intercept)
        spread_mean = spread_series.rolling(self.lookback_window).mean()
        spread_std = spread_series.rolling(self.lookback_window).std()

        current_z_score = (current_spread - spread_mean.iloc[-1]) / (spread_std.iloc[-1] + 1e-8)

        # 生成交易信号
        if current_z_score < -self.entry_threshold:
            signal = 1  # 买入价差（做多asset1，做空asset2）
            position_size = min(1.0, abs(current_z_score) / self.entry_threshold)
        elif current_z_score > self.entry_threshold:
            signal = -1  # 卖出价差（做空asset1，做多asset2）
            position_size = min(1.0, abs(current_z_score) / self.entry_threshold)
        elif abs(current_z_score) < self.exit_threshold:
            signal = 0  # 平仓
            position_size = 0
        else:
            signal = 0  # 持有现有仓位
            position_size = 0

        # 计算预期收益
        expected_reversion = (spread_mean.iloc[-1] - current_spread) / spread_mean.iloc[-1]
        half_life = pair_info['half_life']

        return {
            'signal': signal,
            'position_size': position_size,
            'z_score': current_z_score,
            'current_spread': current_spread,
            'spread_mean': spread_mean.iloc[-1],
            'spread_std': spread_std.iloc[-1],
            'expected_reversion': expected_reversion,
            'half_life': half_life,
            'confidence': min(1.0, abs(current_z_score) / self.entry_threshold)
        }

    def calculate_mean_reversion_signals(self, price_series: pd.Series) -> Dict[str, Any]:
        """
        计算均值回归信号

        Args:
            price_series: 价格序列

        Returns:
            均值回归信号
        """
        if len(price_series) < self.lookback_window:
            return {'error': 'Insufficient data for mean reversion analysis'}

        # 计算收益率
        returns = price_series.pct_change().dropna()

        # 计算移动平均和标准差
        moving_avg = price_series.rolling(self.lookback_window).mean()
        moving_std = price_series.rolling(self.lookback_window).std()

        # 计算z-score
        z_score = (price_series - moving_avg) / (moving_std + 1e-8)
        current_z = z_score.iloc[-1]

        # 计算半衰期
        half_life = self._calculate_half_life(price_series)

        # 计算赫斯特指数（简化版）
        hurst_exponent = self._calculate_hurst_exponent(returns)

        # 生成信号
        if current_z < -self.entry_threshold and hurst_exponent < 0.5:
            signal = 1  # 买入（价格低于均值）
            position_size = min(1.0, abs(current_z) / self.entry_threshold)
        elif current_z > self.entry_threshold and hurst_exponent < 0.5:
            signal = -1  # 卖出（价格高于均值）
            position_size = min(1.0, abs(current_z) / self.entry_threshold)
        elif abs(current_z) < self.exit_threshold:
            signal = 0  # 平仓
            position_size = 0
        else:
            signal = 0  # 持有
            position_size = 0

        # 计算预期收益
        expected_reversion = (moving_avg.iloc[-1] - price_series.iloc[-1]) / moving_avg.iloc[-1]

        return {
            'signal': signal,
            'position_size': position_size,
            'z_score': current_z,
            'current_price': price_series.iloc[-1],
            'moving_average': moving_avg.iloc[-1],
            'moving_std': moving_std.iloc[-1],
            'expected_reversion': expected_reversion,
            'half_life': half_life,
            'hurst_exponent': hurst_exponent,
            'confidence': min(1.0, abs(current_z) / self.entry_threshold)
        }

    def _calculate_hurst_exponent(self, returns: pd.Series) -> float:
        """计算赫斯特指数"""
        try:
            # 简化的赫斯特指数计算
            lags = range(2, min(20, len(returns)//4))
            tau = [np.std(np.subtract(returns.values[lag:], returns.values[:-lag])) for lag in lags]

            if len(tau) > 1 and all(t > 0 for t in tau):
                poly = np.polyfit(np.log(lags), np.log(tau), 1)
                hurst = poly[0]
            else:
                hurst = 0.5

            return hurst

        except:
            return 0.5

    def calculate_statistical_arbitrage_portfolio(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算统计套利组合

        Args:
            price_data: 多个资产的价格数据

        Returns:
            套利组合分析结果
        """
        # 寻找协整对
        cointegration_result = self.find_cointegrated_pairs(price_data)

        if not cointegration_result['cointegrated_pairs']:
            return {'error': 'No cointegrated pairs found'}

        # 为每个协整对计算信号
        portfolio_signals = {}
        total_score = 0
        total_confidence = 0

        for pair_info in cointegration_result['cointegrated_pairs'][:5]:  # 前5个最佳配对
            pair_key = pair_info['pair']

            try:
                signal_result = self.calculate_pairs_trading_signals(price_data, pair_key)
                if 'error' not in signal_result:
                    portfolio_signals[pair_key] = signal_result
                    total_score += signal_result['signal'] * signal_result['confidence']
                    total_confidence += signal_result['confidence']
            except Exception as e:
                continue

        # 综合信号
        if total_confidence > 0:
            combined_signal = total_score / total_confidence
            combined_confidence = total_confidence / len(portfolio_signals)
        else:
            combined_signal = 0
            combined_confidence = 0

        return {
            'individual_pairs': portfolio_signals,
            'combined_signal': combined_signal,
            'combined_confidence': combined_confidence,
            'active_pairs': len(portfolio_signals),
            'portfolio_score': total_score
        }

    def calculate_ornstein_uhlenbeck_process(self, price_series: pd.Series) -> Dict[str, Any]:
        """
        计算Ornstein-Uhlenbeck过程参数

        Args:
            price_series: 价格序列

        Returns:
            OU过程参数
        """
        if len(price_series) < 50:
            return {'error': 'Insufficient data for OU process'}

        # 计算对数价格
        log_prices = np.log(price_series)

        # 计算差分
        delta_log_prices = log_prices.diff().dropna()
        lagged_log_prices = log_prices.shift(1).dropna()

        # 对齐数据
        aligned_data = pd.concat([lagged_log_prices, delta_log_prices], axis=1).dropna()

        if len(aligned_data) < 20:
            return {'error': 'Insufficient aligned data'}

        X = aligned_data.iloc[:, 0].values.reshape(-1, 1)
        y = aligned_data.iloc[:, 1].values

        # 线性回归估计参数
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(X.flatten(), y)

            # OU过程参数
            theta = -slope  # 均值回归速度
            mu = intercept / theta if theta != 0 else log_prices.mean()  # 长期均值
            sigma = std_err  # 波动率

            # 计算半衰期
            half_life = np.log(2) / theta if theta > 0 else float('inf')

            return {
                'theta': theta,
                'mu': mu,
                'sigma': sigma,
                'half_life': half_life,
                'r_squared': r_value**2,
                'mean_reversion_speed': theta,
                'long_term_mean': mu,
                'volatility': sigma
            }

        except Exception as e:
            return {'error': str(e)}

    def calculate_kalman_filter_pairs(self, price_data: pd.DataFrame,
                                    pair_key: str) -> Dict[str, Any]:
        """
        计算卡尔曼滤波配对交易

        Args:
            price_data: 价格数据
            pair_key: 配对键

        Returns:
            卡尔曼滤波结果
        """
        if pair_key not in self.cointegration_pairs:
            return {'error': f'Pair {pair_key} not cointegrated'}

        asset1, asset2 = pair_key.split('_')
        prices1 = price_data[asset1]
        prices2 = price_data[asset2]

        # 卡尔曼滤波参数
        delta = 1e-5  # 过程噪声
        vw = delta / (1 - delta) * np.eye(2)  # 状态转移噪声
        ve = 0.001  # 观测噪声

        # 初始化状态
        beta = np.array([0.0, 0.0])  # [alpha, beta]
        P = np.eye(2)  # 协方差矩阵

        # 存储结果
        betas = []
        residuals = []

        # 运行卡尔曼滤波
        for i in range(1, len(prices1)):
            # 预测步骤
            beta_pred = beta
            P_pred = P + vw

            # 更新步骤
            x = np.array([1.0, prices2.iloc[i]])
            y = prices1.iloc[i]

            # 卡尔曼增益
            R = ve + x.T @ P_pred @ x
            K = P_pred @ x / R

            # 状态更新
            beta = beta_pred + K * (y - x.T @ beta_pred)
            P = (np.eye(2) - K.reshape(-1, 1) @ x.reshape(1, -1)) @ P_pred

            betas.append(beta.copy())
            residuals.append(y - x.T @ beta)

        betas = np.array(betas)
        residuals = np.array(residuals)

        # 计算动态对冲比率
        current_hedge_ratio = betas[-1, 1]
        current_alpha = betas[-1, 0]

        # 计算残差统计量
        residual_mean = np.mean(residuals)
        residual_std = np.std(residuals)
        current_residual = residuals[-1]

        # z-score
        z_score = (current_residual - residual_mean) / (residual_std + 1e-8)

        return {
            'dynamic_hedge_ratio': current_hedge_ratio,
            'dynamic_alpha': current_alpha,
            'current_residual': current_residual,
            'z_score': z_score,
            'residual_mean': residual_mean,
            'residual_std': residual_std,
            'beta_series': betas,
            'residual_series': residuals
        }

    def calculate_statistical_arbitrage_indicators(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算统计套利指标

        Args:
            price_data: 价格数据

        Returns:
            统计套利指标DataFrame
        """
        if len(price_data) < self.lookback_window:
            return pd.DataFrame()

        indicators = pd.DataFrame(index=price_data.index[-1:])

        # 计算单个资产的均值回归信号
        for asset in price_data.columns:
            try:
                mr_signal = self.calculate_mean_reversion_signals(price_data[asset])
                if 'error' not in mr_signal:
                    indicators[f'{asset}_mr_signal'] = mr_signal['signal']
                    indicators[f'{asset}_mr_zscore'] = mr_signal['z_score']
                    indicators[f'{asset}_mr_confidence'] = mr_signal['confidence']
                    indicators[f'{asset}_half_life'] = mr_signal['half_life']
                    indicators[f'{asset}_hurst'] = mr_signal['hurst_exponent']
            except:
                continue

        # 计算协整对信号
        if len(price_data.columns) >= 2:
            portfolio_result = self.calculate_statistical_arbitrage_portfolio(price_data)
            if 'error' not in portfolio_result:
                indicators['portfolio_signal'] = portfolio_result['combined_signal']
                indicators['portfolio_confidence'] = portfolio_result['combined_confidence']
                indicators['active_pairs'] = portfolio_result['active_pairs']

        # 计算OU过程参数
        try:
            ou_result = self.calculate_ornstein_uhlenbeck_process(price_data.iloc[:, 0])
            if 'error' not in ou_result:
                indicators['ou_theta'] = ou_result['theta']
                indicators['ou_mu'] = ou_result['mu']
                indicators['ou_sigma'] = ou_result['sigma']
                indicators['ou_half_life'] = ou_result['half_life']
        except:
            pass

        return indicators


class PairsTradingOptimizer:
    """
    配对交易优化器

    优化配对交易策略的参数和风险管理
    """

    def __init__(self,
                 optimization_window: int = 252,
                 min_half_life: float = 5,
                 max_half_life: float = 60,
                 transaction_cost: float = 0.001):
        """
        初始化配对交易优化器

        Args:
            optimization_window: 优化窗口
            min_half_life: 最小半衰期
            max_half_life: 最大半衰期
            transaction_cost: 交易成本
        """
        self.optimization_window = optimization_window
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        self.transaction_cost = transaction_cost

    def optimize_pair_parameters(self, price_data: pd.DataFrame,
                                pair_key: str) -> Dict[str, Any]:
        """
        优化配对交易参数

        Args:
            price_data: 价格数据
            pair_key: 配对键

        Returns:
            优化结果
        """
        if '_' not in pair_key:
            return {'error': 'Invalid pair key format'}

        asset1, asset2 = pair_key.split('_')
        if asset1 not in price_data.columns or asset2 not in price_data.columns:
            return {'error': 'Assets not found in data'}

        # 提取价格序列
        prices1 = price_data[asset1]
        prices2 = price_data[asset2]

        # 定义目标函数（最大化夏普比率）
        def objective_function(params):
            entry_threshold, exit_threshold = params

            # 简化的回测
            signals = []
            positions = []
            returns = []

            for i in range(self.optimization_window, len(prices1)):
                # 计算滚动统计量
                window_prices1 = prices1.iloc[i-self.optimization_window:i]
                window_prices2 = prices2.iloc[i-self.optimization_window:i]

                # 计算对冲比率
                log_p1 = np.log(window_prices1)
                log_p2 = np.log(window_prices2)
                hedge_ratio = stats.linregress(log_p2, log_p1).slope

                # 计算价差
                spread = log_p1.iloc[-1] - hedge_ratio * log_p2.iloc[-1]

                # 计算z-score
                spread_mean = spread.mean()
                spread_std = spread.std()
                z_score = (spread - spread_mean) / (spread_std + 1e-8)

                # 生成信号
                if z_score < -entry_threshold:
                    signal = 1
                elif z_score > entry_threshold:
                    signal = -1
                elif abs(z_score) < exit_threshold:
                    signal = 0
                else:
                    signal = 0

                signals.append(signal)

            # 计算收益
            for i in range(1, len(signals)):
                if signals[i] != signals[i-1]:  # 信号变化
                    # 交易成本
                    returns.append(-self.transaction_cost)
                else:
                    # 策略收益（简化）
                    ret1 = prices1.pct_change().iloc[self.optimization_window + i]
                    ret2 = prices2.pct_change().iloc[self.optimization_window + i]
                    strategy_return = signals[i-1] * (ret1 - hedge_ratio * ret2)
                    returns.append(strategy_return)

            if not returns:
                return -np.inf

            # 计算夏普比率
            returns_array = np.array(returns)
            sharpe_ratio = returns_array.mean() / (returns_array.std() + 1e-8) * np.sqrt(252)

            return -sharpe_ratio  # 最小化负夏普比率

        # 参数范围
        bounds = [(1.5, 3.0), (0.1, 1.0)]  # entry_threshold, exit_threshold

        # 优化
        try:
            result = minimize(objective_function, [2.0, 0.5], bounds=bounds, method='L-BFGS-B')

            if result.success:
                optimal_entry, optimal_exit = result.x
                optimal_sharpe = -result.fun

                return {
                    'optimal_entry_threshold': optimal_entry,
                    'optimal_exit_threshold': optimal_exit,
                    'optimal_sharpe_ratio': optimal_sharpe,
                    'optimization_successful': True
                }
            else:
                return {'error': 'Optimization failed'}

        except Exception as e:
            return {'error': str(e)}

    def calculate_optimal_position_size(self, price_data: pd.DataFrame,
                                      pair_key: str,
                                      signal_strength: float) -> Dict[str, Any]:
        """
        计算最优仓位大小

        Args:
            price_data: 价格数据
            pair_key: 配对键
            signal_strength: 信号强度

        Returns:
            仓位建议
        """
        # Kelly准则计算最优仓位
        if '_' not in pair_key:
            return {'error': 'Invalid pair key'}

        asset1, asset2 = pair_key.split('_')
        prices1 = price_data[asset1]
        prices2 = price_data[asset2]

        # 计算历史收益
        returns1 = prices1.pct_change().dropna()
        returns2 = prices2.pct_change().dropna()

        # 计算配对收益（使用固定对冲比率）
        hedge_ratio = stats.linregress(np.log(prices2), np.log(prices1)).slope
        pair_returns = returns1 - hedge_ratio * returns2

        # Kelly准则
        win_rate = (pair_returns > 0).mean()
        avg_win = pair_returns[pair_returns > 0].mean()
        avg_loss = abs(pair_returns[pair_returns < 0].mean())

        if avg_loss > 0:
            kelly_fraction = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
            kelly_fraction = max(0, min(0.25, kelly_fraction))  # 限制在25%以内
        else:
            kelly_fraction = 0

        # 根据信号强度调整
        adjusted_fraction = kelly_fraction * abs(signal_strength)

        # 风险调整
        volatility = pair_returns.std() * np.sqrt(252)
        risk_adjusted_fraction = adjusted_fraction / (1 + volatility)

        return {
            'kelly_fraction': kelly_fraction,
            'signal_adjusted_fraction': adjusted_fraction,
            'risk_adjusted_fraction': risk_adjusted_fraction,
            'win_rate': win_rate,
            'volatility': volatility,
            'recommended_position_size': risk_adjusted_fraction
        }


def create_statistical_arbitrage_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    创建统计套利特征

    Args:
        data: 市场数据

    Returns:
        统计套利特征DataFrame
    """
    engine = StatisticalArbitrageEngine()
    indicators = engine.calculate_statistical_arbitrage_indicators(data)
    return indicators


# 主要功能函数
def calculate_statistical_arbitrage_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有统计套利指标

    Args:
        data: 包含价格数据的DataFrame

    Returns:
        包含所有指标值的DataFrame
    """
    if len(data) < 100:
        raise ValueError("数据长度不足，至少需要100个数据点")

    return create_statistical_arbitrage_features(data)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=300, freq='D')

    # 模拟两个相关资产的价格
    base_returns = np.random.normal(0.001, 0.02, 300)
    asset1_prices = 100 * np.exp(np.cumsum(base_returns))

    # 创建相关的第二个资产
    noise = np.random.normal(0, 0.01, 300)
    asset2_prices = asset1_prices * 1.1 * np.exp(noise)

    sample_data = pd.DataFrame({
        'Asset1': asset1_prices,
        'Asset2': asset2_prices
    }, index=dates)

    # 计算指标
    try:
        indicators = calculate_statistical_arbitrage_indicators(sample_data)
        print("统计套利指标计算成功!")
        print(f"指标数量: {indicators.shape[1]}")
        print("最新指标值:")
        print(indicators.iloc[-1])

        # 测试协整关系
        engine = StatisticalArbitrageEngine()
        coint_result = engine.find_cointegrated_pairs(sample_data)
        print("\n协整对分析:")
        for pair in coint_result['cointegrated_pairs']:
            print(f"{pair['pair']}: R²={pair['r_squared']:.3f}, 半衰期={pair['half_life']:.1f}")

    except Exception as e:
        print(f"计算错误: {e}")