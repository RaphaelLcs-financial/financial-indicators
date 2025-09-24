"""
市场微观结构分析器金融指标

本模块实现了基于市场微观结构分析的高频统计套利指标，专注于订单流、市场冲击和流动性分析。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class MarketMicrostructureAnalyzer:
    """
    市场微观结构分析器

    分析订单流、价格形成过程、市场冲击等微观结构特征
    """

    def __init__(self,
                 tick_size: float = 0.01,
                 lot_size: int = 100,
                 analysis_window: int = 100,
                 impact_decay_rate: float = 0.95):
        """
        初始化市场微观结构分析器

        Args:
            tick_size: 最小价格变动单位
            lot_size: 最小交易单位
            analysis_window: 分析窗口大小
            impact_decay_rate: 市场冲击衰减率
        """
        self.tick_size = tick_size
        self.lot_size = lot_size
        self.analysis_window = analysis_window
        self.impact_decay_rate = impact_decay_rate

        # 订单簿模拟
        self.order_book = {
            'bids': [],
            'asks': [],
            'bid_prices': [],
            'ask_prices': [],
            'bid_volumes': [],
            'ask_volumes': []
        }

        # 市场冲击历史
        self.impact_history = []
        self.impact_decay = {}

        # 流动性指标
        self.liquidity_metrics = {}
        self.price_discovery_metrics = {}

    def _simulate_order_book(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        模拟订单簿状态

        Args:
            data: 市场数据

        Returns:
            模拟的订单簿状态
        """
        latest = data.iloc[-1]

        # 基于当前价格模拟订单簿
        mid_price = (latest['high'] + latest['low']) / 2
        spread = latest['high'] - latest['low']

        # 生成买卖盘价格
        bid_prices = []
        ask_prices = []
        bid_volumes = []
        ask_volumes = []

        current_bid = mid_price - spread * 0.3
        current_ask = mid_price + spread * 0.3

        for i in range(10):  # 10档深度
            bid_prices.append(current_bid - i * self.tick_size)
            ask_prices.append(current_ask + i * self.tick_size)

            # 模拟深度分布（指数衰减）
            bid_volumes.append(int(latest['volume'] * 0.1 * np.exp(-i * 0.3)))
            ask_volumes.append(int(latest['volume'] * 0.1 * np.exp(-i * 0.3)))

        return {
            'bid_prices': bid_prices,
            'ask_prices': ask_prices,
            'bid_volumes': bid_volumes,
            'ask_volumes': ask_volumes,
            'mid_price': mid_price,
            'spread': spread,
            'weighted_mid_price': self._calculate_weighted_mid_price(
                bid_prices, ask_prices, bid_volumes, ask_volumes
            )
        }

    def _calculate_weighted_mid_price(self, bid_prices: List[float], ask_prices: List[float],
                                   bid_volumes: List[int], ask_volumes: List[int]) -> float:
        """计算加权中价"""
        if not bid_prices or not ask_prices:
            return 0.0

        total_bid_volume = sum(bid_volumes)
        total_ask_volume = sum(ask_volumes)
        total_volume = total_bid_volume + total_ask_volume

        if total_volume == 0:
            return (bid_prices[0] + ask_prices[0]) / 2

        weighted_bid = sum(p * v for p, v in zip(bid_prices, bid_volumes)) / total_bid_volume if total_bid_volume > 0 else bid_prices[0]
        weighted_ask = sum(p * v for p, v in zip(ask_prices, ask_volumes)) / total_ask_volume if total_ask_volume > 0 else ask_prices[0]

        return (weighted_bid + weighted_ask) / 2

    def calculate_order_flow_imbalance(self, data: pd.DataFrame) -> pd.Series:
        """
        计算订单流不平衡

        Args:
            data: 市场数据

        Returns:
            订单流不平衡序列
        """
        ofi = pd.Series(0.0, index=data.index)

        for i in range(1, len(data)):
            # 模拟订单流
            current = data.iloc[i]
            prev = data.iloc[i-1]

            # 价格变动方向
            price_change = current['close'] - prev['close']

            # 成交量变动
            volume_change = current['volume'] - prev['volume']

            # 买卖压力
            buy_pressure = (current['close'] - current['open']) / (current['high'] - current['low'] + 1e-8)
            sell_pressure = 1 - buy_pressure

            # 订单流不平衡
            net_flow = buy_pressure * current['volume'] - sell_pressure * current['volume']
            normalized_flow = net_flow / (current['volume'] + 1e-8)

            ofi.iloc[i] = normalized_flow

        return ofi

    def calculate_price_impact_function(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        计算价格冲击函数

        Args:
            data: 市场数据

        Returns:
            价格冲击分析结果
        """
        returns = data['close'].pct_change().dropna()
        volumes = data['volume'].iloc[1:]

        # 计算临时冲击和永久冲击
        temporary_impacts = []
        permanent_impacts = []

        for i in range(len(returns)):
            # 基于成交量的冲击估计
            volume_normalized = volumes.iloc[i] / volumes.rolling(20).mean().iloc[i]

            # 临时冲击（价格反转）
            if i > 0:
                reversal = -returns.iloc[i] * returns.iloc[i-1]  # 价格反转程度
                temporary_impact = reversal * volume_normalized
            else:
                temporary_impact = 0

            # 永久冲击（价格持续变动）
            permanent_impact = abs(returns.iloc[i]) * volume_normalized

            temporary_impacts.append(temporary_impact)
            permanent_impacts.append(permanent_impact)

        # 估计冲击函数参数
        impact_params = self._estimate_impact_function(volumes.values, returns.values)

        return {
            'temporary_impacts': temporary_impacts,
            'permanent_impacts': permanent_impacts,
            'impact_parameters': impact_params,
            'average_temporary_impact': np.mean(temporary_impacts),
            'average_permanent_impact': np.mean(permanent_impacts),
            'impact_ratio': np.mean(permanent_impacts) / (np.mean(temporary_impacts) + 1e-8)
        }

    def _estimate_impact_function(self, volumes: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        """估计冲击函数参数"""
        # 简化的平方根冲击函数: impact = a * sqrt(volume)
        try:
            # 准备数据
            valid_indices = ~np.isnan(returns) & ~np.isnan(volumes) & (volumes > 0)
            X = np.sqrt(volumes[valid_indices]).reshape(-1, 1)
            y = np.abs(returns[valid_indices])

            if len(X) > 10:
                # 线性回归
                from sklearn.linear_model import LinearRegression
                model = LinearRegression()
                model.fit(X, y)

                return {
                    'alpha': model.coef_[0],
                    'intercept': model.intercept_,
                    'r_squared': model.score(X, y)
                }
            else:
                return {'alpha': 0.0, 'intercept': 0.0, 'r_squared': 0.0}

        except:
            return {'alpha': 0.0, 'intercept': 0.0, 'r_squared': 0.0}

    def calculate_liquidity_measures(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算流动性度量

        Args:
            data: 市场数据

        Returns:
            流动性指标字典
        """
        measures = {}

        # Amihud非流动性指标
        returns = data['close'].pct_change()
        dollar_volume = data['close'] * data['volume']
        amihud = abs(returns) / (dollar_volume + 1e-8)
        measures['amihud_illiquidity'] = amihud.rolling(20).mean()

        # Roll有效价差
        price_changes = data['close'].diff()
        roll_spread = 2 * np.sqrt(-np.minimum(0, price_changes.rolling(2).cov()))
        measures['roll_effective_spread'] = roll_spread.rolling(20).mean()

        # 成交量加权平均价格偏差
        vwap = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        vwap_deviation = abs(data['close'] - vwap) / data['close']
        measures['vwap_deviation'] = vwap_deviation.rolling(20).mean()

        # 价格影响系数
        volume_normalized = data['volume'] / data['volume'].rolling(20).mean()
        price_impact = abs(returns) / (volume_normalized + 1e-8)
        measures['price_impact_coefficient'] = price_impact.rolling(20).mean()

        # 流动性比率
        high_low_range = (data['high'] - data['low']) / data['close']
        volume_ratio = data['volume'] / data['volume'].rolling(20).mean()
        measures['liquidity_ratio'] = volume_ratio / (high_low_range + 1e-8)

        return measures

    def calculate_price_discovery_metrics(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算价格发现指标

        Args:
            data: 市场数据

        Returns:
            价格发现指标字典
        """
        metrics = {}

        # 信息份额（简化版）
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std()

        # 价格效率指标
        price_efficiency = self._calculate_price_efficiency(data)
        metrics['price_efficiency'] = price_efficiency.rolling(20).mean()

        # 价格发现速度
        price_adjustment_speed = self._calculate_adjustment_speed(data)
        metrics['price_discovery_speed'] = price_adjustment_speed.rolling(20).mean()

        # 市场深度指标
        market_depth = self._estimate_market_depth(data)
        metrics['market_depth'] = market_depth.rolling(20).mean()

        # 订单流毒性
        order_flow_toxicity = self._calculate_order_flow_toxicity(data)
        metrics['order_flow_toxicity'] = order_flow_toxicity.rolling(20).mean()

        # 价格冲击持续性
        impact_persistence = self._calculate_impact_persistence(data)
        metrics['impact_persistence'] = impact_persistence.rolling(20).mean()

        return metrics

    def _calculate_price_efficiency(self, data: pd.DataFrame) -> pd.Series:
        """计算价格效率"""
        returns = data['close'].pct_change()

        # 随机游走检验
        autocorr = returns.rolling(20).apply(lambda x: x.autocorr(lag=1))

        # 方差比率检验
        variance_ratio = self._variance_ratio_test(returns, 5)

        # 综合效率指标
        efficiency = 1 - abs(autocorr) - abs(variance_ratio - 1)

        return efficiency.fillna(0.5)

    def _variance_ratio_test(self, returns: pd.Series, q: int) -> pd.Series:
        """方差比率检验"""
        def vr(ser):
            if len(ser) < q * 2:
                return 1.0

            variance_1 = ser.var()
            variance_q = (ser.rolling(q).sum()).var() / q

            return variance_q / variance_1 if variance_1 > 0 else 1.0

        return returns.rolling(q * 2).apply(vr)

    def _calculate_adjustment_speed(self, data: pd.DataFrame) -> pd.Series:
        """计算价格调整速度"""
        returns = data['close'].pct_change()
        absolute_returns = abs(returns)

        # 价格冲击的半衰期
        half_life = []
        for i in range(20, len(returns)):
            window_returns = returns.iloc[i-20:i]
            try:
                # 简化的均值回归检验
                autocorr = window_returns.autocorr(lag=1)
                if autocorr < 0:  # 均值回归
                    hl = -np.log(2) / np.log(1 + autocorr) if autocorr > -1 else 1
                else:
                    hl = 10  # 无均值回归
                half_life.append(hl)
            except:
                half_life.append(10)

        adjustment_speed = pd.Series(half_life, index=returns.index[19:])
        return 1 / (adjustment_speed + 1)  # 转换为速度指标

    def _estimate_market_depth(self, data: pd.DataFrame) -> pd.Series:
        """估计市场深度"""
        # 基于成交量和价格变动估计深度
        returns = data['close'].pct_change()
        volume_normalized = data['volume'] / data['volume'].rolling(20).mean()

        # 深度与价格冲击成反比
        depth_estimate = volume_normalized / (abs(returns) + 1e-8)

        return depth_estimate.fillna(1.0)

    def _calculate_order_flow_toxicity(self, data: pd.DataFrame) -> pd.Series:
        """计算订单流毒性"""
        # VPIN指标简化版
        returns = data['close'].pct_change()
        volume_changes = data['volume'].pct_change()

        # 买卖不平衡
        buy_sell_imbalance = np.sign(returns) * volume_changes

        # 毒性指标
        toxicity = abs(buy_sell_imbalance).rolling(20).std()

        return toxicity.fillna(0)

    def _calculate_impact_persistence(self, data: pd.DataFrame) -> pd.Series:
        """计算冲击持续性"""
        returns = data['close'].pct_change()
        volumes = data['volume']

        # 计算冲击的自相关性
        impact = abs(returns) * volumes
        impact_autocorr = impact.rolling(20).apply(lambda x: x.autocorr(lag=1))

        return impact_autocorr.fillna(0)

    def calculate_market_stress_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算市场压力指标

        Args:
            data: 市场数据

        Returns:
            市场压力指标字典
        """
        indicators = {}

        # 价格跳跃检测
        returns = data['close'].pct_change()
        price_jumps = self._detect_price_jumps(returns)
        indicators['price_jump_frequency'] = price_jumps.rolling(50).sum() / 50

        # 流动性干涸风险
        volume_spike = data['volume'] / data['volume'].rolling(20).mean()
        liquidity_dry_up = (volume_spike < 0.5).astype(float)
        indicators['liquidity_dry_up_risk'] = liquidity_dry_up.rolling(20).mean()

        # 市场深度恶化
        bid_ask_spread = (data['high'] - data['low']) / data['close']
        spread_widening = bid_ask_spread.rolling(20).pct_change()
        indicators['spread_widening'] = spread_widening.fillna(0)

        # 订单流不平衡
        ofi = self.calculate_order_flow_imbalance(data)
        indicators['order_flow_imbalance_extreme'] = (abs(ofi) > 0.8).astype(float).rolling(20).mean()

        # 价格发现效率恶化
        price_efficiency = self._calculate_price_efficiency(data)
        indicators['price_dislocation'] = (1 - price_efficiency).rolling(20).mean()

        return indicators

    def _detect_price_jumps(self, returns: pd.Series, threshold: float = 3.0) -> pd.Series:
        """检测价格跳跃"""
        # 基于标准化的收益率
        rolling_std = returns.rolling(20).std()
        rolling_mean = returns.rolling(20).mean()

        normalized_returns = (returns - rolling_mean) / (rolling_std + 1e-8)

        # 检测跳跃
        jumps = (abs(normalized_returns) > threshold).astype(int)

        return jumps

    def calculate_high_frequency_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算高频交易信号

        Args:
            data: 市场数据

        Returns:
            高频交易信号DataFrame
        """
        if len(data) < self.analysis_window:
            return pd.DataFrame()

        # 计算各种指标
        ofi = self.calculate_order_flow_imbalance(data)
        liquidity_measures = self.calculate_liquidity_measures(data)
        price_discovery = self.calculate_price_discovery_metrics(data)
        stress_indicators = self.calculate_market_stress_indicators(data)

        # 创建信号DataFrame
        signals = pd.DataFrame(index=data.index[-1:], columns=[
            'momentum_signal', 'mean_reversion_signal', 'liquidity_signal',
            'order_flow_signal', 'market_stress_signal', 'combined_signal'
        ])

        latest_idx = -1

        # 动量信号
        momentum_score = self._calculate_momentum_score(data, ofi)
        signals['momentum_signal'].iloc[latest_idx] = momentum_score

        # 均值回归信号
        mean_reversion_score = self._calculate_mean_reversion_score(data, price_discovery)
        signals['mean_reversion_signal'].iloc[latest_idx] = mean_reversion_score

        # 流动性信号
        liquidity_score = self._calculate_liquidity_score(liquidity_measures)
        signals['liquidity_signal'].iloc[latest_idx] = liquidity_score

        # 订单流信号
        order_flow_score = ofi.iloc[-1]
        signals['order_flow_signal'].iloc[latest_idx] = order_flow_score

        # 市场压力信号
        stress_score = self._calculate_stress_score(stress_indicators)
        signals['market_stress_signal'].iloc[latest_idx] = stress_score

        # 综合信号
        combined_signal = self._combine_signals(
            momentum_score, mean_reversion_score, liquidity_score,
            order_flow_score, stress_score
        )
        signals['combined_signal'].iloc[latest_idx] = combined_signal

        return signals

    def _calculate_momentum_score(self, data: pd.DataFrame, ofi: pd.Series) -> float:
        """计算动量分数"""
        returns = data['close'].pct_change()

        # 短期动量
        short_momentum = returns.iloc[-5:].mean()

        # 订单流确认
        ofi_confirmation = ofi.iloc[-5:].mean()

        # 价格效率
        efficiency = self._calculate_price_efficiency(data).iloc[-1]

        momentum_score = 0.4 * short_momentum + 0.3 * ofi_confirmation + 0.3 * efficiency

        return np.tanh(momentum_score * 10)  # 归一化到[-1,1]

    def _calculate_mean_reversion_score(self, data: pd.DataFrame,
                                      price_discovery: Dict[str, pd.Series]) -> float:
        """计算均值回归分数"""
        returns = data['close'].pct_change()

        # 价格偏离度
        recent_returns = returns.iloc[-10:]
        deviation = abs(recent_returns.mean())

        # 调整速度
        adjustment_speed = price_discovery['price_discovery_speed'].iloc[-1]

        # 流动性支持
        liquidity_ratio = price_discovery['market_depth'].iloc[-1]

        mean_reversion_score = 0.4 * deviation - 0.3 * adjustment_speed + 0.3 * liquidity_ratio

        return np.tanh(mean_reversion_score * 5)

    def _calculate_liquidity_score(self, liquidity_measures: Dict[str, pd.Series]) -> float:
        """计算流动性分数"""
        # 综合流动性指标
        amihud = liquidity_measures['amihud_illiquidity'].iloc[-1]
        liquidity_ratio = liquidity_measures['liquidity_ratio'].iloc[-1]
        vwap_dev = liquidity_measures['vwap_deviation'].iloc[-1]

        # 归一化并综合
        liquidity_score = 0.4 * (1 - amihud) + 0.3 * liquidity_ratio + 0.3 * (1 - vwap_dev)

        return np.clip(liquidity_score, -1, 1)

    def _calculate_stress_score(self, stress_indicators: Dict[str, pd.Series]) -> float:
        """计算市场压力分数"""
        jump_freq = stress_indicators['price_jump_frequency'].iloc[-1]
        liquidity_risk = stress_indicators['liquidity_dry_up_risk'].iloc[-1]
        spread_widen = stress_indicators['spread_widening'].iloc[-1]
        dislocation = stress_indicators['price_dislocation'].iloc[-1]

        stress_score = 0.3 * jump_freq + 0.3 * liquidity_risk + 0.2 * spread_widen + 0.2 * dislocation

        return np.clip(stress_score, 0, 1)

    def _combine_signals(self, momentum: float, mean_reversion: float,
                        liquidity: float, order_flow: float, stress: float) -> float:
        """综合信号"""
        # 权重分配
        weights = np.array([0.25, 0.2, 0.15, 0.25, 0.15])

        # 信号组合
        signals = np.array([momentum, mean_reversion, liquidity, order_flow, -stress])

        # 加权平均
        combined = np.average(signals, weights=weights)

        return np.clip(combined, -1, 1)

    def calculate_microstructure_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算微观结构指标

        Args:
            data: 市场数据

        Returns:
            微观结构指标DataFrame
        """
        if len(data) < self.analysis_window:
            return pd.DataFrame()

        # 计算高频信号
        hf_signals = self.calculate_high_frequency_signals(data)

        # 计算流动性指标
        liquidity_measures = self.calculate_liquidity_measures(data)

        # 计算价格发现指标
        price_discovery = self.calculate_price_discovery_metrics(data)

        # 计算市场压力指标
        stress_indicators = self.calculate_market_stress_indicators(data)

        # 模拟订单簿
        order_book = self._simulate_order_book(data)

        # 创建综合指标DataFrame
        indicators = pd.DataFrame(index=data.index[-1:])

        # 添加高频信号
        if not hf_signals.empty:
            indicators = pd.concat([indicators, hf_signals], axis=1)

        # 添加最新流动性指标
        for name, series in liquidity_measures.items():
            indicators[f'liquidity_{name}'] = series.iloc[-1]

        # 添加最新价格发现指标
        for name, series in price_discovery.items():
            indicators[f'price_discovery_{name}'] = series.iloc[-1]

        # 添加最新压力指标
        for name, series in stress_indicators.items():
            indicators[f'stress_{name}'] = series.iloc[-1]

        # 添加订单簿指标
        indicators['mid_price'] = order_book['mid_price']
        indicators['spread'] = order_book['spread']
        indicators['weighted_mid_price'] = order_book['weighted_mid_price']
        indicators['order_book_imbalance'] = (sum(order_book['bid_volumes']) - sum(order_book['ask_volumes'])) / \
                                            (sum(order_book['bid_volumes']) + sum(order_book['ask_volumes']) + 1e-8)

        # 添加市场冲击指标
        impact_analysis = self.calculate_price_impact_function(data)
        if 'impact_ratio' in impact_analysis:
            indicators['impact_ratio'] = impact_analysis['impact_ratio']

        return indicators


class HighFrequencyArbitrageEngine:
    """
    高频套利引擎

    实现多种高频统计套利策略
    """

    def __init__(self,
                 latency_threshold: float = 0.001,
                 profit_threshold: float = 0.0001,
                 risk_limit: float = 0.01):
        """
        初始化高频套利引擎

        Args:
            latency_threshold: 延迟阈值（秒）
            profit_threshold: 利润阈值
            risk_limit: 风险限额
        """
        self.latency_threshold = latency_threshold
        self.profit_threshold = profit_threshold
        self.risk_limit = risk_limit

        # 套利策略
        self.strategies = {
            'statistical_arbitrage': self._statistical_arbitrage,
            'pairs_trading': self._pairs_trading,
            'market_making': self._market_making,
            'latency_arbitrage': self._latency_arbitrage,
            'liquidity_provision': self._liquidity_provision
        }

        # 风险管理
        self.position_limits = {}
        self.risk_metrics = {}

    def _statistical_arbitrage(self, data: pd.DataFrame) -> Dict[str, Any]:
        """统计套利策略"""
        returns = data['close'].pct_change().dropna()

        # 计算z-score
        rolling_mean = returns.rolling(20).mean()
        rolling_std = returns.rolling(20).std()
        z_score = (returns - rolling_mean) / (rolling_std + 1e-8)

        current_z = z_score.iloc[-1]

        # 生成信号
        if current_z < -2.0:  # 显著低估
            signal = 1  # 买入
            expected_profit = abs(current_z) * rolling_std.iloc[-1]
        elif current_z > 2.0:  # 显著高估
            signal = -1  # 卖出
            expected_profit = abs(current_z) * rolling_std.iloc[-1]
        else:
            signal = 0  # 持有
            expected_profit = 0

        return {
            'signal': signal,
            'expected_profit': expected_profit,
            'z_score': current_z,
            'confidence': min(1.0, abs(current_z) / 2.0)
        }

    def _pairs_trading(self, data1: pd.DataFrame, data2: pd.DataFrame) -> Dict[str, Any]:
        """配对交易策略"""
        if len(data1) != len(data2):
            return {'signal': 0, 'expected_profit': 0, 'confidence': 0}

        # 计算价差
        spread = data1['close'] - data2['close']

        # 计算价差的z-score
        spread_mean = spread.rolling(20).mean()
        spread_std = spread.rolling(20).std()
        spread_z = (spread - spread_mean) / (spread_std + 1e-8)

        current_z = spread_z.iloc[-1]

        # 生成信号
        if current_z < -2.0:
            signal = 1  # 买入data1，卖出data2
            expected_profit = abs(current_z) * spread_std.iloc[-1]
        elif current_z > 2.0:
            signal = -1  # 卖出data1，买入data2
            expected_profit = abs(current_z) * spread_std.iloc[-1]
        else:
            signal = 0
            expected_profit = 0

        return {
            'signal': signal,
            'expected_profit': expected_profit,
            'spread_z_score': current_z,
            'confidence': min(1.0, abs(current_z) / 2.0)
        }

    def _market_making(self, data: pd.DataFrame) -> Dict[str, Any]:
        """做市策略"""
        current_price = data['close'].iloc[-1]
        volatility = data['close'].pct_change().rolling(20).std().iloc[-1]

        # 计算买卖价差
        spread = max(self.tick_size, volatility * current_price * 0.5)

        bid_price = current_price - spread / 2
        ask_price = current_price + spread / 2

        # 基于存货风险调整
        inventory_risk = self._calculate_inventory_risk(data)
        risk_adjustment = inventory_risk * spread * 0.1

        bid_price -= risk_adjustment
        ask_price += risk_adjustment

        return {
            'bid_price': bid_price,
            'ask_price': ask_price,
            'spread': spread,
            'inventory_risk': inventory_risk,
            'signal': 0  # 做市策略同时买卖
        }

    def _latency_arbitrage(self, fast_data: pd.DataFrame, slow_data: pd.DataFrame) -> Dict[str, Any]:
        """延迟套利策略"""
        fast_price = fast_data['close'].iloc[-1]
        slow_price = slow_data['close'].iloc[-1]

        price_diff = fast_price - slow_price
        price_diff_pct = price_diff / slow_price

        if abs(price_diff_pct) > self.profit_threshold:
            signal = 1 if price_diff > 0 else -1
            expected_profit = abs(price_diff_pct)
            confidence = min(1.0, abs(price_diff_pct) / self.profit_threshold)
        else:
            signal = 0
            expected_profit = 0
            confidence = 0

        return {
            'signal': signal,
            'expected_profit': expected_profit,
            'price_difference': price_diff_pct,
            'confidence': confidence
        }

    def _liquidity_provision(self, data: pd.DataFrame) -> Dict[str, Any]:
        """流动性提供策略"""
        current_price = data['close'].iloc[-1]
        volume = data['volume'].iloc[-1]
        avg_volume = data['volume'].rolling(20).mean().iloc[-1]

        # 识别流动性不足的情况
        volume_ratio = volume / avg_volume

        if volume_ratio < 0.5:  # 成交量偏低，提供流动性
            spread = self.tick_size * 2  # 提供较窄的价差
            bid_price = current_price - spread / 2
            ask_price = current_price + spread / 2

            signal = 0  # 同时提供买卖流动性
            confidence = (0.5 - volume_ratio) / 0.5  # 流动性越不足，信心越高
        else:
            signal = 0
            confidence = 0
            bid_price = ask_price = spread = 0

        return {
            'signal': signal,
            'bid_price': bid_price,
            'ask_price': ask_price,
            'spread': spread,
            'confidence': confidence,
            'volume_ratio': volume_ratio
        }

    def _calculate_inventory_risk(self, data: pd.DataFrame) -> float:
        """计算存货风险"""
        returns = data['close'].pct_change()
        volatility = returns.rolling(20).std().iloc[-1]

        # 简化的存货风险度量
        return volatility * np.sqrt(len(data))  # 随时间累积的风险

    def execute_arbitrage_strategy(self, strategy_name: str, data: pd.DataFrame,
                                 **kwargs) -> Dict[str, Any]:
        """
        执行套利策略

        Args:
            strategy_name: 策略名称
            data: 市场数据
            **kwargs: 策略参数

        Returns:
            执行结果
        """
        if strategy_name not in self.strategies:
            return {'error': f'Unknown strategy: {strategy_name}'}

        strategy_func = self.strategies[strategy_name]

        try:
            if strategy_name == 'pairs_trading':
                if 'data2' not in kwargs:
                    return {'error': 'pairs_trading requires data2 parameter'}
                result = strategy_func(data, kwargs['data2'])
            elif strategy_name == 'latency_arbitrage':
                if 'slow_data' not in kwargs:
                    return {'error': 'latency_arbitrage requires slow_data parameter'}
                result = strategy_func(data, kwargs['slow_data'])
            else:
                result = strategy_func(data)

            # 风险检查
            if 'signal' in result and abs(result['signal']) > 0:
                risk_assessment = self._assess_strategy_risk(strategy_name, result)
                result['risk_assessment'] = risk_assessment

                # 如果风险过高，调整信号
                if risk_assessment['risk_level'] == 'high':
                    result['signal'] = 0
                    result['risk_override'] = True

            return result

        except Exception as e:
            return {'error': str(e)}

    def _assess_strategy_risk(self, strategy_name: str, result: Dict[str, Any]) -> Dict[str, str]:
        """评估策略风险"""
        risk_level = 'low'
        risk_factors = []

        # 检查预期收益
        if 'expected_profit' in result and result['expected_profit'] > 0.01:
            risk_level = 'medium'
            risk_factors.append('high_expected_profit')

        # 检查置信度
        if 'confidence' in result and result['confidence'] < 0.5:
            risk_level = 'medium'
            risk_factors.append('low_confidence')

        # 策略特定风险
        if strategy_name == 'statistical_arbitrage':
            if 'z_score' in result and abs(result['z_score']) > 3:
                risk_level = 'high'
                risk_factors.append('extreme_z_score')
        elif strategy_name == 'market_making':
            if 'inventory_risk' in result and result['inventory_risk'] > 0.05:
                risk_level = 'high'
                risk_factors.append('high_inventory_risk')

        return {
            'risk_level': risk_level,
            'risk_factors': risk_factors
        }


def create_microstructure_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    创建市场微观结构特征

    Args:
        data: 市场数据

    Returns:
        微观结构特征DataFrame
    """
    analyzer = MarketMicrostructureAnalyzer()
    indicators = analyzer.calculate_microstructure_indicators(data)
    return indicators


# 主要功能函数
def calculate_microstructure_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有市场微观结构指标

    Args:
        data: 包含OHLCV数据的DataFrame

    Returns:
        包含所有指标值的DataFrame
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("数据必须包含 'open', 'high', 'low', 'close', 'volume' 列")

    if len(data) < 100:
        raise ValueError("数据长度不足，至少需要100个数据点")

    return create_microstructure_features(data)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='5min')  # 5分钟数据

    # 模拟高频价格数据
    initial_price = 100
    returns = np.random.normal(0, 0.001, len(dates))  # 较小的波动率
    prices = [initial_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    sample_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(dates))
    }, index=dates)

    # 计算指标
    try:
        indicators = calculate_microstructure_indicators(sample_data)
        print("市场微观结构指标计算成功!")
        print(f"指标数量: {indicators.shape[1]}")
        print("最新指标值:")
        print(indicators.iloc[-1])

    except Exception as e:
        print(f"计算错误: {e}")