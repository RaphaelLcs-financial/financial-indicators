"""
流动性风险分析器金融指标

本模块实现了基于流动性风险分析的高频交易指标，专注于流动性度量、风险预警和资本管理。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


class LiquidityRiskAnalyzer:
    """
    流动性风险分析器

    分析市场流动性状况、识别流动性风险事件、评估交易成本
    """

    def __init__(self,
                 liquidity_window: int = 20,
                 risk_threshold: float = 0.95,
                 stress_threshold: float = 0.99):
        """
        初始化流动性风险分析器

        Args:
            liquidity_window: 流动性计算窗口
            risk_threshold: 风险阈值（分位数）
            stress_threshold: 压力阈值（分位数）
        """
        self.liquidity_window = liquidity_window
        self.risk_threshold = risk_threshold
        self.stress_threshold = stress_threshold

        # 流动性指标历史
        self.liquidity_history = {}
        self.risk_events = []

        # 流动性基准
        self.liquidity_benchmarks = {}

    def calculate_tradability_metrics(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算可交易性指标

        Args:
            data: 市场数据

        Returns:
            可交易性指标字典
        """
        metrics = {}

        # 成交量加权平均价格偏差
        vwap = (data['close'] * data['volume']).cumsum() / data['volume'].cumsum()
        vwap_deviation = abs(data['close'] - vwap) / data['close']
        metrics['vwap_deviation'] = vwap_deviation.rolling(self.liquidity_window).mean()

        # 价格影响系数
        returns = data['close'].pct_change()
        volume_normalized = data['volume'] / data['volume'].rolling(self.liquidity_window).mean()
        price_impact = abs(returns) / (volume_normalized + 1e-8)
        metrics['price_impact_coefficient'] = price_impact.rolling(self.liquidity_window).mean()

        # 交易频率指标
        volume_activity = data['volume'].rolling(self.liquidity_window).sum()
        metrics['trading_activity'] = volume_activity / volume_activity.rolling(50).mean()

        # 价格连续性
        price_gaps = abs(data['close'] - data['close'].shift(1)) / data['close'].shift(1)
        metrics['price_continuity'] = 1 - price_gaps.rolling(self.liquidity_window).mean()

        # 成交量稳定性
        volume_volatility = data['volume'].rolling(self.liquidity_window).std() / \
                           data['volume'].rolling(self.liquidity_window).mean()
        metrics['volume_stability'] = 1 - volume_volatility

        return metrics

    def calculate_liquidity_depth_metrics(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算流动性深度指标

        Args:
            data: 市场数据

        Returns:
            流动性深度指标字典
        """
        metrics = {}

        # 有效价差估计
        high_low_range = (data['high'] - data['low']) / data['close']
        metrics['estimated_spread'] = high_low_range.rolling(self.liquidity_window).mean()

        # Amihud非流动性指标
        returns = data['close'].pct_change()
        dollar_volume = data['close'] * data['volume']
        amihud = abs(returns) / (dollar_volume + 1e-8)
        metrics['amihud_illiquidity'] = amihud.rolling(self.liquidity_window).mean()

        # 流动性比率
        turnover_ratio = data['volume'] / (data['close'] * 1000000)  # 假设100万股流通
        price_range = (data['high'] - data['low']) / data['close']
        metrics['liquidity_ratio'] = turnover_ratio / (price_range + 1e-8)

        # 成交量深度
        volume_depth = data['volume'].rolling(self.liquidity_window).mean()
        metrics['volume_depth'] = volume_depth / volume_depth.rolling(100).mean()

        # 价格吸收能力
        price_absorption = abs(returns) / (data['volume'] + 1e-8)
        metrics['price_absorption'] = price_absorption.rolling(self.liquidity_window).mean()

        return metrics

    def calculate_liquidity_resilience_metrics(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算流动性弹性指标

        Args:
            data: 市场数据

        Returns:
            流动性弹性指标字典
        """
        metrics = {}

        # 成交量恢复速度
        volume_shock = abs(data['volume'].pct_change())
        volume_recovery = 1 / (1 + volume_shock.rolling(self.liquidity_window).mean())
        metrics['volume_resilience'] = volume_recovery

        # 价格恢复速度
        price_shock = abs(data['close'].pct_change())
        price_recovery = 1 / (1 + price_shock.rolling(self.liquidity_window).mean())
        metrics['price_resilience'] = price_recovery

        # 订单流恢复
        order_flow = data['volume'] * np.sign(data['close'] - data['open'])
        of_volatility = order_flow.rolling(self.liquidity_window).std()
        of_mean = order_flow.rolling(self.liquidity_window).mean()
        metrics['order_flow_stability'] = 1 - (of_volatility / (abs(of_mean) + 1e-8))

        # 市场深度恢复
        depth_proxy = data['volume'] / ((data['high'] - data['low']) / data['close'] + 1e-8)
        depth_volatility = depth_proxy.rolling(self.liquidity_window).std()
        metrics['depth_resilience'] = 1 / (1 + depth_volatility)

        return metrics

    def calculate_liquidity_risk_metrics(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算流动性风险指标

        Args:
            data: 市场数据

        Returns:
            流动性风险指标字典
        """
        metrics = {}

        # 流动性黑洞风险
        volume_spike = data['volume'] / data['volume'].rolling(self.liquidity_window).mean()
        price_move = abs(data['close'].pct_change())
        liquidity_black_hole = volume_spike * price_move
        metrics['liquidity_black_hole_risk'] = liquidity_black_hole.rolling(self.liquidity_window).mean()

        # 流动性枯竭风险
        volume_drought = (data['volume'] < data['volume'].rolling(self.liquidity_window).quantile(0.1)).astype(int)
        metrics['liquidity_drought_risk'] = volume_drought.rolling(self.liquidity_window).mean()

        # 买卖价差扩大风险
        spread_proxy = (data['high'] - data['low']) / data['close']
        spread_widening = spread_proxy.rolling(self.liquidity_window).pct_change()
        metrics['spread_widening_risk'] = spread_widening.clip(-5, 5).rolling(self.liquidity_window).mean()

        # 价格滑点风险
        expected_slippage = self._estimate_expected_slippage(data)
        metrics['slippage_risk'] = expected_slippage.rolling(self.liquidity_window).mean()

        # 流动性转换风险
        conversion_risk = self._calculate_conversion_risk(data)
        metrics['conversion_risk'] = conversion_risk.rolling(self.liquidity_window).mean()

        return metrics

    def _estimate_expected_slippage(self, data: pd.DataFrame) -> pd.Series:
        """估计预期滑点"""
        returns = data['close'].pct_change()
        volume_normalized = data['volume'] / data['volume'].rolling(self.liquidity_window).mean()

        # 基于成交量比例估计滑点
        slippage = abs(returns) * (1 / (volume_normalized + 0.1))

        return slippage

    def _calculate_conversion_risk(self, data: pd.DataFrame) -> pd.Series:
        """计算流动性转换风险"""
        # 估计完全平仓所需时间
        daily_volume = data['volume'].rolling(self.liquidity_window).mean()
        position_size_proxy = daily_volume * 0.1  # 假设持仓为日均成交量的10%

        conversion_time = position_size_proxy / daily_volume
        risk_score = conversion_time / 5  # 超过5天风险增加

        return risk_score.fillna(0)

    def detect_liquidity_events(self, data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        检测流动性事件

        Args:
            data: 市场数据

        Returns:
            检测到的流动性事件列表
        """
        events = []

        # 计算各种流动性指标
        tradability = self.calculate_tradability_metrics(data)
        depth = self.calculate_liquidity_depth_metrics(data)
        resilience = self.calculate_liquidity_resilience_metrics(data)
        risk = self.calculate_liquidity_risk_metrics(data)

        # 综合流动性评分
        liquidity_score = self._calculate_composite_liquidity_score(
            tradability, depth, resilience, risk
        )

        # 检测异常事件
        for i in range(len(data)):
            current_score = liquidity_score.iloc[i] if i < len(liquidity_score) else 0.5

            # 检测流动性危机
            if current_score < 0.2:  # 严重流动性不足
                event_type = 'liquidity_crisis'
                severity = 'critical'
            elif current_score < 0.4:  # 流动性紧张
                event_type = 'liquidity_stress'
                severity = 'high'
            else:
                continue

            # 获取详细信息
            event_details = {
                'timestamp': data.index[i],
                'event_type': event_type,
                'severity': severity,
                'liquidity_score': current_score,
                'price': data['close'].iloc[i],
                'volume': data['volume'].iloc[i],
                'price_change': data['close'].pct_change().iloc[i] if i > 0 else 0,
                'volume_change': data['volume'].pct_change().iloc[i] if i > 0 else 0
            }

            # 添加具体指标
            if 'vwap_deviation' in tradability:
                event_details['vwap_deviation'] = tradability['vwap_deviation'].iloc[i]
            if 'amihud_illiquidity' in depth:
                event_details['amihud_illiquidity'] = depth['amihud_illiquidity'].iloc[i]
            if 'liquidity_black_hole_risk' in risk:
                event_details['black_hole_risk'] = risk['liquidity_black_hole_risk'].iloc[i]

            events.append(event_details)

        return events

    def _calculate_composite_liquidity_score(self,
                                           tradability: Dict[str, pd.Series],
                                           depth: Dict[str, pd.Series],
                                           resilience: Dict[str, pd.Series],
                                           risk: Dict[str, pd.Series]) -> pd.Series:
        """计算综合流动性评分"""
        # 提取各维度指标
        scores = {}

        # 可交易性得分
        if 'vwap_deviation' in tradability and 'price_continuity' in tradability:
            vwap_score = 1 - np.clip(tradability['vwap_deviation'], 0, 1)
            continuity_score = tradability['price_continuity']
            scores['tradability'] = (vwap_score + continuity_score) / 2

        # 深度得分
        if 'amihud_illiquidity' in depth and 'liquidity_ratio' in depth:
            amihud_score = 1 / (1 + depth['amihud_illiquidity'] * 1000)
            ratio_score = np.clip(depth['liquidity_ratio'] / 10, 0, 1)
            scores['depth'] = (amihud_score + ratio_score) / 2

        # 弹性得分
        if 'volume_resilience' in resilience and 'price_resilience' in resilience:
            scores['resilience'] = (resilience['volume_resilience'] + resilience['price_resilience']) / 2

        # 风险得分（反向）
        if 'liquidity_black_hole_risk' in risk and 'liquidity_drought_risk' in risk:
            risk_score = 1 - np.clip(risk['liquidity_black_hole_risk'], 0, 1)
            drought_score = 1 - risk['liquidity_drought_risk']
            scores['risk'] = (risk_score + drought_score) / 2

        # 计算综合得分
        if scores:
            weights = {'tradability': 0.3, 'depth': 0.3, 'resilience': 0.2, 'risk': 0.2}
            composite_score = sum(scores[key] * weights.get(key, 0.25) for key in scores)
        else:
            composite_score = pd.Series(0.5, index=data.index)

        return composite_score

    def calculate_liquidity_adjusted_var(self, data: pd.DataFrame,
                                      confidence_level: float = 0.95,
                                      holding_period: int = 1) -> Dict[str, float]:
        """
        计算流动性调整VaR

        Args:
            data: 市场数据
            confidence_level: 置信水平
            holding_period: 持有期间

        Returns:
            流动性调整VaR结果
        """
        returns = data['close'].pct_change().dropna()
        liquidity_metrics = self.calculate_liquidity_depth_metrics(data)

        # 传统VaR
        traditional_var = returns.quantile(1 - confidence_level)

        # 流动性调整
        if 'amihud_illiquidity' in liquidity_metrics:
            current_illiquidity = liquidity_metrics['amihud_illiquidity'].iloc[-1]
            avg_illiquidity = liquidity_metrics['amihud_illiquidity'].mean()

            # 流动性调整因子
            liquidity_adjustment = current_illiquidity / (avg_illiquidity + 1e-8)

            # 流动性调整VaR
            liquidity_adjusted_var = traditional_var * (1 + liquidity_adjustment)

            # 流动性成本
            liquidity_cost = current_illiquidity * np.sqrt(holding_period)
        else:
            liquidity_adjusted_var = traditional_var
            liquidity_cost = 0

        return {
            'traditional_var': traditional_var,
            'liquidity_adjusted_var': liquidity_adjusted_var,
            'liquidity_cost': liquidity_cost,
            'adjustment_factor': liquidity_adjusted_var / traditional_var if traditional_var != 0 else 1,
            'confidence_level': confidence_level,
            'holding_period': holding_period
        }

    def calculate_optimal_execution_schedule(self, data: pd.DataFrame,
                                          order_size: float,
                                          time_horizon: int = 10) -> Dict[str, Any]:
        """
        计算最优执行策略

        Args:
            data: 市场数据
            order_size: 订单规模
            time_horizon: 时间范围（分钟）

        Returns:
            最优执行策略
        """
        # 估计市场冲击函数
        impact_params = self._estimate_market_impact(data)

        # 估计成交量模式
        volume_profile = self._estimate_volume_profile(data)

        # 优化执行策略
        execution_schedule = self._optimize_execution_strategy(
            order_size, time_horizon, impact_params, volume_profile
        )

        # 计算预期成本
        expected_cost = self._calculate_execution_cost(
            execution_schedule, impact_params, volume_profile
        )

        return {
            'execution_schedule': execution_schedule,
            'expected_cost': expected_cost,
            'total_time': time_horizon,
            'avg_participation_rate': order_size / volume_profile['total_volume'] if volume_profile['total_volume'] > 0 else 0,
            'market_impact_estimate': impact_params,
            'volume_profile': volume_profile
        }

    def _estimate_market_impact(self, data: pd.DataFrame) -> Dict[str, float]:
        """估计市场冲击参数"""
        returns = data['close'].pct_change()
        volumes = data['volume']
        avg_volume = volumes.rolling(self.liquidity_window).mean()

        # 简化的平方根冲击函数
        volume_normalized = volumes / (avg_volume + 1e-8)
        impact_estimate = abs(returns) / (np.sqrt(volume_normalized) + 1e-8)

        # 计算平均冲击参数
        valid_impacts = impact_estimate.dropna()
        if len(valid_impacts) > 0:
            alpha = valid_impacts.median()
            beta = 0.5  # 平方根假设
        else:
            alpha = 0.001
            beta = 0.5

        return {
            'alpha': alpha,
            'beta': beta,
            'permanent_impact': alpha * 0.3,
            'temporary_impact': alpha * 0.7
        }

    def _estimate_volume_profile(self, data: pd.DataFrame) -> Dict[str, Any]:
        """估计成交量分布"""
        volumes = data['volume']

        # 计算历史成交量模式
        avg_volume = volumes.rolling(self.liquidity_window).mean()
        volume_volatility = volumes.rolling(self.liquidity_window).std()

        # 时间分布模式（简化）
        intraday_pattern = {
            'open_volume_ratio': 1.2,  # 开盘成交量比例
            'close_volume_ratio': 1.5,  # 收盘成交量比例
            'middle_volume_ratio': 0.8   # 中间时段成交量比例
        }

        return {
            'average_volume': avg_volume.iloc[-1],
            'volume_volatility': volume_volatility.iloc[-1],
            'total_volume': avg_volume.iloc[-1] * 390,  # 假设390分钟交易时间
            'intraday_pattern': intraday_pattern
        }

    def _optimize_execution_strategy(self, order_size: float, time_horizon: int,
                                    impact_params: Dict[str, float],
                                    volume_profile: Dict[str, Any]) -> List[float]:
        """优化执行策略"""
        # 简化的TWAP（时间加权平均价格）策略
        base_participation = order_size / time_horizon

        schedule = []
        for t in range(time_horizon):
            # 根据成交量模式调整
            if t < time_horizon * 0.1:  # 开盘时段
                participation_rate = base_participation * volume_profile['intraday_pattern']['open_volume_ratio']
            elif t > time_horizon * 0.9:  # 收盘时段
                participation_rate = base_participation * volume_profile['intraday_pattern']['close_volume_ratio']
            else:  # 中间时段
                participation_rate = base_participation * volume_profile['intraday_pattern']['middle_volume_ratio']

            schedule.append(participation_rate)

        return schedule

    def _calculate_execution_cost(self, schedule: List[float],
                                impact_params: Dict[str, float],
                                volume_profile: Dict[str, Any]) -> Dict[str, float]:
        """计算执行成本"""
        total_participation = sum(schedule)
        avg_participation = np.mean(schedule)

        # 临时冲击成本
        temporary_cost = impact_params['temporary_impact'] * np.sqrt(avg_participation)

        # 永久冲击成本
        permanent_cost = impact_params['permanent_impact'] * total_participation

        total_cost = temporary_cost + permanent_cost

        return {
            'temporary_cost': temporary_cost,
            'permanent_cost': permanent_cost,
            'total_cost': total_cost,
            'cost_per_share': total_cost,
            'cost_bps': total_cost * 10000  # 基点
        }

    def calculate_liquidity_stress_testing(self, data: pd.DataFrame,
                                         stress_scenarios: List[Dict[str, float]]) -> Dict[str, Any]:
        """
        流动性压力测试

        Args:
            data: 市场数据
            stress_scenarios: 压力测试场景

        Returns:
            压力测试结果
        """
        results = {}

        # 计算基准流动性指标
        baseline_metrics = self.calculate_liquidity_depth_metrics(data)

        for scenario in stress_scenarios:
            scenario_name = scenario.get('name', 'unnamed_scenario')

            # 应用压力参数
            stressed_metrics = self._apply_stress_scenario(data, scenario)

            # 计算影响
            impact_analysis = self._analyze_stress_impact(baseline_metrics, stressed_metrics)

            results[scenario_name] = {
                'scenario_parameters': scenario,
                'stressed_metrics': stressed_metrics,
                'impact_analysis': impact_analysis,
                'worst_case_liquidity': impact_analysis.get('worst_case_score', 0.5),
                'recovery_time_estimate': self._estimate_recovery_time(stressed_metrics)
            }

        return results

    def _apply_stress_scenario(self, data: pd.DataFrame, scenario: Dict[str, float]) -> Dict[str, pd.Series]:
        """应用压力场景"""
        stressed_data = data.copy()

        # 应用价格冲击
        if 'price_shock' in scenario:
            price_shock = scenario['price_shock']
            stressed_data['close'] *= (1 + price_shock)

        # 应用成交量冲击
        if 'volume_shock' in scenario:
            volume_shock = scenario['volume_shock']
            stressed_data['volume'] *= (1 + volume_shock)

        # 应用价差扩大
        if 'spread_widening' in scenario:
            spread_widening = scenario['spread_widening']
            stressed_data['high'] *= (1 + spread_widening / 2)
            stressed_data['low'] *= (1 - spread_widening / 2)

        # 重新计算流动性指标
        stressed_metrics = self.calculate_liquidity_depth_metrics(stressed_data)

        return stressed_metrics

    def _analyze_stress_impact(self, baseline: Dict[str, pd.Series],
                             stressed: Dict[str, pd.Series]) -> Dict[str, float]:
        """分析压力影响"""
        impact = {}

        for key in baseline:
            if key in stressed:
                baseline_val = baseline[key].iloc[-1]
                stressed_val = stressed[key].iloc[-1]

                if baseline_val != 0:
                    change_pct = (stressed_val - baseline_val) / baseline_val
                    impact[key] = change_pct
                else:
                    impact[key] = 0

        # 计算综合影响得分
        if impact:
            avg_impact = np.mean(list(impact.values()))
            worst_case = max(abs(val) for val in impact.values())

            impact['average_impact'] = avg_impact
            impact['worst_case_impact'] = worst_case
            impact['worst_case_score'] = max(0, 1 - worst_case)
        else:
            impact['average_impact'] = 0
            impact['worst_case_impact'] = 0
            impact['worst_case_score'] = 0.5

        return impact

    def _estimate_recovery_time(self, stressed_metrics: Dict[str, pd.Series]) -> float:
        """估计恢复时间"""
        # 基于流动性弹性估计恢复时间
        recovery_rate = 0.1  # 假设每天恢复10%

        worst_score = 0.5  # 压力后的流动性得分
        target_score = 0.8  # 目标流动性得分

        if recovery_rate > 0:
            recovery_time = (target_score - worst_score) / recovery_rate
        else:
            recovery_time = float('inf')

        return max(0, recovery_time)

    def calculate_liquidity_risk_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算流动性风险指标

        Args:
            data: 市场数据

        Returns:
            流动性风险指标DataFrame
        """
        if len(data) < self.liquidity_window:
            return pd.DataFrame()

        # 计算各类指标
        tradability = self.calculate_tradability_metrics(data)
        depth = self.calculate_liquidity_depth_metrics(data)
        resilience = self.calculate_liquidity_resilience_metrics(data)
        risk = self.calculate_liquidity_risk_metrics(data)

        # 创建指标DataFrame
        indicators = pd.DataFrame(index=data.index[-1:])

        # 添加最新指标值
        for name, series in tradability.items():
            indicators[f'tradability_{name}'] = series.iloc[-1]

        for name, series in depth.items():
            indicators[f'depth_{name}'] = series.iloc[-1]

        for name, series in resilience.items():
            indicators[f'resilience_{name}'] = series.iloc[-1]

        for name, series in risk.items():
            indicators[f'risk_{name}'] = series.iloc[-1]

        # 计算综合流动性评分
        composite_score = self._calculate_composite_liquidity_score(
            tradability, depth, resilience, risk
        )
        indicators['composite_liquidity_score'] = composite_score.iloc[-1]

        # 计算流动性调整VaR
        var_result = self.calculate_liquidity_adjusted_var(data)
        indicators['liquidity_adjusted_var'] = var_result['liquidity_adjusted_var']
        indicators['traditional_var'] = var_result['traditional_var']
        indicators['liquidity_cost'] = var_result['liquidity_cost']

        # 检测流动性事件
        events = self.detect_liquidity_events(data)
        if events:
            latest_event = events[-1]
            indicators['recent_event_severity'] = {'low': 1, 'medium': 2, 'high': 3, 'critical': 4}[latest_event['severity']]
            indicators['recent_event_type'] = latest_event['event_type']
        else:
            indicators['recent_event_severity'] = 0
            indicators['recent_event_type'] = 'none'

        return indicators


class LiquidityPortfolioManager:
    """
    流动性投资组合管理器

    管理投资组合的流动性风险和优化资本配置
    """

    def __init__(self,
                 portfolio_size: float = 1000000,
                 liquidity_requirement: float = 0.1,
                 risk_aversion: float = 1.0):
        """
        初始化流动性投资组合管理器

        Args:
            portfolio_size: 投资组合规模
            liquidity_requirement: 流动性要求比例
            risk_aversion: 风险厌恶系数
        """
        self.portfolio_size = portfolio_size
        self.liquidity_requirement = liquidity_requirement
        self.risk_aversion = risk_aversion

        # 投资组合配置
        self.positions = {}
        self.liquidity_buffer = portfolio_size * liquidity_requirement

    def optimize_liquidity_allocation(self, assets_data: Dict[str, pd.DataFrame],
                                    target_weights: Dict[str, float]) -> Dict[str, Any]:
        """
        优化流动性配置

        Args:
            assets_data: 各资产数据
            target_weights: 目标权重

        Returns:
            优化配置结果
        """
        # 计算各资产流动性指标
        asset_liquidity = {}
        for asset, data in assets_data.items():
            analyzer = LiquidityRiskAnalyzer()
            metrics = analyzer.calculate_liquidity_depth_metrics(data)
            asset_liquidity[asset] = metrics

        # 计算流动性调整后的权重
        adjusted_weights = self._calculate_liquidity_adjusted_weights(
            target_weights, asset_liquidity
        )

        # 计算执行策略
        execution_strategies = {}
        for asset, weight in adjusted_weights.items():
            if asset in assets_data:
                target_position = self.portfolio_size * weight
                execution_strategy = analyzer.calculate_optimal_execution_schedule(
                    assets_data[asset], target_position
                )
                execution_strategies[asset] = execution_strategy

        return {
            'adjusted_weights': adjusted_weights,
            'execution_strategies': execution_strategies,
            'liquidity_buffer': self.liquidity_buffer,
            'portfolio_liquidity_score': self._calculate_portfolio_liquidity_score(adjusted_weights, asset_liquidity)
        }

    def _calculate_liquidity_adjusted_weights(self, target_weights: Dict[str, float],
                                            asset_liquidity: Dict[str, Dict[str, pd.Series]]) -> Dict[str, float]:
        """计算流动性调整权重"""
        adjusted_weights = {}
        total_liquidity_score = 0

        # 计算流动性评分
        liquidity_scores = {}
        for asset in target_weights:
            if asset in asset_liquidity:
                metrics = asset_liquidity[asset]
                # 简化的流动性评分
                if 'amihud_illiquidity' in metrics and 'liquidity_ratio' in metrics:
                    amihud_score = 1 / (1 + metrics['amihud_illiquidity'].iloc[-1] * 1000)
                    ratio_score = min(1, metrics['liquidity_ratio'].iloc[-1] / 5)
                    liquidity_scores[asset] = (amihud_score + ratio_score) / 2
                else:
                    liquidity_scores[asset] = 0.5
            else:
                liquidity_scores[asset] = 0.5

            total_liquidity_score += liquidity_scores[asset] * target_weights[asset]

        # 调整权重
        for asset in target_weights:
            liquidity_adjustment = liquidity_scores[asset] / (total_liquidity_score + 1e-8)
            adjusted_weights[asset] = target_weights[asset] * liquidity_adjustment

        # 归一化
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for asset in adjusted_weights:
                adjusted_weights[asset] /= total_weight

        return adjusted_weights

    def _calculate_portfolio_liquidity_score(self, weights: Dict[str, float],
                                           asset_liquidity: Dict[str, Dict[str, pd.Series]]) -> float:
        """计算投资组合流动性评分"""
        portfolio_score = 0

        for asset, weight in weights.items():
            if asset in asset_liquidity:
                metrics = asset_liquidity[asset]
                if 'amihud_illiquidity' in metrics:
                    amihud = metrics['amihud_illiquidity'].iloc[-1]
                    asset_score = 1 / (1 + amihud * 1000)
                    portfolio_score += weight * asset_score

        return portfolio_score


def create_liquidity_risk_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    创建流动性风险特征

    Args:
        data: 市场数据

    Returns:
        流动性风险特征DataFrame
    """
    analyzer = LiquidityRiskAnalyzer()
    indicators = analyzer.calculate_liquidity_risk_indicators(data)
    return indicators


# 主要功能函数
def calculate_liquidity_risk_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有流动性风险指标

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

    return create_liquidity_risk_features(data)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=500, freq='5min')

    # 模拟高频价格数据
    initial_price = 100
    returns = np.random.normal(0, 0.001, len(dates))
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
        indicators = calculate_liquidity_risk_indicators(sample_data)
        print("流动性风险指标计算成功!")
        print(f"指标数量: {indicators.shape[1]}")
        print("最新指标值:")
        print(indicators.iloc[-1])

        # 检测流动性事件
        analyzer = LiquidityRiskAnalyzer()
        events = analyzer.detect_liquidity_events(sample_data)
        print(f"\n检测到 {len(events)} 个流动性事件")

    except Exception as e:
        print(f"计算错误: {e}")