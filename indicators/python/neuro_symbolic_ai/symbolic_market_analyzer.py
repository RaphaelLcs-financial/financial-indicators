"""
符号市场分析器
Symbolic Market Analyzer

基于符号逻辑和规则推理的市场分析系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import sympy as sp
    from sympy import symbols, Eq, solve, sympify, And, Or, Not
    SYMPY_AVAILABLE = True
    print("🔢 SymPy 已启用")
except ImportError:
    SYMPY_AVAILABLE = False
    print("⚠️ SymPy 不可用")

class SymbolicMarketAnalyzer:
    """
    符号市场分析器

    使用符号逻辑和规则推理进行市场分析
    """

    def __init__(self,
                 knowledge_base: Dict[str, Any] = None,
                 inference_engine: str = 'forward_chaining',
                 logical_depth: int = 5):
        """
        初始化符号分析器

        Args:
            knowledge_base: 知识库
            inference_engine: 推理引擎类型
            logical_depth: 逻辑推理深度
        """
        self.knowledge_base = knowledge_base or self._default_financial_knowledge()
        self.inference_engine = inference_engine
        self.logical_depth = logical_depth

        # 符号变量
        self.symbolic_vars = self._initialize_symbolic_variables()

        # 推理结果
        self.logical_derivations = []
        self.inference_path = []
        self.symbolic_interpretations = {}

        # 事实库
        self.facts = {}
        self.derived_facts = {}

        # 规则库
        self.rule_base = self._initialize_rule_base()

    def _default_financial_knowledge(self) -> Dict[str, Any]:
        """默认金融知识库"""
        return {
            'market_regimes': {
                'bull_market': {
                    'conditions': ['price_trend > 0.02', 'volume > avg_volume', 'momentum > 0'],
                    'characteristics': ['rising_prices', 'high_volume', 'positive_sentiment']
                },
                'bear_market': {
                    'conditions': ['price_trend < -0.02', 'volume > avg_volume', 'momentum < 0'],
                    'characteristics': ['falling_prices', 'high_volume', 'negative_sentiment']
                },
                'sideways_market': {
                    'conditions': ['abs(price_trend) < 0.01', 'volatility < avg_volatility'],
                    'characteristics': ['stable_prices', 'low_volume', 'neutral_sentiment']
                }
            },
            'technical_patterns': {
                'uptrend': {
                    'conditions': ['higher_highs', 'higher_lows', 'rising_sma'],
                    'interpretation': 'bullish_continuation'
                },
                'downtrend': {
                    'conditions': ['lower_highs', 'lower_lows', 'falling_sma'],
                    'interpretation': 'bearish_continuation'
                },
                'reversal': {
                    'conditions': ['divergence', 'overbought_oversold', 'volume_spike'],
                    'interpretation': 'trend_reversal'
                }
            },
            'risk_levels': {
                'low_risk': {
                    'conditions': ['volatility < 0.15', 'liquidity > 0.8', 'correlation < 0.3'],
                    'actions': ['position_sizing_normal', 'stop_loss_tight']
                },
                'medium_risk': {
                    'conditions': ['0.15 <= volatility < 0.25', 'liquidity > 0.6'],
                    'actions': ['position_sizing_reduced', 'stop_loss_wider']
                },
                'high_risk': {
                    'conditions': ['volatility >= 0.25', 'liquidity < 0.4', 'correlation > 0.7'],
                    'actions': ['position_sizing_minimal', 'hedging_strategies']
                }
            }
        }

    def _initialize_symbolic_variables(self) -> Dict[str, Any]:
        """初始化符号变量"""
        if SYMPY_AVAILABLE:
            return {
                'price': symbols('price'),
                'volume': symbols('volume'),
                'momentum': symbols('momentum'),
                'volatility': symbols('volatility'),
                'trend': symbols('trend'),
                'correlation': symbols('correlation'),
                'liquidity': symbols('liquidity'),
                'sentiment': symbols('sentiment')
            }
        else:
            return {
                'price': 'price_var',
                'volume': 'volume_var',
                'momentum': 'momentum_var',
                'volatility': 'volatility_var',
                'trend': 'trend_var',
                'correlation': 'correlation_var',
                'liquidity': 'liquidity_var',
                'sentiment': 'sentiment_var'
            }

    def _initialize_rule_base(self) -> List[Dict[str, Any]]:
        """初始化规则库"""
        return [
            {
                'name': 'trend_identification',
                'conditions': [
                    {'variable': 'price_trend', 'operator': '>', 'value': 0.02},
                    {'variable': 'momentum', 'operator': '>', 'value': 0}
                ],
                'conclusion': 'uptrend',
                'confidence': 0.8
            },
            {
                'name': 'risk_assessment',
                'conditions': [
                    {'variable': 'volatility', 'operator': '>', 'value': 0.25},
                    {'variable': 'liquidity', 'operator': '<', 'value': 0.4}
                ],
                'conclusion': 'high_risk',
                'confidence': 0.9
            },
            {
                'name': 'market_regime_detection',
                'conditions': [
                    {'variable': 'price_trend', 'operator': '>', 'value': 0.02},
                    {'variable': 'volume', 'operator': '>', 'value': 1.0},
                    {'variable': 'sentiment', 'operator': '>', 'value': 0.5}
                ],
                'conclusion': 'bull_market',
                'confidence': 0.85
            }
        ]

    def symbolic_analysis(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """符号分析主函数"""
        print("🔍 执行符号市场分析...")

        # 第一步：数据符号化
        symbolic_data = self._symbolize_market_data(market_data)

        # 第二步：事实提取
        self.facts = self._extract_facts(symbolic_data)

        # 第三步：逻辑推理
        if self.inference_engine == 'forward_chaining':
            reasoning_results = self._forward_chaining_reasoning()
        else:
            reasoning_results = self._backward_chaining_reasoning()

        # 第四步：符号解释
        symbolic_interpretations = self._generate_symbolic_interpretations(reasoning_results)

        # 第五步：逻辑验证
        logical_validation = self._validate_logical_consistency(reasoning_results)

        # 第六步：生成建议
        recommendations = self._generate_recommendations(reasoning_results, symbolic_interpretations)

        results = {
            'symbolic_data': symbolic_data,
            'facts': self.facts,
            'reasoning_results': reasoning_results,
            'symbolic_interpretations': symbolic_interpretations,
            'logical_validation': logical_validation,
            'recommendations': recommendations,
            'inference_path': self.inference_path,
            'logical_derivations': self.logical_derivations,
            'analysis_depth': self.logical_depth
        }

        return results

    def _symbolize_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """市场数据符号化"""
        print("📊 数据符号化...")

        symbolic_data = {}

        # 价格数据符号化
        if 'close' in market_data.columns:
            prices = market_data['close'].dropna()
            if len(prices) > 0:
                symbolic_data['price'] = self._symbolize_price_series(prices)
                symbolic_data['price_trend'] = self._symbolize_trend(prices)

        # 成交量数据符号化
        if 'volume' in market_data.columns:
            volumes = market_data['volume'].dropna()
            if len(volumes) > 0:
                symbolic_data['volume'] = self._symbolize_volume_series(volumes)

        # 多变量关系符号化
        numeric_columns = market_data.select_dtypes(include=[np.number]).columns
        if len(numeric_columns) >= 2:
            symbolic_data['correlations'] = self._symbolize_correlations(market_data[numeric_columns])
            symbolic_data['volatility'] = self._symbolize_volatility(market_data)

        # 市场结构符号化
        symbolic_data['market_structure'] = self._symbolize_market_structure(market_data)

        return symbolic_data

    def _symbolize_price_series(self, prices: pd.Series) -> Dict[str, Any]:
        """价格序列符号化"""
        symbolization = {}

        # 价格水平
        current_price = prices.iloc[-1]
        price_mean = prices.mean()
        price_std = prices.std()

        if current_price > price_mean + price_std:
            symbolization['level'] = 'high'
        elif current_price < price_mean - price_std:
            symbolization['level'] = 'low'
        else:
            symbolization['level'] = 'normal'

        # 价格变化
        if len(prices) > 1:
            price_change = (prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2]
            if price_change > 0.02:
                symbolization['change'] = 'strong_increase'
            elif price_change > 0.005:
                symbolization['change'] = 'moderate_increase'
            elif price_change > -0.005:
                symbolization['change'] = 'stable'
            elif price_change > -0.02:
                symbolization['change'] = 'moderate_decrease'
            else:
                symbolization['change'] = 'strong_decrease'

        return symbolization

    def _symbolize_trend(self, prices: pd.Series) -> Dict[str, Any]:
        """趋势符号化"""
        trend = {}

        if len(prices) < 10:
            return trend

        # 线性趋势
        x = np.arange(len(prices))
        y = prices.values
        slope, _ = np.polyfit(x, y, 1)

        if slope > 0.01:
            trend['direction'] = 'strong_uptrend'
        elif slope > 0.002:
            trend['direction'] = 'moderate_uptrend'
        elif slope > -0.002:
            trend['direction'] = 'sideways'
        elif slope > -0.01:
            trend['direction'] = 'moderate_downtrend'
        else:
            trend['direction'] = 'strong_downtrend'

        # 趋势强度
        r_squared = self._calculate_r_squared(x, y)
        if r_squared > 0.8:
            trend['strength'] = 'strong'
        elif r_squared > 0.5:
            trend['strength'] = 'moderate'
        else:
            trend['strength'] = 'weak'

        return trend

    def _symbolize_volume_series(self, volumes: pd.Series) -> Dict[str, Any]:
        """成交量序列符号化"""
        volume_symbolization = {}

        # 成交量水平
        current_volume = volumes.iloc[-1]
        avg_volume = volumes.mean()
        volume_std = volumes.std()

        if current_volume > avg_volume + 2 * volume_std:
            volume_symbolization['level'] = 'very_high'
        elif current_volume > avg_volume + volume_std:
            volume_symbolization['level'] = 'high'
        elif current_volume < avg_volume - volume_std:
            volume_symbolization['level'] = 'low'
        else:
            volume_symbolization['level'] = 'normal'

        # 成交量趋势
        if len(volumes) > 5:
            volume_trend = np.polyfit(range(5), volumes.iloc[-5:].values, 1)[0]
            if volume_trend > 0:
                volume_symbolization['trend'] = 'increasing'
            elif volume_trend < 0:
                volume_symbolization['trend'] = 'decreasing'
            else:
                volume_symbolization['trend'] = 'stable'

        return volume_symbolization

    def _symbolize_correlations(self, data: pd.DataFrame) -> Dict[str, Any]:
        """相关性符号化"""
        correlations = {}

        corr_matrix = data.corr()
        avg_corr = corr_matrix.abs().mean()

        if avg_corr > 0.7:
            correlations['market_coupling'] = 'high'
        elif avg_corr > 0.4:
            correlations['market_coupling'] = 'moderate'
        else:
            correlations['market_coupling'] = 'low'

        # 相关性模式
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))

        correlations['high_correlation_groups'] = high_corr_pairs

        return correlations

    def _symbolize_volatility(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """波动率符号化"""
        volatility_symbolization = {}

        if 'close' in market_data.columns:
            prices = market_data['close'].dropna()
            if len(prices) > 10:
                returns = prices.pct_change().dropna()
                volatility = returns.std()

                if volatility > 0.25:
                    volatility_symbolization['level'] = 'very_high'
                elif volatility > 0.15:
                    volatility_symbolization['level'] = 'high'
                elif volatility > 0.08:
                    volatility_symbolization['level'] = 'moderate'
                else:
                    volatility_symbolization['level'] = 'low'

                # 波动率趋势
                if len(returns) > 20:
                    recent_vol = returns.iloc[-20:].std()
                    earlier_vol = returns.iloc[-40:-20].std() if len(returns) > 40 else recent_vol

                    if recent_vol > earlier_vol * 1.2:
                        volatility_symbolization['trend'] = 'increasing'
                    elif recent_vol < earlier_vol * 0.8:
                        volatility_symbolization['trend'] = 'decreasing'
                    else:
                        volatility_symbolization['trend'] = 'stable'

        return volatility_symbolization

    def _symbolize_market_structure(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """市场结构符号化"""
        structure = {}

        # 价格层次结构
        if 'close' in market_data.columns and 'high' in market_data.columns and 'low' in market_data.columns:
            prices = market_data['close'].dropna()
            highs = market_data['high'].dropna()
            lows = market_data['low'].dropna()

            if len(prices) > 20:
                # 支撑阻力位
                resistance_levels = self._identify_resistance_levels(highs)
                support_levels = self._identify_support_levels(lows)

                structure['resistance_levels'] = resistance_levels
                structure['support_levels'] = support_levels

                # 价格相对位置
                current_price = prices.iloc[-1]
                if resistance_levels and current_price > resistance_levels[0]:
                    structure['price_position'] = 'above_resistance'
                elif support_levels and current_price < support_levels[-1]:
                    structure['price_position'] = 'below_support'
                else:
                    structure['price_position'] = 'in_range'

        return structure

    def _extract_facts(self, symbolic_data: Dict[str, Any]) -> Dict[str, Any]:
        """提取事实"""
        facts = {}

        # 展平符号化数据
        for category, data in symbolic_data.items():
            if isinstance(data, dict):
                for key, value in data.items():
                    facts[f"{category}_{key}"] = value
            else:
                facts[category] = data

        # 添加元事实
        facts['analysis_timestamp'] = pd.Timestamp.now()
        facts['data_quality'] = self._assess_data_quality(symbolic_data)

        return facts

    def _assess_data_quality(self, symbolic_data: Dict[str, Any]) -> str:
        """评估数据质量"""
        quality_score = 0

        # 基于数据完整性
        if 'price' in symbolic_data:
            quality_score += 0.3
        if 'volume' in symbolic_data:
            quality_score += 0.2
        if 'volatility' in symbolic_data:
            quality_score += 0.2
        if 'correlations' in symbolic_data:
            quality_score += 0.2
        if 'market_structure' in symbolic_data:
            quality_score += 0.1

        if quality_score > 0.8:
            return 'high'
        elif quality_score > 0.5:
            return 'medium'
        else:
            return 'low'

    def _forward_chaining_reasoning(self) -> Dict[str, Any]:
        """前向链推理"""
        print("🔗 执行前向链推理...")

        reasoning_results = {}
        self.inference_path = []

        # 初始化事实集
        working_facts = self.facts.copy()
        iteration = 0

        while iteration < self.logical_depth:
            new_facts_derived = False
            iteration += 1

            # 应用规则库中的每条规则
            for rule in self.rule_base:
                rule_result = self._apply_symbolic_rule(rule, working_facts)

                if rule_result['triggered']:
                    # 添加新事实
                    for new_fact in rule_result['derived_facts']:
                        if new_fact not in working_facts:
                            working_facts[new_fact] = rule_result['conclusion']
                            new_facts_derived = True

                    # 记录推理步骤
                    self.inference_path.append({
                        'iteration': iteration,
                        'rule': rule['name'],
                        'result': rule_result,
                        'new_facts': rule_result['derived_facts']
                    })

            # 如果没有新事实被推导，停止推理
            if not new_facts_derived:
                break

        reasoning_results['final_facts'] = working_facts
        reasoning_results['inference_iterations'] = iteration
        reasoning_results['rules_triggered'] = len([step for step in self.inference_path if step['result']['triggered']])

        return reasoning_results

    def _backward_chaining_reasoning(self) -> Dict[str, Any]:
        """后向链推理"""
        print("🔍 执行后向链推理...")

        reasoning_results = {}
        self.inference_path = []

        # 目标：验证市场状态
        goals = ['bull_market', 'bear_market', 'sideways_market', 'high_risk', 'low_risk']

        for goal in goals:
            goal_result = self._prove_goal(goal, self.facts, self.logical_depth)
            reasoning_results[goal] = goal_result

            if goal_result['proven']:
                self.inference_path.append({
                    'goal': goal,
                    'proven': True,
                    'confidence': goal_result['confidence'],
                    'proof_path': goal_result['proof_path']
                })

        return reasoning_results

    def _apply_symbolic_rule(self, rule: Dict[str, Any], facts: Dict[str, Any]) -> Dict[str, Any]:
        """应用符号规则"""
        result = {
            'triggered': False,
            'confidence': 0,
            'derived_facts': [],
            'conclusion': None,
            'conditions_met': [],
            'conditions_failed': []
        }

        # 检查所有条件
        all_conditions_met = True
        condition_confidences = []

        for condition in rule['conditions']:
            condition_met, confidence = self._evaluate_symbolic_condition(condition, facts)

            if condition_met:
                result['conditions_met'].append(condition)
                condition_confidences.append(confidence)
            else:
                result['conditions_failed'].append(condition)
                all_conditions_met = False

        # 如果所有条件都满足，触发规则
        if all_conditions_met and condition_confidences:
            result['triggered'] = True
            result['confidence'] = min(condition_confidences) * rule['confidence']
            result['conclusion'] = rule['conclusion']
            result['derived_facts'] = [rule['conclusion']]

        return result

    def _evaluate_symbolic_condition(self, condition: Dict[str, Any], facts: Dict[str, Any]) -> Tuple[bool, float]:
        """评估符号条件"""
        variable = condition['variable']
        operator = condition['operator']
        value = condition['value']

        # 在事实中查找变量值
        fact_value = facts.get(variable)

        if fact_value is None:
            return False, 0.0

        # 根据变量类型评估
        if isinstance(fact_value, str):
            # 符号值评估
            return self._evaluate_symbolic_condition_string(fact_value, operator, value)
        elif isinstance(fact_value, (int, float)):
            # 数值评估
            return self._evaluate_symbolic_condition_numeric(fact_value, operator, value)
        else:
            return False, 0.0

    def _evaluate_symbolic_condition_string(self, fact_value: str, operator: str, target_value: str) -> Tuple[bool, float]:
        """评估字符串条件"""
        if operator == '==':
            return fact_value == target_value, 1.0 if fact_value == target_value else 0.0
        elif operator == '!=':
            return fact_value != target_value, 1.0 if fact_value != target_value else 0.0
        else:
            return False, 0.0

    def _evaluate_symbolic_condition_numeric(self, fact_value: float, operator: str, target_value: float) -> Tuple[bool, float]:
        """评估数值条件"""
        if operator == '>':
            return fact_value > target_value, min(1.0, (fact_value - target_value) / target_value) if target_value != 0 else 1.0
        elif operator == '<':
            return fact_value < target_value, min(1.0, (target_value - fact_value) / target_value) if target_value != 0 else 1.0
        elif operator == '>=':
            return fact_value >= target_value, min(1.0, (fact_value - target_value) / target_value) if target_value != 0 else 1.0
        elif operator == '<=':
            return fact_value <= target_value, min(1.0, (target_value - fact_value) / target_value) if target_value != 0 else 1.0
        elif operator == '==':
            return abs(fact_value - target_value) < 1e-6, 1.0 if abs(fact_value - target_value) < 1e-6 else 0.0
        else:
            return False, 0.0

    def _prove_goal(self, goal: str, facts: Dict[str, Any], depth: int) -> Dict[str, Any]:
        """证明目标"""
        result = {
            'proven': False,
            'confidence': 0,
            'proof_path': [],
            'remaining_depth': depth
        }

        if depth <= 0:
            return result

        # 查找能够推导出目标的规则
        relevant_rules = [rule for rule in self.rule_base if rule['conclusion'] == goal]

        for rule in relevant_rules:
            # 递归证明规则的条件
            all_conditions_proven = True
            condition_proofs = []
            total_confidence = 1.0

            for condition in rule['conditions']:
                # 尝试从事实中证明条件
                condition_met, confidence = self._evaluate_symbolic_condition(condition, facts)

                if condition_met:
                    condition_proofs.append({
                        'condition': condition,
                        'proven': True,
                        'confidence': confidence
                    })
                    total_confidence *= confidence
                else:
                    all_conditions_proven = False
                    break

            if all_conditions_proven:
                result['proven'] = True
                result['confidence'] = total_confidence * rule['confidence']
                result['proof_path'] = condition_proofs
                break

        return result

    def _generate_symbolic_interpretations(self, reasoning_results: Dict[str, Any]) -> Dict[str, Any]:
        """生成符号解释"""
        interpretations = {}

        # 市场状态解释
        if 'final_facts' in reasoning_results:
            final_facts = reasoning_results['final_facts']

            # 推导市场状态
            market_state = self._interpret_market_state(final_facts)
            interpretations['market_state'] = market_state

            # 风险评估解释
            risk_assessment = self._interpret_risk_assessment(final_facts)
            interpretations['risk_assessment'] = risk_assessment

            # 交易机会解释
            trading_opportunities = self._interpret_trading_opportunities(final_facts)
            interpretations['trading_opportunities'] = trading_opportunities

        # 推理路径解释
        if self.inference_path:
            path_interpretation = self._interpret_inference_path(self.inference_path)
            interpretations['reasoning_path'] = path_interpretation

        return interpretations

    def _interpret_market_state(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """解释市场状态"""
        market_state = {}

        # 基于事实推断市场状态
        if 'bull_market' in facts:
            market_state['regime'] = 'bull_market'
            market_state['description'] = '牛市：价格上升趋势，成交量放大，市场情绪积极'
        elif 'bear_market' in facts:
            market_state['regime'] = 'bear_market'
            market_state['description'] = '熊市：价格下降趋势，成交量增加，市场情绪消极'
        elif 'sideways_market' in facts:
            market_state['regime'] = 'sideways_market'
            market_state['description'] = '横盘市场：价格稳定，成交量正常，市场情绪中性'
        else:
            market_state['regime'] = 'unknown'
            market_state['description'] = '市场状态不明确'

        # 趋势分析
        if 'price_trend_direction' in facts:
            trend_direction = facts['price_trend_direction']
            market_state['trend_direction'] = trend_direction
            market_state['trend_description'] = self._describe_trend(trend_direction)

        return market_state

    def _interpret_risk_assessment(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """解释风险评估"""
        risk_assessment = {}

        # 风险等级
        if 'high_risk' in facts:
            risk_assessment['level'] = 'high'
            risk_assessment['description'] = '高风险：高波动率，低流动性，需要谨慎操作'
        elif 'medium_risk' in facts:
            risk_assessment['level'] = 'medium'
            risk_assessment['description'] = '中等风险：正常波动率，适度流动性，可进行适量交易'
        elif 'low_risk' in facts:
            risk_assessment['level'] = 'low'
            risk_assessment['description'] = '低风险：低波动率，高流动性，适合积极操作'
        else:
            risk_assessment['level'] = 'unknown'
            risk_assessment['description'] = '风险等级不明确'

        # 风险因素分析
        risk_factors = []
        if 'volatility_level' in facts:
            risk_factors.append(f"波动率: {facts['volatility_level']}")
        if 'volume_level' in facts:
            risk_factors.append(f"成交量: {facts['volume_level']}")

        risk_assessment['risk_factors'] = risk_factors

        return risk_assessment

    def _interpret_trading_opportunities(self, facts: Dict[str, Any]) -> Dict[str, Any]:
        """解释交易机会"""
        opportunities = {}

        # 基于趋势的机会
        if 'price_trend_direction' in facts:
            trend = facts['price_trend_direction']
            if 'uptrend' in trend:
                opportunities['trend_opportunity'] = '买入机会：上升趋势延续'
            elif 'downtrend' in trend:
                opportunities['trend_opportunity'] = '卖出机会：下降趋势延续'
            else:
                opportunities['trend_opportunity'] = '观望：横盘整理'

        # 基于波动率的机会
        if 'volatility_level' in facts:
            volatility = facts['volatility_level']
            if volatility in ['high', 'very_high']:
                opportunities['volatility_opportunity'] = '短线交易机会：高波动率提供短线操作空间'
            else:
                opportunities['volatility_opportunity'] = '长线持有机会：低波动率适合长线投资'

        return opportunities

    def _interpret_inference_path(self, inference_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """解释推理路径"""
        path_analysis = {}

        # 推理深度
        path_analysis['reasoning_depth'] = len(inference_path)

        # 触发的规则
        triggered_rules = [step['rule'] for step in inference_path if step['result']['triggered']]
        path_analysis['triggered_rules'] = triggered_rules

        # 推理质量
        if triggered_rules:
            path_analysis['reasoning_quality'] = 'high'
            path_analysis['quality_description'] = f'通过{len(triggered_rules)}条规则成功推导'
        else:
            path_analysis['reasoning_quality'] = 'low'
            path_analysis['quality_description'] = '未能触发有效规则'

        return path_analysis

    def _validate_logical_consistency(self, reasoning_results: Dict[str, Any]) -> Dict[str, Any]:
        """验证逻辑一致性"""
        validation = {
            'is_consistent': True,
            'conflicts': [],
            'warnings': []
        }

        # 检查事实间的一致性
        if 'final_facts' in reasoning_results:
            final_facts = reasoning_results['final_facts']

            # 检查冲突的市场状态
            market_states = [state for state in final_facts.keys() if 'market' in state]
            if len(market_states) > 1:
                conflict = {
                    'type': 'multiple_market_states',
                    'states': market_states,
                    'description': '检测到多个市场状态，可能存在逻辑冲突'
                }
                validation['conflicts'].append(conflict)
                validation['is_consistent'] = False

            # 检查冲突的风险等级
            risk_levels = [level for level in final_facts.keys() if 'risk' in level]
            if len(risk_levels) > 1:
                conflict = {
                    'type': 'multiple_risk_levels',
                    'levels': risk_levels,
                    'description': '检测到多个风险等级，可能存在逻辑冲突'
                }
                validation['conflicts'].append(conflict)
                validation['is_consistent'] = False

        # 检查推理路径的合理性
        if len(self.inference_path) == 0:
            warning = {
                'type': 'no_reasoning_path',
                'description': '没有产生推理路径，可能规则库不完善'
            }
            validation['warnings'].append(warning)

        return validation

    def _generate_recommendations(self,
                                reasoning_results: Dict[str, Any],
                                symbolic_interpretations: Dict[str, Any]) -> Dict[str, Any]:
        """生成建议"""
        recommendations = {
            'trading_recommendations': [],
            'risk_management_suggestions': [],
            'market_outlook': '',
            'confidence_level': 0
        }

        # 基于推理结果生成交易建议
        if 'final_facts' in reasoning_results:
            final_facts = reasoning_results['final_facts']

            # 交易建议
            if 'bull_market' in final_facts:
                recommendations['trading_recommendations'].append('考虑增加多头仓位')
                recommendations['market_outlook'] = '市场看涨'
            elif 'bear_market' in final_facts:
                recommendations['trading_recommendations'].append('考虑减少仓位或增加空头仓位')
                recommendations['market_outlook'] = '市场看跌'
            else:
                recommendations['trading_recommendations'].append('保持现有仓位，观望市场变化')
                recommendations['market_outlook'] = '市场中性'

            # 风险管理建议
            if 'high_risk' in final_facts:
                recommendations['risk_management_suggestions'].append('设置较紧的止损')
                recommendations['risk_management_suggestions'].append('减少仓位规模')
            elif 'low_risk' in final_facts:
                recommendations['risk_management_suggestions'].append('可以适当增加仓位')

        # 计算置信度
        if self.inference_path:
            avg_confidence = np.mean([
                step['result']['confidence'] for step in self.inference_path
                if step['result']['triggered']
            ])
            recommendations['confidence_level'] = avg_confidence
        else:
            recommendations['confidence_level'] = 0

        return recommendations

    def _calculate_r_squared(self, x: np.ndarray, y: np.ndarray) -> float:
        """计算R平方"""
        if len(x) < 2:
            return 0

        slope, intercept = np.polyfit(x, y, 1)
        y_pred = slope * x + intercept

        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)

        return 1 - (ss_residual / ss_total) if ss_total != 0 else 0

    def _identify_resistance_levels(self, highs: pd.Series, n_levels: int = 3) -> List[float]:
        """识别阻力位"""
        if len(highs) < 10:
            return []

        # 简单的局部最大值识别
        resistance_levels = []

        for i in range(2, len(highs) - 2):
            if (highs.iloc[i] > highs.iloc[i-1] and
                highs.iloc[i] > highs.iloc[i+1] and
                highs.iloc[i] > highs.iloc[i-2] and
                highs.iloc[i] > highs.iloc[i+2]):

                resistance_levels.append(highs.iloc[i])

        # 返回前n个最高的阻力位
        resistance_levels.sort(reverse=True)
        return resistance_levels[:n_levels]

    def _identify_support_levels(self, lows: pd.Series, n_levels: int = 3) -> List[float]:
        """识别支撑位"""
        if len(lows) < 10:
            return []

        # 简单的局部最小值识别
        support_levels = []

        for i in range(2, len(lows) - 2):
            if (lows.iloc[i] < lows.iloc[i-1] and
                lows.iloc[i] < lows.iloc[i+1] and
                lows.iloc[i] < lows.iloc[i-2] and
                lows.iloc[i] < lows.iloc[i+2]):

                support_levels.append(lows.iloc[i])

        # 返回前n个最低的支撑位
        support_levels.sort()
        return support_levels[:n_levels]

    def _describe_trend(self, trend_direction: str) -> str:
        """描述趋势"""
        trend_descriptions = {
            'strong_uptrend': '强劲上升趋势',
            'moderate_uptrend': '中等上升趋势',
            'sideways': '横盘整理',
            'moderate_downtrend': '中等下降趋势',
            'strong_downtrend': '强劲下降趋势'
        }
        return trend_descriptions.get(trend_direction, '趋势不明确')

    def get_analysis_summary(self) -> Dict[str, Any]:
        """获取分析总结"""
        return {
            'total_facts': len(self.facts),
            'derived_facts': len(self.derived_facts),
            'inference_path_length': len(self.inference_path),
            'logical_derivations_count': len(self.logical_derivations),
            'rules_in_base': len(self.rule_base),
            'inference_engine': self.inference_engine,
            'logical_depth': self.logical_depth,
            'model_info': self.get_model_info()
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'Symbolic Market Analyzer',
            'sympy_available': SYMPY_AVAILABLE,
            'inference_engine': self.inference_engine,
            'logical_depth': self.logical_depth,
            'knowledge_base_size': len(self.knowledge_base),
            'rule_base_size': len(self.rule_base),
            'symbolic_variables_count': len(self.symbolic_vars)
        }


# 便捷函数
def create_symbolic_analyzer(knowledge_base: Dict[str, Any] = None) -> SymbolicMarketAnalyzer:
    """创建符号分析器实例"""
    return SymbolicMarketAnalyzer(knowledge_base=knowledge_base)

def quick_symbolic_analysis(market_data: pd.DataFrame) -> Dict[str, Any]:
    """快速符号分析"""
    analyzer = create_symbolic_analyzer()
    return analyzer.symbolic_analysis(market_data)

def get_market_regime_interpretation(market_data: pd.DataFrame) -> str:
    """获取市场状态解释"""
    results = quick_symbolic_analysis(market_data)
    market_state = results.get('symbolic_interpretations', {}).get('market_state', {})
    return market_state.get('description', '无法确定市场状态')