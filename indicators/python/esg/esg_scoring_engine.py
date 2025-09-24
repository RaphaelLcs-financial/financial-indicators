"""
ESG评分引擎

综合的环境、社会和治理(ESG)评分系统，结合多源数据分析公司可持续发展表现。
支持动态权重调整、行业对比分析和风险评级。

算法特点:
- 多维度ESG指标整合
- 行业标准化和相对评分
- 动态权重优化
- 时间序列趋势分析
- 风险暴露评估

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
import logging
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class ESGData:
    """ESG数据结构"""
    company_id: str
    company_name: str
    industry: str
    sector: str
    environmental_metrics: Dict[str, float]
    social_metrics: Dict[str, float]
    governance_metrics: Dict[str, float]
    reporting_year: int
    data_quality_score: float

@dataclass
class ESGScore:
    """ESG评分结果"""
    company_id: str
    overall_score: float
    environmental_score: float
    social_score: float
    governance_score: float
    industry_rank: int
    industry_percentile: float
    risk_rating: str
    trend_score: float
    confidence_level: float

class ESGScoringEngine:
    """ESG评分引擎"""

    def __init__(self,
                 environmental_weight: float = 0.33,
                 social_weight: float = 0.33,
                 governance_weight: float = 0.34,
                 use_dynamic_weights: bool = True,
                 include_industry_adjustment: bool = True,
                 risk_thresholds: Dict[str, float] = None):

        self.environmental_weight = environmental_weight
        self.social_weight = social_weight
        self.governance_weight = governance_weight
        self.use_dynamic_weights = use_dynamic_weights
        self.include_industry_adjustment = include_industry_adjustment

        # 风险评级阈值
        if risk_thresholds is None:
            self.risk_thresholds = {
                'low': 0.8,
                'medium': 0.6,
                'high': 0.4
            }
        else:
            self.risk_thresholds = risk_thresholds

        # 数据标准化器
        self.scalers = {
            'environmental': StandardScaler(),
            'social': StandardScaler(),
            'governance': StandardScaler()
        }

        # 行业基准
        self.industry_benchmarks = {}
        self.sector_benchmarks = {}

        # 动态权重模型
        self.weight_optimizer = None

        # 历史数据存储
        self.historical_scores = {}

        logger.info("ESG Scoring Engine initialized")

    def _define_esg_metrics(self) -> Dict[str, List[str]]:
        """定义ESG指标体系"""
        return {
            'environmental': [
                'carbon_emissions_intensity',
                'renewable_energy_usage',
                'water_usage_efficiency',
                'waste_management_score',
                'biodiversity_impact',
                'environmental_policy_compliance',
                'climate_change_mitigation',
                'pollution_prevention',
                'resource_efficiency',
                'environmental_innovation'
            ],
            'social': [
                'employee_satisfaction_score',
                'diversity_inclusion_index',
                'health_safety_performance',
                'community_investment_ratio',
                'human_rights_policy_score',
                'labor_practices_rating',
                'supply_chain_social_risk',
                'product_safety_quality',
                'data_privacy_protection',
                'stakeholder_engagement'
            ],
            'governance': [
                'board_independence_score',
                'executive_compensation_alignment',
                'shareholder_rights_protection',
                'anti_corruption_measures',
                'ethics_compliance_program',
                'risk_management_effectiveness',
                'transparency_disclosure',
                'auditor_independence',
                'political_contribution_policy',
                'corporate_governance_structure'
            ]
        }

    def calculate_esg_score(self,
                          esg_data: ESGData,
                          industry_peers: List[ESGData] = None,
                          historical_data: List[ESGData] = None) -> ESGScore:
        """计算ESG评分"""
        logger.info(f"Calculating ESG score for {esg_data.company_name}")

        # 数据预处理
        env_metrics = self._preprocess_metrics(esg_data.environmental_metrics, 'environmental')
        social_metrics = self._preprocess_metrics(esg_data.social_metrics, 'social')
        gov_metrics = self._preprocess_metrics(esg_data.governance_metrics, 'governance')

        # 计算各维度得分
        env_score = self._calculate_dimension_score(env_metrics, 'environmental')
        social_score = self._calculate_dimension_score(social_metrics, 'social')
        gov_score = self._calculate_dimension_score(gov_metrics, 'governance')

        # 行业调整
        if self.include_industry_adjustment and industry_peers:
            env_score, social_score, gov_score = self._apply_industry_adjustment(
                env_score, social_score, gov_score, esg_data.industry, industry_peers
            )

        # 动态权重调整
        if self.use_dynamic_weights:
            weights = self._calculate_dynamic_weights(esg_data, industry_peers)
        else:
            weights = {
                'environmental': self.environmental_weight,
                'social': self.social_weight,
                'governance': self.governance_weight
            }

        # 计算综合得分
        overall_score = (
            weights['environmental'] * env_score +
            weights['social'] * social_score +
            weights['governance'] * gov_score
        )

        # 行业排名
        industry_rank, industry_percentile = self._calculate_industry_ranking(
            overall_score, esg_data.industry, industry_peers
        )

        # 风险评级
        risk_rating = self._calculate_risk_rating(overall_score)

        # 趋势分析
        trend_score = self._calculate_trend_score(esg_data, historical_data)

        # 置信度评估
        confidence_level = self._calculate_confidence_level(esg_data, env_metrics, social_metrics, gov_metrics)

        return ESGScore(
            company_id=esg_data.company_id,
            overall_score=overall_score,
            environmental_score=env_score,
            social_score=social_score,
            governance_score=gov_score,
            industry_rank=industry_rank,
            industry_percentile=industry_percentile,
            risk_rating=risk_rating,
            trend_score=trend_score,
            confidence_level=confidence_level
        )

    def _preprocess_metrics(self, metrics: Dict[str, float], dimension: str) -> Dict[str, float]:
        """预处理指标数据"""
        processed_metrics = {}

        # 处理缺失值
        defined_metrics = self._define_esg_metrics()[dimension]
        for metric in defined_metrics:
            if metric in metrics:
                value = metrics[metric]
                # 处理异常值
                if value is not None and not np.isnan(value):
                    # 限制在合理范围内
                    if dimension == 'environmental':
                        # 环境指标：越低越好（如碳排放）或越高越好（如可再生能源使用）
                        if 'emissions' in metric or 'waste' in metric or 'impact' in metric:
                            processed_metrics[metric] = min(max(value, 0), 100)
                        else:
                            processed_metrics[metric] = min(max(value, 0), 100)
                    elif dimension == 'social':
                        # 社会指标：越高越好
                        processed_metrics[metric] = min(max(value, 0), 100)
                    elif dimension == 'governance':
                        # 治理指标：越高越好
                        processed_metrics[metric] = min(max(value, 0), 100)
            else:
                # 使用行业平均值填充缺失值
                processed_metrics[metric] = 50.0  # 默认中等水平

        return processed_metrics

    def _calculate_dimension_score(self, metrics: Dict[str, float], dimension: str) -> float:
        """计算维度得分"""
        if not metrics:
            return 0.0

        # 定义权重
        metric_weights = self._get_metric_weights(dimension)

        # 加权平均
        weighted_sum = 0.0
        total_weight = 0.0

        for metric, value in metrics.items():
            weight = metric_weights.get(metric, 1.0)
            weighted_sum += weight * value
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return min(max(weighted_sum / total_weight, 0), 100)

    def _get_metric_weights(self, dimension: str) -> Dict[str, float]:
        """获取指标权重"""
        weights = {
            'environmental': {
                'carbon_emissions_intensity': 0.15,
                'renewable_energy_usage': 0.12,
                'water_usage_efficiency': 0.10,
                'waste_management_score': 0.10,
                'biodiversity_impact': 0.08,
                'environmental_policy_compliance': 0.10,
                'climate_change_mitigation': 0.12,
                'pollution_prevention': 0.08,
                'resource_efficiency': 0.10,
                'environmental_innovation': 0.05
            },
            'social': {
                'employee_satisfaction_score': 0.12,
                'diversity_inclusion_index': 0.10,
                'health_safety_performance': 0.12,
                'community_investment_ratio': 0.08,
                'human_rights_policy_score': 0.10,
                'labor_practices_rating': 0.10,
                'supply_chain_social_risk': 0.10,
                'product_safety_quality': 0.10,
                'data_privacy_protection': 0.10,
                'stakeholder_engagement': 0.08
            },
            'governance': {
                'board_independence_score': 0.12,
                'executive_compensation_alignment': 0.10,
                'shareholder_rights_protection': 0.10,
                'anti_corruption_measures': 0.12,
                'ethics_compliance_program': 0.10,
                'risk_management_effectiveness': 0.12,
                'transparency_disclosure': 0.10,
                'auditor_independence': 0.08,
                'political_contribution_policy': 0.08,
                'corporate_governance_structure': 0.08
            }
        }

        return weights.get(dimension, {})

    def _apply_industry_adjustment(self,
                                 env_score: float,
                                 social_score: float,
                                 gov_score: float,
                                 industry: str,
                                 industry_peers: List[ESGData]) -> Tuple[float, float, float]:
        """应用行业调整"""
        if not industry_peers:
            return env_score, social_score, gov_score

        # 计算行业平均得分
        peer_scores = []
        for peer in industry_peers:
            peer_env = self._calculate_dimension_score(
                self._preprocess_metrics(peer.environmental_metrics, 'environmental'),
                'environmental'
            )
            peer_social = self._calculate_dimension_score(
                self._preprocess_metrics(peer.social_metrics, 'social'),
                'social'
            )
            peer_gov = self._calculate_dimension_score(
                self._preprocess_metrics(peer.governance_metrics, 'governance'),
                'governance'
            )
            peer_scores.append([peer_env, peer_social, peer_gov])

        peer_scores = np.array(peer_scores)
        industry_means = np.mean(peer_scores, axis=0)
        industry_stds = np.std(peer_scores, axis=0)

        # 行业标准化
        env_adjusted = (env_score - industry_means[0]) / (industry_stds[0] + 1e-8) * 10 + 50
        social_adjusted = (social_score - industry_means[1]) / (industry_stds[1] + 1e-8) * 10 + 50
        gov_adjusted = (gov_score - industry_means[2]) / (industry_stds[2] + 1e-8) * 10 + 50

        # 限制在0-100范围内
        env_adjusted = min(max(env_adjusted, 0), 100)
        social_adjusted = min(max(social_adjusted, 0), 100)
        gov_adjusted = min(max(gov_adjusted, 0), 100)

        return env_adjusted, social_adjusted, gov_adjusted

    def _calculate_dynamic_weights(self,
                                 esg_data: ESGData,
                                 industry_peers: List[ESGData] = None) -> Dict[str, float]:
        """计算动态权重"""
        if not industry_peers:
            return {
                'environmental': self.environmental_weight,
                'social': self.social_weight,
                'governance': self.governance_weight
            }

        # 基于行业特征调整权重
        industry = esg_data.industry.lower()

        # 不同行业的ESG重要性
        industry_weights = {
            'energy': {'environmental': 0.5, 'social': 0.25, 'governance': 0.25},
            'technology': {'environmental': 0.25, 'social': 0.35, 'governance': 0.4},
            'financial': {'environmental': 0.2, 'social': 0.3, 'governance': 0.5},
            'healthcare': {'environmental': 0.25, 'social': 0.45, 'governance': 0.3},
            'manufacturing': {'environmental': 0.4, 'social': 0.35, 'governance': 0.25},
            'retail': {'environmental': 0.3, 'social': 0.4, 'governance': 0.3}
        }

        # 查找匹配的行业权重
        for key in industry_weights:
            if key in industry:
                return industry_weights[key]

        # 默认权重
        return {
            'environmental': self.environmental_weight,
            'social': self.social_weight,
            'governance': self.governance_weight
        }

    def _calculate_industry_ranking(self,
                                  score: float,
                                  industry: str,
                                  industry_peers: List[ESGData] = None) -> Tuple[int, float]:
        """计算行业排名"""
        if not industry_peers:
            return 1, 100.0

        # 计算同行得分
        peer_scores = []
        for peer in industry_peers:
            peer_env = self._calculate_dimension_score(
                self._preprocess_metrics(peer.environmental_metrics, 'environmental'),
                'environmental'
            )
            peer_social = self._calculate_dimension_score(
                self._preprocess_metrics(peer.social_metrics, 'social'),
                'social'
            )
            peer_gov = self._calculate_dimension_score(
                self._preprocess_metrics(peer.governance_metrics, 'governance'),
                'governance'
            )

            weights = self._calculate_dynamic_weights(peer, industry_peers)
            peer_score = (
                weights['environmental'] * peer_env +
                weights['social'] * peer_social +
                weights['governance'] * peer_gov
            )
            peer_scores.append(peer_score)

        # 排名和百分位
        peer_scores = sorted(peer_scores, reverse=True)
        rank = peer_scores.index(score) + 1 if score in peer_scores else len(peer_scores) + 1
        percentile = (len(peer_scores) - rank + 1) / len(peer_scores) * 100

        return rank, percentile

    def _calculate_risk_rating(self, score: float) -> str:
        """计算风险评级"""
        if score >= self.risk_thresholds['low']:
            return 'LOW'
        elif score >= self.risk_thresholds['medium']:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def _calculate_trend_score(self,
                             esg_data: ESGData,
                             historical_data: List[ESGData] = None) -> float:
        """计算趋势得分"""
        if not historical_data or len(historical_data) < 2:
            return 0.0

        # 计算历史得分
        historical_scores = []
        for hist_data in historical_data:
            if hist_data.company_id == esg_data.company_id:
                hist_env = self._calculate_dimension_score(
                    self._preprocess_metrics(hist_data.environmental_metrics, 'environmental'),
                    'environmental'
                )
                hist_social = self._calculate_dimension_score(
                    self._preprocess_metrics(hist_data.social_metrics, 'social'),
                    'social'
                )
                hist_gov = self._calculate_dimension_score(
                    self._preprocess_metrics(hist_data.governance_metrics, 'governance'),
                    'governance'
                )

                weights = self._calculate_dynamic_weights(hist_data)
                hist_score = (
                    weights['environmental'] * hist_env +
                    weights['social'] * hist_social +
                    weights['governance'] * hist_gov
                )
                historical_scores.append(hist_score)

        if len(historical_scores) < 2:
            return 0.0

        # 计算趋势（线性回归斜率）
        x = np.arange(len(historical_scores))
        y = np.array(historical_scores)

        # 简单线性回归
        slope = np.cov(x, y)[0, 1] / np.var(x) if np.var(x) > 0 else 0

        # 标准化趋势得分
        trend_score = np.tanh(slope * 10) * 50 + 50  # 转换到0-100范围

        return min(max(trend_score, 0), 100)

    def _calculate_confidence_level(self,
                                 esg_data: ESGData,
                                 env_metrics: Dict[str, float],
                                 social_metrics: Dict[str, float],
                                 gov_metrics: Dict[str, float]) -> float:
        """计算置信度水平"""
        # 数据完整性评分
        total_metrics = len(self._define_esg_metrics()['environmental']) + \
                       len(self._define_esg_metrics()['social']) + \
                       len(self._define_esg_metrics()['governance'])

        available_metrics = len(env_metrics) + len(social_metrics) + len(gov_metrics)
        completeness_score = available_metrics / total_metrics

        # 数据质量评分
        data_quality_score = esg_data.data_quality_score if esg_data.data_quality_score else 0.5

        # 报告时效性（假设当年报告质量更高）
        reporting_timeliness = 1.0 if esg_data.reporting_year >= 2023 else 0.8

        # 综合置信度
        confidence = (completeness_score * 0.4 + data_quality_score * 0.4 + reporting_timeliness * 0.2)

        return min(max(confidence * 100, 0), 100)

    def batch_analyze_companies(self,
                               companies_data: List[ESGData],
                               include_industry_analysis: bool = True) -> Dict:
        """批量分析公司"""
        logger.info(f"Batch analyzing {len(companies_data)} companies")

        results = {}

        # 按行业分组
        industry_groups = {}
        for company in companies_data:
            if company.industry not in industry_groups:
                industry_groups[company.industry] = []
            industry_groups[company.industry].append(company)

        # 分析每家公司
        for company in companies_data:
            industry_peers = industry_groups.get(company.industry, [])
            industry_peers = [peer for peer in industry_peers if peer.company_id != company.company_id]

            score = self.calculate_esg_score(company, industry_peers)
            results[company.company_id] = score

        # 行业分析
        industry_analysis = {}
        if include_industry_analysis:
            for industry, companies in industry_groups.items():
                industry_scores = [results[comp.company_id].overall_score for comp in companies]
                industry_analysis[industry] = {
                    'average_score': np.mean(industry_scores),
                    'median_score': np.median(industry_scores),
                    'std_score': np.std(industry_scores),
                    'top_performers': sorted(
                        [(comp.company_name, results[comp.company_id].overall_score) for comp in companies],
                        key=lambda x: x[1], reverse=True
                    )[:5]
                }

        return {
            'company_scores': results,
            'industry_analysis': industry_analysis,
            'summary_statistics': {
                'total_companies': len(companies_data),
                'average_score': np.mean([score.overall_score for score in results.values()]),
                'score_distribution': {
                    'low_risk': len([s for s in results.values() if s.risk_rating == 'LOW']),
                    'medium_risk': len([s for s in results.values() if s.risk_rating == 'MEDIUM']),
                    'high_risk': len([s for s in results.values() if s.risk_rating == 'HIGH'])
                }
            }
        }

    def generate_esg_report(self,
                          company_score: ESGScore,
                          company_data: ESGData,
                          benchmark_scores: Dict[str, float] = None) -> Dict:
        """生成ESG报告"""
        logger.info(f"Generating ESG report for {company_data.company_name}")

        # 基础信息
        report = {
            'company_info': {
                'name': company_data.company_name,
                'industry': company_data.industry,
                'sector': company_data.sector,
                'reporting_year': company_data.reporting_year
            },
            'esg_scores': {
                'overall_score': company_score.overall_score,
                'environmental_score': company_score.environmental_score,
                'social_score': company_score.social_score,
                'governance_score': company_score.governance_score,
                'risk_rating': company_score.risk_rating
            },
            'performance_analysis': {
                'industry_rank': company_score.industry_rank,
                'industry_percentile': company_score.industry_percentile,
                'trend_score': company_score.trend_score,
                'confidence_level': company_score.confidence_level
            }
        }

        # 基准对比
        if benchmark_scores:
            report['benchmark_comparison'] = {
                'industry_benchmark': benchmark_scores.get('industry_average', 0),
                'sector_benchmark': benchmark_scores.get('sector_average', 0),
                'global_benchmark': benchmark_scores.get('global_average', 0),
                'performance_vs_industry': company_score.overall_score - benchmark_scores.get('industry_average', 0),
                'performance_vs_sector': company_score.overall_score - benchmark_scores.get('sector_average', 0)
            }

        # 改进建议
        report['recommendations'] = self._generate_recommendations(company_score, company_data)

        # 风险分析
        report['risk_analysis'] = self._analyze_esg_risks(company_score, company_data)

        return report

    def _generate_recommendations(self,
                                score: ESGScore,
                                company_data: ESGData) -> List[Dict]:
        """生成改进建议"""
        recommendations = []

        # 环境改进建议
        if score.environmental_score < 60:
            recommendations.append({
                'category': 'Environmental',
                'priority': 'High' if score.environmental_score < 40 else 'Medium',
                'suggestion': 'Improve carbon emissions reduction targets and increase renewable energy usage',
                'potential_impact': '5-10 point increase in ESG score'
            })

        # 社会改进建议
        if score.social_score < 60:
            recommendations.append({
                'category': 'Social',
                'priority': 'High' if score.social_score < 40 else 'Medium',
                'suggestion': 'Enhance diversity and inclusion programs and strengthen employee engagement',
                'potential_impact': '5-10 point increase in ESG score'
            })

        # 治理改进建议
        if score.governance_score < 60:
            recommendations.append({
                'category': 'Governance',
                'priority': 'High' if score.governance_score < 40 else 'Medium',
                'suggestion': 'Improve board independence and enhance executive compensation alignment',
                'potential_impact': '5-10 point increase in ESG score'
            })

        # 趋势建议
        if score.trend_score < 40:
            recommendations.append({
                'category': 'Overall',
                'priority': 'Medium',
                'suggestion': 'Develop comprehensive ESG improvement strategy with clear timelines and targets',
                'potential_impact': 'Steady improvement in ESG performance over time'
            })

        return recommendations

    def _analyze_esg_risks(self, score: ESGScore, company_data: ESGData) -> Dict:
        """分析ESG风险"""
        risks = {
            'overall_risk_level': score.risk_rating,
            'risk_factors': [],
            'mitigation_suggestions': []
        }

        # 识别关键风险因素
        if score.environmental_score < 40:
            risks['risk_factors'].append({
                'factor': 'Environmental Risk',
                'severity': 'High',
                'description': 'Poor environmental performance may lead to regulatory penalties and reputational damage'
            })

        if score.social_score < 40:
            risks['risk_factors'].append({
                'factor': 'Social Risk',
                'severity': 'High',
                'description': 'Weak social performance may result in employee dissatisfaction and brand damage'
            })

        if score.governance_score < 40:
            risks['risk_factors'].append({
                'factor': 'Governance Risk',
                'severity': 'High',
                'description': 'Poor governance may lead to regulatory scrutiny and investor concerns'
            })

        # 风险缓解建议
        if risks['risk_factors']:
            risks['mitigation_suggestions'] = [
                'Implement comprehensive ESG risk management framework',
                'Regular ESG performance monitoring and reporting',
                'Stakeholder engagement and feedback mechanisms',
                'Integration of ESG factors into corporate strategy'
            ]

        return risks

    def analyze_portfolio_esg(self,
                            portfolio_companies: List[ESGData],
                            portfolio_weights: List[float] = None) -> Dict:
        """分析投资组合ESG表现"""
        logger.info("Analyzing portfolio ESG performance")

        if portfolio_weights is None:
            portfolio_weights = [1.0 / len(portfolio_companies)] * len(portfolio_companies)

        # 计算每家公司的ESG得分
        company_scores = []
        for company in portfolio_companies:
            score = self.calculate_esg_score(company)
            company_scores.append(score)

        # 加权平均
        weights = np.array(portfolio_weights)
        env_scores = [score.environmental_score for score in company_scores]
        social_scores = [score.social_score for score in company_scores]
        gov_scores = [score.governance_score for score in company_scores]
        overall_scores = [score.overall_score for score in company_scores]

        portfolio_env_score = np.average(env_scores, weights=weights)
        portfolio_social_score = np.average(social_scores, weights=weights)
        portfolio_gov_score = np.average(gov_scores, weights=weights)
        portfolio_overall_score = np.average(overall_scores, weights=weights)

        # 风险分析
        high_risk_companies = [
            (portfolio_companies[i].company_name, company_scores[i].risk_rating)
            for i in range(len(portfolio_companies))
            if company_scores[i].risk_rating == 'HIGH'
        ]

        return {
            'portfolio_esg_score': portfolio_overall_score,
            'dimension_scores': {
                'environmental': portfolio_env_score,
                'social': portfolio_social_score,
                'governance': portfolio_gov_score
            },
            'risk_analysis': {
                'high_risk_exposure': len(high_risk_companies),
                'high_risk_companies': high_risk_companies,
                'portfolio_risk_rating': self._calculate_risk_rating(portfolio_overall_score)
            },
            'diversification_analysis': {
                'industry_concentration': self._calculate_industry_concentration(portfolio_companies),
                'esg_score_dispersion': np.std(overall_scores)
            },
            'improvement_potential': {
                'worst_performers': sorted(
                    [(portfolio_companies[i].company_name, overall_scores[i])
                     for i in range(len(portfolio_companies))],
                    key=lambda x: x[1]
                )[:5]
            }
        }

    def _calculate_industry_concentration(self, companies: List[ESGData]) -> Dict:
        """计算行业集中度"""
        industry_counts = {}
        for company in companies:
            industry = company.industry
            industry_counts[industry] = industry_counts.get(industry, 0) + 1

        total = len(companies)
        concentration = {industry: count/total for industry, count in industry_counts.items()}

        return {
            'concentration_ratios': concentration,
            'herfindahl_index': sum([ratio**2 for ratio in concentration.values()])
        }

# 使用示例
def example_usage():
    """使用示例"""
    # 创建ESG数据
    company_data = ESGData(
        company_id="TECH001",
        company_name="TechCorp Inc.",
        industry="Technology",
        sector="Information Technology",
        environmental_metrics={
            'carbon_emissions_intensity': 25.5,
            'renewable_energy_usage': 85.0,
            'water_usage_efficiency': 78.0,
            'waste_management_score': 82.0
        },
        social_metrics={
            'employee_satisfaction_score': 88.0,
            'diversity_inclusion_index': 75.0,
            'health_safety_performance': 92.0,
            'community_investment_ratio': 65.0
        },
        governance_metrics={
            'board_independence_score': 90.0,
            'executive_compensation_alignment': 85.0,
            'shareholder_rights_protection': 88.0,
            'anti_corruption_measures': 95.0
        },
        reporting_year=2024,
        data_quality_score=0.85
    )

    # 创建评分引擎
    engine = ESGScoringEngine()

    # 计算ESG评分
    score = engine.calculate_esg_score(company_data)
    print(f"ESG Score: {score.overall_score:.2f}")
    print(f"Risk Rating: {score.risk_rating}")
    print(f"Industry Rank: {score.industry_rank}")

    # 生成报告
    report = engine.generate_esg_report(score, company_data)
    print(f"Recommendations: {len(report['recommendations'])}")

    return engine, score

if __name__ == "__main__":
    example_usage()