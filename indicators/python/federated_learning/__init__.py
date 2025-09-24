"""
联邦学习金融指标模块

本模块包含基于联邦学习的先进金融分析系统，支持多机构协作建模、
隐私保护数据分析和分布式训练。

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

from .federated_market_predictor import FederatedMarketPredictor
from .privacy_preserving_portfolio import PrivacyPreservingPortfolio
from .distributed_risk_analyzer import DistributedRiskAnalyzer
from .cross_institutional_anomaly import CrossInstitutionalAnomaly

__all__ = [
    'FederatedMarketPredictor',
    'PrivacyPreservingPortfolio',
    'DistributedRiskAnalyzer',
    'CrossInstitutionalAnomaly'
]