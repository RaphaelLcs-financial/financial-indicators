"""
神经演化金融算法模块

本模块包含基于神经演化算法的先进金融分析系统，结合进化计算
和深度学习技术优化交易策略和投资组合。

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

from .neuroevolution_trader import NeuroevolutionTrader
from .evolutionary_strategy_optimizer import EvolutionaryStrategyOptimizer
from .genetic_algorithm_portfolio import GeneticAlgorithmPortfolio
from .neural_architecture_search import NeuralArchitectureSearch

__all__ = [
    'NeuroevolutionTrader',
    'EvolutionaryStrategyOptimizer',
    'GeneticAlgorithmPortfolio',
    'NeuralArchitectureSearch'
]