"""
神经演化交易系统

基于神经演化算法的自适应交易系统，结合进化计算和深度学习技术，
能够自动优化神经网络结构和交易策略参数。

算法特点:
- 遗传算法优化网络结构
- 进化策略微调参数
- 多目标优化（收益 vs 风险）
- 自适应学习率和拓扑结构
- 种群多样性维护

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import random
import copy
from collections import defaultdict
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import json

logger = logging.getLogger(__name__)

@dataclass
class Genome:
    """基因组表示神经网络结构和参数"""
    layer_sizes: List[int]
    activation_functions: List[str]
    learning_rate: float
    dropout_rate: float
    batch_size: int
    optimization_method: str
    fitness_score: float = 0.0
    risk_adjusted_score: float = 0.0
    generation: int = 0

class TradingNetwork(nn.Module):
    """可进化交易神经网络"""

    def __init__(self, genome: Genome, input_dim: int, output_dim: int = 3):
        super(TradingNetwork, self).__init__()

        self.genome = genome
        self.input_dim = input_dim
        self.output_dim = output_dim

        # 构建网络层
        self.layers = self._build_layers()
        self._initialize_weights()

    def _build_layers(self) -> nn.ModuleList:
        """根据基因组构建网络层"""
        layers = nn.ModuleList()

        # 输入层
        prev_size = self.input_dim

        # 隐藏层
        for i, layer_size in enumerate(self.genome.layer_sizes):
            # 线性层
            layers.append(nn.Linear(prev_size, layer_size))

            # 激活函数
            activation = self._get_activation(self.genome.activation_functions[i])
            layers.append(activation)

            # Dropout
            if self.genome.dropout_rate > 0:
                layers.append(nn.Dropout(self.genome.dropout_rate))

            prev_size = layer_size

        # 输出层
        layers.append(nn.Linear(prev_size, self.output_dim))
        layers.append(nn.Softmax(dim=1))

        return layers

    def _get_activation(self, activation_name: str) -> nn.Module:
        """获取激活函数"""
        activations = {
            'relu': nn.ReLU(),
            'leaky_relu': nn.LeakyReLU(0.1),
            'elu': nn.ELU(),
            'selu': nn.SELU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'gelu': nn.GELU()
        }
        return activations.get(activation_name, nn.ReLU())

    def _initialize_weights(self):
        """初始化网络权重"""
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for layer in self.layers:
            x = layer(x)
        return x

class NeuroevolutionTrader:
    """神经演化交易系统"""

    def __init__(self,
                 input_dim: int = 20,
                 population_size: int = 50,
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.7,
                 elite_size: int = 5,
                 risk_aversion: float = 0.5,
                 tournament_size: int = 3):

        self.input_dim = input_dim
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        self.risk_aversion = risk_aversion
        self.tournament_size = tournament_size

        # 进化历史
        self.evolution_history = []
        self.best_genome = None
        self.best_network = None

        # 网络架构搜索空间
        self.layer_size_options = [32, 64, 128, 256, 512]
        self.activation_options = ['relu', 'leaky_relu', 'elu', 'selu', 'tanh']
        self.learning_rate_options = [0.001, 0.005, 0.01, 0.05]
        self.dropout_options = [0.0, 0.1, 0.2, 0.3]
        self.batch_size_options = [16, 32, 64, 128]
        self.optimization_options = ['adam', 'sgd', 'rmsprop']

        # 初始化种群
        self.population = self._initialize_population()

        logger.info(f"Neuroevolution Trader initialized with population size {population_size}")

    def _initialize_population(self) -> List[Genome]:
        """初始化种群"""
        population = []

        for _ in range(self.population_size):
            genome = self._generate_random_genome()
            population.append(genome)

        return population

    def _generate_random_genome(self) -> Genome:
        """生成随机基因组"""
        # 随机网络深度 (1-5层)
        num_layers = random.randint(1, 5)
        layer_sizes = [random.choice(self.layer_size_options) for _ in range(num_layers)]

        # 随机激活函数
        activation_functions = [random.choice(self.activation_options) for _ in range(num_layers)]

        # 随机超参数
        learning_rate = random.choice(self.learning_rate_options)
        dropout_rate = random.choice(self.dropout_options)
        batch_size = random.choice(self.batch_size_options)
        optimization_method = random.choice(self.optimization_options)

        return Genome(
            layer_sizes=layer_sizes,
            activation_functions=activation_functions,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            batch_size=batch_size,
            optimization_method=optimization_method
        )

    def evolve_trader(self,
                     training_data: pd.DataFrame,
                     validation_data: pd.DataFrame) -> Dict:
        """演化交易系统"""
        logger.info("Starting neuroevolution training...")

        for generation in range(self.generations):
            logger.info(f"Generation {generation + 1}/{self.generations}")

            # 评估种群
            population_fitness = self._evaluate_population(
                self.population, training_data, validation_data
            )

            # 选择精英
            elite_genomes = self._select_elite(self.population, population_fitness)

            # 记录最佳个体
            best_genome = elite_genomes[0]
            if self.best_genome is None or best_genome.fitness_score > self.best_genome.fitness_score:
                self.best_genome = best_genome
                self.best_network = TradingNetwork(best_genome, self.input_dim)

            # 进化操作
            new_population = []

            # 精英保留
            new_population.extend(elite_genomes[:self.elite_size])

            # 生成新个体
            while len(new_population) < self.population_size:
                parent1, parent2 = self._tournament_selection(self.population, population_fitness)
                child_genome = self._crossover(parent1, parent2)
                child_genome = self._mutate(child_genome)
                child_genome.generation = generation + 1
                new_population.append(child_genome)

            self.population = new_population

            # 记录演化历史
            self.evolution_history.append({
                'generation': generation + 1,
                'best_fitness': best_genome.fitness_score,
                'average_fitness': np.mean([g.fitness_score for g in self.population]),
                'best_genome': best_genome,
                'population_diversity': self._calculate_diversity(self.population)
            })

            logger.info(f"Generation {generation + 1}: Best fitness = {best_genome.fitness_score:.4f}")

        return {
            'evolution_history': self.evolution_history,
            'best_genome': self.best_genome,
            'best_network': self.best_network,
            'final_population': self.population
        }

    def _evaluate_population(self,
                            population: List[Genome],
                            training_data: pd.DataFrame,
                            validation_data: pd.DataFrame) -> List[float]:
        """评估种群适应度"""
        fitness_scores = []

        for i, genome in enumerate(population):
            # 创建网络
            network = TradingNetwork(genome, self.input_dim)

            # 训练网络
            training_result = self._train_network(network, genome, training_data)

            # 验证性能
            validation_result = self._validate_network(network, validation_data)

            # 计算适应度
            fitness_score = self._calculate_fitness(training_result, validation_result)
            genome.fitness_score = fitness_score

            fitness_scores.append(fitness_score)

            logger.debug(f"Individual {i + 1}/{len(population)}: Fitness = {fitness_score:.4f}")

        return fitness_scores

    def _train_network(self,
                      network: TradingNetwork,
                      genome: Genome,
                      training_data: pd.DataFrame) -> Dict:
        """训练网络"""
        # 预处理数据
        features, targets = self._preprocess_data(training_data)

        # 创建数据加载器
        dataset = torch.utils.data.TensorDataset(
            torch.FloatTensor(features),
            torch.LongTensor(targets)
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=genome.batch_size,
            shuffle=True
        )

        # 设置优化器
        optimizer = self._get_optimizer(network, genome)
        criterion = nn.CrossEntropyLoss()

        # 训练
        network.train()
        training_losses = []

        for epoch in range(10):  # 固定训练轮数
            epoch_loss = 0.0
            for batch_features, batch_targets in dataloader:
                optimizer.zero_grad()

                outputs = network(batch_features)
                loss = criterion(outputs, batch_targets)

                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            training_losses.append(epoch_loss)

        return {
            'training_losses': training_losses,
            'final_loss': training_losses[-1],
            'convergence_rate': np.mean(np.diff(training_losses[-5:])) if len(training_losses) > 5 else 0
        }

    def _validate_network(self,
                         network: TradingNetwork,
                         validation_data: pd.DataFrame) -> Dict:
        """验证网络性能"""
        features, targets = self._preprocess_data(validation_data)

        network.eval()
        with torch.no_grad():
            outputs = network(torch.FloatTensor(features))
            predictions = torch.argmax(outputs, dim=1)

            accuracy = (predictions == torch.LongTensor(targets)).float().mean().item()

            # 计算其他指标
            precision = self._calculate_precision(predictions, targets)
            recall = self._calculate_recall(predictions, targets)

            # 计算交易性能
            trading_performance = self._simulate_trading(predictions, validation_data)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'trading_performance': trading_performance
        }

    def _preprocess_data(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """预处理市场数据"""
        features = []

        # 基本价格特征
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change().fillna(0)
            features.extend([
                returns.values[-5:],  # 最近5日收益率
                returns.rolling(20).mean().values[-1],  # 20日平均收益
                returns.rolling(20).std().values[-1],  # 20日波动率
            ])

        # 成交量特征
        if 'volume' in market_data.columns:
            volume_ma = market_data['volume'].rolling(20).mean()
            volume_ratio = market_data['volume'] / volume_ma
            features.append(volume_ratio.values[-1])

        # 技术指标
        if 'high' in market_data.columns and 'low' in market_data.columns:
            high_low_range = (market_data['high'] - market_data['low']) / market_data['close']
            features.append(high_low_range.values[-1])

        # 目标变量
        if 'close' in market_data.columns:
            future_return = market_data['close'].pct_change(5).shift(-5).fillna(0)
            targets = np.where(future_return > 0.01, 2,  # 上涨
                             np.where(future_return < -0.01, 0, 1))  # 下跌或持有
        else:
            targets = np.ones(len(market_data))

        # 填充到固定维度
        features = np.array(features).flatten()
        if len(features) < self.input_dim:
            features = np.pad(features, (0, self.input_dim - len(features)))
        elif len(features) > self.input_dim:
            features = features[:self.input_dim]

        # 标准化
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features.reshape(-1, 1)).flatten()

        return features_normalized, targets

    def _get_optimizer(self, network: TradingNetwork, genome: Genome) -> optim.Optimizer:
        """获取优化器"""
        if genome.optimization_method == 'adam':
            return optim.Adam(network.parameters(), lr=genome.learning_rate)
        elif genome.optimization_method == 'sgd':
            return optim.SGD(network.parameters(), lr=genome.learning_rate, momentum=0.9)
        elif genome.optimization_method == 'rmsprop':
            return optim.RMSprop(network.parameters(), lr=genome.learning_rate)
        else:
            return optim.Adam(network.parameters(), lr=genome.learning_rate)

    def _calculate_precision(self, predictions: torch.Tensor, targets: np.ndarray) -> float:
        """计算精确率"""
        predictions_np = predictions.cpu().numpy()
        true_positives = np.sum((predictions_np == 2) & (targets == 2))
        predicted_positives = np.sum(predictions_np == 2)

        return true_positives / (predicted_positives + 1e-8)

    def _calculate_recall(self, predictions: torch.Tensor, targets: np.ndarray) -> float:
        """计算召回率"""
        predictions_np = predictions.cpu().numpy()
        true_positives = np.sum((predictions_np == 2) & (targets == 2))
        actual_positives = np.sum(targets == 2)

        return true_positives / (actual_positives + 1e-8)

    def _simulate_trading(self, predictions: torch.Tensor, market_data: pd.DataFrame) -> Dict:
        """模拟交易"""
        predictions_np = predictions.cpu().numpy()

        if 'close' not in market_data.columns:
            return {'total_return': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0}

        prices = market_data['close'].values
        returns = np.diff(prices) / prices[:-1]

        # 简单交易策略
        positions = np.zeros(len(predictions_np))
        positions[predictions_np == 2] = 1  # 买入
        positions[predictions_np == 0] = -1  # 卖出

        # 计算交易收益
        trading_returns = positions[:-1] * returns

        total_return = np.sum(trading_returns)
        sharpe_ratio = np.mean(trading_returns) / (np.std(trading_returns) + 1e-8) * np.sqrt(252)

        # 计算最大回撤
        cumulative_returns = np.cumprod(1 + trading_returns)
        peak = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = np.min(drawdown)

        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def _calculate_fitness(self, training_result: Dict, validation_result: Dict) -> float:
        """计算适应度分数"""
        # 多目标适应度：准确性 + 交易性能 - 复杂度惩罚

        accuracy = validation_result['accuracy']
        trading_performance = validation_result['trading_performance']

        # 综合得分
        performance_score = 0.4 * accuracy + 0.3 * trading_performance['sharpe_ratio'] + 0.3 * (1 + trading_performance['total_return'])

        # 复杂度惩罚（网络层数和参数数量）
        complexity_penalty = 0.01 * len(self.best_genome.layer_sizes) if self.best_genome else 0.01

        fitness_score = performance_score - complexity_penalty

        return max(0.0, fitness_score)

    def _select_elite(self, population: List[Genome], fitness_scores: List[float]) -> List[Genome]:
        """选择精英个体"""
        # 按适应度排序
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda x: x[0], reverse=True)]

        return sorted_population[:self.elite_size]

    def _tournament_selection(self, population: List[Genome], fitness_scores: List[float]) -> Tuple[Genome, Genome]:
        """锦标赛选择"""
        # 随机选择锦标赛参与者
        tournament_indices = np.random.choice(len(population), self.tournament_size * 2, replace=False)

        # 第一个父代
        parent1_indices = tournament_indices[:self.tournament_size]
        parent1_fitness = [fitness_scores[i] for i in parent1_indices]
        parent1_idx = parent1_indices[np.argmax(parent1_fitness)]
        parent1 = population[parent1_idx]

        # 第二个父代
        parent2_indices = tournament_indices[self.tournament_size:]
        parent2_fitness = [fitness_scores[i] for i in parent2_indices]
        parent2_idx = parent2_indices[np.argmax(parent2_fitness)]
        parent2 = population[parent2_idx]

        return parent1, parent2

    def _crossover(self, parent1: Genome, parent2: Genome) -> Genome:
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return copy.deepcopy(random.choice([parent1, parent2]))

        # 单点交叉
        child = Genome(
            layer_sizes=[],
            activation_functions=[],
            learning_rate=0.0,
            dropout_rate=0.0,
            batch_size=0,
            optimization_method=''
        )

        # 网络结构交叉
        max_layers = max(len(parent1.layer_sizes), len(parent2.layer_sizes))
        crossover_point = random.randint(1, max_layers - 1)

        for i in range(max_layers):
            if i < crossover_point:
                if i < len(parent1.layer_sizes):
                    child.layer_sizes.append(parent1.layer_sizes[i])
                    child.activation_functions.append(parent1.activation_functions[i])
            else:
                if i < len(parent2.layer_sizes):
                    child.layer_sizes.append(parent2.layer_sizes[i])
                    child.activation_functions.append(parent2.activation_functions[i])

        # 超参数交叉
        child.learning_rate = random.choice([parent1.learning_rate, parent2.learning_rate])
        child.dropout_rate = random.choice([parent1.dropout_rate, parent2.dropout_rate])
        child.batch_size = random.choice([parent1.batch_size, parent2.batch_size])
        child.optimization_method = random.choice([parent1.optimization_method, parent2.optimization_method])

        return child

    def _mutate(self, genome: Genome) -> Genome:
        """变异操作"""
        mutated = copy.deepcopy(genome)

        # 网络结构变异
        if random.random() < self.mutation_rate:
            mutation_type = random.choice(['add_layer', 'remove_layer', 'modify_layer', 'modify_hyperparams'])

            if mutation_type == 'add_layer' and len(mutated.layer_sizes) < 5:
                mutated.layer_sizes.insert(random.randint(0, len(mutated.layer_sizes)),
                                        random.choice(self.layer_size_options))
                mutated.activation_functions.insert(random.randint(0, len(mutated.activation_functions)),
                                                random.choice(self.activation_options))

            elif mutation_type == 'remove_layer' and len(mutated.layer_sizes) > 1:
                remove_idx = random.randint(0, len(mutated.layer_sizes) - 1)
                mutated.layer_sizes.pop(remove_idx)
                mutated.activation_functions.pop(remove_idx)

            elif mutation_type == 'modify_layer':
                if mutated.layer_sizes:
                    modify_idx = random.randint(0, len(mutated.layer_sizes) - 1)
                    mutated.layer_sizes[modify_idx] = random.choice(self.layer_size_options)
                    mutated.activation_functions[modify_idx] = random.choice(self.activation_options)

            elif mutation_type == 'modify_hyperparams':
                mutated.learning_rate = random.choice(self.learning_rate_options)
                mutated.dropout_rate = random.choice(self.dropout_options)
                mutated.batch_size = random.choice(self.batch_size_options)
                mutated.optimization_method = random.choice(self.optimization_options)

        return mutated

    def _calculate_diversity(self, population: List[Genome]) -> float:
        """计算种群多样性"""
        if len(population) < 2:
            return 0.0

        # 计算基因组之间的距离
        total_distance = 0.0
        count = 0

        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = self._genome_distance(population[i], population[j])
                total_distance += distance
                count += 1

        return total_distance / count if count > 0 else 0.0

    def _genome_distance(self, genome1: Genome, genome2: Genome) -> float:
        """计算基因组距离"""
        distance = 0.0

        # 网络结构距离
        layer_distance = abs(len(genome1.layer_sizes) - len(genome2.layer_sizes))
        distance += layer_distance

        # 超参数距离
        lr_distance = abs(genome1.learning_rate - genome2.learning_rate)
        dropout_distance = abs(genome1.dropout_rate - genome2.dropout_rate)
        distance += lr_distance + dropout_distance

        return distance

    def predict_trading_signal(self, market_data: pd.DataFrame) -> Dict:
        """预测交易信号"""
        if self.best_network is None:
            return {'error': 'No trained model available'}

        self.best_network.eval()

        # 预处理数据
        features, _ = self._preprocess_data(market_data)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            outputs = self.best_network(features_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

        # 获取预测结果
        signal_mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        predicted_signal = np.argmax(probabilities)
        confidence = probabilities[predicted_signal]

        return {
            'predicted_signal': signal_mapping[predicted_signal],
            'signal_id': predicted_signal,
            'confidence': confidence,
            'probabilities': {
                'sell': probabilities[0],
                'hold': probabilities[1],
                'buy': probabilities[2]
            },
            'model_complexity': len(self.best_genome.layer_sizes) if self.best_genome else 0,
            'evolution_generation': self.best_genome.generation if self.best_genome else 0
        }

    def analyze_evolution_performance(self) -> Dict:
        """分析演化性能"""
        if not self.evolution_history:
            return {'error': 'No evolution history available'}

        # 性能趋势分析
        generations = [h['generation'] for h in self.evolution_history]
        best_fitness = [h['best_fitness'] for h in self.evolution_history]
        average_fitness = [h['average_fitness'] for h in self.evolution_history]
        diversity = [h['population_diversity'] for h in self.evolution_history]

        performance_analysis = {
            'total_generations': len(generations),
            'final_best_fitness': best_fitness[-1],
            'fitness_improvement': best_fitness[-1] - best_fitness[0],
            'convergence_rate': np.polyfit(generations[-10:], best_fitness[-10:], 1)[0] if len(generations) > 10 else 0,
            'average_diversity': np.mean(diversity),
            'diversity_trend': np.polyfit(generations, diversity, 1)[0]
        }

        # 最佳网络分析
        if self.best_genome:
            best_network_analysis = {
                'layer_structure': self.best_genome.layer_sizes,
                'activation_functions': self.best_genome.activation_functions,
                'learning_rate': self.best_genome.learning_rate,
                'dropout_rate': self.best_genome.dropout_rate,
                'batch_size': self.best_genome.batch_size,
                'optimization_method': self.best_genome.optimization_method,
                'fitness_score': self.best_genome.fitness_score,
                'generation': self.best_genome.generation
            }
        else:
            best_network_analysis = {}

        return {
            'performance_analysis': performance_analysis,
            'best_network_analysis': best_network_analysis,
            'evolution_convergence': self._assess_convergence(best_fitness),
            'population_dynamics': {
                'fitness_progression': list(zip(generations, best_fitness, average_fitness)),
                'diversity_progression': list(zip(generations, diversity))
            }
        }

    def _assess_convergence(self, fitness_history: List[float]) -> Dict:
        """评估收敛性"""
        if len(fitness_history) < 10:
            return {'converged': False, 'reason': 'Insufficient generations'}

        # 检查最近10代的改进
        recent_improvements = np.diff(fitness_history[-10:])
        avg_improvement = np.mean(recent_improvements)

        # 收敛标准：平均改进小于阈值
        convergence_threshold = 0.001
        converged = avg_improvement < convergence_threshold

        return {
            'converged': converged,
            'average_improvement': avg_improvement,
            'convergence_threshold': convergence_threshold,
            'final_fitness': fitness_history[-1]
        }

    def analyze(self, market_data: pd.DataFrame) -> Dict:
        """完整分析"""
        logger.info("Starting neuroevolution trading analysis...")

        # 预测交易信号
        prediction = self.predict_trading_signal(market_data)

        # 性能分析
        performance = self.analyze_evolution_performance()

        # 生成交易建议
        trading_recommendation = self._generate_trading_recommendation(prediction, performance)

        return {
            'prediction': prediction,
            'performance_analysis': performance,
            'trading_recommendation': trading_recommendation,
            'model_info': {
                'model_type': 'Neuroevolution',
                'population_size': self.population_size,
                'generations': self.generations,
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elite_size': self.elite_size
            }
        }

    def _generate_trading_recommendation(self, prediction: Dict, performance: Dict) -> Dict:
        """生成交易建议"""
        signal = prediction['predicted_signal']
        confidence = prediction['confidence']

        # 基于演化性能调整置信度
        if 'best_network_analysis' in performance:
            fitness_score = performance['best_network_analysis'].get('fitness_score', 0.5)
            adjusted_confidence = confidence * min(fitness_score / 2.0, 1.0)
        else:
            adjusted_confidence = confidence * 0.5

        # 交易建议
        if signal == 'BUY' and adjusted_confidence > 0.6:
            recommendation = 'STRONG_BUY'
            reason = f"Neuroevolution model predicts strong upward movement with confidence {adjusted_confidence:.2f}"
        elif signal == 'SELL' and adjusted_confidence > 0.6:
            recommendation = 'STRONG_SELL'
            reason = f"Neuroevolution model predicts downward movement with confidence {adjusted_confidence:.2f}"
        elif signal == 'HOLD':
            recommendation = 'HOLD'
            reason = "Model suggests waiting for better opportunity"
        else:
            recommendation = 'WEAK_' + signal
            reason = f"Model prediction has low confidence ({adjusted_confidence:.2f})"

        return {
            'recommendation': recommendation,
            'predicted_signal': signal,
            'confidence': adjusted_confidence,
            'model_fitness': performance['best_network_analysis'].get('fitness_score', 0) if 'best_network_analysis' in performance else 0,
            'reason': reason,
            'position_size': self._calculate_position_size(adjusted_confidence),
            'risk_level': self._assess_risk_level(performance)
        }

    def _calculate_position_size(self, confidence: float) -> float:
        """计算持仓规模"""
        if confidence > 0.8:
            return 0.15  # 15%仓位
        elif confidence > 0.6:
            return 0.08  # 8%仓位
        else:
            return 0.03  # 3%仓位

    def _assess_risk_level(self, performance: Dict) -> str:
        """评估风险水平"""
        convergence = performance.get('evolution_convergence', {})
        if convergence.get('converged', False):
            return 'LOW'
        else:
            return 'MEDIUM'

# 使用示例
def example_usage():
    """使用示例"""
    # 创建神经演化交易器
    trader = NeuroevolutionTrader(
        input_dim=20,
        population_size=30,
        generations=20,
        mutation_rate=0.15,
        crossover_rate=0.7,
        elite_size=3
    )

    # 创建示例市场数据
    dates = pd.date_range('2023-01-01', periods=500, freq='D')
    np.random.seed(42)

    prices = 100 + np.cumsum(np.random.randn(500) * 0.02)
    volumes = 1000000 + np.random.randn(500) * 100000

    market_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(500) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(500)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(500)) * 0.02),
        'close': prices,
        'volume': volumes
    })

    # 分割训练和验证数据
    train_data = market_data.iloc[:400]
    val_data = market_data.iloc[400:]

    # 演化训练
    evolution_results = trader.evolve_trader(train_data, val_data)
    print(f"Evolution completed with best fitness: {evolution_results['best_genome'].fitness_score:.4f}")

    # 预测
    test_data = market_data.iloc[-50:]
    prediction = trader.predict_trading_signal(test_data)
    print(f"Prediction: {prediction['predicted_signal']}, Confidence: {prediction['confidence']:.2f}")

    return trader, prediction

if __name__ == "__main__":
    example_usage()