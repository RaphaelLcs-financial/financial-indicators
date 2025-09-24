"""
蚁群优化交易指标
Ant Colony Optimization Trading Indicator

基于蚁群算法的市场路径优化和交易策略生成
"""

import numpy as np
import pandas as pd
import random
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class AntColonyOptimizer:
    """
    蚁群优化交易系统

    模拟蚂蚁觅食行为，优化交易路径和时机选择
    基于信息素浓度和启发式信息进行决策
    """

    def __init__(self, n_ants: int = 20, n_iterations: int = 100,
                 evaporation_rate: float = 0.1, alpha: float = 1.0,
                 beta: float = 2.0, q0: float = 0.9):
        """
        初始化蚁群优化器

        Args:
            n_ants: 蚂蚁数量
            n_iterations: 迭代次数
            evaporation_rate: 信息素蒸发率
            alpha: 信息素重要性参数
            beta: 启发式信息重要性参数
            q0: 贪婪选择概率
        """
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.evaporation_rate = evaporation_rate
        self.alpha = alpha
        self.beta = beta
        self.q0 = q0

        # 交易状态定义
        self.actions = ['BUY', 'SELL', 'HOLD']
        self.n_actions = len(self.actions)

        # 信息素矩阵
        self.pheromone_matrix = None

        # 启发式信息矩阵
        self.heuristic_matrix = None

        # 蚂蚁路径历史
        self.ant_paths = []

        # 最优解
        self.best_path = None
        best_fitness = -np.inf

        # 训练统计
        self.training_stats = {
            'best_fitness_history': [],
            'average_fitness_history': [],
            'convergence_history': []
        }

    def _create_feature_space(self, data: pd.DataFrame) -> np.ndarray:
        """创建特征空间"""
        features = []

        # 价格特征
        features.append(data['close'].pct_change().fillna(0))  # 收益率
        features.append(data['close'].rolling(5).mean().pct_change().fillna(0))  # 短期趋势
        features.append(data['close'].rolling(20).mean().pct_change().fillna(0))  # 长期趋势

        # 波动率特征
        returns = data['close'].pct_change().fillna(0)
        features.append(returns.rolling(10).std().fillna(0))  # 短期波动率
        features.append(returns.rolling(30).std().fillna(0))  # 长期波动率

        # 技术指标
        if len(data) >= 14:
            # RSI计算
            gains = returns[returns > 0]
            losses = -returns[returns < 0]
            avg_gain = np.mean(gains[-14:]) if len(gains) > 0 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) > 0 else 0
            if avg_loss > 0:
                rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            else:
                rsi = 100
            features.append(rsi / 100)  # 归一化RSI
        else:
            features.append(0.5)

        # 成交量特征
        if 'volume' in data.columns:
            features.append(data['volume'].pct_change().fillna(0))
            features.append(data['volume'].rolling(20).mean().fillna(0))
        else:
            features.extend([0, 0])

        # 价格位置
        if len(data) >= 20:
            high_20 = data['high'].rolling(20).max()
            low_20 = data['low'].rolling(20).min()
            price_position = (data['close'] - low_20) / (high_20 - low_20)
            features.append(price_position.fillna(0.5))
        else:
            features.append(0.5)

        # 转换为numpy数组
        feature_matrix = np.column_stack(features)

        return feature_matrix

    def _initialize_pheromone_matrix(self, n_states: int):
        """初始化信息素矩阵"""
        # 信息素矩阵：状态到动作的映射
        self.pheromone_matrix = np.ones((n_states, self.n_actions)) / self.n_actions

    def _calculate_heuristic_matrix(self, features: np.ndarray) -> np.ndarray:
        """计算启发式信息矩阵"""
        n_states = len(features)
        heuristic_matrix = np.zeros((n_states, self.n_actions))

        for i in range(n_states):
            state_features = features[i]

            # 基于特征计算每个动作的启发式信息
            for j, action in enumerate(self.actions):
                if action == 'BUY':
                    # 买入启发式：低价格、上升趋势、低RSI
                    heuristic = self._calculate_buy_heuristic(state_features)
                elif action == 'SELL':
                    # 卖出启发式：高价格、下降趋势、高RSI
                    heuristic = self._calculate_sell_heuristic(state_features)
                else:  # HOLD
                    # 持有启发式：中性状态
                    heuristic = self._calculate_hold_heuristic(state_features)

                heuristic_matrix[i, j] = max(heuristic, 0.01)  # 避免零值

        return heuristic_matrix

    def _calculate_buy_heuristic(self, features: np.ndarray) -> float:
        """计算买入启发式信息"""
        # 价格位置（越低越好）
        price_position = features[7] if len(features) > 7 else 0.5
        price_score = 1 - price_position

        # 趋势（上升越好）
        short_trend = features[1] if len(features) > 1 else 0
        long_trend = features[2] if len(features) > 2 else 0
        trend_score = (short_trend + long_trend) / 2 + 0.5

        # RSI（超卖区域更好）
        rsi = features[4] if len(features) > 4 else 0.5
        rsi_score = 1 - rsi

        # 波动率（适中为好）
        volatility = features[3] if len(features) > 3 else 0
        volatility_score = 1 - min(abs(volatility - 0.02), 0.05) / 0.05

        # 综合启发式信息
        heuristic = (price_score * 0.3 + trend_score * 0.3 +
                   rsi_score * 0.2 + volatility_score * 0.2)

        return heuristic

    def _calculate_sell_heuristic(self, features: np.ndarray) -> float:
        """计算卖出启发式信息"""
        # 价格位置（越高越好）
        price_position = features[7] if len(features) > 7 else 0.5
        price_score = price_position

        # 趋势（下降越好）
        short_trend = features[1] if len(features) > 1 else 0
        long_trend = features[2] if len(features) > 2 else 0
        trend_score = 0.5 - (short_trend + long_trend) / 2

        # RSI（超买区域更好）
        rsi = features[4] if len(features) > 4 else 0.5
        rsi_score = rsi

        # 波动率（适中为好）
        volatility = features[3] if len(features) > 3 else 0
        volatility_score = 1 - min(abs(volatility - 0.02), 0.05) / 0.05

        # 综合启发式信息
        heuristic = (price_score * 0.3 + trend_score * 0.3 +
                   rsi_score * 0.2 + volatility_score * 0.2)

        return heuristic

    def _calculate_hold_heuristic(self, features: np.ndarray) -> float:
        """计算持有启发式信息"""
        # 中性状态的启发式信息
        price_position = features[7] if len(features) > 7 else 0.5
        price_score = 1 - abs(price_position - 0.5) * 2

        # 趋势中性
        short_trend = features[1] if len(features) > 1 else 0
        long_trend = features[2] if len(features) > 2 else 0
        trend_score = 1 - abs(short_trend + long_trend)

        # RSI中性
        rsi = features[4] if len(features) > 4 else 0.5
        rsi_score = 1 - abs(rsi - 0.5) * 2

        # 波动率适中
        volatility = features[3] if len(features) > 3 else 0
        volatility_score = 1 - min(abs(volatility - 0.02), 0.05) / 0.05

        # 综合启发式信息
        heuristic = (price_score * 0.25 + trend_score * 0.25 +
                   rsi_score * 0.25 + volatility_score * 0.25)

        return heuristic

    def _select_action(self, state_idx: int) -> int:
        """选择动作（结合贪婪和随机选择）"""
        if random.random() < self.q0:
            # 贪婪选择
            pheromone = self.pheromone_matrix[state_idx] ** self.alpha
            heuristic = self.heuristic_matrix[state_idx] ** self.beta
            probabilities = pheromone * heuristic
            action = np.argmax(probabilities)
        else:
            # 随机选择
            action = random.randint(0, self.n_actions - 1)

        return action

    def _evaluate_path(self, path: List[int], data: pd.DataFrame,
                      start_idx: int) -> float:
        """评估路径适应度"""
        if len(path) == 0:
            return -np.inf

        # 模拟交易
        portfolio_value = 1.0  # 初始组合价值
        position = 0  # 持仓比例
        transaction_cost = 0.001  # 交易成本

        for i, action_idx in enumerate(path):
            if start_idx + i >= len(data):
                break

            action = self.actions[action_idx]
            current_price = data['close'].iloc[start_idx + i]
            next_price = data['close'].iloc[start_idx + i + 1] if start_idx + i + 1 < len(data) else current_price

            # 执行交易动作
            if action == 'BUY' and position < 0.9:
                target_position = 0.9
                position_change = target_position - position
                cost = abs(position_change) * transaction_cost
                position = target_position

            elif action == 'SELL' and position > -0.9:
                target_position = -0.9
                position_change = target_position - position
                cost = abs(position_change) * transaction_cost
                position = target_position

            else:  # HOLD
                cost = 0

            # 计算收益
            price_return = (next_price - current_price) / current_price
            portfolio_return = position * price_return - cost
            portfolio_value *= (1 + portfolio_return)

        # 计算适应度（考虑收益和风险）
        returns = np.diff([1.0] + [portfolio_value])
        if len(returns) > 0:
            sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
            fitness = portfolio_value * (1 + sharpe_ratio * 0.1)
        else:
            fitness = portfolio_value

        return fitness

    def _update_pheromone(self, all_paths: List[Tuple[List[int], float]]):
        """更新信息素"""
        # 信息素蒸发
        self.pheromone_matrix *= (1 - self.evaporation_rate)

        # 信息素增强
        for path, fitness in all_paths:
            if fitness > 0:  # 只对正收益路径增强
                for i, action in enumerate(path):
                    state_idx = min(i, len(self.pheromone_matrix) - 1)
                    self.pheromone_matrix[state_idx, action] += fitness * 0.1

    def optimize_trading_strategy(self, data: pd.DataFrame,
                                 path_length: int = 20) -> Dict[str, Any]:
        """优化交易策略"""
        print("🐜 开始蚁群优化交易策略...")

        # 创建特征空间
        features = self._create_feature_space(data)
        n_states = min(len(features), 100)  # 限制状态空间大小

        # 初始化矩阵
        self._initialize_pheromone_matrix(n_states)
        self.heuristic_matrix = self._calculate_heuristic_matrix(features[:n_states])

        # 选择起始位置
        start_idx = np.random.randint(0, len(data) - path_length - 10)

        best_fitness = -np.inf
        best_path = None

        for iteration in range(self.n_iterations):
            iteration_paths = []
            iteration_fitnesses = []

            # 每只蚂蚁构建路径
            for ant in range(self.n_ants):
                path = []

                # 构建路径
                for step in range(min(path_length, n_states)):
                    state_idx = step
                    action = self._select_action(state_idx)
                    path.append(action)

                # 评估路径
                fitness = self._evaluate_path(path, data, start_idx)
                iteration_paths.append((path, fitness))
                iteration_fitnesses.append(fitness)

                # 更新最优解
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_path = path.copy()

            # 更新信息素
            self._update_pheromone(iteration_paths)

            # 记录统计信息
            avg_fitness = np.mean(iteration_fitnesses)
            self.training_stats['best_fitness_history'].append(best_fitness)
            self.training_stats['average_fitness_history'].append(avg_fitness)

            if iteration % 10 == 0:
                print(f"Iteration {iteration}/{self.n_iterations}, "
                      f"Best Fitness: {best_fitness:.4f}, "
                      f"Avg Fitness: {avg_fitness:.4f}")

        # 保存最优路径
        self.best_path = best_path

        # 计算收敛性
        convergence_rate = self._calculate_convergence_rate()

        print(f"✅ 优化完成！最佳适应度: {best_fitness:.4f}")
        print(f"收敛率: {convergence_rate:.2%}")

        return {
            'best_fitness': best_fitness,
            'best_path': best_path,
            'convergence_rate': convergence_rate,
            'training_stats': self.training_stats,
            'start_index': start_idx
        }

    def _calculate_convergence_rate(self) -> float:
        """计算收敛率"""
        if len(self.training_stats['best_fitness_history']) < 10:
            return 0.0

        recent_fitness = self.training_stats['best_fitness_history'][-10:]
        early_fitness = self.training_stats['best_fitness_history'][:10]

        if np.mean(early_fitness) > 0:
            convergence_rate = (np.mean(recent_fitness) - np.mean(early_fitness)) / np.mean(early_fitness)
        else:
            convergence_rate = 0.0

        return convergence_rate

    def get_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        if self.best_path is None:
            raise ValueError("模型未训练，请先调用optimize_trading_strategy方法")

        signals = []
        path_length = len(self.best_path)

        for i in range(len(data)):
            if i < path_length:
                # 使用初始路径
                action_idx = self.best_path[i] if i < len(self.best_path) else 2  # 默认HOLD
            else:
                # 基于当前状态选择动作
                features = self._create_feature_space(data.iloc[:i+1])
                if len(features) > 0:
                    current_features = features[-1]
                    heuristic_values = [
                        self._calculate_buy_heuristic(current_features),
                        self._calculate_sell_heuristic(current_features),
                        self._calculate_hold_heuristic(current_features)
                    ]

                    # 使用启发式信息选择动作
                    action_idx = np.argmax(heuristic_values)
                else:
                    action_idx = 2  # 默认HOLD

            # 转换为信号
            action = self.actions[action_idx]
            if action == 'BUY':
                signal = 1
            elif action == 'SELL':
                signal = -1
            else:
                signal = 0

            signals.append(signal)

        signals_df = pd.DataFrame({
            'signal': signals,
            'action': [self.actions[min(s+1, 2)] for s in signals],
            'confidence': np.ones(len(signals))  # 蚁群算法的置信度
        }, index=data.index)

        return signals_df

    def analyze_pheromone_distribution(self) -> Dict[str, Any]:
        """分析信息素分布"""
        if self.pheromone_matrix is None:
            return {'error': 'Pheromone matrix not initialized'}

        analysis = {}

        # 每个动作的平均信息素
        avg_pheromone = np.mean(self.pheromone_matrix, axis=0)
        analysis['average_pheromone'] = dict(zip(self.actions, avg_pheromone))

        # 信息素分布方差
        pheromone_variance = np.var(self.pheromone_matrix, axis=0)
        analysis['pheromone_variance'] = dict(zip(self.actions, pheromone_variance))

        # 最优动作分布
        optimal_actions = np.argmax(self.pheromone_matrix, axis=1)
        action_counts = np.bincount(optimal_actions, minlength=self.n_actions)
        analysis['optimal_action_distribution'] = dict(zip(self.actions, action_counts / len(optimal_actions)))

        # 收敛程度
        convergence_score = np.max(avg_pheromone) / (np.sum(avg_pheromone) + 1e-10)
        analysis['convergence_score'] = convergence_score

        return analysis

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'n_ants': self.n_ants,
            'n_iterations': self.n_iterations,
            'evaporation_rate': self.evaporation_rate,
            'alpha': self.alpha,
            'beta': self.beta,
            'q0': self.q0,
            'actions': self.actions,
            'best_path_length': len(self.best_path) if self.best_path else 0,
            'training_stats': self.training_stats,
            'model_type': 'Ant Colony Optimization Trader'
        }

# 便捷函数
def create_ant_colony_optimizer(n_ants: int = 20, n_iterations: int = 100) -> AntColonyOptimizer:
    """创建蚁群优化器实例"""
    return AntColonyOptimizer(n_ants, n_iterations)

def quick_ant_colony_analysis(data: pd.DataFrame, path_length: int = 20) -> Dict[str, Any]:
    """快速蚁群优化分析"""
    optimizer = AntColonyOptimizer()

    # 优化策略
    optimization_result = optimizer.optimize_trading_strategy(data, path_length)

    # 生成信号
    signals = optimizer.get_trading_signals(data)

    # 分析信息素分布
    pheromone_analysis = optimizer.analyze_pheromone_distribution()

    return {
        'optimization_result': optimization_result,
        'latest_signal': signals['signal'].iloc[-1] if len(signals) > 0 else 0,
        'latest_action': signals['action'].iloc[-1] if len(signals) > 0 else 'HOLD',
        'pheromone_analysis': pheromone_analysis,
        'model_info': optimizer.get_model_info()
    }