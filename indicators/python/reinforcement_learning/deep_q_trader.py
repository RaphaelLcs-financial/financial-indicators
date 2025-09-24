"""
深度Q学习交易系统

基于深度Q网络(DQN)的自适应交易系统，结合经验回放和目标网络，能够学习
最优交易策略并在不同市场环境下自适应调整。

算法特点:
- Double DQN架构减少Q值过高估计
- 优先经验回放提高学习效率
- 噪声网络增强探索能力
- 多时间尺度状态表示
- 风险调整奖励函数

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from typing import Dict, List, Tuple, Optional, Union
from collections import deque, namedtuple
import random
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 定义经验回放元组
Experience = namedtuple('Experience',
    ['state', 'action', 'reward', 'next_state', 'done', 'risk_metrics'])

class DQNetwork(nn.Module):
    """深度Q网络"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(DQNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, action_dim))

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        return self.network(x)

class NoisyDQNetwork(nn.Module):
    """噪声深度Q网络，增强探索能力"""

    def __init__(self, state_dim: int, action_dim: int, hidden_dims: List[int] = None):
        super(NoisyDQNetwork, self).__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256, 128, 64]

        # 噪声层
        self.noisy_layers = []
        prev_dim = state_dim

        for i, hidden_dim in enumerate(hidden_dims):
            if i == len(hidden_dims) - 1:
                # 最后一层使用Factorized Noisy Linear
                linear = NoisyLinear(prev_dim, action_dim)
            else:
                linear = NoisyLinear(prev_dim, hidden_dim)

            self.noisy_layers.append(linear)
            prev_dim = hidden_dim if i < len(hidden_dims) - 1 else action_dim

        self.layers = nn.ModuleList(self.noisy_layers)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation(layer(x))
            x = self.dropout(x)

        return self.layers[-1](x)

    def reset_noise(self):
        """重置噪声"""
        for layer in self.layers:
            layer.reset_noise()

class NoisyLinear(nn.Module):
    """因子化噪声线性层"""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.4):
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """重置参数"""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)

        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))

    def reset_noise(self):
        """重置噪声"""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon = torch.FloatTensor(epsilon_out.ger(epsilon_in))
        self.bias_epsilon = torch.FloatTensor(epsilon_out)

    def _scale_noise(self, size: int) -> torch.Tensor:
        """缩放噪声"""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        self.epsilon = 1e-6

        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.position = 0
        self.max_priority = 1.0

    def add(self, experience: Experience):
        """添加经验"""
        max_priority = self.max_priority

        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience

        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, np.ndarray]:
        """采样经验"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]

        # 计算采样概率
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        # 采样索引
        indices = np.random.choice(len(probabilities), batch_size, p=probabilities)

        # 计算重要性采样权重
        self.beta = min(1.0, self.beta + self.beta_increment)
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()

        # 获取经验
        experiences = [self.buffer[idx] for idx in indices]

        return experiences, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """更新优先级"""
        for idx, td_error in zip(indices, td_errors):
            priority = abs(td_error) + self.epsilon
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)

class DeepQTrader:
    """深度Q学习交易系统"""

    def __init__(self,
                 state_dim: int = 50,
                 action_dim: int = 3,
                 learning_rate: float = 0.0001,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 memory_size: int = 10000,
                 batch_size: int = 64,
                 target_update_freq: int = 100,
                 use_double_dqn: bool = True,
                 use_noisy_net: bool = True,
                 risk_aversion: float = 0.5,
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.use_double_dqn = use_double_dqn
        self.use_noisy_net = use_noisy_net
        self.risk_aversion = risk_aversion
        self.device = device

        # 动作映射
        self.action_mapping = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}

        # 创建网络
        if use_noisy_net:
            self.q_network = NoisyDQNetwork(state_dim, action_dim).to(device)
            self.target_network = NoisyDQNetwork(state_dim, action_dim).to(device)
        else:
            self.q_network = DQNetwork(state_dim, action_dim).to(device)
            self.target_network = DQNetwork(state_dim, action_dim).to(device)

        # 复制目标网络
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # 优化器
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

        # 经验回放缓冲区
        self.replay_buffer = PrioritizedReplayBuffer(memory_size)

        # 训练统计
        self.training_step = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.loss_history = []

        # 性能指标
        self.trade_history = []
        self.portfolio_values = []
        self.sharpe_ratios = []
        self.max_drawdowns = []

        logger.info(f"Deep Q Trader initialized with device: {device}")

    def _preprocess_state(self, state_data: pd.DataFrame) -> torch.Tensor:
        """预处理状态数据"""
        # 提取特征
        features = []

        # 价格相关特征
        if 'close' in state_data.columns:
            returns = state_data['close'].pct_change().fillna(0)
            features.extend([
                returns.values[-5:],  # 最近5日收益率
                returns.rolling(20).mean().values[-1],  # 20日平均收益
                returns.rolling(20).std().values[-1],  # 20日波动率
            ])

        # 技术指标
        if 'volume' in state_data.columns:
            volume_ma = state_data['volume'].rolling(20).mean()
            volume_ratio = state_data['volume'] / volume_ma
            features.append(volume_ratio.values[-1])

        # 市场微观结构
        if 'high' in state_data.columns and 'low' in state_data.columns:
            high_low_range = (state_data['high'] - state_data['low']) / state_data['close']
            features.append(high_low_range.values[-1])

        # 填充或截断到固定维度
        features = np.array(features).flatten()
        if len(features) < self.state_dim:
            features = np.pad(features, (0, self.state_dim - len(features)))
        elif len(features) > self.state_dim:
            features = features[:self.state_dim]

        # 标准化
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features.reshape(-1, 1)).flatten()

        return torch.FloatTensor(features_normalized).to(self.device)

    def _calculate_reward(self,
                         current_state: pd.DataFrame,
                         action: int,
                         next_state: pd.DataFrame,
                         portfolio_value: float,
                         risk_metrics: Dict) -> float:
        """计算风险调整奖励"""
        # 基础奖励：价格变化
        if 'close' in current_state.columns and 'close' in next_state.columns:
            price_change = (next_state['close'].iloc[-1] - current_state['close'].iloc[-1]) / current_state['close'].iloc[-1]
        else:
            price_change = 0

        # 动作奖励
        action_reward = 0
        if action == 2 and price_change > 0:  # 买入且价格上涨
            action_reward = abs(price_change)
        elif action == 0 and price_change < 0:  # 卖出且价格下跌
            action_reward = abs(price_change)
        elif action == 1:  # 持有
            action_reward = -0.001  # 小额惩罚，鼓励交易

        # 风险惩罚
        risk_penalty = 0
        if 'volatility' in risk_metrics:
            risk_penalty = self.risk_aversion * risk_metrics['volatility']

        # 组合价值变化
        portfolio_change = 0
        if hasattr(self, 'last_portfolio_value'):
            portfolio_change = (portfolio_value - self.last_portfolio_value) / self.last_portfolio_value
        self.last_portfolio_value = portfolio_value

        # 总奖励
        total_reward = action_reward + portfolio_change - risk_penalty

        return total_reward

    def select_action(self, state: torch.Tensor, training: bool = True) -> int:
        """选择动作"""
        if training and not self.use_noisy_net:
            # ε-greedy探索
            if random.random() < self.epsilon:
                return random.randint(0, self.action_dim - 1)

        with torch.no_grad():
            if self.use_noisy_net:
                self.q_network.reset_noise()

            q_values = self.q_network(state.unsqueeze(0))
            return q_values.argmax().item()

    def train_step(self, experiences: List[Experience]) -> float:
        """训练一步"""
        # 准备批量数据
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)

        # 当前Q值
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))

        # 下一状态Q值
        if self.use_double_dqn:
            # Double DQN：使用主网络选择动作，目标网络评估
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions).squeeze(1)
        else:
            # 标准DQN
            next_q_values = self.target_network(next_states).max(1)[0]

        # 目标Q值
        target_q_values = rewards + (self.gamma * next_q_values * ~dones)

        # 计算损失
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_network.load_state_dict(self.q_network.state_dict())

    def train_episode(self,
                     training_data: pd.DataFrame,
                     initial_portfolio: float = 10000.0,
                     max_steps: int = 1000) -> Dict:
        """训练一个回合"""
        state = self._preprocess_state(training_data.iloc[:50])
        done = False
        step = 0
        total_reward = 0
        portfolio_value = initial_portfolio
        self.last_portfolio_value = initial_portfolio

        episode_experiences = []

        while not done and step < max_steps and step < len(training_data) - 50:
            # 选择动作
            action = self.select_action(state, training=True)

            # 执行动作
            next_state_data = training_data.iloc[step+1:step+51]
            next_state = self._preprocess_state(next_state_data)

            # 计算奖励
            risk_metrics = self._calculate_risk_metrics(next_state_data)
            reward = self._calculate_reward(
                training_data.iloc[step:step+50],
                action,
                next_state_data,
                portfolio_value,
                risk_metrics
            )

            # 更新组合价值
            portfolio_value += reward * portfolio_value * 0.1  # 控制交易规模

            # 检查终止条件
            done = (step >= max_steps - 1) or (portfolio_value < initial_portfolio * 0.5)

            # 存储经验
            experience = Experience(state, action, reward, next_state, done, risk_metrics)
            episode_experiences.append(experience)

            # 添加到经验回放
            self.replay_buffer.add(experience)

            # 训练
            if len(self.replay_buffer.buffer) >= self.batch_size:
                experiences, indices, weights = self.replay_buffer.sample(self.batch_size)

                # 训练步
                loss = self.train_step(experiences)
                self.loss_history.append(loss)

                # 更新优先级
                if self.use_double_dqn:
                    td_errors = self._calculate_td_errors(experiences)
                    self.replay_buffer.update_priorities(indices, td_errors)

            # 更新状态
            state = next_state
            total_reward += reward
            step += 1
            self.training_step += 1

            # 定期更新目标网络
            if self.training_step % self.target_update_freq == 0:
                self.update_target_network()

            # 衰减探索率
            if not self.use_noisy_net:
                self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

        # 记录回合统计
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(step)

        # 计算性能指标
        sharpe_ratio = self._calculate_sharpe_ratio(episode_experiences)
        max_drawdown = self._calculate_max_drawdown(episode_experiences)

        self.sharpe_ratios.append(sharpe_ratio)
        self.max_drawdowns.append(max_drawdown)

        return {
            'episode_reward': total_reward,
            'episode_length': step,
            'final_portfolio': portfolio_value,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'epsilon': self.epsilon
        }

    def _calculate_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """计算风险指标"""
        risk_metrics = {}

        if 'close' in data.columns:
            returns = data['close'].pct_change().fillna(0)
            risk_metrics['volatility'] = returns.std()
            risk_metrics['var_95'] = returns.quantile(0.05)
            risk_metrics['var_99'] = returns.quantile(0.01)

        if 'volume' in data.columns:
            risk_metrics['volume_volatility'] = data['volume'].pct_change().std()

        return risk_metrics

    def _calculate_td_errors(self, experiences: List[Experience]) -> np.ndarray:
        """计算TD误差"""
        states = torch.stack([exp.state for exp in experiences]).to(self.device)
        actions = torch.LongTensor([exp.action for exp in experiences]).to(self.device)
        rewards = torch.FloatTensor([exp.reward for exp in experiences]).to(self.device)
        next_states = torch.stack([exp.next_state for exp in experiences]).to(self.device)
        dones = torch.BoolTensor([exp.done for exp in experiences]).to(self.device)

        with torch.no_grad():
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            td_errors = (current_q_values.squeeze() - target_q_values).cpu().numpy()

        return td_errors

    def _calculate_sharpe_ratio(self, experiences: List[Experience]) -> float:
        """计算夏普比率"""
        if len(experiences) < 2:
            return 0.0

        rewards = [exp.reward for exp in experiences]
        returns = np.array(rewards)

        if returns.std() == 0:
            return 0.0

        return np.sqrt(252) * returns.mean() / returns.std()

    def _calculate_max_drawdown(self, experiences: List[Experience]) -> float:
        """计算最大回撤"""
        if len(experiences) < 2:
            return 0.0

        portfolio_values = [10000.0]  # 初始组合价值
        for exp in experiences:
            portfolio_values.append(portfolio_values[-1] * (1 + exp.reward * 0.1))

        portfolio_values = np.array(portfolio_values)
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - peak) / peak

        return np.min(drawdown)

    def predict(self, market_data: pd.DataFrame) -> Dict:
        """预测交易信号"""
        self.q_network.eval()

        with torch.no_grad():
            state = self._preprocess_state(market_data.iloc[-50:])

            if self.use_noisy_net:
                self.q_network.reset_noise()

            q_values = self.q_network(state.unsqueeze(0))
            action = q_values.argmax().item()

            # 计算动作概率
            probabilities = F.softmax(q_values, dim=1).squeeze().cpu().numpy()

            # 计算置信度
            confidence = probabilities[action]

            # 风险评估
            risk_metrics = self._calculate_risk_metrics(market_data)

            return {
                'action': self.action_mapping[action],
                'action_id': action,
                'confidence': confidence,
                'q_values': q_values.squeeze().cpu().numpy(),
                'probabilities': probabilities,
                'risk_metrics': risk_metrics,
                'model_confidence': self._calculate_model_confidence()
            }

    def _calculate_model_confidence(self) -> float:
        """计算模型置信度"""
        if not self.loss_history:
            return 0.5

        # 基于最近的损失计算置信度
        recent_losses = self.loss_history[-100:]
        avg_loss = np.mean(recent_losses)

        # 损失越低，置信度越高
        confidence = max(0.1, min(0.9, 1.0 - avg_loss))

        return confidence

    def get_training_statistics(self) -> Dict:
        """获取训练统计信息"""
        return {
            'total_training_steps': self.training_step,
            'total_episodes': len(self.episode_rewards),
            'average_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'average_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
            'current_epsilon': self.epsilon,
            'average_loss': np.mean(self.loss_history[-100:]) if self.loss_history else 0,
            'average_sharpe_ratio': np.mean(self.sharpe_ratios[-10:]) if self.sharpe_ratios else 0,
            'average_max_drawdown': np.mean(self.max_drawdowns[-10:]) if self.max_drawdowns else 0,
            'best_portfolio_value': max(self.portfolio_values) if self.portfolio_values else 0
        }

    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
            'loss_history': self.loss_history
        }, filepath)
        logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint['training_step']
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.loss_history = checkpoint['loss_history']
        logger.info(f"Model loaded from {filepath}")

    def analyze(self, market_data: pd.DataFrame) -> Dict:
        """完整分析"""
        logger.info("Starting Deep Q Trading analysis...")

        # 预测
        prediction = self.predict(market_data)

        # 获取统计信息
        stats = self.get_training_statistics()

        # 生成交易建议
        trading_recommendation = self._generate_trading_recommendation(prediction, stats)

        return {
            'prediction': prediction,
            'statistics': stats,
            'trading_recommendation': trading_recommendation,
            'model_info': {
                'model_type': 'Deep Q Network',
                'use_double_dqn': self.use_double_dqn,
                'use_noisy_net': self.use_noisy_net,
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'risk_aversion': self.risk_aversion
            }
        }

    def _generate_trading_recommendation(self, prediction: Dict, stats: Dict) -> Dict:
        """生成交易建议"""
        action = prediction['action']
        confidence = prediction['confidence']
        model_confidence = prediction['model_confidence']

        # 风险评估
        risk_level = 'LOW'
        if 'risk_metrics' in prediction:
            volatility = prediction['risk_metrics'].get('volatility', 0)
            if volatility > 0.02:
                risk_level = 'HIGH'
            elif volatility > 0.01:
                risk_level = 'MEDIUM'

        # 交易建议
        if action == 'BUY' and confidence > 0.6 and model_confidence > 0.6:
            recommendation = 'STRONG_BUY'
            reason = f"Model predicts upward movement with high confidence ({confidence:.2f})"
        elif action == 'SELL' and confidence > 0.6 and model_confidence > 0.6:
            recommendation = 'STRONG_SELL'
            reason = f"Model predicts downward movement with high confidence ({confidence:.2f})"
        elif action == 'HOLD':
            recommendation = 'HOLD'
            reason = "Model suggests waiting for better opportunity"
        else:
            recommendation = 'WEAK_' + action
            reason = f"Model prediction has low confidence ({confidence:.2f})"

        return {
            'recommendation': recommendation,
            'action': action,
            'confidence': confidence,
            'model_confidence': model_confidence,
            'risk_level': risk_level,
            'reason': reason,
            'position_size': self._calculate_position_size(confidence, risk_level),
            'stop_loss': self._calculate_stop_loss(prediction, risk_level),
            'take_profit': self._calculate_take_profit(prediction, risk_level)
        }

    def _calculate_position_size(self, confidence: float, risk_level: str) -> float:
        """计算持仓规模"""
        base_size = 0.1  # 基础10%仓位

        if confidence > 0.8:
            size_multiplier = 1.5
        elif confidence > 0.6:
            size_multiplier = 1.0
        else:
            size_multiplier = 0.5

        if risk_level == 'HIGH':
            size_multiplier *= 0.5
        elif risk_level == 'MEDIUM':
            size_multiplier *= 0.8

        return min(base_size * size_multiplier, 0.3)  # 最大30%仓位

    def _calculate_stop_loss(self, prediction: Dict, risk_level: str) -> float:
        """计算止损位"""
        if risk_level == 'HIGH':
            return 0.02  # 2%止损
        elif risk_level == 'MEDIUM':
            return 0.015  # 1.5%止损
        else:
            return 0.01  # 1%止损

    def _calculate_take_profit(self, prediction: Dict, risk_level: str) -> float:
        """计算止盈位"""
        confidence = prediction['confidence']

        if confidence > 0.8:
            return 0.04  # 4%止盈
        elif confidence > 0.6:
            return 0.03  # 3%止盈
        else:
            return 0.02  # 2%止盈

# 使用示例
def example_usage():
    """使用示例"""
    # 创建示例数据
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    np.random.seed(42)

    # 生成模拟市场数据
    prices = 100 + np.cumsum(np.random.randn(1000) * 0.02)
    volumes = 1000000 + np.random.randn(1000) * 100000

    market_data = pd.DataFrame({
        'date': dates,
        'open': prices * (1 + np.random.randn(1000) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(1000)) * 0.02),
        'low': prices * (1 - np.abs(np.random.randn(1000)) * 0.02),
        'close': prices,
        'volume': volumes
    })

    # 初始化交易器
    trader = DeepQTrader(
        state_dim=20,
        action_dim=3,
        learning_rate=0.0001,
        use_double_dqn=True,
        use_noisy_net=True
    )

    # 训练
    print("开始训练...")
    for episode in range(10):
        stats = trader.train_episode(market_data)
        print(f"Episode {episode + 1}: Reward={stats['episode_reward']:.2f}, "
              f"Portfolio={stats['final_portfolio']:.2f}")

    # 预测
    prediction = trader.predict(market_data)
    print(f"预测动作: {prediction['action']}, 置信度: {prediction['confidence']:.2f}")

    # 完整分析
    analysis = trader.analyze(market_data)
    print(f"交易建议: {analysis['trading_recommendation']['recommendation']}")

    return trader, analysis

if __name__ == "__main__":
    example_usage()