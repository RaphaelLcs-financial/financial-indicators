"""
深度确定性策略梯度交易指标
Deep Deterministic Policy Gradient Trader

基于DDPG算法的连续动作空间强化学习交易系统
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple
from collections import deque
import random
import warnings
warnings.filterwarnings('ignore')

class ActorNetwork(nn.Module):
    """Actor网络 - 策略网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # 批归一化
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)

        # 初始化权重
        self.initialize_weights()

    def initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        # 使用tanh激活函数将输出限制在[-1, 1]
        action = torch.tanh(self.fc3(x))
        return action

class CriticNetwork(nn.Module):
    """Critic网络 - 价值网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        # Q1网络
        self.q1_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_fc3 = nn.Linear(hidden_dim, 1)

        # Q2网络（用于双Q学习）
        self.q2_fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_fc3 = nn.Linear(hidden_dim, 1)

        # 批归一化
        self.q1_bn1 = nn.BatchNorm1d(hidden_dim)
        self.q1_bn2 = nn.BatchNorm1d(hidden_dim)
        self.q2_bn1 = nn.BatchNorm1d(hidden_dim)
        self.q2_bn2 = nn.BatchNorm1d(hidden_dim)

        self.initialize_weights()

    def initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播，返回两个Q值"""
        x = torch.cat([state, action], dim=1)

        # Q1
        q1 = F.relu(self.q1_bn1(self.q1_fc1(x)))
        q1 = F.relu(self.q1_bn2(self.q1_fc2(q1)))
        q1 = self.q1_fc3(q1)

        # Q2
        q2 = F.relu(self.q2_bn1(self.q2_fc1(x)))
        q2 = F.relu(self.q2_bn2(self.q2_fc2(q2)))
        q2 = self.q2_fc3(q2)

        return q1, q2

    def q1_forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """仅使用Q1网络"""
        x = torch.cat([state, action], dim=1)
        q1 = F.relu(self.q1_bn1(self.q1_fc1(x)))
        q1 = F.relu(self.q1_bn2(self.q1_fc2(q1)))
        q1 = self.q1_fc3(q1)
        return q1

class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, max_size: int = 100000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """采样批量数据"""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def size(self) -> int:
        """缓冲区大小"""
        return len(self.buffer)

class DDPGTrader:
    """
    DDPG交易系统

    实现深度确定性策略梯度算法，支持连续动作空间
    动作：[-1, 1] 表示持仓比例
    """

    def __init__(self, state_dim: int = 20, action_dim: int = 1,
                 hidden_dim: int = 256, learning_rate: float = 1e-4):
        """
        初始化DDPG交易系统

        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            hidden_dim: 隐藏层维度
            learning_rate: 学习率
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.lr = learning_rate

        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ 使用设备: {self.device}")

        # 创建网络
        self.actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # 创建目标网络
        self.target_actor = ActorNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_critic = CriticNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        # 复制参数到目标网络
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # 超参数
        self.gamma = 0.99  # 折扣因子
        self.tau = 0.005   # 软更新系数
        self.batch_size = 64
        self.memory_size = 100000

        # 经验回放
        self.replay_buffer = ReplayBuffer(self.memory_size)

        # 噪声参数（OU噪声）
        self.ou_noise = OUNoise(action_dim)

        # 训练统计
        self.training_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'rewards': [],
            'episode_lengths': []
        }

        # 训练状态
        self.is_trained = False
        self.total_steps = 0

    def _create_state(self, data: pd.DataFrame, idx: int) -> np.ndarray:
        """创建状态向量"""
        if idx < 20:  # 需要足够的历史数据
            return np.zeros(self.state_dim)

        # 提取价格和成交量数据
        prices = data['close'].iloc[max(0, idx-20):idx+1].values
        volumes = data['volume'].iloc[max(0, idx-20):idx+1].values if 'volume' in data.columns else None

        # 计算特征
        features = []

        # 1. 价格特征
        features.extend([
            prices[-1] / prices[-2] - 1,  # 当前收益率
            prices[-1] / prices[-5] - 1,  # 5日收益率
            prices[-1] / prices[-10] - 1, # 10日收益率
            prices[-1] / prices[-20] - 1, # 20日收益率
        ])

        # 2. 移动平均线特征
        features.extend([
            prices[-1] / np.mean(prices[-5:]) - 1,   # 相对5日均线
            prices[-1] / np.mean(prices[-10:]) - 1,  # 相对10日均线
            prices[-1] / np.mean(prices[-20:]) - 1,  # 相对20日均线
        ])

        # 3. 波动率特征
        returns = np.diff(prices) / prices[:-1]
        features.extend([
            np.std(returns[-5:]),    # 短期波动率
            np.std(returns[-10:]),   # 中期波动率
            np.std(returns[-20:]),   # 长期波动率
        ])

        # 4. 技术指标
        if len(prices) >= 14:
            # RSI
            gains = returns[returns > 0]
            losses = -returns[returns < 0]
            avg_gain = np.mean(gains[-14:]) if len(gains) > 0 else 0
            avg_loss = np.mean(losses[-14:]) if len(losses) > 0 else 0
            if avg_loss > 0:
                rsi = 100 - (100 / (1 + avg_gain / avg_loss))
            else:
                rsi = 100
            features.append(rsi / 100)  # 归一化
        else:
            features.append(0.5)

        # 5. 成交量特征
        if volumes is not None and len(volumes) > 0:
            features.extend([
                volumes[-1] / np.mean(volumes[-5:]) - 1,  # 成交量变化
                np.std(volumes[-10:]) / np.mean(volumes[-10:]) if np.mean(volumes[-10:]) > 0 else 0,  # 成交量波动率
            ])
        else:
            features.extend([0, 0])

        # 6. 位置特征
        if len(prices) >= 20:
            high_20 = np.max(prices[-20:])
            low_20 = np.min(prices[-20:])
            price_position = (prices[-1] - low_20) / (high_20 - low_20)
            features.append(price_position)
        else:
            features.append(0.5)

        # 7. 趋势特征
        if len(returns) >= 10:
            trend = np.polyfit(range(10), returns[-10:], 1)[0]
            features.append(trend)
        else:
            features.append(0)

        # 8. 填充到指定维度
        while len(features) < self.state_dim:
            features.append(0)

        return np.array(features[:self.state_dim], dtype=np.float32)

    def _calculate_reward(self, action: float, next_price: float, current_price: float,
                          transaction_cost: float = 0.001) -> float:
        """计算奖励"""
        # 价格变化收益
        price_return = (next_price - current_price) / current_price

        # 策略收益
        strategy_return = action * price_return

        # 交易成本
        cost = abs(action) * transaction_cost

        # 风险调整收益
        risk_penalty = 0.1 * abs(action) * abs(price_return)

        # 总奖励
        reward = strategy_return - cost - risk_penalty

        return reward

    def train_episode(self, data: pd.DataFrame, start_idx: int = 20) -> Dict[str, Any]:
        """训练一个episode"""
        self.actor.train()
        self.critic.train()

        total_reward = 0
        episode_length = 0
        actor_losses = []
        critic_losses = []

        state = self._create_state(data, start_idx)
        done = False
        current_idx = start_idx

        while not done and current_idx < len(data) - 1:
            # 选择动作
            action = self.select_action(state)

            # 执行动作
            next_state = self._create_state(data, current_idx + 1)
            reward = self._calculate_reward(
                action,
                data['close'].iloc[current_idx + 1],
                data['close'].iloc[current_idx]
            )

            done = current_idx >= len(data) - 2

            # 存储经验
            self.replay_buffer.add(state, action, reward, next_state, done)

            # 训练
            if self.replay_buffer.size() >= self.batch_size:
                actor_loss, critic_loss = self._train_step()
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            # 更新状态
            state = next_state
            total_reward += reward
            episode_length += 1
            current_idx += 1

        # 更新统计
        self.training_stats['rewards'].append(total_reward)
        self.training_stats['episode_lengths'].append(episode_length)
        if actor_losses:
            self.training_stats['actor_losses'].append(np.mean(actor_losses))
        if critic_losses:
            self.training_stats['critic_losses'].append(np.mean(critic_losses))

        self.total_steps += episode_length

        return {
            'total_reward': total_reward,
            'episode_length': episode_length,
            'actor_loss': np.mean(actor_losses) if actor_losses else 0,
            'critic_loss': np.mean(critic_losses) if critic_losses else 0
        }

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> float:
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]

        # 添加探索噪声
        if add_noise:
            action += self.ou_noise.noise()

        # 确保动作在有效范围内
        action = np.clip(action, -1, 1)

        return action[0] if isinstance(action, np.ndarray) else action

    def _train_step(self) -> Tuple[float, float]:
        """训练一步"""
        # 采样经验
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 转换为tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 训练Critic
        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q1, current_q2 = self.critic(states, actions)

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        critic_loss.backward()
        self.critic_optimizer.step()

        # 训练Actor
        self.actor_optimizer.zero_grad()

        actor_actions = self.actor(states)
        actor_loss = -self.critic.q1_forward(states, actor_actions).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 软更新目标网络
        self._soft_update()

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self):
        """软更新目标网络"""
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1 - self.tau) + param.data * self.tau)

    def train(self, data: pd.DataFrame, episodes: int = 100) -> Dict[str, Any]:
        """训练模型"""
        print("🤖 开始DDPG交易系统训练...")

        for episode in range(episodes):
            # 随机起始点
            start_idx = np.random.randint(20, len(data) - 50)
            episode_result = self.train_episode(data, start_idx)

            if episode % 10 == 0:
                print(f"Episode {episode}/{episodes}, "
                      f"Reward: {episode_result['total_reward']:.4f}, "
                      f"Actor Loss: {episode_result['actor_loss']:.4f}, "
                      f"Critic Loss: {episode_result['critic_loss']:.4f}")

        self.is_trained = True

        # 计算训练统计
        stats = {
            'total_episodes': episodes,
            'total_steps': self.total_steps,
            'avg_reward': np.mean(self.training_stats['rewards']),
            'avg_episode_length': np.mean(self.training_stats['episode_lengths']),
            'final_actor_loss': np.mean(self.training_stats['actor_losses'][-10:]) if self.training_stats['actor_losses'] else 0,
            'final_critic_loss': np.mean(self.training_stats['critic_losses'][-10:]) if self.training_stats['critic_losses'] else 0,
            'training_stats': self.training_stats
        }

        print(f"✅ 训练完成！平均奖励: {stats['avg_reward']:.4f}")
        return stats

    def get_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """获取交易信号"""
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train方法")

        self.actor.eval()
        signals = []

        for i in range(len(data)):
            if i < 20:
                signals.append(0)
            else:
                state = self._create_state(data, i)
                action = self.select_action(state, add_noise=False)
                signals.append(action)

        signals_df = pd.DataFrame({
            'signal': signals,
            'position': signals,  # 直接使用动作作为持仓
            'confidence': np.abs(signals)  # 绝对值作为置信度
        }, index=data.index)

        return signals_df

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'hidden_dim': self.hidden_dim,
            'learning_rate': self.lr,
            'device': str(self.device),
            'is_trained': self.is_trained,
            'total_steps': self.total_steps,
            'buffer_size': self.replay_buffer.size(),
            'training_stats': self.training_stats
        }

class OUNoise:
    """OU噪声过程"""
    def __init__(self, action_dim: int, mu: float = 0, theta: float = 0.15, sigma: float = 0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(action_dim) * mu
        self.reset()

    def reset(self):
        """重置噪声状态"""
        self.state = np.ones(self.action_dim) * self.mu

    def noise(self) -> np.ndarray:
        """生成噪声"""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

# 便捷函数
def create_ddpg_trader(state_dim: int = 20, learning_rate: float = 1e-4) -> DDPGTrader:
    """创建DDPG交易器实例"""
    return DDPGTrader(state_dim=state_dim, learning_rate=learning_rate)

def quick_ddpg_training(data: pd.DataFrame, episodes: int = 50) -> Dict[str, Any]:
    """快速DDPG训练"""
    trader = DDPGTrader()

    # 训练模型
    training_stats = trader.train(data, episodes=episodes)

    # 生成信号
    signals = trader.get_trading_signals(data)

    return {
        'training_stats': training_stats,
        'latest_signal': signals['signal'].iloc[-1] if len(signals) > 0 else 0,
        'latest_position': signals['position'].iloc[-1] if len(signals) > 0 else 0,
        'model_info': trader.get_model_info()
    }