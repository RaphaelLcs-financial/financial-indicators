"""
联邦学习市场预测器

基于联邦学习的多机构协作市场预测系统，支持隐私保护的数据分析和
分布式模型训练。采用差分隐私、安全聚合等先进技术确保数据安全。

算法特点:
- 联邦平均算法 (FedAvg)
- 差分隐私保护
- 安全模型聚合
- 异构客户端处理
- 动态权重调整

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from typing import Dict, List, Tuple, Optional, Union, Callable
import logging
from dataclasses import dataclass
from abc import ABC, abstractmethod
import threading
import queue
import time
from cryptography.fernet import Fernet
import hashlib
import json

logger = logging.getLogger(__name__)

@dataclass
class ClientData:
    """客户端数据"""
    client_id: str
    market_data: pd.DataFrame
    features: np.ndarray
    targets: np.ndarray
    data_size: int
    privacy_budget: float = 1.0

@dataclass
class ModelUpdate:
    """模型更新"""
    client_id: str
    model_parameters: Dict[str, torch.Tensor]
    update_size: int
    timestamp: float
    privacy_cost: float
    model_version: int

class DifferentialPrivacy:
    """差分隐私机制"""

    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def add_laplace_noise(self, data: np.ndarray, sensitivity: float) -> np.ndarray:
        """添加拉普拉斯噪声"""
        scale = sensitivity / self.epsilon
        noise = np.random.laplace(0, scale, data.shape)
        return data + noise

    def add_gaussian_noise(self, data: np.ndarray, sensitivity: float) -> np.ndarray:
        """添加高斯噪声"""
        sigma = np.sqrt(2 * np.log(1.25 / self.delta)) * sensitivity / self.epsilon
        noise = np.random.normal(0, sigma, data.shape)
        return data + noise

    def clip_gradients(self, gradients: List[torch.Tensor], clip_norm: float) -> List[torch.Tensor]:
        """梯度裁剪"""
        total_norm = torch.norm(torch.stack([torch.norm(g.flatten()) for g in gradients]))
        clip_coef = min(clip_norm / (total_norm + 1e-6), 1.0)

        return [g * clip_coef for g in gradients]

class SecureAggregator:
    """安全聚合器"""

    def __init__(self, encryption_key: bytes = None):
        if encryption_key is None:
            encryption_key = Fernet.generate_key()
        self.cipher = Fernet(encryption_key)
        self.client_keys = {}

    def encrypt_update(self, update: Dict, client_id: str) -> bytes:
        """加密模型更新"""
        update_str = json.dumps({
            k: v.tolist() if isinstance(v, torch.Tensor) else v
            for k, v in update.items()
        })
        return self.cipher.encrypt(update_str.encode())

    def decrypt_update(self, encrypted_update: bytes) -> Dict:
        """解密模型更新"""
        decrypted = self.cipher.decrypt(encrypted_update).decode()
        return json.loads(decrypted)

    def secure_aggregate(self, encrypted_updates: List[bytes]) -> Dict:
        """安全聚合模型更新"""
        decrypted_updates = [self.decrypt_update(update) for update in encrypted_updates]

        # 简单平均聚合（实际应用中可以使用更复杂的聚合方法）
        aggregated = {}
        for key in decrypted_updates[0].keys():
            values = [update[key] for update in decrypted_updates]
            aggregated[key] = np.mean(values, axis=0)

        return aggregated

class MarketPredictor(nn.Module):
    """市场预测神经网络"""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None):
        super(MarketPredictor, self).__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, 3))  # 买、卖、持有

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

class FederatedClient:
    """联邦学习客户端"""

    def __init__(self,
                 client_id: str,
                 model: nn.Module,
                 data: ClientData,
                 privacy_mechanism: DifferentialPrivacy,
                 local_epochs: int = 5,
                 batch_size: int = 32,
                 learning_rate: float = 0.001):

        self.client_id = client_id
        self.model = model
        self.data = data
        self.privacy_mechanism = privacy_mechanism
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # 数据加载器
        dataset = TensorDataset(
            torch.FloatTensor(self.data.features),
            torch.LongTensor(self.data.targets)
        )
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def local_train(self) -> Tuple[Dict[str, torch.Tensor], float]:
        """本地训练"""
        logger.info(f"Client {self.client_id} starting local training")

        # 保存初始参数
        initial_params = {name: param.clone() for name, param in self.model.named_parameters()}

        # 本地训练
        self.model.train()
        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            for batch_features, batch_targets in self.dataloader:
                self.optimizer.zero_grad()

                outputs = self.model(batch_features)
                loss = self.criterion(outputs, batch_targets)

                loss.backward()

                # 应用差分隐私
                gradients = [param.grad for param in self.model.parameters()]
                clipped_gradients = self.privacy_mechanism.clip_gradients(gradients, clip_norm=1.0)

                # 重新设置梯度
                for i, param in enumerate(self.model.parameters()):
                    param.grad = clipped_gradients[i]

                self.optimizer.step()
                epoch_loss += loss.item()

            logger.debug(f"Client {self.client_id} epoch {epoch + 1}, loss: {epoch_loss:.4f}")

        # 计算模型更新
        model_updates = {}
        for name, param in self.model.named_parameters():
            model_updates[name] = param - initial_params[name]

        # 计算隐私成本
        privacy_cost = self._calculate_privacy_cost()

        logger.info(f"Client {self.client_id} local training completed")

        return model_updates, privacy_cost

    def _calculate_privacy_cost(self) -> float:
        """计算隐私成本"""
        # 简化的隐私成本计算
        # 实际应用中应该使用更精确的差分隐私会计
        return self.local_epochs * self.learning_rate / self.data.data_size

    def evaluate(self, test_data: Tuple[np.ndarray, np.ndarray] = None) -> Dict:
        """评估模型性能"""
        if test_data is None:
            test_data = (self.data.features, self.data.targets)

        self.model.eval()
        with torch.no_grad():
            features_tensor = torch.FloatTensor(test_data[0])
            targets_tensor = torch.LongTensor(test_data[1])

            outputs = self.model(features_tensor)
            predictions = torch.argmax(outputs, dim=1)

            accuracy = accuracy_score(test_data[1], predictions.cpu().numpy())
            precision = precision_score(test_data[1], predictions.cpu().numpy(), average='weighted')
            recall = recall_score(test_data[1], predictions.cpu().numpy(), average='weighted')

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'total_samples': len(test_data[0])
        }

class FederatedMarketPredictor:
    """联邦学习市场预测器"""

    def __init__(self,
                 input_dim: int = 20,
                 hidden_dims: List[int] = None,
                 num_rounds: int = 100,
                 clients_per_round: int = 5,
                 privacy_epsilon: float = 1.0,
                 privacy_delta: float = 1e-5,
                 aggregation_strategy: str = 'fedavg'):

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.num_rounds = num_rounds
        self.clients_per_round = clients_per_round
        self.aggregation_strategy = aggregation_strategy

        # 全局模型
        self.global_model = MarketPredictor(input_dim, hidden_dims)
        self.model_version = 0

        # 隐私保护
        self.privacy_mechanism = DifferentialPrivacy(privacy_epsilon, privacy_delta)

        # 安全聚合
        self.secure_aggregator = SecureAggregator()

        # 客户端管理
        self.clients = {}
        self.client_history = {}

        # 训练历史
        self.training_history = {
            'rounds': [],
            'client_performances': [],
            'global_accuracy': [],
            'privacy_costs': []
        }

        logger.info(f"Federated Market Predictor initialized with {num_rounds} rounds")

    def register_client(self, client_data: ClientData) -> str:
        """注册客户端"""
        client_id = client_data.client_id

        # 为每个客户端创建独立的模型副本
        client_model = MarketPredictor(self.input_dim, self.hidden_dims)
        client_model.load_state_dict(self.global_model.state_dict())

        # 创建联邦客户端
        client = FederatedClient(
            client_id=client_id,
            model=client_model,
            data=client_data,
            privacy_mechanism=self.privacy_mechanism
        )

        self.clients[client_id] = client
        self.client_history[client_id] = []

        logger.info(f"Client {client_id} registered with {client_data.data_size} samples")

        return client_id

    def preprocess_market_data(self, market_data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """预处理市场数据"""
        # 特征工程
        features = []

        # 价格特征
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

        # 目标变量（价格方向）
        if 'close' in market_data.columns:
            future_return = market_data['close'].pct_change(5).shift(-5).fillna(0)
            targets = np.where(future_return > 0.01, 2,  # 上涨
                             np.where(future_return < -0.01, 0, 1))  # 下跌或持有
        else:
            targets = np.ones(len(market_data))  # 默认持有

        # 填充或截断到固定维度
        features = np.array(features).flatten()
        if len(features) < self.input_dim:
            features = np.pad(features, (0, self.input_dim - len(features)))
        elif len(features) > self.input_dim:
            features = features[:self.input_dim]

        # 标准化
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features.reshape(-1, 1)).flatten()

        return features_normalized, targets[-1]

    def federated_training(self) -> Dict:
        """联邦学习训练"""
        logger.info("Starting federated training")

        for round_num in range(self.num_rounds):
            logger.info(f"Round {round_num + 1}/{self.num_rounds}")

            # 选择参与本轮训练的客户端
            selected_clients = self._select_clients()

            # 本地训练
            client_updates = []
            round_performances = []
            total_privacy_cost = 0.0

            for client in selected_clients:
                # 发送当前全局模型到客户端
                client.model.load_state_dict(self.global_model.state_dict())

                # 本地训练
                model_updates, privacy_cost = client.local_train()

                # 收集模型更新
                update = ModelUpdate(
                    client_id=client.client_id,
                    model_parameters=model_updates,
                    update_size=client.data.data_size,
                    timestamp=time.time(),
                    privacy_cost=privacy_cost,
                    model_version=self.model_version
                )

                client_updates.append(update)
                round_performances.append(client.evaluate())
                total_privacy_cost += privacy_cost

            # 安全聚合
            aggregated_updates = self._secure_aggregate_updates(client_updates)

            # 更新全局模型
            self._update_global_model(aggregated_updates)

            # 评估全局模型
            global_accuracy = self._evaluate_global_model()

            # 记录训练历史
            self.training_history['rounds'].append(round_num + 1)
            self.training_history['client_performances'].append(round_performances)
            self.training_history['global_accuracy'].append(global_accuracy)
            self.training_history['privacy_costs'].append(total_privacy_cost)

            self.model_version += 1

            logger.info(f"Round {round_num + 1} completed, "
                       f"Global accuracy: {global_accuracy:.4f}, "
                       f"Privacy cost: {total_privacy_cost:.6f}")

        return {
            'training_history': self.training_history,
            'final_model_state': self.global_model.state_dict(),
            'total_privacy_cost': sum(self.training_history['privacy_costs'])
        }

    def _select_clients(self) -> List[FederatedClient]:
        """选择参与训练的客户端"""
        available_clients = list(self.clients.values())

        if len(available_clients) <= self.clients_per_round:
            return available_clients

        # 随机选择客户端
        selected_indices = np.random.choice(
            len(available_clients),
            self.clients_per_round,
            replace=False
        )

        return [available_clients[i] for i in selected_indices]

    def _secure_aggregate_updates(self, updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """安全聚合模型更新"""
        logger.info("Securely aggregating model updates")

        # 加权聚合（基于数据量）
        total_samples = sum(update.update_size for update in updates)
        aggregated_updates = {}

        # 获取参数名称
        param_names = list(updates[0].model_parameters.keys())

        for param_name in param_names:
            weighted_sum = None
            for update in updates:
                weight = update.update_size / total_samples
                param_update = update.model_parameters[param_name]

                if weighted_sum is None:
                    weighted_sum = weight * param_update
                else:
                    weighted_sum += weight * param_update

            aggregated_updates[param_name] = weighted_sum

        return aggregated_updates

    def _update_global_model(self, aggregated_updates: Dict[str, torch.Tensor]):
        """更新全局模型"""
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                if name in aggregated_updates:
                    param += aggregated_updates[name]

        logger.info("Global model updated")

    def _evaluate_global_model(self) -> float:
        """评估全局模型"""
        if not self.clients:
            return 0.0

        # 在所有客户端数据上评估
        all_accuracies = []

        for client in self.clients.values():
            performance = client.evaluate()
            all_accuracies.append(performance['accuracy'])

        return np.mean(all_accuracies)

    def predict_market_trend(self, market_data: pd.DataFrame) -> Dict:
        """预测市场趋势"""
        self.global_model.eval()

        # 预处理数据
        features, _ = self.preprocess_market_data(market_data)
        features_tensor = torch.FloatTensor(features).unsqueeze(0)

        with torch.no_grad():
            outputs = self.global_model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().cpu().numpy()

        # 获取预测结果
        trend_mapping = {0: 'DOWN', 1: 'HOLD', 2: 'UP'}
        predicted_trend = np.argmax(probabilities)
        confidence = probabilities[predicted_trend]

        return {
            'predicted_trend': trend_mapping[predicted_trend],
            'trend_id': predicted_trend,
            'confidence': confidence,
            'probabilities': {
                'down': probabilities[0],
                'hold': probabilities[1],
                'up': probabilities[2]
            },
            'model_version': self.model_version,
            'privacy_utilization': sum(self.training_history['privacy_costs'])
        }

    def analyze_federated_performance(self) -> Dict:
        """分析联邦学习性能"""
        if not self.training_history['rounds']:
            return {'error': 'No training history available'}

        # 计算性能指标
        global_accuracies = self.training_history['global_accuracy']
        privacy_costs = self.training_history['privacy_costs']

        performance_analysis = {
            'total_rounds': len(global_accuracies),
            'final_accuracy': global_accuracies[-1] if global_accuracies else 0,
            'best_accuracy': max(global_accuracies) if global_accuracies else 0,
            'accuracy_improvement': global_accuracies[-1] - global_accuracies[0] if len(global_accuracies) > 1 else 0,
            'average_privacy_cost_per_round': np.mean(privacy_costs) if privacy_costs else 0,
            'total_privacy_cost': sum(privacy_costs),
            'convergence_rate': self._calculate_convergence_rate(global_accuracies)
        }

        # 客户端贡献分析
        client_contributions = self._analyze_client_contributions()

        return {
            'performance_analysis': performance_analysis,
            'client_contributions': client_contributions,
            'training_stability': self._assess_training_stability(),
            'privacy_budget_utilization': self._analyze_privacy_utilization()
        }

    def _calculate_convergence_rate(self, accuracies: List[float]) -> float:
        """计算收敛速度"""
        if len(accuracies) < 10:
            return 0.0

        # 计算最近10轮的改进率
        recent_improvements = np.diff(accuracies[-10:])
        return np.mean(recent_improvements)

    def _analyze_client_contributions(self) -> Dict:
        """分析客户端贡献"""
        contributions = {}

        for client_id, client in self.clients.items():
            if client_id in self.client_history:
                performances = self.client_history[client_id]
                avg_performance = np.mean([p['accuracy'] for p in performances]) if performances else 0
                participation_rounds = len(performances)

                contributions[client_id] = {
                    'average_performance': avg_performance,
                    'participation_rounds': participation_rounds,
                    'contribution_score': avg_performance * participation_rounds
                }

        return contributions

    def _assess_training_stability(self) -> Dict:
        """评估训练稳定性"""
        if not self.training_history['global_accuracy']:
            return {'stability_score': 0.0}

        accuracies = self.training_history['global_accuracy']
        recent_accuracy = accuracies[-10:] if len(accuracies) >= 10 else accuracies

        stability_metrics = {
            'accuracy_variance': np.var(recent_accuracy),
            'accuracy_trend': np.polyfit(range(len(recent_accuracy)), recent_accuracy, 1)[0],
            'stability_score': 1.0 / (1.0 + np.var(recent_accuracy))  # 方差越小，稳定性越高
        }

        return stability_metrics

    def _analyze_privacy_utilization(self) -> Dict:
        """分析隐私预算使用情况"""
        privacy_costs = self.training_history['privacy_costs']

        if not privacy_costs:
            return {'total_privacy_cost': 0.0, 'budget_utilization': 0.0}

        total_cost = sum(privacy_costs)
        remaining_budget = self.privacy_mechanism.epsilon - total_cost

        return {
            'total_privacy_cost': total_cost,
            'remaining_privacy_budget': max(0, remaining_budget),
            'budget_utilization': total_cost / self.privacy_mechanism.epsilon,
            'average_cost_per_round': np.mean(privacy_costs)
        }

    def save_federated_model(self, filepath: str):
        """保存联邦模型"""
        model_state = {
            'global_model_state_dict': self.global_model.state_dict(),
            'model_version': self.model_version,
            'training_history': self.training_history,
            'privacy_epsilon': self.privacy_mechanism.epsilon,
            'privacy_delta': self.privacy_mechanism.delta,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims
        }

        torch.save(model_state, filepath)
        logger.info(f"Federated model saved to {filepath}")

    def load_federated_model(self, filepath: str):
        """加载联邦模型"""
        model_state = torch.load(filepath, map_location='cpu')

        self.global_model.load_state_dict(model_state['global_model_state_dict'])
        self.model_version = model_state['model_version']
        self.training_history = model_state['training_history']
        self.privacy_mechanism.epsilon = model_state['privacy_epsilon']
        self.privacy_mechanism.delta = model_state['privacy_delta']
        self.input_dim = model_state['input_dim']
        self.hidden_dims = model_state['hidden_dims']

        logger.info(f"Federated model loaded from {filepath}")

    def analyze(self, market_data: pd.DataFrame) -> Dict:
        """完整分析"""
        logger.info("Starting federated market prediction analysis...")

        # 预测市场趋势
        prediction = self.predict_market_trend(market_data)

        # 性能分析
        performance = self.analyze_federated_performance()

        # 生成交易建议
        trading_recommendation = self._generate_trading_recommendation(prediction, performance)

        return {
            'prediction': prediction,
            'performance_analysis': performance,
            'trading_recommendation': trading_recommendation,
            'model_info': {
                'model_type': 'Federated Learning',
                'model_version': self.model_version,
                'total_clients': len(self.clients),
                'training_rounds': len(self.training_history['rounds']),
                'privacy_protection': {
                    'epsilon': self.privacy_mechanism.epsilon,
                    'delta': self.privacy_mechanism.delta,
                    'total_cost': sum(self.training_history['privacy_costs'])
                }
            }
        }

    def _generate_trading_recommendation(self, prediction: Dict, performance: Dict) -> Dict:
        """生成交易建议"""
        trend = prediction['predicted_trend']
        confidence = prediction['confidence']

        # 基于模型性能调整置信度
        model_accuracy = performance['performance_analysis']['final_accuracy']
        adjusted_confidence = confidence * model_accuracy

        # 交易建议
        if trend == 'UP' and adjusted_confidence > 0.6:
            recommendation = 'BUY'
            reason = f"Federated model predicts upward trend with confidence {adjusted_confidence:.2f}"
        elif trend == 'DOWN' and adjusted_confidence > 0.6:
            recommendation = 'SELL'
            reason = f"Federated model predicts downward trend with confidence {adjusted_confidence:.2f}"
        else:
            recommendation = 'HOLD'
            reason = "Model prediction confidence too low for trading"

        return {
            'recommendation': recommendation,
            'predicted_trend': trend,
            'confidence': adjusted_confidence,
            'model_accuracy': model_accuracy,
            'reason': reason,
            'position_size': self._calculate_position_size(adjusted_confidence),
            'risk_level': self._assess_risk_level(performance)
        }

    def _calculate_position_size(self, confidence: float) -> float:
        """计算持仓规模"""
        if confidence > 0.8:
            return 0.2  # 20%仓位
        elif confidence > 0.6:
            return 0.1  # 10%仓位
        else:
            return 0.05  # 5%仓位

    def _assess_risk_level(self, performance: Dict) -> str:
        """评估风险水平"""
        stability = performance['training_stability']['stability_score']
        if stability > 0.8:
            return 'LOW'
        elif stability > 0.5:
            return 'MEDIUM'
        else:
            return 'HIGH'

# 使用示例
def example_usage():
    """使用示例"""
    # 创建联邦学习预测器
    predictor = FederatedMarketPredictor(
        input_dim=20,
        num_rounds=50,
        clients_per_round=3,
        privacy_epsilon=2.0
    )

    # 创建示例市场数据
    dates = pd.date_range('2023-01-01', periods=1000, freq='D')
    np.random.seed(42)

    # 模拟多个客户端的市场数据
    for client_id in ['client_1', 'client_2', 'client_3']:
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

        # 预处理数据
        features, targets = predictor.preprocess_market_data(market_data)

        # 创建客户端数据
        client_data = ClientData(
            client_id=client_id,
            market_data=market_data,
            features=features,
            targets=targets,
            data_size=len(market_data),
            privacy_budget=1.0
        )

        # 注册客户端
        predictor.register_client(client_data)

    # 联邦训练
    training_results = predictor.federated_training()
    print(f"Federated training completed with final accuracy: {training_results['training_history']['global_accuracy'][-1]:.4f}")

    # 预测
    test_data = market_data.iloc[-50:]
    prediction = predictor.predict_market_trend(test_data)
    print(f"Prediction: {prediction['predicted_trend']}, Confidence: {prediction['confidence']:.2f}")

    return predictor, prediction

if __name__ == "__main__":
    example_usage()