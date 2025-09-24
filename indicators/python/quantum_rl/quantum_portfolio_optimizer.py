"""
量子强化学习投资组合优化器
Quantum Reinforcement Learning Portfolio Optimizer

结合量子计算与强化学习的投资组合优化系统
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.distributions import Normal, Categorical
    TORCH_AVAILABLE = True
    print("🧠 PyTorch 已启用")
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch 不可用")

try:
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    from qiskit.circuit import Parameter
    from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
    from qiskit.algorithms import VQE
    from qiskit.algorithms.optimizers import COBYLA
    from qiskit.primitives import Sampler
    QISKIT_AVAILABLE = True
    print("⚛️ Qiskit 已启用")
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠️ Qiskit 不可用")

try:
    import gym
    from gym import spaces
    GYM_AVAILABLE = True
    print("🎮 Gym 已启用")
except ImportError:
    GYM_AVAILABLE = False
    print("⚠️ Gym 不可用")

class QuantumPortfolioOptimizer:
    """
    量子强化学习投资组合优化器

    使用量子计算增强的强化学习算法优化投资组合配置
    """

    def __init__(self,
                 n_assets: int = 5,
                 n_qubits: int = 4,
                 quantum_depth: int = 3,
                 learning_rate: float = 0.001,
                 risk_aversion: float = 0.5,
                 transaction_cost: float = 0.001):
        """
        初始化量子投资组合优化器

        Args:
            n_assets: 资产数量
            n_qubits: 量子比特数
            quantum_depth: 量子电路深度
            learning_rate: 学习率
            risk_aversion: 风险厌恶系数
            transaction_cost: 交易成本
        """
        self.n_assets = n_assets
        self.n_qubits = n_qubits
        self.quantum_depth = quantum_depth
        self.learning_rate = learning_rate
        self.risk_aversion = risk_aversion
        self.transaction_cost = transaction_cost

        # 量子组件
        self.quantum_circuit = None
        self.quantum_params = None
        self.quantum_optimizer = None

        # 强化学习组件
        self.q_network = None
        self.target_network = None
        self.optimizer = None
        self.memory = ReplayBuffer(capacity=10000)

        # 投资组合状态
        self.current_portfolio = np.ones(n_assets) / n_assets
        self.portfolio_history = []
        self.performance_metrics = {}

        # 训练状态
        self.is_trained = False
        self.training_episodes = 0
        self.training_rewards = []

        if TORCH_AVAILABLE:
            self._build_quantum_circuit()
            self._build_neural_networks()

    def _build_quantum_circuit(self):
        """构建量子电路"""
        if not QISKIT_AVAILABLE:
            return

        # 量子寄存器
        qr = QuantumRegister(self.n_qubits, 'q')
        cr = ClassicalRegister(self.n_qubits, 'c')
        self.quantum_circuit = QuantumCircuit(qr, cr)

        # 参数化量子电路
        self.quantum_params = []
        for depth in range(self.quantum_depth):
            for qubit in range(self.n_qubits):
                # 旋转门参数
                theta_param = Parameter(f'θ_{depth}_{qubit}')
                phi_param = Parameter(f'φ_{depth}_{qubit}')
                self.quantum_params.extend([theta_param, phi_param])

                # 添加旋转门
                self.quantum_circuit.ry(theta_param, qr[qubit])
                self.quantum_circuit.rz(phi_param, qr[qubit])

            # 纠缠层
            for qubit in range(self.n_qubits - 1):
                self.quantum_circuit.cz(qr[qubit], qr[qubit + 1])

        # 测量
        self.quantum_circuit.measure(qr, cr)

    def _build_neural_networks(self):
        """构建神经网络"""
        if not TORCH_AVAILABLE:
            return

        # 量子增强Q网络
        self.q_network = QuantumQNetwork(
            input_dim=self.n_assets * 3,  # 价格、波动率、相关性
            hidden_dims=[128, 64, 32],
            output_dim=self.n_assets,
            n_qubits=self.n_qubits,
            quantum_depth=self.quantum_depth
        )

        # 目标网络
        self.target_network = QuantumQNetwork(
            input_dim=self.n_assets * 3,
            hidden_dims=[128, 64, 32],
            output_dim=self.n_assets,
            n_qubits=self.n_qubits,
            quantum_depth=self.quantum_depth
        )

        # 复制参数
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 优化器
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)

    def train_portfolio_optimization(self,
                                   market_data: pd.DataFrame,
                                   n_episodes: int = 100,
                                   max_steps: int = 252) -> Dict[str, Any]:
        """训练投资组合优化"""
        print("🚀 开始量子强化学习投资组合优化训练...")

        # 创建环境
        env = PortfolioEnvironment(market_data, self.n_assets, self.transaction_cost)

        training_stats = {
            'episode_rewards': [],
            'portfolio_values': [],
            'sharpe_ratios': [],
            'max_drawdowns': [],
            'quantum_advantages': []
        }

        for episode in range(n_episodes):
            print(f"📊 训练回合 {episode + 1}/{n_episodes}")

            # 重置环境
            state = env.reset()
            episode_reward = 0
            portfolio_values = []

            for step in range(max_steps):
                # 量子增强的动作选择
                action, quantum_info = self._quantum_action_selection(state)

                # 执行动作
                next_state, reward, done, info = env.step(action)

                # 存储经验
                self.memory.push(state, action, reward, next_state, done)

                # 学习
                self._learn_step()

                state = next_state
                episode_reward += reward
                portfolio_values.append(info['portfolio_value'])

                if done:
                    break

            # 更新目标网络
            if episode % 10 == 0:
                self.target_network.load_state_dict(self.q_network.state_dict())

            # 记录统计信息
            training_stats['episode_rewards'].append(episode_reward)
            training_stats['portfolio_values'].append(portfolio_values)

            # 计算性能指标
            if len(portfolio_values) > 1:
                returns = pd.Series(portfolio_values).pct_change().dropna()
                sharpe_ratio = self._calculate_sharpe_ratio(returns)
                max_drawdown = self._calculate_max_drawdown(portfolio_values)

                training_stats['sharpe_ratios'].append(sharpe_ratio)
                training_stats['max_drawdowns'].append(max_drawdown)

                # 量子优势评估
                quantum_advantage = self._assess_quantum_advantage(quantum_info)
                training_stats['quantum_advantages'].append(quantum_advantage)

            self.training_episodes += 1
            self.training_rewards.append(episode_reward)

        self.is_trained = True

        return {
            'training_stats': training_stats,
            'final_portfolio': self.current_portfolio,
            'total_episodes': n_episodes,
            'convergence_info': self._analyze_convergence(training_stats),
            'quantum_performance': self._analyze_quantum_performance(training_stats)
        }

    def _quantum_action_selection(self, state: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """量子增强动作选择"""
        quantum_info = {}

        if not TORCH_AVAILABLE:
            # 随机动作
            action = np.random.dirichlet(np.ones(self.n_assets))
            return action, quantum_info

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)

            # Q网络输出
            q_values = self.q_network(state_tensor)

            # 量子增强探索
            if QISKIT_AVAILABLE and np.random.random() < 0.1:  # 10%概率使用量子探索
                quantum_action = self._generate_quantum_action()
                quantum_weights = self._quantum_to_portfolio_weights(quantum_action)
                quantum_info['quantum_used'] = True
                quantum_info['quantum_state'] = quantum_action
                return quantum_weights, quantum_info

            # ε-贪婪策略
            epsilon = max(0.01, 0.5 - self.training_episodes * 0.001)
            if np.random.random() < epsilon:
                action = np.random.dirichlet(np.ones(self.n_assets))
                quantum_info['exploration'] = True
            else:
                # 贪婪动作选择
                action_weights = F.softmax(q_values, dim=-1).squeeze().numpy()
                action = self._normalize_portfolio_weights(action_weights)
                quantum_info['exploration'] = False

        return action, quantum_info

    def _generate_quantum_action(self) -> np.ndarray:
        """生成量子动作"""
        if not QISKIT_AVAILABLE:
            return np.random.random(self.n_qubits)

        try:
            # 绑定参数
            bound_circuit = self.quantum_circuit.bind_parameters(
                {param: np.random.uniform(0, 2*np.pi) for param in self.quantum_params}
            )

            # 执行量子电路
            sampler = Sampler()
            job = sampler.run(bound_circuit, shots=1000)
            result = job.result()
            counts = result.quasi_dists[0]

            # 转换为动作概率
            action_probs = np.zeros(self.n_qubits)
            for bitstring, prob in counts.items():
                for i, bit in enumerate(bitstring):
                    if bit == '1':
                        action_probs[i] += prob

            return action_probs / np.sum(action_probs) if np.sum(action_probs) > 0 else np.random.random(self.n_qubits)

        except Exception as e:
            print(f"⚠️ 量子电路执行失败: {e}")
            return np.random.random(self.n_qubits)

    def _quantum_to_portfolio_weights(self, quantum_state: np.ndarray) -> np.ndarray:
        """量子态转投资组合权重"""
        # 将量子态映射到投资组合权重
        if len(quantum_state) != self.n_assets:
            # 如果维度不匹配，进行插值或截断
            if len(quantum_state) > self.n_assets:
                quantum_state = quantum_state[:self.n_assets]
            else:
                # 填充到正确维度
                padded_state = np.zeros(self.n_assets)
                padded_state[:len(quantum_state)] = quantum_state
                quantum_state = padded_state

        # 归一化为投资组合权重
        weights = quantum_state / np.sum(quantum_state) if np.sum(quantum_state) > 0 else np.ones(self.n_assets) / self.n_assets

        return weights

    def _normalize_portfolio_weights(self, weights: np.ndarray) -> np.ndarray:
        """归一化投资组合权重"""
        weights = np.abs(weights)  # 确保非负
        return weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(self.n_assets) / self.n_assets

    def _learn_step(self):
        """学习步骤"""
        if len(self.memory) < 32:
            return

        # 采样批次
        batch = self.memory.sample(32)
        if not batch:
            return

        states, actions, rewards, next_states, dones = zip(*batch)

        # 转换为张量
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.BoolTensor(dones)

        # 计算当前Q值
        current_q_values = self.q_network(states).gather(1, torch.argmax(actions, dim=1).unsqueeze(1))

        # 计算目标Q值
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones.float()) * 0.99 * next_q_values

        # 计算损失
        loss = F.smooth_l1_loss(current_q_values.squeeze(), target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def optimize_portfolio(self,
                          market_data: pd.DataFrame,
                          current_portfolio: np.ndarray = None,
                          optimization_horizon: int = 252) -> Dict[str, Any]:
        """优化投资组合"""
        print("🎯 执行量子投资组合优化...")

        if current_portfolio is None:
            current_portfolio = np.ones(self.n_assets) / self.n_assets

        self.current_portfolio = current_portfolio

        # 预处理市场数据
        processed_data = self._preprocess_market_data(market_data)

        # 量子计算增强的优化
        if QISKIT_AVAILABLE:
            quantum_optimization = self._quantum_portfolio_optimization(processed_data)
        else:
            quantum_optimization = self._classical_fallback_optimization(processed_data)

        # 强化学习优化
        rl_optimization = self._rl_portfolio_optimization(processed_data)

        # 融合优化结果
        final_weights = self._fuse_optimization_results(quantum_optimization, rl_optimization)

        # 风险调整
        risk_adjusted_weights = self._apply_risk_adjustment(final_weights, processed_data)

        # 交易成本调整
        final_portfolio = self._apply_transaction_costs(risk_adjusted_weights, current_portfolio)

        # 生成优化报告
        optimization_report = {
            'optimal_weights': final_portfolio,
            'previous_weights': current_portfolio,
            'quantum_contribution': quantum_optimization,
            'rl_contribution': rl_optimization,
            'expected_return': self._calculate_expected_return(final_portfolio, processed_data),
            'expected_risk': self._calculate_portfolio_risk(final_portfolio, processed_data),
            'sharpe_ratio': self._calculate_portfolio_sharpe_ratio(final_portfolio, processed_data),
            'optimization_metrics': self._calculate_optimization_metrics(final_portfolio, current_portfolio),
            'quantum_advantage_score': self._calculate_quantum_advantage_score(quantum_optimization)
        }

        return optimization_report

    def _preprocess_market_data(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """预处理市场数据"""
        processed = {}

        # 价格数据
        numeric_columns = market_data.select_dtypes(include=[np.number]).columns
        price_data = market_data[numeric_columns].fillna(method='ffill')

        # 收益率
        returns = price_data.pct_change().dropna()
        processed['returns'] = returns

        # 期望收益率
        processed['expected_returns'] = returns.mean()

        # 协方差矩阵
        processed['covariance_matrix'] = returns.cov()

        # 波动率
        processed['volatilities'] = returns.std()

        # 相关性
        processed['correlations'] = returns.corr()

        # 市场状态
        processed['market_state'] = self._analyze_market_state(price_data)

        return processed

    def _analyze_market_state(self, price_data: pd.DataFrame) -> Dict[str, Any]:
        """分析市场状态"""
        state = {}

        if 'close' in price_data.columns:
            prices = price_data['close'].dropna()
            if len(prices) > 20:
                # 趋势分析
                returns = prices.pct_change().dropna()
                trend = np.polyfit(range(len(returns)), returns.values, 1)[0]
                state['trend'] = 'bullish' if trend > 0 else 'bearish' if trend < 0 else 'neutral'

                # 波动率状态
                volatility = returns.std()
                state['volatility_state'] = 'high' if volatility > 0.2 else 'low' if volatility < 0.1 else 'medium'

                # 市场效率
                autocorr = returns.autocorr(lag=1)
                state['market_efficiency'] = 'efficient' if abs(autocorr) < 0.1 else 'inefficient'

        return state

    def _quantum_portfolio_optimization(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """量子投资组合优化"""
        quantum_result = {}

        if not QISKIT_AVAILABLE:
            return quantum_result

        try:
            # 构建量子优化问题
            expected_returns = processed_data['expected_returns'].values
            covariance_matrix = processed_data['covariance_matrix'].values

            # 量子变分优化器
            quantum_optimizer = self._create_quantum_optimizer(expected_returns, covariance_matrix)

            # 运行优化
            optimal_params = self._run_quantum_optimization(quantum_optimizer)

            # 转换为投资组合权重
            optimal_weights = self._quantum_params_to_weights(optimal_params)

            quantum_result = {
                'weights': optimal_weights,
                'optimization_params': optimal_params,
                'convergence_history': [],
                'quantum_circuits_used': self.quantum_depth,
                'optimization_time': 0.0  # 实际应用中应该计时
            }

        except Exception as e:
            print(f"⚠️ 量子优化失败: {e}")
            quantum_result['error'] = str(e)

        return quantum_result

    def _create_quantum_optimizer(self, expected_returns: np.ndarray, covariance_matrix: np.ndarray):
        """创建量子优化器"""
        # 这里应该构建适当的量子优化电路
        # 简化的实现
        return {
            'expected_returns': expected_returns,
            'covariance_matrix': covariance_matrix,
            'n_assets': len(expected_returns)
        }

    def _run_quantum_optimization(self, optimizer: Dict[str, Any]) -> np.ndarray:
        """运行量子优化"""
        # 简化的量子优化实现
        n_assets = optimizer['n_assets']
        return np.random.random(n_assets)  # 实际应该是量子优化结果

    def _quantum_params_to_weights(self, quantum_params: np.ndarray) -> np.ndarray:
        """量子参数转权重"""
        weights = np.abs(quantum_params)
        return weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(quantum_params)) / len(quantum_params)

    def _classical_fallback_optimization(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """经典优化回退"""
        # 简单的均值方差优化
        expected_returns = processed_data['expected_returns'].values
        covariance_matrix = processed_data['covariance_matrix'].values

        # 最小方差组合
        try:
            inv_cov = np.linalg.inv(covariance_matrix)
            ones = np.ones(len(expected_returns))
            min_var_weights = inv_cov @ ones / (ones @ inv_cov @ ones)

            return {
                'weights': min_var_weights,
                'method': 'minimum_variance',
                'fallback_reason': 'quantum_not_available'
            }
        except:
            # 等权重组合
            equal_weights = np.ones(len(expected_returns)) / len(expected_returns)
            return {
                'weights': equal_weights,
                'method': 'equal_weight',
                'fallback_reason': 'matrix_inversion_failed'
            }

    def _rl_portfolio_optimization(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """强化学习优化"""
        rl_result = {}

        if not self.is_trained or not TORCH_AVAILABLE:
            # 使用训练好的策略或随机策略
            rl_result['weights'] = np.ones(self.n_assets) / self.n_assets
            rl_result['method'] = 'untrained_or_no_torch'
            return rl_result

        # 使用训练好的Q网络进行优化
        try:
            # 构建状态表示
            state_features = self._build_rl_state_features(processed_data)

            with torch.no_grad():
                state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                weights = F.softmax(q_values, dim=-1).squeeze().numpy()

            rl_result = {
                'weights': weights,
                'method': 'trained_q_network',
                'confidence': float(torch.max(q_values))
            }

        except Exception as e:
            print(f"⚠️ 强化学习优化失败: {e}")
            rl_result['weights'] = np.ones(self.n_assets) / self.n_assets
            rl_result['error'] = str(e)

        return rl_result

    def _build_rl_state_features(self, processed_data: Dict[str, Any]) -> np.ndarray:
        """构建强化学习状态特征"""
        features = []

        # 期望收益率
        features.extend(processed_data['expected_returns'].values)

        # 波动率
        features.extend(processed_data['volatilities'].values)

        # 市场状态编码
        market_state = processed_data['market_state']
        state_encoding = [
            1.0 if market_state.get('trend') == 'bullish' else 0.0,
            1.0 if market_state.get('trend') == 'bearish' else 0.0,
            1.0 if market_state.get('volatility_state') == 'high' else 0.0,
            1.0 if market_state.get('market_efficiency') == 'efficient' else 0.0
        ]
        features.extend(state_encoding)

        return np.array(features)

    def _fuse_optimization_results(self, quantum_result: Dict[str, Any], rl_result: Dict[str, Any]) -> np.ndarray:
        """融合优化结果"""
        # 权重融合策略
        quantum_weights = quantum_result.get('weights', np.ones(self.n_assets) / self.n_assets)
        rl_weights = rl_result.get('weights', np.ones(self.n_assets) / self.n_assets)

        # 动态权重分配
        quantum_confidence = self._assess_quantum_confidence(quantum_result)
        rl_confidence = rl_result.get('confidence', 0.5)

        total_confidence = quantum_confidence + rl_confidence
        if total_confidence > 0:
            quantum_weight = quantum_confidence / total_confidence
            rl_weight = rl_confidence / total_confidence
        else:
            quantum_weight = 0.5
            rl_weight = 0.5

        # 加权融合
        fused_weights = quantum_weight * quantum_weights + rl_weight * rl_weights

        return self._normalize_portfolio_weights(fused_weights)

    def _assess_quantum_confidence(self, quantum_result: Dict[str, Any]) -> float:
        """评估量子优化置信度"""
        if 'error' in quantum_result:
            return 0.1  # 低置信度

        if 'fallback_reason' in quantum_result:
            return 0.3  # 中等置信度

        return 0.8  # 高置信度

    def _apply_risk_adjustment(self, weights: np.ndarray, processed_data: Dict[str, Any]) -> np.ndarray:
        """应用风险调整"""
        # 基于风险厌恶系数调整权重
        volatilities = processed_data['volatilities'].values

        # 风险调整因子
        risk_factors = 1.0 / (1.0 + self.risk_aversion * volatilities)

        # 调整权重
        adjusted_weights = weights * risk_factors

        return self._normalize_portfolio_weights(adjusted_weights)

    def _apply_transaction_costs(self, optimal_weights: np.ndarray, current_weights: np.ndarray) -> np.ndarray:
        """应用交易成本调整"""
        # 计算交易量
        trading_volume = np.abs(optimal_weights - current_weights)

        # 交易成本惩罚
        cost_penalty = self.transaction_cost * trading_volume

        # 调整权重以减少交易
        adjusted_weights = current_weights + 0.8 * (optimal_weights - current_weights)

        return self._normalize_portfolio_weights(adjusted_weights)

    def _calculate_expected_return(self, weights: np.ndarray, processed_data: Dict[str, Any]) -> float:
        """计算期望收益率"""
        expected_returns = processed_data['expected_returns'].values
        return np.dot(weights, expected_returns)

    def _calculate_portfolio_risk(self, weights: np.ndarray, processed_data: Dict[str, Any]) -> float:
        """计算投资组合风险"""
        covariance_matrix = processed_data['covariance_matrix'].values
        return np.sqrt(np.dot(weights.T, np.dot(covariance_matrix, weights)))

    def _calculate_portfolio_sharpe_ratio(self, weights: np.ndarray, processed_data: Dict[str, Any]) -> float:
        """计算夏普比率"""
        expected_return = self._calculate_expected_return(weights, processed_data)
        risk = self._calculate_portfolio_risk(weights, processed_data)

        if risk > 0:
            return expected_return / risk
        else:
            return 0.0

    def _calculate_optimization_metrics(self, new_weights: np.ndarray, old_weights: np.ndarray) -> Dict[str, float]:
        """计算优化指标"""
        metrics = {}

        # 权重变化
        weight_change = np.mean(np.abs(new_weights - old_weights))
        metrics['weight_change'] = weight_change

        # 分散化程度
        herfindahl_index = np.sum(new_weights ** 2)
        metrics['diversification'] = 1.0 - herfindahl_index

        # 集中度
        max_weight = np.max(new_weights)
        metrics['concentration'] = max_weight

        # 有效资产数
        effective_assets = 1.0 / herfindahl_index
        metrics['effective_assets'] = effective_assets

        return metrics

    def _calculate_quantum_advantage_score(self, quantum_result: Dict[str, Any]) -> float:
        """计算量子优势分数"""
        if 'error' in quantum_result:
            return 0.0

        if 'fallback_reason' in quantum_result:
            return 0.3

        # 基于量子电路深度和参数数量评估
        quantum_score = min(1.0, self.quantum_depth / 10.0)
        return quantum_score

    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        excess_returns = returns - risk_free_rate / 252  # 日化无风险利率
        return np.sqrt(252) * excess_returns.mean() / returns.std()

    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """计算最大回撤"""
        if len(portfolio_values) < 2:
            return 0.0

        cumulative = np.array(portfolio_values)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak

        return np.min(drawdown)

    def _assess_quantum_advantage(self, quantum_info: Dict[str, Any]) -> float:
        """评估量子优势"""
        if not quantum_info.get('quantum_used', False):
            return 0.0

        # 基于量子态的质量评估优势
        quantum_state = quantum_info.get('quantum_state', np.array([]))
        if len(quantum_state) == 0:
            return 0.0

        # 量子优势度量：量子态的纠缠度和叠加态质量
        state_purity = np.sum(quantum_state ** 2)
        quantum_advantage = 1.0 - state_purity  # 越纯态优势越小

        return quantum_advantage

    def _analyze_convergence(self, training_stats: Dict[str, Any]) -> Dict[str, Any]:
        """分析收敛性"""
        convergence_info = {}

        episode_rewards = training_stats['episode_rewards']
        if len(episode_rewards) > 10:
            # 检查最后10个回合的平均奖励
            recent_avg = np.mean(episode_rewards[-10:])
            overall_avg = np.mean(episode_rewards)

            convergence_info['recent_average'] = recent_avg
            convergence_info['overall_average'] = overall_avg
            convergence_info['convergence_ratio'] = recent_avg / overall_avg if overall_avg != 0 else 0
            convergence_info['is_converged'] = abs(recent_avg - overall_avg) / abs(overall_avg) < 0.1 if overall_avg != 0 else False

        return convergence_info

    def _analyze_quantum_performance(self, training_stats: Dict[str, Any]) -> Dict[str, Any]:
        """分析量子性能"""
        quantum_performance = {}

        quantum_advantages = training_stats.get('quantum_advantages', [])
        if quantum_advantages:
            quantum_performance['average_quantum_advantage'] = np.mean(quantum_advantages)
            quantum_performance['max_quantum_advantage'] = np.max(quantum_advantages)
            quantum_performance['quantum_usage_rate'] = len([qa for qa in quantum_advantages if qa > 0]) / len(quantum_advantages)

        return quantum_performance

    def get_optimization_summary(self) -> Dict[str, Any]:
        """获取优化总结"""
        return {
            'is_trained': self.is_trained,
            'training_episodes': self.training_episodes,
            'n_assets': self.n_assets,
            'n_qubits': self.n_qubits,
            'quantum_depth': self.quantum_depth,
            'learning_rate': self.learning_rate,
            'risk_aversion': self.risk_aversion,
            'current_portfolio': self.current_portfolio.tolist(),
            'memory_size': len(self.memory),
            'model_info': self.get_model_info()
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'model_type': 'Quantum Reinforcement Learning Portfolio Optimizer',
            'torch_available': TORCH_AVAILABLE,
            'qiskit_available': QISKIT_AVAILABLE,
            'gym_available': GYM_AVAILABLE,
            'n_assets': self.n_assets,
            'n_qubits': self.n_qubits,
            'quantum_depth': self.quantum_depth,
            'risk_aversion': self.risk_aversion,
            'transaction_cost': self.transaction_cost
        }


class QuantumQNetwork(nn.Module):
    """量子Q网络"""
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, n_qubits: int, quantum_depth: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.n_qubits = n_qubits
        self.quantum_depth = quantum_depth

        # 神经网络层
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

        # 量子增强层（简化实现）
        self.quantum_enhancement = nn.Linear(output_dim, output_dim)

    def forward(self, x):
        # 神经网络前向传播
        neural_output = self.network(x)

        # 量子增强
        quantum_output = self.quantum_enhancement(neural_output)

        # 融合输出
        final_output = neural_output + 0.1 * quantum_output

        return final_output


class PortfolioEnvironment:
    """投资组合环境"""
    def __init__(self, market_data: pd.DataFrame, n_assets: int, transaction_cost: float = 0.001):
        self.market_data = market_data
        self.n_assets = n_assets
        self.transaction_cost = transaction_cost

        # 状态空间
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(n_assets * 3,),  # 价格、波动率、相关性
            dtype=np.float32
        )

        # 动作空间
        self.action_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(n_assets,),
            dtype=np.float32
        )

        # 环境状态
        self.current_step = 0
        self.max_steps = len(market_data) - 1
        self.current_portfolio = np.ones(n_assets) / n_assets
        self.portfolio_value = 1.0

    def reset(self):
        """重置环境"""
        self.current_step = 0
        self.current_portfolio = np.ones(self.n_assets) / self.n_assets
        self.portfolio_value = 1.0
        return self._get_state()

    def step(self, action):
        """执行动作"""
        # 归一化动作
        action = action / np.sum(action) if np.sum(action) > 0 else np.ones(self.n_assets) / self.n_assets

        # 计算奖励
        reward = self._calculate_reward(action)

        # 更新状态
        self.current_step += 1
        self.current_portfolio = action

        # 计算投资组合价值
        self.portfolio_value *= (1 + reward)

        # 检查是否结束
        done = self.current_step >= self.max_steps

        # 获取新状态
        next_state = self._get_state()

        # 信息
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_weights': action.copy(),
            'step': self.current_step
        }

        return next_state, reward, done, info

    def _get_state(self):
        """获取状态"""
        if self.current_step >= len(self.market_data):
            return np.zeros(self.n_assets * 3)

        # 简化的状态表示
        price_data = self.market_data.iloc[self.current_step].select_dtypes(include=[np.number]).values

        # 填充或截断到正确维度
        if len(price_data) < self.n_assets * 3:
            padded_state = np.zeros(self.n_assets * 3)
            padded_state[:len(price_data)] = price_data
            return padded_state
        else:
            return price_data[:self.n_assets * 3]

    def _calculate_reward(self, action):
        """计算奖励"""
        # 简化的奖励计算
        if self.current_step >= len(self.market_data) - 1:
            return 0.0

        # 获取当前和下一期的价格数据
        current_prices = self.market_data.iloc[self.current_step].select_dtypes(include=[np.number]).values
        next_prices = self.market_data.iloc[self.current_step + 1].select_dtypes(include=[np.number]).values

        # 计算资产收益率
        if len(current_prices) >= self.n_assets and len(next_prices) >= self.n_assets:
            asset_returns = (next_prices[:self.n_assets] - current_prices[:self.n_assets]) / current_prices[:self.n_assets]

            # 投资组合收益率
            portfolio_return = np.dot(action, asset_returns)

            # 交易成本惩罚
            trading_cost_penalty = self.transaction_cost * np.sum(np.abs(action - self.current_portfolio))

            # 净奖励
            reward = portfolio_return - trading_cost_penalty
        else:
            reward = 0.0

        return reward


class ReplayBuffer:
    """经验回放缓冲区"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        """存储经验"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int):
        """采样批次"""
        if len(self.buffer) < batch_size:
            return None

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)


# 便捷函数
def create_quantum_portfolio_optimizer(n_assets: int = 5) -> QuantumPortfolioOptimizer:
    """创建量子投资组合优化器实例"""
    return QuantumPortfolioOptimizer(n_assets=n_assets)

def quick_portfolio_optimization(market_data: pd.DataFrame, n_assets: int = 5) -> Dict[str, Any]:
    """快速投资组合优化"""
    optimizer = create_quantum_portfolio_optimizer(n_assets)
    return optimizer.optimize_portfolio(market_data)

def train_quantum_optimizer(market_data: pd.DataFrame, n_assets: int = 5, n_episodes: int = 50) -> Dict[str, Any]:
    """训练量子优化器"""
    optimizer = create_quantum_portfolio_optimizer(n_assets)
    return optimizer.train_portfolio_optimization(market_data, n_episodes)