"""
GAN Market Synthesizer
======================

基于生成对抗网络的市场数据合成器和异常检测器，
通过学习真实市场数据的分布来生成合成数据并识别
异常模式。该指标利用GAN的强大生成能力进行市场
分析和模式识别。

特点：
- 生成对抗网络
- 数据合成和增强
- 异常检测
- 模式识别
- 分布学习
- 特征提取
- 风险评估
- 市场状态识别

Author: Claude AI Assistant
Version: 1.0
Date: 2025-09-24
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Union, List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class Generator(nn.Module):
    """生成器网络"""
    def __init__(self, latent_dim: int, output_dim: int, hidden_dims: List[int] = [256, 512, 256]):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = latent_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # 输出到[-1,1]范围

        self.model = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)


class Discriminator(nn.Module):
    """判别器网络"""
    def __init__(self, input_dim: int, hidden_dims: List[int] = [512, 256, 128]):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 输出概率

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def feature_extraction(self, x: torch.Tensor) -> torch.Tensor:
        """提取特征（用于异常检测）"""
        features = []
        input = x

        for layer in self.model:
            if isinstance(layer, nn.Linear):
                input = layer(input)
            elif isinstance(layer, nn.LeakyReLU):
                input = layer(input)
                features.append(input)

        return features[-1] if features else input


class Encoder(nn.Module):
    """编码器网络（用于异常检测）"""
    def __init__(self, input_dim: int, latent_dim: int, hidden_dims: List[int] = [256, 128]):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, latent_dim))

        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class GANMarketSynthesizer:
    """
    基于GAN的市场数据合成器

    利用生成对抗网络进行市场数据
    合成、异常检测和模式识别。
    """

    def __init__(self, latent_dim: int = 100, hidden_dims: List[int] = [256, 512, 256],
                 epochs: int = 200, batch_size: int = 32, learning_rate: float = 0.0002):
        """
        初始化GAN市场合成器

        Args:
            latent_dim: 潜在空间维度，默认100
            hidden_dims: 隐藏层维度，默认[256,512,256]
            epochs: 训练轮数，默认200
            batch_size: 批次大小，默认32
            learning_rate: 学习率，默认0.0002
        """
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.name = f"GAN Market Synthesizer ({latent_dim})"
        self.category = "deep_learning"

        # 设备和模型
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = None
        self.discriminator = None
        self.encoder = None
        self.scaler = StandardScaler()
        self.is_trained = False

        # 训练历史
        self.training_history = {'g_loss': [], 'd_loss': [], 'anomaly_scores': []}

    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        准备特征数据

        Args:
            data: 原始市场数据

        Returns:
            特征数据
        """
        features = pd.DataFrame(index=data.index)

        # 价格特征
        if 'close' in data.columns:
            prices = data['close']
            features['price'] = prices
            features['returns'] = prices.pct_change().fillna(0)
            features['log_returns'] = np.log(prices / prices.shift(1)).fillna(0)

            # 技术指标
            features['sma_5'] = prices.rolling(window=5).mean()
            features['sma_10'] = prices.rolling(window=10).mean()
            features['sma_20'] = prices.rolling(window=20).mean()

            # 指数移动平均
            features['ema_12'] = prices.ewm(span=12).mean()
            features['ema_26'] = prices.ewm(span=26).mean()

            # 相对强弱指数
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            features['rsi'] = 100 - (100 / (1 + rs)).fillna(50)

            # 布林带
            bb_period = 20
            bb_middle = prices.rolling(window=bb_period).mean()
            bb_std = prices.rolling(window=bb_period).std()
            features['bb_upper'] = bb_middle + (bb_std * 2)
            features['bb_lower'] = bb_middle - (bb_std * 2)
            features['bb_width'] = (features['bb_upper'] - features['bb_lower']) / bb_middle

        # 成交量特征
        if 'volume' in data.columns:
            volume = data['volume']
            features['volume'] = volume
            features['volume_sma'] = volume.rolling(window=20).mean()
            features['volume_ratio'] = volume / (features['volume_sma'] + 1e-10)
            features['volume_change'] = volume.pct_change().fillna(0)

        # 波动率特征
        if 'close' in data.columns:
            returns = features['returns']
            features['volatility_5'] = returns.rolling(window=5).std()
            features['volatility_20'] = returns.rolling(window=20).std()
            features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-10)

        # 高低价特征
        if 'high' in data.columns and 'low' in data.columns:
            features['high'] = data['high']
            features['low'] = data['low']
            features['range'] = data['high'] - data['low']
            features['range_sma'] = features['range'].rolling(window=20).mean()

            # 价格位置
            features['price_position'] = (data['close'] - features['low']) / (features['range'] + 1e-10)

        # 清理数据
        features = features.fillna(method='ffill').fillna(0)

        # 移除无限值
        features = features.replace([np.inf, -np.inf], 0)

        return features

    def build_models(self, input_dim: int) -> Tuple[Generator, Discriminator, Encoder]:
        """
        构建模型

        Args:
            input_dim: 输入维度

        Returns:
            生成器、判别器、编码器
        """
        generator = Generator(self.latent_dim, input_dim, self.hidden_dims).to(self.device)
        discriminator = Discriminator(input_dim, self.hidden_dims).to(self.device)
        encoder = Encoder(input_dim, self.latent_dim, [128, 64]).to(self.device)

        return generator, discriminator, encoder

    def train_gan(self, features: pd.DataFrame) -> Dict[str, List[float]]:
        """
        训练GAN模型

        Args:
            features: 特征数据

        Returns:
            训练历史
        """
        # 数据标准化
        features_scaled = self.scaler.fit_transform(features)

        # 构建模型
        input_dim = features_scaled.shape[1]
        self.generator, self.discriminator, self.encoder = self.build_models(input_dim)

        # 优化器
        g_optimizer = optim.Adam(self.generator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        d_optimizer = optim.Adam(self.discriminator.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
        e_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate * 0.5)

        # 损失函数
        criterion = nn.BCELoss()
        mse_criterion = nn.MSELoss()

        # 转换为张量
        real_data = torch.FloatTensor(features_scaled).to(self.device)

        # 训练循环
        for epoch in range(self.epochs):
            # 训练判别器
            for _ in range(1):  # 判别器训练次数
                self.discriminator.train()
                self.generator.eval()

                # 真实数据
                real_labels = torch.ones(real_data.size(0), 1).to(self.device)
                real_output = self.discriminator(real_data)
                d_loss_real = criterion(real_output, real_labels)

                # 生成数据
                noise = torch.randn(real_data.size(0), self.latent_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_labels = torch.zeros(real_data.size(0), 1).to(self.device)
                fake_output = self.discriminator(fake_data.detach())
                d_loss_fake = criterion(fake_output, fake_labels)

                # 总判别器损失
                d_loss = d_loss_real + d_loss_fake

                d_optimizer.zero_grad()
                d_loss.backward()
                d_optimizer.step()

            # 训练生成器
            for _ in range(2):  # 生成器训练次数
                self.generator.train()
                self.discriminator.eval()

                noise = torch.randn(real_data.size(0), self.latent_dim).to(self.device)
                fake_data = self.generator(noise)
                fake_labels = torch.ones(real_data.size(0), 1).to(self.device)
                fake_output = self.discriminator(fake_data)
                g_loss = criterion(fake_output, fake_labels)

                # 编码器损失（辅助训练）
                if epoch > 50:  # 50轮后开始训练编码器
                    self.encoder.train()
                    encoded_features = self.encoder(real_data)
                    reconstructed_data = self.generator(encoded_features)
                    e_loss = mse_criterion(reconstructed_data, real_data)

                    e_optimizer.zero_grad()
                    e_loss.backward()
                    e_optimizer.step()

                g_optimizer.zero_grad()
                g_loss.backward()
                g_optimizer.step()

            # 记录历史
            self.training_history['g_loss'].append(g_loss.item())
            self.training_history['d_loss'].append(d_loss.item())

            # 计算异常分数
            if epoch % 10 == 0:
                anomaly_scores = self.calculate_anomaly_scores(real_data)
                self.training_history['anomaly_scores'].append(anomaly_scores.mean().item())

            if epoch % 50 == 0:
                print(f"Epoch {epoch}/{self.epochs}, G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}")

        self.is_trained = True
        return self.training_history

    def generate_synthetic_data(self, n_samples: int) -> pd.DataFrame:
        """
        生成合成数据

        Args:
            n_samples: 样本数量

        Returns:
            合成数据
        """
        if not self.is_trained or self.generator is None:
            return pd.DataFrame()

        self.generator.eval()
        with torch.no_grad():
            noise = torch.randn(n_samples, self.latent_dim).to(self.device)
            synthetic_data = self.generator(noise).cpu().numpy()

        # 反标准化
        synthetic_data = self.scaler.inverse_transform(synthetic_data)

        # 创建DataFrame
        feature_names = [f'feature_{i}' for i in range(synthetic_data.shape[1])]
        synthetic_df = pd.DataFrame(synthetic_data, columns=feature_names)

        return synthetic_df

    def calculate_anomaly_scores(self, real_data: torch.Tensor) -> torch.Tensor:
        """
        计算异常分数

        Args:
            real_data: 真实数据

        Returns:
            异常分数
        """
        if not self.is_trained:
            return torch.zeros(real_data.size(0))

        self.generator.eval()
        self.discriminator.eval()
        self.encoder.eval()

        with torch.no_grad():
            # 方法1: 判别器不确定性
            real_output = self.discriminator(real_data)
            discriminator_uncertainty = torch.abs(real_output - 0.5) * 2  # 越接近0.5越异常

            # 方法2: 重建误差
            encoded_features = self.encoder(real_data)
            reconstructed_data = self.generator(encoded_features)
            reconstruction_error = torch.mean((real_data - reconstructed_data) ** 2, dim=1)

            # 方法3: 潜在空间距离
            encoded_real = self.encoder(real_data)
            noise_samples = torch.randn(real_data.size(0), self.latent_dim).to(self.device)
            latent_distance = torch.mean((encoded_real - noise_samples) ** 2, dim=1)

            # 综合异常分数
            anomaly_scores = (discriminator_uncertainty.squeeze() +
                            reconstruction_error +
                            latent_distance) / 3

        return anomaly_scores

    def detect_anomalies(self, features: pd.DataFrame) -> pd.Series:
        """
        检测异常

        Args:
            features: 特征数据

        Returns:
            异常分数
        """
        if not self.is_trained:
            return pd.Series()

        # 数据标准化
        features_scaled = self.scaler.transform(features)

        # 转换为张量
        real_data = torch.FloatTensor(features_scaled).to(self.device)

        # 计算异常分数
        anomaly_scores = self.calculate_anomaly_scores(real_data)

        # 转换为pandas Series
        anomaly_series = pd.Series(anomaly_scores.cpu().numpy(), index=features.index)

        return anomaly_series

    def analyze_market_regimes(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        分析市场状态

        Args:
            features: 特征数据

        Returns:
            市场状态分析
        """
        regime_analysis = {}

        # 异常检测
        anomaly_scores = self.detect_anomalies(features)
        regime_analysis['anomaly_scores'] = anomaly_scores

        # 异常阈值
        anomaly_threshold = anomaly_scores.quantile(0.9)
        regime_analysis['anomaly_threshold'] = pd.Series(anomaly_threshold, index=features.index)

        # 异常状态
        is_anomaly = anomaly_scores > anomaly_threshold
        regime_analysis['is_anomaly'] = is_anomaly.astype(int)

        # 异常强度
        anomaly_intensity = (anomaly_scores - anomaly_scores.min()) / (anomaly_scores.max() - anomaly_scores.min() + 1e-10)
        regime_analysis['anomaly_intensity'] = anomaly_intensity

        # 市场正常性得分
        normality_score = 1 - anomaly_intensity
        regime_analysis['normality_score'] = normality_score

        # 异常持续性
        anomaly_persistence = is_anomaly.rolling(window=5).sum()
        regime_analysis['anomaly_persistence'] = anomaly_persistence.fillna(0)

        return regime_analysis

    def generate_market_patterns(self, n_patterns: int = 10) -> List[pd.DataFrame]:
        """
        生成市场模式

        Args:
            n_patterns: 模式数量

        Returns:
            市场模式列表
        """
        patterns = []

        for i in range(n_patterns):
            # 生成合成数据
            synthetic_data = self.generate_synthetic_data(100)

            # 添加一些特定的模式特征
            pattern_type = i % 5

            if pattern_type == 0:  # 趋势模式
                synthetic_data['feature_0'] += np.linspace(0, 1, 100)
            elif pattern_type == 1:  # 周期模式
                synthetic_data['feature_0'] += np.sin(np.linspace(0, 4*np.pi, 100)) * 0.5
            elif pattern_type == 2:  # 突发模式
                spike_indices = np.random.choice(100, 5, replace=False)
                synthetic_data.loc[spike_indices, 'feature_0'] *= 2
            elif pattern_type == 3:  # 均值回归模式
                synthetic_data['feature_0'] = np.random.randn(100) * 0.1
            elif pattern_type == 4:  # 随机游走模式
                synthetic_data['feature_0'] = np.cumsum(np.random.randn(100) * 0.1)

            patterns.append(synthetic_data)

        return patterns

    def calculate_risk_metrics(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算风险指标

        Args:
            features: 特征数据

        Returns:
            风险指标
        """
        risk_metrics = {}

        # 异常检测
        anomaly_scores = self.detect_anomalies(features)
        risk_metrics['anomaly_risk'] = anomaly_scores

        # 异似风险
        anomaly_threshold = anomaly_scores.quantile(0.8)
        high_anomaly_risk = anomaly_scores > anomaly_threshold
        risk_metrics['high_anomaly_risk'] = high_anomaly_risk.astype(int)

        # 系统性风险（基于特征相关性）
        if len(features.columns) > 1:
            correlation_matrix = features.corr()
            systemic_risk = correlation_matrix.abs().mean().mean()
            risk_metrics['systemic_risk'] = pd.Series(systemic_risk, index=features.index)

        # 流动性风险（基于数据波动性）
        if 'volume' in features.columns:
            volume_volatility = features['volume'].rolling(window=20).std()
            liquidity_risk = volume_volatility / (features['volume'].rolling(window=20).mean() + 1e-10)
            risk_metrics['liquidity_risk'] = liquidity_risk.fillna(0)

        # 模型不确定性风险
        if self.is_trained:
            # 生成多个样本来评估不确定性
            n_samples = min(100, len(features))
            synthetic_samples = []

            for _ in range(10):  # 生成10个样本
                synthetic_data = self.generate_synthetic_data(n_samples)
                synthetic_samples.append(synthetic_data.values)

            # 计算生成数据的方差
            synthetic_variance = np.var(synthetic_samples, axis=0)
            model_uncertainty = np.mean(synthetic_variance, axis=1)

            # 创建对应长度的序列
            uncertainty_series = pd.Series(
                np.tile(model_uncertainty, (len(features) // n_samples + 1))[:len(features)],
                index=features.index
            )
            risk_metrics['model_uncertainty'] = uncertainty_series

        return risk_metrics

    def analyze_market_with_gan(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        使用GAN分析市场

        Args:
            data: 市场数据

        Returns:
            GAN分析结果
        """
        results = {}

        # 准备特征
        features = self.prepare_features(data)

        # 训练GAN（如果尚未训练）
        if not self.is_trained:
            print("训练GAN模型...")
            history = self.train_gan(features)
            results['training_history'] = history

        # 市场状态分析
        regime_analysis = self.analyze_market_regimes(features)
        results.update(regime_analysis)

        # 风险指标
        risk_metrics = self.calculate_risk_metrics(features)
        results.update(risk_metrics)

        # 生成合成数据示例
        synthetic_data = self.generate_synthetic_data(100)
        results['synthetic_data_sample'] = synthetic_data

        # 生成市场模式
        market_patterns = self.generate_market_patterns(5)
        results['market_patterns'] = market_patterns

        # 生成交易信号
        results['signals'] = self.generate_gan_signals(results)

        return results

    def generate_gan_signals(self, gan_analysis: Dict[str, Any]) -> pd.Series:
        """
        基于GAN分析生成交易信号

        Args:
            gan_analysis: GAN分析结果

        Returns:
            交易信号
        """
        if 'anomaly_scores' not in gan_analysis:
            return pd.Series()

        anomaly_scores = gan_analysis['anomaly_scores']
        normality_scores = gan_analysis.get('normality_score', 1 - anomaly_scores)

        signals = pd.Series(0, index=anomaly_scores.index)

        # 异常阈值
        high_anomaly_threshold = anomaly_scores.quantile(0.9)
        low_anomaly_threshold = anomaly_scores.quantile(0.1)

        # 正常性阈值
        high_normality_threshold = normality_scores.quantile(0.9)
        low_normality_threshold = normality_scores.quantile(0.1)

        # 异常信号
        extreme_anomaly = anomaly_scores > high_anomaly_threshold
        low_anomaly = anomaly_scores < low_anomaly_threshold

        # 正常性信号
        high_normality = normality_scores > high_normality_threshold
        low_normality = normality_scores < low_normality_threshold

        # 风险确认
        high_anomaly_risk = gan_analysis.get('high_anomaly_risk', pd.Series(0, index=anomaly_scores.index))
        model_uncertainty = gan_analysis.get('model_uncertainty', pd.Series(0, index=anomaly_scores.index))

        # 信号逻辑
        # 在低异常和正常性高时买入
        buy_signal = low_anomaly & high_normality & (high_anomaly_risk == 0)

        # 在高异常和正常性低时卖出
        sell_signal = extreme_anomaly & low_normality

        # 模型不确定性调整
        uncertainty_adjustment = 1 - model_uncertainty / (model_uncertainty.max() + 1e-10)
        buy_signal &= (uncertainty_adjustment > 0.5)
        sell_signal &= (uncertainty_adjustment > 0.5)

        # 分配信号值
        signals[buy_signal] = 2
        signals[sell_signal] = -2

        # 极端信号
        extreme_buy = buy_signal & (normality_scores > normality_scores.quantile(0.95))
        extreme_sell = sell_signal & (anomaly_scores > anomaly_scores.quantile(0.95))

        signals[extreme_buy] = 3
        signals[extreme_sell] = -3

        return signals

    def get_trading_recommendations(self, gan_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            gan_analysis: GAN分析结果

        Returns:
            交易建议DataFrame
        """
        # 找到一个指标序列作为索引
        index_keys = [k for k, v in gan_analysis.items() if isinstance(v, pd.Series) and not v.empty]
        if not index_keys:
            return pd.DataFrame()

        index = gan_analysis[index_keys[0]].index
        recommendations = pd.DataFrame(index=index)

        # 添加GAN分析指标
        for key, value in gan_analysis.items():
            if isinstance(value, pd.Series):
                recommendations[key] = value

        # 交易信号
        recommendations['signals'] = gan_analysis.get('signals', pd.Series(0, index=index))

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - 极低异常风险',
            2: '买入 - 低异常风险',
            0: '持有 - 正常状态',
            -2: '卖出 - 高异常风险',
            -3: '强烈卖出 - 极高异常风险'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 市场状态分类
        def classify_market_state(row):
            anomaly_score = row.get('anomaly_scores', 0)
            normality_score = row.get('normality_score', 1.0)
            high_anomaly_risk = row.get('high_anomaly_risk', 0)

            if high_anomaly_risk > 0:
                return 'anomaly_market'
            elif anomaly_score > anomaly_score.quantile(0.8):
                return 'high_risk_market'
            elif normality_score > normality_score.quantile(0.8):
                return 'normal_market'
            else:
                return 'transitional_market'

        recommendations['market_state'] = recommendations.apply(classify_market_state, axis=1)

        # 仓位建议
        position_map = {3: 0.5, 2: 0.3, 0: 0.2, -2: 0.1, -3: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 异常风险调整
        anomaly_adjustment = 1 - recommendations['anomaly_scores'] / (recommendations['anomaly_scores'].max() + 1e-10)
        recommendations['position_size'] *= anomaly_adjustment

        # 模型不确定性调整
        if 'model_uncertainty' in recommendations.columns:
            uncertainty_adjustment = 1 - recommendations['model_uncertainty'] / (recommendations['model_uncertainty'].max() + 1e-10)
            recommendations['position_size'] *= uncertainty_adjustment

        # 确保仓位在合理范围内
        recommendations['position_size'] = recommendations['position_size'].clip(0, 1)

        # 策略建议
        def get_gan_strategy(market_state, signal_strength):
            if market_state == 'anomaly_market':
                return 'defensive_gan'  # 防御性策略
            elif market_state == 'high_risk_market':
                return 'risk_avoidance'  # 风险规避
            elif market_state == 'normal_market':
                return 'normal_trading'  # 正常交易
            else:
                return 'cautious_trading'  # 谨慎交易

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_gan_strategy(x['market_state'], abs(x['signals'])),
            axis=1
        )

        # GAN风险评估
        def assess_gan_risk(row):
            risk_score = 0

            # 异常风险
            if row.get('anomaly_scores', 0) > row.get('anomaly_scores', 0).quantile(0.8):
                risk_score += 0.4

            # 高异常风险
            if row.get('high_anomaly_risk', 0) > 0:
                risk_score += 0.3

            # 模型不确定性
            if 'model_uncertainty' in row.index:
                if row['model_uncertainty'] > row['model_uncertainty'].quantile(0.8):
                    risk_score += 0.3

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['risk_level'] = recommendations.apply(assess_gan_risk, axis=1)

        return recommendations


# 使用示例
if __name__ == "__main__":
    # 生成模拟市场数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')

    # 生成价格序列
    base_price = 100
    returns = np.random.randn(300) * 0.02

    # 添加一些异常
    anomaly_indices = [50, 120, 200, 250]
    returns[anomaly_indices] += np.random.randn(len(anomaly_indices)) * 0.1

    # 构建价格
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))

    # 生成成交量
    base_volume = 1000000
    volume_changes = np.random.lognormal(0, 0.3, 300)
    volumes = base_volume * volume_changes

    # 创建数据框
    market_data = pd.DataFrame({
        'date': dates,
        'close': prices[1:],
        'volume': volumes
    }).set_index('date')

    # 创建GAN市场合成器
    gms = GANMarketSynthesizer(
        latent_dim=50,
        hidden_dims=[128, 256, 128],
        epochs=100,
        batch_size=16
    )

    # 分析市场
    print("使用GAN分析市场...")
    gan_results = gms.analyze_market_with_gan(market_data)

    # 显示主要结果
    print(f"最新异常分数: {gan_results['anomaly_scores'].iloc[-1]:.3f}")
    print(f"正常性得分: {gan_results['normality_score'].iloc[-1]:.3f}")
    print(f"异常阈值: {gan_results['anomaly_threshold'].iloc[-1]:.3f}")
    print(f"是否异常: {gan_results['is_anomaly'].iloc[-1]}")
    print(f"当前信号: {gan_results['signals'].iloc[-1]}")

    if 'model_uncertainty' in gan_results:
        print(f"模型不确定性: {gan_results['model_uncertainty'].iloc[-1]:.3f}")

    # 显示合成数据统计
    if 'synthetic_data_sample' in gan_results:
        synthetic_stats = gan_results['synthetic_data_sample'].describe()
        print(f"\n合成数据统计:")
        print(synthetic_stats.loc[['mean', 'std']])

    # 获取交易建议
    recommendations = gms.get_trading_recommendations(gan_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['anomaly_scores', 'normality_score', 'market_state',
                                          'signal_description', 'strategy', 'position_size', 'risk_level']])