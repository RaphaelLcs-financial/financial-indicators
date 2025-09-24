"""
Transformer Market Predictor
============================

基于Transformer架构的深度学习市场预测指标，利用自注意力机制
捕捉市场数据中的长期依赖关系和时间模式。该指标结合了
现代深度学习技术，能够处理复杂的市场动态和多变量关系。

特点：
- 自注意力机制
- 多头注意力
- 位置编码
- 层归一化
- 残差连接
- 编码器-解码器结构
- 多变量处理
- 长期依赖建模

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
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


class PositionalEncoding(nn.Module):
    """位置编码层"""
    def __init__(self, d_model: int, max_len: int = 1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=0.1)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.attention = None

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)

        # 线性变换
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)

        # 应用mask（如果有的话）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 计算注意力权重
        attention_weights = torch.softmax(scores, dim=-1)
        self.attention = attention_weights

        # 应用注意力权重
        context = torch.matmul(attention_weights, V)

        # 拼接多头
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)

        # 最终线性变换
        output = self.w_o(context)
        return self.dropout(output)


class PositionwiseFeedForward(nn.Module):
    """位置前馈网络"""
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(self.activation(self.w1(x))))


class TransformerLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super(TransformerLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 自注意力 + 残差连接 + 层归一化
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class MarketTransformer(nn.Module):
    """市场预测Transformer模型"""
    def __init__(self, input_dim: int, d_model: int = 256, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 1024, dropout: float = 0.1):
        super(MarketTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model

        # 输入嵌入
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

        # Transformer层
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # 输出层
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )

        # 预测层（多时间步预测）
        self.prediction_heads = nn.ModuleDict({
            'short_term': nn.Linear(d_model, 1),  # 1天预测
            'medium_term': nn.Linear(d_model, 1),  # 5天预测
            'long_term': nn.Linear(d_model, 1)   # 20天预测
        })

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, input_dim = x.size()

        # 输入嵌入
        x = self.input_embedding(x)
        x = x * np.sqrt(self.d_model)

        # 位置编码
        x = x.transpose(0, 1)  # (seq_len, batch, d_model)
        x = self.positional_encoding(x)
        x = x.transpose(0, 1)  # (batch, seq_len, d_model)

        # Transformer层
        for layer in self.transformer_layers:
            x = layer(x, mask)

        # 使用最后一个时间步的输出进行预测
        final_output = x[:, -1, :]

        # 多时间步预测
        predictions = {}
        for horizon, head in self.prediction_heads.items():
            predictions[horizon] = head(final_output)

        # 主要预测输出
        predictions['main'] = self.output_layer(final_output)

        return predictions

    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """获取注意力权重"""
        for layer in self.transformer_layers:
            if layer.self_attn.attention is not None:
                return layer.self_attn.attention
        return None


class TransformerMarketPredictor:
    """
    基于Transformer的市场预测器

    利用自注意力机制和深度学习技术
    进行市场趋势和价格预测。
    """

    def __init__(self, sequence_length: int = 50, d_model: int = 256,
                 n_heads: int = 8, n_layers: int = 4, batch_size: int = 32,
                 epochs: int = 100, learning_rate: float = 0.001):
        """
        初始化Transformer市场预测器

        Args:
            sequence_length: 序列长度，默认50
            d_model: 模型维度，默认256
            n_heads: 注意力头数，默认8
            n_layers: Transformer层数，默认4
            batch_size: 批次大小，默认32
            epochs: 训练轮数，默认100
            learning_rate: 学习率，默认0.001
        """
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.name = f"Transformer Market Predictor ({sequence_length})"
        self.category = "deep_learning"

        # 模型和设备
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

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
            features['sma_20'] = prices.rolling(window=20).mean()
            features['ema_12'] = prices.ewm(span=12).mean()
            features['ema_26'] = prices.ewm(span=26).mean()

            # RSI
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features['rsi'] = 100 - (100 / (1 + rs)).fillna(50)

            # MACD
            features['macd'] = features['ema_12'] - features['ema_26']
            features['macd_signal'] = features['macd'].ewm(span=9).mean()
            features['macd_histogram'] = features['macd'] - features['macd_signal']

            # 布林带
            sma_20 = prices.rolling(window=20).mean()
            std_20 = prices.rolling(window=20).std()
            features['bollinger_upper'] = sma_20 + (std_20 * 2)
            features['bollinger_lower'] = sma_20 - (std_20 * 2)
            features['bollinger_position'] = (prices - features['bollinger_lower']) / (features['bollinger_upper'] - features['bollinger_lower'] + 1e-10)

        # 成交量特征
        if 'volume' in data.columns:
            volume = data['volume']
            features['volume'] = volume
            features['volume_sma'] = volume.rolling(window=20).mean()
            features['volume_ratio'] = volume / features['volume_sma']
            features['volume_change'] = volume.pct_change().fillna(0)

        # 波动率特征
        if 'close' in data.columns:
            returns = features['returns']
            features['volatility_5'] = returns.rolling(window=5).std()
            features['volatility_20'] = returns.rolling(window=20).std()
            features['volatility_ratio'] = features['volatility_5'] / (features['volatility_20'] + 1e-10)

        # 高低价特征（如果可用）
        if 'high' in data.columns and 'low' in data.columns:
            features['high'] = data['high']
            features['low'] = data['low']
            features['range'] = data['high'] - data['low']
            features['range_sma'] = features['range'].rolling(window=20).mean()
            features['range_position'] = (data['close'] - data['low']) / (features['range'] + 1e-10)

        # 清理数据
        features = features.fillna(method='ffill').fillna(0)

        return features

    def create_sequences(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建训练序列

        Args:
            features: 特征数据
            targets: 目标数据

        Returns:
            序列数据和目标
        """
        X, y = [], []

        for i in range(self.sequence_length, len(features)):
            X.append(features.iloc[i-self.sequence_length:i].values)
            y.append(targets.iloc[i])

        return np.array(X), np.array(y)

    def build_model(self, input_dim: int) -> MarketTransformer:
        """
        构建Transformer模型

        Args:
            input_dim: 输入维度

        Returns:
            Transformer模型
        """
        model = MarketTransformer(
            input_dim=input_dim,
            d_model=self.d_model,
            n_heads=self.n_heads,
            n_layers=self.n_layers,
            dropout=0.1
        ).to(self.device)

        return model

    def train_model(self, features: pd.DataFrame, targets: pd.Series) -> Dict[str, Any]:
        """
        训练模型

        Args:
            features: 特征数据
            targets: 目标数据

        Returns:
            训练历史
        """
        # 数据标准化
        features_scaled = self.scaler.fit_transform(features)
        targets_scaled = (targets - targets.mean()) / (targets.std() + 1e-10)

        # 创建序列
        X, y = self.create_sequences(pd.DataFrame(features_scaled, index=features.index), targets_scaled)

        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 转换为张量
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).unsqueeze(1).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.FloatTensor(y_val).unsqueeze(1).to(self.device)

        # 构建模型
        self.model = self.build_model(X_train.shape[2])

        # 训练设置
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)

        # 训练历史
        history = {'train_loss': [], 'val_loss': []}

        # 训练循环
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(self.epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0.0

            for i in range(0, len(X_train), self.batch_size):
                batch_X = X_train[i:i+self.batch_size]
                batch_y = y_train[i:i+self.batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs['main'], batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(X_train) // self.batch_size

            # 验证阶段
            self.model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for i in range(0, len(X_val), self.batch_size):
                    batch_X = X_val[i:i+self.batch_size]
                    batch_y = y_val[i:i+self.batch_size]

                    outputs = self.model(batch_X)
                    loss = criterion(outputs['main'], batch_y)
                    val_loss += loss.item()

            val_loss /= len(X_val) // self.batch_size

            # 记录历史
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            # 学习率调度
            scheduler.step(val_loss)

            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    break

            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        self.is_trained = True
        return history

    def predict(self, features: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        进行预测

        Args:
            features: 特征数据

        Returns:
            预测结果
        """
        if not self.is_trained or self.model is None:
            return {}

        # 数据标准化
        features_scaled = self.scaler.transform(features)

        # 创建序列
        X = []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])

        if len(X) == 0:
            return {}

        X = torch.FloatTensor(np.array(X)).to(self.device)

        # 预测
        self.model.eval()
        predictions = {}

        with torch.no_grad():
            outputs = self.model(X)

            # 解码预测结果
            for horizon, pred in outputs.items():
                pred_values = pred.cpu().numpy().flatten()

                # 创建对应的索引
                pred_index = features.index[self.sequence_length:]

                predictions[horizon] = pd.Series(pred_values, index=pred_index)

        return predictions

    def calculate_attention_analysis(self, features: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        注意力分析

        Args:
            features: 特征数据

        Returns:
            注意力分析结果
        """
        if not self.is_trained or self.model is None:
            return {}

        # 数据标准化
        features_scaled = self.scaler.transform(features)

        # 创建序列
        X = []
        for i in range(self.sequence_length, len(features_scaled)):
            X.append(features_scaled[i-self.sequence_length:i])

        if len(X) == 0:
            return {}

        X = torch.FloatTensor(np.array(X)).to(self.device)

        # 获取注意力权重
        self.model.eval()
        attention_weights = []

        with torch.no_grad():
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i+self.batch_size]
                self.model(batch_X)

                # 获取注意力权重
                weights = self.model.get_attention_weights()
                if weights is not None:
                    attention_weights.append(weights.cpu().numpy())

        if not attention_weights:
            return {}

        # 分析注意力权重
        attention_analysis = {}

        # 平均注意力权重
        avg_attention = np.mean(attention_weights, axis=0)
        attention_df = pd.DataFrame(avg_attention[0],  # 取第一个样本
                                  index=[f'Feature_{i}' for i in range(avg_attention.shape[1])],
                                  columns=[f'Time_{i}' for i in range(avg_attention.shape[2])])

        attention_analysis['average_attention'] = attention_df

        # 注意力集中度
        attention_concentration = np.std(avg_attention, axis=2)
        concentration_series = pd.Series(attention_concentration[0],
                                        index=[f'Feature_{i}' for i in range(avg_attention.shape[1])])

        attention_analysis['attention_concentration'] = concentration_series

        # 时间步重要性
        time_importance = np.mean(avg_attention, axis=1)
        time_series = pd.Series(time_importance[0],
                               index=[f'Time_{i}' for i in range(avg_attention.shape[2])])

        attention_analysis['time_importance'] = time_series

        return attention_analysis

    def calculate_feature_importance(self, features: pd.DataFrame) -> pd.Series:
        """
        计算特征重要性

        Args:
            features: 特征数据

        Returns:
            特征重要性
        """
        attention_analysis = self.calculate_attention_analysis(features)

        if 'attention_concentration' in attention_analysis:
            return attention_analysis['attention_concentration']
        else:
            return pd.Series()

    def analyze_market_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        分析市场模式

        Args:
            data: 市场数据

        Returns:
            市场模式分析结果
        """
        results = {}

        # 准备特征
        features = self.prepare_features(data)

        # 目标变量（价格收益率）
        if 'close' in data.columns:
            targets = data['close'].pct_change().shift(-1).fillna(0)  # 预测下一个周期的收益率
        else:
            targets = pd.Series(0, index=data.index)

        # 训练模型（如果尚未训练）
        if not self.is_trained:
            print("训练Transformer模型...")
            history = self.train_model(features, targets)
            results['training_history'] = history

        # 进行预测
        predictions = self.predict(features)
        results.update(predictions)

        # 注意力分析
        attention_analysis = self.calculate_attention_analysis(features)
        results.update(attention_analysis)

        # 特征重要性
        feature_importance = self.calculate_feature_importance(features)
        results['feature_importance'] = feature_importance

        # 生成交易信号
        results['signals'] = self.generate_transformer_signals(results)

        return results

    def generate_transformer_signals(self, transformer_analysis: Dict[str, Any]) -> pd.Series:
        """
        基于Transformer分析生成交易信号

        Args:
            transformer_analysis: Transformer分析结果

        Returns:
            交易信号
        """
        if 'main' not in transformer_analysis:
            return pd.Series()

        predictions = transformer_analysis['main']
        signals = pd.Series(0, index=predictions.index)

        # 预测强度
        pred_strength = predictions.abs()
        pred_direction = np.sign(predictions)

        # 多时间步预测确认
        confirmations = []
        for horizon in ['short_term', 'medium_term', 'long_term']:
            if horizon in transformer_analysis:
                horizon_pred = transformer_analysis[horizon]
                if len(horizon_pred) > 0:
                    # 对齐索引
                    aligned_pred = horizon_pred.reindex(predictions.index).fillna(0)
                    horizon_direction = np.sign(aligned_pred)
                    confirmations.append(horizon_direction == pred_direction)

        # 计算确认强度
        if confirmations:
            confirmation_strength = sum(confirmations) / len(confirmations)
        else:
            confirmation_strength = 1.0

        # 信号阈值
        strong_threshold = predictions.std() * 1.5
        moderate_threshold = predictions.std() * 1.0

        # 生成信号
        strong_buy = (pred_direction > 0) & (pred_strength > strong_threshold) & (confirmation_strength > 0.6)
        buy = (pred_direction > 0) & (pred_strength > moderate_threshold) & (confirmation_strength > 0.4)
        strong_sell = (pred_direction < 0) & (pred_strength > strong_threshold) & (confirmation_strength > 0.6)
        sell = (pred_direction < 0) & (pred_strength > moderate_threshold) & (confirmation_strength > 0.4)

        # 分配信号值
        signals[strong_buy] = 3
        signals[buy] = 2
        signals[strong_sell] = -3
        signals[sell] = -2

        return signals

    def get_trading_recommendations(self, transformer_analysis: Dict[str, Any]) -> pd.DataFrame:
        """
        获取交易建议

        Args:
            transformer_analysis: Transformer分析结果

        Returns:
            交易建议DataFrame
        """
        if 'main' not in transformer_analysis:
            return pd.DataFrame()

        predictions = transformer_analysis['main']
        index = predictions.index
        recommendations = pd.DataFrame(index=index)

        # 添加Transformer分析指标
        for key, value in transformer_analysis.items():
            if isinstance(value, pd.Series) and len(value) > 0:
                recommendations[key] = value

        # 交易信号
        recommendations['signals'] = transformer_analysis.get('signals', pd.Series(0, index=index))

        # 信号描述
        signal_descriptions = {
            3: '强烈买入 - Transformer预测强烈上涨',
            2: '买入 - Transformer预测上涨',
            0: '持有 - Transformer预测中性',
            -2: '卖出 - Transformer预测下跌',
            -3: '强烈卖出 - Transformer预测强烈下跌'
        }
        recommendations['signal_description'] = recommendations['signals'].map(signal_descriptions)

        # 预测置信度
        predictions_std = predictions.std()
        prediction_confidence = recommendations['main'].abs() / predictions_std
        recommendations['prediction_confidence'] = prediction_confidence.fillna(0)

        # 多时间步预测一致性
        time_step_consistency = self.calculate_time_step_consistency(recommendations)
        recommendations['time_step_consistency'] = time_step_consistency.fillna(0)

        # 模型确定性分类
        def classify_model_certainty(row):
            confidence = row.get('prediction_confidence', 0)
            consistency = row.get('time_step_consistency', 0)

            if confidence > 1.5 and consistency > 0.7:
                return 'high_certainty'
            elif confidence > 1.0 and consistency > 0.5:
                return 'medium_certainty'
            else:
                return 'low_certainty'

        recommendations['model_certainty'] = recommendations.apply(classify_model_certainty, axis=1)

        # 仓位建议
        position_map = {3: 0.6, 2: 0.4, 0: 0.2, -2: 0.1, -3: 0.0}
        recommendations['position_size'] = recommendations['signals'].map(position_map)

        # 置信度调整
        confidence_adjustment = recommendation['prediction_confidence'] / (recommendation['prediction_confidence'].max() + 1e-10)
        recommendations['position_size'] *= confidence_adjustment

        # 一致性调整
        consistency_adjustment = recommendations['time_step_consistency']
        recommendations['position_size'] *= consistency_adjustment

        # 确保仓位在合理范围内
        recommendations['position_size'] = recommendations['position_size'].clip(0, 1)

        # 策略建议
        def get_transformer_strategy(certainty, signal_strength):
            if certainty == 'high_certainty':
                return 'aggressive_transformer'  # 激进Transformer策略
            elif certainty == 'medium_certainty':
                return 'moderate_transformer'    # 适度Transformer策略
            else:
                return 'conservative_transformer'  # 保守Transformer策略

        recommendations['strategy'] = recommendations.apply(
            lambda x: get_transformer_strategy(x['model_certainty'], abs(x['signals'])),
            axis=1
        )

        # 模型风险评估
        def assess_model_risk(row):
            risk_score = 0

            # 预测风险
            if row.get('prediction_confidence', 0) < 0.5:
                risk_score += 0.4

            # 一致性风险
            if row.get('time_step_consistency', 0) < 0.3:
                risk_score += 0.3

            # 预测强度风险
            if abs(row.get('main', 0)) > row.get('main', 0).std() * 2:
                risk_score += 0.3

            if risk_score > 0.6:
                return 'high'
            elif risk_score > 0.3:
                return 'medium'
            else:
                return 'low'

        recommendations['risk_level'] = recommendations.apply(assess_model_risk, axis=1)

        return recommendations

    def calculate_time_step_consistency(self, recommendations: pd.DataFrame) -> pd.Series:
        """
        计算多时间步预测一致性

        Args:
            recommendations: 推荐数据框

        Returns:
            一致性序列
        """
        time_steps = ['short_term', 'medium_term', 'long_term']
        available_steps = [step for step in time_steps if step in recommendations.columns]

        if len(available_steps) < 2:
            return pd.Series(1.0, index=recommendations.index)

        # 计算预测方向一致性
        directions = []
        for step in available_steps:
            direction = np.sign(recommendations[step])
            directions.append(direction)

        # 计算一致性
        consistency = pd.Series(1.0, index=recommendations.index)
        for i in range(len(directions) - 1):
            for j in range(i + 1, len(directions)):
                step_consistency = (directions[i] == directions[j]).astype(float)
                consistency *= step_consistency

        return consistency.fillna(0)


# 使用示例
if __name__ == "__main__":
    # 生成模拟市场数据
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', periods=300, freq='D')

    # 生成价格序列
    base_price = 100
    returns = np.random.randn(300) * 0.02

    # 添加趋势和周期性
    trend = np.linspace(0, 0.3, 300)
    seasonality = np.sin(np.linspace(0, 4*np.pi, 300)) * 0.01
    returns += trend + seasonality

    # 构建价格
    prices = [base_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))

    # 生成成交量
    base_volume = 1000000
    volume_changes = np.random.lognormal(0, 0.2, 300)
    volumes = base_volume * volume_changes

    # 生成高低价
    high_prices = [p * (1 + abs(np.random.randn() * 0.01)) for p in prices[1:]]
    low_prices = [p * (1 - abs(np.random.randn() * 0.01)) for p in prices[1:]]

    # 创建数据框
    market_data = pd.DataFrame({
        'date': dates,
        'open': prices[1:],
        'high': high_prices,
        'low': low_prices,
        'close': prices[1:],
        'volume': volumes
    }).set_index('date')

    # 创建Transformer市场预测器
    tmp = TransformerMarketPredictor(
        sequence_length=30,
        d_model=128,
        n_heads=4,
        n_layers=2,
        epochs=50,
        batch_size=16
    )

    # 分析市场模式
    print("分析市场模式...")
    transformer_results = tmp.analyze_market_patterns(market_data)

    # 显示主要结果
    if 'main' in transformer_results:
        print(f"最新预测值: {transformer_results['main'].iloc[-1]:.6f}")
        print(f"预测置信度: {transformer_results['main'].abs().iloc[-1] / transformer_results['main'].std():.3f}")

    if 'short_term' in transformer_results:
        print(f"短期预测: {transformer_results['short_term'].iloc[-1]:.6f}")
        print(f"中期预测: {transformer_results['medium_term'].iloc[-1]:.6f}")
        print(f"长期预测: {transformer_results['long_term'].iloc[-1]:.6f}")

    print(f"当前信号: {transformer_results['signals'].iloc[-1]}")

    # 显示特征重要性
    if 'feature_importance' in transformer_results and not transformer_results['feature_importance'].empty:
        print("\n重要特征:")
        top_features = transformer_results['feature_importance'].nlargest(5)
        for feature, importance in top_features.items():
            print(f"  {feature}: {importance:.3f}")

    # 获取交易建议
    recommendations = tmp.get_trading_recommendations(transformer_results)
    if not recommendations.empty:
        print("\n最新交易建议:")
        print(recommendations.iloc[-1][['main', 'prediction_confidence', 'time_step_consistency',
                                          'model_certainty', 'signal_description', 'strategy', 'position_size', 'risk_level']])