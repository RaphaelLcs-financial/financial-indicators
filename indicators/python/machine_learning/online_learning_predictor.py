"""
在线学习预测器金融指标

本模块实现了基于在线学习的自适应金融指标系统，能够实时更新模型并适应市场变化。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from collections import deque
import warnings
warnings.filterwarnings('ignore')

# 尝试导入在线学习库
try:
    from river import linear_model, metrics, preprocessing, compose, drift
    from river import ensemble, tree, naive_bayes, neighbors
    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

# 如果没有River，使用sklearn的partial_fit
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler


class OnlineLearningPredictor:
    """
    在线学习预测器

    使用增量学习方法实时更新预测模型，适应市场变化
    """

    def __init__(self,
                 window_size: int = 100,
                 learning_rate: float = 0.01,
                 update_frequency: int = 5,
                 max_memory: int = 1000):
        """
        初始化在线学习预测器

        Args:
            window_size: 特征计算窗口
            learning_rate: 学习率
            update_frequency: 更新频率
            max_memory: 最大记忆长度
        """
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.max_memory = max_memory

        # 特征和目标记忆
        self.feature_memory = deque(maxlen=max_memory)
        self.target_memory = deque(maxlen=max_memory)
        self.timestamp_memory = deque(maxlen=max_memory)

        # 模型初始化
        self._initialize_models()

        # 性能跟踪
        self.prediction_errors = deque(maxlen=100)
        self.cumulative_return = 0.0
        self.trades_count = 0

        # 概念漂移检测
        self.drift_detector = self._initialize_drift_detector()

        # 自适应参数
        self.current_learning_rate = learning_rate
        self.confidence_threshold = 0.6

    def _initialize_models(self):
        """初始化在线学习模型"""
        if RIVER_AVAILABLE:
            # 使用River在线学习库
            self.models = {
                'linear': linear_model.LinearRegression(
                    optimizer=linear_model.SGD(lr=self.learning_rate)
                ),
                'passive_aggressive': linear_model.PassiveAggressiveRegressor(
                    C=1.0, mode='regression'
                ),
                'hoeffding_tree': tree.HoeffdingTreeRegressor(
                    grace_period=100, max_depth=10
                ),
                'knn': neighbors.KNNRegressor(n_neighbors=5, window_size=100),
                'adaptive_random_forest': ensemble.AdaptiveRandomForestRegressor(
                    n_models=10, seed=42
                )
            }

            # 预处理管道
            self.preprocessor = compose.Pipeline(
                preprocessing.StandardScaler(),
                preprocessing.PolynomialExtender(degree=2, include_bias=False)
            )

            # 模型权重（自适应集成）
            self.model_weights = {name: 1.0/len(self.models) for name in self.models.keys()}

        else:
            # 使用sklearn的增量学习
            self.models = {
                'sgd': SGDRegressor(learning_rate='constant', eta0=self.learning_rate),
                'rf': RandomForestRegressor(n_estimators=50, warm_start=True, random_state=42)
            }

            self.preprocessor = StandardScaler()
            self.model_weights = {'sgd': 0.6, 'rf': 0.4}

        # 特征选择器
        self.feature_importance = {}
        self.selected_features = []

    def _initialize_drift_detector(self):
        """初始化概念漂移检测器"""
        if RIVER_AVAILABLE:
            return drift.ADWIN(delta=0.002)
        else:
            return {
                'error_window': deque(maxlen=50),
                'mean_error': 0.0,
                'drift_detected': False
            }

    def _create_online_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        创建在线学习特征

        Args:
            data: 市场数据

        Returns:
            特征DataFrame
        """
        features = pd.DataFrame(index=data.index)

        # 价格特征
        features['returns'] = data['close'].pct_change()
        features['log_returns'] = np.log(data['close'] / data['close'].shift(1))

        # 在线移动平均
        features['ema_5'] = data['close'].ewm(span=5).mean()
        features['ema_20'] = data['close'].ewm(span=20).mean()
        features['ema_ratio'] = features['ema_5'] / features['ema_20']

        # 在线波动率
        features['volatility_10'] = features['returns'].rolling(10).std()
        features['volatility_30'] = features['returns'].rolling(30).std()
        features['volatility_ratio'] = features['volatility_10'] / features['volatility_30']

        # 在线动量指标
        features['momentum_5'] = data['close'].pct_change(5)
        features['momentum_10'] = data['close'].pct_change(10)
        features['momentum_divergence'] = features['momentum_5'] - features['momentum_10']

        # 在线RSI
        features['rsi_online'] = self._calculate_online_rsi(data['close'])

        # 在线MACD
        features['macd_online'], features['macd_signal_online'] = self._calculate_online_macd(data['close'])

        # 在线成交量指标
        features['volume_ema'] = data['volume'].ewm(span=20).mean()
        features['volume_ratio'] = data['volume'] / features['volume_ema']
        features['volume_momentum'] = features['volume_ratio'].pct_change(5)

        # 价格效率
        features['price_efficiency'] = abs(features['returns']) / (features['volatility_10'] + 1e-8)

        # 在线相关性（如果有多资产）
        if hasattr(data, 'close_related'):
            features['correlation_online'] = data['close'].rolling(20).corr(data['close_related'])

        # 在线统计特征
        features['skewness_20'] = features['returns'].rolling(20).skew()
        features['kurtosis_20'] = features['returns'].rolling(20).kurt()
        features['percentile_80'] = features['returns'].rolling(20).quantile(0.8)
        features['percentile_20'] = features['returns'].rolling(20).quantile(0.2)

        # 在线趋势强度
        features['trend_strength'] = abs(features['ema_5'].pct_change())

        return features

    def _calculate_online_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算在线RSI"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # 使用指数移动平均代替简单移动平均
        avg_gain = gain.ewm(span=period).mean()
        avg_loss = loss.ewm(span=period).mean()

        rs = avg_gain / (avg_loss + 1e-8)
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def _calculate_online_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series]:
        """计算在线MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()

        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()

        return macd, signal_line

    def update_model(self, features: pd.Series, target: float, timestamp: pd.Timestamp) -> Dict[str, Any]:
        """
        更新在线学习模型

        Args:
            features: 当前特征
            target: 目标值
            timestamp: 时间戳

        Returns:
            更新结果
        """
        # 存储新数据
        self.feature_memory.append(features.to_dict())
        self.target_memory.append(target)
        self.timestamp_memory.append(timestamp)

        update_results = {}

        if RIVER_AVAILABLE:
            # 使用River在线学习
            for model_name, model in self.models.items():
                try:
                    # 预处理特征
                    feature_dict = features.to_dict()
                    processed_features = self.preprocessor.transform_one(feature_dict)

                    # 增量学习
                    y_pred = model.predict_one(processed_features)
                    model.learn_one(processed_features, target)

                    # 计算误差
                    error = abs(y_pred - target) if y_pred is not None else float('inf')

                    update_results[model_name] = {
                        'prediction': y_pred,
                        'error': error,
                        'status': 'updated'
                    }

                except Exception as e:
                    update_results[model_name] = {
                        'error': str(e),
                        'status': 'failed'
                    }

        else:
            # 使用sklearn增量学习
            X = np.array([features.values])
            y = np.array([target])

            # SGD更新
            try:
                sgd_pred = self.models['sgd'].predict(X)[0]
                self.models['sgd'].partial_fit(X, y)
                sgd_error = abs(sgd_pred - target)

                update_results['sgd'] = {
                    'prediction': sgd_pred,
                    'error': sgd_error,
                    'status': 'updated'
                }
            except Exception as e:
                update_results['sgd'] = {'error': str(e), 'status': 'failed'}

            # 随机森林更新（需要累积一定数据）
            if len(self.feature_memory) % 20 == 0:  # 每20个样本更新一次
                try:
                    X_batch = np.array([list(f.values()) for f in list(self.feature_memory)[-20:]])
                    y_batch = np.array(list(self.target_memory)[-20:])

                    rf_pred = self.models['rf'].predict(X_batch[-1:])[0]
                    self.models['rf'].fit(X_batch, y_batch)
                    rf_error = abs(rf_pred - y_batch[-1])

                    update_results['rf'] = {
                        'prediction': rf_pred,
                        'error': rf_error,
                        'status': 'updated'
                    }
                except Exception as e:
                    update_results['rf'] = {'error': str(e), 'status': 'not_updated'}

        # 检测概念漂移
        self._detect_concept_drift(update_results)

        # 自适应调整模型权重
        self._adapt_model_weights(update_results)

        return update_results

    def _detect_concept_drift(self, update_results: Dict[str, Any]):
        """检测概念漂移"""
        if RIVER_AVAILABLE:
            # 使用River的漂移检测
            for model_name, result in update_results.items():
                if 'error' in result and result['error'] != float('inf'):
                    self.drift_detector.update(result['error'])
                    if self.drift_detector.drift_detected:
                        # 检测到漂移，重置模型
                        self._reset_model_on_drift(model_name)
                        break
        else:
            # 简单的误差窗口检测
            for model_name, result in update_results.items():
                if 'error' in result:
                    self.drift_detector['error_window'].append(result['error'])

                    if len(self.drift_detector['error_window']) >= 50:
                        current_mean = np.mean(list(self.drift_detector['error_window'])[-25:])
                        historical_mean = np.mean(list(self.drift_detector['error_window'])[:-25])

                        if current_mean > historical_mean * 1.5:  # 误差增加50%
                            self.drift_detector['drift_detected'] = True
                            self._reset_model_on_drift(model_name)
                            self.drift_detector['error_window'].clear()
                            break

    def _reset_model_on_drift(self, model_name: str):
        """在检测到漂移时重置模型"""
        print(f"概念漂移检测！重置模型: {model_name}")

        if RIVER_AVAILABLE:
            # River模型通常自适应能力较强，可以调整学习率
            self.current_learning_rate *= 1.5  # 提高学习率适应新环境
            if hasattr(self.models[model_name], 'optimizer'):
                self.models[model_name].optimizer.lr = self.current_learning_rate
        else:
            # 重置sklearn模型
            if model_name == 'sgd':
                self.models['sgd'] = SGDRegressor(learning_rate='constant', eta0=self.current_learning_rate)
            elif model_name == 'rf':
                self.models['rf'] = RandomForestRegressor(n_estimators=50, warm_start=True, random_state=42)

    def _adapt_model_weights(self, update_results: Dict[str, Any]):
        """自适应调整模型权重"""
        errors = []
        model_names = []

        for model_name, result in update_results.items():
            if 'error' in result and result['error'] != float('inf'):
                errors.append(result['error'])
                model_names.append(model_name)

        if len(errors) > 1:
            # 基于误差调整权重
            weights = 1.0 / (np.array(errors) + 1e-8)
            weights = weights / np.sum(weights)

            # 平滑更新权重
            alpha = 0.1  # 平滑因子
            for i, model_name in enumerate(model_names):
                old_weight = self.model_weights.get(model_name, 1.0/len(self.models))
                new_weight = alpha * weights[i] + (1 - alpha) * old_weight
                self.model_weights[model_name] = new_weight

    def predict_online(self, features: pd.Series) -> Dict[str, Any]:
        """
        在线预测

        Args:
            features: 当前特征

        Returns:
            预测结果
        """
        predictions = {}
        confidence_scores = {}

        if RIVER_AVAILABLE:
            # River预测
            for model_name, model in self.models.items():
                try:
                    feature_dict = features.to_dict()
                    processed_features = self.preprocessor.transform_one(feature_dict)

                    pred = model.predict_one(processed_features)
                    predictions[model_name] = pred

                    # 置信度基于权重和预测稳定性
                    confidence_scores[model_name] = self.model_weights.get(model_name, 0.1)

                except Exception as e:
                    predictions[model_name] = None
                    confidence_scores[model_name] = 0

        else:
            # sklearn预测
            try:
                X = np.array([features.values])

                # SGD预测
                sgd_pred = self.models['sgd'].predict(X)[0]
                predictions['sgd'] = sgd_pred
                confidence_scores['sgd'] = self.model_weights.get('sgd', 0.6)

                # 随机森林预测
                if len(self.feature_memory) >= 20:
                    rf_pred = self.models['rf'].predict(X)[0]
                    predictions['rf'] = rf_pred
                    confidence_scores['rf'] = self.model_weights.get('rf', 0.4)

            except Exception as e:
                predictions['sgd'] = None
                predictions['rf'] = None
                confidence_scores = {'sgd': 0, 'rf': 0}

        # 集成预测
        ensemble_pred, ensemble_confidence = self._ensemble_prediction(predictions, confidence_scores)

        return {
            'individual_predictions': predictions,
            'ensemble_prediction': ensemble_pred,
            'ensemble_confidence': ensemble_confidence,
            'model_weights': self.model_weights,
            'concept_drift_status': getattr(self.drift_detector, 'drift_detected',
                                          self.drift_detector.get('drift_detected', False))
        }

    def _ensemble_prediction(self, predictions: Dict[str, float],
                           confidence_scores: Dict[str, float]) -> Tuple[float, float]:
        """集成预测"""
        valid_predictions = [(p, confidence_scores.get(m, 0))
                            for m, p in predictions.items()
                            if p is not None and confidence_scores.get(m, 0) > 0]

        if not valid_predictions:
            return 0.0, 0.0

        preds, weights = zip(*valid_predictions)
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # 归一化权重

        # 加权平均预测
        ensemble_pred = np.average(preds, weights=weights)

        # 计算集成置信度
        prediction_variance = np.var(preds)
        weight_consensus = np.mean(weights) * 4  # 权重一致性
        ensemble_confidence = min(1.0, (1.0 - prediction_variance) * weight_consensus)

        return ensemble_pred, ensemble_confidence

    def calculate_online_indicator(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        计算在线学习指标

        Args:
            data: 市场数据

        Returns:
            在线学习指标DataFrame
        """
        # 创建特征
        features = self._create_online_features(data)

        if len(features) < self.window_size:
            return pd.DataFrame()

        # 获取最新特征
        latest_features = features.iloc[-1]

        # 在线预测
        prediction_result = self.predict_online(latest_features)

        # 创建指标
        indicators = pd.DataFrame(index=data.index[-1:], columns=[
            'online_prediction', 'online_confidence', 'concept_drift_flag',
            'adaptation_rate', 'model_diversity', 'online_signal'
        ])

        # 填充指标值
        indicators['online_prediction'] = prediction_result['ensemble_prediction']
        indicators['online_confidence'] = prediction_result['ensemble_confidence']
        indicators['concept_drift_flag'] = int(prediction_result['concept_drift_status'])
        indicators['adaptation_rate'] = self.current_learning_rate
        indicators['model_diversity'] = self._calculate_model_diversity(prediction_result['individual_predictions'])

        # 生成交易信号
        current_price = data['close'].iloc[-1]
        predicted_change = prediction_result['ensemble_prediction']
        confidence = prediction_result['ensemble_confidence']

        if confidence > 0.7:
            if predicted_change > 0.01:  # 预测上涨超过1%
                indicators['online_signal'] = 2  # 强烈买入
            elif predicted_change < -0.01:  # 预测下跌超过1%
                indicators['online_signal'] = -2  # 强烈卖出
            else:
                indicators['online_signal'] = 0  # 持有
        elif confidence > 0.5:
            if predicted_change > 0.005:
                indicators['online_signal'] = 1  # 买入
            elif predicted_change < -0.005:
                indicators['online_signal'] = -1  # 卖出
            else:
                indicators['online_signal'] = 0  # 持有
        else:
            indicators['online_signal'] = 0  # 低置信度，持有

        return indicators

    def _calculate_model_diversity(self, predictions: Dict[str, float]) -> float:
        """计算模型多样性"""
        valid_preds = [p for p in predictions.values() if p is not None]

        if len(valid_preds) < 2:
            return 0.0

        # 计算预测的标准差作为多样性指标
        diversity = np.std(valid_preds)
        return min(1.0, diversity)  # 归一化到[0,1]

    def get_learning_statistics(self) -> Dict[str, Any]:
        """获取学习统计信息"""
        return {
            'samples_processed': len(self.feature_memory),
            'current_learning_rate': self.current_learning_rate,
            'model_weights': self.model_weights,
            'concept_drift_count': getattr(self.drift_detector, 'n_detections', 0),
            'average_prediction_error': np.mean(self.prediction_errors) if self.prediction_errors else 0,
            'cumulative_return': self.cumulative_return,
            'trades_executed': self.trades_count
        }


class AdaptiveOnlineEnsemble:
    """
    自适应在线集成

    动态管理多个在线学习模型的自适应集成系统
    """

    def __init__(self,
                 max_models: int = 10,
                 performance_window: int = 50,
                 removal_threshold: float = 0.3):
        """
        初始化自适应在线集成

        Args:
            max_models: 最大模型数量
            performance_window: 性能评估窗口
            removal_threshold: 模型移除阈值
        """
        self.max_models = max_models
        self.performance_window = performance_window
        self.removal_threshold = removal_threshold

        # 模型池
        self.model_pool = {}
        self.model_performance = {}
        self.model_creation_time = {}

        # 集成管理
        self.current_ensemble = []
        self.ensemble_weights = {}

        # 自适应参数
        self.addition_threshold = 0.6
        self.replacement_threshold = 0.4

    def add_model(self, model_name: str, model_config: Dict[str, Any]):
        """添加新模型"""
        if len(self.model_pool) >= self.max_models:
            # 移除性能最差的模型
            self._remove_worst_model()

        # 创建新模型
        new_model = OnlineLearningPredictor(**model_config)
        self.model_pool[model_name] = new_model
        self.model_performance[model_name] = []
        self.model_creation_time[model_name] = pd.Timestamp.now()

        print(f"添加新模型: {model_name}")

    def _remove_worst_model(self):
        """移除性能最差的模型"""
        if not self.model_performance:
            return

        # 计算平均性能
        avg_performance = {}
        for model_name, performance_history in self.model_performance.items():
            if performance_history:
                avg_performance[model_name] = np.mean(performance_history[-self.performance_window:])
            else:
                avg_performance[model_name] = 0

        # 移除性能最差的模型
        worst_model = min(avg_performance, key=avg_performance.get)
        if avg_performance[worst_model] < self.removal_threshold:
            del self.model_pool[worst_model]
            del self.model_performance[worst_model]
            del self.model_creation_time[worst_model]

            print(f"移除低性能模型: {worst_model}")

    def update_ensemble(self, features: pd.Series, target: float, timestamp: pd.Timestamp):
        """更新集成"""
        # 更新每个模型
        for model_name, model in self.model_pool.items():
            try:
                update_result = model.update_model(features, target, timestamp)

                # 记录性能
                if 'error' in update_result:
                    self.model_performance[model_name].append(1.0 / (1.0 + update_result['error']))

                    # 保持性能历史长度
                    if len(self.model_performance[model_name]) > self.performance_window:
                        self.model_performance[model_name] = self.model_performance[model_name][-self.performance_window:]

            except Exception as e:
                print(f"模型 {model_name} 更新失败: {e}")

        # 更新集成成员
        self._update_ensemble_members()

        # 更新集成权重
        self._update_ensemble_weights()

    def _update_ensemble_members(self):
        """更新集成成员"""
        # 计算每个模型的性能得分
        performance_scores = {}
        for model_name, performance_history in self.model_performance.items():
            if len(performance_history) >= 10:  # 至少10个样本
                recent_performance = np.mean(performance_history[-20:])
                performance_scores[model_name] = recent_performance

        # 选择性能最好的模型
        sorted_models = sorted(performance_scores.items(), key=lambda x: x[1], reverse=True)
        self.current_ensemble = [model[0] for model in sorted_models[:5]]  # 选择前5个模型

        # 考虑添加新模型
        self._consider_model_addition(performance_scores)

    def _consider_model_addition(self, current_scores: Dict[str, float]):
        """考虑添加新模型"""
        if len(self.current_ensemble) < 3 and len(self.model_pool) > len(self.current_ensemble):
            # 添加性能超过阈值的模型
            for model_name, score in current_scores.items():
                if model_name not in self.current_ensemble and score > self.addition_threshold:
                    self.current_ensemble.append(model_name)
                    if len(self.current_ensemble) >= 5:
                        break

    def _update_ensemble_weights(self):
        """更新集成权重"""
        if not self.current_ensemble:
            return

        # 计算权重
        weights = {}
        total_weight = 0

        for model_name in self.current_ensemble:
            if model_name in self.model_performance and self.model_performance[model_name]:
                # 基于近期性能计算权重
                recent_performance = np.mean(self.model_performance[model_name][-10:])
                weight = max(0.1, recent_performance)  # 最小权重0.1
                weights[model_name] = weight
                total_weight += weight

        # 归一化权重
        if total_weight > 0:
            for model_name in weights:
                weights[model_name] /= total_weight

        self.ensemble_weights = weights

    def predict(self, features: pd.Series) -> Dict[str, Any]:
        """集成预测"""
        if not self.current_ensemble:
            return {'error': 'No active models in ensemble'}

        predictions = {}
        individual_predictions = {}

        # 获取每个模型的预测
        for model_name in self.current_ensemble:
            if model_name in self.model_pool:
                try:
                    model_prediction = self.model_pool[model_name].predict_online(features)
                    pred = model_prediction['ensemble_prediction']
                    confidence = model_prediction['ensemble_confidence']

                    predictions[model_name] = pred
                    individual_predictions[model_name] = {
                        'prediction': pred,
                        'confidence': confidence
                    }

                except Exception as e:
                    print(f"模型 {model_name} 预测失败: {e}")

        # 计算集成预测
        if predictions:
            ensemble_pred = np.average(list(predictions.values()),
                                     weights=[self.ensemble_weights.get(m, 1.0/len(predictions))
                                            for m in predictions.keys()])

            # 计算集成置信度
            confidence_values = [ind_pred['confidence'] for ind_pred in individual_predictions.values()]
            ensemble_confidence = np.mean(confidence_values)

            # 计算预测一致性
            pred_variance = np.var(list(predictions.values()))
            consistency_score = 1.0 / (1.0 + pred_variance)

            final_confidence = ensemble_confidence * consistency_score
        else:
            ensemble_pred = 0.0
            final_confidence = 0.0

        return {
            'ensemble_prediction': ensemble_pred,
            'ensemble_confidence': final_confidence,
            'individual_predictions': individual_predictions,
            'ensemble_weights': self.ensemble_weights,
            'active_models': self.current_ensemble
        }


def create_online_learning_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    创建在线学习特征

    Args:
        data: 市场数据

    Returns:
        在线学习特征DataFrame
    """
    # 初始化在线学习预测器
    online_predictor = OnlineLearningPredictor()

    # 计算指标
    indicators = online_predictor.calculate_online_indicator(data)

    return indicators


# 主要功能函数
def calculate_online_learning_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有在线学习指标

    Args:
        data: 包含OHLCV数据的DataFrame

    Returns:
        包含所有指标值的DataFrame
    """
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("数据必须包含 'open', 'high', 'low', 'close', 'volume' 列")

    if len(data) < 100:
        raise ValueError("数据长度不足，至少需要100个数据点")

    return create_online_learning_features(data)


# 示例使用
if __name__ == "__main__":
    # 生成示例数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

    # 模拟价格数据
    initial_price = 100
    returns = np.random.normal(0.001, 0.02, 200)
    prices = [initial_price]

    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    sample_data = pd.DataFrame({
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 200)
    }, index=dates)

    # 计算指标
    try:
        indicators = calculate_online_learning_indicators(sample_data)
        print("在线学习指标计算成功!")
        print(f"指标数量: {indicators.shape[1]}")
        print("最新指标值:")
        print(indicators.iloc[-1])

        # 获取学习统计
        predictor = OnlineLearningPredictor()
        stats = predictor.get_learning_statistics()
        print("\n学习统计信息:")
        for key, value in stats.items():
            print(f"{key}: {value}")

    except Exception as e:
        print(f"计算错误: {e}")