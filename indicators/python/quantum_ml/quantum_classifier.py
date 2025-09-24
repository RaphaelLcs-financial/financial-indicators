"""
量子分类器交易指标
Quantum Classifier Trading Indicator

基于量子支持向量机和量子神经网络的市场分类预测指标
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    from qiskit import QuantumCircuit, Aer, execute
    from qiskit.circuit import Parameter
    from qiskit_machine_learning.algorithms import QSVC
    from qiskit_machine_learning.kernels import QuantumKernel
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False

class QuantumClassifier:
    """
    量子分类器交易指标

    结合量子计算和机器学习的市场预测器
    支持量子SVM和量子神经网络方法
    """

    def __init__(self, n_qubits: int = 4, backend_type: str = 'simulator'):
        """
        初始化量子分类器

        Args:
            n_qubits: 量子比特数量
            backend_type: 后端类型 ('simulator', 'real')
        """
        self.n_qubits = n_qubits
        self.backend_type = backend_type
        self.scaler = StandardScaler()

        # 传统机器学习模型作为基准
        self.classical_svm = SVC(kernel='rbf', probability=True)
        self.classical_rf = RandomForestClassifier(n_estimators=100, random_state=42)

        # 量子模型
        self.quantum_model = None
        self.quantum_kernel = None

        # 训练状态
        self.is_trained = False
        self.feature_names = []

        # 模型性能
        self.performance_metrics = {}

    def _create_quantum_circuit(self, n_features: int) -> QuantumCircuit:
        """创建量子电路"""
        if not QISKIT_AVAILABLE:
            return None

        # 创建参数化量子电路
        qc = QuantumCircuit(self.n_qubits)

        # 特征映射
        params = [Parameter(f'θ{i}') for i in range(min(n_features, self.n_qubits * 2))]

        # Hadamard门创建叠加态
        for i in range(self.n_qubits):
            qc.h(i)

        # 特征编码
        for i in range(min(len(params), self.n_qubits)):
            qc.rz(params[i], i % self.n_qubits)

        # 纠缠层
        for i in range(self.n_qubits - 1):
            qc.cx(i, i + 1)

        # 变分层
        for i in range(len(params)):
            if i < self.n_qubits:
                qc.ry(params[i], i)

        return qc

    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """准备特征数据"""
        # 技术指标特征
        features = []

        # 价格特征
        features.append(data['close'].pct_change().fillna(0))  # 收益率
        features.append(data['close'].rolling(5).mean().pct_change().fillna(0))  # 短期趋势
        features.append(data['close'].rolling(20).mean().pct_change().fillna(0))  # 长期趋势

        # 波动率特征
        returns = data['close'].pct_change().fillna(0)
        features.append(returns.rolling(10).std().fillna(0))  # 短期波动率
        features.append(returns.rolling(30).std().fillna(0))  # 长期波动率

        # 成交量特征
        if 'volume' in data.columns:
            volume_change = data['volume'].pct_change().fillna(0)
            features.append(volume_change)
            features.append(data['volume'].rolling(20).mean().fillna(0))
        else:
            features.extend([0] * 2)

        # 价格位置特征
        high_20 = data['high'].rolling(20).max()
        low_20 = data['low'].rolling(20).min()
        price_position = (data['close'] - low_20) / (high_20 - low_20)
        features.append(price_position.fillna(0.5))

        # 动量特征
        features.append(data['close'].pct_change(5).fillna(0))  # 5日动量
        features.append(data['close'].pct_change(10).fillna(0))  # 10日动量

        # 转换为numpy数组
        feature_matrix = np.column_stack(features)

        # 标准化
        feature_matrix = self.scaler.fit_transform(feature_matrix)

        return feature_matrix

    def _create_labels(self, data: pd.DataFrame, lookahead: int = 5) -> np.ndarray:
        """创建标签"""
        future_returns = data['close'].pct_change(lookahead).shift(-lookahead)

        # 三分类：-1=下跌, 0=震荡, 1=上涨
        labels = np.zeros(len(data))
        labels[future_returns > 0.015] = 1    # 上涨超过1.5%
        labels[future_returns < -0.015] = -1  # 下跌超过1.5%

        return labels

    def train(self, data: pd.DataFrame) -> Dict[str, Any]:
        """训练模型"""
        print("🔬 开始训练量子分类器...")

        # 准备数据
        features = self._prepare_features(data)
        labels = self._create_labels(data)

        # 移除无效数据
        valid_indices = ~np.isnan(features).any(axis=1) & ~np.isnan(labels)
        features = features[valid_indices]
        labels = labels[valid_indices]

        # 分割训练测试集
        split_idx = int(len(features) * 0.8)
        X_train, X_test = features[:split_idx], features[split_idx:]
        y_train, y_test = labels[:split_idx], labels[split_idx:]

        results = {}

        # 1. 训练经典SVM基准
        self.classical_svm.fit(X_train, y_train)
        svm_accuracy = self.classical_svm.score(X_test, y_test)
        results['classical_svm_accuracy'] = svm_accuracy
        print(f"📊 经典SVM准确率: {svm_accuracy:.3f}")

        # 2. 训练随机森林基准
        self.classical_rf.fit(X_train, y_train)
        rf_accuracy = self.classical_rf.score(X_test, y_test)
        results['classical_rf_accuracy'] = rf_accuracy
        print(f"📊 随机森林准确率: {rf_accuracy:.3f}")

        # 3. 训练量子模型（如果可用）
        if QISKIT_AVAILABLE and len(X_train) > 0:
            try:
                # 创建量子核
                self.quantum_kernel = QuantumKernel(
                    feature_map=self._create_quantum_circuit(X_train.shape[1]),
                    quantum_instance=Aer.get_backend('qasm_simulator')
                )

                # 训练量子SVM
                self.quantum_model = QSVC(quantum_kernel=self.quantum_kernel)
                self.quantum_model.fit(X_train, y_train)

                quantum_accuracy = self.quantum_model.score(X_test, y_test)
                results['quantum_accuracy'] = quantum_accuracy
                print(f"⚛️ 量子SVM准确率: {quantum_accuracy:.3f}")

            except Exception as e:
                print(f"⚠️ 量子模型训练失败: {e}")
                results['quantum_error'] = str(e)
        else:
            print("⚠️ Qiskit不可用，跳过量子模型训练")
            results['quantum_skipped'] = True

        # 计算集成预测
        self.is_trained = True
        self.performance_metrics = results

        print(f"✅ 训练完成！最佳模型准确率: {max(results.values()):.3f}")
        return results

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """生成预测信号"""
        if not self.is_trained:
            raise ValueError("模型未训练，请先调用train方法")

        # 准备特征
        features = self._prepare_features(data)

        # 获取各模型预测
        predictions = {}

        # 经典SVM预测
        svm_pred = self.classical_svm.predict(features)
        svm_proba = self.classical_svm.predict_proba(features)
        predictions['svm'] = {
            'prediction': svm_pred,
            'confidence': np.max(svm_proba, axis=1)
        }

        # 随机森林预测
        rf_pred = self.classical_rf.predict(features)
        rf_proba = self.classical_rf.predict_proba(features)
        predictions['rf'] = {
            'prediction': rf_pred,
            'confidence': np.max(rf_proba, axis=1)
        }

        # 量子模型预测（如果可用）
        if self.quantum_model is not None:
            try:
                quantum_pred = self.quantum_model.predict(features)
                quantum_proba = self.quantum_model.predict_proba(features)
                predictions['quantum'] = {
                    'prediction': quantum_pred,
                    'confidence': np.max(quantum_proba, axis=1)
                }
            except Exception as e:
                print(f"量子预测失败: {e}")

        # 集成预测
        ensemble_pred = self._ensemble_predictions(predictions)

        return pd.Series(ensemble_pred, index=data.index)

    def _ensemble_predictions(self, predictions: Dict[str, Any]) -> np.ndarray:
        """集成多个模型的预测结果"""
        ensemble_signals = np.zeros(len(list(predictions.values())[0]['prediction']))

        # 权重分配
        weights = {
            'svm': 0.3,
            'rf': 0.4,
            'quantum': 0.3
        }

        # 如果量子模型不可用，调整权重
        if 'quantum' not in predictions:
            weights['svm'] = 0.4
            weights['rf'] = 0.6

        # 加权投票
        for model_name, pred_data in predictions.items():
            weight = weights.get(model_name, 0)
            ensemble_signals += weight * pred_data['prediction']

        # 离散化信号
        final_signals = np.zeros_like(ensemble_signals)
        final_signals[ensemble_signals > 0.3] = 1
        final_signals[ensemble_signals < -0.3] = -1

        return final_signals

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'n_qubits': self.n_qubits,
            'backend_type': self.backend_type,
            'is_trained': self.is_trained,
            'qiskit_available': QISKIT_AVAILABLE,
            'performance_metrics': self.performance_metrics,
            'feature_names': [
                'returns', 'short_trend', 'long_trend',
                'short_volatility', 'long_volatility',
                'volume_change', 'volume_sma',
                'price_position', 'momentum_5d', 'momentum_10d'
            ]
        }

    def get_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """获取详细交易信号"""
        predictions = self.predict(data)

        signals_df = pd.DataFrame(index=data.index)
        signals_df['signal'] = predictions
        signals_df['signal_strength'] = abs(predictions)

        # 计算信号置信度
        features = self._prepare_features(data)
        if self.is_trained:
            try:
                svm_proba = self.classical_svm.predict_proba(features)
                rf_proba = self.classical_rf.predict_proba(features)
                signals_df['confidence'] = np.maximum(
                    np.max(svm_proba, axis=1),
                    np.max(rf_proba, axis=1)
                )
            except:
                signals_df['confidence'] = 0.5

        return signals_df

    def analyze_quantum_advantage(self, data: pd.DataFrame) -> Dict[str, Any]:
        """分析量子优势"""
        if not self.is_trained or 'quantum_accuracy' not in self.performance_metrics:
            return {'quantum_advantage': False, 'reason': 'Quantum model not trained'}

        classical_best = max(
            self.performance_metrics.get('classical_svm_accuracy', 0),
            self.performance_metrics.get('classical_rf_accuracy', 0)
        )
        quantum_acc = self.performance_metrics.get('quantum_accuracy', 0)

        advantage_threshold = 0.02  # 2%的改进阈值

        return {
            'quantum_advantage': quantum_acc > classical_best + advantage_threshold,
            'quantum_accuracy': quantum_acc,
            'classical_best_accuracy': classical_best,
            'improvement': quantum_acc - classical_best,
            'improvement_percentage': (quantum_acc - classical_best) / classical_best * 100 if classical_best > 0 else 0
        }

# 便捷函数
def create_quantum_classifier(n_qubits: int = 4) -> QuantumClassifier:
    """创建量子分类器实例"""
    return QuantumClassifier(n_qubits=n_qubits)

def quick_quantum_analysis(data: pd.DataFrame, n_qubits: int = 4) -> Dict[str, Any]:
    """快速量子分析"""
    classifier = QuantumClassifier(n_qubits=n_qubits)

    # 训练模型
    training_results = classifier.train(data)

    # 生成信号
    signals = classifier.get_trading_signals(data)

    # 分析量子优势
    quantum_advantage = classifier.analyze_quantum_advantage(data)

    return {
        'training_results': training_results,
        'latest_signal': signals['signal'].iloc[-1] if len(signals) > 0 else 0,
        'latest_confidence': signals['confidence'].iloc[-1] if len(signals) > 0 else 0,
        'quantum_advantage': quantum_advantage,
        'model_info': classifier.get_model_info()
    }