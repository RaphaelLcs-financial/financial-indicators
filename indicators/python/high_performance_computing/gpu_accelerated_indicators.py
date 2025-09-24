"""
GPU加速金融指标
GPU Accelerated Financial Indicators

利用CUDA和GPU并行计算加速大规模金融指标计算
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from typing import Dict, Any, List, Tuple, Optional
import time

# GPU加速支持
try:
    import cupy as cp
    from numba import cuda, jit
    CUDA_AVAILABLE = True
    print("🚀 CUDA GPU加速已启用")
except ImportError:
    CUDA_AVAILABLE = False
    print("⚠️ CUDA不可用，将使用CPU版本")

# 多进程支持
try:
    from multiprocessing import Pool, cpu_count
    from joblib import Parallel, delayed
    MULTIPROCESSING_AVAILABLE = True
    N_CORES = cpu_count()
    print(f"🖥️ 多进程已启用，核心数: {N_CORES}")
except ImportError:
    MULTIPROCESSING_AVAILABLE = False
    N_CORES = 1
    print("⚠️ 多进程不可用")

class GPUAcceleratedIndicators:
    """
    GPU加速金融指标计算器

    利用GPU并行计算加速大规模金融数据处理
    支持多种并行计算模式
    """

    def __init__(self, use_gpu: bool = True, use_multiprocessing: bool = True):
        """
        初始化GPU加速指标

        Args:
            use_gpu: 是否使用GPU加速
            use_multiprocessing: 是否使用多进程
        """
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        self.use_multiprocessing = use_multiprocessing and MULTIPROCESSING_AVAILABLE

        self.device_info = self._get_device_info()
        self.performance_stats = {}

    def _get_device_info(self) -> Dict[str, Any]:
        """获取设备信息"""
        info = {
            'gpu_available': CUDA_AVAILABLE,
            'multiprocessing_available': MULTIPROCESSING_AVAILABLE,
            'cpu_cores': N_CORES,
            'use_gpu': self.use_gpu,
            'use_multiprocessing': self.use_multiprocessing
        }

        if CUDA_AVAILABLE:
            try:
                info['gpu_name'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
                info['gpu_memory'] = cp.cuda.runtime.getDeviceProperties(0)['totalGlobalMem'] // (1024**3)
            except:
                info['gpu_name'] = 'Unknown GPU'
                info['gpu_memory'] = 0

        return info

    def _to_gpu(self, data: np.ndarray) -> Any:
        """数据转移到GPU"""
        if self.use_gpu and CUDA_AVAILABLE:
            return cp.asarray(data)
        return data

    def _from_gpu(self, data: Any) -> np.ndarray:
        """数据从GPU转回CPU"""
        if self.use_gpu and CUDA_AVAILABLE and hasattr(data, 'get'):
            return data.get()
        return data

    def accelerated_moving_average(self, data: pd.Series, window: int,
                                method: str = 'sma') -> pd.Series:
        """GPU加速移动平均"""
        start_time = time.time()

        if self.use_gpu and CUDA_AVAILABLE:
            result = self._gpu_moving_average(data.values, window, method)
        else:
            result = self._cpu_moving_average(data.values, window, method)

        execution_time = time.time() - start_time
        self.performance_stats['moving_average'] = execution_time

        return pd.Series(result, index=data.index)

    def _gpu_moving_average(self, data: np.ndarray, window: int, method: str) -> np.ndarray:
        """GPU移动平均计算"""
        gpu_data = self._to_gpu(data)

        if method == 'sma':
            # 使用卷积实现SMA
            kernel = cp.ones(window) / window
            result = cp.convolve(gpu_data, kernel, mode='same')
        elif method == 'ema':
            # EMA计算
            alpha = 2.0 / (window + 1)
            result = cp.zeros_like(gpu_data)
            result[0] = gpu_data[0]
            for i in range(1, len(gpu_data)):
                result[i] = alpha * gpu_data[i] + (1 - alpha) * result[i-1]
        else:
            raise ValueError(f"未知的移动平均方法: {method}")

        return self._from_gpu(result)

    def _cpu_moving_average(self, data: np.ndarray, window: int, method: str) -> np.ndarray:
        """CPU移动平均计算"""
        if method == 'sma':
            return pd.Series(data).rolling(window=window).mean().values
        elif method == 'ema':
            return pd.Series(data).ewm(span=window).mean().values
        else:
            raise ValueError(f"未知的移动平均方法: {method}")

    @cuda.jit
    def _cuda_rsi_kernel(self, data: np.ndarray, result: np.ndarray, window: int):
        """CUDA RSI计算核函数"""
        idx = cuda.grid(1)
        n = len(data)

        if idx < window or idx >= n:
            return

        # 计算收益
        gains = 0.0
        losses = 0.0

        # 初始化窗口
        for i in range(max(0, idx - window + 1), idx):
            change = data[i] - data[i-1] if i > 0 else 0
            if change > 0:
                gains += change
            else:
                losses += -change

        avg_gain = gains / window
        avg_loss = losses / window

        if avg_loss > 0:
            rs = avg_gain / avg_loss
            result[idx] = 100.0 - (100.0 / (1.0 + rs))
        else:
            result[idx] = 100.0

    def accelerated_rsi(self, data: pd.Series, window: int = 14) -> pd.Series:
        """GPU加速RSI计算"""
        start_time = time.time()

        if self.use_gpu and CUDA_AVAILABLE:
            result = self._gpu_rsi(data.values, window)
        else:
            result = self._cpu_rsi(data.values, window)

        execution_time = time.time() - start_time
        self.performance_stats['rsi'] = execution_time

        return pd.Series(result, index=data.index)

    def _gpu_rsi(self, data: np.ndarray, window: int) -> np.ndarray:
        """GPU RSI计算"""
        gpu_data = self._to_gpu(data)
        result = cp.zeros_like(gpu_data)

        # 配置CUDA网格
        threads_per_block = 256
        blocks_per_grid = (len(data) + threads_per_block - 1) // threads_per_block

        # 启动CUDA核函数
        self._cuda_rsi_kernel[blocks_per_grid, threads_per_block](gpu_data, result, window)

        return self._from_gpu(result)

    def _cpu_rsi(self, data: np.ndarray, window: int) -> np.ndarray:
        """CPU RSI计算"""
        delta = np.diff(data, prepend=data[0])
        gains = np.where(delta > 0, delta, 0)
        losses = np.where(delta < 0, -delta, 0)

        avg_gains = pd.Series(gains).rolling(window=window).mean()
        avg_losses = pd.Series(losses).rolling(window=window).mean()

        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi.fillna(50).values

    def accelerated_correlation_matrix(self, data: pd.DataFrame,
                                     window: int = 20) -> np.ndarray:
        """GPU加速相关性矩阵计算"""
        start_time = time.time()

        if self.use_gpu and CUDA_AVAILABLE:
            result = self._gpu_correlation_matrix(data, window)
        else:
            result = self._cpu_correlation_matrix(data, window)

        execution_time = time.time() - start_time
        self.performance_stats['correlation_matrix'] = execution_time

        return result

    def _gpu_correlation_matrix(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """GPU相关性矩阵计算"""
        returns = data.pct_change().fillna(0).values
        n_assets = returns.shape[1]

        # 滚动窗口相关性计算
        correlation_matrices = []

        for i in range(window, len(returns)):
            window_returns = returns[i-window:i]
            gpu_returns = self._to_gpu(window_returns)

            # 计算相关性矩阵
            corr_matrix = cp.corrcoef(gpu_returns.T)
            correlation_matrices.append(self._from_gpu(corr_matrix))

        return np.array(correlation_matrices)

    def _cpu_correlation_matrix(self, data: pd.DataFrame, window: int) -> np.ndarray:
        """CPU相关性矩阵计算"""
        returns = data.pct_change().fillna(0)
        correlation_matrices = []

        for i in range(window, len(returns)):
            window_returns = returns.iloc[i-window:i]
            corr_matrix = window_returns.corr().values
            correlation_matrices.append(corr_matrix)

        return np.array(correlation_matrices)

    def accelerated_portfolio_optimization(self, returns: pd.DataFrame,
                                         n_simulations: int = 10000) -> Dict[str, Any]:
        """GPU加速投资组合优化"""
        start_time = time.time()

        if self.use_gpu and CUDA_AVAILABLE:
            result = self._gpu_portfolio_optimization(returns.values, n_simulations)
        else:
            result = self._cpu_portfolio_optimization(returns.values, n_simulations)

        execution_time = time.time() - start_time
        self.performance_stats['portfolio_optimization'] = execution_time

        return result

    def _gpu_portfolio_optimization(self, returns: np.ndarray, n_simulations: int) -> Dict[str, Any]:
        """GPU投资组合优化"""
        n_assets = returns.shape[1]

        # 生成随机权重
        weights = cp.random.random((n_simulations, n_assets))
        weights = weights / cp.sum(weights, axis=1, keepdims=True)

        # 计算投资组合收益和风险
        gpu_returns = self._to_gpu(returns)
        expected_returns = cp.mean(gpu_returns, axis=0)
        cov_matrix = cp.cov(gpu_returns.T)

        # 并行计算所有组合的指标
        portfolio_returns = cp.dot(weights, expected_returns)
        portfolio_risks = cp.sqrt(cp.sum(cp.dot(weights, cov_matrix) * weights, axis=1))
        sharpe_ratios = portfolio_returns / portfolio_risks

        # 找到最优组合
        best_idx = cp.argmax(sharpe_ratios)
        best_weights = self._from_gpu(weights[best_idx])
        best_sharpe = self._from_gpu(sharpe_ratios[best_idx])

        # 找到有效前沿
        efficient_frontier = self._find_efficient_frontier_gpu(weights, portfolio_returns, portfolio_risks)

        return {
            'optimal_weights': best_weights,
            'optimal_sharpe_ratio': best_sharpe,
            'efficient_frontier': efficient_frontier,
            'all_simulations': n_simulations
        }

    def _find_efficient_frontier_gpu(self, weights, returns, risks) -> Dict[str, np.ndarray]:
        """找到有效前沿"""
        # 简化的有效前沿计算
        cpu_returns = self._from_gpu(returns)
        cpu_risks = self._from_gpu(risks)

        # 按风险排序
        sorted_indices = np.argsort(cpu_risks)
        efficient_returns = cpu_returns[sorted_indices]
        efficient_risks = cpu_risks[sorted_indices]

        return {
            'returns': efficient_returns,
            'risks': efficient_risks
        }

    def _cpu_portfolio_optimization(self, returns: np.ndarray, n_simulations: int) -> Dict[str, Any]:
        """CPU投资组合优化"""
        n_assets = returns.shape[1]

        best_sharpe = -np.inf
        best_weights = None

        all_returns = []
        all_risks = []

        for _ in range(n_simulations):
            weights = np.random.random(n_assets)
            weights /= np.sum(weights)

            portfolio_return = np.dot(weights, np.mean(returns, axis=0))
            portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(np.cov(returns.T), weights)))
            sharpe_ratio = portfolio_return / portfolio_risk

            all_returns.append(portfolio_return)
            all_risks.append(portfolio_risk)

            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_weights = weights

        return {
            'optimal_weights': best_weights,
            'optimal_sharpe_ratio': best_sharpe,
            'efficient_frontier': {
                'returns': np.array(all_returns),
                'risks': np.array(all_risks)
            },
            'all_simulations': n_simulations
        }

    def accelerated_risk_metrics(self, returns: pd.Series,
                                confidence_level: float = 0.95) -> Dict[str, float]:
        """GPU加速风险指标计算"""
        start_time = time.time()

        if self.use_gpu and CUDA_AVAILABLE:
            result = self._gpu_risk_metrics(returns.values, confidence_level)
        else:
            result = self._cpu_risk_metrics(returns.values, confidence_level)

        execution_time = time.time() - start_time
        self.performance_stats['risk_metrics'] = execution_time

        return result

    def _gpu_risk_metrics(self, returns: np.ndarray, confidence_level: float) -> Dict[str, float]:
        """GPU风险指标计算"""
        gpu_returns = self._to_gpu(returns)

        # VaR计算
        var = cp.percentile(gpu_returns, (1 - confidence_level) * 100)

        # CVaR计算
        cvar = cp.mean(gpu_returns[gpu_returns <= var])

        # 波动率
        volatility = cp.std(gpu_returns)

        return {
            'VaR': float(self._from_gpu(var)),
            'CVaR': float(self._from_gpu(cvar)),
            'volatility': float(self._from_gpu(volatility))
        }

    def _cpu_risk_metrics(self, returns: np.ndarray, confidence_level: float) -> Dict[str, float]:
        """CPU风险指标计算"""
        var = np.percentile(returns, (1 - confidence_level) * 100)
        cvar = np.mean(returns[returns <= var])
        volatility = np.std(returns)

        return {
            'VaR': var,
            'CVaR': cvar,
            'volatility': volatility
        }

    def parallel_technical_analysis(self, data: pd.DataFrame,
                                  indicators: List[str]) -> Dict[str, pd.Series]:
        """并行技术分析"""
        start_time = time.time()

        if self.use_multiprocessing and MULTIPROCESSING_AVAILABLE:
            result = self._parallel_technical_analysis(data, indicators)
        else:
            result = self._sequential_technical_analysis(data, indicators)

        execution_time = time.time() - start_time
        self.performance_stats['parallel_analysis'] = execution_time

        return result

    def _parallel_technical_analysis(self, data: pd.DataFrame,
                                   indicators: List[str]) -> Dict[str, pd.Series]:
        """并行技术分析"""
        def calculate_indicator(indicator_name):
            if indicator_name == 'sma_20':
                return self.accelerated_moving_average(data['close'], 20, 'sma')
            elif indicator_name == 'ema_20':
                return self.accelerated_moving_average(data['close'], 20, 'ema')
            elif indicator_name == 'rsi':
                return self.accelerated_rsi(data['close'])
            else:
                return pd.Series(0, index=data.index)

        # 使用joblib并行计算
        results = Parallel(n_jobs=N_CORES)(
            delayed(calculate_indicator)(indicator) for indicator in indicators
        )

        return dict(zip(indicators, results))

    def _sequential_technical_analysis(self, data: pd.DataFrame,
                                     indicators: List[str]) -> Dict[str, pd.Series]:
        """顺序技术分析"""
        results = {}
        for indicator in indicators:
            if indicator == 'sma_20':
                results[indicator] = self.accelerated_moving_average(data['close'], 20, 'sma')
            elif indicator == 'ema_20':
                results[indicator] = self.accelerated_moving_average(data['close'], 20, 'ema')
            elif indicator == 'rsi':
                results[indicator] = self.accelerated_rsi(data['close'])
            else:
                results[indicator] = pd.Series(0, index=data.index)

        return results

    def benchmark_performance(self, data: pd.DataFrame) -> Dict[str, Any]:
        """性能基准测试"""
        print("🏁 开始性能基准测试...")

        benchmark_results = {}

        # 测试移动平均
        start = time.time()
        self.accelerated_moving_average(data['close'], 20)
        benchmark_results['moving_average_gpu'] = time.time() - start

        start = time.time()
        self._cpu_moving_average(data['close'].values, 20, 'sma')
        benchmark_results['moving_average_cpu'] = time.time() - start

        # 测试RSI
        start = time.time()
        self.accelerated_rsi(data['close'])
        benchmark_results['rsi_gpu'] = time.time() - start

        start = time.time()
        self._cpu_rsi(data['close'].values, 14)
        benchmark_results['rsi_cpu'] = time.time() - start

        # 计算加速比
        if benchmark_results['moving_average_cpu'] > 0:
            benchmark_results['ma_speedup'] = benchmark_results['moving_average_cpu'] / benchmark_results['moving_average_gpu']

        if benchmark_results['rsi_cpu'] > 0:
            benchmark_results['rsi_speedup'] = benchmark_results['rsi_cpu'] / benchmark_results['rsi_gpu']

        return benchmark_results

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        return {
            'performance_stats': self.performance_stats,
            'device_info': self.device_info,
            'benchmark_results': getattr(self, 'benchmark_results', None)
        }

    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            'use_gpu': self.use_gpu,
            'use_multiprocessing': self.use_multiprocessing,
            'device_info': self.device_info,
            'performance_stats': self.performance_stats,
            'model_type': 'GPU Accelerated Financial Indicators'
        }

# 便捷函数
def create_gpu_accelerated_indicators(use_gpu: bool = True,
                                    use_multiprocessing: bool = True) -> GPUAcceleratedIndicators:
    """创建GPU加速指标实例"""
    return GPUAcceleratedIndicators(use_gpu, use_multiprocessing)

def quick_gpu_analysis(data: pd.DataFrame) -> Dict[str, Any]:
    """快速GPU分析"""
    accelerator = GPUAcceleratedIndicators()

    # 计算指标
    indicators = accelerator.parallel_technical_analysis(data, ['sma_20', 'ema_20', 'rsi'])

    # 性能基准测试
    benchmark = accelerator.benchmark_performance(data)

    return {
        'indicators': indicators,
        'benchmark_results': benchmark,
        'performance_stats': accelerator.get_performance_stats(),
        'model_info': accelerator.get_model_info()
    }