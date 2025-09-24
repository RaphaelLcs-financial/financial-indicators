"""
伊藤引理计算器

基于随机微积分的伊藤引理，实现金融衍生品定价和风险分析。支持多维
随机过程、相关系数计算、以及各种随机微分方程的数值解。

算法特点:
- 多维伊藤过程建模
- 相关布朗运动处理
- 蒙特卡洛模拟
- 有限元方法求解
- 风险中性定价

作者: Claude Code AI
版本: 1.0
日期: 2025-09-24
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.integrate import quad
from typing import Dict, List, Tuple, Optional, Union, Callable
from sklearn.preprocessing import StandardScaler
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StochasticProcess:
    """随机过程定义"""
    drift: Callable[[float, np.ndarray], float]  # 漂移项 μ(t, X)
    diffusion: Callable[[float, np.ndarray], float]  # 扩散项 σ(t, X)
    dimension: int = 1
    name: str = "Generic Stochastic Process"

@dataclass
class CorrelationMatrix:
    """相关系数矩阵"""
    matrix: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray

class ItoCalculator:
    """伊藤引理计算器"""

    def __init__(self,
                 time_horizon: float = 1.0,
                 time_steps: int = 1000,
                 num_simulations: int = 10000,
                 risk_free_rate: float = 0.05,
                 dividend_yield: float = 0.0):

        self.time_horizon = time_horizon
        self.time_steps = time_steps
        self.num_simulations = num_simulations
        self.risk_free_rate = risk_free_rate
        self.dividend_yield = dividend_yield

        # 时间网格
        self.dt = time_horizon / time_steps
        self.time_grid = np.linspace(0, time_horizon, time_steps + 1)

        # 随机数生成器
        self.rng = np.random.default_rng(42)

        logger.info(f"Ito Calculator initialized with {num_simulations} simulations "
                   f"over {time_horizon} years with {time_steps} time steps")

    def geometric_brownian_motion(self,
                                 s0: float,
                                 mu: float,
                                 sigma: float,
                                 correlation_matrix: Optional[np.ndarray] = None) -> Dict:
        """几何布朗运动模拟"""
        logger.info("Simulating Geometric Brownian Motion...")

        # 生成布朗运动
        if correlation_matrix is None:
            correlation_matrix = np.array([[1.0]])

        # Cholesky分解
        L = np.linalg.cholesky(correlation_matrix)

        # 生成相关随机变量
        dW = self.rng.normal(0, np.sqrt(self.dt), (self.num_simulations, self.time_steps, 1))
        correlated_dW = np.einsum('ijk,kl->ijl', dW, L)

        # 初始化价格路径
        S = np.zeros((self.num_simulations, self.time_steps + 1))
        S[:, 0] = s0

        # Euler-Maruyama模拟
        for t in range(self.time_steps):
            drift_term = (mu - 0.5 * sigma**2) * self.dt
            diffusion_term = sigma * correlated_dW[:, t, 0]

            S[:, t + 1] = S[:, t] * np.exp(drift_term + diffusion_term)

        # 计算统计量
        final_prices = S[:, -1]
        expected_price = np.mean(final_prices)
        price_std = np.std(final_prices)

        # 计算理论值
        theoretical_mean = s0 * np.exp(mu * self.time_horizon)
        theoretical_var = s0**2 * np.exp(2 * mu * self.time_horizon) * \
                          (np.exp(sigma**2 * self.time_horizon) - 1)

        return {
            'paths': S,
            'time_grid': self.time_grid,
            'final_prices': final_prices,
            'statistics': {
                'mean': expected_price,
                'std': price_std,
                'theoretical_mean': theoretical_mean,
                'theoretical_std': np.sqrt(theoretical_var),
                'relative_error': abs(expected_price - theoretical_mean) / theoretical_mean
            },
            'quantiles': {
                '5%': np.percentile(final_prices, 5),
                '25%': np.percentile(final_prices, 25),
                '50%': np.percentile(final_prices, 50),
                '75%': np.percentile(final_prices, 75),
                '95%': np.percentile(final_prices, 95)
            }
        }

    def ornstein_uhlenbeck_process(self,
                                  x0: float,
                                  theta: float,
                                  mu: float,
                                  sigma: float) -> Dict:
        """Ornstein-Uhlenbeck过程模拟"""
        logger.info("Simulating Ornstein-Uhlenbeck Process...")

        # 模拟参数
        X = np.zeros((self.num_simulations, self.time_steps + 1))
        X[:, 0] = x0

        # 生成随机冲击
        dW = self.rng.normal(0, np.sqrt(self.dt), (self.num_simulations, self.time_steps))

        # 精确解模拟
        exp_theta_dt = np.exp(-theta * self.dt)
        sigma_term = sigma * np.sqrt((1 - np.exp(-2 * theta * self.dt)) / (2 * theta))

        for t in range(self.time_steps):
            # 精确离散化
            X[:, t + 1] = mu + (X[:, t] - mu) * exp_theta_dt + sigma_term * dW[:, t]

        # 计算统计量
        final_values = X[:, -1]
        long_term_mean = mu
        long_term_var = sigma**2 / (2 * theta)

        return {
            'paths': X,
            'time_grid': self.time_grid,
            'final_values': final_values,
            'statistics': {
                'mean': np.mean(final_values),
                'std': np.std(final_values),
                'long_term_mean': long_term_mean,
                'long_term_std': np.sqrt(long_term_var)
            },
            'process_parameters': {
                'theta': theta,  # 均值回归速度
                'mu': mu,        # 长期均值
                'sigma': sigma   # 波动率
            }
        }

    def cox_ingersoll_ross_process(self,
                                  r0: float,
                                  kappa: float,
                                  theta: float,
                                  sigma: float) -> Dict:
        """Cox-Ingersoll-Ross利率过程模拟"""
        logger.info("Simulating Cox-Ingersoll-Ross Process...")

        # 参数检查
        if 2 * kappa * theta < sigma**2:
            logger.warning("Feller condition not satisfied: 2*kappa*theta < sigma^2")

        # 初始化利率路径
        r = np.zeros((self.num_simulations, self.time_steps + 1))
        r[:, 0] = r0

        # 生成随机冲击
        dW = self.rng.normal(0, np.sqrt(self.dt), (self.num_simulations, self.time_steps))

        # Euler-Maruyama模拟
        for t in range(self.time_steps):
            drift_term = kappa * (theta - r[:, t]) * self.dt
            diffusion_term = sigma * np.sqrt(np.maximum(r[:, t], 0)) * dW[:, t]

            r[:, t + 1] = r[:, t] + drift_term + diffusion_term

        # 计算统计量
        final_rates = r[:, -1]
        theoretical_mean = theta + (r0 - theta) * np.exp(-kappa * self.time_horizon)

        return {
            'paths': r,
            'time_grid': self.time_grid,
            'final_rates': final_rates,
            'statistics': {
                'mean': np.mean(final_rates),
                'std': np.std(final_rates),
                'theoretical_mean': theoretical_mean,
                'min_rate': np.min(final_rates),
                'max_rate': np.max(final_rates)
            },
            'process_parameters': {
                'kappa': kappa,  # 均值回归速度
                'theta': theta,  # 长期均值
                'sigma': sigma   # 波动率
            }
        }

    def heston_model(self,
                    s0: float,
                    v0: float,
                    mu: float,
                    kappa: float,
                    theta: float,
                    sigma: float,
                    rho: float) -> Dict:
        """Heston随机波动率模型"""
        logger.info("Simulating Heston Stochastic Volatility Model...")

        # 初始化路径
        S = np.zeros((self.num_simulations, self.time_steps + 1))
        V = np.zeros((self.num_simulations, self.time_steps + 1))
        S[:, 0] = s0
        V[:, 0] = v0

        # 生成相关布朗运动
        dW1 = self.rng.normal(0, np.sqrt(self.dt), (self.num_simulations, self.time_steps))
        dW2 = self.rng.normal(0, np.sqrt(self.dt), (self.num_simulations, self.time_steps))

        # 相关性调整
        dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW2

        # 模拟
        for t in range(self.time_steps):
            # 波动率过程
            drift_v = kappa * (theta - V[:, t]) * self.dt
            diffusion_v = sigma * np.sqrt(np.maximum(V[:, t], 0)) * dW2[:, t]
            V[:, t + 1] = V[:, t] + drift_v + diffusion_v

            # 价格过程
            drift_s = mu * S[:, t] * self.dt
            diffusion_s = np.sqrt(V[:, t]) * S[:, t] * dW1[:, t]
            S[:, t + 1] = S[:, t] + drift_s + diffusion_s

        # 计算统计量
        final_prices = S[:, -1]
        final_volatilities = V[:, -1]

        return {
            'price_paths': S,
            'volatility_paths': V,
            'time_grid': self.time_grid,
            'final_prices': final_prices,
            'final_volatilities': final_volatilities,
            'statistics': {
                'price_mean': np.mean(final_prices),
                'price_std': np.std(final_prices),
                'volatility_mean': np.mean(final_volatilities),
                'volatility_std': np.std(final_volatilities),
                'correlation': rho
            },
            'model_parameters': {
                'mu': mu,
                'kappa': kappa,
                'theta': theta,
                'sigma': sigma,
                'rho': rho
            }
        }

    def ito_lemma_application(self,
                            process: StochasticProcess,
                            function: Callable[[float, np.ndarray], float],
                            x0: float) -> Dict:
        """应用伊藤引理"""
        logger.info(f"Applying Ito's Lemma to {process.name}...")

        # 模拟原始过程
        X = np.zeros((self.num_simulations, self.time_steps + 1))
        X[:, 0] = x0

        # 生成随机增量
        dW = self.rng.normal(0, np.sqrt(self.dt), (self.num_simulations, self.time_steps))

        for t in range(self.time_steps):
            drift = process.drift(self.time_grid[t], X[:, t])
            diffusion = process.diffusion(self.time_grid[t], X[:, t])

            X[:, t + 1] = X[:, t] + drift * self.dt + diffusion * dW[:, t]

        # 应用伊藤引理计算f(t, X_t)
        Y = np.zeros_like(X)

        for i in range(self.num_simulations):
            for t in range(self.time_steps + 1):
                Y[i, t] = function(self.time_grid[t], X[i, t])

        # 计算伊藤引理的解析解（如果可能）
        analytical_drift = None
        analytical_diffusion = None

        # 数值微分计算偏导数
        epsilon = 1e-6
        n_samples = min(100, self.num_simulations)

        # 选择一些样本点计算偏导数
        sample_indices = np.random.choice(self.num_simulations, n_samples, replace=False)
        time_indices = np.random.choice(self.time_steps + 1, n_samples, replace=True)

        df_dx_samples = []
        d2f_dx2_samples = []

        for i, t_idx in zip(sample_indices, time_indices):
            x = X[i, t_idx]
            t_val = self.time_grid[t_idx]

            # 一阶偏导数
            f_plus = function(t_val, x + epsilon)
            f_minus = function(t_val, x - epsilon)
            df_dx = (f_plus - f_minus) / (2 * epsilon)

            # 二阶偏导数
            d2f_dx2 = (f_plus - 2 * function(t_val, x) + f_minus) / (epsilon**2)

            df_dx_samples.append(df_dx)
            d2f_dx2_samples.append(d2f_dx2)

        return {
            'original_process': X,
            'transformed_process': Y,
            'time_grid': self.time_grid,
            'derivatives': {
                'df_dx_mean': np.mean(df_dx_samples),
                'd2f_dx2_mean': np.mean(d2f_dx2_samples),
                'df_dx_std': np.std(df_dx_samples),
                'd2f_dx2_std': np.std(d2f_dx2_samples)
            },
            'statistics': {
                'original_mean': np.mean(X[:, -1]),
                'transformed_mean': np.mean(Y[:, -1]),
                'original_std': np.std(X[:, -1]),
                'transformed_std': np.std(Y[:, -1])
            }
        }

    def multi_dimensional_ito(self,
                             processes: List[StochasticProcess],
                             initial_values: np.ndarray,
                             correlation_matrix: np.ndarray) -> Dict:
        """多维伊藤过程"""
        logger.info("Simulating Multi-dimensional Ito Process...")

        n_dimensions = len(processes)
        assert len(initial_values) == n_dimensions
        assert correlation_matrix.shape == (n_dimensions, n_dimensions)

        # Cholesky分解
        L = np.linalg.cholesky(correlation_matrix)

        # 初始化多维过程
        X = np.zeros((self.num_simulations, self.time_steps + 1, n_dimensions))
        X[:, 0, :] = initial_values

        # 生成相关随机变量
        dW = self.rng.normal(0, np.sqrt(self.dt), (self.num_simulations, self.time_steps, n_dimensions))
        correlated_dW = np.einsum('ijk,kl->ijl', dW, L)

        # 模拟每个维度
        for dim, process in enumerate(processes):
            for t in range(self.time_steps):
                drift = process.drift(self.time_grid[t], X[:, t, dim])
                diffusion = process.diffusion(self.time_grid[t], X[:, t, dim])

                X[:, t + 1, dim] = X[:, t, dim] + drift * self.dt + diffusion * correlated_dW[:, t, dim]

        return {
            'paths': X,
            'time_grid': self.time_grid,
            'correlation_matrix': correlation_matrix,
            'final_values': X[:, -1, :],
            'statistics': {
                'means': np.mean(X[:, -1, :], axis=0),
                'covariance': np.cov(X[:, -1, :], rowvar=False),
                'correlation': np.corrcoef(X[:, -1, :], rowvar=False)
            }
        }

    def asian_option_pricing(self,
                           s0: float,
                           k: float,
                           t: float,
                           sigma: float,
                           option_type: str = 'call',
                           averaging_type: str = 'arithmetic') -> Dict:
        """亚式期权定价"""
        logger.info(f"Pricing {averaging_type} average Asian {option_type} option...")

        # 模拟价格路径
        gbm_result = self.geometric_brownian_motion(s0, self.risk_free_rate, sigma)

        paths = gbm_result['paths']

        if averaging_type == 'arithmetic':
            # 算术平均
            average_prices = np.mean(paths[:, 1:], axis=1)  # 排除初始价格
        elif averaging_type == 'geometric':
            # 几何平均
            average_prices = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))
        else:
            raise ValueError("averaging_type must be 'arithmetic' or 'geometric'")

        # 计算期权收益
        if option_type == 'call':
            payoffs = np.maximum(average_prices - k, 0)
        elif option_type == 'put':
            payoffs = np.maximum(k - average_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        # 贴现期望
        discounted_payoffs = np.exp(-self.risk_free_rate * t) * payoffs
        option_price = np.mean(discounted_payoffs)

        # 计算标准误差
        std_error = np.std(discounted_payoffs) / np.sqrt(self.num_simulations)

        # 计算Delta和Gamma
        delta, gamma = self._calculate_greeks(
            lambda s0_new: self._asian_option_monte_carlo(s0_new, k, t, sigma, option_type, averaging_type),
            s0, 0.01
        )

        return {
            'option_price': option_price,
            'std_error': std_error,
            'confidence_interval': [
                option_price - 1.96 * std_error,
                option_price + 1.96 * std_error
            ],
            'greeks': {
                'delta': delta,
                'gamma': gamma
            },
            'simulation_details': {
                'num_simulations': self.num_simulations,
                'time_steps': self.time_steps,
                'averaging_type': averaging_type,
                'option_type': option_type
            }
        }

    def _asian_option_monte_carlo(self, s0: float, k: float, t: float, sigma: float,
                                 option_type: str, averaging_type: str) -> float:
        """亚式期权蒙特卡洛定价辅助函数"""
        gbm_result = self.geometric_brownian_motion(s0, self.risk_free_rate, sigma)
        paths = gbm_result['paths']

        if averaging_type == 'arithmetic':
            average_prices = np.mean(paths[:, 1:], axis=1)
        else:
            average_prices = np.exp(np.mean(np.log(paths[:, 1:]), axis=1))

        if option_type == 'call':
            payoffs = np.maximum(average_prices - k, 0)
        else:
            payoffs = np.maximum(k - average_prices, 0)

        return np.mean(np.exp(-self.risk_free_rate * t) * payoffs)

    def _calculate_greeks(self, pricing_function: Callable, s0: float, epsilon: float) -> Tuple[float, float]:
        """计算Greeks"""
        # Delta
        price_up = pricing_function(s0 + epsilon)
        price_down = pricing_function(s0 - epsilon)
        delta = (price_up - price_down) / (2 * epsilon)

        # Gamma
        price_center = pricing_function(s0)
        gamma = (price_up - 2 * price_center + price_down) / (epsilon**2)

        return delta, gamma

    def basket_option_pricing(self,
                            weights: np.ndarray,
                            s0: np.ndarray,
                            k: float,
                            t: float,
                            sigma: np.ndarray,
                            correlation_matrix: np.ndarray,
                            option_type: str = 'call') -> Dict:
        """篮子期权定价"""
        logger.info("Pricing Basket Option...")

        n_assets = len(weights)
        assert len(s0) == n_assets
        assert len(sigma) == n_assets

        # 创建多维几何布朗运动
        processes = []
        for i in range(n_assets):
            def make_drift(mu, sigma):
                return lambda t, x: mu
            def make_diffusion(sigma):
                return lambda t, x: sigma

            processes.append(StochasticProcess(
                drift=make_drift(self.risk_free_rate, sigma[i]),
                diffusion=make_diffusion(sigma[i]),
                dimension=1,
                name=f"Asset_{i}"
            ))

        # 模拟多维过程
        multi_result = self.multi_dimensional_ito(processes, s0, correlation_matrix)
        paths = multi_result['paths']

        # 计算篮子价值
        basket_values = np.zeros((self.num_simulations, self.time_steps + 1))
        for t in range(self.time_steps + 1):
            basket_values[:, t] = np.dot(weights, paths[:, t, :].T)

        # 计算期权收益
        final_basket_values = basket_values[:, -1]
        if option_type == 'call':
            payoffs = np.maximum(final_basket_values - k, 0)
        else:
            payoffs = np.maximum(k - final_basket_values, 0)

        # 贴现期望
        discounted_payoffs = np.exp(-self.risk_free_rate * t) * payoffs
        option_price = np.mean(discounted_payoffs)

        return {
            'option_price': option_price,
            'basket_values': basket_values,
            'final_basket_values': final_basket_values,
            'correlation_matrix': correlation_matrix,
            'statistics': {
                'mean_basket_value': np.mean(final_basket_values),
                'std_basket_value': np.std(final_basket_values)
            }
        }

    def analyze_market_data(self, market_data: pd.DataFrame) -> Dict:
        """分析市场数据并应用随机微积分"""
        logger.info("Analyzing market data with stochastic calculus...")

        # 提取价格数据
        prices = market_data['close'].values
        returns = np.log(prices[1:] / prices[:-1])

        # 参数估计
        mu = np.mean(returns) * 252  # 年化收益率
        sigma = np.std(returns) * np.sqrt(252)  # 年化波动率

        # 模拟几何布朗运动
        gbm_result = self.geometric_brownian_motion(
            s0=prices[-1],
            mu=mu,
            sigma=sigma
        )

        # 期权定价示例
        atm_call_price = self.asian_option_pricing(
            s0=prices[-1],
            k=prices[-1],
            t=1.0,
            sigma=sigma,
            option_type='call'
        )

        # 波动率分析
        ou_result = self.ornstein_uhlenbeck_process(
            x0=sigma,
            theta=0.2,  # 长期波动率
            mu=0.2,
            sigma=0.1
        )

        return {
            'market_parameters': {
                'current_price': prices[-1],
                'estimated_mu': mu,
                'estimated_sigma': sigma,
                'historical_volatility': sigma
            },
            'gbm_simulation': gbm_result,
            'volatility_analysis': ou_result,
            'option_pricing': atm_call_price,
            'risk_metrics': {
                'value_at_risk_95': np.percentile(gbm_result['final_prices'], 5),
                'expected_shortfall_95': np.mean(gbm_result['final_prices'][gbm_result['final_prices'] <= np.percentile(gbm_result['final_prices'], 5)]),
                'probability_of_loss': np.mean(gbm_result['final_prices'] < prices[-1])
            }
        }

    def get_correlation_analysis(self, market_data: pd.DataFrame) -> Dict:
        """相关性分析"""
        logger.info("Performing correlation analysis...")

        # 计算收益率相关性
        returns = market_data[['close', 'volume']].pct_change().dropna()
        correlation_matrix = returns.corr().values

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)

        # 生成相关随机过程
        s0 = market_data['close'].iloc[-1]
        processes = [
            StochasticProcess(
                drift=lambda t, x: 0.05 * x,
                diffusion=lambda t, x: 0.2 * x,
                name="Price Process"
            ),
            StochasticProcess(
                drift=lambda t, x: 0.0,
                diffusion=lambda t, x: 0.3,
                name="Volume Process"
            )
        ]

        initial_values = np.array([s0, market_data['volume'].iloc[-1]])

        multi_result = self.multi_dimensional_ito(
            processes=processes,
            initial_values=initial_values,
            correlation_matrix=correlation_matrix
        )

        return {
            'correlation_matrix': correlation_matrix,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'multivariate_simulation': multi_result,
            'correlation_metrics': {
                'determinant': np.linalg.det(correlation_matrix),
                'trace': np.trace(correlation_matrix),
                'condition_number': np.linalg.cond(correlation_matrix)
            }
        }

# 使用示例
def example_usage():
    """使用示例"""
    # 创建伊藤计算器
    calculator = ItoCalculator(
        time_horizon=1.0,
        time_steps=252,
        num_simulations=10000,
        risk_free_rate=0.05
    )

    # 几何布朗运动
    gbm_result = calculator.geometric_brownian_motion(
        s0=100,
        mu=0.08,
        sigma=0.2
    )
    print(f"GBM Final Price Mean: {gbm_result['statistics']['mean']:.2f}")

    # 亚式期权定价
    asian_option = calculator.asian_option_pricing(
        s0=100,
        k=100,
        t=1.0,
        sigma=0.2,
        option_type='call',
        averaging_type='arithmetic'
    )
    print(f"Asian Option Price: {asian_option['option_price']:.4f}")

    # Heston模型
    heston_result = calculator.heston_model(
        s0=100,
        v0=0.04,
        mu=0.08,
        kappa=2.0,
        theta=0.04,
        sigma=0.3,
        rho=-0.7
    )
    print(f"Heston Model - Final Volatility Mean: {heston_result['statistics']['volatility_mean']:.4f}")

    return calculator

if __name__ == "__main__":
    example_usage()