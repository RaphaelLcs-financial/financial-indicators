"""
链上分析器金融指标

本模块实现了基于区块链链上数据的加密货币专用指标，包括网络活动、持币者行为、交易所流量等分析。
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional, Union
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class OnChainAnalyzer:
    """
    链上分析器

    分析区块链链上数据，提供加密货币特有的技术指标
    """

    def __init__(self,
                 analysis_window: int = 30,
                 whale_threshold: float = 1000,  # 1000 BTC/ETH为大户阈值
                 exchange_threshold: float = 10000):  # 交易所流量阈值
        """
        初始化链上分析器

        Args:
            analysis_window: 分析窗口大小
            whale_threshold: 大户持币量阈值
            exchange_threshold: 交易所流量阈值
        """
        self.analysis_window = analysis_window
        self.whale_threshold = whale_threshold
        self.exchange_threshold = exchange_threshold

        # 链上数据缓存
        self.on_chain_data = {}
        self.network_metrics = {}

    def calculate_network_activity_metrics(self, on_chain_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算网络活动指标

        Args:
            on_chain_data: 链上数据，包含transactions, active_addresses等

        Returns:
            网络活动指标字典
        """
        metrics = {}

        # 交易活跃度
        if 'transactions' in on_chain_data.columns:
            tx_volume = on_chain_data['transactions']
            metrics['transaction_volume'] = tx_volume.rolling(self.analysis_window).mean()
            metrics['transaction_growth'] = tx_volume.pct_change().rolling(self.analysis_window).mean()

        # 活跃地址数
        if 'active_addresses' in on_chain_data.columns:
            active_addresses = on_chain_data['active_addresses']
            metrics['active_addresses'] = active_addresses.rolling(self.analysis_window).mean()
            metrics['address_growth'] = active_addresses.pct_change().rolling(self.analysis_window).mean()

        # 平均交易规模
        if 'transactions' in on_chain_data.columns and 'active_addresses' in on_chain_data.columns:
            avg_tx_per_address = on_chain_data['transactions'] / (on_chain_data['active_addresses'] + 1e-8)
            metrics['avg_transactions_per_address'] = avg_tx_per_address.rolling(self.analysis_window).mean()

        # 网络利用率
        if 'block_size' in on_chain_data.columns and 'gas_limit' in on_chain_data.columns:
            network_utilization = on_chain_data['block_size'] / (on_chain_data['gas_limit'] + 1e-8)
            metrics['network_utilization'] = network_utilization.rolling(self.analysis_window).mean()

        # Gas费用指标
        if 'gas_price' in on_chain_data.columns:
            gas_price = on_chain_data['gas_price']
            metrics['gas_price_ma'] = gas_price.rolling(self.analysis_window).mean()
            metrics['gas_price_zscore'] = (gas_price - gas_price.rolling(self.analysis_window).mean()) / \
                                         (gas_price.rolling(self.analysis_window).std() + 1e-8)

        # 区块时间
        if 'block_time' in on_chain_data.columns:
            block_time = on_chain_data['block_time']
            metrics['avg_block_time'] = block_time.rolling(self.analysis_window).mean()
            metrics['block_time_volatility'] = block_time.rolling(self.analysis_window).std()

        return metrics

    def calculate_holder_behavior_metrics(self, on_chain_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算持币者行为指标

        Args:
            on_chain_data: 链上数据，包含各种持币者数据

        Returns:
            持币者行为指标字典
        """
        metrics = {}

        # 大户持币比例
        if 'whale_balance' in on_chain_data.columns and 'total_supply' in on_chain_data.columns:
            whale_ratio = on_chain_data['whale_balance'] / (on_chain_data['total_supply'] + 1e-8)
            metrics['whale_concentration'] = whale_ratio.rolling(self.analysis_window).mean()
            metrics['whale_balance_change'] = on_chain_data['whale_balance'].pct_change().rolling(self.analysis_window).mean()

        # 长期持币者指标
        if 'hodler_balance' in on_chain_data.columns:
            hodler_balance = on_chain_data['hodler_balance']
            metrics['hodler_ratio'] = hodler_balance.rolling(self.analysis_window).mean()
            metrics['hodler_accumulation'] = hodler_balance.pct_change().rolling(self.analysis_window).mean()

        # 新地址增长
        if 'new_addresses' in on_chain_data.columns:
            new_addresses = on_chain_data['new_addresses']
            metrics['new_address_growth'] = new_addresses.pct_change().rolling(self.analysis_window).mean()
            metrics['new_address_ma'] = new_addresses.rolling(self.analysis_window).mean()

        # 零余额地址
        if 'zero_balance_addresses' in on_chain_data.columns:
            zero_balance = on_chain_data['zero_balance_addresses']
            total_addresses = on_chain_data.get('total_addresses', zero_balance)
            metrics['zero_balance_ratio'] = zero_balance / (total_addresses + 1e-8)

        # 地址分布
        if 'address_distribution' in on_chain_data.columns:
            # 基尼系数计算地址集中度
            gini_coefficient = self._calculate_gini_coefficient(on_chain_data['address_distribution'])
            metrics['address_gini'] = gini_coefficient.rolling(self.analysis_window).mean()

        # 持币年龄分布
        if 'age_distribution' in on_chain_data.columns:
            # 计算持币年龄加权平均
            age_weighted_avg = self._calculate_age_weighted_average(on_chain_data['age_distribution'])
            metrics['avg_holding_period'] = age_weighted_avg.rolling(self.analysis_window).mean()

        return metrics

    def _calculate_gini_coefficient(self, distribution_data: pd.Series) -> pd.Series:
        """计算基尼系数"""
        gini_values = []

        for i in range(len(distribution_data)):
            try:
                # 解析分布数据
                if pd.notna(distribution_data.iloc[i]):
                    dist = eval(distribution_data.iloc[i]) if isinstance(distribution_data.iloc[i], str) else distribution_data.iloc[i]

                    if isinstance(dist, dict) and 'balances' in dist:
                        balances = np.array(dist['balances'])
                        if len(balances) > 0:
                            # 计算基尼系数
                            n = len(balances)
                            gini = 0
                            for i in range(n):
                                for j in range(n):
                                    gini += abs(balances[i] - balances[j])
                            gini = gini / (2 * n * np.sum(balances)) if np.sum(balances) > 0 else 0
                        else:
                            gini = 0
                    else:
                        gini = 0
                else:
                    gini = 0

                gini_values.append(gini)
            except:
                gini_values.append(0)

        return pd.Series(gini_values, index=distribution_data.index)

    def _calculate_age_weighted_average(self, age_distribution: pd.Series) -> pd.Series:
        """计算持币年龄加权平均"""
        age_values = []

        for i in range(len(age_distribution)):
            try:
                if pd.notna(age_distribution.iloc[i]):
                    dist = eval(age_distribution.iloc[i]) if isinstance(age_distribution.iloc[i], str) else age_distribution.iloc[i]

                    if isinstance(dist, dict) and 'ages' in dist and 'amounts' in dist:
                        ages = np.array(dist['ages'])
                        amounts = np.array(dist['amounts'])

                        if len(ages) > 0 and len(amounts) > 0:
                            weighted_avg = np.average(ages, weights=amounts)
                        else:
                            weighted_avg = 0
                    else:
                        weighted_avg = 0
                else:
                    weighted_avg = 0

                age_values.append(weighted_avg)
            except:
                age_values.append(0)

        return pd.Series(age_values, index=age_distribution.index)

    def calculate_exchange_flow_metrics(self, on_chain_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算交易所流量指标

        Args:
            on_chain_data: 链上数据，包含交易所流入流出数据

        Returns:
            交易所流量指标字典
        """
        metrics = {}

        # 交易所净流入
        if 'exchange_inflow' in on_chain_data.columns and 'exchange_outflow' in on_chain_data.columns:
            net_flow = on_chain_data['exchange_inflow'] - on_chain_data['exchange_outflow']
            metrics['exchange_net_flow'] = net_flow.rolling(self.analysis_window).mean()
            metrics['exchange_net_flow_pct'] = net_flow.pct_change().rolling(self.analysis_window).mean()

            # 流量比率
            total_flow = on_chain_data['exchange_inflow'] + on_chain_data['exchange_outflow']
            metrics['exchange_flow_ratio'] = net_flow / (total_flow + 1e-8)

        # 交易所余额
        if 'exchange_balance' in on_chain_data.columns:
            exchange_balance = on_chain_data['exchange_balance']
            metrics['exchange_balance_ma'] = exchange_balance.rolling(self.analysis_window).mean()
            metrics['exchange_balance_change'] = exchange_balance.pct_change().rolling(self.analysis_window).mean()

        # 稳定币流入
        if 'stablecoin_inflow' in on_chain_data.columns:
            stablecoin_flow = on_chain_data['stablecoin_inflow']
            metrics['stablecoin_inflow_ma'] = stablecoin_flow.rolling(self.analysis_window).mean()
            metrics['stablecoin_inflow_growth'] = stablecoin_flow.pct_change().rolling(self.analysis_window).mean()

        # 交易所存款/提款比率
        if 'exchange_deposits' in on_chain_data.columns and 'exchange_withdrawals' in on_chain_data.columns:
            deposit_withdrawal_ratio = on_chain_data['exchange_deposits'] / (on_chain_data['exchange_withdrawals'] + 1e-8)
            metrics['deposit_withdrawal_ratio'] = deposit_withdrawal_ratio.rolling(self.analysis_window).mean()

        # 巨额交易监测
        if 'large_transactions' in on_chain_data.columns:
            large_tx = on_chain_data['large_transactions']
            metrics['large_transaction_volume'] = large_tx.rolling(self.analysis_window).sum()
            metrics['large_transaction_frequency'] = (large_tx > 0).rolling(self.analysis_window).sum()

        return metrics

    def calculate_mining_staking_metrics(self, on_chain_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算挖矿/质押指标

        Args:
            on_chain_data: 链上数据，包含挖矿或质押相关数据

        Returns:
            挖矿/质押指标字典
        """
        metrics = {}

        # 挖矿难度/算力
        if 'mining_difficulty' in on_chain_data.columns:
            difficulty = on_chain_data['mining_difficulty']
            metrics['difficulty_ma'] = difficulty.rolling(self.analysis_window).mean()
            metrics['difficulty_change'] = difficulty.pct_change().rolling(self.analysis_window).mean()

        # 挖矿收益
        if 'mining_revenue' in on_chain_data.columns:
            revenue = on_chain_data['mining_revenue']
            metrics['mining_revenue_ma'] = revenue.rolling(self.analysis_window).mean()
            metrics['mining_profitability'] = revenue.rolling(self.analysis_window).mean() / \
                                           (difficulty.rolling(self.analysis_window).mean() + 1e-8)

        # 质押相关指标
        if 'staking_reward' in on_chain_data.columns:
            staking_reward = on_chain_data['staking_reward']
            metrics['staking_yield'] = staking_reward.rolling(self.analysis_window).mean()

        if 'total_staked' in on_chain_data.columns:
            total_staked = on_chain_data['total_staked']
            metrics['staking_ratio'] = total_staked.rolling(self.analysis_window).mean()
            metrics['staking_growth'] = total_staked.pct_change().rolling(self.analysis_window).mean()

        # 验证者/矿工数量
        if 'validator_count' in on_chain_data.columns:
            validators = on_chain_data['validator_count']
            metrics['validator_growth'] = validators.pct_change().rolling(self.analysis_window).mean()
            metrics['validator_concentration'] = validators.rolling(self.analysis_window).std() / \
                                                 (validators.rolling(self.analysis_window).mean() + 1e-8)

        return metrics

    def calculate_supply_dynamics_metrics(self, on_chain_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算供应动态指标

        Args:
            on_chain_data: 链上数据，包含供应量相关数据

        Returns:
            供应动态指标字典
        """
        metrics = {}

        # 流通供应量
        if 'circulating_supply' in on_chain_data.columns:
            circulating_supply = on_chain_data['circulating_supply']
            metrics['supply_growth_rate'] = circulating_supply.pct_change().rolling(self.analysis_window).mean()

        # 通胀率
        if 'inflation_rate' in on_chain_data.columns:
            inflation = on_chain_data['inflation_rate']
            metrics['inflation_ma'] = inflation.rolling(self.analysis_window).mean()
            metrics['real_inflation'] = inflation.rolling(self.analysis_window).mean() - \
                                      on_chain_data.get('price_inflation', 0).rolling(self.analysis_window).mean()

        # 销毁/焚烧机制
        if 'burned_tokens' in on_chain_data.columns:
            burned = on_chain_data['burned_tokens']
            metrics['burn_rate'] = burned.rolling(self.analysis_window).mean()
            metrics['burn_ratio'] = burned / (on_chain_data.get('total_supply', burned) + 1e-8)

        # 解锁/释放时间表
        if 'unlocked_tokens' in on_chain_data.columns:
            unlocked = on_chain_data['unlocked_tokens']
            metrics['unlock_pressure'] = unlocked.rolling(self.analysis_window).mean()
            metrics['unlock_ratio'] = unlocked / (on_chain_data.get('circulating_supply', unlocked) + 1e-8)

        # 供应冲击
        if 'supply_shock_events' in on_chain_data.columns:
            supply_shock = on_chain_data['supply_shock_events']
            metrics['supply_shock_frequency'] = (supply_shock > 0).rolling(self.analysis_window).sum()
            metrics['supply_shock_magnitude'] = supply_shock.rolling(self.analysis_window).mean()

        return metrics

    def calculate_market_sentiment_indicators(self, on_chain_data: pd.DataFrame,
                                          market_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算市场情绪指标（链上+市场价格）

        Args:
            on_chain_data: 链上数据
            market_data: 市场价格数据

        Returns:
            市场情绪指标字典
        """
        metrics = {}

        # MVRV比率
        if 'market_cap' in market_data.columns and 'realized_cap' in on_chain_data.columns:
            mvrv = market_data['market_cap'] / (on_chain_data['realized_cap'] + 1e-8)
            metrics['mvrv_ratio'] = mvrv.rolling(self.analysis_window).mean()
            metrics['mvrv_zscore'] = (mvrv - mvrv.rolling(self.analysis_window).mean()) / \
                                    (mvrv.rolling(self.analysis_window).std() + 1e-8)

        # NVT比率
        if 'market_cap' in market_data.columns and 'transaction_volume' in on_chain_data.columns:
            nvt = market_data['market_cap'] / (on_chain_data['transaction_volume'].rolling(30).sum() + 1e-8)
            metrics['nvt_ratio'] = nvt.rolling(self.analysis_window).mean()
            metrics['nvt_zscore'] = (nvt - nvt.rolling(self.analysis_window).mean()) / \
                                   (nvt.rolling(self.analysis_window).std() + 1e-8)

        # SOPR（支出产出利润率）
        if 'sopr' in on_chain_data.columns:
            sopr = on_chain_data['sopr']
            metrics['sopr_ma'] = sopr.rolling(self.analysis_window).mean()
            metrics['sopr_deviation'] = (sopr - 1).rolling(self.analysis_window).mean()  # 偏离1的程度

        # 恐慌贪婪指数（链上版本）
        panic_greed = self._calculate_on_chain_panic_greed(on_chain_data, market_data)
        metrics['on_chain_panic_greed'] = panic_greed.rolling(self.analysis_window).mean()

        # 网络价值与交易比率
        if 'network_value' in on_chain_data.columns and 'economic_activity' in on_chain_data.columns:
            nvt_network = on_chain_data['network_value'] / (on_chain_data['economic_activity'] + 1e-8)
            metrics['network_value_ratio'] = nvt_network.rolling(self.analysis_window).mean()

        return metrics

    def _calculate_on_chain_panic_greed(self, on_chain_data: pd.DataFrame,
                                      market_data: pd.DataFrame) -> pd.Series:
        """计算链上版本的恐慌贪婪指数"""
        panic_greed_values = []

        for i in range(len(on_chain_data)):
            try:
                score = 50  # 中性

                # MVRV因子 (0-100)
                if 'market_cap' in market_data.columns and 'realized_cap' in on_chain_data.columns:
                    mvrv = market_data['market_cap'].iloc[i] / (on_chain_data['realized_cap'].iloc[i] + 1e-8)
                    if mvrv < 1:
                        score += 20  # 恐慌
                    elif mvrv > 3:
                        score -= 20  # 贪婪

                # 交易所流入因子 (0-100)
                if 'exchange_inflow' in on_chain_data.columns:
                    inflow_change = on_chain_data['exchange_inflow'].pct_change().iloc[i] if i > 0 else 0
                    if inflow_change > 0.5:  # 流入增加50%+
                        score -= 15  # 恐慌（可能在卖出）
                    elif inflow_change < -0.5:  # 流入减少50%+
                        score += 15  # 贪婪（可能从交易所提币）

                # 大户持币因子 (0-100)
                if 'whale_balance' in on_chain_data.columns and 'total_supply' in on_chain_data.columns:
                    whale_ratio = on_chain_data['whale_balance'].iloc[i] / (on_chain_data['total_supply'].iloc[i] + 1e-8)
                    if whale_ratio > 0.6:  # 大户持币过高
                        score -= 10  # 恐慌
                    elif whale_ratio < 0.3:  # 大户持币较低
                        score += 10  # 贪婪

                # 新地址增长因子 (0-100)
                if 'new_addresses' in on_chain_data.columns:
                    addr_growth = on_chain_data['new_addresses'].pct_change().iloc[i] if i > 0 else 0
                    if addr_growth > 0.5:
                        score += 15  # 贪婪（新用户增加）
                    elif addr_growth < -0.2:
                        score -= 15  # 恐慌（用户流失）

                panic_greed_values.append(np.clip(score, 0, 100))

            except:
                panic_greed_values.append(50)

        return pd.Series(panic_greed_values, index=on_chain_data.index)

    def detect_on_chain_signals(self, on_chain_data: pd.DataFrame,
                               market_data: pd.DataFrame) -> Dict[str, Any]:
        """
        检测链上交易信号

        Args:
            on_chain_data: 链上数据
            market_data: 市场数据

        Returns:
            链上信号分析结果
        """
        signals = {}

        # 计算各类指标
        network_metrics = self.calculate_network_activity_metrics(on_chain_data)
        holder_metrics = self.calculate_holder_behavior_metrics(on_chain_data)
        exchange_metrics = self.calculate_exchange_flow_metrics(on_chain_data)
        sentiment_metrics = self.calculate_market_sentiment_indicators(on_chain_data, market_data)

        # 大户动向信号
        if 'whale_balance_change' in holder_metrics:
            whale_change = holder_metrics['whale_balance_change'].iloc[-1]
            if whale_change > 0.05:  # 大户增持5%+
                signals['whale_accumulation'] = {'signal': 'buy', 'strength': min(3, whale_change * 20)}
            elif whale_change < -0.05:  # 大户减持5%+
                signals['whale_distribution'] = {'signal': 'sell', 'strength': min(3, abs(whale_change) * 20)}

        # 交易所流量信号
        if 'exchange_net_flow' in exchange_metrics:
            net_flow = exchange_metrics['exchange_net_flow'].iloc[-1]
            flow_ratio = exchange_metrics['exchange_flow_ratio'].iloc[-1]

            if net_flow < 0 and flow_ratio < -0.3:  # 净流出，流出比率高
                signals['exchange_outflow'] = {'signal': 'buy', 'strength': min(3, abs(flow_ratio) * 3)}
            elif net_flow > 0 and flow_ratio > 0.3:  # 净流入，流入比率高
                signals['exchange_inflow'] = {'signal': 'sell', 'strength': min(3, flow_ratio * 3)}

        # MVRV信号
        if 'mvrv_zscore' in sentiment_metrics:
            mvrv_z = sentiment_metrics['mvrv_zscore'].iloc[-1]
            if mvrv_z < -2:  # 严重低估
                signals['mvrv_undervalued'] = {'signal': 'buy', 'strength': min(3, abs(mvrv_z) / 2)}
            elif mvrv_z > 2:  # 严重高估
                signals['mvrv_overvalued'] = {'signal': 'sell', 'strength': min(3, mvrv_z / 2)}

        # SOPR信号
        if 'sopr_ma' in sentiment_metrics:
            sopr = sentiment_metrics['sopr_ma'].iloc[-1]
            if sopr < 0.95:  # 亏损卖出
                signals['sopr_capitulation'] = {'signal': 'buy', 'strength': min(3, (1 - sopr) * 20)}
            elif sopr > 1.05:  # 获利了结
                signals['sopr_profit_taking'] = {'signal': 'sell', 'strength': min(3, (sopr - 1) * 20)}

        # 网络活动信号
        if 'transaction_growth' in network_metrics:
            tx_growth = network_metrics['transaction_growth'].iloc[-1]
            if tx_growth > 0.3:  # 交易活跃度大幅增加
                signals['network_activity_spike'] = {'signal': 'buy', 'strength': min(3, tx_growth * 3)}
            elif tx_growth < -0.2:  # 交易活跃度大幅减少
                signals['network_activity_decline'] = {'signal': 'sell', 'strength': min(3, abs(tx_growth) * 3)}

        # 恐慌贪婪信号
        if 'on_chain_panic_greed' in sentiment_metrics:
            fear_greed = sentiment_metrics['on_chain_panic_greed'].iloc[-1]
            if fear_greed < 25:  # 极度恐慌
                signals['extreme_fear'] = {'signal': 'buy', 'strength': 3}
            elif fear_greed > 75:  # 极度贪婪
                signals['extreme_greed'] = {'signal': 'sell', 'strength': 3}

        return signals

    def calculate_on_chain_indicators(self, on_chain_data: pd.DataFrame,
                                   market_data: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有链上指标

        Args:
            on_chain_data: 链上数据
            market_data: 市场数据

        Returns:
            链上指标DataFrame
        """
        if len(on_chain_data) < self.analysis_window:
            return pd.DataFrame()

        # 计算各类指标
        network_metrics = self.calculate_network_activity_metrics(on_chain_data)
        holder_metrics = self.calculate_holder_behavior_metrics(on_chain_data)
        exchange_metrics = self.calculate_exchange_flow_metrics(on_chain_data)
        mining_metrics = self.calculate_mining_staking_metrics(on_chain_data)
        supply_metrics = self.calculate_supply_dynamics_metrics(on_chain_data)
        sentiment_metrics = self.calculate_market_sentiment_indicators(on_chain_data, market_data)

        # 创建指标DataFrame
        indicators = pd.DataFrame(index=on_chain_data.index[-1:])

        # 添加网络活动指标
        for name, series in network_metrics.items():
            indicators[f'network_{name}'] = series.iloc[-1]

        # 添加持币者行为指标
        for name, series in holder_metrics.items():
            indicators[f'holder_{name}'] = series.iloc[-1]

        # 添加交易所流量指标
        for name, series in exchange_metrics.items():
            indicators[f'exchange_{name}'] = series.iloc[-1]

        # 添加挖矿/质押指标
        for name, series in mining_metrics.items():
            indicators[f'mining_{name}'] = series.iloc[-1]

        # 添加供应动态指标
        for name, series in supply_metrics.items():
            indicators[f'supply_{name}'] = series.iloc[-1]

        # 添加市场情绪指标
        for name, series in sentiment_metrics.items():
            indicators[f'sentiment_{name}'] = series.iloc[-1]

        # 检测信号
        signals = self.detect_on_chain_signals(on_chain_data, market_data)

        # 计算综合信号强度
        buy_strength = sum(signal['strength'] for signal in signals.values() if signal['signal'] == 'buy')
        sell_strength = sum(signal['strength'] for signal in signals.values() if signal['signal'] == 'sell')

        indicators['buy_signal_strength'] = buy_strength
        indicators['sell_signal_strength'] = sell_strength
        indicators['net_signal'] = buy_strength - sell_strength
        indicators['signal_count'] = len(signals)

        return indicators


class DeFiAnalyzer:
    """
    DeFi分析器

    专门分析去中心化金融协议的链上数据
    """

    def __init__(self, defi_window: int = 30):
        """
        初始化DeFi分析器

        Args:
            defi_window: DeFi分析窗口
        """
        self.defi_window = defi_window

    def calculate_defi_metrics(self, defi_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算DeFi指标

        Args:
            defi_data: DeFi协议数据

        Returns:
            DeFi指标字典
        """
        metrics = {}

        # TVL (总锁仓价值)
        if 'tvl' in defi_data.columns:
            tvl = defi_data['tvl']
            metrics['tvl_ma'] = tvl.rolling(self.defi_window).mean()
            metrics['tvl_growth'] = tvl.pct_change().rolling(self.defi_window).mean()

        # 收益率
        if 'apy' in defi_data.columns:
            apy = defi_data['apy']
            metrics['apy_ma'] = apy.rolling(self.defi_window).mean()
            metrics['apy_volatility'] = apy.rolling(self.defi_window).std()

        # 清算率
        if 'liquidation_rate' in defi_data.columns:
            liquidation = defi_data['liquidation_rate']
            metrics['liquidation_risk'] = liquidation.rolling(self.defi_window).mean()

        # 利用率
        if 'utilization_rate' in defi_data.columns:
            utilization = defi_data['utilization_rate']
            metrics['utilization_ma'] = utilization.rolling(self.defi_window).mean()

        # 协议收入
        if 'protocol_revenue' in defi_data.columns:
            revenue = defi_data['protocol_revenue']
            metrics['revenue_growth'] = revenue.pct_change().rolling(self.defi_window).mean()

        return metrics

    def calculate_defi_risk_metrics(self, defi_data: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        计算DeFi风险指标

        Args:
            defi_data: DeFi协议数据

        Returns:
            DeFi风险指标字典
        """
        metrics = {}

        # 智能合约风险
        if 'contract_risk_score' in defi_data.columns:
            contract_risk = defi_data['contract_risk_score']
            metrics['contract_risk_ma'] = contract_risk.rolling(self.defi_window).mean()

        # 流动性风险
        if 'liquidity_score' in defi_data.columns:
            liquidity_score = defi_data['liquidity_score']
            metrics['liquidity_risk'] = 1 - liquidity_score.rolling(self.defi_window).mean()

        # 清算风险
        if 'collateral_ratio' in defi_data.columns:
            collateral_ratio = defi_data['collateral_ratio']
            metrics['collateral_health'] = collateral_ratio.rolling(self.defi_window).mean()
            metrics['liquidation_threshold_distance'] = (collateral_ratio - 1.5).rolling(self.defi_window).mean()

        return metrics


def create_on_chain_features(on_chain_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    """
    创建链上特征

    Args:
        on_chain_data: 链上数据
        market_data: 市场数据

    Returns:
        链上特征DataFrame
    """
    analyzer = OnChainAnalyzer()
    indicators = analyzer.calculate_on_chain_indicators(on_chain_data, market_data)
    return indicators


# 主要功能函数
def calculate_on_chain_indicators(on_chain_data: pd.DataFrame, market_data: pd.DataFrame) -> pd.DataFrame:
    """
    计算所有链上指标

    Args:
        on_chain_data: 链上数据DataFrame
        market_data: 市场数据DataFrame

    Returns:
        包含所有指标值的DataFrame
    """
    if len(on_chain_data) < 30:
        raise ValueError("链上数据长度不足，至少需要30个数据点")

    return create_on_chain_features(on_chain_data, market_data)


# 示例使用
if __name__ == "__main__":
    # 生成示例链上数据
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')

    # 模拟链上数据
    sample_on_chain = pd.DataFrame({
        'transactions': np.random.randint(1000000, 5000000, 200),
        'active_addresses': np.random.randint(500000, 2000000, 200),
        'whale_balance': np.random.uniform(10000000, 50000000, 200),
        'total_supply': 21000000,  # BTC总量
        'exchange_inflow': np.random.uniform(1000, 10000, 200),
        'exchange_outflow': np.random.uniform(1000, 10000, 200),
        'gas_price': np.random.uniform(10, 200, 200),
        'realized_cap': np.random.uniform(100000000000, 500000000000, 200),
        'sopr': np.random.uniform(0.8, 1.2, 200)
    }, index=dates)

    # 模拟市场数据
    sample_market = pd.DataFrame({
        'close': np.random.uniform(20000, 60000, 200),
        'market_cap': sample_on_chain['whale_balance'] * np.random.uniform(20000, 60000, 200),
        'volume': np.random.randint(1000000000, 50000000000, 200)
    }, index=dates)

    # 计算指标
    try:
        indicators = calculate_on_chain_indicators(sample_on_chain, sample_market)
        print("链上指标计算成功!")
        print(f"指标数量: {indicators.shape[1]}")
        print("最新指标值:")
        print(indicators.iloc[-1])

        # 检测信号
        analyzer = OnChainAnalyzer()
        signals = analyzer.detect_on_chain_signals(sample_on_chain, sample_market)
        print(f"\n检测到 {len(signals)} 个链上信号:")
        for signal_name, signal_info in signals.items():
            print(f"{signal_name}: {signal_info['signal']} (强度: {signal_info['strength']:.1f})")

    except Exception as e:
        print(f"计算错误: {e}")