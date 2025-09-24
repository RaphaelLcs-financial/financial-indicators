"""
高级机器学习交易策略示例
Advanced Machine Learning Trading Strategy Example
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入传统指标
from indicators.python.trend.moving_averages import MovingAverages
from indicators.python.momentum.rsi import RSI
from indicators.python.volatility.atr import ATR
from indicators.python.volume.obv import OBV

# 导入机器学习指标
from indicators.python.machine_learning.ensemble_market_predictor import EnsembleMarketPredictor

def generate_sample_data(days=500):
    """生成更复杂的示例数据"""
    np.random.seed(42)

    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='D'
    )

    # 生成带趋势和噪声的价格数据
    t = np.arange(days)
    trend = 0.001 * t
    seasonal = 0.02 * np.sin(2 * np.pi * t / 50)  # 季节性
    noise = np.random.normal(0, 0.015, days)

    returns = trend + seasonal + noise
    price = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'date': dates,
        'open': price * (1 + np.random.normal(0, 0.003, days)),
        'high': price * (1 + np.random.normal(0.008, 0.003, days)),
        'low': price * (1 + np.random.normal(-0.008, 0.003, days)),
        'close': price,
        'volume': np.random.randint(500000, 5000000, days)
    })

    data.set_index('date', inplace=True)
    return data

def create_features(data):
    """创建特征矩阵"""
    features = pd.DataFrame(index=data.index)

    # 1. 价格特征
    features['returns'] = data['close'].pct_change()
    features['log_returns'] = np.log(data['close'] / data['close'].shift(1))
    features['price_range'] = (data['high'] - data['low']) / data['close']

    # 2. 移动平均特征
    ma = MovingAverages()
    features['sma_10'] = ma.sma(data['close'], 10)
    features['sma_30'] = ma.sma(data['close'], 30)
    features['ema_12'] = ma.ema(data['close'], 12)
    features['ema_26'] = ma.ema(data['close'], 26)

    # 3. 动量特征
    rsi = RSI()
    features['rsi_14'] = rsi.calculate(data['close'], 14)
    features['rsi_30'] = rsi.calculate(data['close'], 30)

    # 4. 波动率特征
    atr = ATR()
    features['atr_14'] = atr.calculate(data['high'], data['low'], data['close'], 14)
    features['volatility'] = features['returns'].rolling(20).std()

    # 5. 成交量特征
    obv = OBV()
    features['obv'] = obv.calculate(data['close'], data['volume'])
    features['volume_sma'] = data['volume'].rolling(20).mean()
    features['volume_ratio'] = data['volume'] / features['volume_sma']

    # 6. 价格位置特征
    features['price_position'] = (data['close'] - data['low'].rolling(20).min()) / \
                                 (data['high'].rolling(20).max() - data['low'].rolling(20).min())

    # 7. 滞后特征
    for lag in [1, 2, 3, 5, 10]:
        features[f'returns_lag_{lag}'] = features['returns'].shift(lag)

    return features

def create_labels(data, lookahead=5):
    """创建标签"""
    future_returns = data['close'].pct_change(lookahead).shift(-lookahead)

    # 三分类标签：-1=下跌, 0=震荡, 1=上涨
    labels = pd.Series(0, index=data.index)
    labels[future_returns > 0.02] = 1      # 上涨超过2%
    labels[future_returns < -0.02] = -1     # 下跌超过2%

    return labels

def main():
    """主函数"""
    print("🤖 高级机器学习交易策略示例")
    print("=" * 50)

    # 生成数据
    data = generate_sample_data()
    print(f"📊 生成数据: {len(data)} 天")

    # 创建特征
    print("\n🔧 创建特征...")
    features = create_features(data)
    labels = create_labels(data)

    # 移除缺失值
    combined = pd.concat([features, labels], axis=1)
    combined.columns = list(features.columns) + ['label']
    combined = combined.dropna()

    X = combined.iloc[:, :-1]
    y = combined.iloc[:, -1]

    print(f"✅ 特征矩阵: {X.shape}")
    print(f"✅ 标签分布: {y.value_counts().to_dict()}")

    # 分割数据
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # 训练传统随机森林
    print("\n🌲 训练随机森林模型...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)

    # 评估模型
    rf_pred = rf_model.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_pred)

    print(f"📊 随机森林准确率: {rf_accuracy:.3f}")
    print("\n分类报告:")
    print(classification_report(y_test, rf_pred, zero_division=0))

    # 使用集成学习预测器
    print("\n🚀 使用集成学习预测器...")
    predictor = EnsembleMarketPredictor(
        window_size=100,
        prediction_horizon=5
    )

    # 准备数据格式
    market_data = data.loc[X_test.index, ['open', 'high', 'low', 'close', 'volume']]

    try:
        ml_results = predictor.analyze(market_data)
        ml_predictions = predictor.get_trading_signals()

        print(f"✅ 集成学习预测完成")
        print(f"最新信号: {ml_predictions.iloc[-1] if len(ml_predictions) > 0 else 'N/A'}")
    except Exception as e:
        print(f"⚠️ 集成学习执行失败: {e}")
        ml_predictions = pd.Series(0, index=X_test.index)

    # 组合策略
    print("\n🎯 组合策略信号...")

    # 权重组合
    rf_signal = pd.Series(rf_pred, index=X_test.index)
    combined_signal = 0.6 * rf_signal + 0.4 * ml_predictions

    # 生成最终交易信号
    final_signals = pd.Series(0, index=X_test.index)
    final_signals[combined_signal > 0.3] = 1
    final_signals[combined_signal < -0.3] = -1

    # 回测分析
    print("\n📈 回测分析...")

    test_data = data.loc[X_test.index].copy()
    test_data['signal'] = final_signals
    test_data['returns'] = test_data['close'].pct_change()
    test_data['strategy_returns'] = test_data['signal'].shift(1) * test_data['returns']

    # 计算性能指标
    total_return = test_data['strategy_returns'].sum()
    annual_return = total_return * (252 / len(test_data))
    sharpe_ratio = test_data['strategy_returns'].mean() / test_data['strategy_returns'].std() * np.sqrt(252)
    max_drawdown = (test_data['strategy_returns'].cumsum().expanding().max() - test_data['strategy_returns'].cumsum()).max()

    print(f"总收益率: {total_return:.2%}")
    print(f"年化收益率: {annual_return:.2%}")
    print(f"夏普比率: {sharpe_ratio:.3f}")
    print(f"最大回撤: {max_drawdown:.2%}")

    # 特征重要性
    print("\n📊 特征重要性 (Top 10):")
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    for idx, row in feature_importance.iterrows():
        print(f"{row['feature']}: {row['importance']:.3f}")

    # 可视化结果
    print("\n📈 生成可视化图表...")

    fig, axes = plt.subplots(3, 1, figsize=(15, 12))

    # 价格和信号
    axes[0].plot(test_data.index, test_data['close'], label='Price', color='black')
    buy_signals = test_data[test_data['signal'] == 1]
    sell_signals = test_data[test_data['signal'] == -1]
    axes[0].scatter(buy_signals.index, buy_signals['close'], color='green', marker='^', s=100, label='Buy')
    axes[0].scatter(sell_signals.index, sell_signals['close'], color='red', marker='v', s=100, label='Sell')
    axes[0].set_title('Price and Trading Signals')
    axes[0].legend()
    axes[0].grid(True)

    # 累积收益
    cumulative_returns = test_data['strategy_returns'].cumsum()
    axes[1].plot(cumulative_returns.index, cumulative_returns, label='Strategy Returns', color='blue')
    axes[1].set_title('Cumulative Strategy Returns')
    axes[1].grid(True)

    # 特征重要性
    top_features = feature_importance.head(10)
    axes[2].barh(range(len(top_features)), top_features['importance'])
    axes[2].set_yticks(range(len(top_features)))
    axes[2].set_yticklabels(top_features['feature'])
    axes[2].set_title('Top 10 Feature Importance')
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('ml_strategy_results.png', dpi=300, bbox_inches='tight')
    print("✅ 结果图表已保存为: ml_strategy_results.png")

    print("\n🎉 高级策略示例完成！")
    print("\n💡 改进建议:")
    print("- 添加更多特征工程技术")
    print("- 尝试不同的机器学习模型")
    print("- 实现动态权重调整")
    print("- 加入风险管理模块")

if __name__ == "__main__":
    main()