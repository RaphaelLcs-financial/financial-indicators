"""
金融指标基础使用示例
Basic Usage Examples for Financial Indicators
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# 导入指标库
from indicators.python.trend.moving_averages import MovingAverages
from indicators.python.momentum.rsi import RSI
from indicators.python.volatility.atr import ATR
from indicators.python.volume.obv import OBV

def generate_sample_data(days=252):
    """生成示例数据"""
    np.random.seed(42)

    dates = pd.date_range(
        start=datetime.now() - timedelta(days=days),
        end=datetime.now(),
        freq='D'
    )

    # 生成模拟价格数据
    returns = np.random.normal(0.001, 0.02, days)
    price = 100 * np.exp(np.cumsum(returns))

    data = pd.DataFrame({
        'date': dates,
        'open': price * (1 + np.random.normal(0, 0.005, days)),
        'high': price * (1 + np.random.normal(0.01, 0.005, days)),
        'low': price * (1 + np.random.normal(-0.01, 0.005, days)),
        'close': price,
        'volume': np.random.randint(1000000, 10000000, days)
    })

    data.set_index('date', inplace=True)
    return data

def main():
    """主函数"""
    print("🚀 金融指标基础使用示例")
    print("=" * 50)

    # 生成示例数据
    data = generate_sample_data()
    print(f"📊 生成数据: {len(data)} 天")

    # 1. 移动平均线指标
    print("\n📈 移动平均线指标")
    ma = MovingAverages()

    sma_20 = ma.sma(data['close'], period=20)
    ema_20 = ma.ema(data['close'], period=20)

    print(f"SMA(20) 最新值: {sma_20.iloc[-1]:.2f}")
    print(f"EMA(20) 最新值: {ema_20.iloc[-1]:.2f}")

    # 2. RSI指标
    print("\n📊 RSI指标")
    rsi = RSI()
    rsi_values = rsi.calculate(data['close'], period=14)

    print(f"RSI(14) 最新值: {rsi_values.iloc[-1]:.2f}")

    # 3. ATR指标
    print("\n📉 ATR指标")
    atr = ATR()
    atr_values = atr.calculate(data['high'], data['low'], data['close'], period=14)

    print(f"ATR(14) 最新值: {atr_values.iloc[-1]:.2f}")

    # 4. OBV指标
    print("\n📊 OBV指标")
    obv = OBV()
    obv_values = obv.calculate(data['close'], data['volume'])

    print(f"OBV 最新值: {obv_values.iloc[-1]:,.0f}")

    # 5. 创建交易信号
    print("\n🎯 交易信号生成")

    # 简单的RSI交易策略
    signals = pd.Series(0, index=data.index)
    signals[rsi_values < 30] = 1   # 超卖买入
    signals[rsi_values > 70] = -1  # 超买卖出

    print(f"当前信号: {'买入' if signals.iloc[-1] == 1 else '卖出' if signals.iloc[-1] == -1 else '持有'}")

    # 6. 可视化
    print("\n📈 生成图表...")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # 价格和移动平均线
    axes[0].plot(data.index, data['close'], label='Close Price', color='black')
    axes[0].plot(sma_20.index, sma_20, label='SMA(20)', color='blue')
    axes[0].plot(ema_20.index, ema_20, label='EMA(20)', color='red')
    axes[0].set_title('Price and Moving Averages')
    axes[0].legend()
    axes[0].grid(True)

    # RSI
    axes[1].plot(rsi_values.index, rsi_values, label='RSI(14)', color='purple')
    axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.5)
    axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.5)
    axes[1].set_title('RSI Indicator')
    axes[1].legend()
    axes[1].grid(True)

    # ATR
    axes[2].plot(atr_values.index, atr_values, label='ATR(14)', color='orange')
    axes[2].set_title('ATR Indicator')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig('indicators_example.png', dpi=300, bbox_inches='tight')
    print("✅ 图表已保存为: indicators_example.png")

    # 7. 性能统计
    print("\n📊 指标性能统计")

    # 计算信号准确率（简化版）
    future_returns = data['close'].pct_change().shift(-1)
    signal_accuracy = {
        'buy_signal': (future_returns[signals == 1] > 0).mean() if (signals == 1).any() else 0,
        'sell_signal': (future_returns[signals == -1] < 0).mean() if (signals == -1).any() else 0
    }

    print(f"买入信号准确率: {signal_accuracy['buy_signal']:.2%}")
    print(f"卖出信号准确率: {signal_accuracy['sell_signal']:.2%}")

    print("\n🎉 示例完成！")
    print("\n💡 提示:")
    print("- 查看更多指标在 indicators/python/ 目录")
    print("- 使用真实数据替换模拟数据")
    print("- 结合多个指标构建交易策略")

if __name__ == "__main__":
    main()