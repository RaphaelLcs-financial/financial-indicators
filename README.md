# 金融技术指标开源库
# Financial Technical Indicators Open Source Library

## 📈 项目简介
这是一个收集和分享金融技术指标的开源项目，提供多种编程语言实现的交易指标，帮助量化交易者和开发者快速构建交易策略。

## 🎯 项目目标
- 🔍 收集整理各类金融技术指标
- 📚 提供多语言实现示例
- 🤝 建立技术分析社区
- 🚀 推广量化交易知识

## 📁 项目结构
```
financial-indicators/
├── indicators/           # 指标实现
│   ├── python/         # Python实现
│   ├── javascript/     # JavaScript实现
│   ├── tradingview/    # TradingView Pine Script
│   └── documentation/  # 指标说明文档
├── strategies/         # 策略示例
│   ├── trend-following/  # 趋势跟踪策略
│   ├── mean-reversion/   # 均值回归策略
│   └── momentum/         # 动量策略
├── data/              # 数据处理工具
│   ├── fetchers/       # 数据获取工具
│   ├── cleaners/       # 数据清洗工具
│   └── validators/      # 数据验证工具
├── backtesting/       # 回测框架
│   ├── engines/        # 回测引擎
│   ├── metrics/        # 性能指标
│   └── visualizers/    # 可视化工具
├── docs/              # 文档
│   ├── tutorials/      # 教程
│   ├── api-reference/  # API参考
│   └── examples/       # 示例代码
└── community/         # 社区贡献
    ├── indicators/     # 用户贡献指标
    ├── strategies/     # 用户贡献策略
    └── discussions/    # 讨论区
```

## 🛠️ 技术栈
- **Python**: 主要开发语言
- **JavaScript**: 前端和Web应用
- **Pandas**: 数据处理
- **NumPy**: 数值计算
- **Matplotlib/Plotly**: 数据可视化
- **FastAPI**: Web API服务

## 📋 已实现指标分类

### 🚀 趋势指标 (Trend Indicators)
- ✅ **移动平均线 (MA, EMA, SMA, WMA)** - 完整实现
- ✅ **HMA (赫尔移动平均线)** - 创新移动平均线，显著减少滞后性
- ✅ **Supertrend (超级趋势)** - 结合ATR的趋势跟踪指标，2020年后流行
- ✅ **MACD** - 移动平均收敛发散指标，包含信号线和直方图
- ✅ **Vortex Indicator (涡旋指标)** - 较新的趋势识别指标
- ✅ **ADX (平均趋向指数)** - 趋势强度指标，衡量价格趋势强度
- 🔄 布林带 (Bollinger Bands) - 已完成 (归入波动率指标)
- 🔄 抛物线SAR (Parabolic SAR) - 计划中

### 📈 动量指标 (Momentum Indicators)
- ✅ **RSI** - 相对强弱指数，包含背离分析
- ✅ **KDJ** - 随机指标，多时间框架分析
- ✅ **Williams %R** - 威廉指标，失败摆动检测
- ✅ **TMO (True Momentum Oscillator)** - 真实动量振荡器，较新的动量指标
- ✅ **Woodie's CCI** - 伍迪商品通道指数，传统CCI变种
- ✅ **QQE (定量定性估计)** - RSI的改良版，结合RSI和ATR
- ✅ **Fisher Transform (费雪变换)** - 将价格转换为高斯正态分布
- ✅ **Ultimate Oscillator (终极振荡器)** - 多时间框架动量指标
- 🔄 CCI (商品通道指数) - 计划中

### 📊 成交量指标 (Volume Indicators)
- ✅ **OBV** - 能量潮指标，加权OBV实现
- ✅ **Chaikin Money Flow (蔡金资金流)** - 结合价格和成交量的资金流指标
- 🔄 成交量移动平均 - 计划中
- 🔄 A/D Line (累积/派发线) - 计划中
- 🔄 MFI (资金流量指数) - 计划中

### 📉 波动率指标 (Volatility Indicators)
- ✅ **ATR** - 平均真实范围，止损位和通道计算
- ✅ **布林带** - 波动性突破策略
- 🔄 标准差 - 计划中
- 🔄 历史波动率 - 计划中

## 🔥 最新更新 (v2.0.0) - 统一回测系统上线！

### 🎯 核心新功能
- **🚀 统一回测引擎** - 任意指标一键转换为交易策略
- **🌍 多市场数据源** - 支持A股、期货、外汇、加密货币四大市场
- **📊 完整性能分析** - 自动计算30+项回测指标
- **📈 专业可视化** - 权益曲线、回撤分析、交易统计等图表
- **⚡ 简单易用** - 导入指标即可回测，无需复杂配置

### 🆕 新增最新指标
- **TMO (True Momentum Oscillator)** - 2020年后流行的真实动量振荡器
- **Supertrend (超级趋势)** - 结合ATR的创新趋势指标，非常流行
- **HMA (赫尔移动平均线)** - 显著减少滞后性的创新移动平均线
- **Vortex Indicator (涡旋指标)** - 较新的趋势识别指标
- **Woodie's CCI** - 传统CCI的变种，不同参数和解释方法

### 🌐 多市场数据支持
- ✅ **A股市场** - akshare数据源，支持前复权
- ✅ **期货市场** - 主力合约数据，自动换月
- ✅ **外汇市场** - 主要货币对，24小时数据
- ✅ **加密货币** - 主流数字货币，实时价格

### 🔬 回测系统特性
- **真实交易成本** - 手续费、滑点、冲击成本模拟
- **风险管理** - 止损止盈、仓位管理、资金管理
- **多策略对比** - 同时回测多个策略并对比分析
- **参数优化** - 支持策略参数的批量测试优化
- **样本外验证** - 训练集优化，测试集验证

### 📊 性能指标体系
- **收益指标**: 总收益率、年化收益率、夏普比率、索提诺比率
- **风险指标**: 最大回撤、波动率、VaR、CVaR、贝塔系数
- **交易指标**: 胜率、盈利因子、平均盈亏、连续盈亏次数

## 🚀 快速开始

### 安装依赖
```bash
pip install -r requirements.txt
```

### 基础指标使用
```python
from indicators.python.trend.moving_averages import MovingAverages
from indicators.python.momentum.rsi import RSI
from data.fetchers.unified_fetcher import UnifiedDataFetcher

# 获取数据
fetcher = UnifiedDataFetcher()
data = fetcher.fetch_data('BTC-USD', 'crypto', period='1y')

# 计算指标
sma = MovingAverages.sma(data['close'], period=20)
rsi = RSI.calculate(data['close'], period=14)

print(f"SMA(20): {sma.iloc[-1]:.2f}")
print(f"RSI(14): {rsi.iloc[-1]:.2f}")
```

### 🎯 统一回测系统 - 核心特性！

**一键导入，即刻回测！** 任何指标都可以快速转换为交易策略进行回测：

```python
from backtesting.engines.simple_backtest_engine import SimpleBacktestEngine
from backtesting.engines.strategy_base import StrategyBase

# 创建简单RSI策略
class RSIStrategy(StrategyBase):
    def generate_signals(self, data):
        rsi = RSI.calculate(data['close'], 14)
        signals = pd.Series(0, index=data.index)
        signals[rsi < 30] = 1   # 超卖买入
        signals[rsi > 70] = -1  # 超买卖出
        return signals

# 一键回测
engine = SimpleBacktestEngine(initial_capital=100000)
results = engine.run_backtest(data, RSIStrategy())

print(f"总收益率: {results['total_return_pct']:.2f}%")
print(f"夏普比率: {results['sharpe_ratio']:.3f}")
print(f"最大回撤: {results['max_drawdown']*100:.2f}%")
```

### 🌍 多市场支持
```python
# A股回测
stock_data = fetcher.fetch_data('000001', 'stock', period='2y')

# 外汇回测
forex_data = fetcher.fetch_data('EURUSD=X', 'forex', period='2y')

# 加密货币回测
crypto_data = fetcher.fetch_data('BTC-USD', 'crypto', period='2y')

# 期货回测
futures_data = fetcher.fetch_data('RB2401', 'futures', period='1y')
```

### 📊 完整回测示例
查看 `docs/examples/` 目录获取完整示例：

```bash
# 运行简单RSI策略回测
python docs/examples/simple_rsi_strategy.py

# 运行多指标组合策略回测
python docs/examples/multi_indicator_strategy.py
```

## 🤝 贡献指南
1. Fork 本项目
2. 创建特性分支 (`git checkout -b feature/amazing-indicator`)
3. 提交更改 (`git commit -m 'Add amazing indicator'`)
4. 推送到分支 (`git push origin feature/amazing-indicator`)
5. 创建Pull Request

## 📄 许可证
本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 📞 联系方式
- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]
- 讨论区: [Discussions]

## 🙏 致谢
感谢所有贡献者和社区成员的支持！

---

**免责声明**: 本项目提供的指标和策略仅用于教育和研究目的，不构成投资建议。投资有风险，请谨慎决策。