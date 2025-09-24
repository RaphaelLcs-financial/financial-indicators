"""
金融技术指标Python实现库
Financial Technical Indicators Python Library

这个库包含了多种常用的金融技术指标实现，包括：
- 趋势指标 (Trend Indicators)
- 动量指标 (Momentum Indicators)
- 波动率指标 (Volatility Indicators)
- 成交量指标 (Volume Indicators)
"""

# 趋势指标
from .trend.moving_averages import MovingAverages
from .trend.macd import MACD

# 动量指标
from .momentum.rsi import RSI
from .momentum.kdj import KDJ
from .momentum.williams_r import WilliamsR

# 波动率指标
from .volatility.bollinger_bands import BollingerBands
from .volatility.atr import ATR

# 成交量指标
from .volume.obv import OBV

# 版本信息
__version__ = "1.0.0"
__author__ = "Financial Indicators Team"
__email__ = "contact@financial-indicators.com"

# 所有可用的指标类
__all__ = [
    # 趋势指标
    'MovingAverages',
    'MACD',

    # 动量指标
    'RSI',
    'KDJ',
    'WilliamsR',

    # 波动率指标
    'BollingerBands',
    'ATR',

    # 成交量指标
    'OBV',
]

# 指标分类
INDICATOR_CATEGORIES = {
    'trend': ['MovingAverages', 'MACD'],
    'momentum': ['RSI', 'KDJ', 'WilliamsR'],
    'volatility': ['BollingerBands', 'ATR'],
    'volume': ['OBV']
}

# 指标描述
INDICATOR_DESCRIPTIONS = {
    'MovingAverages': '移动平均线指标集合，包括SMA、EMA、WMA、HMA等',
    'MACD': '移动平均收敛发散指标，趋势跟踪动量指标',
    'RSI': '相对强弱指数，动量指标，识别超买超卖',
    'KDJ': '随机指标，动量指标，比较收盘价与价格范围',
    'WilliamsR': '威廉指标，动量指标，衡量超买超卖状态',
    'BollingerBands': '布林带，波动性指标，三条轨道线组成',
    'ATR': '平均真实范围，波动性指标，衡量市场波动性',
    'OBV': '能量潮，成交量指标，通过成交量累积预测价格走势'
}

def get_all_indicators():
    """
    获取所有可用的指标类

    返回:
        dict: 指标名称到类的映射
    """
    import sys
    current_module = sys.modules[__name__]

    indicators = {}
    for name in __all__:
        if hasattr(current_module, name):
            indicators[name] = getattr(current_module, name)

    return indicators

def get_indicators_by_category(category):
    """
    根据分类获取指标

    参数:
        category (str): 指标分类 ('trend', 'momentum', 'volatility', 'volume')

    返回:
        dict: 该分类下的指标类
    """
    if category not in INDICATOR_CATEGORIES:
        raise ValueError(f"未知的指标分类: {category}")

    import sys
    current_module = sys.modules[__name__]

    indicators = {}
    for name in INDICATOR_CATEGORIES[category]:
        if hasattr(current_module, name):
            indicators[name] = getattr(current_module, name)

    return indicators

def calculate_indicator(indicator_name, data, **kwargs):
    """
    便捷函数：计算指定指标

    参数:
        indicator_name (str): 指标名称
        data: 价格数据
        **kwargs: 指标参数

    返回:
        指标计算结果
    """
    import sys
    current_module = sys.modules[__name__]

    if not hasattr(current_module, indicator_name):
        raise ValueError(f"未知指标: {indicator_name}")

    indicator_class = getattr(current_module, indicator_name)

    # 根据指标类型调用相应的计算方法
    if indicator_name == 'MovingAverages':
        return indicator_class.sma(data, **kwargs)
    elif indicator_name == 'MACD':
        return indicator_class.calculate(data, **kwargs)
    elif indicator_name == 'RSI':
        return indicator_class.calculate(data, **kwargs)
    elif indicator_name == 'KDJ':
        return indicator_class.calculate(data['high'], data['low'], data['close'], **kwargs)
    elif indicator_name == 'WilliamsR':
        return indicator_class.calculate(data['high'], data['low'], data['close'], **kwargs)
    elif indicator_name == 'BollingerBands':
        return indicator_class.calculate(data, **kwargs)
    elif indicator_name == 'ATR':
        return indicator_class.calculate(data['high'], data['low'], data['close'], **kwargs)
    elif indicator_name == 'OBV':
        return indicator_class.calculate(data['close'], data['volume'], **kwargs)
    else:
        raise ValueError(f"不支持的指标: {indicator_name}")

# 使用示例
if __name__ == "__main__":
    print("金融技术指标库 v1.0.0")
    print("可用指标:")
    for category, indicators in INDICATOR_CATEGORIES.items():
        print(f"  {category}: {', '.join(indicators)}")

    print(f"\n总共包含 {len(__all__)} 个专业金融指标")
    print("每个指标都包含完整的计算、信号生成和分析功能")