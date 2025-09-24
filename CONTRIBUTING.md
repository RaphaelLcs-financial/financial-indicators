# 贡献指南 | Contributing Guide

感谢您对金融技术指标开源库的关注！我们欢迎所有形式的贡献。

## 🤝 如何贡献

### 1. 报告问题 (Bug Reports)
- 使用 GitHub Issues 报告 bug
- 提供详细的重现步骤
- 包含错误信息和环境信息

### 2. 功能请求 (Feature Requests)
- 在 Issues 中提出新功能建议
- 描述功能用途和应用场景
- 提供可能的实现思路

### 3. 代码贡献 (Code Contributions)
- Fork 本项目
- 创建功能分支
- 提交代码并创建 Pull Request

## 📝 代码规范

### Python 代码风格
- 遵循 PEP 8 规范
- 使用 4 个空格缩进
- 行长度限制在 120 字符
- 使用类型注解

### 指标实现标准
```python
from typing import Union, Dict, Any
import pandas as pd
import numpy as np

class YourIndicator:
    """
    指标描述

    参数:
        param1: 参数1说明
        param2: 参数2说明

    返回:
        指标值和交易信号
    """

    def __init__(self, param1: int = 14, param2: int = 3):
        self.param1 = param1
        self.param2 = param2

    def calculate(self, data: pd.DataFrame) -> pd.Series:
        """
        计算指标值

        参数:
            data: 包含OHLCV数据的DataFrame

        返回:
            指标值序列
        """
        # 实现指标计算逻辑
        pass

    def get_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        生成交易信号

        参数:
            data: 包含OHLCV数据的DataFrame

        返回:
            交易信号序列 (1=买入, -1=卖出, 0=持有)
        """
        # 实现信号生成逻辑
        pass
```

### 文档要求
- 所有函数和类必须有详细的文档字符串
- 包含参数说明、返回值说明和示例
- 复杂算法需要添加注释说明

### 测试要求
- 新功能必须包含单元测试
- 测试覆盖率不低于 80%
- 使用 pytest 框架

## 🚀 开发环境设置

### 1. 克隆项目
```bash
git clone https://github.com/RaphaelLcs-financial/financial-indicators.git
cd financial-indicators
```

### 2. 创建虚拟环境
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
pip install -e .
```

### 4. 安装开发依赖
```bash
pip install -e .[dev]
```

### 5. 运行测试
```bash
pytest tests/
```

### 6. 代码格式化
```bash
black indicators/
flake8 indicators/
```

## 📋 Pull Request 流程

### 1. 分支命名
- 功能分支: `feature/your-feature-name`
- 修复分支: `fix/your-fix-name`
- 文档分支: `docs/your-doc-change`

### 2. 提交信息
```
类型(范围): 描述

# 例如:
feat(indicators): 添加新的RSI指标实现
fix(utils): 修复数据加载器的内存泄漏
docs(readme): 更新安装说明
```

### 3. PR 描述
- 清晰描述变更内容
- 说明解决的问题
- 包含测试结果
- 关联相关的 Issue

## 🏗️ 项目结构

```
financial-indicators/
├── indicators/           # 指标实现
│   ├── python/         # Python实现
│   │   ├── traditional/   # 传统指标
│   │   ├── machine_learning/ # 机器学习指标
│   │   └── quantum/      # 量子金融指标
│   ├── javascript/     # JavaScript实现
│   └── documentation/  # 指标文档
├── tests/              # 测试代码
├── docs/               # 文档
├── examples/           # 示例代码
├── data/              # 数据和配置
└── tools/             # 工具脚本
```

## 🎯 贡献方向

### 急需的贡献
- 新的金融指标实现
- 性能优化
- 文档完善
- 测试用例补充
- 多语言支持

### 特别欢迎的指标类型
- 传统技术指标的改进版本
- 基于机器学习的预测指标
- 加密货币专用指标
- 高频交易指标
- 风险管理指标

## 📞 联系方式

- GitHub Issues: [项目Issues页面](https://github.com/RaphaelLcs-financial/financial-indicators/issues)
- 邮箱: contact@financial-indicators.com
- QQ群: [群号]

## 📄 许可证

通过贡献代码，您同意您的贡献将在 [MIT License](LICENSE) 下发布。

---

感谢您的贡献！🎉