/**
 * Vortex Indicator (涡旋指标) - JavaScript实现
 * 一种相对较新的技术指标，用于识别趋势的开始和反转
 *
 * @author Financial Indicators Team
 * @version 1.0.0
 */

class VortexIndicator {
    constructor() {
        this.name = 'Vortex Indicator';
        this.category = 'trend';
    }

    /**
     * 计算Vortex指标
     * @param {Array} data - 包含high, low, close的数据数组
     * @param {number} period - 计算周期，默认14
     * @returns {Object} 包含VIPlus, VIMinus和信号的指标数据
     */
    calculate(data, period = 14) {
        if (!data || data.length < period + 1) {
            throw new Error('数据长度不足');
        }

        const high = data.map(d => d.high);
        const low = data.map(d => d.low);
        const close = data.map(d => d.close);

        const viPlus = [];
        const viMinus = [];
        const trueRange = [];

        // 计算Vortex指标
        for (let i = 1; i < data.length; i++) {
            // 计算真实范围
            const tr = Math.max(
                high[i] - low[i],
                Math.abs(high[i] - close[i - 1]),
                Math.abs(low[i] - close[i - 1])
            );
            trueRange.push(tr);

            // 计算VM+和VM-
            const vmPlus = Math.abs(high[i] - low[i - 1]);
            const vmMinus = Math.abs(low[i] - high[i - 1]);

            if (i >= period) {
                // 计算周期内的总和
                let sumTr = 0;
                let sumVmPlus = 0;
                let sumVmMinus = 0;

                for (let j = i - period + 1; j <= i; j++) {
                    sumTr += trueRange[j - 1];
                    sumVmPlus += Math.abs(high[j] - low[j - 1]);
                    sumVmMinus += Math.abs(low[j] - high[j - 1]);
                }

                viPlus.push(sumVmPlus / sumTr);
                viMinus.push(sumVmMinus / sumTr);
            } else {
                viPlus.push(null);
                viMinus.push(null);
            }
        }

        return {
            viPlus: viPlus,
            viMinus: viMinus,
            trueRange: trueRange
        };
    }

    /**
     * 生成交易信号
     * @param {Object} vortexData - Vortex指标数据
     * @returns {Array} 信号数组 (1=买入, -1=卖出, 0=持有)
     */
    generateSignals(vortexData) {
        const signals = [];
        const { viPlus, viMinus } = vortexData;

        for (let i = 0; i < viPlus.length; i++) {
            if (viPlus[i] === null || viMinus[i] === null) {
                signals.push(0);
                continue;
            }

            // VI+上穿VI- - 买入信号
            if (viPlus[i] > viMinus[i] && viPlus[i - 1] <= viMinus[i - 1]) {
                signals.push(1);
            }
            // VI+下穿VI- - 卖出信号
            else if (viPlus[i] < viMinus[i] && viPlus[i - 1] >= viMinus[i - 1]) {
                signals.push(-1);
            }
            // 持有信号
            else {
                signals.push(0);
            }
        }

        return signals;
    }

    /**
     * 计算涡旋振荡器
     * @param {Object} vortexData - Vortex指标数据
     * @returns {Array} 振荡器值数组
     */
    calculateOscillator(vortexData) {
        const oscillator = [];
        const { viPlus, viMinus } = vortexData;

        for (let i = 0; i < viPlus.length; i++) {
            if (viPlus[i] === null || viMinus[i] === null) {
                oscillator.push(null);
            } else {
                oscillator.push(viPlus[i] - viMinus[i]);
            }
        }

        return oscillator;
    }
}

// 导出模块
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VortexIndicator;
} else if (typeof window !== 'undefined') {
    window.VortexIndicator = VortexIndicator;
}

// 使用示例
if (typeof window !== 'undefined' && window.document) {
    // 浏览器环境示例
    window.testVortex = function() {
        const testData = [
            { high: 105, low: 95, close: 100 },
            { high: 106, low: 96, close: 101 },
            { high: 107, low: 97, close: 102 },
            { high: 108, low: 98, close: 103 },
            { high: 109, low: 99, close: 104 },
            { high: 110, low: 100, close: 105 },
            { high: 111, low: 101, close: 106 },
            { high: 112, low: 102, close: 107 },
            { high: 113, low: 103, close: 108 },
            { high: 114, low: 104, close: 109 },
            { high: 115, low: 105, close: 110 },
            { high: 116, low: 106, close: 111 },
            { high: 117, low: 107, close: 112 },
            { high: 118, low: 108, close: 113 },
            { high: 119, low: 109, close: 114 }
        ];

        const vortex = new VortexIndicator();
        const result = vortex.calculate(testData, 14);
        const signals = vortex.generateSignals(result);
        const oscillator = vortex.calculateOscillator(result);

        console.log('Vortex Indicator Test:');
        console.log('VI+:', result.viPlus[result.viPlus.length - 1]);
        console.log('VI-:', result.viMinus[result.viMinus.length - 1]);
        console.log('Signal:', signals[signals.length - 1]);
        console.log('Oscillator:', oscillator[oscillator.length - 1]);
    };
}