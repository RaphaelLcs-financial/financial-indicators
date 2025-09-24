/**
 * Woodie's CCI (伍迪商品通道指数) - JavaScript实现
 * 传统CCI指标的变种，使用不同的参数和解释方法
 *
 * @author Financial Indicators Team
 * @version 1.0.0
 */

class WoodiesCCI {
    constructor() {
        this.name = "Woodie's CCI";
        this.category = 'momentum';
        this.cciPeriod = 14;
        this.turboPeriod = 6;
    }

    /**
     * 计算典型价格
     * @param {number} high - 最高价
     * @param {number} low - 最低价
     * @param {number} close - 收盘价
     * @returns {number} 典型价格
     */
    calculateTypicalPrice(high, low, close) {
        return (high + low + close) / 3;
    }

    /**
     * 计算简单移动平均
     * @param {Array} data - 数据数组
     * @param {number} period - 周期
     * @returns {Array} 移动平均数组
     */
    calculateSMA(data, period) {
        const sma = [];
        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                sma.push(null);
            } else {
                const sum = data.slice(i - period + 1, i + 1).reduce((a, b) => a + b, 0);
                sma.push(sum / period);
            }
        }
        return sma;
    }

    /**
     * 计算平均偏差
     * @param {Array} data - 数据数组
     * @param {Array} sma - 移动平均数组
     * @param {number} period - 周期
     * @returns {Array} 平均偏差数组
     */
    calculateMeanDeviation(data, sma, period) {
        const meanDeviation = [];
        for (let i = 0; i < data.length; i++) {
            if (i < period - 1 || sma[i] === null) {
                meanDeviation.push(null);
            } else {
                const deviations = data.slice(i - period + 1, i + 1)
                    .map((val, idx) => Math.abs(val - sma[i]));
                const sumDeviations = deviations.reduce((a, b) => a + b, 0);
                meanDeviation.push(sumDeviations / period);
            }
        }
        return meanDeviation;
    }

    /**
     * 计算CCI
     * @param {Array} typicalPrices - 典型价格数组
     * @param {number} period - 周期
     * @returns {Array} CCI值数组
     */
    calculateCCI(typicalPrices, period) {
        const sma = this.calculateSMA(typicalPrices, period);
        const meanDeviation = this.calculateMeanDeviation(typicalPrices, sma, period);
        const cci = [];

        for (let i = 0; i < typicalPrices.length; i++) {
            if (sma[i] === null || meanDeviation[i] === null || meanDeviation[i] === 0) {
                cci.push(null);
            } else {
                cci.push((typicalPrices[i] - sma[i]) / (0.015 * meanDeviation[i]));
            }
        }

        return cci;
    }

    /**
     * 计算Woodie's CCI完整系统
     * @param {Array} data - OHLC数据数组
     * @returns {Object} 包含所有Woodie's CCI组件的数据
     */
    calculate(data) {
        const typicalPrices = data.map(d =>
            this.calculateTypicalPrice(d.high, d.low, d.close)
        );

        // 主CCI
        const mainCCI = this.calculateCCI(typicalPrices, this.cciPeriod);

        // Turbo CCI (较快的CCI)
        const turboCCI = this.calculateCCI(typicalPrices, this.turboPeriod);

        // 计算CCI的移动平均线
        const mainCCIMA = this.calculateSMA(mainCCI.filter(c => c !== null), 9);
        const turboCCIMA = this.calculateSMA(turboCCI.filter(c => c !== null), 9);

        // 调整数组长度以匹配原始数据
        const adjustedMainMA = Array(data.length - mainCCIMA.length).fill(null).concat(mainCCIMA);
        const adjustedTurboMA = Array(data.length - turboCCIMA.length).fill(null).concat(turboCCIMA);

        // 计算零线交叉
        const zeroLineCross = this.calculateZeroLineCross(mainCCI);

        // 计算趋势线
        const trendLine = this.calculateTrendLine(data);

        return {
            mainCCI: mainCCI,
            turboCCI: turboCCI,
            mainCCIMA: adjustedMainMA,
            turboCCIMA: adjustedTurboMA,
            zeroLineCross: zeroLineCross,
            trendLine: trendLine
        };
    }

    /**
     * 计算零线交叉
     * @param {Array} cci - CCI值数组
     * @returns {Array} 零线交叉信号数组
     */
    calculateZeroLineCross(cci) {
        const signals = [];
        for (let i = 0; i < cci.length; i++) {
            if (cci[i] === null || cci[i - 1] === null) {
                signals.push(0);
            } else if (cci[i] > 0 && cci[i - 1] <= 0) {
                signals.push(1); // 向上穿越零线
            } else if (cci[i] < 0 && cci[i - 1] >= 0) {
                signals.push(-1); // 向下穿越零线
            } else {
                signals.push(0);
            }
        }
        return signals;
    }

    /**
     * 计算趋势线
     * @param {Array} data - OHLC数据数组
     * @returns {Array} 趋势值数组
     */
    calculateTrendLine(data) {
        const period = 34;
        const trendLine = [];

        for (let i = 0; i < data.length; i++) {
            if (i < period - 1) {
                trendLine.push(null);
            } else {
                const highestHigh = Math.max(...data.slice(i - period + 1, i + 1).map(d => d.high));
                const lowestLow = Math.min(...data.slice(i - period + 1, i + 1).map(d => d.low));
                trendLine.push((highestHigh + lowestLow) / 2);
            }
        }

        return trendLine;
    }

    /**
     * 生成Woodie's CCI交易信号
     * @param {Object} woodiesData - Woodie's CCI数据
     * @returns {Array} 交易信号数组
     */
    generateSignals(woodiesData) {
        const signals = [];
        const { mainCCI, turboCCI, mainCCIMA, turboCCIMA, trendLine } = woodiesData;

        for (let i = 0; i < mainCCI.length; i++) {
            if (mainCCI[i] === null) {
                signals.push(0);
                continue;
            }

            let signal = 0;

            // Woodie's CCI 规则1: CCI穿越+100线
            if (mainCCI[i] > 100 && mainCCI[i - 1] <= 100) {
                signal = 1;
            }
            // Woodie's CCI 规则2: CCI穿越-100线
            else if (mainCCI[i] < -100 && mainCCI[i - 1] >= -100) {
                signal = -1;
            }
            // Woodie's CCI 规则3: 零线交叉确认
            else if (mainCCI[i] > 0 && mainCCI[i - 1] <= 0 && turboCCI[i] > turboCCIMA[i]) {
                signal = 1;
            }
            // Woodie's CCI 规则4: 趋势线突破
            else if (trendLine[i] !== null && mainCCI[i] > 0 && mainCCI[i] > trendLine[i]) {
                signal = 1;
            }
            // Woodie's CCI 规则5: CCI与趋势背离
            else if (mainCCI[i] < 0 && turboCCI[i] < turboCCIMA[i]) {
                signal = -1;
            }

            signals.push(signal);
        }

        return signals;
    }
}

// 导出模块
if (typeof module !== 'undefined' && module.exports) {
    module.exports = WoodiesCCI;
} else if (typeof window !== 'undefined') {
    window.WoodiesCCI = WoodiesCCI;
}

// 使用示例
if (typeof window !== 'undefined' && window.document) {
    window.testWoodiesCCI = function() {
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

        const woodiesCCI = new WoodiesCCI();
        const result = woodiesCCI.calculate(testData);
        const signals = woodiesCCI.generateSignals(result);

        console.log("Woodie's CCI Test:");
        console.log('Main CCI:', result.mainCCI[result.mainCCI.length - 1]);
        console.log('Turbo CCI:', result.turboCCI[result.turboCCI.length - 1]);
        console.log('Signal:', signals[signals.length - 1]);
    };
}