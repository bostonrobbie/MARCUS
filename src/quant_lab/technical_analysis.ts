import { RSI, MACD, SMA, BollingerBands } from 'technicalindicators';

export class TechnicalAnalyst {

    /**
     * Calculates the Relative Strength Index (RSI).
     * @param values Array of closing prices
     * @param period Period (default 14)
     */
    public static calculateRSI(values: number[], period: number = 14): number[] {
        return RSI.calculate({
            values: values,
            period: period
        });
    }

    /**
     * Calculates the Moving Average Convergence Divergence (MACD).
     * @param values Array of closing prices
     */
    public static calculateMACD(values: number[]) {
        return MACD.calculate({
            values: values,
            fastPeriod: 12,
            slowPeriod: 26,
            signalPeriod: 9,
            SimpleMAOscillator: false,
            SimpleMASignal: false
        });
    }

    /**
     * Calculates Simple Moving Average (SMA).
     * @param values Array of closing prices
     * @param period Period (e.g., 50, 200)
     */
    public static calculateSMA(values: number[], period: number): number[] {
        return SMA.calculate({
            period: period,
            values: values
        });
    }

    /**
     * Calculates Bollinger Bands.
     * @param values Array of closing prices
     * @param period Period (default 20)
     * @param stdDev Standard Deviation (default 2)
     */
    public static calculateBollingerBands(values: number[], period: number = 20, stdDev: number = 2) {
        return BollingerBands.calculate({
            period: period,
            values: values,
            stdDev: stdDev
        });
    }

    /**
     * Analyzes the trend based on SMA crossovers (Golden Cross / Death Cross logic).
     */
    public static analyzeTrend(values: number[]): 'BULLISH' | 'BEARISH' | 'NEUTRAL' {
        if (values.length < 200) return 'NEUTRAL';

        const sma50 = this.calculateSMA(values, 50);
        const sma200 = this.calculateSMA(values, 200);

        const last50 = sma50[sma50.length - 1];
        const last200 = sma200[sma200.length - 1];

        if (last50 > last200) return 'BULLISH';
        if (last50 < last200) return 'BEARISH';
        return 'NEUTRAL';
    }
}
