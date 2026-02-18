import YahooFinance from 'yahoo-finance2';

const yahooFinance = new YahooFinance();

export interface Candle {
    date: Date;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
}

export class MarketDataService {

    /**
     * Fetches the current live price for a given symbol.
     * @param symbol Ticker symbol (e.g., "SPY", "BTC-USD", "AAPL")
     */
    public static async getPrice(symbol: string): Promise<number | null> {
        try {
            const quoteResult = await yahooFinance.quote(symbol);
            const quote = quoteResult as any;
            return quote.regularMarketPrice || null;
        } catch (error) {
            console.error(`Failed to fetch price for ${symbol}:`, error);
            return null;
        }
    }

    /**
     * Fetches historical candle data for a given symbol.
     * @param symbol Ticker symbol
     * @param interval Interval (e.g., '1d', '1wk', '1mo')
     * @param daysBack Number of days of history to fetch
     */
    public static async getHistory(symbol: string, interval: '1d' | '1wk' | '1mo' = '1d', daysBack: number = 3650): Promise<Candle[]> {
        try {
            const endDate = new Date();
            const startDate = new Date();
            startDate.setDate(endDate.getDate() - daysBack);

            const resultResult = await yahooFinance.historical(symbol, {
                period1: startDate,
                period2: endDate,
                interval: interval
            });
            const result = resultResult as any[];

            return result.map((row: any) => ({
                date: new Date(row.date),
                open: row.open,
                high: row.high,
                low: row.low,
                close: row.close,
                volume: row.volume
            }));
        } catch (error) {
            console.error(`Failed to fetch history for ${symbol}:`, error);
            return [];
        }
    }

    /**
     * Get a snapshot combining current price and recent history
     */
    public static async getSnapshot(symbol: string) {
        const [price, history] = await Promise.all([
            this.getPrice(symbol),
            this.getHistory(symbol, '1d', 50)
        ]);

        return {
            symbol,
            price,
            history
        };
    }
}
