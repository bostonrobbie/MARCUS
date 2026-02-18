
import { MarketDataService } from '../quant_lab/market_data';
import { TechnicalAnalyst } from '../quant_lab/technical_analysis';

async function main() {
    const symbol = 'SPY';
    console.log(`\n🔍 Use Market Data Service for ${symbol}...`);

    // 1. Fetch Real-Time Price
    const price = await MarketDataService.getPrice(symbol);
    console.log(`💰 Current Price: $${price}`);

    if (!price) {
        console.error('❌ Failed to fetch price. Exiting.');
        process.exit(1);
    }

    // 2. Fetch Historical Data
    console.log(`\n📅 Fetching 100 days of history...`);
    const history = await MarketDataService.getHistory(symbol, '1d', 100);
    console.log(`✅ Fetched ${history.length} candles.`);

    if (history.length === 0) {
        console.error('❌ No history found.');
        process.exit(1);
    }

    const closePrices = history.map(c => c.close);

    // 3. Run Technical Analysis
    console.log(`\n📊 Running Technical Analysis...`);

    // RSI
    const rsi = TechnicalAnalyst.calculateRSI(closePrices);
    const lastRsi = rsi[rsi.length - 1];
    console.log(`📈 RSI (14): ${lastRsi.toFixed(2)}`);

    // MACD
    const macd = TechnicalAnalyst.calculateMACD(closePrices);
    const lastMacd = macd[macd.length - 1];
    console.log(`📉 MACD: Line=${lastMacd.MACD?.toFixed(2)}, Signal=${lastMacd.signal?.toFixed(2)}, Hist=${lastMacd.histogram?.toFixed(2)}`);

    // Trend
    const trend = TechnicalAnalyst.analyzeTrend(closePrices);
    console.log(`🚦 Trend (SMA50 vs SMA200): ${trend}`); // Might be NEUTRAL if <200 days data

    console.log('\n✅ Verification Complete: Market Data & TA Libraries are functional.');
}

main().catch(console.error);
