import * as fs from 'fs';
import { OllamaLLM } from '../core/llm';
import { MemoryStore } from '../hedge_fund/memory';
import { DB } from '../core/db';

export interface Opportunity {
    id: string;
    title: string;
    description: string;
    marketLeverage: string;
    ownerFriction: string; // High/Med/Low
    suggestedAction: string;
}

export class Scout {
    private llm: OllamaLLM;

    constructor() {
        this.llm = new OllamaLLM();
    }

    /**
     * Periodically scouts for opportunities based on recent business state.
     */
    public async scoutOpportunities(projectId: number): Promise<Opportunity[]> {
        // Fetch recent memories and snapshots
        const timeline = await MemoryStore.getTimeline(projectId, 10);

        // [NEW] Gather Real-Time Market Intelligence
        const { MarketDataService } = await import('./market_data');
        const { TechnicalAnalyst } = await import('./technical_analysis');

        const watchlist = ['SPY', 'QQQ', 'BTC-USD', 'NVDA', 'TSLA'];
        let marketBriefing = "REAL-TIME MARKET DATA (Verified):\n";

        for (const symbol of watchlist) {
            try {
                const history = await MarketDataService.getHistory(symbol, '1d', 100);
                if (history.length > 50) {
                    const price = history[history.length - 1].close;
                    const closes = history.map(c => c.close);
                    const rsi = TechnicalAnalyst.calculateRSI(closes);
                    const lastRsi = rsi[rsi.length - 1];
                    const trend = TechnicalAnalyst.analyzeTrend(closes);

                    marketBriefing += `- ${symbol}: $${price.toFixed(2)} | RSI: ${lastRsi.toFixed(1)} | Trend: ${trend}\n`;
                }
            } catch (err) {
                console.error(`Failed to scout ${symbol}`, err);
            }
        }

        const prompt = `
You are the Analysis Lead for an Autonomous Business Unit (Hedge Fund Agent). 
Your goal is to scout for new business opportunities, optimizations, or market pivots using the **SOVEREIGN SIGNAL AUDIT** protocol.

${marketBriefing}

Context (Recent History):
${timeline.map(m => `- ${m.title}: ${m.content}`).join('\n')}

MARKET SIGNAL PROTOCOL (v8):
- Use the REAL-TIME MARKET DATA provided above. Do not hallucinate prices.
- If RSI > 70, considered Overbought. If RSI < 30, considered Oversold.
- "BULLISH" trend means SMA50 > SMA200.
- Identify setups based on this data.
- Any opportunity with >8 leverage will trigger the Autonomy Flywheel.

Based on this context, identify 3 proactive opportunities that would:
1. Increase business leverage (The Love Equation).
2. Require minimal owner friction to execute.
3. Align with a "high functioning" business mindset.

Return the result as a JSON array of objects:
{
  "opportunities": [
    {
      "id": "OPT-001",
      "title": "Opportunity Title (e.g. BTC Oversold Bounce)",
      "description": "Evidence-backed description citing RSI and Price",
      "marketLeverage": 8, // (1-10 scale)
      "ownerFriction": 2, // (1-10 scale, lower is better)
      "suggestedAction": "First step to take"
    }
  ]
}

YOU MUST INCLUDE A <thought> BLOCK AT THE BEGINNING OF YOUR RESPONSE TO SHOW YOUR INTERNAL REASONING.
Ensure the remainder of your response is valid JSON matching your required schema.
`;

        const response = await this.llm.generate(prompt);
        let content = response.content;

        // Trace & Transparency: Log thoughts for the Analysis Lead
        const { Intercom } = await import('../hedge_fund/intercom');
        const thoughtMatch = content.match(/<thought>([\s\S]*?)<\/thought>/);
        if (thoughtMatch) {
            Intercom.logThought('Analysis Lead', thoughtMatch[1]);
            content = content.replace(/<thought>[\s\S]*?<\/thought>/, '').trim();
        }

        try {
            const match = content.match(/\{[\s\S]*\}/);
            if (match) {
                const parsed = JSON.parse(match[0]);
                return parsed.opportunities || [];
            }
        } catch (e) {
            console.error('Failed to parse scout opportunities:', e);
        }
        return [];
    }
}
