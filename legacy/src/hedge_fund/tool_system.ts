
import axios from 'axios';
import { Logger } from '../core/logger';

export class ToolSystem {
    public static async execute(toolName: string, args: any, logger: Logger): Promise<string> {
        await logger.log('INFO', `🔧 Executing Tool: ${toolName} with args: ${JSON.stringify(args)}`);

        try {
            if (toolName === 'web_search') {
                return await this.webSearch(args.query);
            } else if (toolName === 'browser_scrape') {
                return await this.browserScrape(args.url);
            } else {
                return `Error: Tool ${toolName} not found.`;
            }
        } catch (error: any) {
            return `Error executing tool ${toolName}: ${error.message}`;
        }
    }

    private static async webSearch(query: string): Promise<string> {
        // In a real private appliance, this might call a local SearXNG or a privacy-first API
        // For v1, we simulate a robust response or use a simple API if configured.
        return `Search results for "${query}": [Evidence Sample] Found 3 market competitors and 2 new tech trends. Competitor X just released a public API.`;
    }

    private static async browserScrape(url: string): Promise<string> {
        // Simple fetch and text extraction
        try {
            const response = await axios.get(url, { timeout: 5000 });
            const text = response.data.toString().replace(/<[^>]*>?/gm, ' ').substring(0, 2000);
            return `Scraped content from ${url}: ${text}`;
        } catch (e: any) {
            return `Failed to scrape ${url}: ${e.message}`;
        }
    }
}
