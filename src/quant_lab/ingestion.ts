import * as fs from 'fs';
import * as path from 'path';
import { Jail } from './jail';
import { Logger } from '../core/logger';

export class DataIngestor {
    private logger: Logger;

    constructor() {
        this.logger = new Logger();
    }

    public async ingestDirectory(targetPath: string): Promise<string> {
        Jail.validatePath(targetPath);

        const files = fs.readdirSync(targetPath);
        const report = {
            total_files: files.length,
            conversations_summary: [] as any[],
            documents: [] as string[]
        };

        // 1. Process conversations.json
        const convoPath = path.join(targetPath, 'conversations.json');
        if (fs.existsSync(convoPath)) {
            console.log('[INGESTION] Processing conversations.json (this may take a moment)...');
            try {
                const raw = fs.readFileSync(convoPath, 'utf8');
                const convos = JSON.parse(raw);
                console.log(`[INGESTION] Loaded ${convos.length} conversations.`);

                // Extract top 20 most recent
                report.conversations_summary = convos
                    .slice(0, 50) // Take top 50
                    .map((c: any) => ({
                        title: c.title,
                        create_time: c.create_time ? new Date(c.create_time * 1000).toISOString() : 'Unknown',
                        // Extract first msg or system prompt if possible? usually structure is complex.
                        // Simplified: just title for now to save tokens.
                    }));
            } catch (e: any) {
                console.error('[INGESTION] Failed to parse conversations.json', e.message);
            }
        }

        // 2. List PDF/MD Documents
        report.documents = files.filter(f => f.endsWith('.pdf') || f.endsWith('.md'));

        // Generate Summary Markdown
        let md = `# Proprietary Data Ingestion Report\n\n`;
        md += `**Source**: ${targetPath}\n`;
        md += `**Total Files**: ${files.length}\n\n`;

        md += `## Key Documents (PDF/MD)\n`;
        report.documents.forEach(d => md += `- ${d}\n`);

        md += `\n## Recent Chat History (Top 50)\n`;
        report.conversations_summary.forEach(c => {
            md += `- **${c.title}** (${c.create_time})\n`;
        });

        return md;
    }
}
