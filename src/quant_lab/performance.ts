
import { DB } from '../core/db';
import { OllamaLLM } from '../core/llm';
import { DigitalOrgRegistry } from '../hedge_fund/registry';
import * as path from 'path';
import * as fs from 'fs';

export interface ReviewScore {
    title: string;
    efficiency: number;
    quality: number;
    lastMileReadiness: number;
    feedback: string;
}

export class PerformanceReview {
    private llm: OllamaLLM;

    constructor() {
        this.llm = new OllamaLLM();
    }

    /**
     * Conduct a performance review for a manager and their direct reports.
     */
    public async conductCheckIn(managerTitle: string, projectId: number): Promise<string> {
        const manager = DigitalOrgRegistry.getTitle(managerTitle);
        const reports = DigitalOrgRegistry.getReports(managerTitle);

        if (!manager || reports.length === 0) {
            return `No direct reports for ${managerTitle} to review.`;
        }

        // Fetch recent work from the database
        const db = DB.getInstance().getDb();
        const recentSteps: any[] = await new Promise((resolve) => {
            db.all(`
                SELECT ps.*, pr.process_name 
                FROM process_steps ps
                JOIN process_runs pr ON ps.process_run_id = pr.id
                WHERE pr.project_id = ? AND ps.status = 'COMPLETED'
                ORDER BY ps.created_at DESC
                LIMIT 20
            `, [projectId], (err, rows) => resolve(rows || []));
        });

        const prompt = `
You are the ${manager.title} performing a regular performance check-in with your team.

Your Team:
${reports.map(r => `- ${r.title} (Bias: ${r.bias})`).join('\n')}

Recent Activity Log:
${recentSteps.map(s => `[${s.process_name}] Step: ${s.step_id} - Result: ${s.result_json || 'Completed'}`).join('\n')}

Audit this work. As a manager, you must:
1. Rate each direct report on a scale of 1-10 for Quality and "Last Mile" execution.
2. Provide specific, constructive feedback based on the heuristics of their role.
3. Identify if any report is falling behind or deserves a "Promotion" to higher complexity tasks.

Format your response as a professional Audit Report for the CEO.
`;

        const response = await this.llm.generate(prompt);
        return response.content;
    }
}
