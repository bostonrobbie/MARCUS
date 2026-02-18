
import * as fs from 'fs';
import * as path from 'path';
import { DB } from '../core/db';

export class WisdomLayer {
    public static async synthesizeHistory(projectId: number, limit: number = 5): Promise<string> {
        const db = DB.getInstance().getDb();

        // Find last N completed process runs
        const runs: any[] = await new Promise((resolve) => {
            db.all("SELECT * FROM memories WHERE project_id = ? ORDER BY created_at DESC LIMIT 20", [projectId], (err: any, rows: any[]) => {
                if (err) return resolve([]);
                resolve(rows || []);
            });
        });

        if (runs.length === 0) return "No prior history available.";

        let synthesis = "Historical Patterns and Lessons Learned:\n\n";

        for (const run of runs) {
            const processDir = path.join(process.cwd(), 'runs', `process_${run.id}`);
            const reportPath = path.join(processDir, 'PROCESS_REPORT.md');

            if (fs.existsSync(reportPath)) {
                const content = fs.readFileSync(reportPath, 'utf8');
                // Extract summary or results
                synthesis += `--- Run #${run.id} ---\n`;
                synthesis += content.substring(0, 500) + "...\n";
            }
        }

        return synthesis;
    }

    public static applyLoveEquation(actions: any[]): any[] {
        // Roemmele's Love Equation in this context: Empathy for the Owner's Time
        // Friction = Time * Complexity
        // We prioritize low friction, high ROI
        return actions.map(a => {
            const time = parseInt(a.estimated_time) || 30;
            const complexity = a.complexity || 5;
            const friction = time * complexity;
            return { ...a, friction_score: friction };
        }).sort((a, b) => a.friction_score - b.friction_score);
    }
}
