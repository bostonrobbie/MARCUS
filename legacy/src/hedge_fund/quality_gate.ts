
import * as fs from 'fs';
import * as path from 'path';

export class QualityGate {
    public static validate(runDir: string, gate: any): { success: boolean, error?: string } {
        const targetPath = path.join(runDir, gate.target);

        if (!fs.existsSync(targetPath)) {
            return { success: false, error: `Artifact missing: ${gate.target}` };
        }

        const content = fs.readFileSync(targetPath, 'utf8');

        if (gate.type === 'MARKDOWN_SECTION') {
            const missing = [];
            for (const section of gate.sections) {
                // Heuristic: Check for # Section, ## Section, or **Section:**
                const pattern = new RegExp(`(^#+.*${section}|\\*\\*${section}:?\\*\\*)`, 'mi');
                if (!pattern.test(content)) {
                    missing.push(section);
                }
            }
            if (missing.length > 0) {
                return { success: false, error: `MD Sections missing in ${gate.target}: ${missing.join(', ')}` };
            }
        }

        if (gate.type === 'CSV_COLUMNS') {
            const header = content.split('\n')[0].toLowerCase();
            const missing = [];
            for (const col of gate.columns) {
                if (!header.includes(col.toLowerCase())) {
                    missing.push(col);
                }
            }
            if (missing.length > 0) {
                return { success: false, error: `CSV Columns missing in ${gate.target}: ${missing.join(', ')}` };
            }
        }

        if (gate.type === 'CSV_MIN_ROWS') {
            const rows = content.trim().split('\n').filter(l => l.length > 0);
            const count = rows.length - 1; // Exclude header
            if (count < gate.min) {
                return { success: false, error: `CSV ${gate.target} has only ${count} rows, expected at least ${gate.min}` };
            }
        }

        return { success: true };
    }
}
