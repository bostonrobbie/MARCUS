
import * as path from 'path';
import * as fs from 'fs';

export class Jail {
    private static policies: any;

    private static loadPolicies() {
        if (!this.policies) {
            const configPath = path.join(process.cwd(), 'config', 'policies.json');
            if (fs.existsSync(configPath)) {
                this.policies = JSON.parse(fs.readFileSync(configPath, 'utf8'));
            } else {
                // Fallback defaults
                this.policies = {
                    allowed_paths: [process.cwd()]
                };
            }
        }
        return this.policies;
    }

    public static isPathAllowed(targetPath: string): boolean {
        const policies = this.loadPolicies();
        const absoluteTarget = path.resolve(targetPath);

        return policies.allowed_paths.some((allowed: string) => {
            const absoluteAllowed = path.resolve(allowed);
            return absoluteTarget.startsWith(absoluteAllowed);
        });
    }

    public static validatePath(targetPath: string) {
        if (!this.isPathAllowed(targetPath)) {
            throw new Error(`JAIL BREAK DETECTED: Path outside workspace: ${targetPath}`);
        }
    }
}
