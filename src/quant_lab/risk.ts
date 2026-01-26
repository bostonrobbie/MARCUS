
import { RiskLevel } from './types';
import * as path from 'path';
import * as fs from 'fs';

export interface ActionRequest {
    type: 'FILE_WRITE' | 'COMMAND' | 'NETWORK' | 'SPEND' | 'MESSAGE' | 'TRADING';
    payload: any;
}

export class RiskEngine {
    private static policies: any;

    private static loadPolicies() {
        if (!this.policies) {
            const configPath = path.join(process.cwd(), 'config', 'policies.json');
            if (fs.existsSync(configPath)) {
                this.policies = JSON.parse(fs.readFileSync(configPath, 'utf8'));
            } else {
                this.policies = { safe_command_patterns: [] };
            }
        }
        return this.policies;
    }

    public static evaluate(action: ActionRequest): RiskLevel {
        const policies = this.loadPolicies();

        switch (action.type) {
            case 'FILE_WRITE':
                // Jail handled separately in CycleRunner for granular control
                return 'SAFE';

            case 'COMMAND':
                const cmd = (action.payload.command || '').trim();

                // 1. Explicitly Blocked
                if (policies.blocked_commands && policies.blocked_commands.some((p: string) => cmd.includes(p))) {
                    return 'BLOCKED';
                }

                // 2. Safe Allowlist (Regex based)
                if (policies.safe_command_patterns && policies.safe_command_patterns.some((pattern: string) => new RegExp(pattern).test(cmd))) {
                    return 'SAFE';
                }

                // 3. Known Risky
                if (cmd.includes('npm install') || cmd.includes('pip install') || cmd.includes('npm i ')) {
                    return 'REVIEW';
                }

                // Default to REVIEW
                return 'REVIEW';

            case 'NETWORK':
                return 'REVIEW';

            case 'SPEND':
            case 'MESSAGE':
            case 'TRADING':
                return 'BLOCKED';

            default:
                return 'REVIEW';
        }
    }
}
