
import { DB } from './db';
import * as fs from 'fs';
import * as path from 'path';

export class Logger {
    private runId: number | null;
    private logDir: string;

    constructor(runId?: number, logDir?: string) {
        this.runId = runId || null;
        this.logDir = logDir || (runId ? path.join(process.cwd(), 'runs', `run_${runId}`) : path.join(process.cwd(), 'logs'));
    }

    public async info(message: string, taskId?: number) {
        return this.log('INFO', message, taskId);
    }

    public async error(message: string, taskId?: number) {
        return this.log('ERROR', message, taskId);
    }

    public async log(level: string, message: string, taskId?: number) {
        const dbInstance = DB.getInstance();
        const db = dbInstance.getDb();
        const ts = new Date().toISOString();

        // Console log
        console.log(`[${ts}] [${level}] ${message}`);

        // DB Log if runId exists
        if (this.runId) {
            try {
                await dbInstance.getPrisma().log.create({
                    data: {
                        runId: this.runId,
                        taskId: taskId || null,
                        ts: new Date(ts),
                        level: level,
                        message: message
                    }
                });
            } catch (e) {
                console.error('Failed to write log to DB', e);
            }
        }

        // Forward to Intercom
        try {
            const { Intercom } = require('../hedge_fund/intercom');
            if (['INFO', 'WARN', 'ERROR'].includes(level)) {
                Intercom.log('SYSTEM', 'ALL', `[${level}] ${message}`, { runId: this.runId, taskId });
            }
        } catch (e) {
            // Ignore circular dependency
        }
    }
}
