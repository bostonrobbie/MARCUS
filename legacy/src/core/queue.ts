
import { DB } from './db';
import { Logger } from './logger';

export type JobHandler = (data: any) => Promise<any>;

export class JobQueue {
    private static handlers: Map<string, JobHandler> = new Map();
    private static isWorkerRunning = false;
    private static pollInterval = 1000;
    private static logger: Logger;

    private static getLogger() {
        if (!this.logger) {
            this.logger = new Logger(undefined, undefined);
        }
        return this.logger;
    }

    static register(type: string, handler: JobHandler) {
        this.handlers.set(type, handler);
        this.getLogger().info(`Registered job handler for: ${type}`);
    }

    static async add(type: string, data: any, priority: number = 0, runId?: number) {
        const prisma = DB.getInstance().getPrisma();
        const job = await prisma.job.create({
            data: {
                type,
                data: JSON.stringify(data),
                priority,
                runId,
                status: 'PENDING'
            }
        });
        this.getLogger().info(`Job added: #${job.id} (${type})`);
        return job.id;
    }

    static async worker() {
        if (this.isWorkerRunning) return;
        this.isWorkerRunning = true;
        this.getLogger().info('Job Queue Worker started.');

        while (this.isWorkerRunning) {
            try {
                const processed = await this.processNext();
                if (!processed) {
                    // No jobs, wait before next poll
                    await new Promise(resolve => setTimeout(resolve, this.pollInterval));
                }
            } catch (error: any) {
                this.getLogger().error(`Worker Loop Error: ${error.message}`);
                await new Promise(resolve => setTimeout(resolve, 5000)); // Backoff on crash
            }
        }
    }

    static stop() {
        this.isWorkerRunning = false;
    }

    private static async processNext(): Promise<boolean> {
        const prisma = DB.getInstance().getPrisma();

        // 1. Find a actionable job
        const job = await prisma.job.findFirst({
            where: { status: 'PENDING' },
            orderBy: [
                { priority: 'desc' },
                { createdAt: 'asc' }
            ]
        });

        if (!job) return false;

        // 2. Lock it (Optimistic)
        try {
            const lockedJob = await prisma.job.update({
                where: {
                    id: job.id,
                    status: 'PENDING' // Ensure it's still pending
                },
                data: {
                    status: 'PROCESSING',
                    startedAt: new Date()
                }
            });

            // 3. Process
            const handler = this.handlers.get(lockedJob.type);
            if (!handler) {
                throw new Error(`No handler registered for type: ${lockedJob.type}`);
            }

            this.getLogger().info(`Processing Job #${lockedJob.id} (${lockedJob.type})...`);

            try {
                const result = await handler(JSON.parse(lockedJob.data));

                // 4. Complete
                await prisma.job.update({
                    where: { id: lockedJob.id },
                    data: {
                        status: 'COMPLETED',
                        completedAt: new Date(),
                        result: JSON.stringify(result)
                    }
                });
                this.getLogger().info(`Job #${lockedJob.id} COMPLETED.`);
            } catch (execError: any) {
                // 5. Fail (Handler Error)
                await prisma.job.update({
                    where: { id: lockedJob.id },
                    data: {
                        status: 'FAILED',
                        completedAt: new Date(),
                        error: execError.message || String(execError)
                    }
                });
                this.getLogger().error(`Job #${lockedJob.id} FAILED: ${execError.message || execError}`);
            }

            return true;

        } catch (e: any) {
            if (e.code === 'P2025') {
                // Determine if it was a race condition (Record to update not found)
                // Another worker picked it up. Just return true to try again immediately.
                return true;
            }
            throw e;
        }
    }
}
