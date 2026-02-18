import { DB } from './db';
import { CycleRunner } from '../hedge_fund/cycle';
import { Logger } from './logger';
import { EventBus, EVENTS } from './events';
import * as fs from 'fs';
import * as path from 'path';
import { JobQueue } from './queue';

export class Scheduler {
    public static async addSchedule(projectId: number, cadence: 'daily' | 'weekly', time: string) {
        const prisma = DB.getInstance().getPrisma();

        let nextRun = new Date();
        const [hours, minutes] = time.split(':').map(Number);
        nextRun.setHours(hours, minutes, 0, 0);

        if (nextRun < new Date()) {
            nextRun.setDate(nextRun.getDate() + 1);
        }

        await prisma.schedule.create({
            data: {
                projectId: projectId,
                cadence: cadence,
                nextRunAt: nextRun,
                status: 'ACTIVE'
            }
        });
        return true;
    }

    public static async listSchedules(): Promise<any[]> {
        const prisma = DB.getInstance().getPrisma();
        const schedules = await prisma.schedule.findMany({
            include: { project: true }
        });
        return schedules.map(s => ({
            ...s,
            project_name: s.project?.name
        }));
    }

    public static async runDue() {
        const prisma = DB.getInstance().getPrisma();
        const now = new Date();

        // Find due schedules
        const due = await prisma.schedule.findMany({
            where: {
                status: 'ACTIVE',
                nextRunAt: { lte: now }
            }
        });

        if (due.length > 0) {
            console.log(`[DAEMON] Found ${due.length} due schedules.`);
        }

        for (const schedule of due) {
            console.log(`[DAEMON] Queuing schedule ${schedule.id} for project ${schedule.projectId}`);

            // Queue the job
            await JobQueue.add('SCHEDULE_EXECUTION', { scheduleId: schedule.id, projectId: schedule.projectId });

            // Calculate next run immediately to prevent double-queuing
            const nextRun = new Date(schedule.nextRunAt);
            if (schedule.cadence === 'daily') {
                nextRun.setDate(nextRun.getDate() + 1);
            } else if (schedule.cadence === 'weekly') {
                nextRun.setDate(nextRun.getDate() + 7);
            }

            // Update Schedule
            await prisma.schedule.update({
                where: { id: schedule.id },
                data: { nextRunAt: nextRun }
            });
        }
    }

    public static async processSchedule(data: any) {
        console.log(`[JOB] Processing schedule for project ${data.projectId}`);

        // Start Autonomous Flywheel (v9)
        const { ProcessRunner } = await import('../hedge_fund/process_runner');
        console.log(`[JOB] Triggering SCOUTING cycle for project ${data.projectId}`);
        const processRunId = await ProcessRunner.startProcess('opportunity_scout_cycle', data.projectId, {
            objective: "Daily autonomous market audit and opportunity scouting."
        });

        const pRunner = new ProcessRunner(0);
        await pRunner.executeNextStep(processRunId);
    }

    public static async daemon() {
        console.log('[DAEMON] Starting Company OS Event-Driven Scheduler...');

        // Register Handlers
        JobQueue.register('SCHEDULE_EXECUTION', this.processSchedule);

        // Start Job Worker
        JobQueue.worker();

        // Listen for Schedule Checks
        EventBus.getInstance().on(EVENTS.SCHEDULE.DUE, async () => {
            try {
                await this.runDue();
            } catch (e) {
                console.error('[DAEMON] Error running schedules:', e);
            }
        });

        console.log('[DAEMON] Heartbeat active (60s tick)');

        // Heartbeat Loop (Emits events instead of blocking logic)
        setInterval(() => {
            EventBus.getInstance().emit(EVENTS.SCHEDULE.DUE);
            // Update heartbeat file
            const now = new Date().toISOString();
            fs.writeFileSync(path.join(process.cwd(), 'daemon_heartbeat.json'), JSON.stringify({
                last_tick: now,
                status: 'OK'
            }));
        }, 60000);

        // Initial check
        EventBus.getInstance().emit(EVENTS.SCHEDULE.DUE);

        // Keep process alive
        return new Promise(() => { });
    }
}
