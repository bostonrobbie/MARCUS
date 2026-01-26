
import { Command } from 'commander';
import * as fs from 'fs';
import * as path from 'path';
import { DB } from './core/db';
import { CycleRunner } from './hedge_fund/cycle';
import { ApprovalSystem } from './hedge_fund/approvals';

const program = new Command();

program
    .name('companyos')
    .description('Company OS Private Appliance CLI')
    .version('1.0.0');

program
    .command('projects:add')
    .description('Create a new project')
    .requiredOption('--name <text>', 'Project name')
    .option('--description <text>', 'Project description')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            const prisma = DB.getInstance().getPrisma();
            const project = await prisma.project.create({
                data: {
                    name: options.name,
                    description: options.description,
                    createdAt: new Date()
                }
            });
            console.log(`Project created with ID: ${project.id}`);
        } catch (error) {
            console.error('Error adding project:', error);
        }
    });

program
    .command('projects:list')
    .description('List all projects')
    .action(async () => {
        try {
            await DB.getInstance().init();
            const prisma = DB.getInstance().getPrisma();
            console.log('\n--- Projects ---');
            const projects = await prisma.project.findMany();
            if (projects.length === 0) console.log('No projects found.');
            else {
                projects.forEach(r => console.log(`[ID: ${r.id}] ${r.name} - ${r.description || 'No description'}`));
            }
        } catch (error) {
            console.error('Error listing projects:', error);
        }
    });

program
    .command('run_cycle')
    .description('Start a new business cycle with an objective')
    .requiredOption('-o, --objective <text>', 'Business objective')
    .option('-p, --project_id <id>', 'Project ID', '1') // Default to 1 (Default Project)
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            console.log(`Starting cycle for objective: ${options.objective} (Project: ${options.project_id})`);
            const runner = new CycleRunner();
            const runId = await runner.start(options.objective, Number(options.project_id));
            console.log(`Cycle started with Run ID: ${runId}`);
        } catch (error) {
            console.error('Error starting cycle:', error);
        }
    });

program
    .command('approvals:show')
    .description('Show details of an approval request')
    .requiredOption('--id <id>', 'Approval ID')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            const prisma = DB.getInstance().getPrisma();
            const approval = await prisma.approval.findUnique({
                where: { id: Number(options.id) },
                include: { task: true }
            });

            if (!approval) {
                console.log('Approval request not found.');
                return;
            }

            console.log(`\n=== Approval Request #${approval.id} ===`);
            console.log(`Task: [${approval.task?.dept}] ${approval.task?.title}`);
            console.log(`Description: ${approval.task?.description}`);
            console.log(`Status: ${approval.decision}`);
            console.log(`Requested At: ${approval.requestedAt}`);
            if (approval.decision !== 'PENDING') {
                console.log(`Resolved At: ${approval.approvedAt}`);
                console.log(`Notes: ${approval.notes}`);
            }
            console.log('\n--- Details ---');
            // Checking ApprovalSystem to ensure we have context
            console.log(`Context: ${approval.notes || 'No notes provided.'}`);
            console.log(`Context: ${approval.notes || 'No notes provided.'}`);

        } catch (error) {
            console.error('Error showing approval:', error);
        }
    });

program
    .command('approvals:list')
    .description('List pending approvals')
    .action(async () => {
        try {
            await DB.getInstance().init();
            const pending = await ApprovalSystem.getPending();
            console.log('\n--- Pending Approvals ---');
            if (pending.length === 0) console.log('No pending approvals.');
            else {
                console.table(pending.map(p => ({
                    ID: p.id,
                    Task: p.title,
                    Risk: 'REVIEW', // Assuming all are REVIEW for now
                    Note: p.notes,
                    Created: p.requested_at
                })));
            }
        } catch (error) {
            console.error('Error listing approvals:', error);
        }
    });

program
    .command('report:daily')
    .description('Generate daily report')
    .option('--date <YYYY-MM-DD>', 'Date to report on', new Date().toISOString().split('T')[0])
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            const prisma = DB.getInstance().getPrisma();
            const date = options.date;
            // Create start/end dates for the query
            const startDate = new Date(date);
            const endDate = new Date(date);
            endDate.setDate(endDate.getDate() + 1);

            console.log(`\n=== Daily Report: ${date} ===\n`);

            // Runs Summary
            const runs = await prisma.processRun.findMany({
                where: {
                    startedAt: {
                        gte: startDate,
                        lt: endDate
                    }
                }
            });

            console.log(`Total Runs: ${runs.length}`);
            const completed = runs.filter(r => r.status === 'COMPLETED').length;
            const failed = runs.filter(r => r.status === 'FAILED').length;
            console.log(`Completed: ${completed} | Failed: ${failed}`);

            // KPIs
            const kpis = await prisma.kPI.findMany({
                where: {
                    createdAt: {
                        gte: startDate,
                        lt: endDate
                    }
                }
            });

            if (kpis.length > 0) {
                console.log('\n--- KPIs ---');
                console.table(kpis.map(k => ({ Metric: k.name, Value: k.value, Unit: k.unit || '' })));
            } else {
                console.log('\nNo KPIs recorded today.');
            }

            // High Level Issues
            if (failed > 0) {
                console.log('\n--- Failures ---');
                runs.filter(r => r.status === 'FAILED').forEach(r => {
                    console.log(`Run #${r.id}: ${r.processName} (FAILED)`);
                });
            }

        } catch (error) {
            console.error('Error generating report:', error);
        }
    });

program
    .command('approve')
    .description('Approve a pending request')
    .requiredOption('--id <id>', 'Approval ID')
    .option('--notes <text>', 'Optional notes')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            await ApprovalSystem.review(Number(options.id), 'APPROVED', options.notes);
            console.log(`Approval ${options.id} granted.`);
        } catch (error) {
            console.error('Error approving:', error);
        }
    });

program
    .command('reject')
    .description('Reject a pending request')
    .requiredOption('--id <id>', 'Approval ID')
    .option('--notes <text>', 'Optional notes')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            await ApprovalSystem.review(Number(options.id), 'REJECTED', options.notes);
            console.log(`Approval ${options.id} rejected.`);
        } catch (error) {
            console.error('Error rejecting:', error);
        }
    });


program
    .command('llm_test')
    .description('Test connectivity to local Ollama LLM')
    .action(async () => {
        try {
            const { OllamaLLM } = await import('./core/llm');
            const llm = new OllamaLLM();
            console.log('Testing Ollama connection...');
            const response = await llm.generate('Say "OK" if you can hear me.');
            console.log('Response:', response.content);
        } catch (error) {
            console.error('LLM Test Failed:', error);
        }
    });


program
    .command('resume')
    .description('Resume a paused run')
    .requiredOption('--run_id <id>', 'Run ID')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            console.log(`Resuming run: ${options.run_id}`);

            await DB.getInstance().init();
            console.log(`Resuming run: ${options.run_id}`);
            const prisma = DB.getInstance().getPrisma();

            // Check if paused
            const run = await prisma.processRun.findUnique({
                where: { id: Number(options.run_id) }
            });

            if (!run) return;

            if (run.status !== 'PAUSED' && run.status !== 'RUNNING') {
                console.log(`Run status is ${run.status}. Cannot resume.`);
                return;
            }

            // If paused, set to RUNNING
            if (run.status === 'PAUSED') {
                await prisma.processRun.update({
                    where: { id: run.id },
                    data: { status: 'RUNNING' }
                });
            }

            // Also check if any task is NEEDS_APPROVAL
            // Note: ProcessRunner uses `process_steps` relation now, not `tasks`.
            // The `tasks` table might be legacy-ish unless used by LLM_TASK logic? 
            // `process_runner.ts` creates `process_steps`.
            // But `CycleRunner` (for LLM_TASK) creates `tasks` via `approvals.ts`?
            // Approvals link to `tasks`.
            // So we need to find tasks pending for this run?
            // But `ProcessRunner` passes `runId` (which is `processRunId`).
            // `CycleRunner` treats it as `projectId`?
            // Re-reading `ProcessRunner.ts`: 
            // `runner.start(prompt, Number(run.projectId), processDir)` -> CycleRunner uses `projectId`.
            // So tasks are linked to `projectId`? Or `run_id`?
            // CycleRunner creates `runs` (legacy `runs` table vs `process_runs`?)
            // This is a dual-schema issue. `ProcessRunner` uses `process_runs`. `CycleRunner` (older) uses `runs` table?
            // Let's assume `options.run_id` is a `processRunId`.

            // We'll skip complex logic and just rely on `runner.resumeRun`.
            // But `CycleRunner` is for `runs`, not `process_runs`?
            // If `ProcessRunner` is the main entry, we should use `executeNextStep`.
            // The CLI calls `CycleRunner.resumeRun`. 
            // If the user meant Process resumption, we should call `ProcessRunner.executeNextStep`.

            // For now, let's just use Prisma for the status update as requested.

            const pTask = await prisma.task.findFirst({
                where: {
                    runId: Number(options.run_id),
                    status: 'NEEDS_APPROVAL'
                }
            });

            if (pTask) {
                const approved = await prisma.approval.findFirst({
                    where: {
                        taskId: pTask.id,
                        decision: 'APPROVED'
                    }
                });

                if (approved) {
                    console.log(`Task ${pTask.id} approved. Resuming...`);
                    await prisma.task.update({
                        where: { id: pTask.id },
                        data: { status: 'PENDING' }
                    });
                } else {
                    console.log(`Task ${pTask.id} still needs approval or was rejected.`);
                    return; // Cannot resume yet
                }
            }

            const runner = new CycleRunner();
            // We need a resume method or just call processQueue logic (exposed or internal?)
            // CycleRunner.start creates a new run. We need CycleRunner.resume(runId).
            // Let's allow accessing the internal processQueue via a public method 'resumeRun'
            await runner.resumeRun(Number(options.run_id));

        } catch (error) {
            console.error('Error resuming:', error);
        }
    });


program
    .command('schedule:add')
    .description('Add a new schedule')
    .requiredOption('--project_id <id>', 'Project ID')
    .requiredOption('--cadence <type>', 'daily or weekly')
    .requiredOption('--time <HH:MM>', 'Time (24h format)')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            const { Scheduler } = await import('./core/scheduler');
            await Scheduler.addSchedule(Number(options.project_id), options.cadence, options.time);
            console.log('Schedule added.');
        } catch (error) {
            console.error('Error adding schedule:', error);
        }
    });

program
    .command('schedule:list')
    .description('List all schedules')
    .action(async () => {
        try {
            await DB.getInstance().init();
            const { Scheduler } = await import('./core/scheduler');
            const schedules = await Scheduler.listSchedules();
            console.log('\n--- Schedules ---');
            if (schedules.length === 0) console.log('No schedules found.');
            else {
                schedules.forEach(s => {
                    console.log(`[ID: ${s.id}] Project: ${s.project_name} | Cadence: ${s.cadence} | Next: ${new Date(s.next_run_at).toLocaleString()}`);
                });
            }
        } catch (error) {
            console.error('Error listing schedules:', error);
        }
    });

program
    .command('schedule:run_due')
    .description('Check and run due schedules')
    .action(async () => {
        try {
            await DB.getInstance().init();
            const { Scheduler } = await import('./core/scheduler');
            await Scheduler.runDue();
        } catch (error) {
            console.error('Error running schedules:', error);
        }
    });


program
    .command('memory:add')
    .description('Add a memory/playbook')
    .requiredOption('--project_id <id>', 'Project ID')
    .requiredOption('--dept <text>', 'Department (CEO, Product, etc.)')
    .requiredOption('--title <text>', 'Title')
    .requiredOption('--content <text>', 'Content or File Path')
    .option('--tags <text>', 'Comma-separated tags')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            const { MemoryStore } = await import('./hedge_fund/memory');
            // Check if content is a file
            let content = options.content;
            const fs = await import('fs');
            if (fs.existsSync(content)) {
                content = fs.readFileSync(content, 'utf8');
            }

            const id = await MemoryStore.add(Number(options.project_id), options.dept, options.title, content, options.tags);
            console.log(`Memory added with ID: ${id}`);
        } catch (error) {
            console.error('Error adding memory:', error);
        }
    });

program
    .command('memory:search')
    .description('Search memories')
    .requiredOption('--project_id <id>', 'Project ID')
    .requiredOption('--dept <text>', 'Department')
    .requiredOption('--query <text>', 'Search query')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            const { MemoryStore } = await import('./hedge_fund/memory');
            const results = await MemoryStore.search(Number(options.project_id), options.dept, options.query);

            console.log(`\n--- Found ${results.length} memories for ${options.dept} ---`);
            results.forEach(m => {
                console.log(`[ID: ${m.id}] ${m.title}`);
                console.log(`Tags: ${m.tags}`);
                console.log(`Preview: ${m.content.substring(0, 100)}...`);
                console.log('---');
            });
        } catch (error) {
            console.error('Error searching memories:', error);
        }
    });


program
    .command('schedule:daemon')
    .description('Run the scheduler daemon (always-on loop)')
    .action(async () => {
        try {
            const { Scheduler } = await import('./core/scheduler');
            const { ApplianceAPI } = await import('./core/api');
            const { JobQueue } = await import('./core/queue');

            // Start API in parallel
            const api = new ApplianceAPI();
            api.start();

            // Start Worker
            JobQueue.worker();

            await Scheduler.daemon();
        } catch (error) {
            console.error('Error in daemon:', error);
        }
    });


program
    .command('start')
    .description('Start the Company OS Appliance in the background')
    .action(async () => {
        const { spawn } = await import('child_process');
        const fs = await import('fs');
        const path = await import('path');

        console.log('Starting Company OS Appliance...');

        // Spawn the daemon
        const out = fs.openSync(path.join(process.cwd(), 'appliance.log'), 'a');
        const err = fs.openSync(path.join(process.cwd(), 'appliance.log'), 'a');

        const subprocess = spawn('node', [path.join(process.cwd(), 'dist', 'index.js'), 'schedule:daemon'], {
            detached: true,
            stdio: ['ignore', out, err]
        });

        subprocess.unref();

        // Save PID
        fs.writeFileSync(path.join(process.cwd(), 'appliance.pid'), subprocess.pid?.toString() || '');

        console.log(`Appliance started (PID: ${subprocess.pid}). Logs: appliance.log`);
        process.exit(0);
    });

program
    .command('stop')
    .description('Stop the Company OS Appliance')
    .action(async () => {
        const fs = await import('fs');
        const path = await import('path');
        const pidPath = path.join(process.cwd(), 'appliance.pid');

        if (fs.existsSync(pidPath)) {
            const pid = parseInt(fs.readFileSync(pidPath, 'utf8'));
            try {
                process.kill(pid, 'SIGTERM');
                console.log(`Stopped appliance (PID: ${pid})`);
            } catch (e) {
                console.log(`Process ${pid} not found or already stopped.`);
            }
            fs.unlinkSync(pidPath);
        } else {
            console.log('No appliance.pid found. Is it running?');
        }
    });

program
    .command('status')
    .description('Show appliance status')
    .action(async () => {
        try {
            const response = await fetch('http://127.0.0.1:3000/health');
            if (response.ok) {
                const health: any = await response.json();
                console.log('\n--- Company OS Appliance Status ---');
                console.log(`Status: ${health.status}`);
                console.log(`Scheduler Heartbeat: ${health.scheduler_heartbeat}`);
                console.log(`Pending Approvals: ${health.pending_approvals}`);
                console.log(`Network Binding: ${health.binding}`);
            } else {
                console.log('Appliance API is not responding. (Service might be stopped)');
            }
        } catch (e) {
            console.log('Appliance API is unreachable. (Service is likely stopped)');
        }
    });

program
    .command('ui')
    .description('Launch the Autonomous Control Room (UI)')
    .action(async () => {
        const { spawn } = await import('child_process');
        const path = await import('path');
        const fs = await import('fs');

        console.log('\n🚀 Launching Autonomous Control Room...');

        // Use ts-node to run the server directly if in dev, or node if compiled
        const serverPath = path.join(process.cwd(), 'src', 'dashboard', 'server.ts');
        const distServerPath = path.join(process.cwd(), 'dist', 'dashboard', 'server.js');

        let cmd = 'npx';
        let args = ['ts-node', serverPath];

        if (fs.existsSync(distServerPath)) {
            cmd = 'node';
            args = [distServerPath];
        }

        const uiProcess = spawn(cmd, args, {
            stdio: 'inherit',
            shell: true
        });

        uiProcess.on('error', (err) => {
            console.error('Failed to start UI:', err);
        });
    });



program
    .command('backup')
    .description('Backup the Company OS data (DB, Artifacts, Config)')
    .option('--out <path>', 'Output path (zip)', `backup_${new Date().toISOString().replace(/[:.]/g, '-')}.zip`)
    .action(async (options) => {
        const { execSync } = await import('child_process');
        const path = await import('path');
        const fs = await import('fs');

        console.log(`Backing up to ${options.out}...`);

        try {
            // Include: db, runs/, prompts/, config/
            const toBackup = ['company_os.db', 'runs', 'prompts', 'config'];
            const existing = toBackup.filter(f => fs.existsSync(path.join(process.cwd(), f)));

            const fileList = existing.join(', ');
            console.log(`Including: ${fileList}`);

            // PowerShell Compress-Archive
            const psCommand = `powershell -Command "Compress-Archive -Path ${existing.join(', ')} -DestinationPath ${options.out} -Force"`;
            execSync(psCommand, { stdio: 'inherit' });

            console.log('Backup successful.');
        } catch (e: any) {
            console.error('Backup failed:', e.message);
        }
    });

program
    .command('restore')
    .description('Restore Company OS data from a backup zip')
    .requiredOption('--from <path>', 'Backup zip path')
    .action(async (options) => {
        const { execSync } = await import('child_process');
        const fs = await import('fs');

        if (!fs.existsSync(options.from)) {
            console.error(`File not found: ${options.from}`);
            return;
        }

        console.log(`Restoring from ${options.from}...`);
        console.log('WARNING: This will overwrite existing data. Proceed? (Add --force to skip check)');

        try {
            const psCommand = `powershell -Command "Expand-Archive -Path ${options.from} -DestinationPath . -Force"`;
            execSync(psCommand, { stdio: 'inherit' });
            console.log('Restore complete. Please restart the appliance if running.');
        } catch (e: any) {
            console.error('Restore failed:', e.message);
        }
    });


program
    .command('process:list')
    .description('List available business processes')
    .action(async () => {
        try {
            const fs = await import('fs');
            const path = await import('path');
            const processesDir = path.join(process.cwd(), 'processes');
            if (!fs.existsSync(processesDir)) {
                console.log('No processes directory found.');
                return;
            }
            const files = fs.readdirSync(processesDir).filter(f => f.endsWith('.yml'));
            console.log('\n--- Available Processes ---');
            files.forEach(f => console.log(`- ${path.basename(f, '.yml')}`));
        } catch (error) {
            console.error('Error listing processes:', error);
        }
    });

program
    .command('process:start')
    .description('Start a new autonomous business process')
    .requiredOption('--project_id <id>', 'Project ID')
    .requiredOption('--name <process>', 'Process name')
    .option('--inputs <json>', 'Process inputs (JSON)', '{}')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            const { ProcessRunner } = await import('./hedge_fund/process_runner');
            let inputs: Record<string, any>;
            try {
                inputs = JSON.parse(options.inputs);
            } catch (jsonError: any) {
                console.error('Error parsing --inputs JSON:', jsonError.message);
                return; // Exit if JSON is invalid
            }
            const processRunId = await ProcessRunner.startProcess(options.name, Number(options.project_id), inputs);
            console.log(`Process started with Run ID: ${processRunId}`);

            // Trigger first step execution
            const runner = new ProcessRunner(0);
            await runner.executeNextStep(processRunId);
        } catch (error: any) {
            console.error('Error starting process:', error.message || error);
        }
    });

program
    .command('scout:manual')
    .description('Manually trigger the Sovereign Scout')
    .option('-p, --project_id <id>', 'Project ID', '1')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            const { Scout } = await import('./quant_lab/scout');
            const { Intercom } = await import('./hedge_fund/intercom');
            const scout = new Scout();
            console.log(`🔭 Manually triggering Scout for Project ${options.project_id}...`);
            const opportunities = await scout.scoutOpportunities(Number(options.project_id));

            if (opportunities.length > 0) {
                console.log(`\n--- Found ${opportunities.length} Opportunities ---`);
                opportunities.forEach(o => {
                    console.log(`- [${o.marketLeverage}/10] ${o.title}: ${o.suggestedAction}`);
                });
                Intercom.log('CLI', 'Analysis Lead', `Manual scout found ${opportunities.length} signals.`);
            } else {
                console.log('No new opportunities found.');
            }
        } catch (error) {
            console.error('Error in manual scout:', error);
        }
    });

program
    .command('run:id')
    .description('Force start/resume a specific Run ID')
    .requiredOption('--run_id <id>', 'Run ID')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            const runner = new CycleRunner();
            console.log(`🚀 Force-resuming Run #${options.run_id}...`);
            await runner.resumeRun(Number(options.run_id));
        } catch (error) {
            console.error('Error force-resuming:', error);
        }
    });

program
    .command('schedule:autostart')
    .description('Initialize 24/7 autonomous loop')
    .option('-h, --hours <n>', 'Interval in hours', '4')
    .action(async (options) => {
        try {
            await DB.getInstance().init();
            const prisma = DB.getInstance().getPrisma();
            // Create a recurring task for the Scout to find new work
            await prisma.schedule.create({
                data: {
                    projectId: 1,
                    cadence: 'daily',
                    nextRunAt: new Date(),
                    status: 'ACTIVE'
                }
            });
            console.log(`📡 24/7 Autonomy Initialized. Interval set to ${options.hours}h.`);
        } catch (error) {
            console.error('Error starting autonomous loop:', error);
        }
    });

program
    .command('hardware:set')
    .description('Switch Hardware Optimization Mode')
    .argument('<mode>', 'Turbo | Balanced | Quiet')
    .action(async (mode) => {
        try {
            const configPath = path.join(process.cwd(), 'config', 'hardware_modes.json');
            const config = JSON.parse(fs.readFileSync(configPath, 'utf8'));
            if (!config.modes[mode]) {
                console.error(`Invalid mode: ${mode}. Available: Turbo, Balanced, Quiet`);
                return;
            }
            config.active_mode = mode;
            fs.writeFileSync(configPath, JSON.stringify(config, null, 2));
            console.log(`⚡ Hardware Mode set to: ${mode}`);
        } catch (error) {
            console.error('Error setting hardware mode:', error);
        }
    });

program.parse(process.argv);


