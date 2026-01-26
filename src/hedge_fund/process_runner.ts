
import * as fs from 'fs';
import * as path from 'path';
import * as yaml from 'js-yaml';
import { DB } from '../core/db';
import { CycleRunner } from './cycle';
import { Logger } from '../core/logger';

export interface ProcessDefinition {
    name: string;
    objective_template: string;
    steps: ProcessStepDefinition[];
    done_criteria: any[];
}

export interface ProcessStepDefinition {
    id: string;
    dept: string;
    action_type: 'LLM_TASK' | 'WRITE_ARTIFACT' | 'RUN_COMMAND' | 'UPDATE_MEMORY' | 'WISDOM_SYNTHESIS' | 'SNAPSHOT' | 'MULTIPLEX' | 'POLISH' | 'ROUTING' | 'DELIVERY_PACKET' | 'EXECUTIVE_SUMMARY' | 'PERFORMANCE_AUDIT' | 'SCOUTING' | 'DEEP_TRUTH_AUDIT' | 'DATA_INGESTION';
    risk_level: string;
    prompt_template: string;
    expected_artifacts?: string[];
    quality_gates?: any[];
    multiplex_personas?: any[];
}

export class ProcessRunner {
    private logger: Logger;

    constructor(runId: number) {
        this.logger = new Logger(runId);
    }

    public static async startProcess(processName: string, projectId: number, inputs: any): Promise<number> {
        const prisma = DB.getInstance().getPrisma();
        const processPath = path.join(process.cwd(), 'processes', `${processName}.yml`);

        if (!fs.existsSync(processPath)) {
            throw new Error(`Process definition not found: ${processName}`);
        }

        const def = yaml.load(fs.readFileSync(processPath, 'utf8')) as ProcessDefinition;

        const processRun = await prisma.processRun.create({
            data: {
                processName: processName,
                projectId: projectId,
                status: 'RUNNING',
                inputsJson: JSON.stringify(inputs),
                startedAt: new Date()
            }
        });

        // Initialize steps
        for (const step of def.steps) {
            await prisma.processStep.create({
                data: {
                    processRunId: processRun.id,
                    stepId: step.id,
                    status: 'PENDING',
                    createdAt: new Date(),
                    updatedAt: new Date()
                }
            });
        }

        return processRun.id;
    }

    public async executeNextStep(processRunId: number) {
        const prisma = DB.getInstance().getPrisma();

        const step = await prisma.processStep.findFirst({
            where: {
                processRunId: processRunId,
                status: 'PENDING'
            },
            orderBy: { id: 'asc' }
        });

        if (!step) {
            await this.finalizeProcess(processRunId);
            return;
        }

        // Update step status
        await prisma.processStep.update({
            where: { id: step.id },
            data: { status: 'RUNNING', updatedAt: new Date() }
        });

        const run = await prisma.processRun.findUnique({
            where: { id: processRunId },
            include: { project: true }
        });

        if (!run) throw new Error('Process run not found');

        const processDir = path.join(process.cwd(), 'runs', `process_${processRunId}`);
        if (!fs.existsSync(processDir)) fs.mkdirSync(processDir, { recursive: true });

        const processPath = path.join(process.cwd(), 'processes', `${run.processName}.yml`);
        const def = yaml.load(fs.readFileSync(processPath, 'utf8')) as ProcessDefinition;
        const stepDef = def.steps.find((s: any) => s.id === step.stepId)!;

        // Trace & Transparency: Log the start of the step
        const { Intercom } = await import('./intercom');
        Intercom.log('ProcessRunner', stepDef.dept, `Executing ${stepDef.action_type} for step ${stepDef.id}`);

        try {
            // Shared prompt interpolation logic for multiple action types
            let prompt = stepDef.prompt_template;
            const inputs = JSON.parse(run.inputsJson || '{}');
            for (const [key, val] of Object.entries(inputs)) {
                prompt = prompt.replace(new RegExp(`{{${key}}}`, 'g'), String(val));
            }

            // Resolve previous steps results
            const prevSteps = await prisma.processStep.findMany({
                where: {
                    processRunId: processRunId,
                    status: 'COMPLETED'
                },
                select: { stepId: true, resultJson: true }
            });
            for (const ps of prevSteps) {
                if (ps.resultJson) {
                    try {
                        const result = JSON.parse(ps.resultJson);
                        // Support {{step_id.result}} or {{step_id.result.key}}
                        if (prompt.includes(`{{${ps.stepId}.result}}`)) {
                            prompt = prompt.replace(new RegExp(`{{${ps.stepId}.result}}`, 'g'), ps.resultJson);
                        }
                        // Shallow match for keys
                        for (const [resKey, resVal] of Object.entries(result)) {
                            const placeholder = `{{${ps.stepId}.result.${resKey}}}`;
                            if (prompt.includes(placeholder)) {
                                prompt = prompt.replace(new RegExp(placeholder.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'), 'g'), String(resVal));
                            }
                        }
                    } catch (e) { }
                }
            }

            if (stepDef.action_type === 'LLM_TASK') {
                const runner = new CycleRunner();
                const runId = await runner.start(prompt, Number(run.projectId), processDir);

                // Quality Gates Check (with v6.3.2 Recursive Self-Correction)
                if (stepDef.quality_gates) {
                    const { QualityGate } = await import('./quality_gate');
                    const { OllamaLLM } = await import('../core/llm');
                    const managerLLM = new OllamaLLM();

                    for (const gate of stepDef.quality_gates) {
                        const result = QualityGate.validate(processDir, gate);
                        if (!result.success) {
                            Intercom.log('ProcessRunner', 'Manager', `Quality Gate Failed for ${stepDef.id}: ${result.error}. Triggering Self-Correction.`);

                            // 1. Debrief
                            const debriefPrompt = `The agent failed a quality gate for step "${stepDef.id}".
Error: ${result.error}
Objective: ${prompt}
Provide a brief manager's debrief on why it failed and how to fix it. Limit to 1 paragraph.`;
                            const debrief = await managerLLM.generate(debriefPrompt);
                            Intercom.logThought('Manager', `DEBRIEF for ${stepDef.id}: ${debrief.content}`);

                            // 2. Retry with correction context
                            const retryPrompt = `
### RECURSIVE CORRECTION CYCLE ###
Your previous attempt for step "${stepDef.id}" failed a quality gate.
ERROR: ${result.error}
MANAGER FEEDBACK: ${debrief.content}

Please re-execute the original objective with these corrections:
${prompt}
`;
                            const retryRunId = await runner.start(retryPrompt, Number(run.projectId), processDir);

                            // Re-validate
                            const secondResult = QualityGate.validate(processDir, gate);
                            if (!secondResult.success) {
                                throw new Error(`Quality Gate Failed after recursive correction: ${secondResult.error}`);
                            }

                            Intercom.log('ProcessRunner', 'Manager', `Self-Correction SUCCESSFUL for ${stepDef.id}`);
                        }
                    }
                }

                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        taskId: runId,
                        status: 'COMPLETED',
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'WRITE_ARTIFACT') {
                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'WISDOM_SYNTHESIS') {
                const { WisdomLayer } = await import('../quant_lab/wisdom_layer');
                const history = await WisdomLayer.synthesizeHistory(Number(run.projectId));

                const wisdomPath = path.join(processDir, 'WISDOM_AUDIT.md');
                fs.writeFileSync(wisdomPath, history);

                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        resultJson: JSON.stringify({ path: 'WISDOM_AUDIT.md', content: history }),
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'SNAPSHOT') {
                const { Metronome } = await import('../quant_lab/metronome');
                const snapshotId = await Metronome.takeSnapshot(Number(run.projectId));
                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        resultJson: JSON.stringify({ snapshot_id: snapshotId }),
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'MULTIPLEX') {
                const { Multiplexer } = await import('./multiplex');
                const mux = new Multiplexer();
                let result = await mux.synthesizeConsensus(prompt, stepDef.multiplex_personas || []);

                // Extract thoughts if any
                const thoughtMatch = result.match(/<thought>([\s\S]*?)<\/thought>/);
                if (thoughtMatch) {
                    Intercom.logThought(stepDef.dept, thoughtMatch[1]);
                    result = result.replace(/<thought>[\s\S]*?<\/thought>/, '').trim();
                }

                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        resultJson: JSON.stringify({ content: result }),
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'POLISH') {
                const { Polisher } = await import('../quant_lab/polisher');
                const polisher = new Polisher();
                const targetFile = path.join(processDir, prompt); // Use interpolated prompt as filename
                await polisher.polishArtifact(targetFile, `Polishing requested in process ${run.processName}`);
                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'ROUTING') {
                const { UniversalProxy } = await import('./universal_proxy');
                const proxy = new UniversalProxy();
                const route = await proxy.routeTask(prompt);
                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        resultJson: JSON.stringify(route),
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'DELIVERY_PACKET') {
                const { Logger } = await import('../core/logger');
                const logger = new Logger(undefined, processDir);
                const deliveryDir = path.join(processDir, 'DELIVERY_PACKET');
                if (!fs.existsSync(deliveryDir)) fs.mkdirSync(deliveryDir, { recursive: true });

                // Copy expected artifacts to delivery packet
                if (stepDef.expected_artifacts) {
                    for (const art of stepDef.expected_artifacts) {
                        const src = path.join(processDir, art);
                        const dest = path.join(deliveryDir, path.basename(art));
                        if (fs.existsSync(src)) {
                            fs.copyFileSync(src, dest);
                            await logger.info(`Added ${art} to Delivery Packet`);
                        }
                    }
                }

                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'EXECUTIVE_SUMMARY') {
                const { OllamaLLM } = await import('../core/llm');
                const llm = new OllamaLLM();
                const { DigitalOrgRegistry } = await import('./registry');
                const ceo = DigitalOrgRegistry.getTitle('CEO')!;
                const prevSteps = await prisma.processStep.findMany({
                    where: {
                        processRunId: processRunId,
                        status: 'COMPLETED'
                    },
                    select: { stepId: true, resultJson: true }
                });

                const summaryPrompt = `
You are the CEO Synthesis Engine. Summarize the work done in this autonomous cycle.

Objective: ${run.project?.name || 'Project'} - ${run.processName}
Work performed:
${prevSteps.map(ps => `- [${ps.stepId}]: ${ps.resultJson || 'Completed'}`).join('\n')}

Create a high-level Executive Summary for the business owner. 
Focus on:
1. WHAT was delivered.
2. WHY it matters (Business Value).
3. THE LOVE EQUATION (Owner time saved vs leverage gained).
4. NEXT ACTIONS for the owner.

Format as a professional Markdown report.
`;
                const summary = await llm.generate(summaryPrompt, {
                    ...ceo.hardwareProfile,
                    isDeepTruth: ceo.isDeepTruthEnabled
                });

                let summaryContent = summary.content;
                const thoughtMatch = summaryContent.match(/<thought>([\s\S]*?)<\/thought>/);
                if (thoughtMatch) {
                    Intercom.logThought(ceo.title, thoughtMatch[1]);
                    summaryContent = summaryContent.replace(/<thought>[\s\S]*?<\/thought>/, '').trim();
                }

                const summaryPath = path.join(processDir, 'EXECUTIVE_SUMMARY.md');
                fs.writeFileSync(summaryPath, summaryContent);

                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        resultJson: JSON.stringify({ path: 'EXECUTIVE_SUMMARY.md', content: summaryContent }),
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'PERFORMANCE_AUDIT') {
                const { PerformanceReview } = await import('../quant_lab/performance');
                const audit = new PerformanceReview();
                // dept acts as the manager's department or title filter
                const report = await audit.conductCheckIn(stepDef.dept, Number(run.projectId));
                const auditPath = path.join(processDir, 'PERFORMANCE_AUDIT.md');
                fs.writeFileSync(auditPath, report);

                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        resultJson: JSON.stringify({ path: 'PERFORMANCE_AUDIT.md', content: report }),
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'SCOUTING') {
                const { Scout } = await import('../quant_lab/scout');
                const scout = new Scout();
                const opportunities = await scout.scoutOpportunities(Number(run.projectId));
                const scoutPath = path.join(processDir, 'SCOUT_REPORT.md');
                const reportContent = `# Proactive Opportunity Report\n\n${opportunities.map(o => `### ${o.title} (${o.ownerFriction} Friction)\n- **Description**: ${o.description}\n- **Leverage**: ${o.marketLeverage}\n- **Next Action**: ${o.suggestedAction}`).join('\n\n')}`;
                fs.writeFileSync(scoutPath, reportContent);

                // --- v7.3 AUTONOMY FLYWHEEL TRIGGER ---
                const highConfidence = opportunities.filter(o => Number(o.marketLeverage) >= 8 && Number(o.ownerFriction) <= 3);
                for (const opp of highConfidence) {
                    Intercom.log('ProcessRunner', 'Analysis Lead', `FLYWHEEL TRIGGER: High-Confidence Opportunity detected: ${opp.title}. Initiating Auto-Project.`);

                    // Trigger a new process cycle for this specific opportunity
                    // We assume a 'business_plan' process exists or use a generic one
                    try {
                        const newProcessRunId = await ProcessRunner.startProcess('business_plan', Number(run.projectId), {
                            objective: `Develop full business plan and execution strategy for: ${opp.title}. Description: ${opp.description}`
                        });
                        Intercom.log('ProcessRunner', 'CEO', `Autonomous Project Started for ${opp.title}. ID: ${newProcessRunId}`);
                    } catch (e: any) {
                        Intercom.log('ProcessRunner', 'CEO', `Flywheel Trigger Failed for ${opp.title}: ${e.message}`);
                    }
                }

                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        resultJson: JSON.stringify({ opportunities }),
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'DEEP_TRUTH_AUDIT') {
                const { OllamaLLM } = await import('../core/llm');
                const llm = new OllamaLLM();
                const { DigitalOrgRegistry } = await import('./registry');
                const ceo = DigitalOrgRegistry.getTitle('CEO')!;

                const auditPrompt = `
Objective: ${run.project?.name || 'Project'} - ${run.processName}
Context: ${prompt}

Perform a DEEP TRUTH AUDIT on the current progress. 
1. Identify any hidden assumptions.
2. Demand forensic proof for the most recent claims.
3. Label all non-empirical statements as [SUBJECTIVE].
4. State exactly what evidence would prove this strategy wrong.
`;
                const response = await llm.generate(auditPrompt, {
                    ...ceo.hardwareProfile,
                    isDeepTruth: true
                });

                // Extract thoughts if present
                let content = response.content;
                const thoughtMatch = content.match(/<thought>([\s\S]*?)<\/thought>/);
                if (thoughtMatch) {
                    Intercom.logThought(ceo.title, thoughtMatch[1]);
                    content = content.replace(/<thought>[\s\S]*?<\/thought>/, '').trim();
                }

                const auditPath = path.join(processDir, 'DEEP_TRUTH_AUDIT.md');
                fs.writeFileSync(auditPath, content);

                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        resultJson: JSON.stringify({ path: 'DEEP_TRUTH_AUDIT.md', content }),
                        updatedAt: new Date()
                    }
                });
            } else if (stepDef.action_type === 'DATA_INGESTION') {
                const { DataIngestor } = await import('../quant_lab/ingestion');
                const ingestor = new DataIngestor();
                // prompt is used as the target path here
                const targetPath = prompt;
                const summary = await ingestor.ingestDirectory(targetPath);

                const reportPath = path.join(processDir, 'DATA_INGESTION_REPORT.md');
                fs.writeFileSync(reportPath, summary);

                await prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'COMPLETED',
                        resultJson: JSON.stringify({ path: 'DATA_INGESTION_REPORT.md', content: summary }),
                        updatedAt: new Date()
                    }
                });
            }

            // Trigger next step
            await this.executeNextStep(processRunId);
        } catch (e: any) {
            const prisma = DB.getInstance().getPrisma();
            await prisma.$transaction([
                prisma.processStep.update({
                    where: { id: step.id },
                    data: {
                        status: 'FAILED',
                        error: e.message,
                        updatedAt: new Date()
                    }
                }),
                prisma.processRun.update({
                    where: { id: processRunId },
                    data: {
                        status: 'FAILED',
                        finishedAt: new Date()
                    }
                })
            ]);
        }
    }

    private async finalizeProcess(processRunId: number) {
        const prisma = DB.getInstance().getPrisma();
        const processDir = path.join(process.cwd(), 'runs', `process_${processRunId}`);

        // Generate Final PROCESS_REPORT.md
        let report = `# Process Run Report #${processRunId}\n\n`;

        const run = await prisma.processRun.findUnique({
            where: { id: processRunId },
            include: { project: true }
        });

        if (!run) return;

        report += `**Process**: ${run.processName}\n`;
        report += `**Status**: COMPLETED\n\n`;

        // Check for HUMAN_ACTIONS.md to extract top 3
        const humanActionsPath = path.join(processDir, 'HUMAN_ACTIONS.md');
        if (fs.existsSync(humanActionsPath)) {
            const actions = fs.readFileSync(humanActionsPath, 'utf8');
            report += `## Top Human Actions\n`;
            // Get first few list items
            const matches = actions.match(/^- .*/gm);
            if (matches) report += matches.slice(0, 3).join('\n') + '\n\n';
        }

        // Artifacts Summary
        report += `## Packets Generated\n`;
        ['OUTREACH_READY', 'LANDING_READY', 'OPS_READY'].forEach(pkg => {
            const pkgPath = path.join(processDir, pkg);
            if (fs.existsSync(pkgPath)) {
                report += `- **${pkg}**: Ready\n`;
            }
        });

        fs.writeFileSync(path.join(processDir, 'PROCESS_REPORT.md'), report);

        await prisma.processRun.update({
            where: { id: processRunId },
            data: {
                status: 'COMPLETED',
                finishedAt: new Date()
            }
        });
        console.log(`[PROCESS] Run ${processRunId} completed.`);
    }
}
