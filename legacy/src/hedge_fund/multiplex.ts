import { OllamaLLM } from '../core/llm';
import { AgentTitle, DigitalOrgRegistry } from './registry';

export interface AgentPersona {
    name: string;
    title: string; // Refers to a title in the Registry
}

export class Multiplexer {
    private llm: OllamaLLM;

    constructor() {
        this.llm = new OllamaLLM();
    }

    public async synthesizeConsensus(objective: string, personas: AgentPersona[]): Promise<string> {
        // Resolve personas to full titles
        const resolvedPersonas = personas.map(p => {
            const regTitle = DigitalOrgRegistry.getTitle(p.title);
            return {
                ...p,
                details: regTitle || {
                    title: p.title,
                    dept: 'General',
                    description: 'Specialist agent',
                    bias: 'Generalist',
                    heuristics: [],
                    hardwareProfile: { temperature: 0.2 }
                }
            };
        });

        // Parallel reasoning
        const responses = await Promise.all(resolvedPersonas.map(p => this.reasonAs(objective, p)));

        // Synthesize as CEO
        const ceoDetails = DigitalOrgRegistry.getTitle('CEO')!;
        const synthesisPrompt = `
Objective: ${objective}

We have divergent reasoning from ${resolvedPersonas.length} specialist agents:
${responses.map((r, i) => `[${resolvedPersonas[i].name} - ${resolvedPersonas[i].details.title}]: ${r}`).join('\n\n')}

As the CEO Synthesis Engine, synthesize the final, most robust consensus business plan. 
Prioritize "The Last Mile" (actionable execution) while balancing the divergent biases and adhering to the core business heuristics.
`;

        const response = await this.llm.generate(synthesisPrompt, {
            ...ceoDetails.hardwareProfile,
            isDeepTruth: ceoDetails.isDeepTruthEnabled
        });
        return response.content;
    }

    private async reasonAs(objective: string, persona: any): Promise<string> {
        const details = persona.details as AgentTitle;
        const prompt = `You are ${persona.name}, the ${details.title} of this Autonomous Business Unit.
Your Department: ${details.dept}
Your Bias: ${details.bias}
Core Heuristics:
${details.heuristics.map(h => `- ${h}`).join('\n')}

Objective: ${objective}

Reason through this objective from your specific professional perspective. 
YOU MUST INCLUDE A <thought> BLOCK STARTING YOUR RESPONSE TO SHOW YOUR INTERNAL REASONING.
Limit your response to 1 concise paragraph focused on your specialization and how to achieve "The Last Mile" for this task.`;

        const response = await this.llm.generate(prompt, {
            ...details.hardwareProfile,
            isDeepTruth: details.isDeepTruthEnabled
        });
        return response.content;
    }
}
