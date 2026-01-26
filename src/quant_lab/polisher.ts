
import { OllamaLLM } from '../core/llm';
import * as fs from 'fs';
import * as path from 'path';

export class Polisher {
    private llm: OllamaLLM;

    constructor() {
        this.llm = new OllamaLLM();
    }

    public async polishArtifact(filePath: string, context: string): Promise<void> {
        if (!fs.existsSync(filePath)) return;

        const content = fs.readFileSync(filePath, 'utf8');
        const extension = path.extname(filePath);

        const prompt = `
You are the "Universal Proxy" specialized in "The Last Mile" execution.
Your goal is to polish the following file to be professional, execution-ready, and optimized for the owner.

File Path: ${filePath}
Context: ${context}

Original Content:
${content}

Provide the POLISHED version of the content. Maintain the original format (${extension}).
Only return the polished content, nothing else.
`;

        const response = await this.llm.generate(prompt);
        fs.writeFileSync(filePath, response.content);
    }
}
