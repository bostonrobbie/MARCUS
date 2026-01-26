
import * as fs from 'fs';
import * as path from 'path';

export class Intercom {
    private static logFile: string = path.join(process.cwd(), 'logs', 'INTERCOM.log');

    public static log(from: string, to: string, message: string, meta?: any) {
        const ts = new Date().toISOString();
        const logEntry = `[${ts}] [${from} -> ${to}] ${message} ${meta ? JSON.stringify(meta) : ''}\n`;

        if (!fs.existsSync(path.dirname(this.logFile))) {
            fs.mkdirSync(path.dirname(this.logFile), { recursive: true });
        }

        fs.appendFileSync(this.logFile, logEntry);
        console.log(`[INTERCOM] ${from} -> ${to}: ${message.substring(0, 100)}...`);
    }

    public static logThought(agent: string, thought: string) {
        const ts = new Date().toISOString();
        const thoughtFile = path.join(process.cwd(), 'logs', 'THOUGHTS_STREAM.log');
        const logEntry = `[${ts}] [${agent}] THOUGHT: ${thought}\n---\n`;

        if (!fs.existsSync(path.dirname(thoughtFile))) {
            fs.mkdirSync(path.dirname(thoughtFile), { recursive: true });
        }

        fs.appendFileSync(thoughtFile, logEntry);
    }

    public static logConversation(prompt: string, response: string) {
        const ts = new Date().toISOString();
        const logFile = path.join(process.cwd(), 'logs', 'LLM_CONVERSATION.log');

        // Truncate prompt if too long for log readability, but keep enough context
        const safePrompt = prompt.length > 5000 ? prompt.substring(0, 5000) + '...' : prompt;

        const logEntry = `[${ts}] 🧠 USER/SYSTEM >\n${safePrompt}\n\n[${ts}] 🤖 AI >\n${response}\n---\n`;

        if (!fs.existsSync(path.dirname(logFile))) {
            fs.mkdirSync(path.dirname(logFile), { recursive: true });
        }

        fs.appendFileSync(logFile, logEntry);
    }
}
