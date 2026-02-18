
import { CONFIG } from '../config';

export interface LLMResponse {
    content: string;
    files?: Array<{ path: string; content: string }>;
    tasks?: Array<{ title: string; type: string; payload: any }>;
}

export interface LLMOptions {
    temperature?: number;
    num_ctx?: number;
    num_thread?: number;
    num_gpu?: number;
    isDeepTruth?: boolean;
}

export interface LLMProvider {
    generate(prompt: string, options?: LLMOptions): Promise<LLMResponse>;
}

export class OllamaLLM implements LLMProvider {
    public async generate(prompt: string, options?: LLMOptions): Promise<LLMResponse> {
        try {
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), CONFIG.LLM_TIMEOUT || 300000);

            let systemPrompt = "You are a helpful business agent. Return JSON only when requested.";

            if (options?.isDeepTruth) {
                systemPrompt = `
You are in DEEP TRUTH MODE (Forensic Reasoning Protocol). 
Follow these steps for every response:
1. EPISTEMIC SKEPTICISM: Question assumptions.
2. SOURCE AUDITING: Link claims to primary business documents or data.
3. STEEL-MANNING: Present the strongest counter-argument to your own strategy.
4. OPINION LABELING: Start subjective sentences with [SUBJECTIVE].
5. FALSIFICATION: State what evidence would prove your current plan wrong.
6. THE LAST MILE: Ensure output is 1-click execution ready.
`;
            }

            const response = await fetch(`${CONFIG.OLLAMA_BASE_URL}/chat/completions`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    model: CONFIG.OLLAMA_MODEL,
                    messages: [
                        { role: "system", content: systemPrompt },
                        { role: "user", content: prompt }
                    ],
                    temperature: options?.temperature ?? CONFIG.LLM_TEMPERATURE,
                    stream: false,
                    options: {
                        num_ctx: options?.num_ctx ?? 8192,
                        num_thread: options?.num_thread,
                        num_gpu: options?.num_gpu
                    }
                }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);

            if (!response.ok) {
                throw new Error(`Ollama Error: ${response.status} ${response.statusText}`);
            }

            const data: any = await response.json();
            const content = data.choices[0].message.content;

            // Log the conversation for the dashboard feed
            try {
                const { Intercom } = await import('../hedge_fund/intercom');
                Intercom.logConversation(prompt, content);
            } catch (e) {
                // Ignore logging errors to not break execution
            }

            return {
                content: content
            };

        } catch (error: any) {
            console.error("LLM Generation Failed:", error.message);
            throw error;
        }
    }
}
