
import * as dotenv from 'dotenv';
dotenv.config();

export const CONFIG = {
    OLLAMA_BASE_URL: process.env.OLLAMA_BASE_URL || 'http://localhost:11434/v1',
    OLLAMA_MODEL: process.env.OLLAMA_MODEL || 'qwen2.5:14b',
    LLM_TEMPERATURE: parseFloat(process.env.LLM_TEMPERATURE || '0.2'),
    LLM_TIMEOUT: parseInt(process.env.LLM_TIMEOUT || '300000'),
};
