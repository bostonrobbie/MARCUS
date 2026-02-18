
import { z } from 'zod';

export const RiskLevelSchema = z.enum(['SAFE', 'REVIEW', 'BLOCKED']);

// Generic Task Schema for delegation (CEO)
export const DelegatedTaskSchema = z.object({
    dept: z.enum(['product', 'engineering', 'growth', 'ops', 'finance_risk']),
    title: z.string(),
    description: z.string(),
    risk_level: RiskLevelSchema,
    deliverables: z.array(z.string()).optional(),
    payload: z.any().optional()
});

export const CEOResponseSchema = z.object({
    analysis: z.string(),
    strategy: z.string(),
    tasks: z.array(DelegatedTaskSchema),
    kpis: z.array(z.object({
        name: z.string(),
        value: z.number(),
        unit: z.string().optional()
    })).optional()
});

// Product Schema
export const ProductResponseSchema = z.object({
    offer: z.object({
        title: z.string(),
        description: z.string(),
        pricing_model: z.string(),
    }),
    copy: z.object({
        headline: z.string(),
        subheadline: z.string(),
        body: z.string(),
        call_to_action: z.string()
    }),
    files_to_create: z.array(z.object({
        path: z.string(),
        content: z.string()
    })),
    kpis: z.array(z.object({
        name: z.string(),
        value: z.number(),
        unit: z.string().optional()
    })).optional()
});

// Engineering Schema
export const EngineeringResponseSchema = z.object({
    plan: z.string(),
    files_to_create: z.array(z.object({
        path: z.string(),
        content: z.string()
    })),
    commands_to_run: z.array(z.string()).optional(),
    kpis: z.array(z.object({
        name: z.string(),
        value: z.number(),
        unit: z.string().optional()
    })).optional()
});

// Growth Schema
export const GrowthResponseSchema = z.object({
    strategy: z.string(),
    outreach_list: z.array(z.object({
        name: z.string(),
        company: z.string(),
        email: z.string(),
        role: z.string()
    })),
    files_to_create: z.array(z.object({
        path: z.string(),
        content: z.string()
    })),
    kpis: z.array(z.object({
        name: z.string(),
        value: z.number(),
        unit: z.string().optional()
    })).optional()
});

// Unified Union Schema
export const AgentResponseSchema = z.union([
    CEOResponseSchema,
    ProductResponseSchema,
    EngineeringResponseSchema,
    GrowthResponseSchema
]);
