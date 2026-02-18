
export type RiskLevel = 'SAFE' | 'REVIEW' | 'BLOCKED';

export type TaskStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'PAUSED' | 'NEEDS_APPROVAL';

export interface Project {
    id: number;
    name: string;
    description: string;
    created_at: string;
}

export interface Run {
    id: number;
    project_id: number;
    objective: string;
    created_at: string;
    status: string;
}

export interface Task {
    id: number;
    run_id: number;
    dept: string;
    title: string;
    description: string;
    risk_level: RiskLevel;
    status: TaskStatus;
    retries: number;
    payload_json: string;
    result_json: string | null;
    created_at: string;
    updated_at: string;
}

export interface Approval {
    id: number;
    task_id: number;
    requested_at: string;
    approved_at: string | null;
    decision: 'APPROVED' | 'REJECTED' | 'PENDING';
    notes: string | null;
}

export interface Artifact {
    id: number;
    run_id: number;
    task_id: number;
    path: string;
    type: string;
    created_at: string;
}

export interface Log {
    id: number;
    run_id: number;
    task_id: number | null;
    ts: string;
    level: string;
    message: string;
}

export interface Memory {
    id: number;
    project_id: number;
    dept: string;
    title: string;
    content: string;
    tags: string;
    created_at: string;
}

export interface Schedule {
    id: number;
    project_id: number;
    cadence: 'daily' | 'weekly';
    next_run_at: string;
    status: 'ACTIVE' | 'PAUSED';
}
