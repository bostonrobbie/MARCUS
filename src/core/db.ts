import sqlite3 from 'sqlite3';
import { Database } from 'sqlite3';
import { PrismaClient } from '@prisma/client';

import 'dotenv/config'; // Ensure env vars are loaded

export class DB {
  private static instance: DB;
  private db: Database;
  private prisma: PrismaClient;

  private constructor() {
    this.db = new sqlite3.Database('company_os.db');
    // Prisma cleans up connection automatically, usually.
    // We pass log options for debugging if needed
    // Prisma cleans up connection automatically, usually.
    // We pass log options for debugging if needed
    this.prisma = new PrismaClient();
  }

  public static getInstance(): DB {
    if (!DB.instance) {
      DB.instance = new DB();
    }
    return DB.instance;
  }

  public getDb(): Database {
    return this.db;
  }

  public getPrisma(): PrismaClient {
    return this.prisma;
  }

  public async init(): Promise<void> {
    await this.prisma.$connect();
    // Legacy generic table creation
    const runSchema = `
      CREATE TABLE IF NOT EXISTS runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        objective TEXT NOT NULL,
        created_at TEXT NOT NULL,
        status TEXT NOT NULL
      );
    `;

    const taskSchema = `
      CREATE TABLE IF NOT EXISTS tasks (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        dept TEXT NOT NULL,
        title TEXT NOT NULL,
        description TEXT NOT NULL,
        risk_level TEXT NOT NULL,
        status TEXT NOT NULL,
        retries INTEGER DEFAULT 0,
        payload_json TEXT,
        result_json TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES runs(id)
      );
    `;

    const approvalSchema = `
      CREATE TABLE IF NOT EXISTS approvals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        task_id INTEGER NOT NULL,
        requested_at TEXT NOT NULL,
        approved_at TEXT,
        decision TEXT NOT NULL DEFAULT 'PENDING',
        notes TEXT,
        FOREIGN KEY(task_id) REFERENCES tasks(id)
      );
    `;

    const artifactSchema = `
      CREATE TABLE IF NOT EXISTS artifacts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        task_id INTEGER NOT NULL,
        path TEXT NOT NULL,
        type TEXT NOT NULL,
        created_at TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES runs(id),
        FOREIGN KEY(task_id) REFERENCES tasks(id)
      );
    `;

    const logSchema = `
      CREATE TABLE IF NOT EXISTS logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        task_id INTEGER,
        ts TEXT NOT NULL,
        level TEXT NOT NULL,
        message TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES runs(id),
        FOREIGN KEY(task_id) REFERENCES tasks(id)
      );
    `;

    const projectsSchema = `
      CREATE TABLE IF NOT EXISTS projects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        description TEXT,
        created_at TEXT NOT NULL
      );
    `;

    const schedulesSchema = `
      CREATE TABLE IF NOT EXISTS schedules (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        cadence TEXT NOT NULL, -- 'daily', 'weekly'
        next_run_at TEXT NOT NULL,
        status TEXT NOT NULL, -- 'ACTIVE', 'PAUSED'
        FOREIGN KEY(project_id) REFERENCES projects(id)
      );
    `;

    const kpisSchema = `
      CREATE TABLE IF NOT EXISTS kpis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        run_id INTEGER NOT NULL,
        project_id INTEGER NOT NULL,
        name TEXT NOT NULL,
        value REAL NOT NULL,
        unit TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(run_id) REFERENCES runs(id),
        FOREIGN KEY(project_id) REFERENCES projects(id)
      );
    `;

    const memoriesSchema = `
      CREATE TABLE IF NOT EXISTS memories (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        dept TEXT NOT NULL,
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        tags TEXT,
        timeline_index INTEGER DEFAULT 0,
        created_at TEXT NOT NULL,
        FOREIGN KEY(project_id) REFERENCES projects(id)
      );
    `;

    const snapshotsSchema = `
      CREATE TABLE IF NOT EXISTS snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER NOT NULL,
        summary TEXT NOT NULL,
        metadata_json TEXT,
        created_at TEXT NOT NULL,
        FOREIGN KEY(project_id) REFERENCES projects(id)
      );
    `;

    const processRunsSchema = `
      CREATE TABLE IF NOT EXISTS process_runs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        process_name TEXT NOT NULL,
        project_id INTEGER NOT NULL,
        status TEXT NOT NULL, -- 'RUNNING', 'COMPLETED', 'FAILED', 'PAUSED'
        inputs_json TEXT,
        started_at TEXT NOT NULL,
        finished_at TEXT,
        FOREIGN KEY(project_id) REFERENCES projects(id)
      );
    `;

    const processStepsSchema = `
      CREATE TABLE IF NOT EXISTS process_steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        process_run_id INTEGER NOT NULL,
        step_id TEXT NOT NULL,
        status TEXT NOT NULL, -- 'PENDING', 'RUNNING', 'COMPLETED', 'FAILED'
        task_id INTEGER,
        result_json TEXT,
        error TEXT,
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        FOREIGN KEY(process_run_id) REFERENCES process_runs(id),
        FOREIGN KEY(task_id) REFERENCES tasks(id)
      );
    `;

    return new Promise((resolve, reject) => {
      this.db.serialize(() => {
        this.db.run(runSchema);
        this.db.run(taskSchema);
        this.db.run(approvalSchema);
        this.db.run(artifactSchema);
        this.db.run(projectsSchema);
        this.db.run(schedulesSchema);
        this.db.run(kpisSchema);
        this.db.run(memoriesSchema);
        this.db.run(snapshotsSchema);
        this.db.run(processRunsSchema);
        this.db.run(processStepsSchema);
        this.db.run(logSchema, (err) => {
          if (err) reject(err);
          else {
            // Migration: Check if runs table has project_id
            this.db.all("PRAGMA table_info(runs)", (err, rows: any[]) => {
              const hasProjectId = rows.some(r => r.name === 'project_id');
              if (!hasProjectId) {
                this.db.run("ALTER TABLE runs ADD COLUMN project_id INTEGER DEFAULT 1");
              }

              // Migration: Check if memories table has timeline_index
              this.db.all("PRAGMA table_info(memories)", (err, mRows: any[]) => {
                const hasTimelineIndex = mRows.some(r => r.name === 'timeline_index');
                if (!hasTimelineIndex) {
                  this.db.run("ALTER TABLE memories ADD COLUMN timeline_index INTEGER DEFAULT 0");
                }
              });

              // Ensure default project exists
              this.db.get("SELECT count(*) as c FROM projects", (err, row: any) => {
                if (row.c === 0) {
                  this.db.run("INSERT INTO projects (name, description, created_at) VALUES ('Default Project', 'Auto-created default project', datetime('now'))");
                }
                resolve();
              });
            });
          }
        });
      });
    });
  }

  public close(): void {
    this.db.close();
  }
}
