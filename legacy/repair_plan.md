# Repair Plan: System Stabilization & Dashboard "North Star"

## Goal
Restore confidence in the system by ensuring the **Command Center Dashboard** is the single source of truth and clearly displays:
1.  **Agent Thoughts** (Thinking)
2.  **Intercom Logs** (Saying)
3.  **Run Status** (Running)
4.  **Deliverables** (Outputs)

## Current State Analysis
- **Fragmentation**: `launch_command_center.bat` was spawning the wrong process.
- **Logging Disconnect**: The Dashboard reads from specific log files (`logs/INTERCOM.log`, `logs/THOUGHTS_STREAM.log`, `logs/LLM_CONVERSATION.log`). We need to verify the *Core Loop* actually writes to these.
- **Agent Idle**: User says "they haven't had the chance to run yet". We need a reliable "Start Autonomy" button or verified script that actually kicks off the loop and keeps it running.

## Fix Strategy

### 1. Core Loop & Logging Verification
*   **Audit**: Check `src/business/cycle.ts` and `src/business/process_runner.ts`.
*   **Fix**: Ensure every step (Plan, Execute, Verify) explicitly logs to `Intercom` (for "Saying") and `Intercom.logThought` (for "Thinking").
*   **Fix**: Ensure `Scout` execution logs to the dashboard.

### 2. Dashboard Data Plumbing
*   **Audit**: `src/dashboard/server.ts`.
*   **Fix**: Ensure the API endpoints (`/api/intercom`, `/api/thoughts`, `/api/active-tasks`) map 1:1 to the files the Core Loop is writing to.
*   **Fix**: Fix the "Outputs" section to show actual artifacts created (files in `runs/` or `artifacts/`).

### 3. "One Button" Launch
*   **Fix**: The `LAUNCH_COMMAND_CENTER.bat` must reliably start **both** the Dashboard Server AND the Agent Loop (in a "standing by" or "active" state).
*   **Feature**: Add a "START MISSION" button directly in the Dashboard UI to trigger `src/scripts/trigger_next.ts` or similar, so the user doesn't need CLI commands.

### 4. Regression Testing
*   **Test**: Run a full "Proprietary Data Audit" process and verify it appears in all 4 quadrants of the dashboard.
