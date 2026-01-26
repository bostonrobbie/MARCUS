# QUICKSTART: Company OS Control Room

## 👁️ 1. Monitor the System (The Dashboard)
To watch your agents in real-time:
1. **Start Dashboard**: `npm run dashboard`
2. **Open Control Room**: [index.html](file:///C:/Users/User/Documents/AI/local-manus-agent-workspace/ai-company-os/src/dashboard/index.html)

---

## 2. Local-Only Runtime
The appliance is locked to `127.0.0.1`. No external network visibility is required or recommended.

## 3. Managing the Service
The `companyos` CLI manages the background daemon.

- **Start**: `node dist/index.js start`
- **Stop**: `node dist/index.js stop`
- **Status**: `node dist/index.js status`
- **Health Endpoint**: [http://127.0.0.1:3000/health](http://127.0.0.1:3000/health)

## 4. Windows Auto-Start
To ensure Company OS runs even after a reboot:
1. Open PowerShell as Administrator.
2. Run `scripts/setup_appliance.ps1`.
3. The "CompanyOS" task is now registered in Task Scheduler.

## 5. Security & Workspace Jail
- **Filesystem**: File actions are restricted to `allowed_paths` defined in `config/policies.json`.
- **Commands**: Read-only and "Safe" commands (like `git status` or `npm test`) bypass approval.
- **Risky Actions**: Installs (`npm install`) and system changes still require human approval via the Approvals Inbox.

## 6. Backup & Restore
Protect your data using native commands:
- **Backup**: `node dist/index.js backup` (Generates a .zip with DB and artifacts)
- **Restore**: `node dist/index.js restore --from <path-to-zip>`

## 7. Updating
To update safely:
1. `node dist/index.js stop`
2. `git pull`
3. `npm run build` (if applicable)
4. `node dist/index.js start`
