---
description: how to launch the autonomous control room dashboard
---

To monitor your digital corporation in real-time, follow these steps:

1. **Start the Backend Server**:
   Run the following command in your terminal:
   ```powershell
   npx ts-node src/dashboard/server.ts
   ```
   *The server will run on http://localhost:3030*

2. **Open the Interface**:
   Simply open the dashboard file in your browser:
   [index.html](file:///C:/Users/User/Documents/AI/local-manus-agent-workspace/ai-company-os/src/dashboard/index.html)

3. **Observe the Control Room**:
   You will see:
   - **Agent Pods**: Visual pulses when an agent (CEO, CTO, etc.) is active.
   - **Thought Stream**: The "Monologue" of what they are currently reasoning about.
   - **Intercom**: Real-time handovers between departments.
   - **Deliverables**: Links to the latest reports and business plans.
