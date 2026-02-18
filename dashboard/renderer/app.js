// ─── OpenClaw Dashboard - Renderer Application ──────────────────────────────
// Gateway WebSocket client + Chat UI + Activity Feed
// Protocol reference: OpenClaw gateway protocol v3

(function () {
  'use strict';

  // ─── State ───────────────────────────────────────────────────────────────

  let ws = null;
  let config = null;
  let connected = false;
  let sessionKey = 'main';
  let chatRunId = null;
  let streamText = '';
  let lastDeltaAt = null;
  let messages = [];
  let activityItems = [];
  let pendingRequests = new Map();
  let pendingTimers = new Map();
  let helloReceived = false;

  // Autonomy state
  let activeTab = 'tasks';
  let runState = 'idle';
  let activeJobId = null;
  let activeJobStatus = null;
  let swarms = [];
  let governanceSnapshot = null;
  let pendingApprovals = new Map();
  let currentApprovalId = null;
  let taskPollTimer = null;
  let swarmPollTimer = null;
  let governancePollTimer = null;
  let confirmCallback = null;

  // Reconnect state
  let reconnectAttempts = 0;
  let backoffMs = 800;
  let circuitOpen = false;
  let reconnectTimer = null;
  let closed = false;

  // Per-connection connect state (prevents double-connect)
  let connectSentForCurrentWs = false;
  let connectFallbackTimer = null;

  const MAX_RECONNECT_ATTEMPTS = 100;
  const CIRCUIT_RESET_MS = 60000;
  const BACKOFF_MULTIPLIER = 1.7;
  const MAX_BACKOFF_MS = 30000;
  const REQUEST_TIMEOUT_MS = 30000;
  const STUCK_TIMEOUT_MS = 180000; // 3 minutes - long-running agents need more time before flagging stuck
  const MAX_ACTIVITY_ITEMS = 250;

  // ─── DOM References ──────────────────────────────────────────────────────

  const statusDot = document.getElementById('statusDot');
  const statusText = document.getElementById('statusText');
  const btnReconnect = document.getElementById('btnReconnect');
  const btnStartGateway = document.getElementById('btnStartGateway');
  const chatMessagesEl = document.getElementById('chatMessages');
  const chatStreamingEl = document.getElementById('chatStreaming');
  const streamingTextEl = document.getElementById('streamingText');
  const chatInput = document.getElementById('chatInput');
  const btnSend = document.getElementById('btnSend');
  const btnAbort = document.getElementById('btnAbort');
  const activityFeed = document.getElementById('activityFeed');
  const tabHeaderActions = document.getElementById('tabHeaderActions');
  const sessionBadge = document.getElementById('sessionBadge');
  const btnNewChat = document.getElementById('btnNewChat');
  const splashOverlay = document.getElementById('splashOverlay');
  const splashStatus = document.getElementById('splashStatus');

  // Autonomy DOM refs
  const runStateBadge = document.getElementById('runStateBadge');
  const activeRunIdEl = document.getElementById('activeRunId');
  const btnPauseRun = document.getElementById('btnPauseRun');
  const btnResumeRun = document.getElementById('btnResumeRun');
  const btnStopRun = document.getElementById('btnStopRun');
  const btnRunTests = document.getElementById('btnRunTests');
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = {
    tasks: document.getElementById('tabTasks'),
    swarms: document.getElementById('tabSwarms'),
    governance: document.getElementById('tabGovernance'),
    activity: document.getElementById('tabActivity'),
    marcus: document.getElementById('tabMarcus'),
  };
  const marcusContent = document.getElementById('marcusContent');
  const tasksContent = document.getElementById('tasksContent');
  const swarmsContent = document.getElementById('swarmsContent');
  const governanceContent = document.getElementById('governanceContent');
  const approvalOverlay = document.getElementById('approvalOverlay');
  const approvalCommand = document.getElementById('approvalCommand');
  const approvalMeta = document.getElementById('approvalMeta');
  const btnApproveOnce = document.getElementById('btnApproveOnce');
  const btnApproveAlways = document.getElementById('btnApproveAlways');
  const btnDeny = document.getElementById('btnDeny');
  const confirmOverlay = document.getElementById('confirmOverlay');
  const confirmTitle = document.getElementById('confirmTitle');
  const confirmMessage = document.getElementById('confirmMessage');
  const btnConfirmYes = document.getElementById('btnConfirmYes');
  const btnConfirmNo = document.getElementById('btnConfirmNo');

  // ─── Utility ─────────────────────────────────────────────────────────────

  function uuid() {
    return crypto.randomUUID();
  }

  function timestamp() {
    return new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
  }

  function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  function extractText(message) {
    if (!message) return null;
    const content = message.content;
    if (!Array.isArray(content)) {
      if (typeof message === 'string') return message;
      if (typeof message.text === 'string') return message.text;
      return null;
    }
    for (const block of content) {
      if (block.type === 'text' && typeof block.text === 'string') return block.text;
    }
    return null;
  }

  // ─── Tool Call Metrics ──────────────────────────────────────────────────
  let toolCallCounts = {};
  let toolCallTotal = 0;
  let metricsSnapshot = null;
  let metricsPollTimer = null;

  function incrementToolCounter(toolName) {
    toolCallCounts[toolName] = (toolCallCounts[toolName] || 0) + 1;
    toolCallTotal++;
    updateToolMetricsBadge();
  }

  function updateToolMetricsBadge() {
    const badge = document.getElementById('toolMetricsBadge');
    if (badge) badge.textContent = toolCallTotal + ' calls';
  }

  function truncate(str, maxLen) {
    if (!str) return '';
    return str.length > maxLen ? str.substring(0, maxLen) + '\u2026' : str;
  }

  // ─── UI Updates ──────────────────────────────────────────────────────────

  function setConnectionStatus(state, text) {
    statusDot.className = 'status-dot ' + state;
    statusText.textContent = text || state;

    if (state === 'connected') {
      btnReconnect.style.display = 'none';
      btnStartGateway.style.display = 'none';
    } else if (state === 'disconnected') {
      btnReconnect.style.display = '';
      btnStartGateway.style.display = '';
    } else if (state === 'reconnecting') {
      btnReconnect.style.display = 'none';
      btnStartGateway.style.display = 'none';
    }
  }

  function scrollChatToBottom() {
    requestAnimationFrame(() => {
      chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
    });
  }

  function renderMessages() {
    // Remove welcome if we have messages
    const welcome = chatMessagesEl.querySelector('.chat-welcome');
    if (welcome && messages.length > 0) {
      welcome.remove();
    }

    // Only render new messages (append-only)
    const existing = chatMessagesEl.querySelectorAll('.msg');
    const startIdx = existing.length;

    for (let i = startIdx; i < messages.length; i++) {
      const msg = messages[i];
      const el = document.createElement('div');
      const cssClass = msg.role === 'user' ? 'msg-user' : (msg.isError ? 'msg-error' : 'msg-assistant');
      el.className = `msg ${cssClass}`;
      el.textContent = msg.text;

      const timeEl = document.createElement('div');
      timeEl.className = 'msg-time';
      timeEl.textContent = msg.time || '';
      el.appendChild(timeEl);

      chatMessagesEl.appendChild(el);
    }

    scrollChatToBottom();
  }

  function addMessage(role, text, isError = false) {
    messages.push({ role, text, isError, time: timestamp() });
    renderMessages();
  }

  function showStreaming(visible) {
    chatStreamingEl.style.display = visible ? '' : 'none';
    if (!visible) {
      streamingTextEl.textContent = '';
    }
  }

  function updateStreamingText(text) {
    streamingTextEl.textContent = text;
    // Auto-scroll the streaming area
    chatStreamingEl.scrollTop = chatStreamingEl.scrollHeight;
  }

  function setSendState(sending) {
    if (sending) {
      btnSend.style.display = 'none';
      btnAbort.style.display = '';
      chatInput.disabled = true;
    } else {
      btnSend.style.display = '';
      btnAbort.style.display = 'none';
      chatInput.disabled = false;
      chatInput.focus();
    }
  }

  // ─── Activity Feed ───────────────────────────────────────────────────────

  function addActivity(type, label, detail) {
    activityItems.push({ type, label, detail, time: timestamp() });
    if (activityItems.length > MAX_ACTIVITY_ITEMS) {
      activityItems.shift();
    }
    renderActivity();
  }

  function renderActivity() {
    // Remove empty state
    const empty = activityFeed.querySelector('.activity-empty');
    if (empty && activityItems.length > 0) {
      empty.remove();
    }

    // Only render new items
    const existing = activityFeed.querySelectorAll('.activity-item');
    const startIdx = existing.length;

    for (let i = startIdx; i < activityItems.length; i++) {
      const item = activityItems[i];
      const el = document.createElement('div');
      el.className = `activity-item activity-${item.type}`;

      let html = `<span class="activity-time">${escapeHtml(item.time)}</span>`;
      html += `<span class="activity-label">${escapeHtml(item.label)}</span>`;
      if (item.detail) {
        html += `<span>${escapeHtml(item.detail)}</span>`;
      }
      el.innerHTML = html;
      activityFeed.appendChild(el);
    }

    // Auto-scroll
    requestAnimationFrame(() => {
      activityFeed.scrollTop = activityFeed.scrollHeight;
    });
  }

  function clearActivity() {
    showConfirm('Clear Activity Log', 'This will clear the activity log only. Chat messages, trading strategies, and task data will NOT be affected.', function () {
      activityItems = [];
      activityFeed.innerHTML = '<div class="activity-empty"><p>Agent activity will appear here as tasks are processed.</p></div>';
      addActivity('info', 'System', 'Activity log cleared');
    });
  }

  // ─── New Chat / Fresh Workspace ─────────────────────────────────────────
  // Chat history list for session tracking
  let chatSessions = [];
  let currentChatIndex = -1;

  function startNewChat() {
    // Save current messages to session history if there are any
    if (messages.length > 0) {
      chatSessions.push({
        messages: messages.slice(),
        time: timestamp(),
        sessionKey: sessionKey,
      });
      currentChatIndex = chatSessions.length - 1;
    }

    // Reset visible chat without deleting anything from the gateway
    messages = [];
    chatMessagesEl.innerHTML = '';
    streamText = '';

    // Re-show welcome screen
    var welcomeEl = document.createElement('div');
    welcomeEl.className = 'chat-welcome';
    welcomeEl.innerHTML = '<div class="welcome-icon">&#x1F916;</div>' +
      '<h2>OpenClaw Agent</h2>' +
      '<p>Fresh workspace ready. Previous chat history is preserved.<br>Type a message below to get started.</p>';
    chatMessagesEl.appendChild(welcomeEl);

    addActivity('info', 'Chat', 'Started fresh workspace');
  }

  if (btnNewChat) {
    btnNewChat.addEventListener('click', function () {
      if (chatRunId) {
        showConfirm('Active Run', 'An agent task is currently running. Starting a new chat will not stop it. Continue?', startNewChat);
      } else {
        startNewChat();
      }
    });
  }

  // ─── Tab-Contextual Header Actions ──────────────────────────────────────
  function updateTabHeaderActions(tabName) {
    if (!tabHeaderActions) return;
    tabHeaderActions.innerHTML = '';

    if (tabName === 'activity') {
      var btn = document.createElement('button');
      btn.className = 'btn-small btn-ghost';
      btn.title = 'Clear activity log only (does NOT affect chat, strategies, or data)';
      btn.textContent = 'Clear Log';
      btn.addEventListener('click', clearActivity);
      tabHeaderActions.appendChild(btn);
    }
  }

  // ─── Tab Switching ──────────────────────────────────────────────────────

  function switchTab(tabName) {
    activeTab = tabName;
    tabBtns.forEach(function (btn) {
      btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    Object.keys(tabContents).forEach(function (key) {
      var el = tabContents[key];
      if (el) el.style.display = key === tabName ? '' : 'none';
    });
    updateTabHeaderActions(tabName);
    if (tabName === 'tasks' && connected) refreshTaskStatus();
    if (tabName === 'swarms' && connected) loadSwarms();
    if (tabName === 'governance' && connected) loadGovernance();
    if (tabName === 'marcus') loadMarcus();
  }

  tabBtns.forEach(function (btn) {
    btn.addEventListener('click', function () { switchTab(btn.dataset.tab); });
  });

  // ─── Agent Status Panel ────────────────────────────────────────────────

  function updateRunState(state, jobId) {
    runState = state;
    activeJobId = jobId || null;
    runStateBadge.className = 'run-state-badge run-state-' + state;
    runStateBadge.textContent = state;
    activeRunIdEl.textContent = jobId ? jobId.slice(0, 18) + '...' : '';
    activeRunIdEl.title = jobId || '';
    btnPauseRun.style.display = (state === 'running') ? '' : 'none';
    btnResumeRun.style.display = (state === 'paused') ? '' : 'none';
    btnStopRun.style.display = (state === 'running' || state === 'paused') ? '' : 'none';
  }

  // ─── Confirm Dialog ────────────────────────────────────────────────────

  function showConfirm(title, message, onConfirm) {
    confirmTitle.textContent = title;
    confirmMessage.textContent = message;
    confirmCallback = onConfirm;
    confirmOverlay.style.display = '';
  }

  function hideConfirm() {
    confirmOverlay.style.display = 'none';
    confirmCallback = null;
  }

  btnConfirmYes.addEventListener('click', function () {
    if (confirmCallback) confirmCallback();
    hideConfirm();
  });
  btnConfirmNo.addEventListener('click', hideConfirm);

  // ─── Run Control Actions ───────────────────────────────────────────────

  btnPauseRun.addEventListener('click', async function () {
    if (!activeJobId) return;
    try {
      await request('autonomy.task.pause', { jobId: activeJobId });
      updateRunState('paused', activeJobId);
      addActivity('info', 'Paused', 'Task paused');
    } catch (err) {
      addActivity('error', 'Pause Failed', err.message);
    }
  });

  btnResumeRun.addEventListener('click', async function () {
    if (!activeJobId) return;
    try {
      await request('autonomy.task.resume', { jobId: activeJobId });
      updateRunState('running', activeJobId);
      addActivity('info', 'Resumed', 'Task resumed');
    } catch (err) {
      addActivity('error', 'Resume Failed', err.message);
    }
  });

  btnStopRun.addEventListener('click', function () {
    if (!activeJobId) return;
    showConfirm('Stop Run?', 'This will cancel the current task: ' + activeJobId, async function () {
      try {
        await request('autonomy.task.cancel', { jobId: activeJobId });
        updateRunState('idle', null);
        addActivity('info', 'Cancelled', 'Task cancelled');
      } catch (err) {
        addActivity('error', 'Cancel Failed', err.message);
      }
    });
  });

  // ─── Tasks Tab Rendering ───────────────────────────────────────────────

  var TASK_ICONS = {
    pending: '\u25CB',
    decomposing: '\u2699',
    executing: '\u25B6',
    verifying: '\u2714',
    completed: '\u2705',
    failed: '\u274C',
    blocked: '\u26D4',
  };

  function renderTasks(jobStatus) {
    if (!jobStatus) {
      tasksContent.innerHTML = '<div class="activity-empty"><p>No active tasks. Send a message or submit a task to see autonomous execution.</p></div>';
      return;
    }

    var html = '';

    // Job header card
    html += '<div class="task-job-card">';
    html += '<div class="task-job-header">';
    html += '<span class="task-job-id">' + escapeHtml(jobStatus.jobId) + '</span>';
    html += '<span class="task-job-status ' + jobStatus.status + '">' + escapeHtml(jobStatus.status) + '</span>';
    html += '</div>';
    html += '<div class="task-job-instruction">' + escapeHtml(jobStatus.instruction.length > 200 ? jobStatus.instruction.slice(0, 200) + '...' : jobStatus.instruction) + '</div>';

    // Progress bar
    var p = jobStatus.progress;
    var total = p.total || 1;
    var pct = Math.round((p.completed / total) * 100);
    var fillClass = p.failed > 0 ? (p.failed > p.completed ? 'bad' : 'warn') : 'good';
    html += '<div class="task-progress-bar"><div class="task-progress-fill ' + fillClass + '" style="width:' + pct + '%"></div></div>';
    html += '<div class="task-progress-label">' + p.completed + '/' + p.total + ' completed';
    if (p.failed > 0) html += ', ' + p.failed + ' failed';
    html += '</div>';

    // Governance budget
    var g = jobStatus.governance;
    if (g && g.toolCallsBudget > 0) {
      var budgetPct = Math.round((g.toolCallsUsed / g.toolCallsBudget) * 100);
      var budgetCls = budgetPct < 60 ? 'safe' : (budgetPct < 85 ? 'caution' : 'danger');
      html += '<div class="task-budget-row">';
      html += '<span class="task-budget-label">Tool Budget</span>';
      html += '<div class="task-budget-bar"><div class="task-budget-fill ' + budgetCls + '" style="width:' + Math.min(budgetPct, 100) + '%"></div></div>';
      html += '<span class="task-budget-value">' + g.toolCallsUsed + '/' + g.toolCallsBudget + '</span>';
      html += '</div>';
    }
    html += '</div>';

    // Subtask timeline
    if (jobStatus.tasks && jobStatus.tasks.length > 0) {
      for (var i = 0; i < jobStatus.tasks.length; i++) {
        var task = jobStatus.tasks[i];
        var isCurrent = task.id === jobStatus.currentTask;
        var stepClass = task.status;
        if (isCurrent && task.status !== 'completed' && task.status !== 'failed') stepClass = 'executing';
        var icon = TASK_ICONS[task.status] || '\u2022';

        html += '<div class="task-step ' + stepClass + '">';
        html += '<div class="task-step-icon">' + icon + '</div>';
        html += '<div class="task-step-body">';
        html += '<div class="task-step-title">' + escapeHtml(task.title) + '</div>';

        // Meta badges
        html += '<div class="task-step-meta">';
        html += '<span class="priority-badge priority-' + task.priority + '">' + escapeHtml(task.priority) + '</span>';
        if (task.estimatedComplexity) {
          html += '<span class="complexity-badge">' + escapeHtml(task.estimatedComplexity) + '</span>';
        }
        if (task.attempts && task.attempts.length > 0) {
          html += '<span class="attempt-badge">attempt ' + task.attempts.length + '/' + task.maxRetries + '</span>';
        }
        html += '</div>';

        // Description
        if (task.description) {
          var desc = task.description.length > 120 ? task.description.slice(0, 120) + '...' : task.description;
          html += '<div class="task-step-desc">' + escapeHtml(desc) + '</div>';
        }

        // Assigned to
        if (task.assignedTo) {
          html += '<div class="task-step-assigned">Agent: ' + escapeHtml(task.assignedTo) + '</div>';
        }

        // Dependencies
        if (task.dependsOn && task.dependsOn.length > 0) {
          var depNames = task.dependsOn.map(function (depId) {
            var dep = jobStatus.tasks.find(function (t) { return t.id === depId; });
            return dep ? dep.title : depId.slice(0, 8);
          });
          html += '<div class="task-step-deps">Depends on: ' + escapeHtml(depNames.join(', ')) + '</div>';
        }

        html += '</div></div>';
      }
    }

    // Artifacts section
    if (jobStatus.artifacts && jobStatus.artifacts.length > 0) {
      html += '<div class="task-artifacts-section">';
      html += '<div class="task-artifacts-title">Artifacts (' + jobStatus.artifacts.length + ')</div>';
      for (var j = 0; j < jobStatus.artifacts.length; j++) {
        var a = jobStatus.artifacts[j];
        var sizeKb = (a.sizeBytes / 1024).toFixed(1);
        html += '<div class="artifact-item">';
        html += '<span class="artifact-type-badge">' + escapeHtml(a.type) + '</span>';
        html += '<span class="artifact-label" title="' + escapeHtml(a.label) + '">' + escapeHtml(a.label) + '</span>';
        html += '<span class="artifact-size">' + sizeKb + ' KB</span>';
        html += '<button class="artifact-view-btn" data-artifact-id="' + escapeHtml(a.id) + '">View</button>';
        html += '</div>';
      }
      html += '</div>';
    }

    tasksContent.innerHTML = html;

    // Attach artifact view handlers
    tasksContent.querySelectorAll('.artifact-view-btn').forEach(function (btn) {
      btn.addEventListener('click', function () { viewArtifact(btn.dataset.artifactId); });
    });
  }

  // ─── Multi-Task List Rendering (shows all recent tasks) ──────────────

  function renderTasksList(tasksList) {
    if (!tasksList || tasksList.length === 0) {
      tasksContent.innerHTML = '<div class="activity-empty"><div class="empty-explainer">' +
        '<p><strong>No Active Tasks</strong></p>' +
        '<p>Tasks appear here when you send a message to the agent. The agent decomposes your request into subtasks, executes them, and reports progress.</p>' +
        '<p class="empty-hint">Each task shows its status, progress, and budget usage.</p>' +
        '</div></div>';
      return;
    }

    var html = '';

    for (var ti = 0; ti < Math.min(tasksList.length, 20); ti++) {
      var task = tasksList[ti];
      var statusClass = task.status || 'pending';
      var statusIcons = {
        'planning': '\u2699', 'executing': '\u25B6', 'paused': '\u23F8',
        'completed': '\u2705', 'failed': '\u274C', 'cancelled': '\u26D4',
      };
      var statusIcon = statusIcons[task.status] || '\u25CB';

      html += '<div class="task-job-card">';
      html += '<div class="task-job-header">';
      html += '<span class="task-job-id">' + statusIcon + ' ' + escapeHtml((task.jobId || '').slice(0, 20)) + '</span>';
      html += '<span class="task-job-status ' + statusClass + '">' + escapeHtml(task.status || 'unknown') + '</span>';
      html += '</div>';

      // Instruction
      var inst = task.instruction || '';
      html += '<div class="task-job-instruction">' + escapeHtml(inst.length > 150 ? inst.slice(0, 150) + '...' : inst) + '</div>';

      // Progress bar
      var p = task.progress || { total: 0, completed: 0, failed: 0, pending: 0 };
      var total = p.total || 1;
      var pct = Math.round((p.completed / total) * 100);
      var fillClass = p.failed > 0 ? 'warn' : 'good';
      html += '<div class="task-progress-bar"><div class="task-progress-fill ' + fillClass + '" style="width:' + pct + '%"></div></div>';
      html += '<div class="task-progress-label">' + p.completed + '/' + p.total + ' completed';
      if (p.failed > 0) html += ', ' + p.failed + ' failed';
      html += '</div>';

      // Governance budget (if available)
      var g = task.governance;
      if (g && g.toolCallsBudget > 0) {
        var budgetPct = Math.round((g.toolCallsUsed / g.toolCallsBudget) * 100);
        var budgetCls = budgetPct < 60 ? 'safe' : (budgetPct < 85 ? 'caution' : 'danger');
        html += '<div class="task-budget-row">';
        html += '<span class="task-budget-label">Tool Budget</span>';
        html += '<div class="task-budget-bar"><div class="task-budget-fill ' + budgetCls + '" style="width:' + Math.min(budgetPct, 100) + '%"></div></div>';
        html += '<span class="task-budget-value">' + g.toolCallsUsed + '/' + g.toolCallsBudget + '</span>';
        html += '</div>';
      }

      // Timestamp
      if (task.createdAt) {
        var created = new Date(task.createdAt);
        html += '<div class="task-timestamp">Created: ' + created.toLocaleTimeString() + '</div>';
      }

      html += '</div>';
    }

    if (tasksList.length > 20) {
      html += '<div style="text-align:center;color:var(--text-dim);font-size:11px;padding:8px;">+ ' + (tasksList.length - 20) + ' more tasks</div>';
    }

    tasksContent.innerHTML = html;
  }

  async function viewArtifact(artifactId) {
    try {
      var result = await request('autonomy.artifacts.get', { artifactId: artifactId });
      if (result && result.content) {
        showConfirm('Artifact', result.content.slice(0, 5000), function () {});
      }
    } catch (err) {
      addActivity('error', 'Artifact', 'Failed to load: ' + err.message);
    }
  }

  // ─── Swarms Tab Rendering ──────────────────────────────────────────────

  function renderSwarms(swarmsList) {
    if (!swarmsList || swarmsList.length === 0) {
      swarmsContent.innerHTML = '<div class="activity-empty"><div class="empty-explainer">' +
        '<p><strong>No Active Swarms</strong></p>' +
        '<p>Swarms are groups of agents that work together in parallel on complex tasks. ' +
        'They launch automatically when a task is broken into subtasks that benefit from multi-agent coordination.</p>' +
        '<p class="empty-hint">Each swarm has a coordinator, researchers, implementers, and verifiers working concurrently.</p>' +
        '</div></div>';
      return;
    }

    var html = '';
    for (var i = 0; i < swarmsList.length; i++) {
      var s = swarmsList[i];
      html += '<div class="swarm-card">';

      // Header
      html += '<div class="swarm-header">';
      html += '<span class="swarm-id">' + escapeHtml(s.id || '') + '</span>';
      html += '<span class="swarm-status-badge ' + (s.status || '') + '">' + escapeHtml(s.status || '') + '</span>';
      html += '</div>';

      // Instruction
      if (s.instruction) {
        var inst = s.instruction.length > 150 ? s.instruction.slice(0, 150) + '...' : s.instruction;
        html += '<div class="swarm-instruction">' + escapeHtml(inst) + '</div>';
      }

      // Agent roster
      if (s.agents && s.agents.length > 0) {
        html += '<div class="swarm-agents-grid">';
        for (var j = 0; j < s.agents.length; j++) {
          var agent = s.agents[j];
          var roleClass = 'role-' + (agent.role || 'coordinator');
          html += '<div class="swarm-agent">';
          html += '<span class="swarm-agent-role ' + roleClass + '">' + escapeHtml(agent.role || 'agent') + '</span>';
          html += '<div class="swarm-agent-status">';
          html += '<span class="swarm-agent-dot ' + (agent.status || 'idle') + '"></span>';
          html += '<span>' + escapeHtml(agent.status || 'idle') + '</span>';
          html += '</div>';
          if (agent.currentTaskId) {
            var taskTitle = agent.currentTaskId;
            if (s.tasks) {
              var found = s.tasks.find(function (t) { return t.id === agent.currentTaskId; });
              if (found) taskTitle = found.title;
            }
            html += '<div class="swarm-agent-task">' + escapeHtml(taskTitle) + '</div>';
          }
          html += '</div>';
        }
        html += '</div>';
      } else if (s.agentCount != null) {
        html += '<div style="font-size:11px;color:var(--text-dim);">' + s.agentCount + ' agents, ' + s.taskCount + ' tasks</div>';
      }

      // Progress bar
      if (s.tasks && s.tasks.length > 0) {
        var completed = s.tasks.filter(function (t) { return t.status === 'completed'; }).length;
        var failed = s.tasks.filter(function (t) { return t.status === 'failed'; }).length;
        var executing = s.tasks.filter(function (t) { return t.status === 'executing'; }).length;
        var pending = s.tasks.length - completed - failed - executing;
        var t = s.tasks.length;
        html += '<div class="swarm-progress">';
        if (completed > 0) html += '<div class="swarm-progress-completed" style="width:' + (completed / t * 100) + '%"></div>';
        if (executing > 0) html += '<div class="swarm-progress-executing" style="width:' + (executing / t * 100) + '%"></div>';
        if (failed > 0) html += '<div class="swarm-progress-failed" style="width:' + (failed / t * 100) + '%"></div>';
        if (pending > 0) html += '<div class="swarm-progress-pending" style="width:' + (pending / t * 100) + '%"></div>';
        html += '</div>';
        html += '<div class="swarm-progress-label">' + completed + ' done, ' + executing + ' running, ' + failed + ' failed, ' + pending + ' pending</div>';
      }

      // Result
      if (s.result) {
        var resultClass = s.result.success ? '' : ' failed-result';
        html += '<div class="swarm-result' + resultClass + '">';
        html += 'Completed: ' + s.result.completedTasks + ', Failed: ' + s.result.failedTasks + ', Agents: ' + s.result.totalAgentsSpawned;
        html += '\nDuration: ' + (s.result.durationMs / 1000).toFixed(1) + 's';
        if (s.result.consolidatedOutput) {
          html += '\n\n' + escapeHtml(s.result.consolidatedOutput.slice(0, 500));
        }
        if (s.result.errors && s.result.errors.length > 0) {
          html += '\n\nErrors:\n' + escapeHtml(s.result.errors.join('\n'));
        }
        html += '</div>';
      }

      html += '</div>';
    }

    swarmsContent.innerHTML = html;
  }

  // ─── Governance Tab Rendering ──────────────────────────────────────────

  function renderGovernance(snapshot) {
    if (!snapshot) {
      governanceContent.innerHTML = '<div class="activity-empty"><p>Governance data loads when connected.</p></div>';
      return;
    }

    var html = '';

    // Budget explanation header
    html += '<div class="governance-info-banner">';
    html += '<div class="governance-info-title">Resource Budgets</div>';
    html += '<div class="governance-info-desc">Budgets protect against runaway execution and infinite loops. ';
    html += 'When a budget is reached, the agent will <strong>pause</strong> and report the limit. ';
    html += 'Task budgets reset automatically when a new task starts. Session budgets persist until a halt or restart.</div>';
    html += '</div>';

    // Budget meters with inline explanations
    html += '<div class="governance-card">';
    html += '<div class="governance-card-title">Active Budgets</div>';

    function budgetMeter(name, used, total, helpText, behaviorText) {
      var pct = total > 0 ? Math.round((used / total) * 100) : 0;
      var cls = pct < 60 ? 'safe' : (pct < 85 ? 'caution' : 'danger');
      var remaining = Math.max(0, total - used);
      var statusLabel = pct >= 100 ? 'EXHAUSTED' : pct >= 85 ? 'NEAR LIMIT' : pct >= 60 ? 'MODERATE' : 'HEALTHY';
      var h = '<div class="budget-meter">';
      h += '<div class="budget-meter-label">';
      h += '<span class="budget-meter-name">' + name + '</span>';
      h += '<span class="budget-meter-status budget-status-' + cls + '">' + statusLabel + '</span>';
      h += '</div>';
      h += '<div class="budget-meter-bar"><div class="budget-meter-fill ' + cls + '" style="width:' + Math.min(pct, 100) + '%"></div></div>';
      h += '<div class="budget-meter-details">';
      h += '<span>' + used + ' / ' + total + ' used (' + pct + '%)</span>';
      h += '<span>' + remaining + ' remaining</span>';
      h += '</div>';
      if (helpText) {
        h += '<div class="budget-meter-help">' + helpText + '</div>';
      }
      if (behaviorText && pct >= 85) {
        h += '<div class="budget-meter-warning">' + behaviorText + '</div>';
      }
      h += '</div>';
      return h;
    }

    html += budgetMeter(
      'Session Tool Calls',
      snapshot.sessionCalls || 0,
      snapshot.sessionBudget || 0,
      'Total tool calls allowed this session. Includes all tasks. Resets on gateway restart or emergency halt.',
      'When exhausted: Agent stops and reports the limit. Use Emergency Halt to reset.'
    );
    html += budgetMeter(
      'Task Tool Calls',
      snapshot.taskCalls || 0,
      snapshot.taskBudget || 0,
      'Tool calls for the current task only. Resets automatically when a new task begins.',
      'When exhausted: Current task stops. Submit a new task to continue working.'
    );
    html += budgetMeter(
      'Agent Spawns',
      snapshot.spawns || 0,
      snapshot.spawnBudget || 0,
      'Number of sub-agents spawned this session. Controls multi-agent parallelism.',
      'When exhausted: New spawns are blocked. Existing agents continue running.'
    );
    html += '</div>';

    // Warnings
    if (snapshot.warnings && snapshot.warnings.length > 0) {
      html += '<div class="governance-card governance-card-warn">';
      html += '<div class="governance-card-title">Active Warnings (' + snapshot.warnings.length + ')</div>';
      html += '<div class="governance-warnings">';
      for (var i = 0; i < snapshot.warnings.length; i++) {
        html += '<div class="governance-warning-item">' + escapeHtml(snapshot.warnings[i]) + '</div>';
      }
      html += '</div></div>';
    }

    // Loop detections
    if (snapshot.loopDetections && snapshot.loopDetections.length > 0) {
      html += '<div class="governance-card governance-card-danger">';
      html += '<div class="governance-card-title">Loop Detections</div>';
      html += '<div class="governance-info-desc" style="margin-bottom:6px;">A tool was called with identical parameters multiple times, which may indicate a stuck agent.</div>';
      for (var j = 0; j < snapshot.loopDetections.length; j++) {
        var d = snapshot.loopDetections[j];
        html += '<div class="governance-loop-item">Tool "' + escapeHtml(d.toolName) + '" called ' + d.count + ' times with the same inputs</div>';
      }
      html += '</div>';
    }

    // Formatted summary
    if (snapshot.formatted) {
      html += '<div class="governance-card">';
      html += '<div class="governance-card-title">Raw Summary</div>';
      html += '<div class="governance-formatted">' + escapeHtml(snapshot.formatted) + '</div>';
      html += '</div>';
    }

    // Emergency halt button with clear labeling
    html += '<div class="governance-card governance-halt-card">';
    html += '<div class="governance-halt-info">';
    html += '<strong>Emergency Halt</strong> cancels ALL running tasks and swarms, and resets all budget counters. ';
    html += 'Trading strategy results and chat history are NOT affected.';
    html += '</div>';
    html += '<button class="btn-halt" id="btnHaltAll">Emergency Halt All</button>';
    html += '</div>';

    governanceContent.innerHTML = html;

    // Attach halt handler
    var haltBtn = document.getElementById('btnHaltAll');
    if (haltBtn) {
      haltBtn.addEventListener('click', function () {
        showConfirm('Emergency Halt', 'This will cancel ALL running tasks and swarms and reset budget counters.\n\nTrading strategies and chat history will NOT be deleted.\n\nAre you sure?', async function () {
          try {
            var result = await request('autonomy.governance.halt', {});
            addActivity('error', 'HALT', 'Halted ' + (result.tasksHalted || 0) + ' tasks, ' + (result.swarmsHalted || 0) + ' swarms');
            updateRunState('idle', null);
            loadGovernance();
          } catch (err) {
            addActivity('error', 'Halt Failed', err.message);
          }
        });
      });
    }
  }

  // ─── Approval System ───────────────────────────────────────────────────

  function showApproval(approval) {
    currentApprovalId = approval.id;
    approvalCommand.textContent = (approval.request && approval.request.command) || 'Unknown command';
    var meta = '';
    if (approval.request) {
      if (approval.request.cwd) meta += 'CWD: ' + approval.request.cwd + '\n';
      if (approval.request.agentId) meta += 'Agent: ' + approval.request.agentId + '\n';
      if (approval.request.security) meta += 'Security: ' + approval.request.security + '\n';
    }
    if (approval.expiresAtMs) {
      var remaining = Math.max(0, Math.round((approval.expiresAtMs - Date.now()) / 1000));
      meta += 'Expires in: ' + remaining + 's';
    }
    approvalMeta.textContent = meta;
    approvalOverlay.style.display = '';
    addActivity('lifecycle', 'Approval', 'Approval required: ' + ((approval.request && approval.request.command) || '').slice(0, 60));
  }

  function hideApproval() {
    approvalOverlay.style.display = 'none';
    currentApprovalId = null;
  }

  async function resolveApproval(decision) {
    if (!currentApprovalId) return;
    try {
      await request('exec.approval.resolve', { id: currentApprovalId, decision: decision });
      addActivity('info', 'Approval', decision + ': ' + currentApprovalId.slice(0, 16));
    } catch (err) {
      addActivity('error', 'Approval', 'Resolve failed: ' + err.message);
    }
    pendingApprovals.delete(currentApprovalId);
    hideApproval();
    if (pendingApprovals.size > 0) {
      var next = pendingApprovals.values().next().value;
      showApproval(next);
    }
  }

  btnApproveOnce.addEventListener('click', function () { resolveApproval('allow-once'); });
  btnApproveAlways.addEventListener('click', function () { resolveApproval('allow-always'); });
  btnDeny.addEventListener('click', function () { resolveApproval('deny'); });

  // ─── Task & Swarm Data Loading ─────────────────────────────────────────

  // Store the full task list for rendering
  let allTasksList = [];

  async function loadActiveTask() {
    try {
      var result = await request('autonomy.task.list', {});
      if (result && result.tasks && result.tasks.length > 0) {
        allTasksList = result.tasks;
        var active = result.tasks.find(function (t) {
          return t.status === 'planning' || t.status === 'executing' || t.status === 'paused';
        });
        if (active) {
          activeJobId = active.jobId;
          activeJobStatus = active;
          var stateMap = { 'planning': 'running', 'executing': 'running', 'paused': 'paused' };
          updateRunState(stateMap[active.status] || 'idle', active.jobId);
        }
        renderTasksList(result.tasks);
      } else {
        allTasksList = [];
        renderTasksList([]);
      }
    } catch (err) {
      // Autonomy may not be active
    }
  }

  async function refreshTaskStatus() {
    // Always reload the full task list to show all recent activity
    await loadActiveTask();
  }

  async function loadSwarms() {
    try {
      var result = await request('autonomy.swarm.list', {});
      if (result && result.swarms) {
        swarms = result.swarms;
        renderSwarms(result.swarms);
      }
    } catch (err) {
      // Non-critical
    }
  }

  async function loadGovernance() {
    try {
      var result = await request('autonomy.governance.status', {});
      if (result) {
        governanceSnapshot = result;
        renderGovernance(result);
      }
    } catch (err) {
      // Non-critical
    }
  }

  async function loadMetrics() {
    try {
      var result = await request('autonomy.metrics', {});
      if (result) {
        metricsSnapshot = result;
        renderMetricsInGovernance(result);
      }
    } catch (err) {
      // Non-critical
    }
  }

  async function loadHealth() {
    try {
      var result = await request('autonomy.health', {});
      if (result) {
        updateHealthIndicator(result);
      }
    } catch (err) {
      // Non-critical
    }
  }

  function renderMetricsInGovernance(metrics) {
    var metricsEl = document.getElementById('metricsPanel');
    if (!metricsEl) return;

    var b = metrics.bridge || {};
    var g = metrics.governance || {};
    var html = '';

    // Platform health header
    html += '<div class="governance-card">';
    html += '<div class="governance-card-title">\u2764 Platform Metrics</div>';

    // Counters grid
    html += '<div class="metrics-grid">';
    html += metricTile('Tasks Submitted', b.tasksSubmitted || 0, 'info');
    html += metricTile('Tasks Completed', b.tasksCompleted || 0, 'good');
    html += metricTile('Tasks Failed', b.tasksFailed || 0, b.tasksFailed > 0 ? 'warn' : 'good');
    html += metricTile('Tasks Cancelled', b.tasksCancelled || 0, 'info');
    html += metricTile('Stalls Detected', b.stallsDetected || 0, b.stallsDetected > 0 ? 'warn' : 'good');
    html += metricTile('Stall Recoveries', b.stallRecoveries || 0, 'info');
    html += metricTile('Artifacts Stored', b.artifactsStored || 0, 'info');
    html += metricTile('Emergency Halts', b.haltsTriggered || 0, b.haltsTriggered > 0 ? 'bad' : 'good');
    html += metricTile('Active Jobs', b.activeJobs || 0, 'info');
    html += metricTile('Memory (MB)', metrics.memoryUsageMB || 0, (metrics.memoryUsageMB || 0) > 1000 ? 'warn' : 'good');
    html += metricTile('Uptime', formatUptime(metrics.uptimeSeconds || 0), 'info');
    html += metricTile('Local Tool Calls', toolCallTotal, 'info');
    html += '</div>';
    html += '</div>';

    // Local tool call breakdown
    if (toolCallTotal > 0) {
      html += '<div class="governance-card">';
      html += '<div class="governance-card-title">\u26A1 Tool Call Breakdown (This Session)</div>';
      var sorted = Object.entries(toolCallCounts).sort(function (a, b) { return b[1] - a[1]; });
      html += '<div class="tool-breakdown">';
      for (var i = 0; i < sorted.length; i++) {
        var pct = Math.round((sorted[i][1] / toolCallTotal) * 100);
        html += '<div class="tool-breakdown-row">';
        html += '<span class="tool-breakdown-name">' + escapeHtml(sorted[i][0]) + '</span>';
        html += '<span class="tool-breakdown-bar"><span class="tool-breakdown-fill" style="width:' + pct + '%"></span></span>';
        html += '<span class="tool-breakdown-count">' + sorted[i][1] + ' (' + pct + '%)</span>';
        html += '</div>';
      }
      html += '</div></div>';
    }

    metricsEl.innerHTML = html;
  }

  function metricTile(label, value, type) {
    return '<div class="metric-tile metric-' + type + '">' +
      '<div class="metric-value">' + value + '</div>' +
      '<div class="metric-label">' + escapeHtml(label) + '</div>' +
      '</div>';
  }

  function formatUptime(seconds) {
    if (seconds < 60) return seconds + 's';
    if (seconds < 3600) return Math.floor(seconds / 60) + 'm';
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    return h + 'h ' + m + 'm';
  }

  function updateHealthIndicator(healthResult) {
    var badge = document.getElementById('healthBadge');
    if (!badge) return;
    badge.className = 'health-badge health-' + (healthResult.status || 'unknown');
    badge.textContent = (healthResult.status || 'unknown').toUpperCase();
    badge.title = (healthResult.issues || []).join('; ') || 'No issues';
  }

  // ─── Task Update Handler ───────────────────────────────────────────────

  function handleTaskUpdate(payload) {
    if (!payload || !payload.jobId) return;
    var statusMap = {
      'planning': 'running', 'executing': 'running', 'paused': 'paused',
      'completed': 'idle', 'failed': 'error', 'cancelled': 'idle',
    };
    var newRunState = statusMap[payload.status] || 'idle';
    var isTerminal = payload.status === 'completed' || payload.status === 'failed' || payload.status === 'cancelled';
    updateRunState(newRunState, isTerminal ? null : payload.jobId);
    if (!isTerminal) activeJobId = payload.jobId;
    var progressStr = payload.progress ? ' (' + payload.progress.completed + '/' + payload.progress.total + ')' : '';
    var instrSnippet = payload.instruction ? ' - ' + payload.instruction.slice(0, 40) : '';
    addActivity('lifecycle', 'Task ' + payload.status, payload.jobId.slice(0, 16) + progressStr + instrSnippet);
    // Always refresh task list - it should update regardless of active tab
    refreshTaskStatus();
    // Also refresh governance since tool counts may have changed
    loadGovernance();
  }

  function handleSwarmEvent(payload) {
    if (!payload) return;
    var detail = (payload.swarmId || '').slice(0, 12);
    if (payload.eventType) detail += ' ' + payload.eventType;
    if (payload.agentId) detail += ' agent:' + payload.agentId.slice(0, 12);
    addActivity('lifecycle', 'Swarm', detail);
    if (activeTab === 'swarms') loadSwarms();
  }

  // ─── Autonomy Polling ──────────────────────────────────────────────────

  function startAutonomyPolling() {
    stopAutonomyPolling();
    taskPollTimer = setInterval(function () {
      if (connected) refreshTaskStatus();
    }, 3000);
    swarmPollTimer = setInterval(function () {
      if (connected && (activeTab === 'swarms' || swarms.some(function (s) { return s.status !== 'completed' && s.status !== 'failed'; }))) {
        loadSwarms();
      }
    }, 5000);
    governancePollTimer = setInterval(function () {
      if (connected) loadGovernance();
    }, 10000);
    metricsPollTimer = setInterval(function () {
      if (connected) loadMetrics();
    }, 5000);
  }

  function stopAutonomyPolling() {
    if (taskPollTimer) { clearInterval(taskPollTimer); taskPollTimer = null; }
    if (swarmPollTimer) { clearInterval(swarmPollTimer); swarmPollTimer = null; }
    if (governancePollTimer) { clearInterval(governancePollTimer); governancePollTimer = null; }
    if (metricsPollTimer) { clearInterval(metricsPollTimer); metricsPollTimer = null; }
  }

  // ─── Test Runner ───────────────────────────────────────────────────────

  btnRunTests.addEventListener('click', async function () {
    btnRunTests.disabled = true;
    btnRunTests.textContent = 'Running...';
    addActivity('info', 'Tests', 'Running baseline tests...');
    try {
      var result = await window.openclaw.runBaselineTest();
      showTestResult(result);
    } catch (err) {
      showTestResult({ ok: false, message: err.message });
    }
    btnRunTests.disabled = false;
    btnRunTests.textContent = 'Run Tests';
  });

  function showTestResult(result) {
    var toast = document.createElement('div');
    toast.className = 'test-result-toast ' + (result.ok ? 'test-result-pass' : 'test-result-fail');
    toast.textContent = result.ok
      ? 'Tests Passed' + (result.message ? ': ' + result.message : '')
      : 'Tests Failed: ' + (result.message || 'Unknown error');
    document.body.appendChild(toast);
    addActivity(result.ok ? 'lifecycle' : 'error', 'Tests', toast.textContent);
    setTimeout(function () { toast.remove(); }, 6000);
  }

  // ─── WebSocket Gateway Client ────────────────────────────────────────────

  function connectWebSocket() {
    if (closed || !config) return;
    if (circuitOpen) return;

    // Close any existing connection first
    if (ws) {
      try { ws.onclose = null; ws.onerror = null; ws.onmessage = null; ws.close(); } catch (e) {}
      ws = null;
    }

    const url = config.wsUrl;
    setConnectionStatus('reconnecting', 'Connecting...');
    addActivity('info', 'System', `Connecting to ${url}...`);

    try {
      ws = new WebSocket(url);
    } catch (err) {
      setConnectionStatus('disconnected', 'Connection failed');
      scheduleReconnect();
      return;
    }

    // Reset per-connection state
    connectSentForCurrentWs = false;
    if (connectFallbackTimer) {
      clearTimeout(connectFallbackTimer);
      connectFallbackTimer = null;
    }

    ws.onopen = () => {
      addActivity('info', 'System', 'WebSocket opened, waiting for challenge...');
      // The gateway sends connect.challenge immediately on open.
      // Set a fallback timer in case the challenge doesn't arrive.
      connectFallbackTimer = setTimeout(() => {
        if (!connectSentForCurrentWs && ws && ws.readyState === WebSocket.OPEN) {
          connectSentForCurrentWs = true;
          sendConnectRequest(null);
        }
      }, 2000);
    };

    ws.onmessage = (event) => {
      // Handle both string and Blob data (Electron WebSocket may send either)
      const data = event.data;
      if (typeof data === 'string') {
        handleMessage(data);
      } else if (data instanceof Blob) {
        data.text().then((text) => handleMessage(text));
      } else if (data instanceof ArrayBuffer) {
        handleMessage(new TextDecoder().decode(data));
      } else {
        handleMessage(String(data));
      }
    };

    ws.onclose = (event) => {
      const wasConnected = connected;
      connected = false;
      helloReceived = false;
      connectSentForCurrentWs = false;
      if (connectFallbackTimer) {
        clearTimeout(connectFallbackTimer);
        connectFallbackTimer = null;
      }
      ws = null;

      // Flush pending
      for (const [id, pending] of pendingRequests) {
        pending.reject(new Error('Connection closed'));
        clearTimeout(pendingTimers.get(id));
      }
      pendingRequests.clear();
      pendingTimers.clear();

      stopAutonomyPolling();

      if (closed) {
        setConnectionStatus('disconnected', 'Disconnected');
        return;
      }

      // Code 1012 = server restart expected
      if (event.code === 1012) {
        addActivity('info', 'System', 'Gateway restarting...');
        setConnectionStatus('reconnecting', 'Gateway restarting...');
      } else if (wasConnected) {
        addActivity('info', 'System', `Connection lost (code ${event.code})`);
        setConnectionStatus('reconnecting', 'Reconnecting...');
      }

      scheduleReconnect();
    };

    ws.onerror = () => {
      // onclose will fire after this
    };
  }

  function handleMessage(raw) {
    let frame;
    try {
      frame = JSON.parse(raw);
    } catch {
      return;
    }

    if (frame.type === 'event') {
      handleEvent(frame);
    } else if (frame.type === 'res') {
      handleResponse(frame);
    }
  }

  function handleEvent(frame) {
    const { event, payload } = frame;

    if (event === 'connect.challenge') {
      // Received challenge nonce - send connect request if not already sent
      if (!connectSentForCurrentWs) {
        connectSentForCurrentWs = true;
        if (connectFallbackTimer) {
          clearTimeout(connectFallbackTimer);
          connectFallbackTimer = null;
        }
        sendConnectRequest(payload?.nonce || null);
      }
      return;
    }

    if (event === 'chat') {
      handleChatEvent(payload);
      return;
    }

    if (event === 'agent') {
      handleAgentEvent(payload);
      return;
    }

    if (event === 'tick') {
      // Heartbeat - ignore
      return;
    }

    if (event === 'shutdown') {
      addActivity('info', 'System', 'Gateway shutting down...');
      return;
    }

    if (event === 'presence') {
      // Client presence updates - ignore for now
      return;
    }

    // Autonomy events
    if (event === 'autonomy.task.update') {
      handleTaskUpdate(payload);
      return;
    }

    if (event === 'autonomy.swarm.event') {
      handleSwarmEvent(payload);
      return;
    }

    if (event === 'autonomy.governance.warning') {
      addActivity('error', 'Governance', (payload && payload.message) || 'Budget warning');
      if (activeTab === 'governance') loadGovernance();
      return;
    }

    if (event === 'exec.approval.requested') {
      if (payload && payload.id) {
        pendingApprovals.set(payload.id, payload);
        if (!currentApprovalId) showApproval(payload);
      }
      return;
    }

    if (event === 'exec.approval.resolved') {
      if (payload && payload.id) {
        pendingApprovals.delete(payload.id);
        addActivity('info', 'Approval Resolved', (payload.decision || '') + ' by ' + (payload.resolvedBy || 'unknown'));
      }
      return;
    }
  }

  function sendConnectRequest(nonce) {
    // The connect request uses the standard request() flow.
    // However, helloReceived is false at this point, so we must bypass the
    // check in request(). We send the frame directly and track it manually.
    const connectParams = {
      minProtocol: 3,
      maxProtocol: 3,
      client: {
        id: 'webchat-ui',
        version: '1.0.0',
        platform: 'electron-win32',
        mode: 'webchat',
        instanceId: 'electron-desktop-' + Date.now(),
      },
      role: 'operator',
      scopes: ['operator.admin', 'operator.approvals', 'operator.pairing'],
      caps: [],
      auth: { token: config.token },
    };

    const id = uuid();
    const frame = { type: 'req', id, method: 'connect', params: connectParams };

    // Track this request so handleResponse() can resolve it
    const connectPromise = new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        pendingRequests.delete(id);
        pendingTimers.delete(id);
        reject(new Error('Connect timeout'));
      }, REQUEST_TIMEOUT_MS);

      pendingRequests.set(id, { resolve, reject });
      pendingTimers.set(id, timer);
    });

    try {
      ws.send(JSON.stringify(frame));
    } catch (err) {
      pendingRequests.delete(id);
      addActivity('error', 'Error', 'Failed to send connect request');
      return;
    }

    // Handle the hello-ok response
    connectPromise
      .then((hello) => {
        handleHelloOk(hello || {});
      })
      .catch((err) => {
        addActivity('error', 'Connect Failed', err.message);
        ws?.close(4008, 'connect failed');
      });
  }

  function handleHelloOk(hello) {
    helloReceived = true;
    connected = true;
    reconnectAttempts = 0;
    backoffMs = 800;
    circuitOpen = false;

    setConnectionStatus('connected', 'Connected');
    addActivity('lifecycle', 'Connected', 'Gateway handshake complete');

    // Extract session defaults from snapshot
    if (hello?.snapshot?.sessionDefaults?.mainSessionKey) {
      sessionKey = hello.snapshot.sessionDefaults.mainSessionKey;
      sessionBadge.textContent = sessionKey;
    }

    // Load chat history (main session + Telegram sessions)
    loadChatHistory();
    loadTelegramHistory();

    // Load autonomy state and start polling
    loadActiveTask();
    loadSwarms();
    loadGovernance();
    loadMetrics();
    loadHealth();
    startAutonomyPolling();
  }

  function handleResponse(frame) {
    const pending = pendingRequests.get(frame.id);
    if (!pending) return;

    pendingRequests.delete(frame.id);
    const timer = pendingTimers.get(frame.id);
    if (timer) {
      clearTimeout(timer);
      pendingTimers.delete(frame.id);
    }

    if (frame.ok) {
      pending.resolve(frame.payload);
    } else {
      pending.reject(new Error(frame.error?.message || 'Request failed'));
    }
  }

  function scheduleReconnect() {
    if (closed) return;
    if (reconnectTimer) return;

    reconnectAttempts++;
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
      circuitOpen = true;
      setConnectionStatus('disconnected', `Circuit open - auto-retry in ${CIRCUIT_RESET_MS / 1000}s`);
      addActivity('error', 'Circuit Open', `${MAX_RECONNECT_ATTEMPTS} reconnect attempts failed; auto-reset in ${CIRCUIT_RESET_MS / 1000}s`);
      // Auto-reset circuit breaker so overnight agents eventually reconnect
      setTimeout(() => {
        if (circuitOpen && !closed) {
          addActivity('info', 'Circuit Reset', 'Auto-resetting circuit breaker');
          manualReconnect();
        }
      }, CIRCUIT_RESET_MS);
      return;
    }

    const jitter = Math.random() * backoffMs * 0.3;
    const delay = backoffMs + jitter;
    backoffMs = Math.min(backoffMs * BACKOFF_MULTIPLIER, MAX_BACKOFF_MS);

    setConnectionStatus('reconnecting', `Reconnecting in ${Math.round(delay / 1000)}s... (${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS})`);

    reconnectTimer = setTimeout(() => {
      reconnectTimer = null;
      connectWebSocket();
    }, delay);
  }

  function manualReconnect() {
    circuitOpen = false;
    reconnectAttempts = 0;
    backoffMs = 800;
    if (reconnectTimer) {
      clearTimeout(reconnectTimer);
      reconnectTimer = null;
    }
    connectWebSocket();
  }

  // ─── Gateway Requests ────────────────────────────────────────────────────

  function request(method, params) {
    return new Promise((resolve, reject) => {
      if (!ws || ws.readyState !== WebSocket.OPEN || !helloReceived) {
        reject(new Error('Not connected to gateway'));
        return;
      }

      const id = uuid();
      const frame = { type: 'req', id, method, params: params || {} };

      const timer = setTimeout(() => {
        pendingRequests.delete(id);
        pendingTimers.delete(id);
        reject(new Error(`Request timeout: ${method}`));
      }, REQUEST_TIMEOUT_MS);

      pendingRequests.set(id, { resolve, reject });
      pendingTimers.set(id, timer);

      try {
        ws.send(JSON.stringify(frame));
      } catch (err) {
        pendingRequests.delete(id);
        clearTimeout(timer);
        pendingTimers.delete(id);
        reject(err);
      }
    });
  }

  // ─── Chat Logic ──────────────────────────────────────────────────────────

  async function loadChatHistory() {
    try {
      const result = await request('chat.history', { sessionKey, limit: 100 });
      if (result?.messages && Array.isArray(result.messages)) {
        // Clear existing and load from history
        messages = [];
        chatMessagesEl.innerHTML = '';

        for (const msg of result.messages) {
          const text = extractText(msg);
          if (text) {
            messages.push({
              role: msg.role || 'assistant',
              text,
              isError: false,
              time: msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
            });
          }
        }

        if (messages.length > 0) {
          renderMessages();
        }
        addActivity('info', 'History', `Loaded ${messages.length} messages`);
      }
    } catch (err) {
      // History loading is not critical - just show empty chat
      addActivity('info', 'History', 'No previous messages');
    }
  }

  async function loadTelegramHistory() {
    try {
      const res = await request('integration.telegram.threads', {});
      const threads = res?.threads || [];
      const active = threads.filter((t) => t.active && t.forwardToTelegram);
      for (const thread of active.slice(0, 5)) {
        try {
          const hist = await request('chat.history', { sessionKey: thread.sessionKey, limit: 20 });
          if (hist?.messages && Array.isArray(hist.messages)) {
            const label = thread.telegramUsername || 'Telegram';
            for (const msg of hist.messages) {
              const text = extractText(msg);
              if (text) {
                const role = msg.role || 'assistant';
                const prefix = role === 'user' ? '[' + label + '] ' : '';
                messages.push({
                  role,
                  text: prefix + text,
                  isError: false,
                  time: msg.timestamp ? new Date(msg.timestamp).toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' }) : '',
                });
              }
            }
          }
        } catch {
          // Skip individual thread errors
        }
      }
      if (active.length > 0) {
        renderMessages();
        addActivity('info', 'Telegram', `Loaded history from ${active.length} Telegram thread(s)`);
      }
    } catch {
      // Telegram history loading is not critical
    }
  }

  async function sendMessage() {
    const text = chatInput.value.trim();
    if (!text) return;

    if (!connected || !helloReceived) {
      addMessage('assistant', 'Not connected to gateway. Please wait for connection.', true);
      return;
    }

    // Check for stop command
    if (text.toLowerCase() === '/stop' || text.toLowerCase() === '/abort') {
      await abortCurrentRun();
      chatInput.value = '';
      return;
    }

    const idempotencyKey = uuid();
    chatInput.value = '';
    resizeInput();

    // Add user message to chat
    addMessage('user', text);

    // Set sending state
    chatRunId = idempotencyKey;
    streamText = '';
    lastDeltaAt = Date.now();
    setSendState(true);
    showStreaming(true);

    try {
      await request('chat.send', {
        sessionKey,
        message: text,
        idempotencyKey,
        deliver: false,
      });
      addActivity('lifecycle', 'Task Sent', text.substring(0, 80) + (text.length > 80 ? '...' : ''));
    } catch (err) {
      chatRunId = null;
      streamText = '';
      setSendState(false);
      showStreaming(false);
      addMessage('assistant', 'Error sending message: ' + err.message, true);
      addActivity('error', 'Send Failed', err.message);
    }
  }

  async function abortCurrentRun() {
    if (!chatRunId) return;

    try {
      await request('chat.abort', { sessionKey, runId: chatRunId });
      addActivity('info', 'Aborted', 'Run aborted by user');
    } catch (err) {
      addActivity('error', 'Abort Failed', err.message);
    }

    // Clean up state regardless
    chatRunId = null;
    streamText = '';
    lastDeltaAt = null;
    setSendState(false);
    showStreaming(false);
  }

  function handleChatEvent(payload) {
    if (!payload) return;

    // Filter to current session — allow Telegram sessions through
    const isTelegramSession = payload.sessionKey && payload.sessionKey.startsWith('telegram:');
    if (payload.sessionKey && payload.sessionKey !== sessionKey && !isTelegramSession) return;

    const state = payload.state;

    // If this is for a different run than ours, handle gracefully
    if (chatRunId && payload.runId && payload.runId !== chatRunId) {
      if (state === 'final') {
        // Another run finished - might be from a different client, refresh history
        loadChatHistory();
      }
      return;
    }

    switch (state) {
      case 'delta': {
        lastDeltaAt = Date.now();
        const text = extractText(payload.message);
        if (text != null) {
          // The delta contains the full accumulated text so far
          if (text.length >= streamText.length) {
            streamText = text;
          }
          updateStreamingText(streamText);
        }
        break;
      }

      case 'final': {
        const finalText = extractText(payload.message);
        if (finalText) {
          addMessage('assistant', finalText);
        } else if (streamText) {
          addMessage('assistant', streamText);
        }
        chatRunId = null;
        streamText = '';
        lastDeltaAt = null;
        setSendState(false);
        showStreaming(false);
        addActivity('lifecycle', 'Complete', 'Agent finished');
        break;
      }

      case 'error': {
        const errMsg = payload.errorMessage || payload.error?.message || 'Unknown error';
        addMessage('assistant', 'Error: ' + errMsg, true);
        chatRunId = null;
        streamText = '';
        lastDeltaAt = null;
        setSendState(false);
        showStreaming(false);
        addActivity('error', 'Error', errMsg);
        break;
      }

      case 'aborted': {
        if (streamText) {
          addMessage('assistant', streamText + '\n\n[Aborted]');
        } else {
          addMessage('assistant', '[Run aborted]', true);
        }
        chatRunId = null;
        streamText = '';
        lastDeltaAt = null;
        setSendState(false);
        showStreaming(false);
        addActivity('info', 'Aborted', 'Run was aborted');
        break;
      }

      case 'started': {
        // Show inbound Telegram message in chat
        if (isTelegramSession) {
          const userText = payload.metadata?.text || payload.userMessage || '';
          if (userText) {
            const username = payload.metadata?.username || payload.metadata?.displayName || 'Telegram';
            addMessage('user', '[' + username + '] ' + userText);
          }
        }
        addActivity('lifecycle', 'Started', 'Agent processing...');
        break;
      }

      default:
        break;
    }
  }

  function handleAgentEvent(payload) {
    if (!payload) return;

    const stream = payload.stream;
    const data = payload.data || {};

    switch (stream) {
      case 'lifecycle':
        if (data.phase === 'start') {
          addActivity('lifecycle', 'Agent Start', data.agentId || '');
        } else if (data.phase === 'end') {
          const duration = data.durationMs ? ` (${(data.durationMs / 1000).toFixed(1)}s)` : '';
          addActivity('lifecycle', 'Agent End', duration);
        } else if (data.phase === 'error') {
          addActivity('error', 'Agent Error', data.error || 'Unknown');
        }
        break;

      case 'tool': {
        const toolName = data.name || data.tool || 'unknown';
        const phase = data.phase || '';
        if (phase === 'start') {
          addActivity('tool', `\u25B6 ${toolName}`, data.args ? truncate(JSON.stringify(data.args), 120) : '');
          incrementToolCounter(toolName);
        } else if (phase === 'result') {
          const status = data.isError ? '\u274C' : '\u2705';
          const resultPreview = data.result ? truncate(String(data.result), 80) : '';
          addActivity('tool', `${status} ${toolName}`, resultPreview);
        } else if (phase === 'update') {
          // Partial results - only show if significant
          if (data.partialResult && String(data.partialResult).length > 20) {
            addActivity('tool', `\u2026 ${toolName}`, truncate(String(data.partialResult), 60));
          }
        } else {
          addActivity('tool', 'Tool', toolName);
        }
        break;
      }

      case 'assistant':
        // Skip - shown in chat via delta events
        break;

      case 'error':
        addActivity('error', 'Error', data.error || data.message || JSON.stringify(data));
        break;

      default:
        if (stream) {
          addActivity('info', stream, typeof data === 'string' ? data : '');
        }
        break;
    }
  }

  // ─── Marcus Research Tab ──────────────────────────────────────────────────

  let marcusData = null;
  let marcusLastLoad = 0;
  let marcusPollTimer = null;
  let marcusSubTab = 'overview'; // overview, winners, inprogress, archived, alerts, directives, history
  let marcusWinnerExpanded = {};  // track which winner cards are expanded
  let marcusLeaderboardSort = { key: 'sharpe_ratio', asc: false };  // leaderboard sort state
  let marcusFunnelExpanded = {};  // track which funnel stages are expanded for drill-down
  let marcusCommandLog = [];
  const MARCUS_API_URL = 'http://localhost:3456/api/marcus';
  const MARCUS_REFRESH_MS = 30000;

  async function loadMarcus() {
    if (Date.now() - marcusLastLoad < 5000 && marcusData) return;
    try {
      var res = await fetch(MARCUS_API_URL, { signal: AbortSignal.timeout(10000) });
      if (!res.ok) throw new Error('HTTP ' + res.status);
      marcusData = await res.json();
      marcusLastLoad = Date.now();
      renderMarcus(marcusData);
    } catch (err) {
      if (!marcusData) {
        marcusContent.innerHTML = '<div class="activity-empty"><p>Cannot reach Marcus API. Make sure serve-web.js is running on port 3456.</p></div>';
      }
    }
  }

  async function loadWinnerEquity(winnerId) {
    try {
      var res = await fetch(MARCUS_API_URL.replace('/api/marcus', '/api/marcus/winner?id=' + winnerId), { signal: AbortSignal.timeout(10000) });
      if (!res.ok) return null;
      return await res.json();
    } catch (e) {
      return null;
    }
  }

  async function sendMarcusCommand(command, args) {
    try {
      var body = Object.assign({ command: command }, args || {});
      var res = await fetch(MARCUS_API_URL.replace('/api/marcus', '/api/marcus/command'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(5000),
      });
      var data = await res.json();
      marcusCommandLog.unshift({ time: new Date().toISOString(), command: command, result: data.ok ? 'OK' : 'FAIL' });
      if (marcusCommandLog.length > 50) marcusCommandLog = marcusCommandLog.slice(0, 50);
      return data;
    } catch (e) {
      marcusCommandLog.unshift({ time: new Date().toISOString(), command: command, result: 'ERROR: ' + e.message });
      return { ok: false, error: e.message };
    }
  }

  function renderMarcus(data) {
    if (!data) return;
    var html = '';

    // ── Daemon Status Bar ──
    var d = data.daemon || {};
    var statusClass = d.status === 'RUNNING' ? 'marcus-status-running' :
                      d.status === 'SLOW' ? 'marcus-status-slow' : 'marcus-status-offline';
    html += '<div class="marcus-status-bar ' + statusClass + '">';
    html += '<span class="marcus-status-dot"></span>';
    html += '<span class="marcus-status-label">' + escapeHtml(d.status || 'UNKNOWN') + '</span>';
    if (d.heartbeat) {
      var ageStr = d.age_seconds < 60 ? d.age_seconds + 's ago' :
                   d.age_seconds < 3600 ? Math.floor(d.age_seconds / 60) + 'm ago' :
                   Math.floor(d.age_seconds / 3600) + 'h ago';
      html += '<span class="marcus-heartbeat">Heartbeat: ' + ageStr + '</span>';
    }
    if (d.state) {
      html += '<span class="marcus-heartbeat">Cycle: ' + (d.state.total_cycles || 0) + '</span>';
    }
    html += '</div>';

    // ── Stats Tiles (always visible) ──
    var s = data.stats || {};
    html += '<div class="marcus-stats-grid">';
    html += marcusTile('Cycles', s.total_cycles || 0, 'info');
    html += marcusTile('Ideas', s.total_ideas || 0, 'info');
    html += marcusTile('S1 Pass', s.total_s1 || 0, (s.total_s1 || 0) > 0 ? 'good' : 'info');
    html += marcusTile('S2 Pass', s.total_s2 || 0, (s.total_s2 || 0) > 0 ? 'good' : 'warn');
    html += marcusTile('S5 Winners', s.total_winners || s.total_s5 || 0, (s.total_winners || s.total_s5 || 0) > 0 ? 'good' : 'warn');
    html += marcusTile('S1 Rate', (s.s1_pass_rate || 0) + '%', 'info');
    html += marcusTile('S2 Rate', (s.s2_pass_rate || 0) + '%', (s.s2_pass_rate || 0) > 0 ? 'good' : 'warn');
    html += marcusTile('Kill Rate', (s.kill_rate_pct || 100) + '%', 'info');
    html += marcusTile('Best Sharpe', (s.best_sharpe_ever || 0).toFixed(2), (s.best_sharpe_ever || 0) >= 0.3 ? 'good' : 'info');
    html += marcusTile('Avg Cycle', formatCycleDuration(s.avg_cycle_sec || 0), 'info');
    html += '</div>';

    // ── Sub-Tab Navigation ──
    var subTabs = [
      { id: 'overview', label: 'Overview' },
      { id: 'winners', label: 'Winners (' + (data.winners || []).length + ')' },
      { id: 'inprogress', label: 'In Progress (' + (data.in_progress || []).length + ')' },
      { id: 'archived', label: 'Failed (' + (data.graveyard || []).length + ')' },
      { id: 'alerts', label: 'Alerts' },
      { id: 'directives', label: 'Directives' },
      { id: 'history', label: 'History' },
    ];
    html += '<div class="marcus-subtabs">';
    for (var ti = 0; ti < subTabs.length; ti++) {
      var t = subTabs[ti];
      var activeClass = marcusSubTab === t.id ? ' marcus-subtab-active' : '';
      html += '<button class="marcus-subtab' + activeClass + '" data-marcus-subtab="' + t.id + '">' + t.label + '</button>';
    }
    html += '</div>';

    // ── Render active sub-tab content ──
    html += '<div class="marcus-subtab-content">';
    if (marcusSubTab === 'overview') {
      html += renderMarcusOverview(data);
    } else if (marcusSubTab === 'winners') {
      html += renderMarcusWinners(data);
    } else if (marcusSubTab === 'inprogress') {
      html += renderMarcusInProgress(data);
    } else if (marcusSubTab === 'archived') {
      html += renderMarcusArchived(data);
    } else if (marcusSubTab === 'alerts') {
      html += renderMarcusAlerts(data);
    } else if (marcusSubTab === 'directives') {
      html += renderMarcusDirectives(data);
    } else if (marcusSubTab === 'history') {
      html += renderMarcusHistory(data);
    }
    html += '</div>';

    marcusContent.innerHTML = html;

    // Bind sub-tab click events
    var subTabBtns = marcusContent.querySelectorAll('.marcus-subtab');
    subTabBtns.forEach(function (btn) {
      btn.addEventListener('click', function () {
        marcusSubTab = btn.dataset.marcusSubtab;
        renderMarcus(marcusData);
      });
    });

    // Bind winner expand/collapse events
    var winnerCards = marcusContent.querySelectorAll('.marcus-winner-header');
    winnerCards.forEach(function (hdr) {
      hdr.addEventListener('click', function () {
        var wid = hdr.dataset.winnerId;
        marcusWinnerExpanded[wid] = !marcusWinnerExpanded[wid];
        renderMarcus(marcusData);
        if (marcusWinnerExpanded[wid]) {
          loadAndRenderEquityCurve(wid);
        }
      });
    });

    // Bind leaderboard sort headers
    var sortHeaders = marcusContent.querySelectorAll('.marcus-th-sort');
    sortHeaders.forEach(function (th) {
      th.addEventListener('click', function () {
        var key = th.dataset.sortKey;
        if (marcusLeaderboardSort.key === key) {
          marcusLeaderboardSort.asc = !marcusLeaderboardSort.asc;
        } else {
          marcusLeaderboardSort.key = key;
          marcusLeaderboardSort.asc = false;
        }
        renderMarcus(marcusData);
      });
    });

    // Bind pipeline funnel drill-down
    var funnelStages = marcusContent.querySelectorAll('.marcus-funnel-stage[data-stage]');
    funnelStages.forEach(function (stage) {
      stage.addEventListener('click', function () {
        var stageName = stage.dataset.stage;
        marcusFunnelExpanded[stageName] = !marcusFunnelExpanded[stageName];
        if (marcusFunnelExpanded[stageName]) {
          loadFunnelDrillDown(stageName, stage);
        } else {
          var drilldown = stage.parentElement.querySelector('.marcus-drilldown-' + stageName);
          if (drilldown) drilldown.remove();
        }
      });
    });

    // Bind command buttons
    var cmdBtns = marcusContent.querySelectorAll('[data-marcus-cmd]');
    cmdBtns.forEach(function (btn) {
      btn.addEventListener('click', async function () {
        var cmd = btn.dataset.marcusCmd;
        btn.disabled = true;
        btn.textContent = 'Sending...';
        await sendMarcusCommand(cmd);
        btn.textContent = 'Sent!';
        setTimeout(function () { loadMarcus(); }, 1500);
      });
    });
  }

  // Load funnel drill-down data for a stage
  async function loadFunnelDrillDown(stageName, stageEl) {
    try {
      var res = await fetch('http://localhost:3456/api/marcus/stage?stage=' + stageName + '&limit=10');
      var data = await res.json();
      var strategies = data.strategies || [];

      var html = '<div class="marcus-drilldown marcus-drilldown-' + stageName + '">';
      if (strategies.length === 0) {
        html += '<div class="marcus-drilldown-empty">No strategies currently at this stage</div>';
      } else {
        html += '<table class="marcus-table marcus-table-mini"><thead><tr>';
        html += '<th>Strategy</th><th>Updated</th><th>Sharpe</th><th>Trades</th>';
        html += '</tr></thead><tbody>';
        for (var i = 0; i < strategies.length; i++) {
          var s = strategies[i];
          var m = s.metrics || {};
          html += '<tr>';
          html += '<td class="marcus-strat-name">' + escapeHtml(s.strategy_name || '') + '</td>';
          html += '<td>' + (s.updated_at || '').slice(0, 16) + '</td>';
          html += '<td>' + (m.sharpe_ratio || 0).toFixed(2) + '</td>';
          html += '<td>' + (m.total_trades || 0) + '</td>';
          html += '</tr>';
        }
        html += '</tbody></table>';
      }
      html += '</div>';

      // Insert after the stage element
      var existing = stageEl.parentElement.querySelector('.marcus-drilldown-' + stageName);
      if (existing) existing.remove();
      stageEl.insertAdjacentHTML('afterend', html);
    } catch (e) {
      console.error('Funnel drill-down error:', e);
    }
  }

  // ── Sub-tab: Overview ──
  function renderMarcusOverview(data) {
    var html = '';

    // Pipeline Funnel (cumulative)
    var p = data.pipeline_cumulative || {};
    var stages = ['CANDIDATE', 'TESTING', 'STAGE1_PASS', 'STAGE2_PASS', 'STAGE3_PASS', 'STAGE4_PASS', 'STAGE5_PASS', 'DEPLOYED'];
    var labels = ['Ideas', 'Tested', 'S1 Pass', 'S2 Gauntlet', 'S3 Regime', 'S4 Robust', 'S5 Comp', 'Winners'];
    html += '<div class="marcus-section">';
    html += '<div class="marcus-section-title">Pipeline Funnel (Cumulative Flow)</div>';
    html += '<div class="marcus-funnel">';
    for (var fi = 0; fi < stages.length; fi++) {
      var cnt = p[stages[fi]] || 0;
      var funnelClass = cnt > 0 ? ' marcus-funnel-active' : '';
      var expanded = marcusFunnelExpanded[stages[fi]] ? ' marcus-funnel-expanded' : '';
      html += '<div class="marcus-funnel-stage' + funnelClass + expanded + '" data-stage="' + stages[fi] + '" title="Click to see strategies at this stage">';
      html += '<div class="marcus-funnel-count">' + cnt + '</div>';
      html += '<div class="marcus-funnel-label">' + labels[fi] + '</div>';
      html += '</div>';
      if (fi < stages.length - 1) {
        html += '<div class="marcus-funnel-arrow">&rarr;</div>';
      }
    }
    html += '</div></div>';

    // Leaderboard (sortable)
    var lb = data.leaderboard || [];
    if (lb.length > 0) {
      // Sort leaderboard based on current sort state
      var sortKey = marcusLeaderboardSort.key || 'sharpe_ratio';
      var sortAsc = marcusLeaderboardSort.asc;
      lb = lb.slice().sort(function(a, b) {
        var va = a[sortKey] || 0;
        var vb = b[sortKey] || 0;
        return sortAsc ? va - vb : vb - va;
      });

      html += '<div class="marcus-section">';
      html += '<div class="marcus-section-title">Strategy Leaderboard (' + lb.length + ')</div>';
      html += '<table class="marcus-table marcus-table-sortable"><thead><tr>';

      var cols = [
        { key: 'strategy_name', label: 'Strategy', sortable: false },
        { key: 'sharpe_ratio', label: 'Sharpe' },
        { key: 'profit_factor', label: 'PF' },
        { key: 'max_drawdown_pct', label: 'DD%' },
        { key: 'total_trades', label: 'Trades' },
        { key: 'net_profit', label: 'Net P/L' },
        { key: 'win_rate', label: 'Win%' },
        { key: 'source', label: 'Source', sortable: false },
      ];
      for (var ci = 0; ci < cols.length; ci++) {
        var col = cols[ci];
        if (col.sortable === false) {
          html += '<th>' + col.label + '</th>';
        } else {
          var arrow = sortKey === col.key ? (sortAsc ? ' ▲' : ' ▼') : ' ⇅';
          html += '<th class="marcus-th-sort" data-sort-key="' + col.key + '">' + col.label + arrow + '</th>';
        }
      }
      html += '</tr></thead><tbody>';

      for (var li = 0; li < lb.length; li++) {
        var row = lb[li];
        var sharpeClass = (row.sharpe_ratio || 0) >= 0.3 ? 'marcus-val-good' :
                          (row.sharpe_ratio || 0) >= 0.15 ? 'marcus-val-ok' : '';
        var srcBadge = row.source === 'winner' ? '<span class="marcus-badge marcus-badge-winner">WINNER</span>' : '<span class="marcus-badge marcus-badge-bt">BT</span>';
        html += '<tr>';
        html += '<td class="marcus-strat-name">' + escapeHtml(row.strategy_name || 'Unknown') + '</td>';
        html += '<td class="' + sharpeClass + '">' + (row.sharpe_ratio || 0).toFixed(2) + '</td>';
        html += '<td>' + (row.profit_factor || 0).toFixed(2) + '</td>';
        html += '<td>' + ((row.max_drawdown_pct || 0) * 100).toFixed(1) + '%</td>';
        html += '<td>' + (row.total_trades || 0) + '</td>';
        html += '<td class="' + ((row.net_profit || 0) >= 0 ? 'marcus-val-good' : 'marcus-val-bad') + '">$' + Math.round(row.net_profit || 0).toLocaleString() + '</td>';
        html += '<td>' + (row.win_rate || 0).toFixed(1) + '%</td>';
        html += '<td>' + srcBadge + '</td>';
        html += '</tr>';
      }
      html += '</tbody></table></div>';
    }

    // Exploration Coverage
    var expl = data.exploration || [];
    if (expl.length > 0) {
      html += '<div class="marcus-section">';
      html += '<div class="marcus-section-title">Exploration Coverage</div>';
      html += '<div class="marcus-exploration-grid">';
      for (var ei = 0; ei < expl.length; ei++) {
        var e = expl[ei];
        var profRate = e.tested > 0 ? Math.round((e.profitable / e.tested) * 100) : 0;
        html += '<div class="marcus-exploration-card">';
        html += '<div class="marcus-exploration-name">' + escapeHtml(e.archetype) + '</div>';
        html += '<div class="marcus-exploration-stats">';
        html += '<span>' + e.tested + ' tested</span>';
        html += '<span class="marcus-val-good">' + e.profitable + ' profitable (' + profRate + '%)</span>';
        html += '<span>Best: ' + (e.best_sharpe || 0).toFixed(2) + '</span>';
        html += '</div></div>';
      }
      html += '</div></div>';
    }

    // Recent Cycles (compact)
    var cycles = data.cycles || [];
    if (cycles.length > 0) {
      html += '<div class="marcus-section">';
      html += '<div class="marcus-section-title">Recent Cycles (' + cycles.length + ')</div>';
      html += '<table class="marcus-table"><thead><tr>';
      html += '<th>#</th><th>Started</th><th>Dur</th><th>Ideas</th><th>S1</th><th>S2</th><th>S3</th><th>S4</th><th>S5</th><th>Best</th><th>Err</th>';
      html += '</tr></thead><tbody>';
      for (var ci = 0; ci < Math.min(cycles.length, 15); ci++) {
        var c = cycles[ci];
        html += '<tr>';
        html += '<td>' + (c.cycle_num || ci + 1) + '</td>';
        html += '<td class="marcus-timestamp">' + formatMarcusTime(c.started_at) + '</td>';
        html += '<td>' + formatCycleDuration(c.duration_seconds) + '</td>';
        html += '<td>' + (c.ideas_generated || 0) + '</td>';
        html += '<td>' + (c.stage1_passed || 0) + '</td>';
        html += '<td class="' + ((c.stage2_passed || 0) > 0 ? 'marcus-val-good' : '') + '">' + (c.stage2_passed || 0) + '</td>';
        html += '<td>' + (c.stage3_passed || 0) + '</td>';
        html += '<td>' + (c.stage4_passed || 0) + '</td>';
        html += '<td class="' + ((c.stage5_passed || 0) > 0 ? 'marcus-val-good' : '') + '">' + (c.stage5_passed || 0) + '</td>';
        html += '<td>' + (c.best_sharpe || 0).toFixed(2) + '</td>';
        html += '<td class="' + ((c.errors || 0) > 0 ? 'marcus-val-bad' : '') + '">' + (c.errors || 0) + '</td>';
        html += '</tr>';
      }
      html += '</tbody></table></div>';
    }

    return html;
  }

  // ── Sub-tab: Winners ──
  function renderMarcusWinners(data) {
    var winners = data.winners || [];
    var html = '';

    if (winners.length === 0) {
      html += '<div class="activity-empty"><p>No winning strategies yet. Strategies that pass all 5 stages appear here.</p></div>';
      return html;
    }

    html += '<div class="marcus-section-title">Winning Strategies (' + winners.length + ')</div>';

    for (var wi = 0; wi < winners.length; wi++) {
      var w = winners[wi];
      var isExpanded = marcusWinnerExpanded[w.id];
      var expandIcon = isExpanded ? '&#x25BC;' : '&#x25B6;';
      var winnerStatusClass = w.is_active ? 'marcus-winner-active' : 'marcus-winner-inactive';

      html += '<div class="marcus-winner-card ' + winnerStatusClass + '">';

      // Header (always visible, clickable)
      html += '<div class="marcus-winner-header" data-winner-id="' + w.id + '">';
      html += '<span class="marcus-winner-expand">' + expandIcon + '</span>';
      html += '<span class="marcus-winner-name">' + escapeHtml(w.strategy_name || 'Unknown') + '</span>';
      html += '<span class="marcus-winner-sharpe">' + (w.sharpe_ratio || 0).toFixed(2) + ' Sharpe</span>';
      html += '<span class="marcus-winner-profit ' + ((w.net_profit || 0) >= 0 ? 'marcus-val-good' : 'marcus-val-bad') + '">$' + Math.round(w.net_profit || 0).toLocaleString() + '</span>';
      html += '<span class="marcus-winner-status">' + (w.is_active ? 'ACTIVE' : 'INACTIVE') + '</span>';
      html += '</div>';

      // Expanded detail
      if (isExpanded) {
        html += '<div class="marcus-winner-detail">';

        // Equity Curve placeholder
        html += '<div class="marcus-winner-equity" id="equity-chart-' + w.id + '">';
        if (w.has_equity_curve) {
          html += '<div class="marcus-loading">Loading equity curve...</div>';
        } else {
          html += '<div class="activity-empty"><p>No equity curve data stored.</p></div>';
        }
        html += '</div>';

        // Stats grid
        html += '<div class="marcus-winner-stats">';
        html += marcusStatRow('Sharpe Ratio', (w.sharpe_ratio || 0).toFixed(4));
        html += marcusStatRow('Profit Factor', (w.profit_factor || 0).toFixed(4));
        html += marcusStatRow('Net Profit', '$' + Math.round(w.net_profit || 0).toLocaleString());
        html += marcusStatRow('Total Return', ((w.total_return || 0) * 100).toFixed(2) + '%');
        html += marcusStatRow('Max Drawdown', ((w.max_drawdown_pct || 0) * 100).toFixed(2) + '%');
        html += marcusStatRow('Win Rate', (w.win_rate || 0).toFixed(2) + '%');
        html += marcusStatRow('Total Trades', w.total_trades || 0);
        html += marcusStatRow('Win / Loss', (w.win_trades || 0) + ' / ' + (w.loss_trades || 0));
        html += marcusStatRow('Avg Trade P/L', '$' + (w.avg_trade_pnl || 0).toFixed(2));
        html += marcusStatRow('Quality Score', (w.quality_score || 0).toFixed(4));
        html += marcusStatRow('MC VaR 95%', w.monte_carlo_var95 ? '$' + Math.round(w.monte_carlo_var95).toLocaleString() : 'N/A');
        html += marcusStatRow('Symbol', w.symbol || 'NQ');
        html += marcusStatRow('Interval', w.interval || '5m');
        html += marcusStatRow('Data Range', (w.data_range_start || '?') + ' to ' + (w.data_range_end || '?'));
        html += marcusStatRow('Created', w.timestamp || '?');
        html += '</div>';

        // Parameters
        if (w.params_json) {
          html += '<div class="marcus-winner-params">';
          html += '<div class="marcus-detail-label">Parameters</div>';
          html += '<pre class="marcus-code">' + escapeHtml(formatJSON(w.params_json)) + '</pre>';
          html += '</div>';
        }

        // Source Code
        if (w.source_code) {
          html += '<div class="marcus-winner-source">';
          html += '<div class="marcus-detail-label">Source Code / Config Snapshot</div>';
          html += '<pre class="marcus-code">' + escapeHtml(w.source_code.substring(0, 2000)) + '</pre>';
          html += '</div>';
        }

        // Regime Analysis
        if (w.regime_analysis_json) {
          html += '<div class="marcus-winner-regime">';
          html += '<div class="marcus-detail-label">Regime Analysis</div>';
          html += '<pre class="marcus-code">' + escapeHtml(formatJSON(w.regime_analysis_json)) + '</pre>';
          html += '</div>';
        }

        // Notes / Tags
        if (w.notes || w.tags) {
          html += '<div class="marcus-winner-meta">';
          if (w.notes) html += '<div><strong>Notes:</strong> ' + escapeHtml(w.notes) + '</div>';
          if (w.tags) html += '<div><strong>Tags:</strong> ' + escapeHtml(w.tags) + '</div>';
          html += '</div>';
        }

        html += '</div>'; // end detail
      }

      html += '</div>'; // end card
    }

    return html;
  }

  async function loadAndRenderEquityCurve(winnerId) {
    var container = document.getElementById('equity-chart-' + winnerId);
    if (!container) return;

    // Show loading spinner
    container.innerHTML = '<div class="marcus-equity-loading"><div class="marcus-spinner"></div> Loading equity curve...</div>';

    var detail = await loadWinnerEquity(winnerId);
    if (!detail) {
      container.innerHTML = '<div class="marcus-equity-error">Failed to load winner data.</div>';
      return;
    }
    if (!detail.equity_curve) {
      container.innerHTML = '<div class="marcus-equity-empty">No equity curve data stored for this strategy.</div>';
      return;
    }

    // Parse and render equity curve with audit info
    try {
      var curveData = JSON.parse(detail.equity_curve);
      var values = [];
      if (Array.isArray(curveData)) {
        values = curveData.map(function (v) { return typeof v === 'number' ? v : (v.equity || v.value || 0); });
      } else if (typeof curveData === 'object') {
        values = Object.values(curveData);
      }

      if (values.length < 2) {
        container.innerHTML = '<div class="marcus-equity-empty">Insufficient equity data (' + values.length + ' points).</div>';
        return;
      }

      var svgHtml = renderEquitySVG(values, 600, 150);

      // ── Equity Curve Audit ──
      var auditHtml = '<div class="marcus-equity-audit">';
      auditHtml += '<span class="marcus-audit-item">📊 ' + values.length + ' data points</span>';

      // Check final PnL consistency
      var winner = detail.winner || {};
      var storedProfit = winner.net_profit || 0;
      var curveProfit = values[values.length - 1] - values[0];
      auditHtml += '<span class="marcus-audit-item">💰 Curve P/L: $' + Math.round(curveProfit).toLocaleString() + '</span>';

      if (storedProfit !== 0) {
        var pnlDiff = Math.abs(curveProfit - storedProfit) / Math.abs(storedProfit);
        if (pnlDiff > 0.05) {
          auditHtml += '<span class="marcus-audit-warn">⚠️ PnL mismatch: stored $' + Math.round(storedProfit).toLocaleString() + ' vs curve $' + Math.round(curveProfit).toLocaleString() + ' (' + (pnlDiff * 100).toFixed(1) + '% diff)</span>';
        } else {
          auditHtml += '<span class="marcus-audit-ok">✅ PnL verified (matches stored)</span>';
        }
      }

      // Show statistical verification notes if available
      if (winner.quality_notes) {
        auditHtml += '<span class="marcus-audit-item">🔬 ' + escapeHtml(winner.quality_notes) + '</span>';
      }

      auditHtml += '</div>';

      container.innerHTML = svgHtml + auditHtml;
    } catch (e) {
      container.innerHTML = '<div class="marcus-equity-error">Error parsing equity data: ' + escapeHtml(e.message) + '</div>';
    }
  }

  function renderEquitySVG(values, width, height) {
    var pad = 30;
    var w = width - pad * 2;
    var h = height - pad * 2;
    var min = Math.min.apply(null, values);
    var max = Math.max.apply(null, values);
    var range = max - min || 1;

    var points = values.map(function (v, i) {
      var x = pad + (i / (values.length - 1)) * w;
      var y = pad + h - ((v - min) / range) * h;
      return x.toFixed(1) + ',' + y.toFixed(1);
    });

    var svg = '<svg class="marcus-equity-svg" viewBox="0 0 ' + width + ' ' + height + '" preserveAspectRatio="none">';
    // Background grid
    for (var gi = 0; gi <= 4; gi++) {
      var gy = pad + (gi / 4) * h;
      var val = max - (gi / 4) * range;
      svg += '<line x1="' + pad + '" y1="' + gy + '" x2="' + (width - pad) + '" y2="' + gy + '" stroke="rgba(255,255,255,0.05)" />';
      svg += '<text x="' + (pad - 4) + '" y="' + (gy + 3) + '" fill="rgba(255,255,255,0.3)" font-size="8" text-anchor="end">$' + Math.round(val / 1000) + 'k</text>';
    }
    // Line
    svg += '<polyline fill="none" stroke="var(--success)" stroke-width="1.5" points="' + points.join(' ') + '" />';
    // Start/End labels
    svg += '<text x="' + pad + '" y="' + (height - 5) + '" fill="var(--text-dim)" font-size="8">Start</text>';
    svg += '<text x="' + (width - pad) + '" y="' + (height - 5) + '" fill="var(--text-dim)" font-size="8" text-anchor="end">$' + Math.round(values[values.length - 1]).toLocaleString() + '</text>';
    svg += '</svg>';
    return svg;
  }

  // ── Sub-tab: In Progress ──
  function renderMarcusInProgress(data) {
    var items = data.in_progress || [];
    var html = '';

    if (items.length === 0) {
      html += '<div class="activity-empty"><p>No strategies currently in the pipeline.</p></div>';
      return html;
    }

    html += '<div class="marcus-section-title">Active Pipeline (' + items.length + ')</div>';
    html += '<table class="marcus-table"><thead><tr>';
    html += '<th>Strategy</th><th>Stage</th><th>S1</th><th>S2</th><th>S3</th><th>S4</th><th>S5</th><th>Updated</th>';
    html += '</tr></thead><tbody>';
    for (var i = 0; i < items.length; i++) {
      var r = items[i];
      var stageClass = r.current_stage === 'STAGE5_PASS' ? 'marcus-val-good' :
                       r.current_stage === 'TESTING' ? 'marcus-val-ok' : '';
      html += '<tr>';
      html += '<td class="marcus-strat-name">' + escapeHtml(r.strategy_name || '?') + '</td>';
      html += '<td class="' + stageClass + '">' + escapeHtml(r.current_stage || '?') + '</td>';
      html += '<td>' + (r.s1_passed_at ? '&#x2714;' : '') + '</td>';
      html += '<td>' + (r.s2_passed_at ? '&#x2714;' : '') + '</td>';
      html += '<td>' + (r.s3_passed_at ? '&#x2714;' : '') + '</td>';
      html += '<td>' + (r.s4_passed_at ? '&#x2714;' : '') + '</td>';
      html += '<td>' + (r.s5_passed_at ? '&#x2714;' : '') + '</td>';
      html += '<td class="marcus-timestamp">' + formatMarcusTime(r.updated_at) + '</td>';
      html += '</tr>';
    }
    html += '</tbody></table>';
    return html;
  }

  // ── Sub-tab: Archived/Failed ──
  function renderMarcusArchived(data) {
    var html = '';

    // Graveyard (failed candidates with reason)
    var gv = data.graveyard || [];
    if (gv.length > 0) {
      html += '<div class="marcus-section">';
      html += '<div class="marcus-section-title">Failed Candidates (' + gv.length + ' recent)</div>';
      html += '<div class="marcus-graveyard-list">';
      for (var gi = 0; gi < Math.min(gv.length, 25); gi++) {
        var g = gv[gi];
        html += '<div class="marcus-graveyard-item">';
        html += '<span class="marcus-graveyard-name">' + escapeHtml(g.strategy_name || 'Unknown') + '</span>';
        html += '<span class="marcus-graveyard-stage">' + escapeHtml(g.killed_at_stage || '?') + '</span>';
        html += '<span class="marcus-graveyard-reason">' + escapeHtml(truncate(g.reason || '', 80)) + '</span>';
        html += '</div>';
      }
      html += '</div></div>';
    }

    // Archived lifecycle entries
    var arch = data.archived || [];
    if (arch.length > 0) {
      html += '<div class="marcus-section">';
      html += '<div class="marcus-section-title">Archived Strategies (' + arch.length + ')</div>';
      html += '<table class="marcus-table"><thead><tr>';
      html += '<th>Strategy</th><th>Stage</th><th>Strikes</th><th>Updated</th>';
      html += '</tr></thead><tbody>';
      for (var ai = 0; ai < arch.length; ai++) {
        var a = arch[ai];
        html += '<tr>';
        html += '<td class="marcus-strat-name">' + escapeHtml(a.strategy_name || '?') + '</td>';
        html += '<td>' + escapeHtml(a.current_stage || '?') + '</td>';
        html += '<td>' + (a.degradation_strikes || 0) + '</td>';
        html += '<td class="marcus-timestamp">' + formatMarcusTime(a.updated_at) + '</td>';
        html += '</tr>';
      }
      html += '</tbody></table></div>';
    }

    if (gv.length === 0 && arch.length === 0) {
      html += '<div class="activity-empty"><p>No archived or failed strategies yet.</p></div>';
    }

    return html;
  }

  // ── Sub-tab: Alerts ──
  function renderMarcusAlerts(data) {
    var html = '';
    var evts = data.events || [];

    // Filter for warnings and errors
    var alerts = evts.filter(function (ev) {
      return ev.severity === 'ERROR' || ev.severity === 'CRITICAL' || ev.severity === 'WARNING';
    });

    var infoEvts = evts.filter(function (ev) {
      return ev.severity === 'INFO';
    });

    if (alerts.length > 0) {
      html += '<div class="marcus-section">';
      html += '<div class="marcus-section-title">Alerts &amp; Warnings (' + alerts.length + ')</div>';
      html += '<div class="marcus-events-list">';
      for (var ai = 0; ai < alerts.length; ai++) {
        var ev = alerts[ai];
        var evClass = (ev.severity === 'ERROR' || ev.severity === 'CRITICAL') ? 'marcus-event-error' : 'marcus-event-warn';
        html += '<div class="marcus-event-item ' + evClass + '">';
        html += '<span class="marcus-event-time">' + formatMarcusTime(ev.timestamp) + '</span>';
        html += '<span class="marcus-event-component">' + escapeHtml(ev.component || '') + '</span>';
        html += '<span class="marcus-event-severity">' + escapeHtml(ev.severity || '') + '</span>';
        html += '<span class="marcus-event-msg">' + escapeHtml(truncate(ev.message || '', 120)) + '</span>';
        html += '</div>';
      }
      html += '</div></div>';
    } else {
      html += '<div class="marcus-section">';
      html += '<div class="marcus-section-title">Alerts</div>';
      html += '<div class="marcus-alert-ok">No warnings or errors. System is healthy.</div>';
      html += '</div>';
    }

    // System Events (all)
    if (infoEvts.length > 0) {
      html += '<div class="marcus-section">';
      html += '<div class="marcus-section-title">System Events (' + infoEvts.length + ')</div>';
      html += '<div class="marcus-events-list">';
      for (var ei = 0; ei < Math.min(infoEvts.length, 30); ei++) {
        var evi = infoEvts[ei];
        html += '<div class="marcus-event-item marcus-event-info">';
        html += '<span class="marcus-event-time">' + formatMarcusTime(evi.timestamp) + '</span>';
        html += '<span class="marcus-event-component">' + escapeHtml(evi.component || '') + '</span>';
        html += '<span class="marcus-event-msg">' + escapeHtml(truncate(evi.message || '', 120)) + '</span>';
        html += '</div>';
      }
      html += '</div></div>';
    }

    return html;
  }

  // ── Sub-tab: Directives / Controls ──
  function renderMarcusDirectives(data) {
    var html = '';
    var d = data.daemon || {};
    var state = d.state || {};

    html += '<div class="marcus-section">';
    html += '<div class="marcus-section-title">Marcus Controls</div>';
    html += '<div class="marcus-controls-grid">';
    html += '<button class="marcus-ctrl-btn marcus-ctrl-green" data-marcus-cmd="force_cycle">Force Cycle Now</button>';
    html += '<button class="marcus-ctrl-btn marcus-ctrl-yellow" data-marcus-cmd="pause">Pause Daemon</button>';
    html += '<button class="marcus-ctrl-btn marcus-ctrl-green" data-marcus-cmd="resume">Resume Daemon</button>';
    html += '<button class="marcus-ctrl-btn marcus-ctrl-red" data-marcus-cmd="clear_graveyard">Clear Graveyard</button>';
    html += '</div></div>';

    html += '<div class="marcus-section">';
    html += '<div class="marcus-section-title">Current Directives</div>';
    html += '<div class="marcus-directives-list">';
    html += marcusDirectiveRow('Status', d.status || 'UNKNOWN', d.status === 'RUNNING' ? 'good' : 'warn');
    html += marcusDirectiveRow('Total Cycles', state.total_cycles || 0, 'info');
    html += marcusDirectiveRow('Total Errors', state.total_errors || 0, (state.total_errors || 0) > 0 ? 'warn' : 'good');
    html += marcusDirectiveRow('Ideas/Cycle', state.ideas_per_cycle || 10, 'info');
    html += marcusDirectiveRow('Max Active Strats', state.max_active_strategies || 20, 'info');
    html += marcusDirectiveRow('Cycle Interval', (state.cycle_interval_minutes || 1) + ' min', 'info');
    html += marcusDirectiveRow('GPU Enabled', state.use_gpu !== false ? 'Yes' : 'No', state.use_gpu !== false ? 'good' : 'warn');
    html += marcusDirectiveRow('LLM Enabled', state.llm_enabled ? 'Yes' : 'No', 'info');
    html += marcusDirectiveRow('Paused', state.paused ? 'YES' : 'No', state.paused ? 'warn' : 'good');
    html += '</div></div>';

    // Command Log
    html += '<div class="marcus-section">';
    html += '<div class="marcus-section-title">Command Log</div>';
    if (marcusCommandLog.length === 0) {
      html += '<div class="activity-empty"><p>No commands sent yet. Use the controls above.</p></div>';
    } else {
      html += '<div class="marcus-events-list">';
      for (var ci = 0; ci < marcusCommandLog.length; ci++) {
        var cmd = marcusCommandLog[ci];
        var cmdClass = cmd.result === 'OK' ? 'marcus-event-info' : 'marcus-event-error';
        html += '<div class="marcus-event-item ' + cmdClass + '">';
        html += '<span class="marcus-event-time">' + formatMarcusTime(cmd.time) + '</span>';
        html += '<span class="marcus-event-component">CMD</span>';
        html += '<span class="marcus-event-msg">' + escapeHtml(cmd.command) + ' → ' + escapeHtml(cmd.result) + '</span>';
        html += '</div>';
      }
      html += '</div>';
    }
    html += '</div>';

    // Chat / Command Input
    html += '<div class="marcus-section">';
    html += '<div class="marcus-section-title">Send Command to Marcus</div>';
    html += '<div class="marcus-chat-input">';
    html += '<input type="text" class="marcus-cmd-input" id="marcusCmdInput" placeholder="Type a command (e.g., force_cycle, pause, resume, clear_graveyard)..." />';
    html += '<button class="marcus-ctrl-btn marcus-ctrl-blue" id="marcusCmdSend">Send</button>';
    html += '</div></div>';

    // After render, bind the chat input
    setTimeout(function () {
      var cmdInput = document.getElementById('marcusCmdInput');
      var cmdSend = document.getElementById('marcusCmdSend');
      if (cmdInput && cmdSend) {
        cmdSend.addEventListener('click', async function () {
          var cmd = cmdInput.value.trim();
          if (!cmd) return;
          cmdInput.value = '';
          await sendMarcusCommand(cmd);
          renderMarcus(marcusData);
        });
        cmdInput.addEventListener('keydown', function (e) {
          if (e.key === 'Enter') {
            cmdSend.click();
          }
        });
      }
    }, 50);

    return html;
  }

  // ── Sub-tab: History ──
  function renderMarcusHistory(data) {
    var html = '';
    var cycles = data.cycles || [];

    html += '<div class="marcus-section">';
    html += '<div class="marcus-section-title">Full Cycle History (' + cycles.length + ')</div>';

    if (cycles.length === 0) {
      html += '<div class="activity-empty"><p>No cycle history yet.</p></div>';
    } else {
      html += '<table class="marcus-table"><thead><tr>';
      html += '<th>#</th><th>Started</th><th>Finished</th><th>Duration</th><th>Ideas</th>';
      html += '<th>Tests</th><th>S1</th><th>S2</th><th>S3</th><th>S4</th><th>S5</th>';
      html += '<th>Rejected</th><th>Errors</th><th>Best Sharpe</th><th>Best Strategy</th><th>GPU</th>';
      html += '</tr></thead><tbody>';
      for (var i = 0; i < cycles.length; i++) {
        var c = cycles[i];
        var errClass = (c.errors || 0) > 0 ? ' class="marcus-val-bad"' : '';
        var s5Class = (c.stage5_passed || 0) > 0 ? ' class="marcus-val-good"' : '';
        html += '<tr>';
        html += '<td>' + (c.cycle_num || '') + '</td>';
        html += '<td class="marcus-timestamp">' + formatMarcusTime(c.started_at) + '</td>';
        html += '<td class="marcus-timestamp">' + formatMarcusTime(c.finished_at) + '</td>';
        html += '<td>' + formatCycleDuration(c.duration_seconds) + '</td>';
        html += '<td>' + (c.ideas_generated || 0) + '</td>';
        html += '<td>' + (c.backtests_run || 0) + '</td>';
        html += '<td>' + (c.stage1_passed || 0) + '</td>';
        html += '<td>' + (c.stage2_passed || 0) + '</td>';
        html += '<td>' + (c.stage3_passed || 0) + '</td>';
        html += '<td>' + (c.stage4_passed || 0) + '</td>';
        html += '<td' + s5Class + '>' + (c.stage5_passed || 0) + '</td>';
        html += '<td>' + (c.rejected || 0) + '</td>';
        html += '<td' + errClass + '>' + (c.errors || 0) + '</td>';
        html += '<td>' + (c.best_sharpe || 0).toFixed(2) + '</td>';
        html += '<td class="marcus-strat-name">' + escapeHtml(c.best_strategy_name || '') + '</td>';
        html += '<td>' + (c.gpu_used ? 'GPU' : 'CPU') + '</td>';
        html += '</tr>';
      }
      html += '</tbody></table>';
    }
    html += '</div>';

    return html;
  }

  // ── Marcus Helper Functions ──
  // KPI tooltip definitions
  var KPI_TOOLTIPS = {
    'Cycles': { def: 'Complete research cycles run', good: '>100', bad: '<10', action: 'Let Marcus run longer' },
    'Ideas': { def: 'Total strategy ideas generated', good: '>1000', bad: '<100', action: 'Check idea generator' },
    'S1 Pass': { def: 'Strategies passing basic profitability (net profit > 0, trades >= 200)', good: '>100', bad: '0', action: 'Check S1 thresholds' },
    'S2 Pass': { def: 'Strategies passing gauntlet stress test (1.5x costs, Sharpe CI positive, permutation test)', good: '>5', bad: '0', action: 'S2 thresholds may be too strict' },
    'S5 Winners': { def: 'Strategies passing all 5 gates with DSR > 95%', good: '>3', bad: '0', action: 'Pipeline is working if >0' },
    'S1 Rate': { def: '% of ideas that pass Stage 1', good: '10-30%', bad: '<5% or >50%', action: 'Adjust min_profit or min_trades' },
    'S2 Rate': { def: '% of S1 passers that pass Stage 2', good: '5-20%', bad: '0% or >40%', action: 'Adjust S2 Sharpe/PF thresholds' },
    'Kill Rate': { def: '% of candidates that fail all stages', good: '95-99%', bad: '<90% (too easy) or 100% (broken)', action: 'Adjust thresholds' },
    'Best Sharpe': { def: 'Highest Sharpe ratio from any cycle', good: '>=0.3', bad: '<0.1', action: 'Check strategy diversity' },
    'Avg Cycle': { def: 'Average time per research cycle', good: '<120s', bad: '>600s (stalling)', action: 'Check for timeouts' },
  };

  function marcusTile(label, value, type) {
    var tip = KPI_TOOLTIPS[label];
    var tooltipHtml = '';
    if (tip) {
      tooltipHtml = '<div class="marcus-kpi-tooltip">' +
        '<div class="marcus-kpi-tooltip-def">' + tip.def + '</div>' +
        '<div class="marcus-kpi-tooltip-range"><span class="marcus-val-good">Good: ' + tip.good + '</span> · ' +
        '<span class="marcus-val-bad">Bad: ' + tip.bad + '</span></div>' +
        '<div class="marcus-kpi-tooltip-action">💡 ' + tip.action + '</div>' +
        '</div>';
    }
    return '<div class="marcus-tile marcus-tile-' + type + ' marcus-tile-interactive">' +
      '<div class="marcus-tile-value">' + value + '</div>' +
      '<div class="marcus-tile-label">' + escapeHtml(label) + '</div>' +
      tooltipHtml +
      '</div>';
  }

  function marcusStatRow(label, value) {
    return '<div class="marcus-stat-row"><span class="marcus-stat-label">' + escapeHtml(label) + '</span><span class="marcus-stat-value">' + value + '</span></div>';
  }

  function marcusDirectiveRow(label, value, type) {
    var cls = type === 'good' ? 'marcus-val-good' : type === 'warn' ? 'marcus-val-bad' : '';
    return '<div class="marcus-directive-row"><span class="marcus-directive-label">' + escapeHtml(label) + '</span><span class="marcus-directive-value ' + cls + '">' + value + '</span></div>';
  }

  function formatCycleDuration(sec) {
    if (!sec || sec <= 0) return '0s';
    if (sec < 60) return Math.round(sec) + 's';
    if (sec < 3600) return Math.floor(sec / 60) + 'm ' + Math.round(sec % 60) + 's';
    return Math.floor(sec / 3600) + 'h ' + Math.floor((sec % 3600) / 60) + 'm';
  }

  function formatMarcusTime(ts) {
    if (!ts) return '--';
    try {
      var d = new Date(ts);
      return d.toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit' }) +
             ' ' + d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
    } catch (e) {
      return ts.slice(0, 16);
    }
  }

  function formatJSON(jsonStr) {
    try {
      return JSON.stringify(JSON.parse(jsonStr), null, 2);
    } catch (e) {
      return jsonStr;
    }
  }

  function startMarcusPolling() {
    if (marcusPollTimer) return;
    marcusPollTimer = setInterval(function () {
      if (activeTab === 'marcus') loadMarcus();
    }, MARCUS_REFRESH_MS);
  }

  function stopMarcusPolling() {
    if (marcusPollTimer) {
      clearInterval(marcusPollTimer);
      marcusPollTimer = null;
    }
  }

  // Start Marcus polling immediately (it's independent of gateway connection)
  startMarcusPolling();

  // ─── Stuck Run Detection ─────────────────────────────────────────────────

  let stuckCheckInterval = null;

  function startStuckDetection() {
    if (stuckCheckInterval) return;
    stuckCheckInterval = setInterval(() => {
      if (!chatRunId || !lastDeltaAt) return;

      const elapsed = Date.now() - lastDeltaAt;
      if (elapsed > STUCK_TIMEOUT_MS) {
        // Show stuck warning
        const existing = document.querySelector('.stuck-warning');
        if (!existing) {
          const warning = document.createElement('div');
          warning.className = 'stuck-warning';
          const span = document.createElement('span');
          span.textContent = 'Agent appears stuck (no response for ' + Math.round(elapsed / 1000) + 's)';
          warning.appendChild(span);
          const abortBtn = document.createElement('button');
          abortBtn.textContent = 'Abort Run';
          abortBtn.addEventListener('click', function () {
            warning.remove();
            abortCurrentRun();
          });
          warning.appendChild(abortBtn);
          chatStreamingEl.parentNode.insertBefore(warning, chatStreamingEl);
        }
      }
    }, 10000);
  }

  // Stuck abort is handled directly via addEventListener on the abort button

  // ─── Input Handling ──────────────────────────────────────────────────────

  function resizeInput() {
    chatInput.style.height = 'auto';
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
  }

  chatInput.addEventListener('input', resizeInput);

  chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  btnSend.addEventListener('click', () => sendMessage());
  btnAbort.addEventListener('click', () => abortCurrentRun());
  // Clear activity now handled by tab-contextual header action button

  btnReconnect.addEventListener('click', () => manualReconnect());
  btnStartGateway.addEventListener('click', async () => {
    splashOverlay.style.display = '';
    splashStatus.textContent = 'Starting gateway...';
    btnStartGateway.style.display = 'none';
    btnReconnect.style.display = 'none';

    const result = await window.openclaw.startGateway();
    splashOverlay.style.display = 'none';

    if (result.ok) {
      manualReconnect();
    } else {
      setConnectionStatus('disconnected', 'Gateway failed to start');
      btnStartGateway.style.display = '';
      addActivity('error', 'Gateway', 'Failed to start gateway');
    }
  });

  // ─── Gateway Status from Main Process ────────────────────────────────────

  window.openclaw.onGatewayStatusChange((status) => {
    // Only trigger reconnect if gateway comes up and we have no active connection/attempt
    if (status === 'connected' && !connected && !ws) {
      manualReconnect();
    }
    if (splashOverlay.style.display !== 'none') {
      splashStatus.textContent = status;
    }
  });

  // ─── Initialization ──────────────────────────────────────────────────────

  async function init() {
    config = await window.openclaw.getConfig();

    // Check if gateway is running
    const status = await window.openclaw.getGatewayStatus();

    if (!status.healthy) {
      // Show splash and auto-start
      splashOverlay.style.display = '';
      splashStatus.textContent = 'Starting OpenClaw Gateway...';

      const result = await window.openclaw.startGateway();
      splashOverlay.style.display = 'none';

      if (!result.ok) {
        setConnectionStatus('disconnected', 'Gateway not running');
        btnStartGateway.style.display = '';
        addActivity('error', 'Gateway', 'Could not start gateway automatically');
        return;
      }
    }

    // Connect WebSocket
    connectWebSocket();
    startStuckDetection();
  }

  // Start when DOM is ready
  init();
})();
