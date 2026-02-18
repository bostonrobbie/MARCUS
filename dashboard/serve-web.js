// Simple HTTP server to serve the dashboard in a regular browser
const http = require('http');
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');

const PORT = 3456;
const RENDERER_DIR = path.join(__dirname, 'renderer');

// Marcus paths
const MARCUS_DB = path.join('C:', 'Users', 'User', 'Desktop', 'Zero_Human_HQ', 'Quant_Lab', 'StrategyPipeline', 'src', 'backtesting', 'marcus_registry.db');
const MARCUS_HEARTBEAT = path.join('C:', 'Users', 'User', 'Desktop', 'Zero_Human_HQ', 'Quant_Lab', 'Marcus_Research', 'logs', 'heartbeat.txt');
const MARCUS_HEALTH_LOG = path.join('C:', 'Users', 'User', 'Desktop', 'Zero_Human_HQ', 'Quant_Lab', 'Marcus_Research', 'logs', 'system_health.jsonl');
const MARCUS_STATE_FILE = path.join('C:', 'Users', 'User', 'Desktop', 'Zero_Human_HQ', 'Quant_Lab', 'Marcus_Research', 'marcus_daemon_state.json');
const MARCUS_CONFIG_FILE = path.join('C:', 'Users', 'User', 'Desktop', 'Zero_Human_HQ', 'Quant_Lab', 'StrategyPipeline', 'src', 'backtesting', 'marcus_config.py');
const MARCUS_QUERY_SCRIPT = path.join(__dirname, 'marcus-query.py');

const MIME_TYPES = {
  '.html': 'text/html',
  '.css': 'text/css',
  '.js': 'application/javascript',
  '.json': 'application/json',
  '.png': 'image/png',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
};

// ─── Marcus Data API ─────────────────────────────────────────────────────

function runMarcusQuery(args, callback) {
  const allArgs = [MARCUS_QUERY_SCRIPT, MARCUS_DB].concat(args || []);
  execFile('python', allArgs, { timeout: 15000, windowsHide: true }, (err, stdout) => {
    if (!err && stdout) {
      try {
        callback(null, JSON.parse(stdout.trim()));
      } catch (e) {
        callback(e, null);
      }
    } else {
      callback(err, null);
    }
  });
}

function getDaemonInfo() {
  const info = { heartbeat: null, age_seconds: -1, status: 'OFFLINE', state: null };

  // Heartbeat / daemon status
  try {
    const hb = fs.readFileSync(MARCUS_HEARTBEAT, 'utf8').trim();
    const hbTime = new Date(hb);
    const ageSec = (Date.now() - hbTime.getTime()) / 1000;
    info.heartbeat = hb;
    info.age_seconds = Math.round(ageSec);
    info.status = ageSec < 600 ? 'RUNNING' : ageSec < 3600 ? 'SLOW' : 'STALE';
  } catch (e) { /* file not found = OFFLINE */ }

  // Daemon state file
  try {
    const state = JSON.parse(fs.readFileSync(MARCUS_STATE_FILE, 'utf8'));
    info.state = state;
  } catch (e) { /* ignore */ }

  return info;
}

function getRecentEvents(limit) {
  limit = limit || 50;
  try {
    const raw = fs.readFileSync(MARCUS_HEALTH_LOG, 'utf8');
    const lines = raw.split('\n').filter(Boolean);
    const recent = lines.slice(-limit);
    return recent.map(l => { try { return JSON.parse(l); } catch { return null; } }).filter(Boolean).reverse();
  } catch (e) { return []; }
}

function getMarcusData(callback) {
  const daemon = getDaemonInfo();
  const events = getRecentEvents(50);

  runMarcusQuery(['dashboard'], (err, dbData) => {
    // Merge: dbData first, then override daemon/events from Node.js (more reliable)
    const data = Object.assign({}, dbData || {}, { daemon: daemon, events: events });
    callback(data);
  });
}

function sendJson(res, data) {
  res.writeHead(200, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
  res.end(JSON.stringify(data));
}

function sendError(res, code, msg) {
  res.writeHead(code, { 'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*' });
  res.end(JSON.stringify({ error: msg }));
}

// Parse URL query params
function parseQuery(url) {
  const idx = url.indexOf('?');
  if (idx < 0) return {};
  const qs = url.slice(idx + 1);
  const params = {};
  qs.split('&').forEach(p => {
    const [k, v] = p.split('=');
    params[decodeURIComponent(k)] = decodeURIComponent(v || '');
  });
  return params;
}

// Read POST body
function readBody(req, callback) {
  let body = '';
  req.on('data', chunk => { body += chunk; });
  req.on('end', () => {
    try {
      callback(null, JSON.parse(body));
    } catch (e) {
      callback(e, null);
    }
  });
}

const server = http.createServer((req, res) => {
  let urlPath = req.url.split('?')[0];
  const params = parseQuery(req.url);

  // CORS preflight
  if (req.method === 'OPTIONS') {
    res.writeHead(204, {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type',
    });
    res.end();
    return;
  }

  // ─── Marcus API endpoints ────────────────────────────────────
  if (urlPath === '/api/marcus') {
    getMarcusData((data) => sendJson(res, data));
    return;
  }

  if (urlPath === '/api/marcus/winner') {
    const id = params.id || '1';
    runMarcusQuery(['winner', id], (err, data) => {
      if (err) return sendError(res, 500, 'Query failed');
      sendJson(res, data);
    });
    return;
  }

  if (urlPath === '/api/marcus/history') {
    const limit = params.limit || '200';
    runMarcusQuery(['history', limit], (err, data) => {
      if (err) return sendError(res, 500, 'Query failed');
      sendJson(res, data);
    });
    return;
  }

  if (urlPath === '/api/marcus/events') {
    const limit = parseInt(params.limit || '100');
    sendJson(res, { events: getRecentEvents(limit) });
    return;
  }

  if (urlPath === '/api/marcus/daemon') {
    sendJson(res, getDaemonInfo());
    return;
  }

  if (urlPath === '/api/marcus/config') {
    // Read the current config from the state file
    const info = getDaemonInfo();
    sendJson(res, { config: info.state || {}, status: info.status });
    return;
  }

  if (urlPath === '/api/marcus/stage') {
    const stage = params.stage || 'STAGE1_PASS';
    const limit = params.limit || '20';
    runMarcusQuery(['stage', stage, limit], (err, data) => {
      if (err) return sendError(res, 500, 'Stage query failed');
      sendJson(res, data);
    });
    return;
  }

  if (urlPath === '/api/marcus/command' && req.method === 'POST') {
    // Handle commands: pause, resume, force_cycle, etc.
    readBody(req, (err, body) => {
      if (err || !body) return sendError(res, 400, 'Invalid JSON body');

      const cmd = body.command;
      const logEntry = {
        timestamp: new Date().toISOString(),
        component: 'Dashboard',
        type: 'COMMAND',
        severity: 'INFO',
        message: 'User command: ' + cmd,
        metadata: body
      };

      // Write command to the health log for Marcus to see
      try {
        fs.appendFileSync(MARCUS_HEALTH_LOG, JSON.stringify(logEntry) + '\n');
      } catch (e) { /* ignore */ }

      // Handle specific commands
      if (cmd === 'get_directives') {
        // Read directives from daemon state
        const info = getDaemonInfo();
        const state = info.state || {};
        sendJson(res, {
          ok: true,
          directives: {
            max_active_strategies: state.max_active_strategies || 20,
            ideas_per_cycle: state.ideas_per_cycle || 10,
            cycle_interval_minutes: state.cycle_interval_minutes || 1,
            llm_enabled: state.llm_enabled || false,
            use_gpu: state.use_gpu !== false,
            paused: state.paused || false,
            total_cycles: state.total_cycles || 0,
            total_errors: state.total_errors || 0,
            status: info.status,
          }
        });
      } else {
        // Log the command - Marcus daemon will pick it up from the health log
        sendJson(res, { ok: true, message: 'Command logged: ' + cmd });
      }
    });
    return;
  }

  // ─── Static file serving ─────────────────────────────────────
  if (urlPath === '/' || urlPath === '/index.html') {
    urlPath = '/index-web.html';
  }

  const filePath = path.join(RENDERER_DIR, urlPath);

  // Security: don't serve files outside renderer dir
  if (!filePath.startsWith(RENDERER_DIR)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  const ext = path.extname(filePath);
  const contentType = MIME_TYPES[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, data) => {
    if (err) {
      res.writeHead(404);
      res.end('Not found: ' + urlPath);
      return;
    }
    res.writeHead(200, { 'Content-Type': contentType });
    res.end(data);
  });
});

server.listen(PORT, () => {
  console.log(`Dashboard web server running at http://localhost:${PORT}`);
  console.log(`Open in Chrome: http://localhost:${PORT}/?token=YOUR_TOKEN`);
});
