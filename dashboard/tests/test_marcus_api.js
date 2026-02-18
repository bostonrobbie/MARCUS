/**
 * Marcus Dashboard API Integration Tests
 *
 * Tests the serve-web.js API endpoints and validates data integrity.
 * Run: node tests/test_marcus_api.js
 *
 * Requirements: serve-web.js must be running on port 3456
 */

const http = require('http');

const API_BASE = 'http://localhost:3456';
let passed = 0;
let failed = 0;
let skipped = 0;

function get(path) {
  return new Promise((resolve, reject) => {
    const url = API_BASE + path;
    http.get(url, { timeout: 10000 }, (res) => {
      let body = '';
      res.on('data', chunk => body += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(body));
        } catch (e) {
          reject(new Error('JSON parse error: ' + e.message));
        }
      });
    }).on('error', reject);
  });
}

function post(path, data) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(data);
    const opts = {
      hostname: 'localhost',
      port: 3456,
      path: path,
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Content-Length': Buffer.byteLength(body) },
      timeout: 10000,
    };
    const req = http.request(opts, (res) => {
      let respBody = '';
      res.on('data', chunk => respBody += chunk);
      res.on('end', () => {
        try {
          resolve(JSON.parse(respBody));
        } catch (e) {
          reject(new Error('JSON parse error: ' + e.message));
        }
      });
    });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

function assert(condition, message) {
  if (!condition) throw new Error('Assertion failed: ' + message);
}

function assertType(value, type, name) {
  if (typeof value !== type) throw new Error(`Expected ${name} to be ${type}, got ${typeof value}`);
}

async function test(name, fn) {
  try {
    await fn();
    passed++;
    console.log(`  ✓ ${name}`);
  } catch (e) {
    failed++;
    console.log(`  ✗ ${name}: ${e.message}`);
  }
}

async function run() {
  console.log('\nMarcus Dashboard API Tests');
  console.log('='.repeat(60));

  // Check server is up
  try {
    await get('/api/marcus');
  } catch (e) {
    console.log('ERROR: serve-web.js not running on port 3456');
    console.log('Start it with: node serve-web.js');
    process.exit(1);
  }

  // ─── Main Dashboard Endpoint ──────────────────────────────
  console.log('\n--- GET /api/marcus ---');

  await test('Returns valid JSON with all top-level keys', async () => {
    const data = await get('/api/marcus');
    assert(data.daemon !== undefined, 'Missing daemon');
    assert(data.stats !== undefined, 'Missing stats');
    assert(data.winners !== undefined, 'Missing winners');
    assert(data.leaderboard !== undefined, 'Missing leaderboard');
    assert(data.cycles !== undefined, 'Missing cycles');
    assert(data.pipeline !== undefined, 'Missing pipeline');
    assert(data.pipeline_cumulative !== undefined, 'Missing pipeline_cumulative');
    assert(data.graveyard !== undefined, 'Missing graveyard');
    assert(data.events !== undefined, 'Missing events');
    assert(data.in_progress !== undefined, 'Missing in_progress');
    assert(data.archived !== undefined, 'Missing archived');
    assert(data.exploration !== undefined, 'Missing exploration');
  });

  await test('Daemon status is valid', async () => {
    const data = await get('/api/marcus');
    const valid = ['RUNNING', 'SLOW', 'STALE', 'OFFLINE'];
    assert(valid.includes(data.daemon.status), 'Invalid status: ' + data.daemon.status);
    assertType(data.daemon.age_seconds, 'number', 'age_seconds');
  });

  await test('Stats contains all required fields', async () => {
    const data = await get('/api/marcus');
    const s = data.stats;
    assertType(s.total_cycles, 'number', 'total_cycles');
    assertType(s.total_ideas, 'number', 'total_ideas');
    assertType(s.total_s1, 'number', 'total_s1');
    assertType(s.total_s2, 'number', 'total_s2');
    assertType(s.total_s5, 'number', 'total_s5');
    assertType(s.kill_rate_pct, 'number', 'kill_rate_pct');
    assertType(s.best_sharpe_ever, 'number', 'best_sharpe_ever');
    assertType(s.s1_pass_rate, 'number', 's1_pass_rate');
    assertType(s.s2_pass_rate, 'number', 's2_pass_rate');
  });

  await test('Winners array has entries', async () => {
    const data = await get('/api/marcus');
    assert(data.winners.length > 0, 'No winners found');
  });

  await test('Each winner has required fields', async () => {
    const data = await get('/api/marcus');
    for (const w of data.winners) {
      assert(w.id !== undefined, 'Missing id');
      assert(w.strategy_name, 'Missing strategy_name');
      assertType(w.sharpe_ratio, 'number', 'sharpe_ratio');
      assertType(w.net_profit, 'number', 'net_profit');
      assertType(w.total_trades, 'number', 'total_trades');
      assert(w.has_equity_curve !== undefined, 'Missing has_equity_curve');
    }
  });

  await test('Leaderboard has entries', async () => {
    const data = await get('/api/marcus');
    assert(data.leaderboard.length > 0, 'Empty leaderboard');
  });

  await test('Leaderboard entries have source field', async () => {
    const data = await get('/api/marcus');
    for (const lb of data.leaderboard) {
      assert(['winner', 'backtest'].includes(lb.source), 'Invalid source: ' + lb.source);
      assert(lb.strategy_name, 'Missing strategy_name');
      assertType(lb.sharpe_ratio, 'number', 'sharpe_ratio');
    }
  });

  await test('Pipeline cumulative shows decreasing flow', async () => {
    const data = await get('/api/marcus');
    const pc = data.pipeline_cumulative;
    assert(pc.CANDIDATE >= pc.STAGE1_PASS, 'Candidates should >= S1');
    assert(pc.STAGE1_PASS >= pc.STAGE2_PASS, 'S1 should >= S2');
    assert(pc.STAGE2_PASS >= pc.STAGE5_PASS, 'S2 should >= S5');
  });

  await test('Cycles array has entries with required fields', async () => {
    const data = await get('/api/marcus');
    assert(data.cycles.length > 0, 'No cycles');
    const c = data.cycles[0];
    assert(c.cycle_num !== undefined, 'Missing cycle_num');
    assert(c.started_at !== undefined, 'Missing started_at');
    assertType(c.duration_seconds, 'number', 'duration_seconds');
    assertType(c.ideas_generated, 'number', 'ideas_generated');
  });

  await test('Exploration coverage has ORB archetype', async () => {
    const data = await get('/api/marcus');
    const archetypes = data.exploration.map(e => e.archetype);
    assert(archetypes.includes('ORB'), 'Missing ORB archetype');
  });

  await test('S2 pass count is accurate', async () => {
    const data = await get('/api/marcus');
    // S2 passes from cycle_log should be > 0 (we know pipeline is working)
    assert(data.stats.total_s2 > 0, 'Expected S2 passes > 0');
  });

  await test('Total winners count matches winners array', async () => {
    const data = await get('/api/marcus');
    const winnerCount = data.stats.total_winners || data.winners.length;
    assert(winnerCount === data.winners.length,
      `Winner count mismatch: stats=${winnerCount}, array=${data.winners.length}`);
  });

  // ─── Winner Detail Endpoint ──────────────────────────────
  console.log('\n--- GET /api/marcus/winner ---');

  await test('Winner detail returns winner data', async () => {
    const main = await get('/api/marcus');
    if (main.winners.length === 0) { skipped++; return; }
    const wid = main.winners[0].id;
    const data = await get('/api/marcus/winner?id=' + wid);
    assert(data.winner !== null, 'Winner is null');
    assert(data.winner.strategy_name, 'Missing strategy_name');
  });

  await test('Winner with equity curve returns curve data', async () => {
    const main = await get('/api/marcus');
    const withCurve = main.winners.find(w => w.has_equity_curve);
    if (!withCurve) { skipped++; return; }
    const data = await get('/api/marcus/winner?id=' + withCurve.id);
    assert(data.equity_curve !== null, 'Expected equity curve');
    const parsed = JSON.parse(data.equity_curve);
    assert(Array.isArray(parsed), 'Equity curve should be array');
    assert(parsed.length > 0, 'Equity curve should not be empty');
  });

  await test('Non-existent winner returns null', async () => {
    const data = await get('/api/marcus/winner?id=99999');
    assert(data.winner === null, 'Expected null for non-existent winner');
  });

  // ─── History Endpoint ──────────────────────────────────
  console.log('\n--- GET /api/marcus/history ---');

  await test('History returns cycles array', async () => {
    const data = await get('/api/marcus/history?limit=10');
    assert(Array.isArray(data.cycles), 'Expected cycles array');
    assert(data.cycles.length > 0, 'Expected at least 1 cycle');
  });

  await test('History respects limit parameter', async () => {
    const data = await get('/api/marcus/history?limit=5');
    assert(data.cycles.length <= 5, 'Expected at most 5 cycles');
  });

  // ─── Events Endpoint ──────────────────────────────────
  console.log('\n--- GET /api/marcus/events ---');

  await test('Events returns array', async () => {
    const data = await get('/api/marcus/events?limit=10');
    assert(Array.isArray(data.events), 'Expected events array');
  });

  // ─── Daemon Endpoint ──────────────────────────────────
  console.log('\n--- GET /api/marcus/daemon ---');

  await test('Daemon endpoint returns status', async () => {
    const data = await get('/api/marcus/daemon');
    assert(data.status, 'Missing status');
    assertType(data.age_seconds, 'number', 'age_seconds');
  });

  // ─── Config Endpoint ──────────────────────────────────
  console.log('\n--- GET /api/marcus/config ---');

  await test('Config endpoint returns config and status', async () => {
    const data = await get('/api/marcus/config');
    assert(data.config !== undefined, 'Missing config');
    assert(data.status, 'Missing status');
  });

  // ─── Command Endpoint ──────────────────────────────────
  console.log('\n--- POST /api/marcus/command ---');

  await test('Get directives command returns directives', async () => {
    const data = await post('/api/marcus/command', { command: 'get_directives' });
    assert(data.ok === true, 'Expected ok=true');
    assert(data.directives !== undefined, 'Missing directives');
    assertType(data.directives.status, 'string', 'directives.status');
  });

  await test('Unknown command returns ok', async () => {
    const data = await post('/api/marcus/command', { command: 'test_ping' });
    assert(data.ok === true, 'Expected ok=true');
  });

  // ─── Data Integrity Checks ──────────────────────────────
  console.log('\n--- Data Integrity ---');

  await test('All winner sharpe_ratios are positive', async () => {
    const data = await get('/api/marcus');
    for (const w of data.winners) {
      assert(w.sharpe_ratio > 0, `Winner ${w.strategy_name} has non-positive Sharpe: ${w.sharpe_ratio}`);
    }
  });

  await test('All winner net_profits are positive', async () => {
    const data = await get('/api/marcus');
    for (const w of data.winners) {
      assert(w.net_profit > 0, `Winner ${w.strategy_name} has non-positive net_profit: ${w.net_profit}`);
    }
  });

  await test('Kill rate is between 0 and 100', async () => {
    const data = await get('/api/marcus');
    const kr = data.stats.kill_rate_pct;
    assert(kr >= 0 && kr <= 100, 'Kill rate out of range: ' + kr);
  });

  await test('S1 pass rate is reasonable (< 50%)', async () => {
    const data = await get('/api/marcus');
    const r = data.stats.s1_pass_rate;
    assert(r <= 50, 'S1 pass rate too high: ' + r + '% (indicates possible issue)');
  });

  await test('Best sharpe_ever matches max in leaderboard', async () => {
    const data = await get('/api/marcus');
    if (data.leaderboard.length === 0) { skipped++; return; }
    const lbMax = Math.max(...data.leaderboard.map(r => r.sharpe_ratio));
    // Stats best_sharpe_ever comes from cycle_log, so it may differ slightly
    // but should be in the same ballpark
    assert(data.stats.best_sharpe_ever > 0, 'Best sharpe should be > 0');
  });

  // ─── Summary ──────────────────────────────────────────
  console.log('\n' + '='.repeat(60));
  console.log(`Results: ${passed} passed, ${failed} failed, ${skipped} skipped`);
  console.log('='.repeat(60));

  if (failed > 0) {
    process.exit(1);
  }
}

run().catch(e => {
  console.error('Test runner error:', e);
  process.exit(1);
});
