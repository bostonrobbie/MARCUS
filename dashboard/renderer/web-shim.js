// Web shim: replaces Electron's preload.js IPC with direct HTTP/fetch calls
// This allows the dashboard to run in a regular browser (Chrome) instead of Electron
(function() {
  const GATEWAY_PORT = 18789;
  const WS_URL = `ws://localhost:${GATEWAY_PORT}`;
  const HEALTH_URL = `http://localhost:${GATEWAY_PORT}/health`;

  // Read token from URL param or use empty
  const urlParams = new URLSearchParams(window.location.search);
  const token = urlParams.get('token') || '';

  window.openclaw = {
    getConfig: async () => ({
      token,
      port: GATEWAY_PORT,
      wsUrl: WS_URL,
      healthUrl: HEALTH_URL,
    }),
    getGatewayStatus: async () => {
      try {
        const res = await fetch(HEALTH_URL, { signal: AbortSignal.timeout(5000) });
        const data = await res.json();
        return { healthy: data.status === 'ok' || res.ok };
      } catch {
        return { healthy: false };
      }
    },
    restartGateway: async () => ({ ok: false, error: 'Not available in web mode' }),
    startGateway: async () => {
      // In web mode, just check if gateway is already running
      try {
        const res = await fetch(HEALTH_URL, { signal: AbortSignal.timeout(5000) });
        return { ok: res.ok };
      } catch {
        return { ok: false };
      }
    },
    openExternal: (url) => window.open(url, '_blank'),
    runBaselineTest: async () => ({ ok: false, message: 'Not available in web mode' }),
    onGatewayStatusChange: (callback) => {
      // Poll health every 10s
      setInterval(async () => {
        try {
          const res = await fetch(HEALTH_URL, { signal: AbortSignal.timeout(3000) });
          callback(res.ok ? 'connected' : 'disconnected');
        } catch {
          callback('disconnected');
        }
      }, 10000);
    },
  };
})();
