
Write-Host "--- Company OS Diagnostics ---" -ForegroundColor Cyan

# 1. Environment
Write-Host "[1] Node: $(node -v)"
Write-Host "[2] NPM: $(npm -v)"

# 2. Main Service
if (Test-Path "appliance.pid") {
    $appPid = Get-Content "appliance.pid"
    Write-Host "[3] Main Service PID: $appPid" -ForegroundColor Green
}
else {
    Write-Host "[3] Main Service: STOPPED" -ForegroundColor Red
}

# 3. Dashboard
try {
    $h = Invoke-RestMethod -Uri "http://localhost:3030/api/health" -Method Get -ErrorAction Stop
    Write-Host "[4] Dashboard: ONLINE ($($h.status))" -ForegroundColor Green
}
catch {
    Write-Host "[4] Dashboard: OFFLINE" -ForegroundColor Red
}

# 4. Ollama
try {
    $o = Invoke-RestMethod -Uri "http://localhost:11434/api/tags" -Method Get -ErrorAction Stop
    Write-Host "[5] Ollama: CONNECTED" -ForegroundColor Green
}
catch {
    Write-Host "[5] Ollama: DISCONNECTED" -ForegroundColor Red
}

# 5. DB
if (Test-Path "company_os.db") {
    Write-Host "[6] Database: FOUND" -ForegroundColor Green
}
else {
    Write-Host "[6] Database: MISSING" -ForegroundColor Red
}

Write-Host "--- Done ---"
