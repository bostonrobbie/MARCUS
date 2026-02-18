
# setup_appliance.ps1
# This script registers the Company OS Daemon in Windows Task Scheduler

$TaskName = "CompanyOS"
$ProjectDir = Get-Location
$NodePath = "node" # Assumes node is in PATH
$ScriptPath = "$ProjectDir\dist\index.js"
$Arguments = "$ScriptPath schedule:daemon"

# Check if task already exists
$existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Task '$TaskName' already exists. Unregistering first..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Write-Host "Registering Company OS Task Scheduler Entry..."
Write-Host "Working Directory: $ProjectDir"

$action = New-ScheduledTaskAction -Execute $NodePath -Argument $Arguments -WorkingDirectory $ProjectDir
$trigger = New-ScheduledTaskTrigger -AtStartup
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -User "INTERACTIVE" -RunLevel Highest

Write-Host "Success! Company OS will now start automatically on reboot."
Write-Host "To start it now manually: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To stop it: Stop-ScheduledTask -TaskName '$TaskName'"
