
$WshShell = New-Object -comObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "Synapse AI Control Room.lnk"
$Target = "C:\Users\User\Desktop\Zero_Human_HQ\LAUNCH_COMMAND_CENTER.bat"
$IconPath = "C:\Users\User\Documents\AI\local-manus-agent-workspace\ai-company-os\app_icon.ico"

$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Target
$Shortcut.IconLocation = $IconPath
$Shortcut.Save()

Write-Host "Created shortcut: $ShortcutPath"
