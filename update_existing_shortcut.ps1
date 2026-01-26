
$WshShell = New-Object -comObject WScript.Shell
$ShortcutPath = "C:\Users\User\Desktop\Zero Human Command Center.lnk"
$IconPath = "C:\Users\User\Documents\AI\local-manus-agent-workspace\ai-company-os\app_icon.ico"

if (Test-Path $ShortcutPath) {
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    $Shortcut.IconLocation = $IconPath
    $Shortcut.Save()
    Write-Host "Updated icon for $ShortcutPath"
}
else {
    Write-Host "Shortcut not found at $ShortcutPath"
}
