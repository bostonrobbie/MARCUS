
Add-Type -AssemblyName System.Drawing

$source = "app_icon.png"
$dest = "app_icon.ico"

$bmp = [System.Drawing.Bitmap]::FromFile((Resolve-Path $source))
$icon = [System.Drawing.Icon]::FromHandle($bmp.GetHicon())

$fs = [System.IO.File]::OpenWrite((Join-Path (Get-Location) $dest))
$icon.Save($fs)
$fs.Close()
$bmp.Dispose()
$icon.Dispose()

Write-Host "Created $dest"
