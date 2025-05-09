
# toggle_venv_path.ps1
param (
    [switch]$Remove
)

$venvPath = "C:\Users\lociuser\Codes\smartpath\smart-wsi-scanner\.venv\Scripts"

if ($Remove) {
    $env:PATH = ($env:PATH -split ";" | Where-Object { $_ -ne $venvPath }) -join ";"
    Write-Host "❌ Removed VENV path from PATH"
} else {
    if (-not ($env:PATH -split ";" | Where-Object { $_ -eq $venvPath })) {
        $env:PATH = "$venvPath;$env:PATH"
        Write-Host "✅ Added VENV path to PATH"
    } else {
        Write-Host "⚠️ VENV path is already in PATH"
    }
}

Write-Host "`n🔁 Current PATH:" 
Write-Host $env:PATH
