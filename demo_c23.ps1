## 2.3 Add a one-command demo script (Windows)

```powershell
# demo_c23.ps1
# One-command prototype demo:
# - runs C23 snapshots fast
# - shows snapshot directory
# - stops after a few seconds

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "=== Dimensional Core v0.1 Demo (C20/C22 + C23) ==="
Write-Host ""

# Run for ~8 seconds, then stop (CTRL+C-like)
$proc = Start-Process -PassThru -NoNewWindow `
python `
-ArgumentList "-m","dimensional_core.run_demo","--resume","--c23-demo","--c23-every","0.5","--c23-target","storage"

Start-Sleep -Seconds 8
try { $proc.Kill() } catch {}

Write-Host ""
Write-Host "Snapshots written to:"
Write-Host "  .\dimensional_core\state\dimensions\storage\"
Write-Host ""

Get-ChildItem .\dimensional_core\state\dimensions\storage\ -ErrorAction SilentlyContinue |
Sort-Object LastWriteTime -Descending |
Select-Object -First 8 Name, LastWriteTime

Write-Host ""
Write-Host "Done."
