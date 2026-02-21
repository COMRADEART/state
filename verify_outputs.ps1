Write-Host "`n=== Verify Outputs ===`n"

$events = ".\dimensional_core\state\events.jsonl"
$ip = ".\dimensional_core\state\instance_point.json"
$dim = ".\dimensional_core\state\dimensions\storage"

if (Test-Path $events) { Write-Host "OK events.jsonl" } else { Write-Host "MISSING events.jsonl" }
if (Test-Path $ip)     { Write-Host "OK instance_point.json" } else { Write-Host "MISSING instance_point.json" }
if (Test-Path $dim)    { Write-Host "OK dimensions/storage" } else { Write-Host "MISSING dimensions/storage" }

Write-Host "`nLatest snapshots:`n"
Get-ChildItem $dim -ErrorAction SilentlyContinue |
Sort-Object LastWriteTime -Descending |
Select-Object -First 5 FullName
