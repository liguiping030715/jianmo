$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$workspace = Join-Path $root "workspace"
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$archiveRoot = Join-Path $root "practice_cases"
$archive = Join-Path $archiveRoot "workspace_$stamp"

New-Item -ItemType Directory -Force -Path $archiveRoot | Out-Null
Copy-Item -Recurse -Force $workspace $archive
Write-Host "Archived current workspace to: $archive"

# Preserve knowledge library; reset task-specific directories only.
$taskDirs = @(
  "analysis", "data", "evidence", "experiments", "figures", "models", "paper", "problem", "references", "results", "submissions"
)
foreach ($d in $taskDirs) {
  $p = Join-Path $workspace $d
  if (Test-Path $p) { Remove-Item -Recurse -Force $p }
  New-Item -ItemType Directory -Force -Path $p | Out-Null
}
New-Item -ItemType Directory -Force -Path (Join-Path $workspace "data/raw") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $workspace "data/processed") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $workspace "evidence") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $workspace "references") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $workspace "submissions") | Out-Null

Write-Host "Task workspace reset. Historical knowledge under workspace/knowledge was preserved."
Write-Host "Put the 2026 problem in workspace/problem/ and attachments in workspace/data/raw/."
