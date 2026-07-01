$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$bootstrap = Join-Path $repoRoot "runpod_bootstrap.sh"

if (-not (Test-Path $bootstrap)) {
    throw "Missing bootstrap script: $bootstrap"
}

Write-Host "Testing SSH alias runpod-phase2..."
ssh -o BatchMode=yes runpod-phase2 "echo CONNECTED; hostname; pwd"

Write-Host "Uploading bootstrap script..."
scp $bootstrap runpod-phase2:/workspace/phase2_bootstrap.sh

Write-Host "Running bootstrap on RunPod..."
ssh runpod-phase2 "bash /workspace/phase2_bootstrap.sh"

Write-Host "Done. Open an interactive shell with: ssh runpod-phase2"
